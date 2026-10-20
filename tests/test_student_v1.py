"""CPU-only v1 contracts and small optimization checks; no simulator or runner.learn."""
from copy import deepcopy
from dataclasses import asdict
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"source/rl_training"))
sys.path.insert(0, str(ROOT/"deploy/deploy_mujoco"))
from m20_student_v1.configuration import StudentV1PerceptionCfg, DeeproboticsM20StudentV1DistillationCfg
from m20_student_v1.contract import student_dim, observation_contract, validate_contract
from m20_student_v1.perception import (StudentObservationState, ActionDelayQueue, grid_points,
                                       terrain_visibility, pose_height_variance, confidence_from_quality, tilt_height_jacobian)
from m20_student_v1.distillation import StudentV1Distillation, grouped_loss, stratified_indices, stratum_keys
from m20_student_v1.model_io import initialize_student, actor_from_state
from m20_student_v1.profiles import profile_config, PROFILE_NAMES
from m20_student_v1.symmetry import mirror_actions, mirror_observations
from m20_student_v1.output_storage import output_budget
from m20_student_v1 import environment as glue
from student_v1_adapter import StudentMapAdapter, SensorExtras, load_student
torch.set_num_threads(1)


def fixture(n=2):
    grid = grid_points("cpu")
    hits = torch.cat((grid, (.2*grid[:, :1])), -1)[None].repeat(n, 1, 1)
    base = torch.tensor([[0., 0., .5]]).repeat(n, 1)
    yaw = torch.zeros(n)
    rotation = torch.eye(3)[None].repeat(n, 1, 1)
    proprio = torch.zeros(n, 57)
    proprio[:, 5] = -1.
    return proprio, hits, base, yaw, rotation


def clean_cfg(**overrides):
    cfg = profile_config("clean", {})
    for key, value in overrides.items():
        setattr(cfg, key, value)
    cfg.validate()
    return cfg


def force_first_capture(state):
    state._initialize()
    state.next_capture.zero_()


class SensorTests(unittest.TestCase):
    def test_contract_clean_and_unknown(self):
        p, hits, base, yaw, rotation = fixture()
        hits[1, 10] = torch.nan
        state = StudentObservationState(2, "cpu", .02, clean_cfg())
        obs = state.observe(p, hits, base, yaw, rotation, 0)
        self.assertEqual(obs.shape, (2, 618))
        self.assertTrue(torch.allclose(obs[0, 57:244], -hits[0, :, 2]))
        self.assertTrue(torch.equal(obs[0, 244:618], torch.ones(374)))
        self.assertEqual([obs[1, k].item() for k in (67, 254, 441)], [0., 0., 0.])
        known = torch.arange(187) != 10
        self.assertTrue(torch.equal(obs[1, 431:618][known], torch.ones(186)))
        self.assertTrue(torch.isfinite(obs).all())
        validate_contract(observation_contract())
        bad = observation_contract()
        bad["grid"]["order"] = "x then y"
        with self.assertRaises(ValueError):
            validate_contract(bad)

    def test_fixed_timing_across_noise_levels(self):
        snapshots = []
        for level in (.25, 1.):
            torch.manual_seed(17)
            cfg = clean_cfg(level=level, map_update_period_s=.1, update_jitter_s=.02, max_delay_s=.06,
                            height_noise_std=.02)
            state = StudentObservationState(2, "cpu", .02, cfg)
            p, hits, base, yaw, rotation = fixture()
            for step in range(20):
                state.observe(p, hits, base, yaw, rotation, step)
            snapshots.append((state.packet_capture.clone(), state.packet_arrival.clone(), state.next_capture.clone()))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(*snapshots)))

    def test_world_motion_is_not_lost_by_repeated_rounding(self):
        p, hits, base, yaw, rotation = fixture()
        state = StudentObservationState(2, "cpu", .02, clean_cfg(map_update_period_s=1., max_map_age_s=2.))
        force_first_capture(state)
        before = state.observe(p, hits, base, yaw, rotation, 0)
        for step in range(1, 4):
            moved = base.clone()
            moved[:, 0] = step*.02
            after = state.observe(p, hits, moved, yaw, rotation, step)
        self.assertAlmostEqual((after[0, 150]-before[0, 150]).item(), -.02, places=5)
        self.assertEqual(state.capture[0].item(), 0)

    def test_cells_retain_individual_age_then_expire(self):
        p, hits, base, yaw, rotation = fixture()
        state = StudentObservationState(2, "cpu", .02, clean_cfg(max_map_age_s=.05))
        state.observe(p, hits, base, yaw, rotation, 0)
        hidden = hits.clone()
        hidden[:, 93] = torch.nan
        obs = state.observe(p, hidden, base, yaw, rotation, 1)
        self.assertEqual(obs[0, 244+93].item(), 1.)
        self.assertAlmostEqual(state.cell_age[0, 93].item(), .02, places=6)
        self.assertEqual(state.cell_age[0, 94].item(), 0.)
        obs = state.observe(p, hidden, base, yaw, rotation, 3)
        self.assertEqual(obs[0, 244+93].item(), 0.)
        self.assertEqual(obs[0, 57+93].item(), 0.)
        self.assertEqual(obs[0, 431+93].item(), 0.)

    def test_partial_reset_and_history_are_idempotent(self):
        p, hits, base, yaw, rotation = fixture()
        state = StudentObservationState(2, "cpu", .02, clean_cfg(), 4)
        state.observe(p, hits, base, yaw, rotation, 0)
        p[:, 0] = .1
        saved = state.observe(p, hits, base, yaw, rotation, 1)
        self.assertEqual(saved.shape, (2, 789))
        self.assertEqual(saved[0, -57+5].item(), -1.)
        history = state.history.clone()
        self.assertTrue(torch.equal(saved, state.observe(p, hits, base, yaw, rotation, 1)))
        self.assertTrue(torch.equal(history, state.history))
        state.reset(torch.tensor([0]))
        current = state.observe(p, hits, base, yaw, rotation, 1)
        self.assertTrue(torch.equal(current[1], saved[1]))
        self.assertEqual(current[0, 618:].abs().sum().item(), 0.)
        self.assertTrue(torch.equal(state.history[1], history[1]))

    def test_front_rear_fov_and_rigid_transform_invariance(self):
        p, hits, base, yaw, rotation = fixture(1)
        cfg = clean_cfg(lidar_horizontal_fov_deg=135.)
        visible = terrain_visibility(hits, base, yaw, rotation, cfg)
        self.assertTrue(visible[0, 5*17+16])
        self.assertTrue(visible[0, 5*17])
        self.assertFalse(visible[0, 8])
        self.assertFalse(visible[0, 10*17+8])
        r = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        moved_hits = hits @ r.T + torch.tensor([2., -3., .7])
        moved_base = base @ r.T + torch.tensor([2., -3., .7])
        moved_rotation = r[None] @ rotation
        rotated = terrain_visibility(moved_hits, moved_base, yaw+torch.pi/2, moved_rotation, cfg)
        self.assertTrue(torch.equal(visible, rotated))

    def test_pose_quality_depends_on_measured_geometry(self):
        grid = grid_points("cpu")
        valid = torch.ones(2, 187, dtype=torch.bool)
        z = torch.zeros(2, 187)
        z[1, grid[:, 0] >= .1] = .2
        variance = pose_height_variance(z, valid, grid, torch.full((2,), .03), torch.full((2,), .02), torch.zeros(2))
        self.assertEqual(variance[0].max().item(), 0.)
        self.assertGreater(variance[1].max().item(), 0.)
        self.assertTrue(torch.allclose(variance, pose_height_variance(z+7, valid, grid, torch.full((2,), .03), torch.full((2,), .02), torch.zeros(2)), atol=1e-6, rtol=1e-5))
        score = confidence_from_quality(valid, variance, torch.zeros_like(z), clean_cfg())
        self.assertLess(score[1].min().item(), score[0].min().item())
        invalid = confidence_from_quality(torch.ones(3), torch.tensor([0., torch.nan, .01]), torch.tensor([1., 0., 0.]), clean_cfg())
        self.assertEqual(invalid[0].item(), 0.)
        self.assertEqual(invalid[1].item(), 0.)

    def test_map_tilt_and_projected_gravity_have_consistent_frame_sign(self):
        p, hits, base, yaw, rotation = fixture(1)
        hits[:, :, 2] = 0.
        cfg = clean_cfg()
        state = StudentObservationState(1, "cpu", .02, cfg)
        force_first_capture(state)
        state.severity.fill_(1.)
        state.bias.zero_()
        state.bias[:, 4] = .01
        obs = state.observe(p, hits, base, yaw, rotation, 0)
        # R_hat = R_true*RotX(+epsilon): estimated heights tilt +epsilon*y;
        # body-frame gravity is RotX(-epsilon)*[0,0,-1], hence negative y.
        expected_gravity = torch.tensor([0., -.01, -1.])
        expected_gravity /= expected_gravity.norm()
        self.assertTrue(torch.allclose(obs[0, 3:6], expected_gravity))
        self.assertTrue(torch.allclose(obs[0, 57:244], -.01*state.grid[:, 1], atol=1e-7))

    def test_tilt_height_propagation_matches_full_rotation_at_inclined_pose(self):
        angle = torch.tensor(.3, dtype=torch.float64)
        rotation = torch.tensor([[angle.cos(), 0., angle.sin()], [0., 1., 0.],
                                 [-angle.sin(), 0., angle.cos()]], dtype=torch.float64)[None]
        points = torch.tensor([[[.5, .2, 0.], [-.3, .4, .1]]], dtype=torch.float64)
        base = torch.tensor([[0., 0., .5]], dtype=torch.float64)
        jacobian = tilt_height_jacobian(points, base, rotation)
        eps = torch.tensor(1e-6, dtype=torch.float64)
        body = (points-base[:, None]) @ rotation
        for column, perturbation in enumerate((
            torch.tensor([[1., 0., 0.], [0., eps.cos(), -eps.sin()], [0., eps.sin(), eps.cos()]], dtype=torch.float64),
            torch.tensor([[eps.cos(), 0., eps.sin()], [0., 1., 0.], [-eps.sin(), 0., eps.cos()]], dtype=torch.float64),
        )):
            estimated = body @ (rotation@perturbation).transpose(-1, -2)+base[:, None]
            finite_difference = (estimated[..., 2]-points[..., 2])/eps
            self.assertTrue(torch.allclose(jacobian[..., column], finite_difference, atol=1e-6, rtol=1e-6))

    def test_missing_packets_do_not_invent_observations(self):
        p, hits, base, yaw, rotation = fixture()
        cfg = clean_cfg(packet_dropout=1.)
        state = StudentObservationState(2, "cpu", .02, cfg)
        for step in range(10):
            obs = state.observe(p, hits, base, yaw, rotation, step)
        self.assertEqual(obs[:, 57:].abs().sum().item(), 0.)

    def test_all_profiles_validate_and_are_single_factors(self):
        for name in PROFILE_NAMES:
            profile_config(name, {"perception": asdict(StudentV1PerceptionCfg())}).validate()
        height = profile_config("height", {})
        self.assertGreater(height.height_noise_std, 0.)
        self.assertEqual(height.xy_bias_std, 0.)
        self.assertEqual(height.max_delay_s, 0.)
        fov = profile_config("lidar_fov", {})
        self.assertEqual(fov.lidar_horizontal_fov_deg, 135.)
        self.assertEqual(fov.gyro_std_rad_s, 0.)
        self.assertEqual(fov.height_noise_std, 0.)
        latency = profile_config("latency", {})
        self.assertGreater(latency.max_delay_s, 0.)
        self.assertEqual(latency.action_delay_steps, 0)
        reference = asdict(profile_config("quality_reference", {}))
        imperfect = asdict(profile_config("confidence", {}))
        self.assertEqual({key for key in reference if reference[key] != imperfect[key]},
                         {"confidence_log_std", "confidence_overestimate_probability"})
        for kwargs in ({"level": float("nan")}, {"clean_probability": .7, "stress_probability": .7}, {"action_delay_steps": 1.5}):
            with self.assertRaises(ValueError):
                StudentV1PerceptionCfg(**kwargs).validate()

    def test_action_delay_and_partial_reset(self):
        queue = ActionDelayQueue(2, "cpu", 1)
        a, b, c = torch.ones(2, 16), torch.full((2, 16), 2.), torch.full((2, 16), 3.)
        self.assertEqual(queue.push(a).sum().item(), 0.)
        self.assertTrue(torch.equal(queue.push(b), a))
        queue.reset(torch.tensor([0]))
        applied = queue.push(c)
        self.assertEqual(applied[0].sum().item(), 0.)
        self.assertTrue(torch.equal(applied[1], b[1]))
        self.assertTrue(torch.equal(queue.previous_issued, c))


def make_algorithm(history_steps=1, **overrides):
    dim = student_dim(history_steps)
    teacher_obs = torch.randn(4, 244)*.1
    obs = TensorDict({"teacher": teacher_obs, "policy": torch.cat((teacher_obs, torch.ones(4, 374), torch.zeros(4, dim-618)), -1)}, batch_size=[4])
    config = dict(hidden_dims=[16, 8], activation="elu", obs_normalization=False,
                  distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1., "std_type": "log"})
    teacher = MLPModel(obs, {"teacher": ["teacher"]}, "teacher", 16, **deepcopy(config))
    student = MLPModel(obs, {"student": ["policy"]}, "student", 16, **deepcopy(config))
    initialize_student(student, {"actor_state_dict": teacher.state_dict()})
    storage = RolloutStorage("distillation", 4, 2, obs, [16], "cpu")
    options = dict(teacher_ratio=0., teacher_ratio_end=0., batch_size=4, replay_rollouts=2, hard_replay_rollouts=1,
                   teacher_warmup_updates=0, teacher_decay_updates=2, gate_min_episodes=1)
    options.update(overrides)
    alg = StudentV1Distillation(student, teacher, storage, **options)
    alg.teacher_loaded = True
    return alg, obs, config


class AlgorithmTests(unittest.TestCase):
    def test_group_loss_and_critical_weights_with_analytical_counterexample(self):
        target = torch.zeros(2, 16)
        predicted = torch.zeros_like(target)
        predicted[1, :12] = 1.
        plain, _, _ = grouped_loss(predicted, target, torch.zeros(2, dtype=torch.long))
        weighted, _, _ = grouped_loss(predicted, target, torch.tensor([0, 1]))
        self.assertAlmostEqual(plain.item(), .25)
        self.assertAlmostEqual(weighted.item(), 1/3, places=6)
        wheel_only = target.clone()
        wheel_only[:, 12:] = 1.
        leg_only = target.clone()
        leg_only[:, :12] = 1.
        self.assertEqual(grouped_loss(wheel_only, target, torch.zeros(2, dtype=torch.long))[0].item(),
                         grouped_loss(leg_only, target, torch.zeros(2, dtype=torch.long))[0].item())
        with torch.enable_grad():
            predicted.requires_grad_()
            grouped_loss(predicted, target, torch.tensor([0, 1]))[0].backward()
        self.assertGreater(predicted.grad[1].abs().sum().item(), 0.)

    def test_stratification_rebalances_rare_cases(self):
        torch.manual_seed(13)
        keys = torch.cat((torch.zeros(990, dtype=torch.long), torch.ones(10, dtype=torch.long)))
        sample = keys[stratified_indices(keys, 10000)].float().mean().item()
        self.assertTrue(.45 < sample < .55)
        self.assertLess(keys.float().mean().item(), .02)

    def test_terrain_difficulty_actually_changes_sampling_stratum(self):
        torch.manual_seed(13)
        tags = torch.zeros(1000, 4, dtype=torch.long)
        tags[-10:, 3] = 9
        keys = stratum_keys(tags)
        self.assertEqual(keys.unique().numel(), 2)
        sampled_level = tags[stratified_indices(keys, 10000), 3]
        self.assertTrue(.45 < (sampled_level == 9).float().mean().item() < .55)

    def test_labels_owned_pre_step_hard_bank_and_resume(self):
        alg, obs, _ = make_algorithm()
        before = obs["policy"].clone()
        labels = alg.teacher(obs).detach().clone()
        with torch.inference_mode():
            alg.act(obs)
            obs["policy"] += 10.
            info = {"student_v1": {"phase": torch.tensor([0, 1, 2, 3]), "terrain": torch.tensor([0, 1, 2, 3]),
                                      "direction": torch.tensor([0, 1, 3, 5]), "terrain_level": torch.tensor([0, 3, 6, 9]),
                                      "terminated": torch.zeros(4, dtype=torch.bool)}}
            alg.process_env_step(obs, torch.zeros(4), torch.zeros(4), info)
        self.assertTrue(torch.equal(alg.replay_obs[:4], before))
        self.assertTrue(torch.equal(alg.replay_labels[:4], labels))
        self.assertGreater(alg.hard_size, 0)
        batch, target, tags = alg._batch(16)
        self.assertEqual(batch.shape, (16, 618))
        self.assertTrue(torch.isfinite(target).all())
        saved = alg.save()
        restored, _, _ = make_algorithm()
        restored.load(saved)
        self.assertTrue(torch.equal(alg.replay_obs, restored.replay_obs))
        self.assertEqual(alg.hard_size, restored.hard_size)
        self.assertEqual(len(restored.optimizer.state), 0)
        bad = deepcopy(saved)
        bad["student_v1"]["version"] = "v0"
        with self.assertRaises(ValueError):
            restored.load(bad)
        for kind in ("schedule", "tags"):
            bad = deepcopy(saved)
            if kind == "schedule":
                bad["student_v1"]["schedule_progress"] = restored.decay+1
            else:
                bad["replay_tags"] = bad["replay_tags"].float()
            with self.assertRaises(ValueError):
                restored.load(bad)

    def test_teacher_decay_gate_and_zero_teacher_stage(self):
        alg, obs, _ = make_algorithm(teacher_ratio=.5)
        alg.num_updates = 1
        self.assertFalse(alg._advance_schedule())
        self.assertEqual(alg.teacher_ratio, .5)
        alg.student_episodes = 100
        alg.failure_ema = .5
        self.assertFalse(alg._advance_schedule())
        self.assertEqual(alg.teacher_ratio, .5)
        alg.failure_ema = 0.
        self.assertTrue(alg._advance_schedule())
        self.assertAlmostEqual(alg.teacher_ratio, .25)
        alg._advance_schedule()
        self.assertEqual(alg.teacher_ratio, 0.)
        alg.teacher_mask = torch.ones(4, dtype=torch.bool)
        with torch.inference_mode():
            issued = alg.act(obs)
        self.assertFalse(alg.teacher_mask.any())
        self.assertTrue(torch.equal(issued, alg.student(obs)))
        self.assertFalse(any(p.requires_grad for p in alg.teacher.parameters()))

    def test_light_checkpoint_preserves_training_state_without_serializing_replay(self):
        alg, obs, _ = make_algorithm(4, save_replay=False)
        labels = torch.randn(4, 16)
        tags = torch.tensor([[1, 1, 1, 0]]*4)
        for prefix in ("replay", "hard"):
            alg._append(prefix, obs["policy"], labels, tags, torch.zeros(4, dtype=torch.bool))
        alg.num_updates, alg.student_episodes, alg.schedule_progress = 7, 11, 1
        alg.failure_ema = .1
        parameter = next(iter(alg.optimizer.param_groups[0]["params"]))
        # Exercise Adam-state serialization without performing an optimizer step.
        alg.optimizer.state[parameter] = {"step": torch.tensor(3.),
                                          "exp_avg": torch.ones_like(parameter)*.1,
                                          "exp_avg_sq": torch.ones_like(parameter)*.2}
        before = alg.replay_obs.clone()
        sizes = (alg.replay_size, alg.hard_size, alg.replay_cursor, alg.hard_cursor)
        with tempfile.TemporaryDirectory() as directory:
            light_path, full_path = Path(directory)/"light.pt", Path(directory)/"full.pt"
            torch.save(alg.save(), light_path)
            alg.save_replay = True
            torch.save(alg.save(), full_path)
            self.assertLess(light_path.stat().st_size, full_path.stat().st_size)
            saved = torch.load(light_path, weights_only=True)
        self.assertTrue(torch.equal(alg.replay_obs, before))
        self.assertEqual(sizes, (alg.replay_size, alg.hard_size, alg.replay_cursor, alg.hard_cursor))
        self.assertFalse(saved["student_v1"]["replay_saved"])
        for prefix in ("replay", "hard"):
            self.assertEqual(saved["student_v1"][prefix+"_size"], 0)
            for field in ("obs", "labels", "tags", "holdout"):
                self.assertEqual(saved[prefix+"_"+field].untyped_storage().nbytes(), 0)
        restored, _, _ = make_algorithm(4, save_replay=False)
        restored.load(saved)
        self.assertEqual((restored.replay_size, restored.hard_size), (0, 0))
        self.assertEqual((restored.num_updates, restored.student_episodes, restored.schedule_progress), (7, 11, 1))
        self.assertAlmostEqual(restored.failure_ema, .1)
        self.assertTrue(torch.equal(restored.student(obs), alg.student(obs)))
        restored_state = next(iter(restored.optimizer.state.values()))
        self.assertEqual(restored_state["step"].item(), 3.)
        self.assertTrue(torch.equal(restored_state["exp_avg"], torch.ones_like(parameter)*.1))

    def test_weights_only_history_and_export_parity(self):
        alg, obs, config = make_algorithm(4)
        first = alg.student.state_dict()["mlp.0.weight"]
        self.assertEqual(first[:, 244:].abs().sum().item(), 0.)
        self.assertEqual(len(alg.optimizer.state), 0)
        self.assertTrue(torch.equal(alg.student(obs), alg.teacher(obs)))
        actor = actor_from_state(alg.student.state_dict(), 789, {**config, "class_name": "rsl_rl.models.MLPModel"})
        exported = torch.jit.script(alg.student.as_jit())
        with torch.inference_mode():
            self.assertTrue(torch.allclose(actor(obs["policy"]), exported(obs["policy"])))

    def test_tags_required_and_critic_excluded(self):
        alg, obs, _ = make_algorithm()
        obs["critic"] = torch.full_like(obs["policy"], 999.)
        with torch.inference_mode():
            self.assertTrue(torch.equal(alg.act(obs), alg.student(obs)))
            with self.assertRaises(RuntimeError):
                alg.process_env_step(obs, torch.zeros(4), torch.zeros(4), {})


class IntegrationTests(unittest.TestCase):
    def test_evaluation_frozen_plan_and_changed_input_rejected(self):
        spec = importlib.util.spec_from_file_location("evaluation_v1_plan", ROOT/"scripts/tools/evaluate_student_v1.py")
        evaluation = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(evaluation)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            weights = base/"model_0.pt"
            weights.write_bytes(b"plan-only fixture; not a trained v1")
            model = base/"robot.xml"
            model.write_text("<mujoco/>")
            args = evaluation.parse_args(["--v0", str(weights), "--v1", str(weights), "--model", str(model),
                                          "--terrain", "descent", "--profiles", "quality_reference", "confidence",
                                          "--seeds", "42", "--output-dir", str(base/"evaluation")])
            plan = evaluation.freeze_plan(args)
            self.assertFalse(args.run)
            self.assertEqual(len(plan["cases"]), 4)
            self.assertEqual(Path(plan["terrain_xml"]).name, "stairs_descent.xml")
            evaluation.verify_plan(plan)
            execute = evaluation.parse_args(["--run-plan", str(base/"evaluation/plan.json")])
            self.assertIsNotNone(execute.run_plan)
            weights.write_bytes(b"changed input")
            with self.assertRaises(ValueError):
                evaluation.verify_plan(plan)

    def test_evaluation_distance_and_touchdown_window_analytical_counterexamples(self):
        spec = importlib.util.spec_from_file_location("evaluation_v1", ROOT/"scripts/tools/evaluate_student_v1.py")
        evaluation = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(evaluation)
        # At 1 m/s the policy boundary moved 20 mm. A stale 5 ms position would inflate this to 25 mm.
        self.assertAlmostEqual(evaluation.planar_distance([0., 0., .5], [.02, 0., .5]), .02)
        self.assertAlmostEqual(evaluation.planar_distance([-.005, 0., .5], [.02, 0., .5]), .025)
        event = np.zeros((8, 4), dtype=bool)
        event[0, 0] = True
        event[-1, 1] = True
        result = evaluation.touchdown_windows({"event": event, "force_norm_N": np.full((8, 4), 100.)}, .005)
        self.assertEqual(result["full_window_events"], 1)
        self.assertEqual(result["truncated_events"], 1)
        self.assertEqual(result["force_peak_p95_N"], 100.)
        self.assertEqual(result["force_impulse_p95_N_s"], 4.)

    def test_training_glue_and_deployment_tensor_parity(self):
        p, hits, base, yaw, rotation = fixture(1)
        cfg = StudentV1PerceptionCfg(clean_probability=0., stress_probability=0.)
        names = glue.PROPRIO_TERMS
        slices = ((0, 3), (3, 6), (6, 9), (9, 25), (25, 41), (41, 57))
        terms = {}
        for name, (start, end) in zip(names, slices):
            terms[name] = SimpleNamespace(func=lambda env, start=start, end=end: p[:, start:end],
                                          params={}, modifiers=None, history_length=0, clip=None, scale=1.)
        data = SimpleNamespace(root_quat_w=torch.tensor([[1., 0., 0., 0.]]), root_pos_w=base)
        env = SimpleNamespace(num_envs=1, device="cpu", step_dt=.02, common_step_counter=0,
                              scene={"robot": SimpleNamespace(data=data)},
                              cfg=SimpleNamespace(student_perception=cfg, student_history_steps=4,
                                                  observations=SimpleNamespace(teacher=SimpleNamespace(**terms))))
        env.scene = type("Scene", (dict,), {"sensors": {"height_scanner": SimpleNamespace(data=SimpleNamespace(ray_hits_w=hits))}})(env.scene)
        class Contract:
            base_id = 0
            last_scan_hits = hits[0].numpy()
            def observe(self, data, command, previous):
                return np.concatenate((p[0].numpy(), np.zeros(187))).astype(np.float32)
        adapter = StudentMapAdapter(Contract(), deepcopy(cfg), SensorExtras(1, 4))
        mj_data = SimpleNamespace(xpos=base.numpy(), xmat=rotation.reshape(1, 9).numpy())
        for step in range(12):
            env.common_step_counter = step
            torch.manual_seed(100+step)
            train_obs = glue.student_observation(env)
            torch.manual_seed(100+step)
            deploy_obs = adapter.observe(mj_data, np.zeros(3), np.zeros(16), step)
            self.assertTrue(torch.equal(train_obs[0], torch.from_numpy(deploy_obs)))

    def test_training_only_phase_and_direction_tags(self):
        p, hits, base, yaw, rotation = fixture()
        command = torch.tensor([[.5, 0., 0.], [0., .5, .3]])
        clearance = torch.full((2, 4), .03)
        velocity = torch.full((2, 4), -.1)
        contact = torch.ones(2, 4, dtype=torch.bool)
        contact[0, 0] = False
        tags = glue.classify_state(hits, p[:, 3:6], command, clearance, velocity, contact, torch.tensor([3, 7]))
        self.assertEqual(tags["phase"][0].item(), 1)
        self.assertEqual(tags["direction"].tolist(), [1, 5])
        self.assertEqual(tags["terrain_level"].tolist(), [3, 7])

    def test_v1_checkpoint_deployment_and_v0_rejection(self):
        alg, obs, config = make_algorithm()
        alg.provenance = {"contract": observation_contract(), "perception": asdict(StudentV1PerceptionCfg()),
                          "training_config": {"student": config}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"model.pt"
            torch.save(alg.save(), path)
            actor, provenance, identity = load_student(path)
            self.assertEqual(identity["actor_input_dim"], 618)
            self.assertTrue(torch.allclose(actor(obs["policy"]), alg.student(obs)))
            torch.save({"student_v0": {}}, path)
            with self.assertRaises(ValueError):
                load_student(path)

    def test_budget_includes_tags_critical_replay_and_history(self):
        cfg = DeeproboticsM20StudentV1DistillationCfg().to_dict()
        single = output_budget(cfg, 64, 10000, StudentV1PerceptionCfg())
        cfg["history_steps"] = 4
        history = output_budget(cfg, 64, 10000, StudentV1PerceptionCfg())
        self.assertGreater(history["checkpoint_estimate_bytes"], single["checkpoint_estimate_bytes"])
        self.assertGreater(single["required_free_bytes"], 2*1024**3)

    def test_checkpoint_policy_changes_disk_budget_but_not_live_replay(self):
        cfg = DeeproboticsM20StudentV1DistillationCfg().to_dict()
        cfg["history_steps"] = 4
        light = output_budget(cfg, 4096, 10000, StudentV1PerceptionCfg())
        cfg["algorithm"]["save_replay"] = True
        full = output_budget(cfg, 4096, 10000, StudentV1PerceptionCfg())
        self.assertEqual(light["sample_buffer_bytes"], full["sample_buffer_bytes"])
        self.assertEqual(light["checkpoint_replay_bytes"], 0)
        self.assertGreater(full["required_free_bytes"], 200*1024**3)
        self.assertLess(light["required_free_bytes"], 4*1024**3)
        del cfg["algorithm"]["save_replay"]
        legacy = output_budget(cfg, 4096, 10000, StudentV1PerceptionCfg())
        self.assertEqual(legacy["checkpoint_estimate_bytes"], full["checkpoint_estimate_bytes"])


class SymmetryTests(unittest.TestCase):
    def test_reflection_involution_and_map_axis(self):
        for dim in (244, 618, 789):
            x = torch.arange(dim, dtype=torch.float32)[None]
            self.assertTrue(torch.equal(mirror_observations(mirror_observations(x)), x))
            y = mirror_observations(x)
            self.assertEqual(y[0, 57], x[0, 57+170])
            self.assertEqual(y[0, 6], x[0, 6])
            self.assertEqual(y[0, 7], -x[0, 7])
            self.assertEqual(y[0, 8], -x[0, 8])
        a = torch.randn(5, 16)
        self.assertTrue(torch.equal(mirror_actions(mirror_actions(a)), a))

    def test_actual_update_uses_mirror_gradient(self):
        def run(weight):
            torch.manual_seed(23)
            obs = TensorDict({"policy": torch.randn(4,789), "teacher": torch.randn(4,244)}, batch_size=[4])
            cfg = DeeproboticsM20StudentV1DistillationCfg().to_dict()
            models=[]
            for key in ("student", "teacher"):
                m = dict(cfg[key]);m.pop("class_name");m["hidden_dims"]=[16,8]
                models.append(MLPModel(obs, cfg["obs_groups"], key, 16, **m))
            alg = StudentV1Distillation(*models, RolloutStorage("distillation",4,1,obs,[16],"cpu"),
                  num_learning_epochs=1, batch_size=4, replay_rollouts=1, hard_replay_rollouts=1,
                  symmetry_loss_weight=weight)
            alg._append("replay",obs["policy"],torch.randn(4,16),torch.zeros(4,4,dtype=torch.long),torch.zeros(4,dtype=torch.bool))
            alg.rollout_samples=4;alg.executed_total=4
            stats=alg.update()
            return stats,torch.cat([p.detach().flatten() for p in alg.student.mlp.parameters()])
        off,a=run(0);on,b=run(.1)
        self.assertEqual(off["mirror_mse"],0)
        self.assertGreater(on["mirror_mse"],0)
        self.assertFalse(torch.equal(a,b), "Mirror loss must change the optimizer update")



if __name__ == "__main__":
    unittest.main()
