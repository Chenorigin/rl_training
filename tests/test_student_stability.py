"""Stability loss gradients, data isolation, reset boundaries and start/stop coverage."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/rl_training"))
from m20_student_v1.configuration import DeeproboticsM20StudentV1DistillationCfg, StudentV1PerceptionCfg
from m20_student_v1.distillation import StudentV1Distillation
from m20_student_v1.stability import (adjacent_training_pairs, sensor_noise_std,
                                      perturb_sensors, temporal_residual_loss)
from task_support.student_v3_safe.start_stop import practice_masks, validate_practice
from task_support.student_v3_safe.configuration import default_settings

torch.set_num_threads(1)


def collect_fixture(temporal=0., spatial=0., **retention_cfg):
    torch.manual_seed(251)
    obs = TensorDict({"policy": torch.randn(4, 789) * .1,
                      "teacher": torch.randn(4, 244) * .1,
                      "motion_control": torch.zeros(4, 3)}, batch_size=[4])
    obs["policy"][:, 6:9] = 0.
    obs["policy"][:, 244:618] = 1.
    config = DeeproboticsM20StudentV1DistillationCfg().to_dict()
    models = []
    for key in ("student", "teacher"):
        model = deepcopy(config[key]); model.pop("class_name"); model["hidden_dims"] = [16, 8]
        models.append(MLPModel(obs, config["obs_groups"], key, 16, **model))
    alg = StudentV1Distillation(*models, RolloutStorage("distillation", 4, 3, obs, [16], "cpu"),
        teacher_ratio=0., teacher_ratio_end=0., batch_size=12, num_learning_epochs=1,
        replay_rollouts=1, hard_replay_rollouts=1, stratified_fraction=0., hard_fraction=0.,
        temporal_loss_weight=temporal, spatial_loss_weight=spatial, stability_batch_size=12,
        stability_noise_std=sensor_noise_std(StudentV1PerceptionCfg()), **retention_cfg)
    alg.teacher_loaded = True
    tags = {k: torch.zeros(4, dtype=torch.long) for k in ("phase", "terrain", "direction", "terrain_level")}
    tags["terminated"] = torch.zeros(4, dtype=torch.bool)
    with torch.inference_mode():
        for step in range(3):
            current = obs.clone()
            current["policy"][:, 0] += .1 * step
            current["policy"][:, 8] += .01 * step
            current["teacher"][:, 0] -= .03 * step
            alg.act(current)
            alg.process_env_step(current, torch.zeros(4), torch.zeros(4), {"student_v1": tags})
    return alg


class StabilityTests(unittest.TestCase):
    def test_real_adjacent_pairs_exclude_resets_holdouts_phase_and_command_changes(self):
        obs = torch.zeros(3, 5, 789)
        dones = torch.zeros(3, 5, 1, dtype=torch.bool)
        training = torch.ones(3, 5, dtype=torch.bool)
        regular = torch.ones_like(training)
        # At t=1 only env0 is eligible. The exclusions differ intentionally.
        dones[0, 1] = True
        training[0, 2] = False
        regular[1, 3] = False
        obs[1, 4, 6] = 1.
        time, env = adjacent_training_pairs(obs, dones, training, regular)
        actual = set(zip(time.tolist(), env.tolist()))
        self.assertEqual(actual, {(1, 0), (2, 0), (2, 1), (2, 2)})
        self.assertEqual(adjacent_training_pairs(obs[:1], dones[:1], training[:1], regular[:1])[0].numel(), 0)

    def test_heading_feedback_changes_do_not_hide_zero_request_pairs(self):
        obs = torch.zeros(3, 2, 789)
        obs[1, :, 8] = .01
        raw_commands = torch.zeros(3, 2, 3)
        raw_commands[2, 1, 0] = .7
        mask = torch.ones(3, 2, dtype=torch.bool)
        done = torch.zeros(3, 2, 1, dtype=torch.bool)
        time, env = adjacent_training_pairs(obs, done, mask, mask, raw_commands)
        self.assertEqual(set(zip(time.tolist(), env.tolist())), {(1, 0), (1, 1), (2, 0)})

    def test_teacher_required_motion_not_penalized_but_added_oscillation_is(self):
        current, previous = torch.randn(4, 16), torch.randn(4, 16)
        # A constant offset has no extra temporal oscillation; BC still corrects it.
        offset = torch.randn(1, 16)
        self.assertLess(temporal_residual_loss(current+offset, previous+offset, current, previous).item(), 1.e-12)
        candidate = (current + 1.).requires_grad_()
        target = current.clone().requires_grad_()
        loss = temporal_residual_loss(candidate, previous, target, previous)
        self.assertGreater(loss.item(), .9)
        loss.backward()
        self.assertGreater(candidate.grad.abs().sum().item(), 0.)
        self.assertIsNone(target.grad)

    def test_sensor_perturbation_preserves_commands_actions_history_and_unknowns(self):
        torch.manual_seed(16)
        obs = torch.randn(8, 789)
        obs[:, 244:431] = 1.; obs[:, 244:254] = 0.; obs[:, 57:67] = 0.
        before = obs.clone()
        altered = perturb_sensors(obs, sensor_noise_std(StudentV1PerceptionCfg()), .5)
        self.assertTrue(torch.equal(obs, before))
        for start, end in ((6, 9), (21, 25), (41, 57), (244, 789), (57, 67)):
            self.assertTrue(torch.equal(altered[:, start:end], obs[:, start:end]))
        self.assertGreater((altered[:, :3]-obs[:, :3]).abs().sum().item(), 0.)
        self.assertGreater((altered[:, 67:244]-obs[:, 67:244]).abs().sum().item(), 0.)
        torch.testing.assert_close(altered[:, 3:6].norm(dim=-1), obs[:, 3:6].norm(dim=-1))

    def test_actual_update_receives_each_stability_gradient_and_keeps_teacher_frozen(self):
        outputs = []
        for temporal, spatial in ((0., 0.), (.5, 0.), (0., .1)):
            alg = collect_fixture(temporal, spatial)
            teacher_before = {k: v.clone() for k, v in alg.teacher.state_dict().items()}
            gradients = []
            # Compare gradients, not only Adam's first signed step: different
            # gradients can otherwise produce nearly identical parameter deltas.
            original_step = alg.optimizer.step
            def capture_step():
                gradients.append(torch.cat([p.grad.flatten().clone() for p in alg.student.mlp.parameters()]))
                return original_step()
            alg.optimizer.step = capture_step
            torch.manual_seed(24)
            stats = alg.update()
            self.assertGreater(stats["temporal_pair_count"], 0.)
            self.assertTrue(all(torch.equal(v, alg.teacher.state_dict()[k]) for k, v in teacher_before.items()))
            self.assertTrue(all(p.grad is None for p in alg.teacher.parameters()))
            outputs.append((stats, gradients[0]))
        self.assertEqual(outputs[0][0]["temporal_residual_mse"], 0.)
        self.assertEqual(outputs[0][0]["spatial_action_mse"], 0.)
        self.assertGreater(outputs[1][0]["temporal_residual_mse"], 0.)
        self.assertGreater(outputs[2][0]["spatial_action_mse"], 0.)
        self.assertFalse(torch.equal(outputs[0][1], outputs[1][1]))
        self.assertFalse(torch.equal(outputs[0][1], outputs[2][1]))

    def test_disabled_terms_preserve_legacy_defaults_and_invalid_config_fails(self):
        cfg = DeeproboticsM20StudentV1DistillationCfg().algorithm
        self.assertEqual(cfg["temporal_loss_weight"], 0.)
        self.assertEqual(cfg["spatial_loss_weight"], 0.)
        alg = collect_fixture()
        alg.stability_training_mask.zero_()
        self.assertEqual(alg.update()["temporal_pair_count"], 0.)
        for temporal, spatial in ((-1., 0.), (0., float("nan"))):
            with self.assertRaises(ValueError):
                collect_fixture(temporal, spatial)

    def test_start_stop_episode_rotation_and_original_stall_bound(self):
        cfg = default_settings()["training"]["idle_practice"]
        validate_practice(cfg, 4.)
        ids, resets = torch.arange(8), torch.zeros(8, dtype=torch.long)
        selected, first = practice_masks(ids, resets, torch.zeros(8), 42, cfg)
        self.assertTrue(torch.equal(selected, first))
        self.assertEqual(int(selected.sum()), 2)
        move_time = cfg["initial_hold_s"] + cfg["move_s"] / 2
        stop_time = cfg["initial_hold_s"] + cfg["move_s"] + cfg["stop_s"] / 2
        final_time = cfg["initial_hold_s"] + cfg["move_s"] + cfg["stop_s"]
        self.assertFalse(practice_masks(ids, resets, torch.full((8,), move_time), 42, cfg)[1].any())
        self.assertTrue(torch.equal(practice_masks(ids, resets, torch.full((8,), stop_time), 42, cfg)[1], selected))
        self.assertFalse(practice_masks(ids, resets, torch.full((8,), final_time), 42, cfg)[1].any())
        covered = sum(practice_masks(ids, resets+i, torch.zeros(8), 42, cfg)[0].long() for i in range(cfg["every"]))
        self.assertTrue(torch.equal(covered, torch.ones(8, dtype=torch.long)))
        invalid = dict(cfg, initial_hold_s=4.)
        with self.assertRaisesRegex(ValueError, "stall timeout"):
            validate_practice(invalid, 4.)


if __name__ == "__main__":
    unittest.main()
