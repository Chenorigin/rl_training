"""CPU contract checks only: no physics, optimizer steps or training."""
from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
from pathlib import Path
import sys
from types import SimpleNamespace
from types import ModuleType
import importlib.util
import tempfile
import unittest
from unittest.mock import patch

import torch
from tensordict import TensorDict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/rl_training"))
from task_support.stair_trot_v1.signals import TrotSettings, ascent_geometry, gait_scores, phase_targets, paired_events
from task_support.stair_trot_v1.runner import StairTrotRunner
from task_support.idle_stop_motion_v1.motion_residual_actor import MotionResidualMLPModel

torch.set_num_threads(1)
SOURCE = ROOT / "deploy/models/idle_stop_motion_v1/idle_stop_motion_v1_46400.pt"


def train_config():
    return dict(
        num_steps_per_env=1, save_interval=100, logger="tensorboard", resume=False,
        baseline_checkpoint=str(SOURCE),
        actor=dict(class_name="MLPModel", hidden_dims=[512, 256, 128], activation="elu",
                   obs_normalization=False, distribution_cfg=dict(
                       class_name="GaussianDistribution", init_std=1., std_type="log")),
        critic=dict(class_name="MLPModel", hidden_dims=[512, 256, 128],
                    activation="elu", obs_normalization=False),
        obs_groups=dict(actor=["policy"], critic=["critic"]),
        algorithm=dict(
            class_name="task_support.idle_stop_motion_v1.training:StairGuardedPPO",
            value_loss_coef=1., use_clipped_value_loss=True, clip_param=.1,
            entropy_coef=.0001, num_learning_epochs=1, num_mini_batches=1,
            learning_rate=.0001, schedule="fixed", gamma=.99, lam=.95,
            desired_kl=.01, max_grad_norm=1., rnd_cfg=None, symmetry_cfg=None,
            motion_regularization_coefficient=.005, critic_warmup_iterations=20),
    )


class TensorEnv:
    num_envs = 32
    num_actions = 16
    device = "cpu"

    def __init__(self):
        self.unwrapped = self
        self.cfg = SimpleNamespace(trot=TrotSettings(), rewards=SimpleNamespace())
        self.obs = TensorDict({
            "policy": torch.randn(self.num_envs, 244),
            "critic": torch.randn(self.num_envs, 285),
            "motion_control": torch.randn(self.num_envs, 3),
            "trot_control": torch.zeros(self.num_envs, 6),
        }, [self.num_envs])
        self.obs["motion_control"][0] = 0
        self.obs["motion_control"][1:] = torch.tensor([.4, 0., 0.])

    def get_observations(self):
        return self.obs


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.cfg = TrotSettings()
        x, y = torch.meshgrid(torch.linspace(-.8, .8, 17), torch.linspace(-.5, .5, 11), indexing="ij")
        self.rays = torch.stack((x.flatten(), y.flatten()), -1)
        self.command = torch.tensor([[.4, 0., 0.]])

    def gate(self, heights, command=None):
        return ascent_geometry(heights[None], self.rays,
                               self.command if command is None else command, self.cfg).item()

    def test_flat_slope_downhill_and_invalid_do_not_activate(self):
        x = self.rays[:, 0]
        for height in (x * 0, x * .2, (x < 0).float() * .2, x * float("nan")):
            self.assertEqual(self.gate(height), 0)

    def test_nearby_upstairs_including_rear_riser_activates(self):
        x = self.rays[:, 0]
        for edge in (.3, -.3):
            self.assertGreater(self.gate((x > edge).float() * .2), .9)
        self.assertEqual(self.gate((x > .7).float() * .2), 0)

    def test_stop_reverse_turn_and_missing_central_column_close_gate(self):
        height = (self.rays[:, 0] > .2).float() * .2
        for command in ([0, 0, 0], [-.4, 0, 0], [.4, 0, .8], [.4, .5, 0]):
            self.assertEqual(self.gate(height, torch.tensor([command])), 0)
        height[self.rays[:, 0].abs() < .01] = float("nan")
        self.assertEqual(self.gate(height), 0)

    def test_contact_score_rejects_pace_bound_flight_and_standing(self):
        cfg = self.cfg
        phase = torch.tensor([.2])
        swing, height = phase_targets(phase, cfg)
        self.assertEqual(swing.tolist(), [[True, False, False, True]])
        good = gait_scores(~swing, swing, height, torch.ones_like(swing), phase, torch.ones(1), cfg)
        self.assertEqual(good["diagonal"].item(), 1)
        for airborne in ([0, 0, 0, 0], [1, 1, 1, 1], [1, 0, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0]):
            air = torch.tensor([airborne], dtype=torch.bool)
            bad = gait_scores(~air, air, height, torch.ones_like(air), phase, torch.ones(1), cfg)
            self.assertEqual(bad["diagonal"].item(), 0)
            self.assertLess(bad["phase"].item(), good["phase"].item())
        stationary = gait_scores(~swing, swing, height, torch.ones_like(swing), phase, torch.zeros(1), cfg)
        self.assertEqual(stationary["phase"].item() + stationary["diagonal"].item(), 0)
        for value in (.45, .95):
            self.assertFalse(phase_targets(torch.tensor([value]), cfg)[0].any())

    def test_event_reward_rejects_jump_and_consumes_supported_pair_once(self):
        zero = torch.zeros(1, 4)
        all_true = torch.ones(1, 4, dtype=torch.bool)
        none = ~all_true
        phase = torch.tensor([.2])
        armed = torch.zeros(1, 2, dtype=torch.bool)
        jump = paired_events(zero, zero, all_true, none, none, all_true, all_true, phase, armed, self.cfg)
        self.assertEqual(jump[0].item(), 0)
        # A four-wheel landing without an earlier qualified diagonal swing fails.
        landing = paired_events(zero, zero, none, all_true, all_true, none,
                                all_true, torch.tensor([.45]), armed, self.cfg)
        self.assertEqual(landing[0].item(), 0)
        air = torch.tensor([[True, False, False, True]])
        first = paired_events(zero, zero, air, none, ~air, air, all_true, phase, armed, self.cfg)
        self.assertEqual(first[0].item(), 1)
        repeated = paired_events(first[1], first[2], air, none, ~air, air,
                                 all_true, phase, first[3], self.cfg)
        self.assertEqual(repeated[0].item(), 0)
        landed = paired_events(first[1], zero, none, air, all_true, none,
                               all_true, torch.tensor([.45]), first[3], self.cfg)
        self.assertEqual(landed[0].item(), 1)
        self.assertFalse(landed[3].any())
        lost_support = paired_events(first[1], first[2], all_true, none, none, all_true,
                                     all_true, torch.tensor([.3]), first[3], self.cfg)
        self.assertFalse(lost_support[3].any())
        hop_landing = paired_events(lost_support[1], zero, none, all_true, all_true, none,
                                    all_true, torch.tensor([.45]), lost_support[3], self.cfg)
        self.assertEqual(hop_landing[0].item(), 0)
        expired = paired_events(first[1], first[2], none, none, ~air, air,
                                all_true, torch.tensor([.6]), first[3], self.cfg)
        self.assertFalse(expired[3].any())


class EnvironmentStateTests(unittest.TestCase):
    """Exercise the real context/reset functions with sensor tensors, no Isaac."""
    def fixture(self):
        module = ModuleType("isaaclab.envs")

        class Base:
            def _reset_idx(self, env_ids):
                pass

        module.ManagerBasedRLEnv = Base
        path = (ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity"
                / "mdp/stair_trot_v1_env.py")
        spec = importlib.util.spec_from_file_location("trot_env_under_test", path)
        loaded = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"isaaclab.envs": module}):
            spec.loader.exec_module(loaded)
        env = object.__new__(loaded.M20StairTrotV1Env)
        env.num_envs, env.device, env.step_dt, env.common_step_counter = 2, "cpu", .02, 0
        env.cfg = SimpleNamespace(trot=TrotSettings())
        env.action_manager = SimpleNamespace(action=torch.zeros(2, 16))
        env.episode_length_buf = torch.zeros(2, dtype=torch.long)
        command = torch.tensor([[.4, 0., 0.], [.4, 0., 0.]])
        env.command_manager = SimpleNamespace(get_command=lambda name: command)
        x, y = torch.meshgrid(torch.linspace(-.8, .8, 17), torch.linspace(-.5, .5, 11), indexing="ij")
        xy = torch.stack((x.flatten(), y.flatten()), -1)
        heights = (xy[:, 0] > .1).float() * .2
        hits = torch.cat((xy, heights[:, None]), -1).repeat(2, 1, 1)
        wheel_xy = torch.tensor([[.3, .2], [.3, -.2], [-.3, .2], [-.3, -.2]])
        wheel_z = torch.tensor([.2, .2, 0., 0.]) + loaded.TUNING["wheel_radius"]
        wheel_pos = torch.cat((wheel_xy, wheel_z[:, None]), -1).repeat(2, 1, 1)
        env.scene = {
            "robot": SimpleNamespace(data=SimpleNamespace(
                body_pos_w=wheel_pos, root_lin_vel_b=command.clone(),
                root_pos_w=torch.zeros(2, 3),
                root_quat_w=torch.tensor([1., 0., 0., 0.]).repeat(2, 1),
                body_quat_w=torch.tensor([1., 0., 0., 0.]).repeat(2, 4, 1),
                body_link_lin_vel_w=torch.zeros(2, 4, 3),
                body_ang_vel_w=torch.zeros(2, 4, 3),
                root_ang_vel_b=torch.zeros(2, 3),
                projected_gravity_b=torch.tensor([0., 0., -1.]).repeat(2, 1),
                applied_torque=torch.zeros(2, 16), joint_vel=torch.zeros(2, 16))),
            "height_scanner": SimpleNamespace(ray_starts=torch.cat((xy, torch.zeros_like(heights[:, None])), -1),
                                               data=SimpleNamespace(ray_hits_w=hits)),
            "contact_forces": SimpleNamespace(data=SimpleNamespace(
                net_forces_w=torch.tensor([0., 0., 50.]).repeat(2, 4, 1),
                net_forces_w_history=torch.tensor([0., 0., 50.]).repeat(2, 3, 4, 1),
                current_air_time=torch.zeros(2, 4))),
        }
        entities = [SimpleNamespace(name=name, body_ids=[0, 1, 2, 3])
                    for name in ("robot", "height_scanner", "contact_forces")]
        return loaded, env, entities, command

    def test_partial_reset_does_not_advance_other_env_twice(self):
        module, env, entities, _ = self.fixture()
        module._context(env, *entities)
        env.common_step_counter = 1
        module._context(env, *entities)
        before = {key: env._trot_state[key][1].clone()
                  for key in ("phase", "distance", "gate")}
        samples = env._trot_state["stats"]["steps"][1].clone()
        env._reset_idx(torch.tensor([0]))
        module._context(env, *entities)
        for key, value in before.items():
            self.assertTrue(torch.equal(value, env._trot_state[key][1]), key)
        self.assertEqual(env._trot_state["stats"]["steps"][1], samples)
        self.assertEqual(env._trot_state["phase"][0], 0.)

    def test_command_resample_closes_observation_gate_in_same_tick(self):
        module, env, entities, command = self.fixture()
        self.assertTrue((module.trot_control(env, *entities)[:, 0] > 0).all())
        command.zero_()
        self.assertTrue((module.trot_control(env, *entities) == 0).all())

    def test_world_progress_cannot_be_reearned_after_gate_flicker(self):
        module, env, entities, command = self.fixture()
        module._context(env, *entities)
        robot = env.scene["robot"].data
        robot.root_pos_w[:, 0] = .1
        env.common_step_counter += 1
        self.assertTrue((module._context(env, *entities)["progress"] > 0).all())
        command.zero_()
        robot.root_pos_w[:, 0] = 0
        env.common_step_counter += 1
        module._context(env, *entities)
        command[:, 0] = .4
        robot.root_pos_w[:, 0] = .1
        env.common_step_counter += 1
        self.assertTrue((module._context(env, *entities)["progress"] == 0).all())

    def test_episode_metrics_weight_by_actual_ascent_samples(self):
        from task_support.idle_stop_motion_v1.runner import pool_episode_metrics
        module, env, entities, _ = self.fixture()
        module._context(env, *entities)
        metrics = module.trot_metrics(env, torch.tensor([0, 1]))
        self.assertEqual(metrics["active_samples"].item(), 2)
        self.assertEqual(metrics["__weighted_count__/diagonal_observed"].item(), 2)
        pooled = pool_episode_metrics([metrics])
        self.assertNotIn("__weighted_count__/diagonal_observed", pooled[0])
        self.assertEqual(pooled[0]["diagonal_observed"].item(), 0.)


class ActorTests(unittest.TestCase):
    def runner(self):
        torch.manual_seed(11)
        with redirect_stdout(StringIO()):
            runner = StairTrotRunner(TensorEnv(), train_config(), device="cpu")
        runner.initialize_task({})
        return runner

    def teacher(self, obs):
        config = deepcopy(train_config()["actor"])
        config.pop("class_name")
        actor = MotionResidualMLPModel(obs, {"actor": ["policy"]}, "actor", 16, **config)
        actor.load_state_dict(torch.load(SOURCE, map_location="cpu", weights_only=True)["actor_state_dict"], strict=True)
        return actor.eval()

    def test_full_teacher_equal_at_initialization_and_flat_after_arbitrary_residual(self):
        runner = self.runner()
        actor, obs = runner.alg.actor, runner.env.obs
        teacher = self.teacher(obs)
        with torch.no_grad():
            baseline = teacher(obs)
            obs["trot_control"][:, 0] = 1
            self.assertTrue(torch.equal(actor(obs), baseline))
            # This is deliberate test corruption, not learning/optimizer execution.
            actor.trot_mlp[-1].bias.fill_(2.)
            changed = actor(obs)
            self.assertFalse(torch.equal(changed, baseline))
            self.assertTrue(((changed - baseline).abs() <= actor.trot_limits + 1.e-6).all())
            obs["trot_control"].zero_()
            self.assertTrue(torch.equal(actor(obs), baseline))
            torch.manual_seed(91)
            reference_sample = teacher(obs, stochastic_output=True)
            torch.manual_seed(91)
            self.assertTrue(torch.equal(actor(obs, stochastic_output=True), reference_sample))
        self.assertFalse(any(p.requires_grad for name, p in actor.named_parameters()
                             if not name.startswith("trot_mlp.")))
        self.assertTrue(all(p.requires_grad for p in actor.trot_mlp.parameters()))
        self.assertFalse(runner.alg.optimizer.state)

    def test_gradient_boundary_without_optimizer_update(self):
        runner = self.runner()
        actor, obs = runner.alg.actor, runner.env.obs
        actor(obs).sum().backward()
        self.assertTrue(all(p.grad is None or torch.count_nonzero(p.grad) == 0
                            for p in actor.parameters()))
        actor.zero_grad(set_to_none=True)
        obs["trot_control"][:, 0] = 1
        actor(obs).sum().backward()
        self.assertGreater(actor.trot_mlp[-1].bias.grad.abs().sum(), 0)
        self.assertTrue(all(p.grad is None for name, p in actor.named_parameters()
                            if not name.startswith("trot_mlp.")))
        self.assertFalse(runner.alg.optimizer.state)

    def test_actor_blocks_stale_positive_gate_after_command_change(self):
        runner = self.runner()
        actor, obs = runner.alg.actor, runner.env.obs
        teacher = self.teacher(obs)
        with torch.no_grad():
            actor.trot_mlp[-1].bias.fill_(2.)
            obs["trot_control"][:, 0] = 1
            for command in ([0., 0., 0.], [-.4, 0., 0.], [.4, 0., .8]):
                obs["motion_control"][:] = torch.tensor(command)
                self.assertTrue(torch.equal(actor(obs), teacher(obs)))

    def test_full_save_load_and_wrong_task_rejection_without_updates(self):
        runner = self.runner()
        state = runner.alg.save()
        state.update(iter=7, infos={})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save(state, path)
            resumed = self.runner()
            resumed.load(path)
            self.assertEqual(resumed.current_learning_iteration, 7)
            for name, value in runner.alg.actor.state_dict().items():
                self.assertTrue(torch.equal(value, resumed.alg.actor.state_dict()[name]))
            state["m20_training_manifest"]["task_contract"]["trot_control_size"] = 3
            torch.save(state, path)
            with self.assertRaisesRegex(ValueError, "configuration"):
                resumed.load(path)
            state["m20_training_manifest"]["task_contract"]["trot_control_size"] = 6
            state["m20_training_manifest"]["trot_settings"]["swing_fraction"] = .9
            torch.save(state, path)
            with self.assertRaisesRegex(ValueError, "configuration"):
                resumed.load(path)
        with self.assertRaisesRegex(ValueError, "resume requires StairTrot"):
            self.runner().load(SOURCE)


if __name__ == "__main__":
    unittest.main()
