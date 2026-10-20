"""Residual-task initialization, full resume and CPU PPO freeze boundaries."""
from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from tensordict import TensorDict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/rl_training"))
from task_support.idle_stop_motion_v1.runner import (
    IdleStopMotionRunner, _descent_guide_scale, validate_resume_manifest,
)

CHECKPOINT = ROOT / "deploy/models/idle_stop_motion_v1/idle_stop_motion_v1_46400.pt"
torch.set_num_threads(1)


def train_cfg():
    return dict(
        num_steps_per_env=1, save_interval=100, logger="tensorboard", resume=False,
        actor=dict(class_name="MLPModel", hidden_dims=[512, 256, 128], activation="elu",
                   obs_normalization=False, distribution_cfg=dict(
                       class_name="GaussianDistribution", init_std=1.0, std_type="log")),
        critic=dict(class_name="MLPModel", hidden_dims=[512, 256, 128],
                    activation="elu", obs_normalization=False),
        obs_groups=dict(actor=["policy", "motion_control"], critic=["critic", "motion_control"]),
        algorithm=dict(
            class_name="task_support.idle_stop_motion_v1.training:StairGuardedPPO",
            value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=.1,
            entropy_coef=1.e-4, num_learning_epochs=1, num_mini_batches=1,
            learning_rate=1.e-4, schedule="fixed", gamma=.99, lam=.95,
            desired_kl=.01, max_grad_norm=1.0, rnd_cfg=None, symmetry_cfg=None,
            anchor_coefficient=.1, critic_warmup_iterations=20,
            reference_std_multiplier=1.02, anchor_stair_weight=1.,
            anchor_stop_weight=.02, anchor_stop_command_epsilon=1.e-4,
            motion_regularization_coefficient=.02,
        ),
        descent_guide_iters=20000, descent_guide_decay_iters=10000,
    )


class ObservationEnv:
    """Tensor fixture; contains no simulator and has no physics step method."""
    num_envs = 4
    num_actions = 16
    device = "cpu"
    common_step_counter = 0

    def __init__(self):
        self.unwrapped = self
        self.cfg = SimpleNamespace(teacher_detailed_diagnostics=False, rewards=SimpleNamespace())
        self.obs = TensorDict({
            "policy": torch.randn(4, 244), "critic": torch.randn(4, 285),
            "motion_control": torch.tensor([[0., 0., 0.], [.5, 0., 0.],
                                            [0., 0., .6], [-.5, 0., 0.]]),
        }, [4])

    def get_observations(self):
        return self.obs


class MotionTaskTests(unittest.TestCase):
    def runner(self):
        torch.manual_seed(42)
        with redirect_stdout(StringIO()):
            return IdleStopMotionRunner(ObservationEnv(), train_cfg(), device="cpu")

    def test_baseline_initialization_keeps_critic_fresh_and_freezes_base(self):
        runner = self.runner()
        critic = deepcopy(runner.alg.critic.state_dict())
        runner.initialize_task({})
        actor = runner.alg.actor
        self.assertTrue(all(torch.equal(v, runner.alg.critic.state_dict()[k]) for k, v in critic.items()))
        self.assertFalse(any(p.requires_grad for p in actor.mlp.parameters()))
        self.assertFalse(any(p.requires_grad for p in actor.distribution.parameters()))
        self.assertTrue(all(p.requires_grad for p in actor.residual_mlp.parameters()))
        self.assertFalse(runner.alg.optimizer.state)
        self.assertIsNone(runner.alg.reference_actor)
        with torch.inference_mode():
            self.assertTrue(torch.equal(actor(runner.env.obs), actor.mlp(runner.env.obs["policy"])))
        with self.assertRaisesRegex(ValueError, "recorded idle-stop"):
            self.runner().initialize_task({"init_actor_from": str(CHECKPOINT)})

    def test_real_full_resume_preserves_optimizer_counters_and_finished_guide(self):
        saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
        runner = self.runner()
        runner.load(CHECKPOINT)
        runner.initialize_task({"resume": True})
        self.assertEqual(runner.current_learning_iteration, saved["iter"])
        self.assertEqual(runner.alg.guard_updates, saved["preservation_updates"])
        for name in ("actor", "critic"):
            self.assertTrue(all(torch.equal(v, getattr(runner.alg, name).state_dict()[k])
                                for k, v in saved[name + "_state_dict"].items()))
        actual = runner.alg.optimizer.state_dict()
        expected = saved["optimizer_state_dict"]
        self.assertEqual(actual["param_groups"], expected["param_groups"])
        self.assertEqual(set(actual["state"]), set(expected["state"]))
        for key, fields in expected["state"].items():
            for name, value in fields.items():
                other = actual["state"][key][name]
                self.assertTrue(torch.equal(value, other) if torch.is_tensor(value) else value == other)
        self.assertIsNone(runner.alg.reference_actor)
        schedule = runner.env._m20_descent_entry_guide_schedule
        self.assertEqual(schedule["iteration_offset"], saved["iter"])
        self.assertEqual(_descent_guide_scale(schedule["iteration_offset"],
                         schedule["guided_iterations"], schedule["decay_iterations"]), 0.)

    def test_resume_contract_rejects_wrong_profile_and_source_hash(self):
        saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
        manifest = saved["m20_training_manifest"]
        validate_resume_manifest(manifest)
        canonical = deepcopy(manifest)
        canonical["actor_model_config"]["class_name"] = (
            "task_support.idle_stop_motion_v1.motion_residual_actor:MotionResidualMLPModel")
        validate_resume_manifest(canonical)
        for field, value in (("m20_profile", "wrong"), ("motion_control_version", "wrong")):
            bad = deepcopy(manifest)
            bad[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_resume_manifest(bad)
        bad = deepcopy(manifest)
        bad["source_sha256"]["motion_residual_actor.py"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "source"):
            validate_resume_manifest(bad)
        for field in ("preservation_updates", "optimizer_state_dict", "iter"):
            bad = dict(saved)
            bad.pop(field)
            runner = self.runner()
            with self.subTest(missing=field), patch(
                    "task_support.idle_stop_motion_v1.runner.torch.load", return_value=bad):
                with self.assertRaisesRegex(ValueError, field):
                    runner.load(CHECKPOINT)

    def test_cpu_updates_respect_warmup_and_only_train_motion_residual(self):
        runner = self.runner()
        runner.initialize_task({})
        alg, obs = runner.alg, runner.env.obs
        original = deepcopy(alg.actor.state_dict())

        def update():
            with torch.inference_mode():
                alg.act(obs)
                alg.process_env_step(obs, torch.ones(4), torch.zeros(4), {})
                alg.compute_returns(obs)
            return alg.update()

        first = update()
        self.assertEqual(first["critic_warmup"], 1.)
        self.assertTrue(all(torch.equal(value, alg.actor.state_dict()[key])
                            for key, value in original.items()))
        alg.guard_updates = 20
        second = update()
        self.assertEqual(second["critic_warmup"], 0.)
        after = alg.actor.state_dict()
        self.assertTrue(all(torch.equal(value, after[key]) for key, value in original.items()
                            if not key.startswith("residual_mlp.")))
        self.assertTrue(any(not torch.equal(value, after[key]) for key, value in original.items()
                           if key.startswith("residual_mlp.")))


if __name__ == "__main__":
    unittest.main()
