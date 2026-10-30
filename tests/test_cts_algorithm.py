"""Small CPU contract tests: actual CTS optimizer wiring, never simulation."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/rl_training"))
from task_support.cts.model import build_models, sample_observations, teacher_observations
from task_support.cts.algorithm import CTSAlgorithm, balanced_mean, clipped_policy_objective
from task_support.stair_quiet_v1_safe.actor import SafeResidualActor

torch.set_num_threads(1)


def configuration():
    return dict(teacher_fraction=.75, cohort_seed=42, learning_rate=3.e-4, student_learning_rate=1.e-3,
                gamma=.99, lam=.95, clip_param=.1, num_learning_epochs=2, num_mini_batches=2,
                value_loss_coef=1., max_grad_norm=1., student_learning_epochs=2, student_batch_size=8,
                latent_loss_weight=1., decoder_hidden_dims=[16, 8], decoder_bound=[.18, .35, .35]*4+[.16]*4,
                anchor_coefficient=.5, entropy_coef=0., desired_kl=.01)


def fixture():
    torch.manual_seed(12)
    obs = sample_observations(8).apply(lambda value: torch.randn_like(value)*.2)
    obs["motion_control"][:, 0] = .4
    actor_cfg = dict(hidden_dims=[32, 16, 8], activation="elu", obs_normalization=False,
                     residual_hidden_dims=[16, 8],
                     distribution_cfg={"class_name": "GaussianDistribution", "init_std": .1, "std_type": "log"})
    teacher = SafeResidualActor(teacher_observations(obs), {"actor": ["policy"]}, "actor", 16, **deepcopy(actor_cfg))
    with torch.no_grad():
        teacher.residual_mlp[-1].weight.fill_(.01)
        teacher.safety_mlp[-1].weight.fill_(.02)
    student = MLPModel(obs, {"student": ["policy"]}, "student", 16, hidden_dims=[32, 16, 8],
                       activation="elu", obs_normalization=False,
                       distribution_cfg={"class_name": "GaussianDistribution", "init_std": .1, "std_type": "log"})
    critic = MLPModel(obs, {"critic": ["teacher_critic", "motion_control"]}, "critic", 1,
                     hidden_dims=[32, 16, 8], activation="elu", obs_normalization=False)
    saved_teacher = {"actor_state_dict": deepcopy(teacher.state_dict()), "critic_state_dict": deepcopy(critic.state_dict()),
                     "m20_training_manifest": {"actor_model_config": {"class_name": "SafeResidualActor", **actor_cfg}}}
    saved_student = {"student_state_dict": deepcopy(student.state_dict())}
    cfg = configuration()
    models = build_models(saved_teacher, saved_student, cfg)
    return models, CTSAlgorithm(models, cfg, 8, 4), obs, teacher, student, critic, saved_teacher, saved_student


def collect(algorithm, obs):
    with torch.inference_mode():
        for step in range(4):
            actions = algorithm.act(obs)
            rewards = torch.arange(8, dtype=torch.float32)*.1 + step*.2 - actions.square().sum(-1)
            algorithm.process_env_step(obs, rewards, torch.zeros(8, dtype=torch.bool), {})
        algorithm.compute_returns(obs)


class CTSModelTests(unittest.TestCase):
    def test_warm_start_and_export_preserve_actual_source_paths(self):
        models, algorithm, obs, teacher, student, critic, *_ = fixture()
        with torch.no_grad():
            expected_teacher = teacher(teacher_observations(obs))
            expected_student = student(obs)
            actual, latent = models.mean(obs, algorithm.teacher_mask)
            expected = torch.where(algorithm.teacher_mask[:, None], expected_teacher, expected_student)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(models.value(obs, latent), critic(obs).squeeze(-1), rtol=1.e-6, atol=1.e-7)
            exported = torch.jit.script(models.student_export())
            torch.testing.assert_close(exported(obs["policy"], obs["motion_control"]), expected_student, rtol=0, atol=0)
            models.decoder.network[-1].bias.fill_(1.)
            zero = torch.zeros_like(obs["motion_control"])
            torch.testing.assert_close(models.decoder(obs["policy"][:, :57], latent, zero), latent, rtol=0, atol=0)

    def test_student_export_has_no_privileged_dependency(self):
        models, _, obs, *_ = fixture()
        exported = models.student_export()
        before = exported(obs["policy"], obs["motion_control"])
        obs["teacher"].fill_(1000.)
        obs["teacher_critic"].fill_(-1000.)
        obs["safety"].fill_(700.)
        torch.testing.assert_close(exported(obs["policy"], obs["motion_control"]), before, rtol=0, atol=0)

    def test_critic_detaches_encoders(self):
        models, algorithm, obs, *_ = fixture()
        latent = models.latent(obs, algorithm.teacher_mask, detach_student=False)
        models.value(obs, latent).sum().backward()
        self.assertTrue(any(p.grad is not None for p in models.critic.parameters()))
        self.assertTrue(all(p.grad is None for p in models.teacher.parameters()))
        self.assertTrue(all(p.grad is None for p in models.student.parameters()))


class CTSAlgorithmTests(unittest.TestCase):
    def test_group_balancing_and_clipping_known_cases(self):
        self.assertEqual(float(balanced_mean(torch.tensor([2., 2., 2., 10.]), torch.tensor([1, 1, 1, 0]).bool())), 6.)
        with self.assertRaises(ValueError):
            balanced_mean(torch.ones(4), torch.ones(4, dtype=torch.bool))
        loss = clipped_policy_objective(torch.tensor([2., .5]).log(), torch.zeros(2), torch.tensor([1., -1.]), .1)
        torch.testing.assert_close(loss, torch.tensor([-1.1, .9]))

    def test_cohort_permutation_does_not_alias_idle_practice_indices(self):
        models, algorithm, *_ = fixture()
        larger = CTSAlgorithm(models, algorithm.cfg, 64, 4)
        idle = torch.arange(64) % 4 == 0
        self.assertTrue((larger.teacher_mask & idle).any())
        self.assertTrue((~larger.teacher_mask & idle).any())
        self.assertTrue((~larger.teacher_mask & ~idle).any())

    def test_initial_ratios_use_executed_cohort_policy(self):
        models, algorithm, obs, *_ = fixture()
        action = algorithm.act(obs)
        mean, _ = models.mean(obs, algorithm.teacher_mask)
        log_prob = torch.distributions.Normal(mean, models.log_std.exp()).log_prob(action).sum(-1)
        torch.testing.assert_close((log_prob-algorithm.pending["log_prob"]).exp(), torch.ones(8), rtol=0, atol=0)
        wrong = torch.distributions.Normal(models.teacher_latent(obs), models.log_std.exp()).log_prob(action).sum(-1)
        self.assertGreater(float((wrong[~algorithm.teacher_mask]-log_prob[~algorithm.teacher_mask]).abs().max()), 1.e-3)
        self.assertEqual(int(algorithm.teacher_mask.sum()), 6)

    def test_ppo_and_distillation_update_disjoint_parameters(self):
        models, algorithm, obs, *_ = fixture()
        collect(algorithm, obs)
        before = deepcopy(models.state_dict())
        algorithm.ppo_update()
        after = models.state_dict()
        self.assertTrue(all(torch.equal(v, after[k]) for k, v in before.items() if k.startswith("student.")))
        self.assertTrue(all(torch.equal(v, after[k]) for k, v in before.items() if k.startswith(("teacher.mlp.", "teacher.distribution.", "reference_teacher."))))
        for prefix in ("teacher.residual_mlp.", "teacher.safety_mlp.", "decoder.", "critic."):
            self.assertTrue(any(not torch.equal(v, after[k]) for k, v in before.items() if k.startswith(prefix)), prefix)
        after_ppo = deepcopy(after)
        algorithm.distill_update()
        self.assertTrue(any(not torch.equal(v, models.state_dict()[k]) for k, v in after_ppo.items() if k.startswith("student.mlp.")))
        self.assertTrue(all(torch.equal(v, models.state_dict()[k]) for k, v in after_ppo.items() if not k.startswith("student.")))

    def test_timeout_bootstrap_does_not_cross_episode_boundary(self):
        models, algorithm, obs, *_ = fixture()
        algorithm.num_steps_per_env = 1
        with torch.no_grad():
            for parameter in models.critic.parameters(): parameter.zero_()
            models.critic.mlp[-1].bias.fill_(2.)
        algorithm.act(obs)
        done = torch.zeros(8, dtype=torch.bool)
        done[:2] = True
        timeout = torch.zeros_like(done)
        timeout[0] = True
        algorithm.process_env_step(obs, torch.ones(8), done, {"time_outs": timeout})
        algorithm.compute_returns(obs)
        self.assertAlmostEqual(float(algorithm.returns[0, 0]), 2.98, places=5)
        self.assertAlmostEqual(float(algorithm.returns[0, 1]), 1., places=5)
        self.assertAlmostEqual(float(algorithm.returns[0, 2]), 2.98, places=5)

    def test_kl_gate_stops_before_another_parameter_update(self):
        models, algorithm, obs, *_ = fixture()
        collect(algorithm, obs)
        with torch.no_grad():
            models.decoder.network[-1].bias.fill_(3.)
        before = deepcopy(models.state_dict())
        result = algorithm.ppo_update()
        self.assertEqual(result["ppo_minibatches"], 0)
        self.assertTrue(all(torch.equal(value, models.state_dict()[key]) for key, value in before.items()))

    def test_supervision_recomputes_current_teacher_after_ppo(self):
        models, algorithm, obs, *_ = fixture()
        collect(algorithm, obs)
        old_target = models.teacher_latent(algorithm._data()["obs"]).detach().clone()
        algorithm.ppo_update()
        current_target = models.teacher_latent(algorithm._data()["obs"]).detach().clone()
        self.assertFalse(torch.equal(old_target, current_target))
        with patch.object(models, "teacher_latent", wraps=models.teacher_latent) as target_call:
            algorithm.distill_update()
            self.assertEqual(target_call.call_count, 1)

    def test_full_checkpoint_roundtrip_and_fail_closed(self):
        models, algorithm, obs, _, _, _, teacher, student = fixture()
        collect(algorithm, obs)
        algorithm.update()
        saved = deepcopy(algorithm.save())
        rebuilt = build_models(teacher, student, algorithm.cfg)
        restored = CTSAlgorithm(rebuilt, algorithm.cfg, 8, 4)
        restored.load(saved)
        self.assertEqual(restored.num_updates, 1)
        self.assertTrue(restored.optimizer.state)
        self.assertTrue(restored.student_optimizer.state)
        for key, value in models.state_dict().items():
            torch.testing.assert_close(value, rebuilt.state_dict()[key], rtol=0, atol=0)
        saved["teacher_mask"] = ~saved["teacher_mask"]
        with self.assertRaises(ValueError): restored.load(saved)
        bad = deepcopy(algorithm.save())
        bad["models_state_dict"]["log_std"][0] = float("nan")
        with self.assertRaises(ValueError): restored.load(bad)

    def test_resume_rejections_are_atomic_for_models_optimizer_and_rng(self):
        _, trained, obs, _, _, _, teacher, student = fixture()
        collect(trained, obs)
        trained.update()
        saved = deepcopy(trained.save())
        models = build_models(teacher, student, trained.cfg)
        target = CTSAlgorithm(models, trained.cfg, 8, 4)
        before = deepcopy(target.save())
        def assert_equal(actual, expected):
            if isinstance(expected, torch.Tensor):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            elif isinstance(expected, dict):
                self.assertEqual(set(actual), set(expected))
                for key in expected: assert_equal(actual[key], expected[key])
            elif isinstance(expected, list):
                self.assertEqual(len(actual), len(expected))
                for a, e in zip(actual, expected): assert_equal(a, e)
            else:
                self.assertEqual(actual, expected)
        def bounds_bad(state): state["models_state_dict"]["decoder.bounds"][0] = 500.
        def lr_bad(state): state["optimizer_state_dict"]["param_groups"][0]["lr"] = -1.
        def student_lr_bad(state): state["student_optimizer_state_dict"]["param_groups"][0]["lr"] = .9
        def model_dtype_bad(state): state["models_state_dict"]["decoder.network.0.weight"] = state["models_state_dict"]["decoder.network.0.weight"].double()
        def moment_bad(state):
            first = next(iter(state["optimizer_state_dict"]["state"].values()))
            first["exp_avg"] = first["exp_avg"].flatten()[:1]
        def squared_moment_bad(state):
            first = next(iter(state["optimizer_state_dict"]["state"].values()))
            first["exp_avg_sq"].fill_(-1.)
        def rng_bad(state): state["torch_rng_state"] = torch.zeros(3, dtype=torch.uint8)
        for corrupt in (bounds_bad, lr_bad, student_lr_bad, model_dtype_bad, moment_bad, squared_moment_bad, rng_bad):
            with self.subTest(corruption=corrupt.__name__):
                bad = deepcopy(saved)
                corrupt(bad)
                with self.assertRaises(ValueError): target.load(bad)
                assert_equal(target.save(), before)
        target.load(saved)
        self.assertEqual(target.num_updates, trained.num_updates)


if __name__ == "__main__":
    unittest.main()
