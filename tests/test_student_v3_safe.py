"""Complete Safe labels, sensor isolation, persistence and fail-closed deployment.

All checks are CPU-only and forbid optimizer updates. Actual teacher weights
are used when available; simulator wiring is checked separately by --validate-only.
"""
from copy import deepcopy
from dataclasses import asdict
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/rl_training"))
sys.path.insert(0, str(ROOT / "deploy/deploy_mujoco"))
from task_support.student_v3_safe.configuration import TASK, CONFIG_PATH, TEACHER_GROUPS
from task_support.student_v3_safe import entry, deployment
from task_support.student_v3_safe.teacher import SafeTeacherModel, SafeResidualActor, teacher_manifest
from m20_student_v1.distillation import StudentV1Distillation
from m20_student_v1.contract import observation_contract
from task_support.stair_quiet_v1.deployment import execution_contract
from task_support.stair_quiet_v1_width.signals import failure_mask
from student_v1_adapter import load_student, profile_config

CHECKPOINT = ROOT / "logs/rsl_rl/deeprobotics_m20_stair_quiet_v1_safe/2026-10-22_13-29-16_teacher_safe_from_width10000/model_18400.pt"
spec = importlib.util.spec_from_file_location("student_safe_cli_test", ROOT / "scripts/reinforcement_learning/rsl_rl/distill_student.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)
torch.set_num_threads(1)


class FailureSemanticsTests(unittest.TestCase):
    def test_completed_success_is_not_failure_but_simultaneous_failure_wins(self):
        values = {"tail_success": torch.tensor([True, True, False]),
                  "tail_failure": torch.tensor([False, True, False]),
                  "timeout": torch.tensor([False, False, True])}
        manager = SimpleNamespace(active_terms=list(values), get_term=lambda name: values[name],
                                  get_term_cfg=lambda name: SimpleNamespace(time_out=name == "timeout"))
        result = failure_mask(SimpleNamespace(termination_manager=manager, num_envs=3, device="cpu"))
        self.assertEqual(result.tolist(), [False, True, False])

    def test_safe_timeout_is_failed_and_success_with_failure_is_failed(self):
        from task_support.student_v3_safe.environment import completed_failure
        terms = {"tail_success": torch.tensor([False, True, True, False]),
                 "tail_failure": torch.tensor([False, False, True, False]),
                 "timeout": torch.tensor([True, False, False, False])}
        manager = SimpleNamespace(active_terms=list(terms), get_term=lambda name: terms[name],
                                  get_term_cfg=lambda name: SimpleNamespace(time_out=name == "timeout"),
                                  terminated=terms["tail_success"] | terms["tail_failure"], time_outs=terms["timeout"])
        env = SimpleNamespace(termination_manager=manager, num_envs=4, device="cpu",
                              tail_tracker=SimpleNamespace(success=terms["tail_success"]))
        self.assertEqual(completed_failure(env).tolist(), [True, False, True, False])


@unittest.skipUnless(CHECKPOINT.is_file(), "real v1_safe model_18400 checkpoint required")
class SafeStudentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
        cls.manifest = teacher_manifest(cls.saved)

    def make(self):
        args = cli.parse_args(["--task", TASK, "--teacher-checkpoint", str(CHECKPOINT), "--device", "cpu", "--num-envs", "4"])
        training, perception, n, meta, manifest = entry.settings(args, self.saved)
        cfg = deepcopy(training)
        cfg.update(num_steps_per_env=1, multi_gpu=None)
        cfg["algorithm"].update(batch_size=4, replay_rollouts=2, hard_replay_rollouts=1)
        obs = entry.sample_observations(4).apply(lambda value: torch.randn_like(value))
        obs["motion_control"][0] = 0.
        alg = StudentV1Distillation.construct_algorithm(obs, SimpleNamespace(num_envs=4, num_actions=16), deepcopy(cfg), "cpu")
        # This fixture has no physical course; exploration is explicitly off.
        alg.pg_provider = lambda: torch.zeros(4, dtype=torch.bool)
        entry.initialize(alg, self.saved, args)
        control = execution_contract(self.saved, required=True)
        alg.provenance = deployment.provenance_for({
            "version": "v1", "task": TASK, "task_version": "student_v3_safe", "num_envs": 4,
            "teacher_manifest": manifest, "training_config": cfg, "perception": asdict(perception),
            "contract": observation_contract(4),
            "command_feedback": {"version": "legacy_v1_v3_5", "tuning": manifest["stair_tuning"]},
        }, control)
        return args, alg, obs, cfg, perception

    def test_default_configuration_and_plain_student_initialization(self):
        args, alg, obs, cfg, _ = self.make()
        self.assertEqual(args.config, CONFIG_PATH)
        self.assertEqual(cfg["obs_groups"], {"student": ["policy"], "teacher": TEACHER_GROUPS})
        self.assertEqual(cfg["history_steps"], 4)
        initial = torch.load(args.init_student, map_location="cpu", weights_only=True)["student_state_dict"]
        for key, value in initial.items():
            self.assertTrue(torch.equal(alg.student.state_dict()[key], value))
        self.assertEqual(len(alg.optimizer.state), 0)
        args.xr = True
        with self.assertRaisesRegex(ValueError, "XR"):
            entry.settings(args, self.saved)

    def test_complete_teacher_parity_safety_contribution_and_zero_raw_command(self):
        _, alg, obs, cfg, _ = self.make()
        packed = SafeTeacherModel.pack(obs)
        model = deepcopy(cfg["teacher"])
        model.pop("class_name")
        reference = SafeResidualActor(packed, {"actor": ["policy"]}, "actor", 16, **model)
        reference.load_state_dict(self.saved["actor_state_dict"], strict=True)
        with torch.inference_mode():
            actual = alg.teacher(obs)
            torch.testing.assert_close(actual, reference(packed), atol=0, rtol=0)
            torch.testing.assert_close(actual[0], alg.teacher.mlp(obs["teacher"])[0], atol=0, rtol=0)
            for parameter in reference.safety_mlp.parameters():
                parameter.zero_()
            self.assertGreater(float((actual[1:] - reference(packed)[1:]).abs().max()), 1e-6)
        self.assertEqual(set(alg.teacher.state_dict()), set(self.saved["actor_state_dict"]))
        self.assertFalse(any(parameter.requires_grad for parameter in alg.teacher.parameters()))

    def test_student_ignores_all_privileged_groups(self):
        _, alg, obs, _, _ = self.make()
        changed = obs.clone()
        for name in TEACHER_GROUPS:
            changed[name] = torch.randn_like(changed[name]) * 10
        with torch.inference_mode():
            torch.testing.assert_close(alg.student(obs), alg.student(changed), atol=0, rtol=0)

    def test_labels_roundtrip_and_success_allows_teacher_decay(self):
        with patch.object(torch.optim.Adam, "step", side_effect=AssertionError("No optimizer update authorized")):
            _, alg, obs, cfg, _ = self.make()
            with torch.inference_mode():
                expected = alg.teacher(obs).clone()
                alg.act(obs)
                self.assertTrue(torch.equal(alg.transition.privileged_actions, expected))
                alg.teacher_mask.zero_()
                tags = {key: torch.zeros(4, dtype=torch.long) for key in ("phase", "terrain", "direction", "terrain_level")}
                tags["terminated"] = torch.zeros(4, dtype=torch.bool)
                alg.process_env_step(obs, torch.zeros(4), torch.ones(4), {"student_v1": tags})
            self.assertEqual(alg.failure_ema, 0.)
            self.assertEqual(alg.student_episodes, 4)
            self.assertTrue(torch.equal(alg.replay_labels[:4], expected))
            saved = alg.save()
            _, restored, _, _, _ = self.make()
            restored.load(saved)
            with torch.inference_mode():
                torch.testing.assert_close(restored.teacher(obs), expected, atol=0, rtol=0)
            self.assertEqual(restored.num_updates, 0)
            self.assertEqual(len(restored.optimizer.state), 0)

    def test_rejects_wrong_teacher_missing_branch_changed_hash_and_configuration(self):
        for change in ("groups", "hash", "class"):
            bad = deepcopy(self.saved)
            manifest = bad["m20_training_manifest"]
            if change == "groups":
                manifest["actor_input_groups"].pop("safety")
            elif change == "class":
                manifest["actor_model_config"]["class_name"] = "ordinary MLP"
            else:
                sources = manifest["noise_refinement"]["source_sha256"]
                sources["source/rl_training/task_support/stair_quiet_v1_safe/actor.py"] = "0" * 64
            with self.subTest(change=change), self.assertRaises(ValueError):
                teacher_manifest(bad)
        args, alg, _, _, _ = self.make()
        bad_weights = deepcopy(self.saved)
        bad_weights["actor_state_dict"].pop("safety_mlp.4.weight")
        with self.assertRaises(RuntimeError):
            alg.load(bad_weights, {"teacher": True}, True)
        args.history_steps = 1
        with self.assertRaisesRegex(ValueError, "789D"):
            entry.settings(args, self.saved)
        args.history_steps = None
        args.init_student = CHECKPOINT
        with self.assertRaisesRegex(ValueError, "plain student"):
            entry.initialize(alg, self.saved, args)

    def test_export_load_and_execution_controller_required(self):
        args, alg, obs, cfg, perception = self.make()
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            state = alg.save()
            path = folder / "student.pt"
            torch.save(state, path)
            with self.assertRaises(ValueError):
                load_student(path)
            control = alg.provenance["quiet_execution_control"]
            policy, provenance, identity = load_student(path, execution_control=control)
            _, extras = profile_config("student-v1", provenance)
            self.assertIsNotNone(extras.command_feedback)
            with torch.inference_mode():
                torch.testing.assert_close(policy(obs["policy"]), alg.student(obs), atol=0, rtol=0)
            runner = SimpleNamespace(device="cpu", alg=alg)
            runner.export_policy_to_jit = lambda directory, name: torch.jit.script(alg.student.as_jit()).save(str(Path(directory) / name))
            deployment.export_bundle(runner, folder / "exported", provenance)
            exported = folder / "exported/student_v3_safe.pt"
            with self.assertRaises(ValueError):
                load_student(exported, jit=True)
            load_student(exported, jit=True, execution_control=control)
            from deploy_student_v3_safe import load_policy
            actor, actual = load_policy(path)
            self.assertEqual(actual, control)
            with torch.inference_mode():
                torch.testing.assert_close(actor(obs), alg.student(obs), atol=0, rtol=0)

    def test_resume_is_independent_of_original_teacher_file_and_rejects_drift(self):
        args, alg, _, cfg, perception = self.make()
        state = alg.save()
        state["student_v1"]["provenance"]["teacher_checkpoint"] = "/missing/original/teacher.pt"
        resume = cli.parse_args(["--resume", "/unused/checkpoint.pt", "--device", "cpu"])
        training, _, n, _, manifest = entry.settings(resume, state)
        self.assertEqual(n, 4)
        self.assertEqual(training, cfg)
        changed = deepcopy(state)
        changed["student_v1"]["provenance"]["student_v3_safe_sources"]["bad"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "changed"):
            entry.settings(cli.parse_args(["--resume", "/unused.pt"]), changed)
        bad = cli.parse_args(["--resume", "/unused.pt", "--num-envs", "8"])
        with self.assertRaisesRegex(ValueError, "num_envs"):
            entry.settings(bad, state)


if __name__ == "__main__":
    unittest.main()
