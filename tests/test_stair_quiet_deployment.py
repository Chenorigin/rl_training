"""CPU counterexamples for quiet execution contracts and offline response fits."""
from copy import deepcopy
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/rl_training"))
sys.path.insert(0, str(ROOT / "deploy/deploy_mujoco"))

from task_support.stair_quiet_v1 import deployment, distill, actuator_id
from task_support.stair_quiet_v1.landing import LandingSettings, control_contract
from task_support.stair_quiet_v1.baselines import TEACHER, STUDENT


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def teacher_fixture():
    saved = torch.load(ROOT / TEACHER, map_location="cpu", weights_only=True)
    saved["m20_training_manifest"] = deepcopy(saved["m20_training_manifest"])
    saved["m20_training_manifest"]["noise_refinement"] = {
        "control_contract": control_contract(LandingSettings())
    }
    return saved


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.contract = control_contract(LandingSettings())
        self.marked = {"m20_training_manifest": {"noise_refinement": {"control_contract": self.contract}}}

    def test_legacy_rejects_even_empty_or_false_marker(self):
        for value in (self.contract, {}, None, False):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "cannot execute"):
                deployment.reject_unsupported_quiet_checkpoint({"quiet_execution_control": value}, "old runtime")
        deployment.reject_unsupported_quiet_checkpoint({}, "baseline runtime")

    def test_runtime_requires_identical_source_settings_and_no_hardware_claim(self):
        self.assertEqual(deployment.require_runtime(self.marked, self.contract), self.contract)
        changed = control_contract(LandingSettings(mode="off"))
        with self.assertRaisesRegex(ValueError, "change or omit"):
            deployment.require_runtime(self.marked, changed)
        with self.assertRaisesRegex(ValueError, "malformed"):
            deployment.require_runtime(self.marked, None)
        with self.assertRaisesRegex(ValueError, "hardware execution"):
            deployment.require_runtime(self.marked, self.contract, platform="hardware")
        corrupt = deepcopy(self.contract)
        corrupt["source_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "changed"):
            deployment.validate_control_contract(corrupt)

    def test_false_is_not_integer_zero_and_marker_conflict_rejected(self):
        corrupt = deepcopy(self.contract)
        corrupt["actual_acoustic_pass"] = 0
        with self.assertRaisesRegex(ValueError, "changed"):
            deployment.validate_control_contract(corrupt)
        saved = {**self.marked, "quiet_execution_control": control_contract(LandingSettings(mode="off"))}
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            deployment.execution_contract(saved)

    def test_resume_requires_source_and_control_provenance(self):
        provenance = distill.provenance_for({"task": deployment.STUDENT_TASK}, self.contract)
        saved = {"student_v1": {"provenance": provenance}}
        self.assertEqual(distill.check_source(deployment.STUDENT_TASK, saved, resume=True), self.contract)
        provenance["quiet_distillation_sources"]["changed.py"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "source/environment changed"):
            distill.check_source(deployment.STUDENT_TASK, saved, resume=True)

    def test_old_teacher_adapter_rejects_new_control(self):
        from m20_student_v1.v3_5_teacher import teacher_manifest
        saved = teacher_fixture()
        with self.assertRaisesRegex(ValueError, "cannot execute"):
            teacher_manifest(saved)
        self.assertEqual(teacher_manifest(saved, allow_quiet=True)["m20_profile"], "idle_stop_motion_v1")

    def test_dedicated_route_uses_pinned_student_and_complete_teacher(self):
        entry = import_file("quiet_test_distill", ROOT / "scripts/reinforcement_learning/rsl_rl/distill_student.py")
        args = entry.parse_args(["--teacher-checkpoint", str(ROOT / TEACHER),
                                 "--task", deployment.STUDENT_TASK, "--device", "cpu", "--preflight-only"])
        saved = teacher_fixture()
        self.assertEqual(entry.configure_run(args, saved), "v1")
        self.assertEqual(args.init_student, (ROOT / STUDENT).resolve())
        training, perception, task, n, _ = entry.settings_v1(args, saved)
        self.assertEqual(training["obs_groups"]["teacher"], ["teacher", "motion_control"])
        self.assertEqual(training["history_steps"], 4)
        self.assertEqual(training["teacher"]["class_name"], "m20_student_v1.v3_5_teacher.MotionTeacherModel")
        cfg = SimpleNamespace(landing_control=None)
        distill.configure_environment(cfg, self.contract)
        self.assertEqual(control_contract(cfg.landing_control), self.contract)
        old = entry.parse_args(["--teacher-checkpoint", str(ROOT / TEACHER),
                                "--task", "Rough-Deeprobotics-M20-Student-v3.5"])
        with self.assertRaisesRegex(ValueError, "cannot execute"):
            entry.configure_run(old, saved)

    def test_real_student_loader_refuses_missing_execution_layer(self):
        from student_v1_adapter import load_student
        saved = torch.load(ROOT / STUDENT, map_location="cpu", weights_only=True)
        original = saved["student_v1"]["provenance"]
        saved["student_v1"]["provenance"] = deployment.add_execution_provenance(
            {**original, "task": deployment.STUDENT_TASK}, self.contract)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "quiet.pt"
            torch.save(saved, path)
            with self.assertRaisesRegex(ValueError, "malformed"):
                load_student(path)
            policy, provenance, identity = load_student(path, execution_control=self.contract)
            with torch.inference_mode():
                self.assertEqual(tuple(policy(torch.zeros(1, 789)).shape), (1, 16))
            self.assertEqual(provenance["quiet_execution_control"], self.contract)
            self.assertEqual(identity["actor_input_dim"], 789)

    def test_real_teacher_loader_default_refuses_new_control(self):
        import deploy_mujoco
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "quiet_teacher.pt"
            torch.save(teacher_fixture(), path)
            with self.assertRaisesRegex(ValueError, "cannot execute"):
                deploy_mujoco.load_actor(path)
            actor = deploy_mujoco.load_actor(path, allow_execution_control=True)
            self.assertTrue(any(k.startswith("residual_mlp.") for k in actor.state_dict()))

    def test_formal_sdk_exporter_rejects_before_weights_or_sdk_work(self):
        path = ROOT.parent / "m20_student_v3_5_deploy/scripts/export_checkpoint.py"
        exporter = import_file("quiet_test_sdk_export", path)
        for field in ({"quiet_execution_control": self.contract},
                      {"teacher_manifest": {"noise_refinement": {"control_contract": self.contract}}}):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "existing SDK"):
                exporter.student_actor({"student_v1": {"provenance": field}}, "v1")

    def test_actual_dedicated_cli_preflight_has_no_simulation_or_updates(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            source, receipt = folder / "synthetic_teacher_manifest_fixture.pt", folder / "preflight.json"
            # Untouched real baseline weights with a TEST-ONLY control marker,
            # not a trained quiet teacher and never a deployment candidate.
            torch.save(teacher_fixture(), source)
            with contextlib.redirect_stdout(io.StringIO()):
                distill.main(["--teacher-checkpoint", str(source), "--preflight-only", "--device", "cpu",
                              "--num-envs", "2", "--max-iterations", "1",
                              "--log-root", str(folder), "--preflight-report", str(receipt)])
            report = json.loads(receipt.read_text())
            self.assertFalse(report["simulator_started"])
            self.assertEqual(report["optimizer_updates"], 0)
            self.assertEqual(report["actor_dim"], 789)
            self.assertEqual(report["teacher_reference_max_action_error"], 0.)
            self.assertTrue(report["teacher_frozen"])
            self.assertEqual(report["initialization"]["source_input_dim"], 789)
            self.assertEqual(report["quiet_execution_control"], self.contract)

    def test_export_embeds_contract_and_formal_onnx_reader_refuses_it(self):
        from m20_student_v1.model_io import actor_from_state
        from student_v1_adapter import load_student
        saved = torch.load(ROOT / STUDENT, map_location="cpu", weights_only=True)
        provenance = deployment.add_execution_provenance(
            {**saved["student_v1"]["provenance"], "task": deployment.STUDENT_TASK}, self.contract)
        actor = actor_from_state(saved["student_state_dict"], 789, provenance["training_config"]["student"])
        runner = SimpleNamespace(device="cpu", alg=SimpleNamespace(student=lambda obs: actor(obs["policy"])))
        runner.export_policy_to_jit = lambda directory, name: torch.jit.script(actor).save(str(Path(directory) / name))
        with tempfile.TemporaryDirectory() as folder:
            exported = Path(folder) / "exported"
            document = distill.export_bundle(runner, exported, provenance)
            self.assertFalse(document["standalone_controller"])
            self.assertFalse(document["hardware_approved"])
            model = exported / "quiet_nominal_actor.pt"
            with self.assertRaises(ValueError):
                load_student(model, jit=True)
            load_student(model, jit=True, execution_control=self.contract)
            sdk = ROOT.parent / "m20_student_v3_5_deploy/scripts"
            sys.path.insert(0, str(sdk))
            export = import_file("quiet_test_direct_onnx", sdk / "export_onnx.py")
            # Strip ONLY the sidecar marker. The embedded artifact still
            # prevents a direct legacy ONNX export from dropping control.
            sidecar = json.loads((exported / "observation_contract.json").read_text())
            sidecar.pop("quiet_execution_control")
            path = exported / "legacy_sidecar_fixture.json"
            path.write_text(json.dumps(sidecar))
            argv = ["export_onnx", "--model", str(model), "--contract", str(path),
                    "--profile", "v1", "--output", str(exported / "must_not_exist.onnx")]
            with patch.object(sys, "argv", argv), self.assertRaisesRegex(ValueError, "nominal actor"):
                export.main()
            self.assertFalse((exported / "must_not_exist.onnx").exists())


def synthetic_recording():
    # Exact discrete response fixture, explicitly NOT a hardware recording.
    dt, delay = .005, .015
    t = np.arange(0., 12., dt)
    tc = np.arange(0., 12., .02)
    rng = np.random.default_rng(29)
    command = rng.uniform(-.35, .35, (len(tc), 1))
    q, dq = np.zeros((len(t), 1)), np.zeros((len(t), 1))
    for i in range(len(t) - 1):
        ci = np.searchsorted(tc, t[i] - delay, side="right") - 1
        u = command[max(ci, 0), 0]
        acc = 80 * (u - q[i, 0]) - 9 * dq[i, 0] - 1.5 * np.tanh(dq[i, 0] / .05) + .1
        dq[i + 1, 0] = dq[i, 0] + acc * dt
        q[i + 1, 0] = q[i, 0] + dq[i, 0] * dt + .5 * acc * dt ** 2
    arrays = dict(state_time_s=t, command_time_s=tc, command=command, q_rad=q, dq_rad_s=dq)
    metadata = dict(schema=actuator_id.SCHEMA, origin="synthetic_test_fixture",
                    joint_names=["test_hip"], command_kinds=["position"],
                    common_clock="synthetic_fixture_clock", state_timestamp_kind="capture", condition="suspended")
    return arrays, metadata


class IdentificationTests(unittest.TestCase):
    def test_identifies_delay_with_untouched_final_holdout_and_no_calibration_claim(self):
        arrays, metadata = synthetic_recording()
        report = actuator_id.fit_recording(arrays, metadata, allow_synthetic=True,
                                          delay_candidates_s=[0., .005, .01, .015, .02, .03])
        joint = report["joints"][0]
        self.assertEqual(joint["selected_delay_s"], .015)
        self.assertLess(joint["heldout_velocity_rmse_rad_s"], 1e-10)
        self.assertAlmostEqual(joint["coefficients"]["response_gain_s2"], 80., places=8)
        self.assertFalse(report["hardware_calibrated"])
        self.assertFalse(report["hardware_authenticity_verified"])
        # Corrupt only the final holdout: selected delay and fitted parameters
        # must remain unchanged while the final residual increases.
        broken = {key: value.copy() for key, value in arrays.items()}
        broken["dq_rad_s"][int(.9 * len(broken["dq_rad_s"])):] += .5
        other = actuator_id.fit_recording(broken, metadata, allow_synthetic=True,
                                         delay_candidates_s=[0., .005, .01, .015, .02, .03])
        self.assertEqual(other["joints"][0]["selected_delay_s"], joint["selected_delay_s"])
        self.assertEqual(other["joints"][0]["coefficients"], joint["coefficients"])
        self.assertGreater(other["joints"][0]["heldout_velocity_rmse_rad_s"], .01)

    def test_no_real_data_no_calibration_and_duplicate_clock_rejected(self):
        arrays, metadata = synthetic_recording()
        with self.assertRaisesRegex(ValueError, "hardware recording"):
            actuator_id.fit_recording(arrays, metadata)
        metadata["origin"] = "hardware_recording"  # Check only the declared format.
        arrays["state_time_s"][3] = arrays["state_time_s"][2]
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            actuator_id.fit_recording(arrays, metadata)

    def test_condition_and_excitation_fail_closed(self):
        arrays, metadata = synthetic_recording()
        metadata["condition"] = "mixed"
        with self.assertRaisesRegex(ValueError, "Separate"):
            actuator_id.fit_recording(arrays, metadata, allow_synthetic=True)
        metadata["condition"] = "suspended"
        arrays["q_rad"][:] = 0
        arrays["dq_rad_s"][:] = 0
        arrays["command"][:] = 0
        with self.assertRaisesRegex(ValueError, "excitation"):
            actuator_id.fit_recording(arrays, metadata, allow_synthetic=True)

    def test_future_command_never_leaks_into_selected_response(self):
        arrays, metadata = synthetic_recording()
        arrays["command_time_s"] += 100.
        with self.assertRaisesRegex(ValueError, "overlap"):
            actuator_id.fit_recording(arrays, metadata, allow_synthetic=True)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
