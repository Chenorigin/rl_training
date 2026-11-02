"""CTS argument/contract checks: CPU only, no simulator or optimizer steps."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/rl_training"))

from task_support.cts import entry
from task_support.cts.configuration import TASK, VERSION, default_settings, validate_config
from m20_student_v1.contract import observation_contract
from task_support.stair_quiet_v1.deployment import contract_hash, RUNTIME_SCOPE
from task_support.stair_quiet_v1.landing import LandingSettings, control_contract


class CTSArgumentsTests(unittest.TestCase):
    def args(self, *extra):
        with patch.dict("os.environ", {"WORLD_SIZE": "1"}):
            return entry.parse_args(["--teacher-checkpoint", "/tmp/teacher.pt",
                                     "--init-student", "/tmp/student.pt", *extra])

    def test_new_run_requires_both_models(self):
        with patch.dict("os.environ", {"WORLD_SIZE": "1"}):
            for arguments in ([], ["--teacher-checkpoint", "/tmp/teacher.pt"]):
                with self.subTest(arguments=arguments), self.assertRaises(SystemExit):
                    entry.parse_args(arguments)

    def test_init_cts_is_explicit_weight_transfer_and_allows_new_configuration(self):
        with patch.dict('os.environ', {'WORLD_SIZE': '1'}):
            args = entry.parse_args(['--init-cts', '/tmp/cts.pt', '--config', '/tmp/new.json', '--num-envs', '4'])
            self.assertIsNone(args.resume)
            self.assertEqual(args.init_cts, Path('/tmp/cts.pt'))
            with self.assertRaises(SystemExit):
                entry.parse_args(['--init-cts', '/tmp/cts.pt', '--init-student', '/tmp/student.pt'])

    def test_resume_rejects_replacing_sources_configuration_or_seed(self):
        with patch.dict("os.environ", {"WORLD_SIZE": "1"}):
            for flag, value in (("--init-student", "/tmp/student.pt"),
                                ("--config", "/tmp/config.json"), ("--seed", "7")):
                with self.subTest(flag=flag), self.assertRaises(SystemExit):
                    entry.parse_args(["--resume", "/tmp/cts.pt", flag, value])

    def test_bounded_no_update_mode_rejects_unbounded_steps(self):
        args = self.args("--validate-only", "--validation-steps", "16", "--num-envs", "4")
        self.assertTrue(args.validate_only)
        self.assertEqual(args.validation_steps, 16)
        for steps in ("0", "-1", "1025"):
            with self.subTest(steps=steps), self.assertRaises(SystemExit):
                self.args("--validate-only", "--validation-steps", steps)

    def test_modes_are_exclusive(self):
        with self.assertRaises(SystemExit):
            self.args("--preflight-only", "--validate-only")

    def test_invalid_run_label_and_resource_limit_rejected(self):
        for arguments in (("--run-name", "../other"), ("--run-name", ""),
                          ("--min-free-disk-gb", "nan"), ("--min-free-disk-gb", "0"),
                          ("--num-envs", "0"), ("--max-iterations", "0")):
            with self.subTest(arguments=arguments), self.assertRaises(SystemExit):
                self.args(*arguments)

    def test_distributed_environment_is_rejected(self):
        with patch.dict("os.environ", {"WORLD_SIZE": "2"}), self.assertRaises(SystemExit):
            entry.parse_args(["--resume", "/tmp/cts.pt"])

    def test_preflight_returns_without_simulator_import(self):
        expected = {"mode": "preflight"}
        with patch.dict("os.environ", {"WORLD_SIZE": "1"}), \
                patch.object(entry, "prepare", return_value=(None, {}, {}, {})), \
                patch.object(entry, "preflight", return_value=expected) as preflight:
            self.assertIs(entry.main(["--teacher-checkpoint", "/tmp/t.pt", "--init-student",
                                      "/tmp/s.pt", "--preflight-only"]), expected)
        preflight.assert_called_once()


class CTSMetadataTests(unittest.TestCase):
    def meta(self):
        from task_support.cts.auxiliary import validate_context
        from task_support.student_v3_safe.configuration import default_settings as student_settings
        control = control_contract(LandingSettings())
        config = default_settings()
        perception = student_settings()['perception']
        perception['zero_delay_probability'] = config['map_delay_mixture']['zero_delay_probability']
        perception['map_capture_padding_m'] = config['map_capture']['padding_m']
        perception['support_surface_visibility'] = config['map_capture']['support_surface_visibility']
        perception['continuous_delay'] = config.get('all_terrain', {}).get('enabled', False)
        auxiliary = {'version': 1, 'config': deepcopy(config['auxiliary']),
                     'sensor_noise_std': validate_context(config['auxiliary'], perception),
                     'retention': {'manifest_sha256': config['auxiliary']['retention_sha256']}}
        return {"version": VERSION, "task": TASK, "num_envs": 4,
                "config": config, "contract": observation_contract(4),
                "perception": perception, "environment": {"perception": deepcopy(perception)},
                "auxiliary_contract": auxiliary,
                "quiet_execution_control": control,
                "quiet_execution_control_sha256": contract_hash(control),
                "quiet_runtime_scope": RUNTIME_SCOPE, "source_sha256": {"source.py": "abc"}}

    def test_deployment_checks_contract_without_requiring_training_tree_hash(self):
        meta = self.meta()
        with patch.object(entry, "source_hashes", side_effect=AssertionError("unexpected source read")), \
                patch('task_support.cts.auxiliary.context_contract', side_effect=AssertionError('retention must not be read')):
            self.assertIs(entry.validate_metadata(meta, resume=False), meta)

    def test_resume_requires_exact_source_receipt(self):
        meta = self.meta()
        with patch.object(entry, "source_hashes", return_value=meta["source_sha256"]), \
                patch('task_support.cts.auxiliary.context_contract', return_value=meta['auxiliary_contract']):
            self.assertIs(entry.validate_metadata(meta), meta)
        with patch.object(entry, "source_hashes", return_value={"source.py": "changed"}), \
                self.assertRaisesRegex(ValueError, "source contract mismatch"):
            entry.validate_metadata(meta)

    def test_rejects_wrong_task_version_and_missing_cohort(self):
        for key, value in (("task", "other"), ("version", "other"),
                           ("num_envs", 1), ("num_envs", True)):
            meta = self.meta()
            meta[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                entry.validate_metadata(meta, resume=False)

    def test_resume_rejects_changed_auxiliary_receipt(self):
        meta = self.meta()
        changed = deepcopy(meta['auxiliary_contract'])
        changed['retention']['manifest_sha256'] = 'changed'
        with patch.object(entry, 'source_hashes', return_value=meta['source_sha256']), \
                patch('task_support.cts.auxiliary.context_contract', return_value=changed), \
                self.assertRaisesRegex(ValueError, 'retention/context contract changed'):
            entry.validate_metadata(meta)

    def test_auxiliary_noise_config_and_version_are_embedded_contracts(self):
        for target, key, value in (('auxiliary_contract', 'version', 0),
                                   ('auxiliary_contract', 'sensor_noise_std', {}),
                                   ('auxiliary_contract', 'config', {}),
                                   ('perception', 'height_noise_std', 10.)):
            meta = self.meta()
            meta[target][key] = value
            with self.subTest(target=target, key=key), self.assertRaises(ValueError):
                entry.validate_metadata(meta, resume=False)
        meta = self.meta()
        meta['version'] = 'm20_cts_action_latent_v1'
        with self.assertRaisesRegex(ValueError, 'older CTS runs cannot resume'):
            entry.validate_metadata(meta)

    def test_live_auxiliary_contract_must_match_saved_metadata(self):
        meta = self.meta()
        alg = SimpleNamespace(auxiliary=SimpleNamespace(contract=deepcopy(meta['auxiliary_contract'])))
        entry.validate_live_auxiliary(alg, meta)
        alg.auxiliary.contract['version'] = 0
        with self.assertRaisesRegex(ValueError, 'live auxiliary contract'):
            entry.validate_live_auxiliary(alg, meta)

    def test_execution_contract_and_sensor_dimensions_cannot_be_changed(self):
        for field in ("quiet_execution_control", "contract"):
            meta = deepcopy(self.meta())
            if field == "quiet_execution_control":
                meta[field]["settings"]["mode"] = "invalid"
            else:
                meta[field]["student_observation_dim"] = 244
            with self.subTest(field=field), self.assertRaises(ValueError):
                entry.validate_metadata(meta, resume=False)


class CTSAuxiliaryConfigurationTests(unittest.TestCase):
    def test_defaults_preserve_existing_stability_and_retention_weights(self):
        old = json.loads((ROOT/'configs/distillation/student_v3_safe.json').read_text())['training']['algorithm']
        new = default_settings()['auxiliary']
        for key in ('temporal_loss_weight', 'spatial_loss_weight', 'stability_batch_size',
                    'spatial_noise_scale', 'retention_loss_weight', 'retention_batch_size'):
            with self.subTest(key=key):
                self.assertEqual(new[key], old[key])
        self.assertEqual(new['leg_loss_weight'], new['wheel_loss_weight'])
        self.assertNotEqual(new['retention_manifest'], old['retention_manifest'])
        validate_config(default_settings())

    def test_v2_remains_deployable_but_cannot_resume_new_objectives(self):
        from task_support.cts.configuration import LEGACY_VERSION
        meta = CTSMetadataTests().meta()
        meta['version'] = LEGACY_VERSION
        self.assertIs(entry.validate_metadata(meta, resume=False), meta)
        with self.assertRaisesRegex(ValueError, 'older CTS runs cannot resume'):
            entry.validate_metadata(meta)

    def test_old_and_unknown_auxiliary_configuration_is_rejected(self):
        for mutation in ('missing', 'unknown'):
            config = default_settings()
            if mutation == 'missing':
                del config['auxiliary']
            else:
                config['auxiliary']['unsupported'] = True
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_config(config)

    def test_auxiliary_memory_guard_runs_before_model_or_dataset_allocation(self):
        gib = 1024**3
        host = SimpleNamespace(total=16*gib, available=12*gib)
        meta = {'config': default_settings(), 'num_envs': 4}
        with patch('psutil.virtual_memory', return_value=host), \
                patch('task_support.cts.auxiliary.retention_bytes', return_value=8*gib), \
                patch('task_support.cts.model.build_models') as models, \
                patch('task_support.cts.algorithm.CTSAlgorithm') as algorithm, \
                self.assertRaisesRegex(RuntimeError, 'CPU preflight'):
            entry.preflight(SimpleNamespace(), {}, {}, meta)
        models.assert_not_called()
        algorithm.assert_not_called()

    def test_disabled_auxiliaries_have_no_additional_memory_reserve(self):
        config = default_settings()
        config.pop('straight_practice')  # This objective also has an autograd batch reserve.
        config['all_terrain']['wall_stability']['enabled'] = False
        for key in ('temporal_loss_weight', 'spatial_loss_weight', 'retention_loss_weight'):
            config['auxiliary'][key] = 0.
        budget = entry.auxiliary_memory_budget(config)
        self.assertEqual(budget['retention_data_bytes'], 0)
        self.assertEqual(budget['auxiliary_temporary_bytes'], 0)
        self.assertEqual(budget['auxiliary_reserve_bytes'], 0)

    def test_metadata_context_does_not_load_dataset_without_host_headroom(self):
        gib = 1024**3
        host = SimpleNamespace(total=16*gib, available=12*gib)
        with patch('psutil.virtual_memory', return_value=host), \
                patch('task_support.cts.auxiliary.retention_bytes', return_value=8*gib), \
                patch('task_support.cts.auxiliary.context_contract') as context, \
                self.assertRaisesRegex(RuntimeError, 'CPU preflight'):
            entry.guarded_auxiliary_contract(default_settings(), {})
        context.assert_not_called()


class CTSDeploymentTests(unittest.TestCase):
    def deployment(self):
        sys.path.insert(0, str(ROOT / "deploy/deploy_mujoco"))
        import deploy_cts_student
        return deploy_cts_student

    def test_actor_packs_raw_command_without_privileged_groups(self):
        deployment = self.deployment()

        class ProbePolicy(torch.nn.Module):
            def forward(self, policy, raw_command):
                return policy[:, :16] + raw_command[:, :1]

        actor = deployment.CTSStudentActor(ProbePolicy())
        import numpy as np
        obs = deployment.quiet.baseline.policy_inputs(np.zeros(789, dtype=np.float32),
                                                       np.array([.4, 0., 0.], dtype=np.float32), actor)
        self.assertEqual(set(obs.keys()), {"policy", "motion_control"})
        self.assertTrue(torch.equal(actor(obs), torch.full((1, 16), .4)))
        self.assertFalse(getattr(actor, "uses_safe_control", False))
        self.assertIsNone(deployment.sensor_heading_controller(object(), actor))

    def test_loader_uses_trained_weights_and_rejects_bound_corruption(self):
        deployment = self.deployment()
        config = default_settings()["algorithm"]
        control = control_contract(LandingSettings())
        meta = {"config": {"algorithm": config}, "student_inputs": {"policy": 789, "motion_control": 3},
                "command_feedback": {"version": "legacy_v1_v3_5", "tuning": {}},
                "teacher_manifest": {"stair_tuning": {}}, "quiet_execution_control": control,
                "auxiliary_contract": {"version": 1}}

        class ProbePolicy(torch.nn.Module):
            def __init__(self, value):
                super().__init__()
                self.register_buffer("offset", value.clone())

            def forward(self, policy, raw_command):
                return policy[:, :16] + self.offset

        class ProbeModels(torch.nn.Module):
            def __init__(self, value):
                super().__init__()
                self.register_buffer("offset", torch.tensor(value))
                self.decoder = torch.nn.Module()
                self.decoder.register_buffer("bounds", torch.tensor(config["decoder_bound"]))

            def student_export(self):
                return ProbePolicy(self.offset)

        state = ProbeModels(2.).state_dict()
        saved = {"cts": meta, "initialization": {"teacher": {}, "student": {}},
                 "algorithm": {"cts_version": 2, "algorithm_config": config, "models_state_dict": state,
                               "auxiliary_contract": meta['auxiliary_contract']}}
        with patch.object(deployment.torch, "load", return_value=saved), \
                patch.object(deployment, "validate_metadata", return_value=meta), \
                patch.object(deployment, "build_models", side_effect=lambda *_args, **_kwargs: ProbeModels(0.)):
            actor, returned = deployment.load_policy("/tmp/cts.pt")
            self.assertEqual(returned, control)
            output = actor({"policy": torch.zeros(1, 789), "motion_control": torch.zeros(1, 3)})
            self.assertTrue(torch.equal(output, torch.full((1, 16), 2.)))
            state["decoder.bounds"][0] += .1
            with self.assertRaisesRegex(ValueError, "decoder bounds"):
                deployment.load_policy("/tmp/cts.pt")


if __name__ == "__main__":
    unittest.main()
