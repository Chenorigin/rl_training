"""Unified entry routing checks: CPU only, no simulator or optimizer updates."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / 'scripts/reinforcement_learning/rsl_rl/distill_student.py'
spec = importlib.util.spec_from_file_location('distill_student', ENTRY)
entry = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = entry
spec.loader.exec_module(entry)


class EntryTests(unittest.TestCase):
    def args(self, *extra, resume=False):
        return entry.parse_args(['--resume' if resume else '--teacher-checkpoint', '/tmp/checkpoint.pt', *extra])

    def saved(self, version, task=None):
        task = task or f'Rough-Deeprobotics-M20-Student-{version}'
        return {f'student_{version}': {'provenance': {'task': task, 'num_envs': 64}}}

    def test_task_selects_version_and_output(self):
        for task, version in entry.TASK_VERSIONS.items():
            with self.subTest(task=task):
                args = self.args('--task', task)
                self.assertEqual(entry.configure_run(args, {}), version)
                output_version = {
                    'Rough-Deeprobotics-M20-Student-v3': 'v3',
                    'Rough-Deeprobotics-M20-Student-v3.5': 'v3_5',
                }.get(task, version)
                self.assertEqual(args.log_root.name, 'deeprobotics_m20_student_' + output_version)
                self.assertEqual(args.run_name, 'student_' + output_version)

    def test_fresh_default_preserves_v0(self):
        args = self.args()
        self.assertEqual(entry.configure_run(args, {}), 'v0')

    def test_resume_infers_task(self):
        for version in ('v0', 'v1'):
            args = self.args(resume=True)
            self.assertEqual(entry.configure_run(args, self.saved(version)), version)
            self.assertEqual(args.task, f'Rough-Deeprobotics-M20-Student-{version}')

    def test_resume_rejects_wrong_task(self):
        for task in ('Rough-Deeprobotics-M20-Student-v1', 'Rough-Deeprobotics-M20-Student-v3'):
            with self.subTest(task=task), self.assertRaisesRegex(ValueError, 'task differs'):
                entry.configure_run(self.args('--task', task, resume=True), self.saved('v0'))

    def test_resume_rejects_missing_or_ambiguous_provenance(self):
        for saved in ({}, {**self.saved('v0'), **self.saved('v1')}):
            with self.assertRaisesRegex(ValueError, 'unambiguous'):
                entry.configure_run(self.args(resume=True), saved)

    def test_resume_rejects_corrupt_version_task_pair(self):
        with self.assertRaisesRegex(ValueError, 'version'):
            entry.configure_run(self.args(resume=True), self.saved('v0', 'Rough-Deeprobotics-M20-Student-v1'))

    def test_v0_rejects_v1_only_flags(self):
        for option, value in (('--history-steps', '4'), ('--init-student', '/tmp/a.pt'), ('--config', '/tmp/a.json'), ('--teacher-ratio-end', '0')):
            with self.subTest(option=option), self.assertRaisesRegex(ValueError, 'only supported for v1'):
                entry.configure_run(self.args(option, value), {})

    def test_v1_history4_4096_settings(self):
        args = self.args('--task', 'Rough-Deeprobotics-M20-Student-v1', '--history-steps', '4', '--num-envs', '4096')
        entry.configure_run(args, {})
        training, perception, task, n, _ = entry.settings_v1(args, {})
        self.assertEqual(n, 4096)
        self.assertEqual(entry.student_dim(training['history_steps']), 789)
        self.assertFalse(training['algorithm']['save_replay'])

    def test_v0_preserves_original_environment_cap(self):
        args = self.args('--num-envs', '4096')
        entry.configure_run(args, {})
        with self.assertRaisesRegex(ValueError, 'V0 is limited'):
            entry.settings_v0(args, {})

    def test_v1_resume_rejects_configuration_override(self):
        args = self.args('--history-steps', '4', resume=True)
        entry.configure_run(args, self.saved('v1'))
        with self.assertRaisesRegex(ValueError, 'preserves v1'):
            entry.settings_v1(args, self.saved('v1'))

    def test_main_routes_each_backend_after_preflight(self):
        for version in ('v0', 'v1'):
            with self.subTest(version=version):
                task = f'Rough-Deeprobotics-M20-Student-{version}'
                setting = ({}, None, task, 64, {})
                with patch.object(entry, 'resolve_checkpoint', return_value=Path('/tmp/checkpoint.pt')), patch('torch.load', return_value={}), patch.object(entry, 'settings_' + version, return_value=setting), patch.object(entry, 'preflight_' + version, return_value={'checked': True}) as preflight, patch.object(entry, 'train_or_validate_' + version) as backend, patch.object(entry, 'train_or_validate_' + ('v1' if version == 'v0' else 'v0')) as other:
                    entry.main(['--teacher-checkpoint', '/tmp/checkpoint.pt', '--task', task])
                    preflight.assert_called_once()
                    backend.assert_called_once()
                    self.assertEqual(backend.call_args.args[-1], {'checked': True})
                    other.assert_not_called()

    def test_preflight_only_never_calls_backend(self):
        for version in ('v0', 'v1'):
            task = f'Rough-Deeprobotics-M20-Student-{version}'
            with patch.object(entry, 'resolve_checkpoint', return_value=Path('/tmp/checkpoint.pt')), patch('torch.load', return_value={}), patch.object(entry, 'settings_' + version, return_value=({}, None, task, 64, {})), patch.object(entry, 'preflight_' + version, return_value={}), patch.object(entry, 'train_or_validate_v0') as v0, patch.object(entry, 'train_or_validate_v1') as v1:
                entry.main(['--teacher-checkpoint', '/tmp/checkpoint.pt', '--task', task, '--preflight-only'])
                v0.assert_not_called()
                v1.assert_not_called()

    def test_legacy_alias_defaults_to_v1(self):
        alias_spec = importlib.util.spec_from_file_location('legacy_v1', ENTRY.with_name('distill_student_v1.py'))
        alias = importlib.util.module_from_spec(alias_spec)
        alias_spec.loader.exec_module(alias)
        with patch.object(alias, 'unified_main') as main:
            alias.main(['--teacher-checkpoint', '/tmp/checkpoint.pt'])
            self.assertEqual(main.call_args.kwargs['default_task'], 'Rough-Deeprobotics-M20-Student-v1')


if __name__ == '__main__':
    unittest.main()
