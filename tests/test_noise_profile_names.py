"""Profile names change selectors, never policy weights or preset values."""
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'deploy/deploy_mujoco'))
sys.path.insert(0, str(ROOT/'source/rl_training'))
from student_v0_adapter import profile_config as v0_config
from student_v1_adapter import profile_config as v1_config
from m20_student_v0.perception import StudentV0PerceptionCfg
from m20_student_v1.configuration import StudentV1PerceptionCfg
from m20_student_v0.noise_profiles import PROFILE_ALIASES
import deploy_student_v0
import deploy_student_v1


class ProfileNameTests(unittest.TestCase):
    def test_old_and_new_names_have_identical_effective_configs(self):
        for config, meta in ((v0_config, {'perception': asdict(StudentV0PerceptionCfg(level=.25))}),
                             (v1_config, {'perception': asdict(StudentV1PerceptionCfg()), 'contract': {'history_steps': 4}})):
            for old, new in PROFILE_ALIASES.items():
                with self.subTest(version=config.__module__, name=new):
                    old_cfg, old_extra = config(old, meta)
                    new_cfg, new_extra = config(new, meta)
                    self.assertEqual(asdict(old_cfg), asdict(new_cfg))
                    self.assertEqual(asdict(old_extra), asdict(new_extra))

    def test_cli_normalizes_aliases_and_default(self):
        for parser in (deploy_student_v0.parse_args, deploy_student_v1.parse_args):
            base = ['--checkpoint', '/tmp/model.pt', '--export-only', '--export-dir', '/tmp/export']
            self.assertEqual(parser(base).noise_profile, 'student-v1')
            for old, new in PROFILE_ALIASES.items():
                self.assertEqual(parser(base+['--noise-profile', old]).noise_profile, new)
                self.assertEqual(parser(base+['--noise-profile', new]).noise_profile, new)

    def test_custom_files_support_both_name_schemas(self):
        for config, values, meta in (
            (v0_config, {'perception': {'height_noise_std': .123}}, {'perception': asdict(StudentV0PerceptionCfg())}),
            (v1_config, {'height_noise_std': .123}, {'perception': asdict(StudentV1PerceptionCfg()), 'contract': {'history_steps': 1}})):
            for key in ('realistic', 'student-v0'):
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory)/'profiles.json'
                    path.write_text(json.dumps({key: values}))
                    for selector in ('realistic', 'student-v0'):
                        cfg, _ = config(selector, meta, path)
                        self.assertEqual(cfg.height_noise_std, .123)

    def test_evaluation_cli_normalizes_old_names(self):
        spec=importlib.util.spec_from_file_location('name_evaluation', ROOT/'scripts/tools/evaluate_student_v1.py')
        module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        args=module.parse_args(['--v0','/tmp/v0.pt','--v1','/tmp/v1.pt','--model','/tmp/M20.xml','--terrain','descent','--output-dir','/tmp/unused','--profiles','clean','realistic','train'])
        self.assertEqual(args.profiles, ['teacher','student-v0','student-v1'])
        with self.assertRaisesRegex(ValueError, 'no-noise teacher'):
            module.run_case({}, {'model': 'teacher', 'profile': 'student-v0'}, Path('/tmp/unused'))


if __name__ == '__main__':
    unittest.main()
