"""Profile selectors, spatial restrictions and unchanged surviving observations."""
from dataclasses import asdict, replace
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/rl_training"))
sys.path.insert(0, str(ROOT / "deploy/deploy_mujoco"))
from m20_student_v0.noise_profiles import PROFILE_ALIASES
from m20_student_v0.perception import StudentV0PerceptionCfg
from m20_student_v1.configuration import StudentV1PerceptionCfg
from m20_student_v1.perception import StudentObservationState, grid_points, map_output_mask
from m20_student_v1.profiles import (FRONT_ONLY_PROFILE, LEGACY_FRONT_ONLY_PROFILE,
                                    FRONT_DEPLOYMENT_OFFSET_M, profile_config)
from student_v0_adapter import profile_config as v0_config
from student_v1_adapter import profile_config as v1_config
import deploy_student_v0
import deploy_student_v1

torch.set_num_threads(1)


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


class FrontProfileTests(unittest.TestCase):
    def test_inherits_checkpoint_and_changes_only_spatial_restriction(self):
        provenance = {"perception": asdict(StudentV1PerceptionCfg(level=.73, action_delay_steps=3))}
        original = asdict(profile_config("student-v1", provenance))
        restricted = asdict(profile_config(FRONT_ONLY_PROFILE, provenance))
        changed = {key for key in original if original[key] != restricted[key]}
        self.assertEqual(changed, {"front_only", "front_sensor_offset_m"})
        self.assertEqual(restricted["front_sensor_offset_m"], FRONT_DEPLOYMENT_OFFSET_M)
        self.assertTrue(restricted["front_only"])
        self.assertEqual(restricted["front_blind_distance_m"], 0.)
        self.assertEqual(restricted, asdict(profile_config(LEGACY_FRONT_ONLY_PROFILE, provenance)))
        with self.assertRaisesRegex(ValueError, "training provenance"):
            profile_config(FRONT_ONLY_PROFILE, {})
        with self.assertRaisesRegex(ValueError, "distance"):
            StudentV1PerceptionCfg(front_blind_distance_m=True).validate()

    def test_metric_boundary_sides_and_actual_grid(self):
        cfg = profile_config(FRONT_ONLY_PROFILE, {"perception": asdict(StudentV1PerceptionCfg())})
        probes = torch.tensor([[0., 0.], [.01, 0.], [.10, 0.], [.30, 0.],
                               [-.8, 0.], [0., .5], [0., -.5], [.15, .26], [.15, -.26],
                               [.1, .4], [.1, -.4], [.2, .3], [.2, -.3]])
        probes += probes.new_tensor(cfg.front_sensor_offset_m[:2])
        self.assertEqual(map_output_mask(probes, cfg).tolist(),
                         [False, True, True, True, False, False, False, True, True,
                          False, False, True, True])
        grid = grid_points("cpu")
        keep = map_output_mask(grid, cfg)
        self.assertEqual(int(keep.sum()), 45)
        self.assertTrue(keep[5 * 17 + 12])  # x=.4 is only .07972m ahead of the LiDAR.
        self.assertTrue(keep[5 * 17 + 14])  # No additional 30cm blind region.
        self.assertFalse(keep[10 * 17 + 12])  # x=.4,y=.5 is outside the sector.
        self.assertTrue(torch.equal(keep.reshape(11, 17), keep.reshape(11, 17).flip(0)))
        self.assertTrue(map_output_mask(grid, profile_config("teacher", {})).all())

    def test_sector_follows_configured_lidar_position(self):
        cfg = profile_config(FRONT_ONLY_PROFILE, {"perception": asdict(StudentV1PerceptionCfg())})
        shifted = replace(cfg, front_sensor_offset_m=(.6, .2, .1))
        points = torch.tensor([[.95, .2], [.6, .55], [.85, .2], [.95, -.7]])
        self.assertEqual(map_output_mask(points, shifted).tolist(), [True, False, True, False])
        self.assertFalse(torch.equal(map_output_mask(points, cfg), map_output_mask(points, shifted)))

    def test_noisy_motion_retains_exact_baseline_channels_and_history(self):
        provenance = {"perception": asdict(StudentV1PerceptionCfg(level=1., clean_probability=0., stress_probability=0.))}
        restricted = profile_config(FRONT_ONLY_PROFILE, provenance)
        # Compare identical acquisition geometry; only the final output differs.
        baseline = replace(restricted, front_only=False, front_blind_distance_m=0.)
        keep = map_output_mask(grid_points("cpu"), restricted)

        def rollout(cfg):
            torch.manual_seed(76)
            state = StudentObservationState(1, "cpu", .02, cfg, history_steps=4)
            rows, visibility = [], []
            for step in range(60):
                yaw = torch.tensor([step * .05])
                c, s = yaw.cos().item(), yaw.sin().item()
                # Yaw plus pitch: the output sector remains in the gravity/yaw grid.
                yaw_rotation = torch.tensor([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
                pitch = torch.tensor([[.9950042, 0., .0998334], [0., 1., 0.], [-.0998334, 0., .9950042]])
                rotation = (yaw_rotation @ pitch)[None]
                base = torch.tensor([[step * .007, 0., .55]])
                grid = state.grid @ yaw_rotation[:2, :2].T + base[:, :2]
                hits = torch.cat((grid, .08 * grid[:, :1] + .05), -1)[None]
                if step % 3 == 0:
                    hits[:, 99] = torch.nan
                proprio = torch.zeros(1, 57)
                proprio[:, 5] = -1.
                proprio[:, 6] = .4
                rows.append(state.observe(proprio, hits, base, yaw, rotation, step))
                visibility.append(state.last_visibility.clone())
            return torch.cat(rows), torch.cat(visibility)

        old, old_visibility = rollout(baseline)
        new, new_visibility = rollout(restricted)
        self.assertTrue(torch.equal(old[:, :57], new[:, :57]))
        self.assertTrue(torch.equal(old[:, 618:], new[:, 618:]))
        self.assertGreater(int(new[:, 244:431][:, keep].sum()), 0)
        for start in (57, 244, 431):
            self.assertTrue(torch.equal(old[:, start:start+187][:, keep], new[:, start:start+187][:, keep]))
            self.assertEqual(int(torch.count_nonzero(new[:, start:start+187][:, ~keep])), 0)
        self.assertTrue(torch.equal(old_visibility[:, keep], new_visibility[:, keep]))
        self.assertFalse(new_visibility[:, ~keep].any())

    def test_cached_packets_cannot_fill_forbidden_region(self):
        clean = asdict(profile_config("teacher", {}))
        cfg = profile_config(FRONT_ONLY_PROFILE, {"perception": clean})
        state = StudentObservationState(1, "cpu", .02, cfg)
        grid = state.grid
        hits = torch.cat((grid, torch.full((187, 1), .1)), -1)[None]
        base = torch.tensor([[0., 0., .5]])
        proprio = torch.zeros(1, 57)
        proprio[:, 5] = -1.
        state.observe(proprio, hits, base, torch.zeros(1), torch.eye(3)[None], 0)
        state.next_capture.fill_(1000)
        base[:, 0] = .2
        rotation = torch.diag(torch.tensor([-1., -1., 1.]))[None]
        output = state.observe(proprio, hits, base, torch.tensor([torch.pi]), rotation, 1)
        self.assertEqual(state.capture.item(), 0)
        self.assertGreater(int(output[:, 244:431].sum()), 0)
        for start in (57, 244, 431):
            self.assertEqual(int(torch.count_nonzero(output[:, start:start+187][:, ~state.output_mask])), 0)
        self.assertTrue(torch.equal(output, state.observe(proprio, hits, base, torch.tensor([torch.pi]), rotation, 1)))
        state.reset()
        state.next_capture.fill_(1000)
        missing = hits.clone().fill_(torch.nan)
        reset = state.observe(proprio, missing, base, torch.tensor([torch.pi]), rotation, 2)
        self.assertEqual(int(torch.count_nonzero(reset[:, 57:618])), 0)

    def test_deployment_cli_accepts_new_profile_without_changing_default(self):
        args = ["--checkpoint", "unused.pt", "--export-only", "--export-dir", "unused"]
        self.assertEqual(deploy_student_v1.parse_args(args + ["--noise-profile", FRONT_ONLY_PROFILE]).noise_profile,
                         FRONT_ONLY_PROFILE)
        self.assertEqual(deploy_student_v1.parse_args(args).noise_profile, "student-v1")


if __name__ == "__main__":
    unittest.main()
