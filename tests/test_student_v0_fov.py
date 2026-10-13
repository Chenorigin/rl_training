"""Offline FOV/observation checks; no MuJoCo model, rollout or policy load."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deploy/deploy_mujoco"))
from student_v0_adapter import (SensorExtras, StudentMapAdapter,
                                horizontal_fov_mask, profile_config, visible_terrain)


def pose(rotation=None, position=(0., 0., .6)):
    rotation = np.eye(3) if rotation is None else rotation
    return SimpleNamespace(xpos=np.array([position]), xmat=rotation.reshape(1, 9))


def flat_ray(model, data, origin, direction, *args):
    """Analytic ground intersection in place of MuJoCo's ray query."""
    return -origin[2] / direction[2] if direction[2] < 0 else -1.


class FlatContract:
    base_id = 0
    model = object()

    def observe(self, data, command, previous_action):
        yy, xx = np.meshgrid(np.arange(11)*.1-.5, np.arange(17)*.1-.8, indexing="ij")
        rotation = data.xmat[0].reshape(3, 3)
        yaw = np.arctan2(rotation[1,0], rotation[0,0])
        self.last_scan_hits = np.column_stack((
            data.xpos[0,0] + np.cos(yaw)*xx.ravel() - np.sin(yaw)*yy.ravel(),
            data.xpos[0,1] + np.sin(yaw)*xx.ravel() + np.cos(yaw)*yy.ravel(), np.zeros(187)))
        clean = np.zeros(244, dtype=np.float32)
        clean[5] = -1.
        clean[57:] = .1
        return clean


class FovTests(unittest.TestCase):
    def test_front_rear_and_sides(self):
        # Both +/- boundaries are included, just outside is excluded.
        degrees = np.array([0., 67.5, -67.5, 67.51, -67.51, 90., -90., 180.])
        angles = np.deg2rad(degrees)
        hits = np.column_stack((np.cos(angles), np.sin(angles), np.zeros(len(angles))))
        front = horizontal_fov_mask(hits, np.zeros(3), np.eye(3), 135.)
        rear = horizontal_fov_mask(hits, np.zeros(3), np.eye(3), 135., rear=True)
        np.testing.assert_array_equal(front, [True, True, True, False, False, False, False, False])
        np.testing.assert_array_equal(rear, [False, False, False, False, False, False, False, True])

    def test_fov_follows_body_pose_not_world_axis(self):
        yaw, pitch = np.deg2rad(73.), np.deg2rad(24.)
        rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0,0,1]])
        ry = np.array([[np.cos(pitch),0,np.sin(pitch)], [0,1,0], [-np.sin(pitch),0,np.cos(pitch)]])
        rotation = rz @ ry
        origin = np.array([3., -2., .8])
        local_hits = np.array([[1.,0,-.4], [-1.,0,-.4], [0,1,-.4], [0,-1,-.4]])
        hits = local_hits @ rotation.T + origin
        front = horizontal_fov_mask(hits, origin, rotation, 135.)
        rear = horizontal_fov_mask(hits, origin, rotation, 135., rear=True)
        np.testing.assert_array_equal(front, [True, False, False, False])
        np.testing.assert_array_equal(front | rear, [True, True, False, False])

    def test_nonfinite_and_vertical_points_cannot_leak(self):
        hits = np.array([[np.nan,0,0], [np.inf,0,0], [0,0,-1.], [1.,0,0]])
        with np.errstate(invalid="ignore"):
            np.testing.assert_array_equal(horizontal_fov_mask(hits, np.zeros(3), np.eye(3), 135.),
                                          [False, False, False, True])
        np.testing.assert_array_equal(horizontal_fov_mask(hits, np.zeros(3), np.eye(3), 360.),
                                      [False, False, True, True])

    def test_each_sensor_uses_its_own_origin_and_los(self):
        hits = np.array([[1.,0,0], [-1.,0,0], [0,1.,0], [0,-1.,0]])
        seen_origins = []
        def ray(model, data, origin, direction, *args):
            seen_origins.append(origin.copy())
            # Front ground is occluded, rear ground is visible.
            return .01 if origin[0] > 0 else flat_ray(model, data, origin, direction)
        with patch("student_v0_adapter.mujoco.mj_ray", side_effect=ray):
            result = visible_terrain(object(), pose(), 0, hits, (.1,0,.15), 135., (-.1,0,.15))
        np.testing.assert_array_equal(result, [False, True, False, False])
        self.assertEqual(len(seen_origins), 2)  # Side blind points never get LOS rays.
        np.testing.assert_allclose(seen_origins, [[.1,0,.75], [-.1,0,.75]])

    def test_actual_adapter_keeps_618_and_unknown_channels_zero(self):
        cfg, extras = profile_config("lidar_fov", {})
        adapter = StudentMapAdapter(FlatContract(), cfg, extras)
        with patch("student_v0_adapter.mujoco.mj_ray", side_effect=flat_ray):
            obs = adapter.observe(pose(), np.zeros(3), np.zeros(16), 0)
        self.assertEqual(obs.shape, (618,))
        self.assertTrue(np.isfinite(obs).all())
        h, m, c = (obs[57:244], obs[244:431], obs[431:618])
        mask = adapter.last_visibility
        self.assertGreater(mask.sum(), 0)
        self.assertLess(mask.sum(), 187)
        np.testing.assert_array_equal(m.astype(bool), mask)
        np.testing.assert_array_equal(h[~mask], np.zeros((~mask).sum()))
        np.testing.assert_array_equal(c[~mask], np.zeros((~mask).sum()))
        np.testing.assert_array_equal(c[mask], np.ones(mask.sum()))
        for y in (-.5, .5):
            idx = np.where(np.isclose(adapter.contract.last_scan_hits[:,0], 0) &
                           np.isclose(adapter.contract.last_scan_hits[:,1], y))[0]
            self.assertEqual(len(idx), 1)
            self.assertEqual(m[idx[0]], 0.)
        # Existing clean reference is still a full map.
        clean_cfg, clean_extras = profile_config("clean", {})
        full = StudentMapAdapter(FlatContract(), clean_cfg, clean_extras).observe(pose(), np.zeros(3), np.zeros(16), 0)
        np.testing.assert_array_equal(full[244:], np.ones(374))

    def test_config_profiles_and_invalid_options(self):
        for name in ("lidar_fov", "realistic", "stress"):
            cfg, extras = profile_config(name, {})
            self.assertTrue(extras.lidar_visibility)
            self.assertEqual(extras.lidar_horizontal_fov_deg, 135.)
            self.assertIsNotNone(extras.rear_sensor_offset_body_m)
        for value in (0., -1., 361., float("nan")):
            with self.assertRaises(ValueError):
                SensorExtras(lidar_horizontal_fov_deg=value).validate()
        with self.assertRaises(ValueError):
            SensorExtras(rear_sensor_offset_body_m=(0., 0.)).validate()
        # Legacy one-origin configuration remains accepted.
        SensorExtras().validate()

    def test_cached_observation_is_allowed_in_current_blind_sector_then_expires(self):
        cfg, extras = profile_config("lidar_fov", {})
        cfg.max_map_age_s = .03
        extras.blackout_start_s, extras.blackout_duration_s = .02, .1
        adapter = StudentMapAdapter(FlatContract(), cfg, extras)
        with patch("student_v0_adapter.mujoco.mj_ray", side_effect=flat_ray):
            adapter.observe(pose(), np.zeros(3), np.zeros(16), 0)
            turned = pose(np.array([[0.,-1.,0.], [1.,0.,0.], [0.,0.,1.]]))
            obs = adapter.observe(turned, np.zeros(3), np.zeros(16), 1)
            # The side point now corresponds to previously observed front ground.
            idx = 10*17 + 8  # current body x=0, y=+0.5
            self.assertFalse(adapter.last_visibility[idx])
            self.assertEqual(obs[244+idx], 1.)
            self.assertGreater(obs[431+idx], 0.)
            self.assertLess(obs[431+idx], 1.)
            expired = adapter.observe(turned, np.zeros(3), np.zeros(16), 2)
        np.testing.assert_array_equal(expired[57:], np.zeros(561))

    def test_legacy_full_view_and_cli_profile_selection(self):
        hits = np.array([[1.,0,0], [-1.,0,0], [0,1.,0], [0,-1.,0]])
        with patch("student_v0_adapter.mujoco.mj_ray", side_effect=flat_ray):
            np.testing.assert_array_equal(visible_terrain(object(), pose(), 0, hits, (.1,0,.15)),
                                          np.ones(4, dtype=bool))
        from deploy_student_v0 import parse_args
        args = parse_args(["--checkpoint", "unused.pt", "--model", "unused.xml",
                           "--noise-profile", "lidar_fov", "--viewer"])
        self.assertEqual(args.noise_profile, "lidar_fov")
        self.assertTrue(args.viewer)
        self.assertFalse(args.autoplay)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
