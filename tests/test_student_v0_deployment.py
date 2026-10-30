"""Offline visibility and CPU MuJoCo deployment contracts; no optimization."""
from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deploy/deploy_mujoco"))
import deploy_mujoco as baseline
from student_v0_adapter import (SensorExtras, StudentMapAdapter, load_student,
                                horizontal_fov_mask, profile_config, visible_terrain,
                                observation_contract, validate_contract)
from deploy_student_v0 import parse_args, rollout, export_student
from m20_student_v0.perception import StudentV0PerceptionCfg

torch.set_num_threads(1)


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


def fixture_model(path):
    # Reverse XML storage order to expose direction errors in joint permutations.
    bodies = []
    for name in reversed(baseline.JOINT_NAMES):
        body = name.removesuffix("_joint")
        bodies.append(f'<body name="{body}" pos="0 0 -0.3"><joint name="{name}" axis="0 1 0"/><geom type="sphere" size="0.03" mass="0.1" group="1"/></body>')
    motors = ''.join(f'<motor name="{name}" joint="{name}"/>' for name in reversed(baseline.JOINT_NAMES))
    path.write_text('<mujoco><worldbody><geom type="plane" size="10 10 0.1" group="0"/>'
                    '<body name="base_link"><freejoint name="floating_base"/><geom type="box" size="0.2 0.1 0.1" mass="10" group="1"/>'
                    + ''.join(bodies) + '</body></worldbody><actuator>' + motors + '</actuator></mujoco>')


def fixture_checkpoint(path):
    network = torch.nn.Sequential(torch.nn.Linear(618, 512), torch.nn.ELU(), torch.nn.Linear(512, 256),
                                  torch.nn.ELU(), torch.nn.Linear(256, 128), torch.nn.ELU(), torch.nn.Linear(128, 16))
    provenance = {"contract": observation_contract(), "perception": asdict(StudentV0PerceptionCfg()),
                  "training_config": {"student": {"obs_normalization": False, "activation": "elu", "hidden_dims": [512, 256, 128]}}}
    state = {"mlp." + key: value for key, value in network.state_dict().items()}
    state["distribution.log_std_param"] = torch.zeros(16)
    torch.save({"student_state_dict": state, "student_v0": {"provenance": provenance}}, path)
    return network, provenance


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
        args = parse_args(["--checkpoint", "unused.pt", "--model", "unused.xml",
                           "--noise-profile", "lidar_fov", "--viewer"])
        self.assertEqual(args.noise_profile, "lidar_fov")
        self.assertTrue(args.viewer)
        self.assertFalse(args.autoplay)


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.robot = self.root / "robot.xml"
        fixture_model(self.robot)
        self.checkpoint = self.root / "model.pt"
        self.network, self.provenance = fixture_checkpoint(self.checkpoint)
        self.args = parse_args(["--checkpoint", str(self.checkpoint), "--model", str(self.robot), "--terrain", "flat", "--noise-profile", "clean", "--steps", "2"])
        self.model = baseline.load_model(self.robot, self.args)
        self.contract = baseline.M20Contract(self.model, self.args.terrain_config)
        self.data = mujoco.MjData(self.model)
        self.contract.reset(self.data)

    def test_clean_matches_baseline_and_both_joint_directions(self):
        action = np.arange(16, dtype=np.float32) / 10
        command = np.array([.4, -.2, .3], dtype=np.float32)
        clean = self.contract.observe(self.data, command, action)
        cfg, extras = profile_config("clean", self.provenance)
        adapter = StudentMapAdapter(self.contract, cfg, extras)
        actual = adapter.observe(self.data, command, action, 0)
        np.testing.assert_allclose(actual[:244], clean, atol=1e-6)
        np.testing.assert_array_equal(actual[244:], np.ones(374))
        reference = np.arange(16) * .01
        self.data.qpos[self.contract.qpos_ids] = reference
        # Reverse mapping must assign by name, not reuse the gathering direction.
        for i, name in enumerate(baseline.JOINT_NAMES):
            self.assertAlmostEqual(self.data.joint(name).qpos[0], reference[i])
        np.testing.assert_array_equal(self.data.qpos[self.contract.qpos_ids], reference)
        self.contract.apply_pd(self.data, action)
        np.testing.assert_array_equal(self.data.ctrl[self.contract.actuator_ids],
                                      [self.data.actuator(name).ctrl[0] for name in baseline.JOINT_NAMES])

    def test_packet_loss_and_blackout_expiry_are_unknown(self):
        cfg = StudentV0PerceptionCfg(level=1., packet_dropout=1.)
        adapter = StudentMapAdapter(self.contract, cfg, SensorExtras())
        obs = adapter.observe(self.data, np.zeros(3), np.zeros(16), 0)
        np.testing.assert_array_equal(obs[57:], np.zeros(561))
        cfg = StudentV0PerceptionCfg(level=0., max_map_age_s=.03)
        extras = SensorExtras(blackout_start_s=.02, blackout_duration_s=.2)
        adapter = StudentMapAdapter(self.contract, cfg, extras)
        self.assertEqual(adapter.observe(self.data, np.zeros(3), np.zeros(16), 0)[244:431].sum(), 187)
        self.assertEqual(adapter.observe(self.data, np.zeros(3), np.zeros(16), 2)[244:431].sum(), 0)

    def test_visibility_positive_and_occluded_negative(self):
        # The positive oracle is an unobstructed plane. A physical starting
        # mat legitimately occludes low rays just beyond its far edge.
        self.args.no_spawn_mat = True
        self.model = baseline.load_model(self.robot, self.args)
        self.contract = baseline.M20Contract(self.model, self.args.terrain_config)
        self.data = mujoco.MjData(self.model)
        self.contract.reset(self.data)
        self.contract.height_scan(self.data)
        hits = self.contract.last_scan_hits
        self.assertTrue(visible_terrain(self.model, self.data, self.contract.base_id, hits, (0,0,.15)).all())
        # Independent minimal world: a wall between a sensor and a ground point.
        model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="plane" size="10 10 .1" group="0"/>'
                  '<geom type="box" pos=".5 0 .5" size=".05 1 .5" group="0"/>'
                  '<body name="sensor" pos="0 0 .7"/></worldbody></mujoco>')
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        visible = visible_terrain(model, data, model.body("sensor").id,
                                  np.array([[.1, 0, 0.], [1., 0, 0.]]), (0,0,0))
        np.testing.assert_array_equal(visible, [True, False])

    def test_load_export_reload_equivalence_and_teacher_rejection(self):
        policy, provenance, identity = load_student(self.checkpoint)
        inputs = torch.randn(16, 618)
        with torch.inference_mode():
            self.assertTrue(torch.equal(policy(inputs), self.network(inputs)))
        self.args.export_dir = self.root / "exported"
        export_student(self.args, policy, provenance, identity)
        jit, provenance2, _ = load_student(self.args.export_dir / "student_v0.pt", jit=True)
        with torch.inference_mode():
            self.assertTrue(torch.equal(policy(inputs), jit(inputs)))
        self.assertEqual(provenance2, provenance)
        bad = self.root / "teacher.pt"
        torch.save({"actor_state_dict": {"mlp.0.weight": torch.zeros(512,244)}}, bad)
        with self.assertRaises(ValueError):
            load_student(bad)
        sidecar = self.args.export_dir / "observation_contract.json"
        wrong = json.loads(sidecar.read_text())
        wrong["slices"]["confidence"] = [244,431]
        sidecar.write_text(json.dumps(wrong))
        with self.assertRaises(ValueError):
            load_student(self.args.export_dir / "student_v0.pt", jit=True)

    def test_real_entry_physics_and_output_no_weight_update(self):
        policy, provenance, identity = load_student(self.checkpoint)
        before = {k: v.clone() for k,v in policy.state_dict().items()}
        self.args.output_dir = self.root / "rollout"
        result = rollout(self.args, policy, provenance, identity)
        self.assertAlmostEqual(result["duration_s"], .04)
        self.assertEqual(result["teacher_executed_fraction"], 0.)
        self.assertFalse(result["weights_updated"])
        values = np.load(self.args.output_dir / "trajectory.npz")
        self.assertEqual(values["observation"].shape, (2,618))
        self.assertTrue(np.isfinite(values["observation"]).all())
        np.testing.assert_allclose(values["post_time_s"] - values["time_s"], .02)
        for key, value in policy.state_dict().items():
            self.assertTrue(torch.equal(before[key], value))

    def test_reject_same_dimension_wrong_input_semantics(self):
        for key in ("normalization", "actions", "confidence"):
            incompatible = observation_contract()
            incompatible[key] = "incompatible semantics"
            with self.assertRaises(ValueError):
                validate_contract(incompatible)


if __name__ == "__main__":
    unittest.main()
