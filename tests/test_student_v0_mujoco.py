"""Behavioral contract checks for CPU MuJoCo deployment; no optimization."""
from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
import unittest

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deploy/deploy_mujoco"))
import deploy_mujoco as baseline
from student_v0_adapter import (SensorExtras, StudentMapAdapter, load_student,
                                profile_config, visible_terrain, observation_contract, validate_contract)
from deploy_student_v0 import parse_args, rollout, export_student
from m20_student_v0.perception import StudentV0PerceptionCfg

torch.set_num_threads(1)


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
