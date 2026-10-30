"""StairTrot MuJoCo contracts: no training, optimizer or GPU allocations."""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import asdict
from io import StringIO
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import mujoco
import numpy as np
import torch
from tensordict import TensorDict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/rl_training"))
sys.path.insert(0, str(ROOT / "deploy/deploy_mujoco"))
import deploy_mujoco as deployment
from motion_residual_adapter import policy_inputs, heading_controller
from stair_trot_adapter import TrotControlState, StairTrotMujocoController
from task_support.stair_trot_v1.actor import StairTrotActor
from task_support.stair_trot_v1.signals import TrotSettings

SOURCE = ROOT / "deploy/models/idle_stop_motion_v1/idle_stop_motion_v1_46400.pt"
torch.set_num_threads(1)


def checkpoint_fixture():
    source = torch.load(SOURCE, map_location="cpu", weights_only=True)
    cfg = TrotSettings()
    config = deepcopy(source["m20_training_manifest"]["actor_model_config"])
    config["class_name"] = "task_support.stair_trot_v1.actor:StairTrotActor"
    config.update(trot_hidden_dims=[256, 128],
                  correction_limits=[cfg.hipx_residual_limit, cfg.hipy_residual_limit,
                                     cfg.knee_residual_limit]*4 + [cfg.wheel_residual_limit]*4,
                  command_limits=[cfg.forward_command_min, cfg.lateral_command_max, cfg.yaw_command_max])
    obs = TensorDict({"policy": torch.zeros(1, 244), "motion_control": torch.zeros(1, 3),
                      "trot_control": torch.zeros(1, 6)}, [1])
    kwargs = deepcopy({key: value for key, value in config.items() if key != "class_name"})
    with redirect_stdout(StringIO()):
        actor = StairTrotActor(obs, {"actor": ["policy"]}, "actor", 16, **kwargs)
    actor.load_teacher_state(source["actor_state_dict"])
    # Deliberately nonzero fixture verifies that deployment really loads trot.
    with torch.no_grad():
        actor.trot_mlp[-1].bias.fill_(.3)
    manifest = dict(refinement_version="m20_stair_trot_v1", actor_model_config=config,
                    actor_input_groups={"policy": 244, "motion_control": 3, "trot_control": 6},
                    source_teacher_manifest=source["m20_training_manifest"], trot_settings=asdict(cfg),
                    task_contract={"objective": "stable_ascent", "trot_control_size": 6})
    return actor.eval(), dict(actor_state_dict=actor.state_dict(), m20_training_manifest=manifest)


class ActorDeploymentTests(unittest.TestCase):
    def test_public_loader_preserves_full_actor_and_inactive_teacher(self):
        expected, saved = checkpoint_fixture()
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            path = Path(directory)/"model.pt"
            torch.save(saved, path)
            actor = deployment.load_actor(path)
        with redirect_stdout(StringIO()):
            teacher = deployment.load_actor(SOURCE)
        np.random.seed(42)
        obs = np.random.normal(size=244).astype(np.float32)
        for command in ([.4, 0, 0], [0, 0, 0], [-.4, 0, 0], [.4, 0, .8]):
            for gate in (0., .5, 1.):
                control = np.array([gate, 1, 0, .1, .1, .995], dtype=np.float32)
                values = policy_inputs(obs, command, actor, control)
                with torch.inference_mode():
                    self.assertTrue(torch.equal(actor(values), expected(values)))
                    if gate == 0 or command != [.4, 0, 0]:
                        self.assertTrue(torch.equal(actor(values), teacher(policy_inputs(obs, command, teacher))))
        self.assertIsNotNone(heading_controller(SimpleNamespace(), actor))

    def test_loader_rejects_incomplete_contract_and_limits(self):
        _, original = checkpoint_fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"model.pt"
            for bad in ("dimension", "settings", "limits", "source"):
                saved = deepcopy(original)
                manifest = saved["m20_training_manifest"]
                if bad == "dimension":
                    manifest["actor_input_groups"]["trot_control"] = 3
                elif bad == "settings":
                    del manifest["trot_settings"]["min_rise"]
                elif bad == "limits":
                    saved["actor_state_dict"]["trot_limits"][0] = 9
                else:
                    manifest["source_teacher_manifest"]["motion_control_version"] = "unknown"
                torch.save(saved, path)
                with self.assertRaises(ValueError):
                    deployment.load_actor(path)

    def test_missing_sensor_control_fails_instead_of_silent_teacher_fallback(self):
        actor = SimpleNamespace(uses_motion_control=True, uses_trot_control=True)
        for control in (None, np.zeros(3), np.full(6, float("nan"))):
            with self.assertRaises(ValueError):
                policy_inputs(np.zeros(244, dtype=np.float32), [.4, 0, 0], actor, control)


class ControlParityTests(unittest.TestCase):
    def test_same_sensor_snapshots_match_actual_isaac_context(self):
        spec = importlib.util.spec_from_file_location("trot_training_contract_fixture", ROOT/"tests/test_stair_trot_v1.py")
        test_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(test_module)
        module, env, entities, command = test_module.EnvironmentStateTests().fixture()
        robot = env.scene["robot"].data
        sensor = env.scene["contact_forces"].data
        controller = TrotControlState(TrotSettings(), env.step_dt)
        for tick in range(30):
            env.common_step_counter = tick
            env.episode_length_buf[:] = tick
            command[:, 0] = 0 if 10 <= tick < 13 else .4
            yaw = .15 if tick >= 15 else 0.
            robot.root_quat_w[:] = torch.tensor([np.cos(yaw/2), 0, 0, np.sin(yaw/2)])
            robot.root_pos_w[:, 1] = .1 if tick >= 15 else 0.
            hits = env.scene["height_scanner"].data.ray_hits_w
            actual = module.trot_control(env, *entities)[0].numpy()
            expected = controller.observe(hits[0].numpy(), env.scene["height_scanner"].ray_starts[:, :2].numpy(),
                                          robot.root_pos_w[0, :2].numpy(), yaw, robot.body_pos_w[0].numpy(),
                                          sensor.net_forces_w[0].numpy(), sensor.current_air_time[0].numpy(),
                                          command[0].numpy(), tick * env.step_dt)
            np.testing.assert_allclose(actual, expected, atol=1.e-6, rtol=1.e-6)
            # Same timestamp reads do not advance phase; stop takes effect now.
            np.testing.assert_array_equal(expected, controller.observe(
                hits[0].numpy(), env.scene["height_scanner"].ray_starts[:, :2].numpy(),
                robot.root_pos_w[0, :2].numpy(), yaw, robot.body_pos_w[0].numpy(), sensor.net_forces_w[0].numpy(),
                sensor.current_air_time[0].numpy(), command[0].numpy(), tick * env.step_dt))
        phase = controller.phase
        stopped = controller.observe(None, None, None, 0, None, None, None, [0, 0, 0], controller.last_time)
        self.assertFalse(stopped.any())
        self.assertEqual(controller.phase, phase)
        controller.reset()
        self.assertIsNone(controller.route)

    def test_real_physics_contact_normal_points_up_and_air_time_counts_substeps(self):
        bodies = ''.join(f'<body name="wheel{i}" pos="{x} {y} .085"><freejoint/>'
                         '<geom type="sphere" size=".09" mass="1" group="1"/></body>'
                         for i, (x, y) in enumerate(((.3,.2),(.3,-.2),(-.3,.2),(-.3,-.2))))
        model = mujoco.MjModel.from_xml_string(
            '<mujoco><option timestep=".005"/><worldbody><geom type="plane" size="4 4 .1" group="0"/>'
            '<body name="base_link" pos="0 0 .5"/>'+bodies+'</worldbody></mujoco>')
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        contract = SimpleNamespace(model=model, terrain=SimpleNamespace(kind="flat"),
                                   wheel_body_ids=np.array([model.body(f"wheel{i}").id for i in range(4)]))
        actor = SimpleNamespace(m20_training_manifest={"trot_settings": asdict(TrotSettings())})
        controller = StairTrotMujocoController(contract, actor, .02)
        controller.sample_contacts(data)
        self.assertTrue((controller.forces[:, 2] > 0).all())
        np.testing.assert_allclose(controller.forces[:, :2], 0, atol=1.e-5)
        # Raise all spheres: no contacts, elapsed physics time contributes to air.
        for i in range(4):
            address = model.joint(i).qposadr[0]
            data.qpos[address+2] = .5
        for _ in range(4):
            mujoco.mj_step(model, data)
            controller.sample_contacts(data)
        np.testing.assert_allclose(controller.air_time, .02, atol=1.e-6)
        np.testing.assert_array_equal(controller.forces, 0)
        contract.terrain = SimpleNamespace(kind="ascent", spawn_xy=(-1., .1), spawn_yaw=.2)
        aligned = StairTrotMujocoController(contract, actor, .02)
        np.testing.assert_array_equal(aligned.state.fixed_route[0], [-1., 0.])
        self.assertEqual(aligned.state.fixed_route[1], 0.)


if __name__ == "__main__":
    unittest.main()
