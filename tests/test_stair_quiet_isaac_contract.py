"""CPU adapter contracts; these do not simulate impacts or prove climbing."""
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/rl_training"))
from task_support.stair_quiet_v1.isaac_landing import (
    IsaacLandingAdapter, SampleDelay, support_kinematics,
)
from task_support.stair_quiet_v1.landing import LandingSettings
from task_support.stair_quiet_v1.signals import LEGS, StairQuietSettings

torch.set_num_threads(1)


class FakeDelayedPD:
    """A causal actuator double; never imports or starts Isaac."""
    def __init__(self, joints, n):
        self.joint_indices = joints
        self.cfg = SimpleNamespace(max_delay=1)
        self.stiffness = torch.full((n, len(joints)), 80.)
        self.damping = torch.full((n, len(joints)), 2.)
        self.positions_delay_buffer = SimpleNamespace(time_lags=torch.arange(n) % 2)


class FakeRobot:
    def __init__(self, n):
        # Deliberately use a different physical joint order from the canonical
        # per-leg order. Mapping a flat first-12 slice would fail this fixture.
        self.names = [f"{leg}_{joint}_joint" for joint in ("hipx", "hipy", "knee", "wheel") for leg in LEGS]
        self.num_joints, self.is_fixed_base = 16, False
        pos = torch.tensor([[.3, .2, .13], [.3, -.2, .09],
                            [-.3, .2, .09], [-.3, -.2, .09]]).repeat(n, 1, 1)
        self.data = SimpleNamespace(
            body_link_pos_w=pos.clone(), body_com_pos_w=pos.clone(),
            body_link_quat_w=torch.tensor([1., 0., 0., 0.]).repeat(n, 4, 1),
            body_lin_vel_w=torch.zeros(n, 4, 3), body_ang_vel_w=torch.zeros(n, 4, 3),
            joint_pos=torch.zeros(n, 16), joint_vel=torch.zeros(n, 16),
            joint_pos_target=torch.zeros(n, 16), joint_vel_target=torch.zeros(n, 16),
            root_pos_w=torch.tensor([0., 0., .5]).repeat(n, 1),
            root_quat_w=torch.tensor([1., 0., 0., 0.]).repeat(n, 1),
        )
        self.data.body_lin_vel_w[:, 0, 2] = -.9
        self.jacobian = torch.zeros(n, 4, 6, 22)
        for leg in range(4):
            for coordinate, joint in enumerate(("hipx", "hipy", "knee")):
                col = self.names.index(f"{LEGS[leg]}_{joint}_joint") + 6
                self.jacobian[:, leg, coordinate, col] = .2
        self.root_physx_view = SimpleNamespace(get_jacobians=lambda: self.jacobian)
        self.actuators = {"legs": FakeDelayedPD(list(range(12)), n),
                          "wheels": FakeDelayedPD(list(range(12, 16)), n)}
        self.actuators["wheels"].stiffness.zero_()
        self.actuators["wheels"].damping.fill_(.6)

    def find_joints(self, names, preserve_order):
        assert preserve_order
        return [self.names.index(name) for name in names], names

    def set_joint_position_target(self, target, joint_ids):
        self.data.joint_pos_target[:, joint_ids] = target

    def set_joint_velocity_target(self, target, joint_ids):
        self.data.joint_vel_target[:, joint_ids] = target


def fixture(n=2, student=False, **settings):
    robot = FakeRobot(n)
    points = robot.data.body_link_pos_w.clone()
    points[..., 2] = 0.
    terrain = SimpleNamespace(data=SimpleNamespace(ray_hits_w=points),
                              _timestamp=torch.zeros(n), _timestamp_last_update=torch.zeros(n))
    force = torch.zeros(n, 4, 3)
    force[:, 1:, 2] = 100.
    contact = SimpleNamespace(data=SimpleNamespace(net_forces_w=force),
                              _timestamp=torch.zeros(n), _timestamp_last_update=torch.zeros(n))
    class Scene(dict):
        pass
    scene = Scene(robot=robot)
    scene.sensors = dict(height_scanner=terrain, contact_forces=contact)
    env = SimpleNamespace(
        scene=scene, num_envs=n, device="cpu", common_step_counter=0,
        cfg=SimpleNamespace(stair_quiet=StairQuietSettings(), stair_quiet_student=student,
                            landing_control=LandingSettings(**settings)),
        command_manager=SimpleNamespace(get_command=lambda _: torch.tensor([.5, 0., 0.]).repeat(n, 1)),
    )
    module = ModuleType("isaaclab.actuators")
    module.DelayedPDActuator = FakeDelayedPD
    with patch.dict(sys.modules, {"isaaclab.actuators": module}):
        adapter = IsaacLandingAdapter(env, list(range(4)), list(range(4)))
    if not student:
        adapter.begin_step()
    return env, adapter


class IsaacAdapterContracts(unittest.TestCase):
    def test_delay_first_sample_and_partial_reset(self):
        delay = SampleDelay(1, 2, "cpu")
        lag = torch.tensor([0, 1])
        first, ready = delay.push(torch.tensor([[1.], [10.]]), lag)
        torch.testing.assert_close(first, torch.tensor([[1.], [10.]]))
        self.assertEqual(ready.tolist(), [True, False])
        second, _ = delay.push(torch.tensor([[2.], [20.]]), lag)
        torch.testing.assert_close(second, torch.tensor([[2.], [10.]]))
        delay.reset(torch.tensor([0]))
        third, ready = delay.push(torch.tensor([[3.], [30.]]), lag)
        torch.testing.assert_close(third, torch.tensor([[3.], [20.]]))
        self.assertEqual(ready.tolist(), [True, True])

    def test_com_offset_jacobian_and_velocity_share_the_same_point(self):
        env, adapter = fixture()
        robot = env.scene["robot"]
        robot.data.body_link_pos_w[..., 0] += .04
        robot.data.body_ang_vel_w[..., 1] = 2.
        robot.data.body_lin_vel_w.zero_()
        # Set one joint column to pure world-y angular motion.
        robot.jacobian.zero_()
        for leg in range(4):
            robot.jacobian[:, leg, 4, adapter.leg_ids[leg, 0] + 6] = 1.
        point, vn, jn, jp = support_kinematics(robot, adapter.body_ids, adapter.leg_ids, env.cfg.stair_quiet)
        torch.testing.assert_close(point[..., 2], robot.data.body_link_pos_w[..., 2] - .09)
        torch.testing.assert_close(vn, torch.full((2, 4), -.08))
        torch.testing.assert_close(jn[..., 0], torch.full((2, 4), -.04))
        torch.testing.assert_close(jp[..., 0, 0], torch.full((2, 4), -.09))
        # Fixed-base views omit the root row and floating-base columns.
        robot.is_fixed_base = True
        robot.jacobian = robot.jacobian[..., 6:]
        # Use a leading root body in the data with identical wheel values.
        for name in ("body_link_pos_w", "body_com_pos_w", "body_link_quat_w",
                     "body_lin_vel_w", "body_ang_vel_w"):
            value = getattr(robot.data, name)
            setattr(robot.data, name, torch.cat((value[:, :1], value), 1))
        _, _, fixed_jn, _ = support_kinematics(robot, [1, 2, 3, 4], adapter.leg_ids, env.cfg.stair_quiet)
        torch.testing.assert_close(fixed_jn, jn)

    def test_kinematics_diagnostic_detects_a_velocity_disagreement_without_writing_state(self):
        env, adapter = fixture()
        robot = env.scene["robot"]
        robot.data.root_com_vel_w = torch.tensor([.1, .2, .3, .4, .5, .6]).repeat(2, 1)
        robot.data.joint_vel = torch.linspace(-.2, .2, 16).repeat(2, 1)
        robot.jacobian[..., :6] = torch.eye(6)
        generalized = torch.cat((robot.data.root_com_vel_w, robot.data.joint_vel), -1)
        spatial = torch.einsum("nbij,nj->nbi", robot.jacobian, generalized)
        robot.data.body_lin_vel_w = spatial[..., :3].clone()
        robot.data.body_ang_vel_w = spatial[..., 3:].clone()
        before = robot.data.joint_pos_target.clone()
        result = adapter.kinematics_diagnostic()
        self.assertEqual(result["body_com_linear_error_m_s"], 0.)
        self.assertEqual(result["support_point_velocity_error_m_s"], 0.)
        self.assertEqual(result["floating_root_identity_block_max_error"], 0.)
        self.assertGreater(result["observed_support_speed_m_s"], 0.)
        robot.data.body_lin_vel_w[:, 0, 2] += .2
        self.assertAlmostEqual(adapter.kinematics_diagnostic()["support_point_velocity_error_m_s"], .2, places=5)
        torch.testing.assert_close(before, robot.data.joint_pos_target)

    def test_actual_core_writes_nonzero_targets_and_gain_lag_matches_target_lag(self):
        env, adapter = fixture(startup_grace_s=0, minimum_air_time_s=0)
        robot = env.scene["robot"]
        leg = robot.actuators["legs"]
        requested_gains = []
        for _ in range(4):
            # This is the action manager's nominal refresh, once per substep.
            robot.data.joint_pos_target.zero_()
            adapter.apply(.005)
            requested_gains.append(adapter.last["kp_scale"][:, 0].clone() * 80.)
            torch.testing.assert_close(robot.data.joint_pos_target[:, adapter.leg_ids], adapter.last["leg_target"])
        self.assertGreater(adapter.last["correction"].abs().max().item(), 0)
        self.assertLess(leg.stiffness[0, 0], 80)
        torch.testing.assert_close(leg.stiffness[0, 0], requested_gains[-1][0])
        torch.testing.assert_close(leg.stiffness[1, 0], requested_gains[-2][1])
        torch.testing.assert_close(robot.actuators["wheels"].damping, torch.full((2, 4), .6))
        self.assertEqual(adapter.summary()["terrain_valid_fraction"], 1.)
        self.assertEqual(adapter.applied_substeps, 4)

    def test_reset_restores_base_gains_then_captures_randomization_without_compounding(self):
        env, adapter = fixture(startup_grace_s=0, minimum_air_time_s=0)
        for _ in range(4):
            env.scene["robot"].data.joint_pos_target.zero_()
            adapter.apply(.005)
        leg = env.scene["robot"].actuators["legs"]
        other = leg.stiffness[1].clone()
        ids = torch.tensor([0])
        adapter.before_reset(ids)
        torch.testing.assert_close(leg.stiffness[0], torch.full((12,), 80.))
        leg.stiffness[0] = 88.  # existing reset randomization
        adapter.reset(ids)
        torch.testing.assert_close(leg.stiffness[1], other)
        self.assertEqual(adapter.controller.correction[0].abs().sum(), 0)
        env.scene["robot"].data.joint_pos_target.zero_()
        adapter.apply(.005)
        torch.testing.assert_close(leg.stiffness[0], torch.full((12,), 88.))

    def test_off_preserves_nominal_targets_and_gains(self):
        env, adapter = fixture(mode="off")
        robot = env.scene["robot"]
        robot.data.joint_pos_target = torch.randn(2, 16)
        robot.data.joint_vel_target = torch.randn(2, 16)
        position, velocity = robot.data.joint_pos_target.clone(), robot.data.joint_vel_target.clone()
        for _ in range(4):
            adapter.apply(.005)
        torch.testing.assert_close(robot.data.joint_pos_target, position, rtol=0, atol=0)
        torch.testing.assert_close(robot.data.joint_vel_target, velocity, rtol=0, atol=0)
        self.assertEqual(adapter.summary()["modified_samples"], 0)
        torch.testing.assert_close(robot.actuators["legs"].stiffness, torch.full((2, 12), 80.))

    def test_contact_delay_dropout_stale_and_missing_timing_fail_closed(self):
        env, adapter = fixture(contact_delay_steps=1)
        adapter.apply(.005)
        self.assertFalse(adapter.contact_valid.any())
        env.scene.sensors["contact_forces"].data.net_forces_w[..., 2] = 222.
        adapter.apply(.005)
        self.assertTrue(adapter.contact_valid.all())
        # Core sees the previous completed physics sample.
        torch.testing.assert_close(adapter.controller.contact, torch.zeros(2, 4, dtype=torch.bool))
        env, adapter = fixture(contact_dropout_probability=1.)
        adapter.apply(.005)
        self.assertFalse(adapter.contact_valid.any())
        env, adapter = fixture()
        env.scene.sensors["contact_forces"]._timestamp.fill_(.1)
        adapter.apply(.005)
        self.assertFalse(adapter.contact_valid.any())
        del env.scene.sensors["contact_forces"]._timestamp
        with self.assertRaisesRegex(RuntimeError, "capture timestamps"):
            adapter.apply(.005)

    def test_student_uses_delivered_map_and_freezes_capture_pose(self):
        env, adapter = fixture(student=True, startup_grace_s=0, minimum_air_time_s=0)
        robot = env.scene["robot"]
        height = torch.zeros(2, 4)
        state = SimpleNamespace(last_channels=(height, torch.ones_like(height), torch.ones_like(height)),
                                last_step=0, grid=robot.data.body_link_pos_w[0, :, :2].clone(),
                                cell_age=torch.full((2, 4), .35))
        env._m20_student_v1_state = state
        # Oracle is deliberately unusable: a hidden fallback would fail.
        env.scene.sensors["height_scanner"].data.ray_hits_w.fill_(float("nan"))
        adapter.elapsed.fill_(.35)
        adapter.begin_step()
        frozen = adapter._student_map[0].clone()
        robot.data.root_pos_w[:, :2] += 5.
        for _ in range(4):
            robot.data.joint_pos_target.zero_()
            adapter.apply(.005)
        torch.testing.assert_close(adapter._student_map[0], frozen)
        self.assertTrue(adapter.terrain_valid.all())
        self.assertGreater(adapter.last["correction"].abs().max().item(), 0)
        self.assertEqual(adapter.summary()["terrain_valid_fraction"], 1.)
        adapter.elapsed.fill_(.55)  # Original observed cells survive beyond ingest age.
        adapter.apply(.005)
        self.assertTrue(adapter.terrain_valid.all())
        # Expiry follows the sole settings ledger, not a test-local duration.
        adapter.elapsed.fill_(adapter.settings.terrain_memory_s + .01)
        adapter.apply(.005)
        self.assertFalse(adapter.terrain_valid.any())
        self.assertFalse(adapter.summary()["real_deployment_ready"])

    def test_student_missing_clipped_or_unconfident_cells_do_not_become_floor(self):
        env, adapter = fixture(student=True)
        env._m20_student_v1_state = SimpleNamespace(last_channels=None, last_step=0)
        adapter.begin_step()
        adapter.apply(.005)
        self.assertFalse(adapter.terrain_valid.any())
        self.assertTrue(torch.isnan(adapter.clearance).all())
        height = torch.tensor([[1., -1., 0., 0.]]).repeat(2, 1)
        state = SimpleNamespace(last_channels=(height, torch.ones_like(height), torch.tensor([[1., 1., 0., float("nan")]]).repeat(2, 1)),
                                last_step=0, grid=env.scene["robot"].data.body_link_pos_w[0, :, :2],
                                cell_age=torch.zeros(2, 4))
        env._m20_student_v1_state = state
        adapter.begin_step()
        adapter.apply(.005)
        self.assertFalse(adapter.terrain_valid.any())

    def test_rear_wheel_uses_previously_seen_tread_without_an_oracle_fallback(self):
        env, adapter = fixture(student=True, startup_grace_s=0, minimum_air_time_s=0)
        robot = env.scene["robot"]
        height = torch.zeros(2, 4)
        observed = torch.tensor([[1., 1., 0., 0.]]).repeat(2, 1)
        state = SimpleNamespace(
            last_channels=(height, observed, observed.clone()), last_step=0,
            grid=robot.data.body_link_pos_w[0, :, :2].clone(), cell_age=torch.full((2, 4), .35),
        )
        env._m20_student_v1_state = state
        env.scene.sensors["height_scanner"].data.ray_hits_w.fill_(float("nan"))
        adapter.elapsed.fill_(.35)
        adapter.begin_step()
        adapter.apply(.005)
        self.assertTrue(adapter.terrain_valid[:, :2].all())
        self.assertFalse(adapter.terrain_valid[:, 2:].any())
        state.last_channels = None  # no current map during the next traversal
        contact = env.scene.sensors["contact_forces"].data.net_forces_w
        contact[:, 0, 2], contact[:, 2, 2] = 100., 0.
        robot.data.body_lin_vel_w[:, 2, 2] = -.9
        for _ in range(6):
            # Small consecutive pose changes cannot be mistaken for teleport.
            robot.data.root_pos_w[:, 0] += .1
            robot.data.body_link_pos_w[..., 0] += .1
            robot.data.body_com_pos_w[..., 0] += .1
            robot.data.joint_pos_target.zero_()
            adapter.begin_step()
            adapter.apply(.005)
        self.assertTrue(adapter.terrain_valid[:, 2:].all())
        self.assertGreater(adapter.last["correction"][:, 2].abs().max().item(), 0.)
        self.assertGreater(adapter.summary()["terrain_valid_samples_per_wheel"][2], 0)
        adapter.before_reset(torch.tensor([0]))
        adapter.reset(torch.tensor([0]))
        adapter.apply(.005)
        self.assertFalse(adapter.terrain_valid[0].any())
        self.assertTrue(adapter.terrain_valid[1, 2:].all())
        robot.data.root_pos_w[:, 0] += 1.
        adapter.begin_step()
        adapter.apply(.005)
        self.assertFalse(adapter.terrain_valid.any())


if __name__ == "__main__":
    unittest.main()
