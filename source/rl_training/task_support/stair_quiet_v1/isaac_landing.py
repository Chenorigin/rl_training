"""Isaac binding of the shared landing controller, with explicit oracle inputs.

No simulator imports at module load: kinematics and delay semantics are CPU
testable. PhysX Jacobians map to link COM velocities in WORLD coordinates:
https://nvidia-omniverse.github.io/PhysX/physx/5.3.0/docs/Articulations.html
"""
from __future__ import annotations

import torch

from .landing import LandingController
from .signals import LEGS, contact_point_kinematics, wheel_axis_world
from .terrain_memory import TerrainMemory


class SampleDelay:
    """Batched delay with per-environment reset and oldest-available warmup.

    Gain warmup matches DelayedPDActuator: repeat the first issued sample.
    Contact callers additionally use ``ready`` to reject unavailable history.
    """

    def __init__(self, maximum_delay, num_envs, device):
        self.length = int(maximum_delay) + 1
        self.count = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.index = -1
        self.history = None

    def reset(self, ids):
        self.count[ids] = 0

    def push(self, value, lag):
        if self.history is None:
            self.history = torch.zeros((self.length, *value.shape), device=value.device, dtype=value.dtype)
        if self.history.shape[1:] != value.shape:
            raise ValueError("Delayed sample shape changed")
        self.index = (self.index + 1) % self.length
        self.history[self.index] = value
        self.count.add_(1).clamp_(max=self.length)
        lag = torch.as_tensor(lag, device=value.device, dtype=torch.long).expand(value.shape[0])
        available = torch.minimum(lag, self.count - 1)
        result = self.history[(self.index - available) % self.length,
                              torch.arange(value.shape[0], device=value.device)]
        return result.clone(), self.count > lag


def support_kinematics(robot, body_ids, leg_ids, settings):
    """Horizontal capsule support point and its 3-leg-joint normal Jacobian.

    Root columns are excluded from the correction Jacobian, while measured
    point velocity retains base motion. Offsets are measured from COM in BOTH
    the Jacobian and v_com + omega x r expressions.
    """
    data = robot.data
    quat = data.body_link_quat_w[:, body_ids]
    axis = wheel_axis_world(quat, settings)
    up = torch.zeros_like(data.body_lin_vel_w[:, body_ids])
    up[..., 2] = 1.
    velocity, _, offset = contact_point_kinematics(
        data.body_lin_vel_w[:, body_ids], data.body_ang_vel_w[:, body_ids], up, axis,
        data.body_link_pos_w[:, body_ids] - data.body_com_pos_w[:, body_ids],
        settings, validate=False,
    )
    point = data.body_com_pos_w[:, body_ids] + offset
    body_rows = [i - 1 if robot.is_fixed_base else i for i in body_ids]
    if any(i < 0 for i in body_rows):
        raise ValueError("A wheel cannot be the fixed root body")
    jacobian = robot.root_physx_view.get_jacobians()[:, body_rows]
    columns = leg_ids + (0 if robot.is_fixed_base else 6)
    selected = torch.gather(
        jacobian, -1, columns[None, :, None, :].expand(jacobian.shape[0], 4, 6, 3),
    )
    # Cross each angular-velocity Jacobian column with COM-to-support offset.
    point_jacobian = selected[..., :3, :] + torch.linalg.cross(
        selected[..., 3:, :].transpose(-1, -2), offset[..., None, :], dim=-1,
    ).transpose(-1, -2)
    return point, velocity[..., 2], point_jacobian[..., 2, :], point_jacobian


class IsaacLandingAdapter:
    """Task-local hook after nominal action application, before actuator compute."""

    def __init__(self, env, body_ids, contact_ids):
        self.env = env
        self.robot = env.scene["robot"]
        self.settings = env.cfg.landing_control
        self.body_ids, self.contact_ids = body_ids, contact_ids
        expected = [f"{leg}_{joint}_joint" for leg in LEGS for joint in ("hipx", "hipy", "knee")]
        ids, names = self.robot.find_joints(expected, preserve_order=True)
        if names != expected:
            raise ValueError("Landing requires FL/FR/HL/HR hipx/hipy/knee joint order")
        self.leg_ids = torch.tensor(ids, device=env.device, dtype=torch.long).reshape(4, 3)
        expected = [f"{leg}_wheel_joint" for leg in LEGS]
        self.wheel_ids, names = self.robot.find_joints(expected, preserve_order=True)
        if names != expected:
            raise ValueError("Landing wheel joint order changed")
        self.controller = LandingController(env.num_envs, env.device, self.settings)
        self.contact_delay = SampleDelay(self.settings.contact_delay_steps, env.num_envs, env.device)
        self.elapsed = torch.zeros(env.num_envs, device=env.device)
        self.last = None
        self.clearance = torch.zeros(env.num_envs, 4, device=env.device)
        self.terrain_capture_age = torch.full_like(self.clearance, torch.nan)
        self.terrain_valid = torch.zeros_like(self.clearance, dtype=torch.bool)
        self.contact_valid = torch.zeros_like(self.terrain_valid)
        self.applied_substeps = 0
        self.input_counts = torch.zeros(3, dtype=torch.long, device=env.device)
        self.terrain_counts_per_wheel = torch.zeros(4, dtype=torch.long, device=env.device)
        self._student_map = None
        self.terrain_memory = None
        self._memory_empty = None
        self._gains = []
        # Use the explicit PD model, never PhysX drive gains (which would add a
        # second controller and bypass the explicit actuator's effort clipping).
        from isaaclab.actuators import DelayedPDActuator
        global_leg = {joint: i // 3 for i, joint in enumerate(ids)}
        for name, actuator in self.robot.actuators.items():
            if not isinstance(actuator, DelayedPDActuator):
                raise TypeError(f"Landing only supports the pinned explicit delayed PD actuator: {name}")
            joint_indices = actuator.joint_indices
            if isinstance(joint_indices, slice):
                indices = list(range(self.robot.num_joints))[joint_indices]
            else:
                indices = torch.as_tensor(joint_indices).tolist()
            leg = torch.tensor([global_leg.get(j, -1) for j in indices], device=env.device)
            self._gains.append({
                "actuator": actuator, "leg": leg, "base_kp": actuator.stiffness.clone(),
                "base_kd": actuator.damping.clone(),
                "delay": SampleDelay(actuator.cfg.max_delay, env.num_envs, env.device),
            })

    def contract(self):
        student = getattr(self.env.cfg, "stair_quiet_student", False)
        return {
            "adapter": "isaac_explicit_delayed_pd_v1",
            "terrain_source": ("student_cached_noisy_map_valid_confidence_age" if student else
                               "simulation_oracle_height_scanner_horizontal_treads_only"),
            "terrain_memory": "bounded_previously_observed_world_points_with_original_capture_age",
            "terrain_memory_pose_source": "simulation_ground_truth_pose; real_LIO_not_validated",
            "terrain_memory_scope": "static_treads; conflict_expiry_pose_jump_and_reset_invalidate",
            "contact_source": "simulation_oracle_pre_step_net_force_world_z",
            "contact_is_real_sensor": False,
            "normal_frame": "world_positive_z",
            "jacobian_reference": "PhysX_link_COM_to_verified_capsule_support",
            "target_gain_delay": "same_explicit_actuator_per_environment_lag",
            "previous_action": "nominal_policy_action_before_landing",
            "real_deployment_ready": False,
        }

    def before_reset(self, ids):
        # Reset events may change baseline gains. Restore first, then recapture
        # AFTER the original reset/randomization so scaling never compounds.
        for entry in self._gains:
            actuator = entry["actuator"]
            actuator.stiffness[ids] = entry["base_kp"][ids]
            actuator.damping[ids] = entry["base_kd"][ids]

    def reset(self, ids):
        self.controller.reset(ids)
        self.contact_delay.reset(ids)
        self.elapsed[ids] = 0
        self.clearance[ids] = 0
        self.terrain_capture_age[ids] = torch.nan
        self.terrain_valid[ids] = False
        self.contact_valid[ids] = False
        if self.terrain_memory is not None:
            self.terrain_memory.reset(ids)
        for entry in self._gains:
            actuator = entry["actuator"]
            entry["base_kp"][ids] = actuator.stiffness[ids]
            entry["base_kd"][ids] = actuator.damping[ids]
            entry["delay"].reset(ids)

    def begin_step(self):
        """Freeze the student's actual observation map at its capture pose.

        ObservationState.last_channels is on the current yaw-aligned grid,
        including its delivery delay/occlusion/noise. Never replace a missing
        student cell by an oracle ray hit. No observation/history is advanced.
        """
        if not getattr(self.env.cfg, "stair_quiet_student", False):
            terrain = self.env.scene.sensors["height_scanner"]
            points = terrain.data.ray_hits_w
            age = self._sensor_age(terrain)[:, None].expand(points.shape[:2])
            self._update_memory(points, torch.isfinite(points).all(-1), age)
            return
        from task_support.student_v1 import environment as glue
        from task_support.student_v1.contract import HEIGHT_OFFSET, HEIGHT_CLIP
        from task_support.student_v1.perception import quaternion_rotation
        state = glue.sensor_state(self.env)
        if state.last_channels is None or state.last_step != self.env.common_step_counter:
            self._student_map = None
            if self._memory_empty is not None:
                points, valid, age = self._memory_empty
                self._update_memory(points, valid, age)
            return
        height, valid, confidence = state.last_channels
        data = self.robot.data
        rotation = quaternion_rotation(data.root_quat_w)
        yaw = torch.atan2(rotation[:, 1, 0], rotation[:, 0, 0])
        grid = state.grid
        points = torch.empty((*height.shape, 3), device=height.device, dtype=height.dtype)
        points[..., 0] = data.root_pos_w[:, 0:1] + yaw.cos()[:, None] * grid[:, 0] - yaw.sin()[:, None] * grid[:, 1]
        points[..., 1] = data.root_pos_w[:, 1:2] + yaw.sin()[:, None] * grid[:, 0] + yaw.cos()[:, None] * grid[:, 1]
        points[..., 2] = data.root_pos_w[:, 2:3] - HEIGHT_OFFSET - height
        usable = (valid > .5) & (confidence > 0) & torch.isfinite(confidence)
        usable &= torch.isfinite(points).all(-1) & (height > HEIGHT_CLIP[0]) & (height < HEIGHT_CLIP[1])
        self._student_map = (points, usable, state.cell_age.clone(), self.elapsed.clone())
        self._update_memory(points, usable, state.cell_age, yaw=yaw)

    def _update_memory(self, points, valid, age, *, yaw=None):
        """Ingest only policy-tick observations, retaining their capture age.

        World placement currently uses simulation pose truth. Real deployment
        needs a validated LIO estimate and its uncertainty/jump handling.
        """
        if self.terrain_memory is None:
            self.terrain_memory = TerrainMemory(
                self.env.num_envs, points.shape[1], self.env.device, self.settings)
            self._memory_empty = (torch.zeros_like(points), torch.zeros_like(valid), torch.zeros_like(age))
        if yaw is None:
            from task_support.student_v1.perception import quaternion_rotation
            rotation = quaternion_rotation(self.robot.data.root_quat_w)
            yaw = torch.atan2(rotation[:, 1, 0], rotation[:, 0, 0])
        self.terrain_memory.update(
            points_w=points, valid=valid, age_s=age,
            pose_w=self.robot.data.root_pos_w, yaw_w=yaw, time_s=self.elapsed,
        )

    @staticmethod
    def _sensor_age(sensor):
        # Isaac SensorBase maintains these per-env timestamps. Missing timing
        # cannot be interpreted as a fresh measurement.
        if not hasattr(sensor, "_timestamp") or not hasattr(sensor, "_timestamp_last_update"):
            raise RuntimeError("Landing requires sensor capture timestamps")
        return sensor._timestamp - sensor._timestamp_last_update

    def apply(self, dt):
        point, normal_velocity, normal_jacobian, point_jacobian = support_kinematics(
            self.robot, self.body_ids, self.leg_ids, self.env.cfg.stair_quiet,
        )
        if self.terrain_memory is None:
            height = torch.full_like(self.clearance, torch.nan)
            terrain_valid = torch.zeros_like(self.terrain_valid)
            self.terrain_capture_age.fill_(torch.nan)
        else:
            height, terrain_valid, self.terrain_capture_age = self.terrain_memory.sample(
                point[..., :2], time_s=self.elapsed)
        self.clearance = point[..., 2] - height
        self.terrain_valid = terrain_valid
        sensor = self.env.scene.sensors["contact_forces"]
        force = sensor.data.net_forces_w[:, self.contact_ids]
        age = self._sensor_age(sensor)
        contact_valid = torch.isfinite(force).all(-1) & torch.isfinite(age[:, None])
        contact_valid &= age[:, None] >= 0
        if self.settings.contact_dropout_probability:
            contact_valid &= torch.rand_like(contact_valid, dtype=torch.float32) >= self.settings.contact_dropout_probability
        # A vertical riser force is not evidence of load-bearing tread contact.
        normal_force = torch.nan_to_num(force[..., 2], nan=0., posinf=0., neginf=0.).clamp_min(0)
        captured = self.elapsed[:, None].expand_as(normal_force) - torch.nan_to_num(age[:, None], nan=0.)
        packed = torch.stack((normal_force, captured, contact_valid.to(normal_force.dtype)), -1)
        delayed, ready = self.contact_delay.push(packed, self.settings.contact_delay_steps)
        contact_age = self.elapsed[:, None] - delayed[..., 1]
        self.contact_valid = (delayed[..., 2] > .5) & ready[:, None]
        self.contact_valid &= (contact_age >= 0) & (contact_age <= self.settings.contact_max_age_s)
        self.input_counts += torch.stack((
            torch.ones_like(self.terrain_valid).sum(),
            self.terrain_valid.sum(), self.contact_valid.sum(),
        ))
        self.terrain_counts_per_wheel += self.terrain_valid.sum(0)
        command = self.env.command_manager.get_command("base_velocity")
        active = command.abs().amax(-1) > self.settings.command_active_threshold
        data = self.robot.data
        surface_normal = torch.zeros_like(point)
        surface_normal[..., 2] = 1.
        self.last = self.controller.step(
            nominal_leg_target=data.joint_pos_target[:, self.leg_ids],
            nominal_wheel_velocity=data.joint_vel_target[:, self.wheel_ids],
            joint_position=data.joint_pos[:, self.leg_ids],
            joint_velocity=data.joint_vel[:, self.leg_ids],
            normal_jacobian=normal_jacobian, normal_velocity=normal_velocity,
            point_jacobian=point_jacobian, surface_normal=surface_normal,
            clearance=self.clearance, terrain_valid=self.terrain_valid,
            contact_force=delayed[..., 0], contact_valid=self.contact_valid,
            command_active=active, dt=dt,
        )
        self.robot.set_joint_position_target(self.last["leg_target"].flatten(1), joint_ids=self.leg_ids.flatten())
        self.robot.set_joint_velocity_target(self.last["wheel_velocity"], joint_ids=self.wheel_ids)
        for entry in self._gains:
            actuator, leg = entry["actuator"], entry["leg"]
            kp_scale = torch.where(leg[None] >= 0, self.last["kp_scale"][:, leg.clamp_min(0)], 1.)
            kd_scale = torch.where(leg[None] >= 0, self.last["kd_scale"][:, leg.clamp_min(0)], 1.)
            request = torch.stack((entry["base_kp"] * kp_scale, entry["base_kd"] * kd_scale), -1)
            gains, _ = entry["delay"].push(request, actuator.positions_delay_buffer.time_lags)
            actuator.stiffness.copy_(gains[..., 0])
            actuator.damping.copy_(gains[..., 1])
        self.elapsed += dt
        self.applied_substeps += 1

    def summary(self):
        samples, terrain, contact = self.input_counts.tolist()
        return {
            **self.controller.summary(), "applied_physics_substeps": self.applied_substeps,
            "terrain_valid_samples": terrain, "contact_valid_samples": contact,
            "terrain_valid_samples_per_wheel": self.terrain_counts_per_wheel.tolist(),
            "terrain_valid_fraction": terrain / samples if samples else None,
            "contact_valid_fraction": contact / samples if samples else None,
            "terrain_source": self.contract()["terrain_source"],
            "contact_source": self.contract()["contact_source"],
            "terrain_memory": self.terrain_memory.summary() if self.terrain_memory is not None else None,
            "real_deployment_ready": False,
        }

    def kinematics_diagnostic(self):
        """Read-only full-J velocity check, called by bounded validation only.

        The pinned tensor API documents world-frame floating-root velocities.
        ArticulationData.root_com_vel_w gives [linear, angular] at root COM.
        The root 6x6 identity block independently checks this column ordering.
        No joint state, target, or simulator time is changed here.
        """
        data = self.robot.data
        point, _, _, _ = support_kinematics(
            self.robot, self.body_ids, self.leg_ids, self.env.cfg.stair_quiet)
        jacobians = self.robot.root_physx_view.get_jacobians()
        rows = [i - 1 if self.robot.is_fixed_base else i for i in self.body_ids]
        generalized = data.joint_vel if self.robot.is_fixed_base else torch.cat(
            (data.root_com_vel_w, data.joint_vel), -1)
        predicted = torch.einsum("nbij,nj->nbi", jacobians[:, rows], generalized)
        offset = point - data.body_com_pos_w[:, self.body_ids]
        predicted_point = predicted[..., :3] + torch.linalg.cross(predicted[..., 3:], offset)
        measured_linear = data.body_lin_vel_w[:, self.body_ids]
        measured_angular = data.body_ang_vel_w[:, self.body_ids]
        measured_point = measured_linear + torch.linalg.cross(measured_angular, offset)
        values = [
            (predicted[..., :3] - measured_linear).norm(dim=-1).amax(),
            (predicted[..., 3:] - measured_angular).norm(dim=-1).amax(),
            (predicted_point - measured_point).norm(dim=-1).amax(),
            measured_point.norm(dim=-1).amax(),
            generalized.abs().amax(),
        ]
        names = ["body_com_linear_error_m_s", "body_angular_error_rad_s",
                 "support_point_velocity_error_m_s", "observed_support_speed_m_s",
                 "observed_generalized_velocity_abs"]
        if not self.robot.is_fixed_base:
            values.append((jacobians[:, 0, :, :6] - torch.eye(
                6, device=jacobians.device, dtype=jacobians.dtype)).abs().amax())
            names.append("floating_root_identity_block_max_error")
        stacked = torch.stack(values)
        if not torch.isfinite(stacked).all():
            raise RuntimeError("Nonfinite PhysX kinematics diagnostic")
        return dict(zip(names, stacked.tolist()))
