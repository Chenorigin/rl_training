"""MuJoCo adapter for the shared landing controller.

No terrain oracle is queried here. Teacher rays or the student's already
delivered map are frozen at the policy observation time. Contact feedback is
from the preceding completed integration, never the future contact solve.
MuJoCo's mj_jac uses a world point attached to a body; its translational
Jacobian includes root motion when computing support-point normal velocity.
See https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-jac
"""
from dataclasses import asdict
from collections import deque
import math

import numpy as np


MODES = ("off", "precontact", "phase", "combined")
ROLES = ("baseline-teacher", "baseline-student", "candidate-teacher",
         "candidate-student", "control-teacher", "control-student")


def validate_role(role, mode):
    if role not in ROLES or mode not in MODES:
        raise ValueError("Unknown rollout role or landing mode")
    if role.startswith("baseline-") and mode != "off":
        raise ValueError("Pinned baseline roles require landing control off")
    if role.startswith("control-") and mode == "off":
        raise ValueError("Control roles require an explicitly enabled controller")


def validate_checkpoint_control(saved, settings, actual_contract):
    """A trained controller contract cannot silently become an off rollout."""
    noise = saved.get("m20_training_manifest", {}).get("noise_refinement", {})
    expected = noise.get("contract", {}).get("environment", {}).get("landing_control")
    declared = noise.get("control_contract")
    if expected is not None and expected != asdict(settings):
        raise ValueError("Checkpoint landing settings differ from requested controller")
    if declared is not None and declared != actual_contract:
        raise ValueError("Checkpoint control contract differs from current controller")
    return expected is not None or declared is not None


def map_points(contract, data, adapter):
    """Reconstruct only the map available to the actor, including its holes."""
    if adapter is None:
        points = np.array(contract.last_scan_hits, dtype=np.float64, copy=True)
        return points, np.isfinite(points).all(axis=1)
    if adapter.last_channels is None:
        return np.empty((0, 3)), np.empty(0, dtype=bool)
    height, mask, confidence = adapter.last_channels
    height, mask, confidence = map(np.asarray, (height, mask, confidence))
    grid = np.asarray(contract.grid)
    if height.shape != (len(grid),) or mask.shape != height.shape or confidence.shape != height.shape:
        raise ValueError("Student map channels do not match the deployment grid")
    base = data.xpos[contract.base_id]
    rotation = data.xmat[contract.base_id].reshape(3, 3)
    yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    co, si = math.cos(yaw), math.sin(yaw)
    points = np.column_stack((base[0] + co * grid[:, 0] - si * grid[:, 1],
                              base[1] + si * grid[:, 0] + co * grid[:, 1],
                              base[2] - .5 - height))
    # Clipped height contains no recoverable metric distance.
    valid = ((mask > .5) & (confidence > 0) & (np.abs(height) < 1.)
             & np.isfinite(points).all(axis=1) & np.isfinite(confidence))
    return points, valid


def pd_effort(joint_position, joint_velocity, leg_target, wheel_velocity,
              kp_scale, kd_scale, constants):
    """Exactly the deployment PD expression when scales are one."""
    output = np.empty(16, dtype=np.float64)
    output[:12] = np.clip(
        80. * np.repeat(kp_scale, 3) * (leg_target.reshape(12) - joint_position[:12])
        - 2. * np.repeat(kd_scale, 3) * joint_velocity[:12],
        -constants.LEG_EFFORT_LIMIT, constants.LEG_EFFORT_LIMIT)
    output[12:] = np.clip(.6 * (wheel_velocity - joint_velocity[12:]),
                          -constants.WHEEL_EFFORT_LIMIT, constants.WHEEL_EFFORT_LIMIT)
    return output


def upward_support_force(normal, force, wheel_is_first, minimum_z=.8):
    """Force on the wheel, not |normal_z|; riser collisions are excluded."""
    upward = (-1. if wheel_is_first else 1.) * float(normal[2])
    return max(0., float(force)) * upward if upward >= minimum_z else 0.


class MujocoLanding:
    def __init__(self, model, contract, settings, constants):
        import mujoco
        from .landing import LandingController, control_contract, contract_hash
        self.model, self.contract, self.settings, self.constants = model, contract, settings, constants
        self.control = LandingController(1, "cpu", settings)
        self.points = np.empty((0, 3))
        self.valid = np.empty(0, dtype=bool)
        self.point_age_s = np.empty(0)
        self.map_time_s = None
        self.memory = None
        self.geoms = []
        for body in contract.wheel_body_ids:
            ids = [i for i in range(model.ngeom)
                   if model.geom_bodyid[i] == body and
                   (model.geom_contype[i] != 0 or model.geom_conaffinity[i] != 0)]
            if len(ids) != 1 or model.geom_type[ids[0]] != mujoco.mjtGeom.mjGEOM_CYLINDER:
                raise ValueError("Landing adapter requires one physical cylinder per wheel")
            self.geoms.append(ids[0])
        self.contract_document = control_contract(settings)
        self.contract_sha256 = contract_hash(self.contract_document)
        self.contact_history = deque([(np.zeros(4), np.zeros(4, dtype=bool))
                                      for _ in range(settings.contact_delay_steps)])
        # Seed set by benchmark, no global torch random stream consumption.
        self.contact_rng = np.random.default_rng(0)

    def observe(self, data, adapter):
        import torch
        from .terrain_memory import TerrainMemory
        self.points, self.valid = map_points(self.contract, data, adapter)
        self.point_age_s = (np.array(adapter.map.cell_age[0].numpy(), copy=True)
                            if adapter is not None else np.zeros(len(self.points)))
        self.map_time_s = float(data.time)
        count = len(self.points) if len(self.points) else len(self.contract.grid)
        if self.memory is None:
            self.memory = TerrainMemory(1, count, "cpu", self.settings)
        if not len(self.points):
            points, valid, age = np.zeros((count, 3)), np.zeros(count, bool), np.zeros(count)
        else:
            points, valid, age = self.points, self.valid, self.point_age_s
        rotation = data.xmat[self.contract.base_id].reshape(3, 3)
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
        self.memory.update(
            points_w=torch.as_tensor(points[None], dtype=torch.float32),
            valid=torch.as_tensor(valid[None], dtype=torch.bool),
            age_s=torch.as_tensor(age[None], dtype=torch.float32),
            pose_w=torch.as_tensor(data.xpos[self.contract.base_id][None], dtype=torch.float32),
            yaw_w=torch.tensor([yaw], dtype=torch.float32), time_s=float(data.time))

    def apply(self, data, action, contact_force, contact_valid, command_active, dt, feedback_age_s):
        import mujoco
        import torch
        c, const = self.contract, self.constants
        action = np.clip(action, -100., 100.)
        target = (const.DEFAULT_JOINT_POS[:12] + action[:12] * const.ACTION_SCALE[:12]).reshape(4, 3)
        wheel_target = action[12:] * const.ACTION_SCALE[12:]
        support = np.empty((4, 3))
        jacobian = np.empty((4, 3))
        point_jacobian = np.empty((4, 3, 3))
        velocity = np.empty(4)
        for i, (geom, body) in enumerate(zip(self.geoms, c.wheel_body_ids)):
            axis = data.geom_xmat[geom].reshape(3, 3)[:, 2]
            z_axis = np.array([0., 0., 1.])
            radial = z_axis - axis[2] * axis
            norm = np.linalg.norm(radial)
            radius, half_width = self.model.geom_size[geom, :2]
            offset = -radius * radial / max(norm, 1e-12) - half_width * np.sign(axis[2]) * axis
            support[i] = data.geom_xpos[geom] + offset
            jac = np.empty((3, self.model.nv))
            mujoco.mj_jac(self.model, data, jac, None, support[i], int(body))
            jacobian[i] = jac[2, c.qvel_ids[3*i:3*i+3]]
            point_jacobian[i] = jac[:, c.qvel_ids[3*i:3*i+3]]
            velocity[i] = jac[2] @ data.qvel
        tensor = lambda a: torch.as_tensor(np.asarray(a)[None], dtype=torch.float32)
        boolean = lambda a: torch.as_tensor(np.asarray(a)[None], dtype=torch.bool)
        if self.memory is not None:
            height, valid, capture_age = self.memory.sample(
                tensor(support[:, :2]), time_s=float(data.time))
            clearance = tensor(support[:, 2]) - height
        else:
            valid = torch.zeros((1, 4), dtype=torch.bool)
            clearance = torch.zeros((1, 4))
            capture_age = torch.full((1, 4), torch.nan)
        self.contact_history.append((np.array(contact_force), np.array(contact_valid)))
        contact_force, contact_valid = self.contact_history.popleft()
        contact_valid &= self.contact_rng.random(4) >= self.settings.contact_dropout_probability
        contact_valid &= feedback_age_s + self.settings.contact_delay_steps * dt <= self.settings.contact_max_age_s
        with torch.inference_mode():
            result = self.control.step(
                nominal_leg_target=tensor(target), nominal_wheel_velocity=tensor(wheel_target),
                joint_position=tensor(data.qpos[c.qpos_ids[:12]].reshape(4, 3)),
                joint_velocity=tensor(data.qvel[c.qvel_ids[:12]].reshape(4, 3)),
                normal_jacobian=tensor(jacobian), normal_velocity=tensor(velocity),
                point_jacobian=tensor(point_jacobian),
                surface_normal=tensor(np.tile([0., 0., 1.], (4, 1))),
                clearance=clearance, terrain_valid=valid,
                contact_force=tensor(contact_force), contact_valid=boolean(contact_valid),
                command_active=torch.tensor([command_active], dtype=torch.bool), dt=dt)
        result = {k: v[0].detach().numpy().copy() for k, v in result.items()}
        # Preserve original double-precision PD targets exactly on a bypass;
        # a float32 round trip must not itself count as a controller action.
        result["leg_target"] = np.where(result["modified"][:, None], result["leg_target"], target)
        result["wheel_velocity"] = wheel_target
        effort = pd_effort(data.qpos[c.qpos_ids], data.qvel[c.qvel_ids],
                           result["leg_target"], result["wheel_velocity"],
                           result["kp_scale"], result["kd_scale"], const)
        diagnostics = {
            "landing_" + k: result[k] for k in
            ("leg_target", "wheel_velocity", "kp_scale", "kd_scale", "phase", "modified",
             "invalid", "infeasible", "safe_speed", "correction")
        }
        diagnostics.update(landing_clearance_m=clearance[0].numpy(), landing_terrain_valid=valid[0].numpy(),
                           landing_contact_force_N=np.array(contact_force),
                           landing_contact_valid=np.array(contact_valid),
                           landing_normal_velocity_m_s=velocity,
                           landing_map_cache_age_s=float(data.time) - self.map_time_s,
                           landing_capture_age_s=capture_age[0].numpy())
        return effort, diagnostics
