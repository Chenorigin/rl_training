"""Isaac glue. Simulator-independent observation machinery stays in perception.py."""
from copy import deepcopy

import torch

from .perception import StudentObservationState, ActionDelayQueue, quaternion_rotation

PROPRIO_TERMS = ("base_ang_vel", "projected_gravity", "velocity_commands", "joint_pos", "joint_vel", "actions")


def clean_teacher_proprio(env, raw_command_override=None):
    """Evaluate the existing clean teacher terms, preserving clip/scale/order.

    Explicit entity resolution also works while ObservationManager is measuring
    the new packed term during its constructor (env.observation_manager not set).
    """
    if not hasattr(env, "_v1_proprio_terms"):
        terms = []
        for name in PROPRIO_TERMS:
            cfg = deepcopy(getattr(env.cfg.observations.teacher, name))
            if cfg.modifiers or cfg.history_length:
                raise ValueError("Unsupported teacher modifier/history in v1 proprio contract")
            for value in cfg.params.values():
                if hasattr(value, "resolve"):
                    value.resolve(env.scene)
            terms.append(cfg)
        env._v1_proprio_terms = terms
    values = []
    for name, cfg in zip(PROPRIO_TERMS, env._v1_proprio_terms):
        value = (raw_command_override.clone() if name == 'velocity_commands' and raw_command_override is not None
                 else cfg.func(env, **cfg.params).clone())
        if cfg.clip:
            value.clamp_(*cfg.clip)
        if cfg.scale is not None:
            value *= cfg.scale
        values.append(value)
    result = torch.cat(values, -1)
    if result.shape != (env.num_envs, 57):
        raise RuntimeError("Teacher proprio must be 57D in declared joint order")
    return result


def sensor_state(env):
    if not hasattr(env, "_m20_student_v1_state"):
        env._m20_student_v1_state = StudentObservationState(env.num_envs, env.device, env.step_dt,
                                                          env.cfg.student_perception, env.cfg.student_history_steps)
    return env._m20_student_v1_state


def action_delay(env):
    if not hasattr(env, "_m20_student_v1_delay"):
        env._m20_student_v1_delay = ActionDelayQueue(env.num_envs, env.device,
                                                    env.cfg.student_perception.action_delay_steps)
    return env._m20_student_v1_delay


def last_issued_action(env):
    return action_delay(env).previous_issued


def student_observation(env):
    robot = env.scene["robot"]
    rotation = quaternion_rotation(robot.data.root_quat_w)
    yaw = torch.atan2(rotation[:, 1, 0], rotation[:, 0, 0])
    return sensor_state(env).observe(clean_teacher_proprio(env),
                                     env.scene.sensors["height_scanner"].data.ray_hits_w,
                                     robot.data.root_pos_w, yaw, rotation, int(env.common_step_counter))


def reset_student(env, env_ids):
    sensor_state(env).reset(env_ids)
    action_delay(env).reset(env_ids)


def classify_state(hits, projected_gravity, command, clearance, wheel_velocity_z, contact, terrain_level):
    """Training-only tags; none are appended to the deployed actor tensor."""
    finite = torch.isfinite(hits[..., 2])
    z = torch.where(finite, hits[..., 2], 0.).reshape(-1, 11, 17)
    mask = finite.reshape_as(z)
    high = torch.where(mask, z, -torch.inf).flatten(1).max(-1).values
    low = torch.where(mask, z, torch.inf).flatten(1).min(-1).values
    rough = (high-low > .06) & finite.any(-1)
    # Height trend follows the commanded travel direction, including lateral/backward.
    x = torch.arange(-8, 9, device=hits.device)*.1
    y = torch.arange(-5, 6, device=hits.device)*.1
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    along = command[:, 0, None, None]*xx+command[:, 1, None, None]*yy
    ahead = (along > .03) & mask
    behind = (along < -.03) & mask
    delta = (z*ahead).sum((1, 2))/ahead.sum((1, 2)).clamp_min(1)-(z*behind).sum((1, 2))/behind.sum((1, 2)).clamp_min(1)
    terrain = torch.where(rough, torch.where(delta > .04, 1, torch.where(delta < -.04, 2, 3)), 0)
    approaching = ((clearance > 0) & (clearance < .06) & (wheel_velocity_z < -.05) & ~contact).any(-1)
    recovery = (projected_gravity[:, 2] > -.90) | (contact.sum(-1) <= 1)
    phase = torch.where(recovery, 3, torch.where(approaching, 1, 0))
    translating = command[:, :2].norm(dim=-1) > .1
    lateral = command[:, 1].abs() > command[:, 0].abs()
    direction = torch.where(translating, torch.where(lateral, 3, torch.where(command[:, 0] >= 0, 1, 2)),
                            torch.where(command[:, 2].abs() > .1, 4, 0))
    mixed = translating & ((command[:, 2].abs() > .1) | ((command[:, 0].abs() > .1) & (command[:, 1].abs() > .1)))
    direction = torch.where(mixed, 5, direction)
    return {"phase": phase.long(), "terrain": terrain.long(), "direction": direction.long(),
            "terrain_level": terrain_level.long().clone()}
