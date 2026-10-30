"""Versioned training objectives for supported starts and non-stair motion.

These are training terms, not a replacement for hardware handover validation.
In particular, randomized joint resets do not reproduce an identified actuator
or a settled manufacturer standing controller.
"""
import math

import torch


MOTION_KEYS = {'lateral_speed', 'lateral_travel_m', 'command_ramp_s',
               'abrupt_command_probability', 'arc_transition_probability'}
STARTUP_KEYS = {'joint_offset_rad', 'joint_velocity_rad_s', 'roll_pitch_rate_cost',
                'idle_wheel_speed_cost', 'idle_joint_speed_cost', 'idle_action_rate_cost',
                'joint_limit_cost', 'joint_limit_margin_fraction'}


def validate_stability(cfg):
    motion, startup = cfg['motion_stability'], cfg['startup_stability']
    if not isinstance(motion, dict) or set(motion) != MOTION_KEYS:
        raise ValueError('Unknown or missing CTS motion stability settings')
    if not isinstance(startup, dict) or set(startup) != STARTUP_KEYS:
        raise ValueError('Unknown or missing CTS startup stability settings')
    for key in ('lateral_speed', 'command_ramp_s'):
        interval = motion[key]
        if (not isinstance(interval, (list, tuple)) or len(interval) != 2
                or any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in interval)
                or interval[0] > interval[1]):
            raise ValueError('Invalid CTS motion stability interval: ' + key)
    for key in ('lateral_travel_m', 'abrupt_command_probability', 'arc_transition_probability'):
        x = motion[key]
        if type(x) not in (int, float) or not math.isfinite(x) or x < 0:
            raise ValueError('Invalid CTS motion stability value: ' + key)
    for key in ('abrupt_command_probability', 'arc_transition_probability'):
        if not 0 < motion[key] < 1:
            raise ValueError('CTS must retain both fixed/transition and abrupt/ramped commands')
    if not 0 < motion['lateral_travel_m'] < cfg['geometry']['tile_width']/2-cfg['rewards']['boundary_margin']:
        raise ValueError('CTS lateral travel must fit the physical tile')
    if motion['command_ramp_s'][1] * 8 >= cfg['commands']['move_s']:
        raise ValueError('CTS ramps must leave a plateau inside each arc segment')
    if motion['lateral_travel_m']/motion['lateral_speed'][1] < motion['command_ramp_s'][1]:
        raise ValueError('CTS lateral travel must allow the full command ramp')
    for key, x in startup.items():
        if type(x) not in (int, float) or not math.isfinite(x) or x < 0:
            raise ValueError('Invalid CTS startup stability value: ' + key)
    if not 0 < startup['joint_limit_margin_fraction'] < .5:
        raise ValueError('CTS joint limit margin must be inside each joint range')


def reset_supported_start_joints(env, env_ids):
    """Widen non-stair initial joint coverage, inside existing soft limits.

The v13 reset was multiplicative around its nominal pose (about 2%). A bounded
additive perturbation also exercises joints whose default angle is zero. Both
sides have identical distributions; no fixed default-pose target is rewarded.
The original reset still runs first, and staircase resets remain unchanged.
"""
    from .all_terrain_course import course
    state = course(env)
    cfg = state.cfg.get('startup_stability')
    if cfg is None or not len(env_ids):
        return
    ids = env_ids[state.course_ids[env_ids] >= 3]
    if not len(ids):
        return
    robot = env.scene['robot']
    joints = robot.find_joints(env.cfg.leg_joint_names, preserve_order=True)[0]
    q = robot.data.default_joint_pos[ids][:, joints].clone()
    q += torch.empty_like(q).uniform_(-cfg['joint_offset_rad'], cfg['joint_offset_rad'])
    limits = robot.data.soft_joint_pos_limits[ids][:, joints]
    if not torch.isfinite(limits).all() or (limits[..., 1] <= limits[..., 0]).any():
        raise ValueError('Supported-start reset requires finite ordered leg position limits')
    margin = (limits[..., 1]-limits[..., 0])*cfg['joint_limit_margin_fraction']
    q = torch.maximum(torch.minimum(q, limits[..., 1]-margin), limits[..., 0]+margin)
    dq = torch.empty_like(q).uniform_(-cfg['joint_velocity_rad_s'], cfg['joint_velocity_rad_s'])
    robot.write_joint_state_to_sim(q, dq, joint_ids=joints, env_ids=ids)


def stability_costs(env, state, geometry):
    """Measured-rate penalties; keep necessary stepping during lateral motion."""
    cfg = state.cfg.get('startup_stability')
    if cfg is None:
        return {}
    robot = env.scene['robot']
    data = robot.data
    legs = robot.find_joints(env.cfg.leg_joint_names, preserve_order=True)[0]
    wheels = robot.find_joints(env.cfg.wheel_joint_names, preserve_order=True)[0]
    nonstairs = (~state.stairs).to(data.joint_pos.dtype)
    idle = (geometry['command'].abs().amax(-1) <= 1.e-4).to(data.joint_pos.dtype)*nonstairs
    delta = env.action_manager.action-env.action_manager.prev_action
    q = data.joint_pos[:, legs]
    limits = data.soft_joint_pos_limits[:, legs]
    span = (limits[..., 1]-limits[..., 0]).clamp_min(1.e-6)
    margin = span*cfg['joint_limit_margin_fraction']
    near_limit = ((limits[..., 0]+margin-q).clamp_min(0)
                  +(q-limits[..., 1]+margin).clamp_min(0))/margin
    return {
        'roll_pitch_rate': cfg['roll_pitch_rate_cost']*data.root_ang_vel_b[:, :2].square().sum(-1)*nonstairs,
        'idle_wheel_speed': cfg['idle_wheel_speed_cost']*data.joint_vel[:, wheels].square().mean(-1)*idle,
        'idle_joint_speed': cfg['idle_joint_speed_cost']*data.joint_vel[:, legs].square().mean(-1)*idle,
        'idle_action_rate': cfg['idle_action_rate_cost']*delta.square().mean(-1)*idle,
        'joint_limit': cfg['joint_limit_cost']*near_limit.square().mean(-1)*nonstairs
            * (0. if getattr(env, 'cts_limit_monitor', None) is not None else 1.),
    }


def linear_tracking_reward(state, geometry):
    """Keep a useful tracking gradient for small lateral joystick commands.

The legacy fixed variance gives standing still almost full credit at 0.08 m/s.
Use the existing relative tracking tolerance and floor for lateral episodes;
old courses and the zero-command settling phase keep their original reward.
"""
    result = torch.exp(-geometry['error']/.25)
    if 'motion_stability' not in state.cfg:
        return result
    command = geometry['command'][:, :2].norm(dim=-1)
    cfg = state.cfg['commands']
    sigma = (command*cfg['relative_tracking_tolerance']).clamp_min(cfg['tracking_floor_m_s'])
    lateral = ((state.mode == 9) | (state.mode == 10)) & (command > 1.e-4)
    return torch.where(lateral, torch.exp(-geometry['error']/sigma.square()), result)


def stability_diagnostics(env):
    """Current-frame measured costs, separate from episode success statistics."""
    from .all_terrain_course import course
    state = course(env)
    costs = stability_costs(env, state, state.observe())
    output = {}
    for label, mask in (('teacher', state.teacher), ('student', ~state.teacher)):
        for name, values in costs.items():
            output[f'CTSStability/{label}/{name}_cost'] = float(values[mask].mean())
    return output
