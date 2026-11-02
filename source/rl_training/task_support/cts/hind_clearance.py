"""Training-only ascent shaping in world geometry; no action or sensor changes.

FL/FR/HL/HR wheel order. Targets come from the collision atlas, never the
student's noisy/expired map. No gait clock, desired knee angle or torque bonus.
"""
import math

import torch


def validate_hind_clearance(cfg):
    keys = {'enabled', 'margin_m', 'lookahead_m', 'min_rise_m',
            'support_tolerance_m', 'command_min_m_s', 'force_scale_N',
            'clearance_weight', 'riser_weight', 'support_weight'}
    if not isinstance(cfg, dict) or set(cfg) != keys or type(cfg['enabled']) is not bool:
        raise ValueError('Invalid CTS hind clearance settings')
    for key in keys - {'enabled'}:
        value = cfg[key]
        if (type(value) not in (int, float) or not math.isfinite(value)
                or value < 0 or (not key.endswith('_weight') and value == 0)):
            raise ValueError('Invalid CTS hind clearance numeric setting: '+key)
    if not cfg['support_tolerance_m'] < cfg['min_rise_m']:
        raise ValueError('Hind clearance support tolerance must distinguish adjacent treads')
    if cfg['margin_m'] >= cfg['lookahead_m']:
        raise ValueError('Hind clearance approach must allow a gradual lift')
    return cfg


def rising_edges(rectangles, min_rise):
    """Rectangles [tile, surface, xmin/xmax/ymin/ymax/z], local z (floor=0).

    Match each left edge to the top immediately before it. The atlas stair
    rectangles are disjoint in x and share a corridor; equal-height landing
    seams and all falling edges are excluded. Padding has nonfinite height.
    This runs once per atlas, not at every reward call.
    """
    r = rectangles
    valid = torch.isfinite(r).all(-1) & (r[..., 1] > r[..., 0])
    adjacent = torch.isclose(r[:, :, None, 0], r[:, None, :, 1], atol=1.e-5, rtol=0)
    adjacent &= valid[:, None, :] & (r[:, :, None, 2] < r[:, None, :, 3])
    adjacent &= r[:, :, None, 3] > r[:, None, :, 2]
    before = torch.where(adjacent, r[:, None, :, 4], -torch.inf).amax(-1)
    before = torch.where(torch.isfinite(before), before, 0.)
    rise = r[..., 4]-before
    return torch.stack((r[..., 0], r[..., 2], r[..., 3], before, r[..., 4]), -1), valid & (rise >= min_rise)


def clearance_signals(position, axis, force, supported, support_height, edges,
                      valid_edges, moving, radius, half_length, cfg):
    """Pure batched instrument, also usable on measured MuJoCo trajectories.

    Full capsule envelope clears the edge before it advances. A smooth ramp
    starts before the leading wheel surface meets the riser; grounded wheels
    are deliberately included. Supported upper-tread wheels can settle without
    maintaining the swing margin. Negative force-X is only a riser-force proxy.
    """
    bottom = position[..., 2]-radius-half_length*axis[..., 2].abs()
    reach = radius+half_length*axis[..., 0].abs()
    lateral = radius+half_length*axis[..., 1].abs()
    distance = edges[:, None, :, 0]-position[..., 0, None]
    valid = valid_edges[:, None, :] & moving[:, None, None]
    valid = valid & (distance >= -reach[..., None])
    valid &= distance <= reach[..., None]+cfg['lookahead_m']
    valid &= position[..., 1, None]-lateral[..., None] >= edges[:, None, :, 1]
    valid &= position[..., 1, None]+lateral[..., None] <= edges[:, None, :, 2]
    # A settled current tread must not hide the NEXT riser on short treads.
    # Filter landed candidates BEFORE selecting the first unresolved edge.
    landed = (supported[..., None]
        & ((support_height[..., None]-edges[:, None, :, 4]).abs() <= cfg['support_tolerance_m'])
        & ((bottom[..., None]-edges[:, None, :, 4]).abs() <= cfg['support_tolerance_m']))
    valid &= ~landed
    # First still-overlapping ascending edge. Never switch to the next riser
    # merely because the centre crossed this one while its back still overlaps.
    index = torch.where(valid, distance, torch.inf).argmin(-1, keepdim=True)
    active = valid.any(-1)
    selected = edges[:, None].expand(-1, position.shape[1], -1, -1).gather(
        2, index[..., None].expand(-1, -1, 1, 5)).squeeze(2)
    selected = torch.where(active[..., None], selected, 0.)
    distance = selected[..., 0]-position[..., 0]
    phase = (1-(distance-reach)/cfg['lookahead_m']).clamp(0., 1.)
    phase = phase.square()*(3-2*phase)
    rise = (selected[..., 4]-selected[..., 3]).clamp_min(0.)
    target = selected[..., 3]+phase*(rise+cfg['margin_m'])
    deficit = torch.where(active, (target-bottom).clamp_min(0.), 0.)
    # Bounded costs cannot grow arbitrarily when stuck. Above the target,
    # additional height earns nothing; existing effort/impact/limits still act.
    clearance = (deficit/(rise+cfg['margin_m'])).clamp(0., 1.).square()
    near_face = active & (distance.abs() <= reach) & (bottom < selected[..., 4])
    raw_finite = torch.isfinite(torch.cat((position, axis, force), -1)).all(-1) & torch.isfinite(support_height)
    force_cost = (-torch.nan_to_num(force[..., 0])/cfg['force_scale_N']).clamp(0., 1.).square()
    riser = torch.where(near_face, force_cost, 0.)
    invalid = moving[:, None] & ~raw_finite
    clearance = torch.where(invalid, 1., torch.nan_to_num(clearance, nan=1.).clamp(0.,1.))
    riser = torch.where(invalid, 1., riser)
    rear_task = active[:, 2:].any(-1)
    support = rear_task & ~supported[:, 2:].any(-1)
    return dict(active=active, bottom=bottom, target=target, deficit=deficit,
                edge_distance=distance, top=selected[..., 4], phase=phase,
                clearance=clearance, riser=riser, support=support.float())


def signals(env):
    from .all_terrain_course import course
    state = course(env)
    cfg = state.cfg['hind_clearance']
    if not hasattr(state, '_hind_edges'):
        state._hind_edges = rising_edges(state.surface.rectangles, cfg['min_rise_m'])
    g = state.observe()
    edges, valid = state._hind_edges
    origin = state.origins[state.tile].clone()
    origin[:,2] = 0.  # Atlas rectangles use absolute physical Z; spawn Z is not a terrain offset.
    position = g['position']-origin[:, None]
    height = g['height']-origin[:, 2, None]
    moving = state.stairs & (g['command'][:, 0] > cfg['command_min_m_s'])
    tail = env.cfg.stair_tail
    return clearance_signals(position, g['axis'], g['force'], g['supported'], height,
        edges[state.tile], valid[state.tile], moving, tail.wheel_radius, tail.wheel_half_length, cfg)


def hind_step_clearance(env):
    return torch.nan_to_num(-signals(env)['clearance'][:, 2:].mean(-1), nan=-1., posinf=-1., neginf=-1.)


def hind_riser_contact(env):
    return torch.nan_to_num(-signals(env)['riser'][:, 2:].mean(-1), nan=-1., posinf=-1., neginf=-1.)


def hind_support_transfer(env):
    return -signals(env)['support']
