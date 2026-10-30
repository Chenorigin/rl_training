"""Simulator-independent geometry and gait scores (wheel order FL, FR, HL, HR)."""
from dataclasses import dataclass
import math

import torch


@dataclass
class TrotSettings:
    min_rise: float = 0.06
    full_rise: float = 0.08
    lateral_half_width: float = 0.30
    rear_edge: float = -0.60
    front_edge: float = 0.65
    edge_fade: float = 0.15
    forward_command_min: float = 0.10
    lateral_command_max: float = 0.20
    yaw_command_max: float = 0.50
    gate_rise_time: float = 0.20
    frequency_min: float = 0.80
    frequency_max: float = 1.60
    nominal_stride: float = 0.40
    swing_fraction: float = 0.40
    clearance_target: float = 0.08
    clearance_min: float = 0.025
    ground_distance_max: float = 0.16
    contact_force: float = 15.0
    air_time_min: float = 0.04
    lift_event_window: float = 0.08
    touchdown_event_window: float = 0.10
    # Action units; leg scale is 0.25 rad, wheel scale comes from inherited cfg.
    hipx_residual_limit: float = 0.25
    hipy_residual_limit: float = 1.20
    knee_residual_limit: float = 1.60
    wheel_residual_limit: float = 0.15
    # Actual default USD collision: capsule radius .09, axial length .054 m.
    capsule_half_length: float = 0.027
    slip_speed_scale: float = 0.30
    edge_margin: float = 0.06
    edge_height_threshold: float = 0.045
    route_lateral_scale: float = 0.20
    route_heading_scale: float = 0.25
    power_scale_w: float = 1800.0


def yaw_from_quaternion(q):
    """World heading, Isaac scalar-first quaternion convention."""
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def quat_rotate(q, v):
    xyz = q[..., 1:]
    cross = 2 * torch.linalg.cross(xyz, v, dim=-1)
    return v + q[..., :1] * cross + torch.linalg.cross(xyz, cross, dim=-1)


def new_route_progress(position_xy, origin_xy, yaw, frontier):
    """World-fixed projection; revisiting an earlier position earns no progress."""
    relative = position_xy - origin_xy
    forward = relative[:, 0] * yaw.cos() + relative[:, 1] * yaw.sin()
    lateral = -relative[:, 0] * yaw.sin() + relative[:, 1] * yaw.cos()
    return (forward - frontier).clamp_min(0), torch.maximum(frontier, forward), lateral


def ascent_geometry(hit_z, ray_xy, command, cfg):
    """Reject flat/downhill/invalid scans; include the final riser under rear legs.

    A column needs at least two finite hits and half its central rays. This is
    local geometry, never a terrain-type label or a whole-map height range.
    """
    x_levels, inverse = torch.unique(ray_xy[:, 0], sorted=True, return_inverse=True)
    if len(x_levels) < 2:
        raise ValueError("Trot requires at least two forward scan columns")
    heights, valid = [], []
    for i in range(len(x_levels)):
        selected = (inverse == i) & (ray_xy[:, 1].abs() <= cfg.lateral_half_width)
        values = hit_z[:, selected]
        if values.shape[1] < 2:
            raise ValueError("Trot requires at least two central rays per column")
        finite = torch.isfinite(values) & (values.abs() < 1.e6)
        heights.append(torch.nanmedian(torch.where(finite, values, float("nan")), dim=-1).values)
        valid.append(finite.sum(-1) >= max(2, math.ceil(values.shape[1] / 2)))
    profile = torch.nan_to_num(torch.stack(heights, -1))
    valid = torch.stack(valid, -1)
    delta = profile[:, 1:] - profile[:, :-1]
    edge_x = (x_levels[1:] + x_levels[:-1]) * 0.5
    distance_gain = torch.minimum(
        (edge_x - cfg.rear_edge) / cfg.edge_fade,
        (cfg.front_edge - edge_x) / cfg.edge_fade,
    ).clamp(0, 1)
    rise_gain = ((delta - cfg.min_rise) / (cfg.full_rise - cfg.min_rise)).clamp(0, 1)
    # Missing columns anywhere under the robot must not look like known stairs.
    local = (x_levels >= cfg.rear_edge) & (x_levels <= cfg.front_edge)
    usable = valid[:, local].all(-1)
    gate = (rise_gain * distance_gain * (valid[:, 1:] & valid[:, :-1])).amax(-1)
    return gate * usable if command is None else gate * usable * command_gate(command, cfg)


def command_gate(command, cfg):
    return ((command[:, 0] > cfg.forward_command_min)
            & (command[:, 1].abs() <= cfg.lateral_command_max)
            & (command[:, 2].abs() <= cfg.yaw_command_max))


def phase_targets(phase, cfg):
    """Diagonal pairs are half a cycle apart, with two double-support windows."""
    offsets = phase.new_tensor([0.0, 0.5, 0.5, 0.0])
    leg_phase = (phase[:, None] + offsets) % 1.0
    swing = leg_phase < cfg.swing_fraction
    lift = torch.sin(math.pi * (leg_phase / cfg.swing_fraction).clamp(0, 1))
    return swing, torch.where(swing, lift * cfg.clearance_target, 0.0)


def gait_scores(contact, air, clearance, ground_valid, phase, progress, cfg):
    """Scores cannot confuse four-wheel contact or flight with a diagonal trot."""
    swing, target = phase_targets(phase, cfg)
    stance = ~swing
    clearance_score = torch.exp(-((clearance - target) / cfg.clearance_target).square())
    # The lift trajectory has zero endpoints; contact matching supplies timing.
    phase_match = (torch.where(swing, air, contact).float() * clearance_score).mean(-1)
    a = air[:, 0] & air[:, 3] & contact[:, 1] & contact[:, 2]
    b = air[:, 1] & air[:, 2] & contact[:, 0] & contact[:, 3]
    diagonal = a | b
    mismatch = ((air[:, 0] != air[:, 3]).float() + (air[:, 1] != air[:, 2]).float()) * 0.5
    support_deficit = (2 - contact.sum(-1)).clamp_min(0).float() / 2
    stance_missing = (stance & ~contact).float().mean(-1)
    valid = ground_valid.all(-1)
    return {
        "phase": phase_match * progress * valid,
        "diagonal": diagonal.float() * progress * valid,
        "pair_mismatch": mismatch,
        "support_deficit": support_deficit,
        "stance_missing": stance_missing,
        "clearance_error": ((clearance - target).abs() * swing).mean(-1),
        "diagonal_observed": diagonal.float() * valid,
    }


def paired_events(lift_age, touch_age, new_lift, new_touch, contact, air,
                  ground_valid, phase, armed, cfg):
    """Consume each paired event once; a landing needs a prior supported swing."""
    lift_age, touch_age, armed = lift_age.clone(), touch_age.clone(), armed.clone()
    score = torch.zeros_like(phase)
    swing, _ = phase_targets(phase, cfg)
    valid = ground_valid.all(-1)
    for index, (i, j, k, l) in enumerate(((0, 3, 1, 2), (1, 2, 0, 3))):
        support = contact[:, k] & contact[:, l] & valid
        pair_phase = (phase + 0.5 * index) % 1.0
        expired = armed[:, index] & (~support | (pair_phase >= 0.5))
        armed[:, index] &= ~expired
        for leg in (i, j):
            lift_age[:, leg] = torch.where(expired, float("inf"), lift_age[:, leg])
            touch_age[:, leg] = torch.where(expired, float("inf"), touch_age[:, leg])
        lift = (support & air[:, i] & air[:, j] & swing[:, i]
                & ~armed[:, index]
                & (lift_age[:, [i, j]].amax(-1) <= cfg.lift_event_window)
                & (new_lift[:, i] | new_lift[:, j]))
        armed[:, index] |= lift
        touch = (support & contact[:, i] & contact[:, j] & ~swing[:, i]
                 & armed[:, index]
                 & (touch_age[:, [i, j]].amax(-1) <= cfg.touchdown_event_window)
                 & (new_touch[:, i] | new_touch[:, j]))
        armed[:, index] &= ~touch
        for leg in (i, j):
            lift_age[:, leg] = torch.where(lift | touch, float("inf"), lift_age[:, leg])
            touch_age[:, leg] = torch.where(lift | touch, float("inf"), touch_age[:, leg])
        score += lift.float() + touch.float()
    return score, lift_age, touch_age, armed
