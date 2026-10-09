# Copyright (c) 2026 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause

"""Training-only, adjacent-riser alternation ledger for the M20 stair teacher."""

from __future__ import annotations

import torch

from isaaclab.utils.math import quat_apply, yaw_quat


def _new_state(env):
    n, device = env.num_envs, env.device
    zeros = lambda *shape: torch.zeros((n, *shape), device=device)
    flags = lambda *shape: torch.zeros((n, *shape), device=device, dtype=torch.bool)
    return {
        "active": flags(),
        "edge_xy": zeros(2),
        "heading": zeros(2),
        "source_z": zeros(),
        "target_z": zeros(),
        "age": zeros(),
        "radius": torch.full((n, 4), float("nan"), device=device),
        "front_prev_x": zeros(2),
        "front_prev_valid": flags(2),
        "front_safe": flags(2),
        "front_failed": flags(2),
        "front_face": flags(2),
        "front_face_event": flags(2),
        "front_lift": zeros(2),
        "front_lift_max": zeros(),
        "front_lift_gain": zeros(),
        "front_lift_lead": torch.full((n,), -1, device=device, dtype=torch.long),
        "front_lift_credit": zeros(),
        "lift_history_xy": zeros(64, 2),
        "lift_history_heading": zeros(64, 2),
        "lift_history_z": zeros(64),
        "lift_history_max": zeros(64),
        "lift_history_count": torch.zeros(n, device=device, dtype=torch.long),
        "lift_history_index": torch.zeros(n, device=device, dtype=torch.long),
        "targets_started": zeros(),
        "retry_count": zeros(),
        "forward_steps": zeros(),
        "retreat_steps": zeros(),
        "stationary_steps": zeros(),
        "abandoned_targets": zeros(),
        "front_physical_time": zeros(2),
        "front_event": flags(2),
        "front_lift_settlement": zeros(2),
        "front_violation_event": flags(),
        "front_expected": torch.full((n,), -1, device=device, dtype=torch.long),
        "front_count": torch.zeros(n, device=device, dtype=torch.long),
        "front_successes": zeros(),
        "front_violations": zeros(),
        "front_history_xy": zeros(64, 2),
        "front_history_heading": zeros(64, 2),
        "front_history_source_z": zeros(64),
        "front_history_target_z": zeros(64),
        "retired_valid": flags(),
        "retired_xy": zeros(2),
        "retired_z": zeros(),
        "rear_target_index": torch.full((n,), -1, device=device, dtype=torch.long),
        "rear_target_age": zeros(),
        "rear_count": torch.zeros(n, device=device, dtype=torch.long),
        "rear_expected": torch.full((n,), -1, device=device, dtype=torch.long),
        "rear_prev_x": zeros(2),
        "rear_prev_valid": flags(2),
        "rear_safe": flags(2),
        "rear_failed": flags(2),
        "rear_physical_time": zeros(2),
        "rear_event": flags(2),
        "rear_violation_event": flags(),
        "rear_successes": zeros(),
        "rear_violations": zeros(),
        "front_same_tread_paid": torch.full((n,), -1, device=device, dtype=torch.long),
        "rear_same_tread_paid": torch.full((n,), -1, device=device, dtype=torch.long),
        "front_same_tread_event": flags(),
        "rear_same_tread_event": flags(),
    }


def _reset(state, reset):
    if not reset.any():
        return
    for key, value in state.items():
        if key in {"radius"}:
            value[reset] = float("nan")
        elif key in {"front_expected", "rear_expected", "rear_target_index", "front_lift_lead",
                     "front_same_tread_paid", "rear_same_tread_paid"}:
            value[reset] = -1
        else:
            value[reset] = 0


def _contact_on_tread(wheel_z, target_z, radius, force_z, force_norm, x, edge_x, inset):
    return (
        torch.isfinite(radius)
        & ((wheel_z - target_z - radius).abs() <= 0.05)
        & (x > edge_x + inset)
        & (force_z > 0.5 * force_norm)
        & (force_z > 5.0)
    )


def ascent_state(env, asset_cfg, sensor_cfg, contact_sensor_cfg, command_name="base_velocity"):
    """Advance independent front and rear one-wheel-per-riser sequences once per step."""
    step = int(env.common_step_counter)
    cache = getattr(env, "_m20_ascent_cache", None)
    if cache is not None and cache["step"] == step:
        return cache["state"]

    from .stair_teacher import _stair_scan_context

    context = _stair_scan_context(env, sensor_cfg, command_name)
    asset = env.scene[asset_cfg.name]
    scanner = env.scene[sensor_cfg.name]
    contact_sensor = env.scene.sensors[contact_sensor_cfg.name]
    state = getattr(env, "_m20_ascent_state", None)
    if state is None or state["active"].shape[0] != env.num_envs:
        state = _new_state(env)
        env._m20_ascent_state = state
    if hasattr(env, "episode_length_buf"):
        _reset(state, env.episode_length_buf <= 1)
    for key in ("front_event", "rear_event", "front_face_event", "front_violation_event",
                "rear_violation_event", "front_same_tread_event", "rear_same_tread_event"):
        state[key].zero_()
    state["front_lift_settlement"].zero_()
    state["front_lift_gain"].zero_()

    wheel = asset.data.body_pos_w[:, asset_cfg.body_ids]
    velocity = asset.data.body_lin_vel_w[:, asset_cfg.body_ids]
    force = contact_sensor.data.net_forces_w[:, contact_sensor_cfg.body_ids]
    force_z = force[..., 2]
    force_norm = torch.linalg.vector_norm(force, dim=-1)
    contact = (contact_sensor.data.current_contact_time[:, contact_sensor_cfg.body_ids] > 0) & (
        force_z > 0.5 * force_norm
    )
    # Calibrate on the actual surface under each wheel, including before a
    # stair target starts. The target's source tread need not be the tread
    # under a trailing wheel in a one-leg-per-riser gait.
    hits = scanner.data.ray_hits_w
    valid_hits = torch.isfinite(hits).all(dim=-1)
    distance = torch.linalg.vector_norm(wheel[:, :, None, :2] - hits[:, None, :, :2], dim=-1)
    distance = torch.where(valid_hits[:, None], distance, torch.inf)
    nearest_distance, nearest = distance.min(dim=-1)
    surface_z = hits[..., 2].gather(1, nearest)
    measured_radius = wheel[..., 2] - surface_z
    local = (distance < 0.16) & valid_hits[:, None]
    local_max = torch.where(local, hits[:, None, :, 2], -torch.inf).amax(dim=-1)
    local_min = torch.where(local, hits[:, None, :, 2], torch.inf).amin(dim=-1)
    flat_support = (local.sum(dim=-1) >= 2) & ((local_max - local_min) <= 0.025)
    calibrate = (~torch.isfinite(state["radius"]) & contact & (force_z > 5.0)
                 & (contact_sensor.data.current_contact_time[:, contact_sensor_cfg.body_ids] >= 0.06)
                 & flat_support & (nearest_distance < 0.08)
                 & (measured_radius >= 0.02) & (measured_radius <= 0.25))
    state["radius"][calibrate] = measured_radius[calibrate]
    forward = torch.zeros((env.num_envs, 3), device=env.device)
    forward[:, 0] = 1.0
    heading = quat_apply(yaw_quat(scanner.data.quat_w), forward)[:, :2]
    candidate_xy = scanner.data.pos_w[:, :2] + heading * context["edge_x"][:, None]

    # The first riser permits either lead. Afterwards, only the next physical
    # riser on the same flight is eligible; skipped or repeated edges earn zero.
    count = state["front_count"]
    last_i = (count - 1).clamp_min(0)
    row = torch.arange(env.num_envs, device=env.device)
    last_xy = state["front_history_xy"][row, last_i]
    last_heading = state["front_history_heading"][row, last_i]
    delta_xy = candidate_xy - last_xy
    along = (delta_xy * last_heading).sum(dim=1)
    lateral = torch.abs(delta_xy[:, 0] * last_heading[:, 1] - delta_xy[:, 1] * last_heading[:, 0])
    height_step = context["upper_z"] - state["front_history_target_z"][row, last_i]
    adjacent = (count == 0) | (
        (along >= 0.15) & (along <= 0.50) & (lateral <= 0.20)
        & (height_step >= 0.06) & (height_step <= 0.30)
    )
    retired_distance = torch.linalg.vector_norm(candidate_xy - state["retired_xy"], dim=1)
    repeated = state["retired_valid"] & (retired_distance < 0.12) & (
        (context["upper_z"] - state["retired_z"]).abs() < 0.05
    )
    start = (
        ~state["active"] & (context["up_gate"] > 0)
        & adjacent & ~repeated & (count < 64)
    )
    # Credit survives visiting another target and returning. Remember the
    # physical riser plane, not just the last abandoned edge, so A-B-A cannot
    # reset the lift budget. Quantized scans can shift the estimated edge 5 cm.
    history_delta = candidate_xy[:, None] - state["lift_history_xy"]
    history_along = (history_delta * state["lift_history_heading"]).sum(dim=-1).abs()
    history_aligned = (heading[:, None] * state["lift_history_heading"]).sum(dim=-1) > 0.8
    history_valid = torch.arange(64, device=env.device)[None] < state["lift_history_count"][:, None]
    same_point = torch.linalg.vector_norm(history_delta, dim=-1) < 0.15
    matches = (history_valid & (same_point | (history_aligned & (history_along < 0.15)))
               & ((context["upper_z"][:, None] - state["lift_history_z"]).abs() < 0.05))
    known = matches.any(dim=1)
    start &= known | (state["lift_history_count"] < 64)
    target_slot = torch.where(known, matches.long().argmax(dim=1), state["lift_history_count"]).clamp_max(63)
    new_ids = torch.nonzero(start & ~known, as_tuple=False).squeeze(-1)
    new_slots = target_slot[new_ids]
    state["lift_history_xy"][new_ids, new_slots] = candidate_xy[new_ids]
    state["lift_history_heading"][new_ids, new_slots] = heading[new_ids]
    state["lift_history_z"][new_ids, new_slots] = context["upper_z"][new_ids]
    state["lift_history_count"][new_ids] += 1
    state["lift_history_index"][start] = target_slot[start]
    state["active"][start] = True
    state["edge_xy"][start] = candidate_xy[start]
    state["heading"][start] = heading[start]
    state["source_z"][start] = context["lower_z"][start]
    state["target_z"][start] = context["upper_z"][start]
    state["age"][start] = 0.0
    state["targets_started"][start] += 1
    state["front_lift_max"][start] = state["lift_history_max"][row[start], target_slot[start]]
    state["front_lift_lead"][start] = -1
    for key in ("front_prev_valid", "front_safe", "front_failed", "front_face",
                "front_lift", "front_physical_time"):
        state[key][start] = 0

    front_x = ((wheel[:, :2, :2] - state["edge_xy"][:, None]) * state["heading"][:, None]).sum(dim=-1)
    front_radius = state["radius"][:, :2]
    # A failed crossing can be retried after returning behind the edge. A
    # previous face contact remains an event cost, not a permanent veto.
    retry = state["active"][:, None] & (front_x < -0.06) & state["front_failed"]
    state["retry_count"] += retry.any(dim=1).float()
    state["front_failed"][retry] = False
    state["front_safe"][front_x < -0.06] = False

    face_force = torch.linalg.vector_norm(force[:, :2, :2], dim=-1)
    face_now = (
        state["active"][:, None] & (front_x >= -0.12) & (front_x <= 0.03)
        & torch.isfinite(front_radius)
        & (wheel[:, :2, 2] < state["target_z"][:, None] + front_radius + 0.05)
        & (face_force > 50.0) & (face_force > 1.5 * force_z[:, :2].clamp_min(0))
    )
    state["front_face_event"] = face_now & ~state["front_face"]
    state["front_face"] |= face_now

    lift = (
        state["active"][:, None] & (front_x < 0) & ~contact[:, :2]
        & torch.isfinite(front_radius) & (velocity[:, :2, 2] > 0.10)
    )
    state["front_lift"] = torch.maximum(
        state["front_lift"],
        torch.where(lift, (velocity[:, :2, 2] / 0.5).clamp_max(1.0),
                    torch.zeros_like(state["front_lift"])),
    )
    # Bounded feedback during preparation, instead of withholding all lift
    # credit until touchdown. Only the expected leg (either first lead) may
    # earn it, with two other wheels supporting and no repeated-height credit.
    rise = (state["target_z"] - state["source_z"] + 0.02).clamp_min(0.06)
    height_fraction = ((wheel[:, :2, 2] - state["source_z"][:, None] - front_radius)
                       / rise[:, None]).clamp(0, 1)
    support_others = contact.sum(dim=1, keepdim=True) - contact[:, :2].long()
    lift_eligible = (state["active"][:, None] & torch.isfinite(front_radius)
                     & (front_x >= -0.45) & (front_x <= 0.02) & ~contact[:, :2]
                     & (support_others >= 2) & (context["command_x"][:, None] > 0.1))
    fractions = torch.where(lift_eligible, height_fraction, torch.zeros_like(height_fraction))
    lead = torch.where(state["front_expected"] >= 0, state["front_expected"], state["front_lift_lead"])
    lead = torch.where(lead >= 0, lead, fractions.argmax(dim=1))
    selected_fraction = fractions.gather(1, lead[:, None]).squeeze(1)
    gain = (selected_fraction - state["front_lift_max"]).clamp_min(0)
    # Even ambiguous edge identities under turns cannot turn exploration into
    # an infinite reward source: each confirmed landing unlocks at most one
    # further preparation credit for the episode.
    remaining = (1.0 + state["front_successes"] - state["front_lift_credit"]).clamp_min(0)
    gain = torch.minimum(gain, remaining)
    state["front_lift_lead"] = torch.where(gain > 0, lead, state["front_lift_lead"])
    state["front_lift_max"] = torch.maximum(state["front_lift_max"], selected_fraction)
    active_ids = torch.nonzero(state["active"], as_tuple=False).squeeze(-1)
    state["lift_history_max"][active_ids, state["lift_history_index"][active_ids]] = state["front_lift_max"][active_ids]
    state["front_lift_gain"] = gain
    state["front_lift_credit"] += gain
    crossing = (
        state["active"][:, None] & state["front_prev_valid"]
        & (state["front_prev_x"] < 0) & (front_x >= 0)
    )
    clear = (
        torch.isfinite(front_radius)
        & (wheel[:, :2, 2] - state["target_z"][:, None] - front_radius >= 0.02)
    )
    state["front_failed"] |= crossing & ~clear
    state["front_safe"] |= crossing & clear & ~state["front_failed"]
    state["front_lift"][crossing & ~clear] = 0.0
    state["front_prev_x"] = front_x.clone()
    state["front_prev_valid"] = state["active"][:, None].expand_as(state["front_prev_valid"]).clone()

    front_physical = (
        state["active"][:, None]
        & _contact_on_tread(
            wheel[:, :2, 2], state["target_z"][:, None], front_radius,
            force_z[:, :2], force_norm[:, :2], front_x, torch.zeros_like(front_x), 0.06,
        )
    )
    state["front_physical_time"] = torch.where(
        front_physical, state["front_physical_time"] + env.step_dt,
        torch.zeros_like(state["front_physical_time"]),
    )
    candidates = state["front_physical_time"] >= 0.04
    one = candidates.sum(dim=1) == 1
    side = candidates.long().argmax(dim=1)
    expected = state["front_expected"]
    ordered = (expected < 0) | (side == expected)
    safe_side = state["front_safe"].gather(1, side[:, None]).squeeze(1)
    valid = state["active"] & one & ordered & safe_side
    physical = state["active"] & candidates.any(dim=1)
    violation = physical & ~valid
    state["front_event"] = candidates & valid[:, None]
    state["front_violation_event"] = violation
    state["front_lift_settlement"] = state["front_event"].float() * state["front_lift"]
    state["front_successes"] += valid.float()
    state["front_violations"] += violation.float()

    # Even a wrong landing advances the physical target, so one mistake does
    # not close all later local learning. It earns no completion credit.
    retired = valid | violation
    ids = torch.nonzero(retired, as_tuple=False).squeeze(-1)
    slots = count[ids]
    state["front_history_xy"][ids, slots] = state["edge_xy"][ids]
    state["front_history_heading"][ids, slots] = state["heading"][ids]
    state["front_history_source_z"][ids, slots] = state["source_z"][ids]
    state["front_history_target_z"][ids, slots] = state["target_z"][ids]
    state["front_count"][ids] += 1
    state["front_expected"][ids] = torch.where(
        one[ids], 1 - side[ids], torch.full_like(side[ids], -1),
    )
    state["retired_valid"][retired] = True
    state["retired_xy"][retired] = state["edge_xy"][retired]
    state["retired_z"][retired] = state["target_z"][retired]
    state["active"][retired] = False
    state["age"][state["active"]] += env.step_dt
    # Keep a blocked target retryable for the episode. A ten-second timeout
    # used to blacklist the same edge, closing all subsequent local feedback.
    far_from_target = torch.linalg.vector_norm(asset.data.root_pos_w[:, :2] - state["edge_xy"], dim=-1) > 1.5
    abandoned = state["active"] & far_from_target & (context["up_gate"] == 0)
    state["abandoned_targets"] += abandoned.float()
    state["retired_valid"][abandoned] = True
    state["retired_xy"][abandoned] = state["edge_xy"][abandoned]
    state["retired_z"][abandoned] = state["target_z"][abandoned]
    state["active"][abandoned] = False
    commanded_forward = state["active"] & (context["command_x"] > 0.1)
    forward_speed = asset.data.root_lin_vel_b[:, 0]
    state["forward_steps"] += commanded_forward.float()
    state["retreat_steps"] += (commanded_forward & (forward_speed < -0.05)).float()
    state["stationary_steps"] += (commanded_forward & (forward_speed.abs() < 0.05)).float()

    # Rear target k is the physical riser previously reached by a front wheel.
    rear_i = state["rear_count"]
    target_available = rear_i < state["front_count"]
    ri = rear_i.clamp(max=63)
    rear_edge = state["front_history_xy"][row, ri]
    rear_heading = state["front_history_heading"][row, ri]
    rear_target_z = state["front_history_target_z"][row, ri]
    new_rear_target = target_available & (state["rear_target_index"] != rear_i)
    state["rear_target_index"][new_rear_target] = rear_i[new_rear_target]
    state["rear_target_age"][new_rear_target] = 0.0
    for key in ("rear_prev_valid", "rear_safe", "rear_failed", "rear_physical_time"):
        state[key][new_rear_target] = 0

    rear_x = ((wheel[:, 2:, :2] - rear_edge[:, None]) * rear_heading[:, None]).sum(dim=-1)
    rear_radius = state["radius"][:, 2:]
    rear_retry = target_available[:, None] & (rear_x < -0.06) & state["rear_failed"]
    state["rear_failed"][rear_retry] = False
    state["rear_safe"][rear_x < -0.06] = False
    rear_crossing = (
        target_available[:, None] & state["rear_prev_valid"]
        & (state["rear_prev_x"] < 0) & (rear_x >= 0)
    )
    rear_clear = (
        torch.isfinite(rear_radius)
        & (wheel[:, 2:, 2] - rear_target_z[:, None] - rear_radius >= 0.02)
    )
    state["rear_failed"] |= rear_crossing & ~rear_clear
    state["rear_safe"] |= rear_crossing & rear_clear & ~state["rear_failed"]
    state["rear_prev_x"] = rear_x.clone()
    state["rear_prev_valid"] = target_available[:, None].expand_as(state["rear_prev_valid"]).clone()
    rear_physical = (
        target_available[:, None]
        & _contact_on_tread(
            wheel[:, 2:, 2], rear_target_z[:, None], rear_radius,
            force_z[:, 2:], force_norm[:, 2:], rear_x, torch.zeros_like(rear_x), 0.06,
        )
    )
    state["rear_physical_time"] = torch.where(
        rear_physical, state["rear_physical_time"] + env.step_dt,
        torch.zeros_like(state["rear_physical_time"]),
    )
    rear_candidates = state["rear_physical_time"] >= 0.04
    rear_one = rear_candidates.sum(dim=1) == 1
    rear_side = rear_candidates.long().argmax(dim=1)
    rear_ordered = (state["rear_expected"] < 0) | (rear_side == state["rear_expected"])
    rear_safe_side = state["rear_safe"].gather(1, rear_side[:, None]).squeeze(1)
    rear_valid = target_available & rear_one & rear_ordered & rear_safe_side
    rear_physical_event = target_available & rear_candidates.any(dim=1)
    rear_violation = rear_physical_event & ~rear_valid
    state["rear_event"] = rear_candidates & rear_valid[:, None]
    state["rear_violation_event"] = rear_violation
    state["rear_successes"] += rear_valid.float()
    state["rear_violations"] += rear_violation.float()
    rear_retired = rear_valid | rear_violation
    state["rear_target_age"][target_available] += env.step_dt
    # Waiting is not a physical crossing; the rear target also stays retryable.
    state["rear_count"][rear_retired] += 1
    state["rear_expected"][rear_retired] = torch.where(
        rear_one[rear_retired], 1 - rear_side[rear_retired],
        torch.full_like(rear_side[rear_retired], -1),
    )

    # Two wheels on the same stair are a style violation even if one arrived
    # after the target was already retired. The event is paid once per riser.
    front_last = (state["front_count"] - 1).clamp_min(0)
    front_last_z = state["front_history_target_z"][row, front_last]
    front_last_xy = state["front_history_xy"][row, front_last]
    front_last_heading = state["front_history_heading"][row, front_last]
    front_last_x = ((wheel[:, :2, :2] - front_last_xy[:, None]) * front_last_heading[:, None]).sum(-1)
    front_pair = (
        (state["front_count"] > 0) & contact[:, :2].all(dim=1)
        & (front_last_x > 0.06).all(dim=1)
        & ((wheel[:, :2, 2] - front_last_z[:, None] - state["radius"][:, :2]).abs() <= 0.05).all(dim=1)
    )
    state["front_same_tread_event"] = front_pair & (state["front_same_tread_paid"] != front_last)
    state["front_same_tread_paid"] = torch.where(
        state["front_same_tread_event"], front_last, state["front_same_tread_paid"],
    )
    rear_last = (state["rear_count"] - 1).clamp_min(0)
    rear_last_z = state["front_history_target_z"][row, rear_last]
    rear_last_xy = state["front_history_xy"][row, rear_last]
    rear_last_heading = state["front_history_heading"][row, rear_last]
    rear_last_x = ((wheel[:, 2:, :2] - rear_last_xy[:, None]) * rear_last_heading[:, None]).sum(-1)
    rear_pair = (
        (state["rear_count"] > 0) & contact[:, 2:].all(dim=1)
        & (rear_last_x > 0.06).all(dim=1)
        & ((wheel[:, 2:, 2] - rear_last_z[:, None] - state["radius"][:, 2:]).abs() <= 0.05).all(dim=1)
    )
    state["rear_same_tread_event"] = rear_pair & (state["rear_same_tread_paid"] != rear_last)
    state["rear_same_tread_paid"] = torch.where(
        state["rear_same_tread_event"], rear_last, state["rear_same_tread_paid"],
    )

    env._m20_ascent_cache = {"step": step, "state": state}
    return state
