# Copyright (c) 2026 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""Unified M20 stair teacher: observations, terrain gates, event ledgers and rewards.

Terrain gates are derived from the same yaw-aligned ray scan exposed to the
teacher actor. Contact and actuator state are used only for training signals
and critic observations.

All stair-teacher reward implementations and gait envelopes live here. The env
configuration owns reward weights and entity bindings; TUNING below remains
the numeric source for gait envelope parameters.
"""

from __future__ import annotations

import numpy as np
import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse, yaw_quat

from .commands import UniformThresholdVelocityCommand
from .rewards import joint_pos_penalty


# =============================================================================
# 1. Gait envelope parameters (soft constraints, not hardware safety limits)
# =============================================================================

TUNING = {
    "translation_threshold": 0.10, "yaw_threshold": 0.10,
    "turn_posture_scale": 0.20,
    "air_min": 0.10, "air_target": 0.20, "air_max": 0.35,
    "turn_height_flat": 0.08, "turn_height_terrain": 0.16,
    "turn_reach_flat": 0.22, "turn_reach_terrain": 0.30,
    "height_scale": 0.08, "reach_scale": 0.10,
    "rear_min_extension": 0.35, "extension_scale": 0.10,
    "rear_max_knee": 1.90, "knee_scale": 0.50,
    "front_min_extension": 0.35, "front_max_knee": 2.0,
    "front_support_grace": 0.20, "front_fold_weight": -0.5,
    "front_swing_min_extension": 0.22, "front_swing_max_knee": 2.50,
    "front_swing_scale": 0.50,
    "support_grace": 0.12, "same_tread_grace": 0.12,
    "wheel_radius": 0.09, "nearest_valid_distance": 0.12,
    "turn_size_weight": -1.0, "rear_fold_weight": -1.0,
    "same_tread_event_weight": -1.0, "same_tread_dwell_weight": -1.0,
    "turn_status_weight": 1.0, "turn_symmetry_weight": 2.0,
}


# =============================================================================
# 2. Velocity command exposure and privileged critic observations
# =============================================================================

class StairTeacherVelocityCommand(UniformThresholdVelocityCommand):
    """Track translational command exposure for the stair terrain curriculum."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.episode_commanded_translation_m = torch.zeros(self.num_envs, device=self.device)

    def _update_metrics(self):
        super()._update_metrics()
        self.episode_commanded_translation_m += (
            torch.linalg.vector_norm(self.vel_command_b[:, :2], dim=1) * self._env.step_dt
        )

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self.episode_commanded_translation_m[env_ids] = 0.0
        return super().reset(env_ids)


def critic_joint_acceleration(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Privileged joint acceleration in the configured joint order."""
    asset = env.scene[asset_cfg.name]
    return asset.data.joint_acc[:, asset_cfg.joint_ids]


def critic_applied_torque(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Privileged actuator torque in the configured joint order."""
    asset = env.scene[asset_cfg.name]
    return asset.data.applied_torque[:, asset_cfg.joint_ids]


def critic_wheel_contact_force_magnitude(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Four wheel contact-force magnitudes, used by the critic only."""
    sensor = env.scene.sensors[sensor_cfg.name]
    force = sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    return torch.linalg.vector_norm(force, dim=-1)


def critic_material_friction(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Mean randomized static and dynamic robot material friction per env.

    The ground material in this M20 scene has unit friction and multiply
    combine mode, so the randomized robot material is also the effective
    contact coefficient. This returns two values per environment.
    """
    asset = env.scene[asset_cfg.name]
    material = asset.root_physx_view.get_material_properties().to(env.device)
    if material.ndim < 2 or material.shape[-1] < 2:
        raise RuntimeError(f"Unexpected PhysX material-property shape: {tuple(material.shape)}")
    material = material[..., :2]
    if material.shape[0] == env.num_envs or material.shape[0] % env.num_envs == 0:
        return material.reshape(env.num_envs, -1, 2).mean(dim=1)
    if material.shape[0] == 1:
        return material.reshape(1, -1, 2).mean(dim=1).expand(env.num_envs, -1)
    raise RuntimeError(
        "PhysX material properties cannot be grouped by environment: "
        f"shape={tuple(material.shape)}, num_envs={env.num_envs}"
    )


# =============================================================================
# 3. Height-scan geometry and terrain gates
# =============================================================================

def _stair_scan_context(
    env,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    min_edge_height: float = 0.06,
    gate_ramp_height: float = 0.04,
    lateral_half_width: float = 0.35,
    edge_min_x: float = 0.05,
    edge_max_x: float = 0.70,
    command_threshold: float = 0.10,
) -> dict[str, torch.Tensor]:
    """Find the nearest usable height discontinuity along positive body-x travel.

    RayCaster's exposed local ray starts identify each grid column without
    assuming a hard-coded flattened grid shape. Invalid or incomplete scans
    produce a closed (zero) reward gate.
    """
    sensor = env.scene[sensor_cfg.name]
    counter = int(getattr(env, "common_step_counter", -1))
    cache = getattr(env, "_m20_stair_scan_context", None)
    if cache is not None and cache["counter"] == counter and cache["sensor_name"] == sensor_cfg.name:
        return cache["context"]

    hits = sensor.data.ray_hits_w
    ray_starts = getattr(sensor, "ray_starts", None)
    if ray_starts is None:
        raise RuntimeError("M20 stair rewards require RayCaster.ray_starts for height-grid geometry.")
    ray_starts = torch.as_tensor(ray_starts, device=hits.device, dtype=hits.dtype)
    if ray_starts.ndim == 3:
        # Isaac Lab stores one ray-start grid per sensor instance; the pattern
        # is identical across environments, so one grid defines x/y columns.
        ray_starts = ray_starts[0]
    if ray_starts.ndim != 2 or ray_starts.shape[0] != hits.shape[1] or ray_starts.shape[1] < 2:
        raise RuntimeError(
            "Height scan geometry does not match ray hits: "
            f"starts={tuple(ray_starts.shape)}, hits={tuple(hits.shape)}"
        )

    x_values = ray_starts[:, 0]
    y_values = ray_starts[:, 1]
    x_levels, x_inverse = torch.unique(x_values, sorted=True, return_inverse=True)
    column_heights = []
    column_valid = []
    hit_z = hits[..., 2]
    hit_valid = torch.isfinite(hit_z) & (hit_z.abs() < 1.0e6)
    for column_idx in range(x_levels.numel()):
        in_column = (x_inverse == column_idx) & (y_values.abs() <= lateral_half_width)
        valid = hit_valid[:, in_column]
        values = hit_z[:, in_column]
        nan_values = torch.where(valid, values, torch.full_like(values, float("nan")))
        medians = torch.nanmedian(nan_values, dim=1).values
        has_data = valid.sum(dim=1) >= 2
        column_heights.append(torch.nan_to_num(medians, nan=0.0))
        column_valid.append(has_data)

    if x_levels.numel() < 2:
        raise RuntimeError("M20 height scan must contain at least two distinct forward-x columns.")

    profile = torch.stack(column_heights, dim=1)
    profile_valid = torch.stack(column_valid, dim=1)
    delta = profile[:, 1:] - profile[:, :-1]
    edge_x = 0.5 * (x_levels[1:] + x_levels[:-1])
    command = env.command_manager.get_command(command_name)
    command_x = command[:, 0]
    geometry = profile_valid[:, 1:] & profile_valid[:, :-1] & (delta.abs() >= min_edge_height)
    # A recently passed edge still matters while the rear wheels cross it.
    # Forward style rewards continue to require a command and an edge ahead.
    nearby = geometry & (edge_x.unsqueeze(0) >= -0.45) & (edge_x.unsqueeze(0) <= edge_max_x)
    edge_strength = ((delta.abs() - min_edge_height) / max(gate_ramp_height, 1.0e-6)).clamp(0.0, 1.0)
    terrain_gate = torch.where(nearby, edge_strength, torch.zeros_like(edge_strength)).max(dim=1).values
    candidates = (
        geometry
        & (edge_x.unsqueeze(0) >= edge_min_x)
        & (edge_x.unsqueeze(0) <= edge_max_x)
        & (command_x.unsqueeze(1) > command_threshold)
    )
    candidate_distance = torch.where(candidates, edge_x.abs().unsqueeze(0), torch.inf)
    selected_idx = candidate_distance.argmin(dim=1)
    has_edge = candidates.any(dim=1)
    selected_delta = delta.gather(1, selected_idx[:, None]).squeeze(1)
    selected_x = edge_x[selected_idx]
    lower_i = selected_idx
    upper_i = selected_idx + 1
    lower_z = profile.gather(1, lower_i[:, None]).squeeze(1)
    upper_z = profile.gather(1, upper_i[:, None]).squeeze(1)
    ramp = ((selected_delta.abs() - min_edge_height) / max(gate_ramp_height, 1.0e-6)).clamp(0.0, 1.0)
    gate = ramp * has_edge.to(ramp.dtype)
    up_gate = gate * (selected_delta > 0.0).to(gate.dtype)
    down_gate = gate * (selected_delta < 0.0).to(gate.dtype)

    context = {
        "gate": gate,
        "terrain_gate": terrain_gate,
        "up_gate": up_gate,
        "down_gate": down_gate,
        # Keep all visible ascending edges for style checks. The nearest edge
        # can still be the riser the front wheels have already climbed.
        "ascending_edges_x": edge_x,
        "ascending_edges_valid": nearby & (delta > 0.0),
        "ascending_edges_upper_z": profile[:, 1:],
        "edge_x": torch.where(has_edge, selected_x, torch.zeros_like(selected_x)),
        "edge_delta": torch.where(has_edge, selected_delta, torch.zeros_like(selected_delta)),
        "lower_z": torch.where(has_edge, lower_z, torch.zeros_like(lower_z)),
        "upper_z": torch.where(has_edge, upper_z, torch.zeros_like(upper_z)),
        "command_x": command_x,
    }
    stats = getattr(env, "_m20_stair_gate_stats", None)
    if stats is None or stats["steps"].shape[0] != env.num_envs:
        stats = {key: torch.zeros(env.num_envs, device=env.device)
                 for key in ("steps", "forward", "nearby")}
        setattr(env, "_m20_stair_gate_stats", stats)
    if hasattr(env, "episode_length_buf"):
        reset = env.episode_length_buf <= 1
        for value in stats.values():
            value[reset] = 0
    stats["steps"] += 1
    stats["forward"] += (gate > 0).to(stats["forward"].dtype)
    stats["nearby"] += (terrain_gate > 0).to(stats["nearby"].dtype)
    setattr(env, "_m20_stair_scan_context", {"counter": counter, "sensor_name": sensor_cfg.name, "context": context})
    return context


# =============================================================================
# 4. Adjacent-riser event ledger, clearance credit and landing order
# =============================================================================

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
        "same_tread_time": zeros(2),
        "same_tread_riser": torch.full((n, 2), -1, device=device, dtype=torch.long),
        "same_tread_dwell": zeros(2),
        "same_tread_events_total": zeros(2),
        "same_tread_seconds_total": zeros(2),
    }


def _reset(state, reset):
    if not reset.any():
        return
    for key, value in state.items():
        if key in {"radius"}:
            value[reset] = float("nan")
        elif key in {"front_expected", "rear_expected", "rear_target_index", "front_lift_lead",
                     "front_same_tread_paid", "rear_same_tread_paid", "same_tread_riser"}:
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


def _visible_higher_riser(context, scanner_xy, heading, last_xy, last_heading, last_z):
    """Check all visible risers on the recorded flight, not only the nearest."""
    if "ascending_edges_x" in context:
        x = context["ascending_edges_x"][None].expand(scanner_xy.shape[0], -1)
        valid = context["ascending_edges_valid"]
        upper = context["ascending_edges_upper_z"]
    else:
        # Compatibility with minimal scan-context fixtures/older callers.
        x = context["edge_x"][:, None]
        valid = (context["up_gate"] > 0)[:, None]
        upper = context["upper_z"][:, None]
    edges = scanner_xy[:, None] + heading[:, None] * x[..., None]
    delta = edges - last_xy[:, None]
    along = (delta * last_heading[:, None]).sum(dim=-1)
    lateral = (delta[..., 0] * last_heading[:, None, 1]
               - delta[..., 1] * last_heading[:, None, 0]).abs()
    rise = upper - last_z[:, None]
    aligned = (heading * last_heading).sum(dim=1) > 0.8
    return (valid & (along >= 0.15) & (along <= 0.50) & (lateral <= 0.20)
            & (rise >= 0.06) & (rise <= 0.30) & aligned[:, None]).any(dim=1)


def ascent_state(env, asset_cfg, sensor_cfg, contact_sensor_cfg, command_name="base_velocity"):
    """Advance independent front and rear one-wheel-per-riser sequences once per step."""
    step = int(env.common_step_counter)
    cache = getattr(env, "_m20_ascent_cache", None)
    if cache is not None and cache["step"] == step:
        return cache["state"]

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
    # A platform is allowed to carry both legs. Require evidence of another
    # higher riser on this flight before labelling the pair a style violation.
    front_intermediate = _visible_higher_riser(
        context, scanner.data.pos_w[:, :2], heading, front_last_xy, front_last_heading, front_last_z)
    commanded_forward = context["command_x"] > 0.1
    front_pair &= front_intermediate & commanded_forward
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
    # Front scan may already be flat on the top landing while rear wheels
    # still have recorded intermediate risers to cross.
    next_rear = state["rear_count"].clamp_max(63)
    queued_higher = ((state["rear_count"] < state["front_count"])
                     & (state["front_history_target_z"][row, next_rear] > rear_last_z + 0.06))
    visible_higher = _visible_higher_riser(
        context, scanner.data.pos_w[:, :2], heading, rear_last_xy, rear_last_heading, rear_last_z)
    rear_intermediate = queued_higher | visible_higher
    rear_pair &= rear_intermediate & commanded_forward
    state["rear_same_tread_event"] = rear_pair & (state["rear_same_tread_paid"] != rear_last)
    state["rear_same_tread_paid"] = torch.where(
        state["rear_same_tread_event"], rear_last, state["rear_same_tread_paid"],
    )
    pairs = torch.stack((front_pair, rear_pair), dim=1)
    risers = torch.stack((front_last, rear_last), dim=1)
    new_riser = risers != state["same_tread_riser"]
    state["same_tread_time"][new_riser] = 0.0
    state["same_tread_riser"] = risers
    # A command pause, turn, or temporary scan loss only closes the gate;
    # it cannot renew the already consumed grace on this physical riser.
    # The grace budget belongs to the tread, not every contact episode: briefly
    # lifting/replacing a wheel on that same tread cannot renew the free time.
    state["same_tread_time"] += pairs.float() * env.step_dt
    # Permit a brief transfer of support, then charge time rather than only
    # a one-shot event. The rate is bounded per axle and stops when a leg leaves.
    state["same_tread_dwell"] = (pairs & (state["same_tread_time"] > TUNING["same_tread_grace"])).float()
    state["same_tread_events_total"] += torch.stack(
        (state["front_same_tread_event"], state["rear_same_tread_event"]), dim=1).float()
    state["same_tread_seconds_total"] += state["same_tread_dwell"] * env.step_dt

    env._m20_ascent_cache = {"step": step, "state": state}
    return state


# =============================================================================
# 5. Stair task rewards, posture, contact and energy costs
# =============================================================================

def stair_terrain_aware_orientation_cost(
    env,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    flat_weight: float = 50.0,
    stair_weight: float = 5.0,
) -> torch.Tensor:
    """Relax posture near stairs and permit pitch during descent rear follow."""
    asset = env.scene[asset_cfg.name]
    context = _stair_scan_context(env, sensor_cfg, command_name)
    gravity = asset.data.projected_gravity_b
    state = getattr(env, "_m20_stair_step_state", None)
    descent_rear_follow = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if state is not None:
        descent_rear_follow = (
            state["active"] & ~state["up"] & state["front_reached"].all(dim=1)
            & ~state["rear_support_paid"].all(dim=1)
        )
    stair_gate = torch.maximum(context["terrain_gate"], descent_rear_follow.float())
    effective_weight = flat_weight * (1.0 - stair_gate) + stair_weight * stair_gate
    # During rear follow, pitch is needed to transfer support down a step.
    tilt = torch.where(descent_rear_follow, gravity[:, 1].square(), gravity[:, :2].square().sum(dim=1))
    return effective_weight * tilt


def stair_joint_pos_cost(
    env,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float,
    velocity_threshold: float,
    command_threshold: float,
    stair_sensor_cfg: SceneEntityCfg,
    stair_penalty_scale: float,
    disable_turn_side_cmd: bool = False,
    ang_cmd_threshold: float = 0.1,
    y_cmd_threshold: float = 0.1,
    xy_norm_max: float = 0.1,
    xz_norm_max: float = 0.1,
    turn_penalty_scale: float = 0.0,
) -> torch.Tensor:
    """Keep the M20 posture cost, relaxing it near a visible stair edge."""
    cost = joint_pos_penalty(
        env, command_name, asset_cfg, stand_still_scale, velocity_threshold, command_threshold
    )
    if disable_turn_side_cmd:
        cmd = env.command_manager.get_command(command_name)
        turn = (cmd[:, 2].abs() > ang_cmd_threshold) & (torch.linalg.vector_norm(cmd[:, :2], dim=1) < xy_norm_max)
        side = (cmd[:, 1].abs() > y_cmd_threshold) & (
            torch.linalg.vector_norm(cmd[:, [0, 2]], dim=1) < xz_norm_max
        )
        # Sideways relaxation remains available. Pure yaw keeps a weak posture
        # reference rather than removing the hip/knee constraint altogether.
        scale = torch.where(turn, turn_penalty_scale, 1.0)
        cost = cost * torch.where(side, 0.0, scale)
    gate = _stair_scan_context(env, stair_sensor_cfg, command_name)["terrain_gate"]
    return cost * (1.0 - (1.0 - stair_penalty_scale) * gate)


def _stair_new_progress(env, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg, command_name: str):
    """New forward distance along the heading fixed at the first stair edge.

    A per-episode high-water mark makes reversing and crossing the same ground
    again worth zero. The returned value has velocity units for rate rewards.
    """
    counter = int(getattr(env, "common_step_counter", -1))
    cache = getattr(env, "_m20_stair_progress_cache", None)
    if cache is not None and cache["counter"] == counter:
        return cache["rate"], cache["distance"]

    asset = env.scene[asset_cfg.name]
    scanner = env.scene[sensor_cfg.name]
    context = _stair_scan_context(env, sensor_cfg, command_name)
    state = getattr(env, "_m20_stair_progress_state", None)
    if state is None or state["origin"].shape[0] != env.num_envs:
        state = {
            "origin": torch.zeros((env.num_envs, 2), device=env.device),
            "direction": torch.zeros((env.num_envs, 2), device=env.device),
            "maximum": torch.zeros(env.num_envs, device=env.device),
            "desired_lateral": torch.zeros(env.num_envs, device=env.device),
            "active": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
        }
        setattr(env, "_m20_stair_progress_state", state)
    if hasattr(env, "episode_length_buf"):
        reset = env.episode_length_buf <= 1
        state["active"][reset] = False
        state["maximum"][reset] = 0.0
        state["desired_lateral"][reset] = 0.0

    start = (context["gate"] > 0) & ~state["active"]
    forward = torch.zeros((env.num_envs, 3), device=env.device)
    forward[:, 0] = 1.0
    heading = quat_apply(yaw_quat(scanner.data.quat_w), forward)[:, :2]
    state["origin"][start] = asset.data.root_pos_w[start, :2]
    state["direction"][start] = heading[start]
    state["desired_lateral"][start] = 0.0
    state["active"][start] = True

    command = env.command_manager.get_command(command_name)
    commanded_body = torch.zeros((env.num_envs, 3), device=env.device)
    commanded_body[:, :2] = command[:, :2]
    commanded_world = quat_apply(yaw_quat(scanner.data.quat_w), commanded_body)[:, :2]
    route = state["direction"]
    commanded_side = -commanded_world[:, 0] * route[:, 1] + commanded_world[:, 1] * route[:, 0]
    state["desired_lateral"] += commanded_side * env.step_dt * state["active"].float()

    displacement = ((asset.data.root_pos_w[:, :2] - state["origin"]) * state["direction"]).sum(dim=1)
    new_maximum = torch.maximum(state["maximum"], displacement.clamp(min=0.0))
    gained = new_maximum - state["maximum"]
    state["maximum"] = new_maximum
    rate = context["gate"] * (gained / env.step_dt).clamp(max=1.0)
    setattr(env, "_m20_stair_progress_cache", {"counter": counter, "rate": rate, "distance": new_maximum})
    return rate, new_maximum


def _stair_route_quality(env, sensor_cfg: SceneEntityCfg, command_name: str = "base_velocity") -> torch.Tensor:
    """Gate stair rewards by a route fixed when the first riser is seen."""
    _stair_new_progress(env, SceneEntityCfg("robot"), sensor_cfg, command_name)
    context = _stair_scan_context(env, sensor_cfg, command_name)
    state = env._m20_stair_progress_state
    root_xy = env.scene["robot"].data.root_pos_w[:, :2]
    offset = root_xy - state["origin"]
    direction = state["direction"]
    lateral_actual = -offset[:, 0] * direction[:, 1] + offset[:, 1] * direction[:, 0]
    lateral = torch.abs(lateral_actual - state.get("desired_lateral", 0.0))
    excess = (lateral - 0.25).clamp_min(0.0)
    quality = torch.exp(-torch.square(excess / 0.25))
    quality = torch.where(lateral >= 0.75, torch.zeros_like(quality), quality)
    active = state["active"] & (context["command_x"] > 0.10)
    return torch.where(active, quality, torch.ones_like(quality))


def stair_forward_progress(
    env,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """Reward only new ground covered while a stair edge is visible."""
    rate, _ = _stair_new_progress(env, asset_cfg, sensor_cfg, command_name)
    return rate * _stair_route_quality(env, sensor_cfg, command_name)


def _stair_front_sequence_events(
    env, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg, command_name: str = "base_velocity",
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Return per-riser ascent events and command-aware route quality."""
    state = ascent_state(env, asset_cfg, sensor_cfg, contact_sensor_cfg, command_name)
    return state, _stair_route_quality(env, sensor_cfg, command_name)


def stair_pure_yaw_air_time(
    env, command_name: str, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg,
    threshold: float, stair_sensor_cfg: SceneEntityCfg,
    cmd_threshold: float = 0.1, foot_height_threshold: float = 0.05,
) -> torch.Tensor:
    """Keep the original turning reward out of commanded stair traversal."""
    from .rewards import feet_air_time_ang_z_cmd_M20
    native = feet_air_time_ang_z_cmd_M20(
        env, command_name, sensor_cfg, threshold, cmd_threshold,
        foot_height_threshold, asset_cfg,
    )
    command = env.command_manager.get_command(command_name)
    terrain_gate = _stair_scan_context(env, stair_sensor_cfg, command_name)["terrain_gate"]
    state = getattr(env, "_m20_stair_step_state", None)
    if state is not None:
        terrain_gate = torch.maximum(terrain_gate, state["active"].float())
    ascent = getattr(env, "_m20_ascent_state", None)
    if ascent is not None:
        terrain_gate = torch.maximum(terrain_gate, ascent["active"].float())
    stair_forward = (terrain_gate > 0) & (torch.linalg.vector_norm(command[:, :2], dim=1) > 0.10)
    return native * (~stair_forward).float()


def stair_front_wheel_lift(
    env, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg, command_name: str = "base_velocity",
) -> torch.Tensor:
    """Pay bounded lift-height improvement before the expected leg lands."""
    state = ascent_state(env, asset_cfg, sensor_cfg, contact_sensor_cfg, command_name)
    return state["front_lift_gain"] / env.step_dt


def stair_forward_motion_cost(
    env, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """Bounded forward-command shortfall near a visible or tracked stair.

    A target stays active after retreat makes it disappear from the scan.
    Backward commands and deliberate turns do not activate this cost.
    """
    asset = env.scene[asset_cfg.name]
    scanner = env.scene[sensor_cfg.name]
    context = _stair_scan_context(env, sensor_cfg, command_name)
    gate = context["gate"].clone()
    forward = torch.zeros((env.num_envs, 3), device=env.device)
    forward[:, 0] = 1.0
    heading = quat_apply(yaw_quat(scanner.data.quat_w), forward)[:, :2]
    for name in ("_m20_ascent_state", "_m20_stair_step_state"):
        state = getattr(env, name, None)
        if state is not None:
            aligned = (heading * state["heading"]).sum(dim=1) > 0.8
            nearby = torch.linalg.vector_norm(asset.data.root_pos_w[:, :2] - state["edge_xy"], dim=1) <= 1.5
            gate = torch.maximum(gate, (state["active"] & aligned & nearby).float())
    command = env.command_manager.get_command(command_name)
    traversing = (command[:, 0] > 0.1) & (command[:, 0] > command[:, 1].abs())
    shortfall = ((command[:, 0] - asset.data.root_lin_vel_b[:, 0])
                 / command[:, 0].clamp_min(0.3)).clamp(0.0, 2.0)
    return gate * traversing.float() * shortfall


def stair_front_wheel_touchdown(
    env, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg, command_name: str = "base_velocity",
) -> torch.Tensor:
    """Credit one safely landed front wheel per new ascending riser, plus descent support."""
    state = ascent_state(env, asset_cfg, sensor_cfg, contact_sensor_cfg, command_name)
    _stair_step_completion_events(env, asset_cfg, sensor_cfg, contact_sensor_cfg, command_name)
    down_state = env._m20_stair_step_state
    down_front = down_state["front_support_event"].float().sum(dim=1)
    return (state["front_event"].float().sum(dim=1) + down_front) / env.step_dt


def stair_front_single_support(
    env,
    asset_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """Prefer single front support only when the robot covers new ground."""
    contact_sensor = env.scene.sensors[contact_sensor_cfg.name]
    progress_rate, _ = _stair_new_progress(env, asset_cfg, sensor_cfg, command_name)
    in_contact = contact_sensor.data.current_contact_time[:, contact_sensor_cfg.body_ids] > 0.0
    one_support = in_contact.sum(dim=1) == 1
    return progress_rate * one_support.to(progress_rate.dtype) * _stair_route_quality(env, sensor_cfg, command_name)


def stair_diagonal_support_symmetry(
    env,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """Prefer diagonal support only during new forward stair progress."""
    sensor = env.scene.sensors[contact_sensor_cfg.name]
    progress_rate, _ = _stair_new_progress(env, asset_cfg, sensor_cfg, command_name)
    contact = sensor.data.current_contact_time[:, contact_sensor_cfg.body_ids] > 0.0
    # body_ids are configured in FL, FR, HL, HR order.
    matches = (contact[:, 0] == contact[:, 3]) & (contact[:, 1] == contact[:, 2])
    return progress_rate * matches.to(progress_rate.dtype) * _stair_route_quality(env, sensor_cfg, command_name)


def stair_front_alternation(
    env, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg, command_name: str = "base_velocity",
) -> torch.Tensor:
    """Bonus after the first, only for the opposite front leg on the next riser."""
    state = ascent_state(env, asset_cfg, sensor_cfg, contact_sensor_cfg, command_name)
    return (state["front_event"].any(dim=1) & (state["front_count"] > 1)).float() / env.step_dt


def _stair_step_completion_events(
    env,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    tread_inset: float = 0.06,
    stable_time_s: float = 0.06,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Track a visible edge until both rear wheels reach its destination tread.

    Front wheels may already be on a later tread, so requiring all four wheels
    to stand on one tread would reject valid traversals of narrow stairs.
    """
    counter = int(getattr(env, "common_step_counter", -1))
    cache = getattr(env, "_m20_stair_step_completion_cache", None)
    if cache is not None and cache["counter"] == counter:
        return cache["up"], cache["down"]

    asset = env.scene[asset_cfg.name]
    scanner = env.scene[sensor_cfg.name]
    contact_sensor = env.scene.sensors[contact_sensor_cfg.name]
    context = _stair_scan_context(env, sensor_cfg, command_name)
    state = getattr(env, "_m20_stair_step_state", None)
    if state is None or state["active"].shape[0] != env.num_envs:
        state = {
            "active": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
            "edge_xy": torch.zeros(env.num_envs, 2, device=env.device),
            "heading": torch.zeros(env.num_envs, 2, device=env.device),
            "target_z": torch.zeros(env.num_envs, device=env.device),
            "source_z": torch.zeros(env.num_envs, device=env.device),
            "rear_source_clearance": torch.full((env.num_envs, 2), float("nan"), device=env.device),
            "up": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
            "front_reached": torch.zeros(env.num_envs, 2, dtype=torch.bool, device=env.device),
            "front_source_clearance": torch.full((env.num_envs, 2), float("nan"), device=env.device),
            "front_previous_distance": torch.zeros(env.num_envs, 2, device=env.device),
            "front_previous_valid": torch.zeros(env.num_envs, 2, dtype=torch.bool, device=env.device),
            "front_crossed_safe": torch.zeros(env.num_envs, 2, dtype=torch.bool, device=env.device),
            "front_crossing_failed": torch.zeros(env.num_envs, 2, dtype=torch.bool, device=env.device),
            "front_face_collision": torch.zeros(env.num_envs, 2, dtype=torch.bool, device=env.device),
            "front_face_event": torch.zeros(env.num_envs, 2, dtype=torch.bool, device=env.device),
            "front_lift_escrow": torch.zeros(env.num_envs, 2, device=env.device),
            "front_support_time": torch.zeros(env.num_envs, 2, device=env.device),
            "front_support_paid": torch.zeros(env.num_envs, 2, dtype=torch.bool, device=env.device),
            "front_support_event": torch.zeros(env.num_envs, 2, dtype=torch.bool, device=env.device),
            "front_pair_paid": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
            "front_pair_event": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
            "front_lead_paid": torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
            "front_lead_event": torch.zeros(env.num_envs, 2, dtype=torch.bool, device=env.device),
            "rear_support_time": torch.zeros(env.num_envs, 2, device=env.device),
            "rear_support_paid": torch.zeros(env.num_envs, 2, dtype=torch.bool, device=env.device),
            "rear_support_event": torch.zeros(env.num_envs, 2, dtype=torch.bool, device=env.device),
            "stable_time": torch.zeros(env.num_envs, device=env.device),
            "age": torch.zeros(env.num_envs, device=env.device),
            "completed_edge_xy": torch.zeros(env.num_envs, 64, 2, device=env.device),
            "completed_heading": torch.zeros(env.num_envs, 64, 2, device=env.device),
            "completed_target_z": torch.zeros(env.num_envs, 64, device=env.device),
            "completed_count": torch.zeros(env.num_envs, dtype=torch.long, device=env.device),
            "up_attempts": torch.zeros(env.num_envs, device=env.device),
            "down_attempts": torch.zeros(env.num_envs, device=env.device),
            "up_successes": torch.zeros(env.num_envs, device=env.device),
            "down_successes": torch.zeros(env.num_envs, device=env.device),
        }
        setattr(env, "_m20_stair_step_state", state)
    if hasattr(env, "episode_length_buf"):
        reset = env.episode_length_buf <= 1
        for key in ("active", "front_reached", "front_previous_valid", "front_crossed_safe", "front_crossing_failed",
                    "front_face_collision", "front_face_event", "front_lift_escrow", "front_support_time", "front_support_paid", "front_support_event",
                    "front_pair_paid", "front_pair_event", "front_lead_paid", "front_lead_event", "rear_support_time", "rear_support_paid",
                    "rear_support_event", "completed_count", "up_attempts", "down_attempts",
                    "up_successes", "down_successes", "stable_time", "age"):
            state[key][reset] = 0
        state["rear_source_clearance"][reset] = float("nan")
        state["front_source_clearance"][reset] = float("nan")

    forward = torch.zeros(env.num_envs, 3, device=env.device)
    forward[:, 0] = 1.0
    heading = quat_apply(yaw_quat(scanner.data.quat_w), forward)[:, :2]
    candidate_edge_xy = scanner.data.pos_w[:, :2] + heading * context["edge_x"].unsqueeze(1)
    # Remember every completed edge in this episode. Checking only the last
    # target would allow up-down-up oscillation to collect the first bonus twice.
    history_valid = (torch.arange(64, device=env.device).unsqueeze(0)
                     < state["completed_count"].unsqueeze(1))
    along_edge = (((candidate_edge_xy.unsqueeze(1) - state["completed_edge_xy"])
                   * state["completed_heading"]).sum(dim=-1)).abs()
    aligned = ((heading.unsqueeze(1) * state["completed_heading"]).sum(dim=-1)).abs() > 0.8
    same_edge = (history_valid & aligned & (along_edge < 0.15)
                 & ((context["upper_z"].unsqueeze(1) - state["completed_target_z"]).abs() < 0.06))
    start = (~state["active"] & (context["down_gate"] > 0) & ~same_edge.any(dim=1)
             & (state["completed_count"] < 64))
    state["edge_xy"][start] = candidate_edge_xy[start]
    state["heading"][start] = heading[start]
    state["target_z"][start] = context["upper_z"][start]
    state["source_z"][start] = context["lower_z"][start]
    state["rear_source_clearance"][start] = float("nan")
    state["front_source_clearance"][start] = float("nan")
    for key in ("front_previous_valid", "front_crossed_safe", "front_crossing_failed",
                "front_face_collision", "front_lift_escrow",
                "front_support_time", "front_support_paid", "rear_support_time", "rear_support_paid"):
        state[key][start] = 0
    state["front_pair_paid"][start] = False
    state["front_lead_paid"][start] = False
    state["up"][start] = context["edge_delta"][start] > 0
    state["front_reached"][start] = False
    state["stable_time"][start] = 0
    state["age"][start] = 0
    state["active"][start] = True
    state["up_attempts"][start & state["up"]] += 1
    state["down_attempts"][start & ~state["up"]] += 1

    wheel_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    signed_distance = ((wheel_pos[:, :, :2] - state["edge_xy"].unsqueeze(1))
                       * state["heading"].unsqueeze(1)).sum(dim=-1)
    wheel_height = wheel_pos[:, :, 2] - state["target_z"].unsqueeze(1)
    forces = contact_sensor.data.net_forces_w[:, contact_sensor_cfg.body_ids, :]
    contact = (contact_sensor.data.current_contact_time[:, contact_sensor_cfg.body_ids] > 0) & (
        forces[:, :, 2] > 0.5 * torch.linalg.vector_norm(forces, dim=-1)
    )
    # Learn the wheel-center offset from a real source-tread contact instead
    # of assuming the reference model's wheel radius.
    front_baseline = state["front_source_clearance"]
    source_front = (
        state["active"][:, None] & contact[:, :2] & (signed_distance[:, :2] < -tread_inset)
        & ~torch.isfinite(front_baseline)
    )
    source_front_height = wheel_pos[:, :2, 2] - state["source_z"][:, None]
    source_front &= (source_front_height >= 0.02) & (source_front_height <= 0.25)
    front_baseline[source_front] = source_front_height[source_front]

    before = signed_distance[:, :2] < 0.0
    front_force_xy = torch.linalg.vector_norm(forces[:, :2, :2], dim=-1)
    face_now = (
        state["active"][:, None] & state["up"][:, None]
        & (signed_distance[:, :2] >= -0.12) & (signed_distance[:, :2] <= 0.03)
        & (wheel_height[:, :2] < front_baseline + 0.05)
        & (front_force_xy > 50.0)
        & (front_force_xy > 1.5 * forces[:, :2, 2].clamp_min(0.0))
    )
    state["front_face_event"] = face_now & ~state["front_face_collision"]
    state["front_face_collision"] |= face_now
    upward_speed = asset.data.body_lin_vel_w[:, asset_cfg.body_ids[:2], 2].clamp_min(0.0)
    lift = (
        state["active"][:, None] & state["up"][:, None] & before
        & ~contact[:, :2] & torch.isfinite(front_baseline) & (upward_speed >= 0.10)
    )
    state["front_lift_escrow"] = torch.maximum(
        state["front_lift_escrow"],
        torch.where(lift, (upward_speed / 0.5).clamp_max(1.0), torch.zeros_like(upward_speed)),
    )
    # Height is checked on the control step that crosses the physical riser.
    crossing = (
        state["active"][:, None] & state["up"][:, None] & state["front_previous_valid"]
        & (state["front_previous_distance"] < 0.0) & (signed_distance[:, :2] >= 0.0)
    )
    clear = (
        torch.isfinite(front_baseline)
        & (wheel_height[:, :2] - front_baseline >= 0.05)
        & ~state["front_face_collision"]
    )
    state["front_crossing_failed"] |= crossing & ~clear
    state["front_crossed_safe"] |= crossing & clear & ~state["front_crossing_failed"]
    state["front_lift_escrow"][crossing & ~clear] = 0.0
    state["front_previous_distance"] = signed_distance[:, :2].clone()
    state["front_previous_valid"] = state["active"][:, None].expand_as(state["front_previous_valid"]).clone()

    # Reward upper/lower support only after a true tread contact persisted.
    destination_height = (
        torch.isfinite(front_baseline)
        & (wheel_height[:, :2] - front_baseline <= 0.05)
    )
    valid_crossing = torch.where(
        state["up"][:, None], state["front_crossed_safe"],
        state["front_previous_valid"] & (signed_distance[:, :2] > 0.0),
    )
    front_now = (
        state["active"][:, None] & contact[:, :2]
        & (signed_distance[:, :2] > tread_inset)
        & destination_height & valid_crossing
    )
    state["front_support_time"] = torch.where(
        front_now, state["front_support_time"] + env.step_dt,
        torch.zeros_like(state["front_support_time"]),
    )
    confirmed_front = state["front_support_time"] >= 0.04
    state["front_support_event"] = confirmed_front & ~state["front_support_paid"]
    state["front_support_paid"] |= state["front_support_event"]
    state["front_reached"] |= confirmed_front
    one_lead = (state["front_support_event"].sum(dim=1) == 1) & ~state["front_lead_paid"]
    state["front_lead_event"] = state["front_support_event"] & one_lead[:, None]
    state["front_lead_paid"] |= one_lead
    state["front_pair_event"] = (
        state["active"] & confirmed_front.all(dim=1) & ~state["front_pair_paid"]
    )
    state["front_pair_paid"] |= state["front_pair_event"]

    # Calibrate each rear wheel's center height while it contacts the source
    # tread. A fixed 2-25 cm range would mistake an upper-source contact for a
    # successful descent of a 10 cm step.
    source_clearance = wheel_pos[:, 2:, 2] - state["source_z"].unsqueeze(1)
    baseline = state["rear_source_clearance"]
    source_contact = (state["active"].unsqueeze(1) & contact[:, 2:]
                      & (signed_distance[:, 2:] < -tread_inset)
                      & (source_clearance >= 0.02) & (source_clearance <= 0.25)
                      & ~torch.isfinite(baseline))
    baseline[source_contact] = source_clearance[source_contact]
    target_contact = (contact[:, 2:] & (signed_distance[:, 2:] > tread_inset)
                      & torch.isfinite(baseline)
                      & (wheel_height[:, 2:] - baseline <= 0.05))
    state["rear_support_time"] = torch.where(
        target_contact & state["front_reached"].all(dim=1, keepdim=True),
        state["rear_support_time"] + env.step_dt,
        torch.zeros_like(state["rear_support_time"]),
    )
    rear_confirmed = state["rear_support_time"] >= 0.04
    state["rear_support_event"] = rear_confirmed & ~state["rear_support_paid"]
    state["rear_support_paid"] |= state["rear_support_event"]
    rear_now = state["rear_support_paid"].all(dim=1) & target_contact.all(dim=1)
    upright = torch.linalg.vector_norm(asset.data.projected_gravity_b[:, :2], dim=1) < 0.57
    stable = state["active"] & state["front_reached"].all(dim=1) & rear_now & upright
    state["stable_time"] = torch.where(stable, state["stable_time"] + env.step_dt,
                                       torch.zeros_like(state["stable_time"]))
    completed = stable & (state["stable_time"] >= stable_time_s)
    up = completed & state["up"]
    down = completed & ~state["up"]
    state["up_successes"][up] += 1
    state["down_successes"][down] += 1
    completed_ids = torch.nonzero(completed, as_tuple=False).squeeze(-1)
    completed_slots = state["completed_count"][completed_ids]
    state["completed_edge_xy"][completed_ids, completed_slots] = state["edge_xy"][completed_ids]
    state["completed_heading"][completed_ids, completed_slots] = state["heading"][completed_ids]
    state["completed_target_z"][completed_ids, completed_slots] = state["target_z"][completed_ids]
    state["completed_count"][completed_ids] += 1
    state["active"][completed] = False

    state["age"][state["active"]] += env.step_dt
    root_distance = ((asset.data.root_pos_w[:, :2] - state["edge_xy"]) * state["heading"]).sum(dim=1)
    stale = state["active"] & ((state["age"] > 10.0) | (root_distance < -0.70))
    state["active"][stale] = False
    setattr(env, "_m20_stair_step_completion_cache", {"counter": counter, "up": up, "down": down})
    return up, down


def stair_up_step_completion(
    env, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Credit a safe front landing on the next ascending riser, one leg only."""
    state = ascent_state(env, asset_cfg, sensor_cfg, contact_sensor_cfg)
    return state["front_event"].any(dim=1).float() / env.step_dt


def stair_up_rear_step_completion(
    env, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Credit one alternating rear landing per front-confirmed ascending riser."""
    state = ascent_state(env, asset_cfg, sensor_cfg, contact_sensor_cfg)
    return state["rear_event"].any(dim=1).float() / env.step_dt


def stair_up_order_violation_cost(
    env, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """One-shot cost for unsafe, simultaneous, or wrong-side riser landings."""
    state = ascent_state(env, asset_cfg, sensor_cfg, contact_sensor_cfg)
    return (state["front_violation_event"] | state["rear_violation_event"]).float() / env.step_dt


def stair_up_same_tread_cost(
    env, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Cost once per riser if both front or both rear wheels share its tread."""
    state = ascent_state(env, asset_cfg, sensor_cfg, contact_sensor_cfg)
    return (state["front_same_tread_event"] | state["rear_same_tread_event"]).float() / env.step_dt


def stair_up_same_tread_dwell_cost(env, asset_cfg, sensor_cfg, contact_sensor_cfg):
    """Bounded time cost on intermediate treads; short support transfers are free."""
    state = ascent_state(env, asset_cfg, sensor_cfg, contact_sensor_cfg)
    return state["same_tread_dwell"].mean(dim=1)


def stair_down_step_completion(
    env, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """One event when a whole riser has been descended and rear support is stable."""
    _, down = _stair_step_completion_events(env, asset_cfg, sensor_cfg, contact_sensor_cfg)
    return down.to(torch.float32) / env.step_dt


def stair_front_riser_face_cost(
    env, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg, command_name: str = "base_velocity",
) -> torch.Tensor:
    """One-shot cost for a loaded front wheel striking the rising face."""
    state = ascent_state(env, asset_cfg, sensor_cfg, contact_sensor_cfg, command_name)
    return state["front_face_event"].float().sum(dim=1) / env.step_dt


def stair_descent_rear_support(
    env, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg, command_name: str = "base_velocity",
) -> torch.Tensor:
    """Small one-shot credit for each rear wheel safely supporting the lower tread."""
    _stair_step_completion_events(env, asset_cfg, sensor_cfg, contact_sensor_cfg, command_name)
    state = env._m20_stair_step_state
    return (
        state["rear_support_event"].float().sum(dim=1) * (~state["up"]).float()
        / env.step_dt
    )


def stair_route_track_lin_vel_xy_exp(
    env, std: float, command_name: str,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Keep native tracking for every direction, including commands after a turn.

    Stair completion still needs an actual riser crossing. The old frozen
    route multiplier could erase tracking reward after a legitimate turn.
    """
    from .rewards import track_lin_vel_xy_exp
    native = track_lin_vel_xy_exp(env, std, command_name, asset_cfg)
    return native


def stair_step_completion_metrics(env, env_ids) -> dict[str, torch.Tensor]:
    """Report front and rear ascent sequence results alongside descent results."""
    down_state = getattr(env, "_m20_stair_step_state", None)
    up_state = getattr(env, "_m20_ascent_state", None)
    zero = torch.zeros((), device=env.device)
    result = {}
    if up_state is None:
        result.update({key: zero for key in ("up_attempts", "up_successes", "up_success_rate",
                 "up_rear_successes", "up_front_violations", "up_rear_violations")})
    else:
        attempts = up_state["front_count"][env_ids].float()
        successes = up_state["front_successes"][env_ids]
        result["up_attempts"] = attempts.mean()
        result["up_successes"] = successes.mean()
        result["up_success_rate"] = successes.sum() / attempts.sum().clamp(min=1.0)
        result["up_rear_successes"] = up_state["rear_successes"][env_ids].mean()
        result["up_front_violations"] = up_state["front_violations"][env_ids].mean()
        result["up_rear_violations"] = up_state["rear_violations"][env_ids].mean()
        for axle, name in enumerate(("front", "rear")):
            result[f"up_{name}_same_tread_events"] = up_state["same_tread_events_total"][env_ids, axle].mean()
            result[f"up_{name}_same_tread_dwell_s"] = up_state["same_tread_seconds_total"][env_ids, axle].mean()
        for metric, field in (("up_targets_started", "targets_started"),
                              ("up_lift_credit", "front_lift_credit"), ("up_retries", "retry_count"),
                              ("up_abandoned_targets", "abandoned_targets")):
            result[metric] = up_state[field][env_ids].mean()
        forward_steps = up_state["forward_steps"][env_ids].sum().clamp_min(1.0)
        result["up_retreat_fraction"] = up_state["retreat_steps"][env_ids].sum() / forward_steps
        result["up_stationary_fraction"] = up_state["stationary_steps"][env_ids].sum() / forward_steps
    if down_state is None:
        result.update({key: zero for key in ("down_attempts", "down_successes", "down_success_rate")})
    else:
        attempts = down_state["down_attempts"][env_ids]
        successes = down_state["down_successes"][env_ids]
        result["down_attempts"] = attempts.mean()
        result["down_successes"] = successes.mean()
        result["down_success_rate"] = successes.sum() / attempts.sum().clamp(min=1.0)
    gate_stats = getattr(env, "_m20_stair_gate_stats", None)
    if gate_stats is not None:
        steps = gate_stats["steps"][env_ids].sum().clamp(min=1.0)
        result["forward_gate_fraction"] = gate_stats["forward"][env_ids].sum() / steps
        result["nearby_gate_fraction"] = gate_stats["nearby"][env_ids].sum() / steps
    else:
        result["forward_gate_fraction"] = zero
        result["nearby_gate_fraction"] = zero
    return result


def stair_impact_cost(
    env,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    free_fall_speed: float = 0.10,
) -> torch.Tensor:
    """Penalize hard wheel touchdowns during a stair traversal."""
    asset = env.scene[asset_cfg.name]
    contact_sensor = env.scene.sensors[contact_sensor_cfg.name]
    context = _stair_scan_context(env, sensor_cfg, command_name)
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, contact_sensor_cfg.body_ids].to(torch.float32)
    downward_speed = (-asset.data.body_lin_vel_w[:, asset_cfg.body_ids, 2] - free_fall_speed).clamp(min=0.0)
    return context["terrain_gate"] * torch.sum(first_contact * downward_speed.square(), dim=1)


def stair_non_wheel_contact_cost(
    env, sensor_cfg: SceneEntityCfg, stair_sensor_cfg: SceneEntityCfg,
    force_threshold: float = 20.0, force_scale: float = 180.0,
) -> torch.Tensor:
    """Bounded cost for shank or leg contact with a stair."""
    sensor = env.scene.sensors[sensor_cfg.name]
    force = torch.linalg.vector_norm(
        sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids], dim=-1
    ).amax(dim=1)
    excess = ((force - force_threshold).clamp_min(0.0) / force_scale).square()
    gate = _stair_scan_context(env, stair_sensor_cfg)["terrain_gate"]
    state = getattr(env, "_m20_stair_step_state", None)
    if state is not None:
        gate = torch.maximum(gate, state["active"].float())
    ascent = getattr(env, "_m20_ascent_state", None)
    if ascent is not None:
        gate = torch.maximum(gate, ascent["active"].float())
    return gate * excess.sum(dim=1).clamp_max(1.0)


def stair_non_wheel_scrape_cost(
    env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg,
    stair_sensor_cfg: SceneEntityCfg, force_threshold: float = 8.0,
    speed_threshold: float = 0.08, speed_scale: float = 0.55,
) -> torch.Tensor:
    """Bounded cost when a loaded non-wheel link slides against a stair."""
    sensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]
    force = torch.linalg.vector_norm(
        sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids], dim=-1
    ).amax(dim=1)
    speed = torch.linalg.vector_norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=-1)
    scrape = ((speed - speed_threshold).clamp_min(0.0) / speed_scale).square()
    gate = _stair_scan_context(env, stair_sensor_cfg)["terrain_gate"]
    state = getattr(env, "_m20_stair_step_state", None)
    if state is not None:
        gate = torch.maximum(gate, state["active"].float())
    ascent = getattr(env, "_m20_ascent_state", None)
    if ascent is not None:
        gate = torch.maximum(gate, ascent["active"].float())
    return gate * (scrape * (force > force_threshold).float()).sum(dim=1).clamp_max(1.0)


def stair_joint_effort_cost(
    env,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """Normalized squared actuator effort while stepping on stairs."""
    asset = env.scene[asset_cfg.name]
    context = _stair_scan_context(env, sensor_cfg, command_name)
    limits = getattr(env, "_m20_stair_actuator_effort_limits", None)
    if limits is None or limits.shape != asset.data.applied_torque.shape:
        limits = torch.zeros_like(asset.data.applied_torque)
        assigned = torch.zeros(limits.shape[1], dtype=torch.bool, device=limits.device)
        for actuator in asset.actuators.values():
            joint_ids = actuator.joint_indices
            limits[:, joint_ids] = actuator.effort_limit.abs()
            assigned[joint_ids] = True
        if not torch.all(assigned) or not torch.all(torch.isfinite(limits)) or not torch.all(limits > 0):
            raise RuntimeError("M20 stair effort cost requires finite physical limits for every actuator joint.")
        setattr(env, "_m20_stair_actuator_effort_limits", limits)
    torque = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return context["terrain_gate"] * torch.mean((torque / limits[:, asset_cfg.joint_ids]).square(), dim=1)


def stair_joint_acceleration_cost(
    env,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    normalization_acceleration: float = 60.0,
) -> torch.Tensor:
    """Bounded leg acceleration cost; wheel collisions cannot dominate it."""
    asset = env.scene[asset_cfg.name]
    context = _stair_scan_context(env, sensor_cfg, command_name)
    acceleration = asset.data.joint_acc[:, asset_cfg.joint_ids]
    normalized = acceleration / max(normalization_acceleration, 1.0)
    return context["terrain_gate"] * torch.mean(normalized.square(), dim=1).clamp_max(1.0)


def stair_mechanical_power_cost(
    env,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    normalization_power: float = 1800.0,
) -> torch.Tensor:
    """Normalized absolute joint mechanical power while traversing stairs."""
    asset = env.scene[asset_cfg.name]
    context = _stair_scan_context(env, sensor_cfg, command_name)
    torque = asset.data.applied_torque[:, asset_cfg.joint_ids]
    velocity = asset.data.joint_vel[:, asset_cfg.joint_ids]
    power = torch.abs(torque * velocity).mean(dim=1) / max(normalization_power, 1.0)
    return context["terrain_gate"] * power




# =============================================================================
# 6. Phase-aware leg folding, compact turns and gait metrics
# =============================================================================

def pure_turn(command):
    return ((torch.linalg.vector_norm(command[:, :2], dim=1) < TUNING["translation_threshold"])
            & (command[:, 2].abs() > TUNING["yaw_threshold"]))


def turn_tracking_quality(yaw, command):
    """Stationary leg dancing or turning opposite to the command earns zero."""
    progress = (yaw * command.sign() / command.abs().clamp_min(TUNING["yaw_threshold"])).clamp(0, 1)
    return progress * torch.exp(-((yaw - command) / 0.5).square())


def bounded_air_score(air, low, target, high):
    """Time-valued triangle; zero for nonflight or prolonged flight."""
    peak = target - low
    return torch.minimum((air - low).clamp_min(0),
                         ((high - air) / (high - target) * peak).clamp_min(0))


def fold_excess(extension, knee, min_extension=None, max_knee=None):
    min_extension = TUNING["rear_min_extension"] if min_extension is None else min_extension
    max_knee = TUNING["rear_max_knee"] if max_knee is None else max_knee
    length = ((min_extension - extension).clamp_min(0)
              / TUNING["extension_scale"]).square()
    angle = ((knee.abs() - max_knee).clamp_min(0)
             / TUNING["knee_scale"]).square()
    return torch.maximum(length, angle).clamp_max(1)


def turn_size_excess(clearance, reach, terrain):
    height_limit = TUNING["turn_height_flat"] + terrain[:, None] * (
        TUNING["turn_height_terrain"] - TUNING["turn_height_flat"])
    reach_limit = TUNING["turn_reach_flat"] + terrain[:, None] * (
        TUNING["turn_reach_terrain"] - TUNING["turn_reach_flat"])
    return (((clearance - height_limit).clamp_min(0) / TUNING["height_scale"]).square()
            + ((reach - reach_limit).clamp_min(0) / TUNING["reach_scale"]).square()).clamp_max(1)


def _context(env, asset_cfg, hip_cfg, knee_cfg, sensor_cfg, contact_sensor_cfg,
             command_name="base_velocity"):
    step = int(env.common_step_counter)
    cached = getattr(env, "_m20_gait_context", None)
    if cached is not None and cached["step"] == step:
        return cached
    scan = _stair_scan_context(env, sensor_cfg, command_name)
    robot = env.scene[asset_cfg.name]
    contact = env.scene.sensors[contact_sensor_cfg.name]
    wheels = robot.data.body_pos_w[:, asset_cfg.body_ids]
    hips = robot.data.body_pos_w[:, hip_cfg.body_ids]
    hits = env.scene[sensor_cfg.name].data.ray_hits_w
    valid = torch.isfinite(hits).all(dim=-1)
    distance = torch.linalg.vector_norm(wheels[:, :, None, :2] - hits[:, None, :, :2], dim=-1)
    distance = torch.where(valid[:, None], distance, torch.inf)
    closest, index = distance.min(dim=-1)
    ground = hits[..., 2].gather(1, index)
    ground_valid = (closest < TUNING["nearest_valid_distance"]) & torch.isfinite(ground)
    # Missing terrain cannot turn into a fabricated large clearance reward/cost.
    ground = torch.where(ground_valid, ground, wheels[..., 2] - TUNING["wheel_radius"])
    force = contact.data.net_forces_w[:, contact_sensor_cfg.body_ids]
    airborne = contact.data.current_contact_time[:, contact_sensor_cfg.body_ids] <= 0
    support = ((force[..., 2] > 5.0)
               & (force[..., 2] > 0.5 * torch.linalg.vector_norm(force, dim=-1))
               & (contact.data.current_contact_time[:, contact_sensor_cfg.body_ids] >= TUNING["support_grace"]))
    clearance = wheels[..., 2] - ground - TUNING["wheel_radius"]
    support &= ground_valid & (clearance.abs() <= 0.05)
    command = env.command_manager.get_command(command_name)
    down = scan["down_gate"] > 0
    descent = getattr(env, "_m20_stair_step_state", None)
    if descent is not None:
        down |= descent["active"] & ~descent["up"] & (env.episode_length_buf > 1)
    down &= command[:, 0] > TUNING["translation_threshold"]
    # All visible upward edges include the one recently passed by the trunk.
    # An active target also keeps the gate through a temporary scan transition.
    rising_edges = scan.get("ascending_edges_valid")
    up = (scan["up_gate"] > 0) if rising_edges is None else rising_edges.any(dim=1)
    ascent = getattr(env, "_m20_ascent_state", None)
    if ascent is not None:
        up |= ascent["active"] & (env.episode_length_buf > 1)
    up &= (command[:, 0] > TUNING["translation_threshold"]) & ~down
    extension = torch.linalg.vector_norm(wheels - hips, dim=-1)
    reach = torch.linalg.vector_norm((wheels - hips)[..., :2], dim=-1)
    knee = robot.data.joint_pos[:, knee_cfg.joint_ids]
    turn = pure_turn(command)
    size = turn_size_excess(clearance, reach, scan["terrain_gate"])
    size = torch.where(ground_valid & airborne, size, 0.0)
    fold = torch.where(support[:, 2:] & down[:, None], fold_excess(extension[:, 2:], knee[:, 2:]), 0.0)
    front_support = (support[:, :2]
                     & (contact.data.current_contact_time[:, contact_sensor_cfg.body_ids][:, :2]
                        >= TUNING["front_support_grace"]) & up[:, None])
    front_fold = torch.where(front_support, fold_excess(
        extension[:, :2], knee[:, :2], TUNING["front_min_extension"], TUNING["front_max_knee"]), 0.0)
    front_swing = airborne[:, :2] & ground_valid[:, :2] & up[:, None]
    # Swing needs substantial flexion to clear a riser. Only the wider severe
    # envelope is charged, at a weaker rate than settled support.
    front_swing_fold = torch.where(front_swing, fold_excess(
        extension[:, :2], knee[:, :2], TUNING["front_swing_min_extension"],
        TUNING["front_swing_max_knee"]), 0.0)
    tracking = turn_tracking_quality(robot.data.root_ang_vel_b[:, 2], command[:, 2])
    cached = dict(step=step, turn=turn, tracking=tracking, support=support, airborne=airborne, clearance=clearance,
                  ground_valid=ground_valid, extension=extension, knee=knee, reach=reach,
                  size=size, fold=fold, down=down, up=up,
                  front_fold=front_fold, front_support=front_support,
                  front_swing=front_swing, front_swing_fold=front_swing_fold)
    env._m20_gait_context = cached
    # Once per control step, independently of how many reward terms use it.
    stats = getattr(env, "_m20_gait_stats", None)
    if stats is None:
        stats = {name: torch.zeros(env.num_envs, device=env.device) for name in
                 ("turn_steps", "turn_clearance_sum", "turn_reach_sum", "turn_size_sum",
                  "rear_support_samples", "rear_extension_sum", "rear_fold_samples",
                  "down_steps", "turn_yaw_error_sum", "up_steps", "front_support_samples",
                  "front_extension_sum", "front_knee_abs_sum", "front_fold_samples",
                  "front_swing_samples", "front_swing_fold_samples")}
        env._m20_gait_stats = stats
    reset = env.episode_length_buf <= 1
    for value in stats.values():
        value[reset] = 0
    valid_turn = turn & ground_valid.all(dim=1)
    stats["turn_steps"] += valid_turn.float()
    stats["turn_clearance_sum"] += torch.where(valid_turn, clearance.clamp_min(0).mean(dim=1), 0)
    stats["turn_reach_sum"] += torch.where(valid_turn, reach.mean(dim=1), 0)
    stats["turn_size_sum"] += size.mean(dim=1) * turn.float()
    stats["turn_yaw_error_sum"] += (robot.data.root_ang_vel_b[:, 2] - command[:, 2]).abs() * valid_turn.float()
    rear_samples = support[:, 2:] & down[:, None]
    stats["rear_support_samples"] += rear_samples.sum(dim=1)
    stats["rear_extension_sum"] += (extension[:, 2:] * rear_samples.float()).sum(dim=1)
    stats["rear_fold_samples"] += ((fold > 0) & rear_samples).sum(dim=1)
    stats["down_steps"] += down.float()
    stats["up_steps"] += up.float()
    stats["front_support_samples"] += front_support.sum(dim=1)
    stats["front_extension_sum"] += (extension[:, :2] * front_support.float()).sum(dim=1)
    stats["front_knee_abs_sum"] += (knee[:, :2].abs() * front_support.float()).sum(dim=1)
    stats["front_fold_samples"] += ((front_fold > 0) & front_support).sum(dim=1)
    stats["front_swing_samples"] += front_swing.sum(dim=1)
    stats["front_swing_fold_samples"] += ((front_swing_fold > 0) & front_swing).sum(dim=1)
    return cached


def turn_swing_size_cost(env, asset_cfg, hip_cfg, knee_cfg, sensor_cfg, contact_sensor_cfg):
    c = _context(env, asset_cfg, hip_cfg, knee_cfg, sensor_cfg, contact_sensor_cfg)
    return c["size"].mean(dim=1) * c["turn"].float()


def descent_rear_fold_cost(env, asset_cfg, hip_cfg, knee_cfg, sensor_cfg, contact_sensor_cfg):
    return _context(env, asset_cfg, hip_cfg, knee_cfg, sensor_cfg, contact_sensor_cfg)["fold"].mean(dim=1)


def ascent_front_fold_cost(env, asset_cfg, hip_cfg, knee_cfg, sensor_cfg, contact_sensor_cfg):
    """Bounded cost for excessive front flexion with separate support/swing envelopes.

    Normal swing and initial touchdown compliance are free; only severe swing
    flexion is charged. Commands and missing local ground close the gates.
    """
    c = _context(env, asset_cfg, hip_cfg, knee_cfg, sensor_cfg, contact_sensor_cfg)
    return (c["front_fold"] + TUNING["front_swing_scale"] * c["front_swing_fold"]).mean(dim=1)


def compact_turn_air_time(env, command_name, sensor_cfg, asset_cfg, threshold,
                          stair_sensor_cfg, cmd_threshold=0.1, foot_height_threshold=0.05):
    """Bound airtime at touchdown; moving forward cannot activate this term.

    Keep the inherited call signature so saved config parameters still resolve.
    Height shaping is handled by turn_swing_size_cost using local ground.
    """
    from .rewards import get_gait_level_tensor
    sensor = env.scene.sensors[sensor_cfg.name]
    first = sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    air = sensor.data.last_air_time[:, sensor_cfg.body_ids]
    command = env.command_manager.get_command(command_name)
    score = bounded_air_score(air, TUNING["air_min"], threshold, TUNING["air_max"])
    yaw = env.scene[asset_cfg.name].data.root_ang_vel_b[:, 2]
    tracking = turn_tracking_quality(yaw, command[:, 2])
    return (score * first).sum(dim=1) * pure_turn(command).float() * tracking * get_gait_level_tensor(env)


def compact_turn_status(env, asset_cfg, hip_cfg, knee_cfg, sensor_cfg, contact_sensor_cfg):
    """Only actual airborne diagonal pairs count, measured above local terrain."""
    from .rewards import get_gait_level_tensor
    c = _context(env, asset_cfg, hip_cfg, knee_cfg, sensor_cfg, contact_sensor_cfg)
    pattern = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    for grounded, lifted in (((0, 3), (1, 2)), ((1, 2), (0, 3))):
        lo = c["clearance"][:, lifted]
        airborne = c["airborne"][:, lifted]
        pattern |= (c["support"][:, grounded].all(dim=1) & airborne.all(dim=1)
                    & c["ground_valid"][:, lifted].all(dim=1)
                    & (lo > 0.02).all(dim=1) & (lo < TUNING["turn_height_terrain"]).all(dim=1))
    return pattern.float() * c["turn"].float() * c["tracking"] * get_gait_level_tensor(env)


def gait_quality_metrics(env, env_ids):
    stats = getattr(env, "_m20_gait_stats", None)
    if stats is None:
        return {"samples": torch.zeros((), device=env.device)}
    s = {key: value[env_ids].sum() for key, value in stats.items()}
    turns = s["turn_steps"].clamp_min(1)
    rear = s["rear_support_samples"].clamp_min(1)
    front = s["front_support_samples"].clamp_min(1)
    swing = s["front_swing_samples"].clamp_min(1)
    return {"turn_samples": s["turn_steps"], "descent_samples": s["down_steps"],
            "rear_support_samples": s["rear_support_samples"],
            "turn_clearance_mean_m": s["turn_clearance_sum"] / turns,
            "turn_reach_mean_m": s["turn_reach_sum"] / turns,
            "turn_yaw_error_mean": s["turn_yaw_error_sum"] / turns,
            "rear_extension_mean_m": s["rear_extension_sum"] / rear,
            "rear_fold_support_fraction": s["rear_fold_samples"] / rear,
            "ascent_samples": s["up_steps"], "front_support_samples": s["front_support_samples"],
            "front_extension_mean_m": s["front_extension_sum"] / front,
            "front_knee_abs_mean_rad": s["front_knee_abs_sum"] / front,
            "front_fold_support_fraction": s["front_fold_samples"] / front,
            "front_swing_samples": s["front_swing_samples"],
            "front_extreme_swing_fold_fraction": s["front_swing_fold_samples"] / swing}


# =============================================================================
# 7. Terrain curriculum and per-terrain level metrics
# =============================================================================

def stair_terrain_level_mean(env, env_ids, terrain_name: str) -> torch.Tensor:
    """Mean current row level for one terrain type, or the full population.

    Isaac Lab assigns terrain types to columns using cumulative proportions.
    Recreate that assignment from the active generator configuration so a
    type's curve follows the same environments through level changes.
    """
    terrain = env.scene.terrain
    levels = terrain.terrain_levels.float()
    if terrain_name == "overall":
        return levels.mean()

    generator = terrain.cfg.terrain_generator
    names = list(generator.sub_terrains)
    if terrain_name not in names:
        raise ValueError(f"Unknown terrain type for level logging: {terrain_name}")
    proportions = np.array(
        [generator.sub_terrains[name].proportion for name in names], dtype=np.float64
    )
    if proportions.sum() <= 0:
        raise RuntimeError("Cannot log terrain levels without positive terrain proportions")
    proportions /= proportions.sum()
    column_samples = np.arange(generator.num_cols) / generator.num_cols + 0.001
    column_type_ids = np.searchsorted(np.cumsum(proportions), column_samples, side="right")
    if np.any(column_type_ids >= len(names)):
        raise RuntimeError("Terrain type columns do not match Isaac Lab's generator proportions")

    columns = np.flatnonzero(column_type_ids == names.index(terrain_name))
    selected = torch.isin(
        terrain.terrain_types,
        torch.as_tensor(columns, device=levels.device, dtype=torch.long),
    )
    if not torch.any(selected):
        return levels.new_tensor(float("nan"))
    return levels[selected].mean()


def stair_distance_terrain_levels(
    env,
    env_ids,
    min_commanded_translation_m: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Apply the M20 distance curriculum requested for the stair course.

    Drop one level when displacement is below half a terrain tile; promote one
    level after reaching half the commanded episode distance. Promotion wins
    if the two thresholds overlap. Episodes with almost no translational
    command exposure keep their level.
    """
    asset = env.scene[asset_cfg.name]
    terrain = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")[env_ids]
    distance = torch.linalg.vector_norm(
        asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1
    )
    command_speed = torch.linalg.vector_norm(command[:, :2], dim=1)
    promotion_distance = command_speed * env.max_episode_length_s * 0.5
    command_term = env.command_manager.get_term("base_velocity")
    commanded_translation = command_term.episode_commanded_translation_m[env_ids]
    has_translation_opportunity = commanded_translation >= min_commanded_translation_m
    move_up = has_translation_opportunity & (command_speed > 0.1) & (distance >= promotion_distance)
    tile_half_length = terrain.cfg.terrain_generator.size[0] * 0.5
    move_down = has_translation_opportunity & (distance < tile_half_length) & ~move_up
    terrain.update_env_origins(env_ids, move_up, move_down)

    mean_level = torch.mean(terrain.terrain_levels.float())
    from .rewards import update_gait_level_from_terrain_mean

    update_gait_level_from_terrain_mean(mean_level)
    return mean_level
