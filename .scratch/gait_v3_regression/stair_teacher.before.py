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
    "front_support_grace": 0.0, "front_fold_weight": -3.0,
    "front_swing_min_extension": 0.22, "front_swing_max_knee": 2.50,
    "front_swing_scale": 0.50,
    "support_grace": 0.12, "same_tread_grace": 0.12,
    "wheel_radius": 0.09, "nearest_valid_distance": 0.12,
    "turn_size_weight": -1.0, "rear_fold_weight": -1.0,
    "same_tread_event_weight": -4.5, "same_tread_dwell_weight": -1.0,
    "turn_status_weight": 1.0, "turn_symmetry_weight": 2.0,
    "ascent_direction_weight": -2.0,
    "direction_lateral_tolerance": 0.10, "direction_lateral_scale": 0.25,
    "direction_speed_tolerance": 0.05, "direction_speed_scale": 0.20,
    "direction_heading_tolerance": np.deg2rad(5.0),
    "direction_heading_scale": np.deg2rad(15.0),
    "direction_velocity_mix": 0.5, "direction_heading_mix": 0.5,
    "direction_command_change": np.deg2rad(15.0),
    "direction_exit_inset": 0.06, "direction_exit_height_tolerance": 0.05,
    "direction_exit_contact_time": 0.06,
    # Persistent physical tread ledger; these are perception/style tolerances.
    "riser_identity_distance": 0.12, "riser_height_tolerance": 0.04,
    "riser_lateral_corridor": 0.45, "riser_min_spacing": 0.15,
    "riser_observation_half_width": 0.35,
    "tread_observation_corridor": 0.80,
    "riser_max_spacing": 0.50, "riser_cross_clearance": 0.01,
    "riser_default_half_width": 0.05,
    "tread_inset": 0.025, "tread_height_tolerance": 0.04,
    "landing_time": 0.04, "prep_distance": 0.60,
    "prep_weight": 0.5, "rear_completion_weight": 3.0,
    "front_touchdown_weight": 0.25,
    "front_a_support_low_deg": 90.0, "front_a_support_high_deg": 80.0,
    "front_a_swing_low_deg": 75.0, "front_a_swing_high_deg": 55.0,
    "pose_low_rise": 0.15, "pose_high_rise": 0.25,
    "pose_angle_scale": np.deg2rad(20.0), "pose_length_scale": 0.10,
    "pose_load_force_scale": 60.0,
    "rear_b_limit": np.deg2rad(35.0), "rear_g_limit": np.deg2rad(25.0),
    "rear_forward_relax": np.deg2rad(5.0), "rear_forward_weight": -2.0,
    "front_swing_min_extension_v3": 0.30,
    "front_swing_length_high": 0.26, "front_support_length_high": 0.33,
    "body_gap_target": 0.22, "body_gap_scale": 0.10,
    "body_bottom_z": -0.07, "body_gap_weight": -1.0,
    "pose_tracking_discount": 0.25,
    "pose_phase_radius": 1.50,

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
        "ascending_edges_lower_z": profile[:, :-1],
        "ascending_edges_half_width": 0.5 * (x_levels[1:] - x_levels[:-1]),
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
    state = {
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

    state.update({
        "map_count": torch.zeros(n,device=device,dtype=torch.long),
        "map_xy": zeros(64,2), "map_heading": zeros(64,2),
        "map_z": zeros(64), "map_lower_z": zeros(64),
        "map_next_x": torch.full((n,64),torch.inf,device=device),
        "map_intermediate": flags(64), "map_landed": flags(64,4),
        "map_safe": flags(64,4), "map_failed": flags(64,4),
        "map_clear": flags(64,4), "map_crossed": flags(64,4),
        "map_face_struck": flags(64,4), "map_half_width": zeros(64),
        "map_band_min": zeros(64), "map_band_max": zeros(64),
        "map_stable_time": zeros(64,4), "map_paid": flags(64,2),
        "map_correct": flags(64,2), "map_side": torch.full((n,64,2),-1,device=device,dtype=torch.long),
        "map_duplicate_paid": flags(64,2), "map_prep_max": zeros(64,2),
        "map_lead": torch.full((n,64,2),-1,device=device,dtype=torch.long),
        "map_dwell": zeros(64,2), "map_face_paid": flags(64,2),
        "map_flight": torch.full((n,64),-1,device=device,dtype=torch.long),
        "flight_count": torch.zeros(n,device=device,dtype=torch.long),
        "axis_last_slot": torch.full((n,2),-1,device=device,dtype=torch.long),
        "axis_target_slot": torch.full((n,2),-1,device=device,dtype=torch.long),
        "previous_wheel": zeros(4,3), "wheel_history_valid": flags(),
        "radius_calibrated": flags(4), "rear_lift_gain": zeros(),
        "rear_lift_credit": zeros(), "skip_total": zeros(2),
        "physical_total": zeros(2), "unsafe_total": zeros(2),
        "front_transition_successes": zeros(), "rear_transition_successes": zeros(),
        "axis_run": zeros(2), "axis_max_run": zeros(2),
        "duplicate_event_count": zeros(2),
    })
    state["radius"].fill_(TUNING["wheel_radius"])
    return state


def _reset(state, reset):
    if not reset.any():
        return
    for key, value in state.items():
        if key in {"radius"}:
            value[reset] = TUNING["wheel_radius"]
        elif key in {"front_expected", "rear_expected", "rear_target_index", "front_lift_lead",
                     "front_same_tread_paid", "rear_same_tread_paid", "same_tread_riser",
                     "map_side", "map_lead", "axis_last_slot", "axis_target_slot", "map_flight"}:
            value[reset] = -1
        elif key == "map_next_x":
            value[reset] = torch.inf
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


def _register_ascent_risers(state, scan, scanner, heading, command):
    """Observe every ascending riser; registration never depends on actor success."""
    n, device = heading.shape[0], heading.device
    row = torch.arange(n, device=device)
    xs = scan.get("ascending_edges_x", scan["edge_x"])
    if "ascending_edges_x" in scan:
        valid = scan["ascending_edges_valid"]
        upper = scan["ascending_edges_upper_z"]
        lower = scan["ascending_edges_lower_z"]
        xy = scanner.data.pos_w[:, None, :2] + heading[:, None] * xs[None, :, None]
    else:
        valid = (scan["up_gate"] > 0)[:, None]
        upper, lower = scan["upper_z"][:, None], scan["lower_z"][:, None]
        xy = scanner.data.pos_w[:, None, :2] + heading[:, None] * xs[:, None, None]
    for j in range(valid.shape[1]):
        enabled = valid[:, j] & (command[:, 0] > TUNING["translation_threshold"])
        delta = xy[:, j, None] - state["map_xy"]
        along = (delta * state["map_heading"]).sum(-1)
        lateral_signed = (-delta[...,0]*state["map_heading"][...,1]
                          +delta[...,1]*state["map_heading"][...,0])
        known = torch.arange(64, device=device)[None] < state["map_count"][:, None]
        aligned = (state["map_heading"] * heading[:, None]).sum(-1) > 0.8
        half_band = TUNING["riser_observation_half_width"]
        overlap_band = (lateral_signed-half_band <= state["map_band_max"]) & (
            lateral_signed+half_band >= state["map_band_min"])
        match = known & aligned & (along.abs() < TUNING["riser_identity_distance"]) & overlap_band & (
            (state["map_z"] - upper[:, j, None]).abs() < TUNING["riser_height_tolerance"]) & (
            (state["map_lower_z"]-lower[:,j,None]).abs() < TUNING["riser_height_tolerance"])
        add = enabled & ~match.any(-1) & (state["map_count"] < 64)
        slot = torch.where(match.any(-1), match.long().argmax(-1), state["map_count"]).clamp_max(63)
        ids = torch.nonzero(add, as_tuple=False).squeeze(-1)
        state["map_xy"][ids, slot[ids]] = xy[ids, j]
        state["map_heading"][ids, slot[ids]] = heading[ids]
        state["map_z"][ids, slot[ids]] = upper[ids, j]
        state["map_lower_z"][ids, slot[ids]] = lower[ids, j]
        state["map_band_min"][ids,slot[ids]] = -half_band
        state["map_band_max"][ids,slot[ids]] = half_band
        matched = enabled & match.any(-1)
        mid = torch.nonzero(matched,as_tuple=False).squeeze(-1)
        band_center = lateral_signed[mid,slot[mid]]
        state["map_band_min"][mid,slot[mid]] = torch.minimum(state["map_band_min"][mid,slot[mid]],band_center-half_band)
        state["map_band_max"][mid,slot[mid]] = torch.maximum(state["map_band_max"][mid,slot[mid]],band_center+half_band)
        half_width = scan.get("ascending_edges_half_width")
        state["map_half_width"][ids,slot[ids]] = (
            half_width[j] if half_width is not None else TUNING["riser_default_half_width"])
        state["map_count"][ids] += 1
        # A successor confirms an intermediate tread and bounds its upper end.
        # Update both directions so rear observations/registration order cannot
        # change tread identity. No N*64*64 pairwise allocation is needed.
        own_xy = state["map_xy"][row, slot]
        own_h = state["map_heading"][row, slot]
        own_z = state["map_z"][row, slot]
        delta = own_xy[:, None] - state["map_xy"]
        dist = (delta * state["map_heading"]).sum(-1)
        cross = (delta[..., 0] * state["map_heading"][..., 1]
                 - delta[..., 1] * state["map_heading"][..., 0]).abs()
        known = torch.arange(64, device=device)[None] < state["map_count"][:, None]
        parallel = (state["map_heading"] * own_h[:, None]).sum(-1) > 0.8
        pred = known & parallel & (dist >= TUNING["riser_min_spacing"]) & (
            dist <= TUNING["riser_max_spacing"]) & (cross < TUNING["riser_lateral_corridor"]) & (
            (state["map_z"] - lower[:, j, None]).abs() < TUNING["riser_height_tolerance"]) & enabled[:, None]
        state["map_next_x"] = torch.minimum(state["map_next_x"], torch.where(pred, dist, torch.inf))
        dz = state["map_lower_z"] - own_z[:, None]
        successor = known & parallel & (-dist >= TUNING["riser_min_spacing"]) & (
            -dist <= TUNING["riser_max_spacing"]) & (cross < TUNING["riser_lateral_corridor"]) & (
            dz.abs() < TUNING["riser_height_tolerance"]) & enabled[:, None]
        next_dist = torch.where(successor, -dist, torch.inf).amin(-1)
        state["map_next_x"][row, slot] = torch.minimum(state["map_next_x"][row, slot], next_dist)
        neighbors = pred | successor
        connected = neighbors.any(-1)
        neighbor_slot = neighbors.long().argmax(-1)
        new_flight = add & ~connected
        state["map_flight"][ids, slot[ids]] = torch.where(connected[ids],
            state["map_flight"][ids, neighbor_slot[ids]], state["flight_count"][ids])
        state["flight_count"] += new_flight.long()
    state["map_intermediate"] = torch.isfinite(state["map_next_x"])


def ascent_state(env, asset_cfg, sensor_cfg, contact_sensor_cfg, command_name="base_velocity"):
    """Physical tread ledger shared by two independently learning alternating axles.

    Every observed riser is registered before either axle's reward is considered.
    Stable landings, clearance, style, duplicate history and preparation budgets
    are separate records. Missing a tread advances physical progress without
    rewarding the skipped/wrong step or permanently trapping the next target.
    """
    step = int(env.common_step_counter)
    cache = getattr(env, "_m20_ascent_cache", None)
    if cache is not None and cache["step"] == step:
        return cache["state"]
    scan = _stair_scan_context(env, sensor_cfg, command_name)
    robot, scanner = env.scene[asset_cfg.name], env.scene[sensor_cfg.name]
    sensor = env.scene.sensors[contact_sensor_cfg.name]
    state = getattr(env, "_m20_ascent_state", None)
    if state is None or state["active"].shape[0] != env.num_envs:
        state = _new_state(env)
        env._m20_ascent_state = state
    reset = env.episode_length_buf <= 1
    _reset(state, reset)
    wheels = robot.data.body_pos_w[:, asset_cfg.body_ids]
    force = sensor.data.net_forces_w[:, contact_sensor_cfg.body_ids]
    norm = torch.linalg.vector_norm(force, dim=-1)
    loaded = (force[..., 2] > 5) & (force[..., 2] > 0.5 * norm)
    loaded &= sensor.data.current_contact_time[:, contact_sensor_cfg.body_ids] > 0
    command = env.command_manager.get_command(command_name)
    enabled = command[:, 0] > TUNING["translation_threshold"]
    row = torch.arange(env.num_envs, device=env.device)
    forward = torch.zeros(env.num_envs, 3, device=env.device); forward[:, 0] = 1
    heading = quat_apply(yaw_quat(scanner.data.quat_w), forward)[:, :2]
    _register_ascent_risers(state, scan, scanner, heading, command)
    known = torch.arange(64, device=env.device)[None] < state["map_count"][:, None]
    # Known model wheel radius is available immediately. Sensor-based estimates
    # only calibrate it on flat loaded support, never disable landing attribution.
    hits = scanner.data.ray_hits_w
    valid_hits = torch.isfinite(hits).all(-1)
    distance = torch.linalg.vector_norm(wheels[:, :, None, :2] - hits[:, None, :, :2], dim=-1)
    distance = torch.where(valid_hits[:, None], distance, torch.inf)
    near, nearest = distance.min(-1)
    surface = hits[..., 2].gather(1, nearest)
    measured = wheels[..., 2] - surface
    local = distance < 0.16
    lo = torch.where(local, hits[:, None, :, 2], torch.inf).amin(-1)
    hi = torch.where(local, hits[:, None, :, 2], -torch.inf).amax(-1)
    flat = (local.sum(-1) >= 2) & ((hi-lo) < 0.025)
    calibrate = ~state["radius_calibrated"] & loaded & flat & (near < 0.08) & (
        (measured - TUNING["wheel_radius"]).abs() < 0.025)
    state["radius"][calibrate] = measured[calibrate]
    state["radius_calibrated"] |= calibrate
    relative = wheels[:, None, :, :2] - state["map_xy"][:, :, None]
    x = (relative * state["map_heading"][:, :, None]).sum(-1)
    cross = (relative[..., 0] * state["map_heading"][:, :, None, 1]
             - relative[..., 1] * state["map_heading"][:, :, None, 0]).abs()
    prev = state["previous_wheel"][:, None, :, :2] - state["map_xy"][:, :, None]
    prev_x = (prev * state["map_heading"][:, :, None]).sum(-1)
    width = state["map_half_width"][:,:,None]
    history = state["wheel_history_valid"][:,None,None] & known[:,:,None]
    advance = x > prev_x
    overlap = history & advance & (prev_x <= width) & (x >= -width)
    # The two rays bracket the real riser. Preserve clearance observed anywhere
    # in that interval; a midpoint can be 5cm too early to see the final lift.
    def interval_height(at_x):
        fraction = ((at_x-prev_x)/(x-prev_x).clamp_min(1e-6)).clamp(0,1)
        return state["previous_wheel"][:,None,:,2]+fraction*(
            wheels[:,None,:,2]-state["previous_wheel"][:,None,:,2])
    high = torch.maximum(interval_height(torch.maximum(prev_x,-width)),
                         interval_height(torch.minimum(x,width)))
    clearance = high-state["map_z"][:,:,None]-state["radius"][:,None]
    state["map_clear"] |= overlap & (clearance >= TUNING["riser_cross_clearance"])
    crossing = history & (prev_x < width) & (x >= width)
    state["map_crossed"] |= crossing
    face_horizontal = torch.linalg.vector_norm(force[...,:2],dim=-1)
    face_all = history & (x >= -0.12) & (x <= width+0.03) & (
        wheels[:,None,:,2] < state["map_z"][:,:,None]+state["radius"][:,None]+0.05) & (
        face_horizontal[:,None]>50) & (face_horizontal[:,None]>1.5*force[:,None,:,2].clamp_min(0))
    state["map_face_struck"] |= face_all
    state["map_face_struck"] &= ~(x < -width-0.06)
    state["map_clear"] &= ~face_all
    safe_cross = state["map_clear"] & state["map_crossed"] & ~state["map_face_struck"]
    state["retry_count"] += (safe_cross & state["map_failed"]).sum((1,2))
    state["map_failed"] |= crossing & ~safe_cross & known[:, :, None]
    state["map_failed"] &= ~safe_cross
    state["map_safe"] |= safe_cross
    state["map_safe"] &= ~face_all
    # A ray pair bounds the real riser. Settle a landing only beyond that
    # interval, so an early upper-tread contact cannot consume its credit
    # before the completed crossing is observable.
    landing_boundary = torch.maximum(width, torch.full_like(width, TUNING["tread_inset"]))
    occupancy = known[:, :, None] & loaded[:, None] & (x > landing_boundary) & (
        x < state["map_next_x"][:, :, None] - TUNING["tread_inset"]) & (
        cross < TUNING["tread_observation_corridor"]) & (
        (wheels[:, None, :, 2] - state["map_z"][:, :, None] - state["radius"][:, None]).abs()
        <= TUNING["tread_height_tolerance"])
    state["map_stable_time"] = torch.where(occupancy, state["map_stable_time"] + env.step_dt, 0)
    stable = state["map_stable_time"] >= TUNING["landing_time"]
    state["map_landed"] |= stable
    for key in ("front_event", "rear_event", "front_face_event", "front_violation_event",
                "rear_violation_event", "front_same_tread_event", "rear_same_tread_event"):
        state[key].zero_()
    state["front_lift_gain"].zero_(); state["rear_lift_gain"].zero_()
    state["front_lift_settlement"].zero_()
    face_force = torch.linalg.vector_norm(force[:, :2, :2], dim=-1)
    face = known[:,:,None] & (x[:,:,:2] >= -0.12) & (x[:,:,:2] <= 0.03) & (
        wheels[:,None,:2,2] < state["map_z"][:,:,None] + state["radius"][:,None,:2] + 0.05) & (
        face_force[:,None] > 50) & (face_force[:,None] > 1.5*force[:,None,:2,2].clamp_min(0))
    new_face = face & ~state["map_face_paid"] & enabled[:,None,None]
    state["front_face_event"] = new_face.any(1)
    state["map_face_paid"] |= new_face
    for axle, name in enumerate(("front", "rear")):
        sl = slice(2*axle, 2*axle+2)
        pair = state["map_landed"][:, :, sl]
        first = pair.any(-1) & ~state["map_paid"][:, :, axle] & known
        candidate = torch.where(first, state["map_z"], torch.inf).argmin(-1)
        available = first.any(-1) & enabled
        landed = pair[row, candidate]
        single = landed.sum(-1) == 1
        side = landed.long().argmax(-1)
        last = state["axis_last_slot"][:, axle]
        candidate_flight = state["map_flight"][row, candidate]
        new_flight = available & (last >= 0) & (
            candidate_flight != state["map_flight"][row,last.clamp_min(0)])
        last = torch.where(new_flight, -1, last)
        state[name+"_expected"][new_flight] = -1
        state["axis_run"][new_flight,axle] = 0
        last_z = state["map_z"][row, last.clamp_min(0)]
        last_xy = state["map_xy"][row, last.clamp_min(0)]
        last_h = state["map_heading"][row, last.clamp_min(0)]
        advance = ((state["map_xy"][row, candidate] - last_xy) * last_h).sum(-1)
        source = state["map_lower_z"][row, candidate]
        adjacent = (last < 0) | ((source-last_z).abs() < TUNING["riser_height_tolerance"]) & (
            advance >= TUNING["riser_min_spacing"]) & (advance <= TUNING["riser_max_spacing"])
        # If a first observed landing skips a known lower tread, it is a skip,
        # including when neither front leg ever paid a completion reward.
        lower_before = known & (state["map_z"] < state["map_z"][row, candidate, None] - 0.05)
        lower_before &= state["map_flight"] == candidate_flight[:,None]
        lower_before &= (last < 0)[:,None] | (state["map_z"] > last_z[:,None]+0.05)
        lower_unvisited = (lower_before & ~state["map_paid"][:, :, axle]).any(-1)
        skipped = (~adjacent | lower_unvisited) & available
        expected = state[name+"_expected"]
        ordered = (expected < 0) | (side == expected)
        safe = state["map_safe"][row, candidate, sl].gather(1, side[:, None]).squeeze(-1)
        correct = available & single & ordered & safe & ~skipped
        violation = available & ~correct
        previous = last.clamp_min(0)
        previous_side = state["map_side"][row,previous,axle]
        previous_pair = state["map_landed"][row,previous,sl]
        previous_safe = state["map_safe"][row,previous,sl].gather(
            1,previous_side.clamp_min(0)[:,None]).squeeze(-1)
        # Pay the main style reward for a demonstrated alternating transition,
        # rather than paying the first foot before its partner's plan is known.
        # This prevents HL1,HR1,HL2,HR2 from collecting a completion per tread
        # and only facing a temporally discounted same-tread penalty later.
        transition = correct & (last >= 0) & (previous_side >= 0) & (
            previous_pair.sum(-1)==1) & previous_safe & (side!=previous_side) & (
            ~state["map_duplicate_paid"][row,previous,axle])
        state[name+"_event"] = landed & transition[:, None]
        state[name+"_transition_successes"] += transition.float()
        state[name+"_violation_event"] = violation
        state[name+"_successes"] += correct.float()
        state[name+"_violations"] += violation.float()
        state["skip_total"][:, axle] += skipped.float()
        state["physical_total"][:, axle] += available.float()
        state["unsafe_total"][:, axle] += (available & ~safe).float()
        state["axis_run"][:, axle] = torch.where(available,
            torch.where(correct, state["axis_run"][:, axle]+1, 0), state["axis_run"][:, axle])
        state["axis_max_run"][:, axle] = torch.maximum(state["axis_max_run"][:, axle], state["axis_run"][:, axle])
        ids = torch.nonzero(available, as_tuple=False).squeeze(-1)
        slots = candidate[ids]
        state["map_paid"][ids, slots, axle] = True
        state["map_correct"][ids, slots, axle] = correct[ids]
        state["map_side"][ids, slots, axle] = torch.where(single[ids], side[ids], -1)
        # Physical progress is monotonic; late lower contacts cannot rewind the
        # sequence or unlock more preparation credit.
        further = available & ((last < 0) | (state["map_z"][row, candidate] > last_z+0.05))
        state["axis_last_slot"][further, axle] = candidate[further]
        state[name+"_expected"][further] = torch.where(single[further], 1-side[further], -1)
        state[name+"_count"] += available.long()
        # All historical treads remain checked, including after the first leg
        # has left. Each axle/tread pays once, and both axles add independently.
        duplicate = pair.all(-1) & state["map_intermediate"] & known
        new_duplicate = duplicate & ~state["map_duplicate_paid"][:, :, axle] & enabled[:, None]
        state["map_duplicate_paid"][:, :, axle] |= new_duplicate
        state["map_correct"][:, :, axle] &= ~duplicate
        state[name+"_same_tread_event"] = new_duplicate.any(-1)
        state["duplicate_event_count"][:, axle] = new_duplicate.sum(-1)
        state["same_tread_events_total"][:, axle] += new_duplicate.sum(-1)
        state["axis_run"][new_duplicate.any(-1), axle] = 0
        current_last = state["axis_last_slot"][:,axle]
        ambiguous_pair = (current_last >= 0) & new_duplicate[row,current_last.clamp_min(0)]
        # After both feet occupy one tread, either may restart the sequence.
        # Requiring an invisible historical leader would make a memoryless
        # actor's identical current posture receive contradictory targets.
        state[name+"_expected"][ambiguous_pair] = -1
        current_pair = stable[:, :, sl].all(-1) & state["map_intermediate"] & known & enabled[:, None]
        state["map_dwell"][:, :, axle] += current_pair.float()*env.step_dt
        dwelling = current_pair & (state["map_dwell"][:, :, axle] > TUNING["same_tread_grace"])
        state["same_tread_dwell"][:, axle] = dwelling.any(-1).float()
        state["same_tread_seconds_total"][:, axle] += dwelling.any(-1)*env.step_dt
        # Geometric next target per axle, independent of the other axle queue.
        last = state["axis_last_slot"][:, axle]
        last_z = state["map_z"][row, last.clamp_min(0)]
        last_flight = state["map_flight"][row,last.clamp_min(0)]
        same_flight = state["map_flight"] == last_flight[:,None]
        wheel_x = x[:,:,sl].mean(-1)
        new_flight_ahead = ~same_flight & (wheel_x > -TUNING["prep_distance"]) & (wheel_x < 0.03)
        ahead = known & ~state["map_paid"][:, :, axle] & (
            (last < 0)[:, None] | (same_flight & (state["map_z"] > last_z[:, None]+0.05)) | new_flight_ahead)
        target = torch.where(ahead, state["map_z"], torch.inf).argmin(-1)
        has_target = ahead.any(-1)
        changed = has_target & (target != state["axis_target_slot"][:, axle])
        state["targets_started"] += changed.float() if axle == 0 else 0
        state["axis_target_slot"][:, axle] = torch.where(has_target, target, -1)
        expected = state[name+"_expected"]
        lead = state["map_lead"][row, target, axle]
        lead = torch.where(expected >= 0, expected, lead)
        hfrac = (wheels[:, sl, 2] - state["radius"][:, sl] - state["map_lower_z"][row, target, None]) / (
            state["map_z"][row, target, None] - state["map_lower_z"][row, target, None] + 0.02).clamp_min(0.06)
        tx = x[row, target, sl]
        progress = hfrac.clamp(0,1) * ((tx+TUNING["prep_distance"])/TUNING["prep_distance"]).clamp(0,1)
        other_support = loaded.sum(-1,keepdim=True) - loaded[:, sl].long()
        eligible = has_target[:, None] & enabled[:, None] & ~loaded[:, sl] & (other_support >= 2) & (
            tx > -TUNING["prep_distance"]) & (tx < 0.03)
        progress = torch.where(eligible, progress, 0)
        lead = torch.where(lead >= 0, lead, progress.argmax(-1))
        current = progress.gather(1,lead[:, None]).squeeze(-1)
        gain = (current-state["map_prep_max"][row, target, axle]).clamp_min(0)
        budget = (1+state[name+"_successes"]-state[name+"_lift_credit"]).clamp_min(0)
        gain = torch.minimum(gain,budget)*has_target
        state["map_lead"][row, target, axle] = torch.where(gain > 0,lead,state["map_lead"][row,target,axle])
        state["map_prep_max"][row,target,axle] = torch.maximum(state["map_prep_max"][row,target,axle],current)
        state[name+"_lift_gain"] = gain
        state[name+"_lift_credit"] += gain
    state["previous_wheel"] = wheels.clone()
    state["wheel_history_valid"].fill_(True)
    # Retain unpaid tread history on departure, but do not apply stair posture
    # shaping on unrelated flat terrain. Rear feet can keep the phase active
    # after the forward scan has cleared, while an actual lateral exit cannot.
    nearby = known[:,:,None] & (cross < TUNING["tread_observation_corridor"]) & (
        x > -TUNING["prep_distance"]) & (
        x < torch.where(torch.isfinite(state["map_next_x"]),
                        state["map_next_x"],TUNING["riser_max_spacing"])[:,:,None]+0.20) & (
        wheels[:,None,:,2]-state["radius"][:,None] >= state["map_lower_z"][:,:,None]-0.10) & (
        wheels[:,None,:,2]-state["radius"][:,None] <= state["map_z"][:,:,None]+0.50)
    state["active"] = enabled & (state["axis_target_slot"] >= 0).any(-1) & nearby.any((1,2))
    target = torch.where(state["axis_target_slot"][:,0]>=0,state["axis_target_slot"][:,0],
        torch.where(state["axis_target_slot"][:,1]>=0,state["axis_target_slot"][:,1],state["axis_last_slot"][:,0])).clamp_min(0)
    state["edge_xy"] = state["map_xy"][row,target].clone()
    state["heading"] = state["map_heading"][row,target].clone()
    state["source_z"] = state["map_lower_z"][row,target].clone()
    state["target_z"] = state["map_z"][row,target].clone()
    state["forward_steps"] += state["active"].float()
    state["retreat_steps"] += (state["active"] & (robot.data.root_lin_vel_b[:,0] < -0.05)).float()
    state["stationary_steps"] += (state["active"] & (robot.data.root_lin_vel_b[:,0].abs() < 0.05)).float()
    env._m20_ascent_cache = {"step":step,"state":state}
    return state


def stair_rear_wheel_lift(env, asset_cfg, sensor_cfg, contact_sensor_cfg, command_name="base_velocity"):
    """One bounded preparation budget per rear target; no cyclic leg-dance pay."""
    return ascent_state(env,asset_cfg,sensor_cfg,contact_sensor_cfg,command_name)["rear_lift_gain"] / env.step_dt




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


def _ascent_direction_context(env, asset_cfg, sensor_cfg, contact_sensor_cfg,
                              command_name="base_velocity"):
    """Track ascent drift against a command-integrated heading, once per step.

    Current body yaw must not redefine the desired route: a policy can otherwise
    turn and drive sideways while retaining perfect body-frame velocity tracking.
    Cross-track increments use the previous commanded path tangent, so slowing
    down on a stair does not create an along-track position error penalty.
    """
    step = int(env.common_step_counter)
    cached = getattr(env, "_m20_ascent_direction_cache", None)
    if cached is not None and cached["step"] == step:
        return cached
    robot = env.scene[asset_cfg.name]
    scanner = env.scene[sensor_cfg.name]
    sensor = env.scene.sensors[contact_sensor_cfg.name]
    scan = _stair_scan_context(env, sensor_cfg, command_name)
    command = env.command_manager.get_command(command_name)
    root_xy = robot.data.root_pos_w[:, :2]
    forward = torch.zeros(env.num_envs, 3, device=env.device)
    forward[:, 0] = 1.0
    body_heading = quat_apply(yaw_quat(robot.data.root_quat_w), forward)[:, :2]
    actual_yaw = torch.atan2(body_heading[:, 1], body_heading[:, 0])
    moving_up = command[:, 0] > TUNING["translation_threshold"]
    state = getattr(env, "_m20_ascent_direction_state", None)
    if state is None or state["engaged"].shape[0] != env.num_envs:
        zero = lambda *shape: torch.zeros(env.num_envs, *shape, device=env.device)
        state = dict(engaged=zero().bool(), reference_yaw=actual_yaw.clone(),
                     previous_xy=root_xy.clone(), previous_command=command.clone(),
                     cross_track=zero(), edge_xy=zero(2), edge_heading=zero(2),
                     top_z=zero(), stats={key: zero() for key in
                         ("samples", "lateral_sum", "speed_sum", "heading_sum",
                          "outside_samples", "cost_sum", "max_lateral")})
        env._m20_ascent_direction_state = state
    reset = env.episode_length_buf <= 1
    state["engaged"][reset] = False
    state["cross_track"][reset] = 0.0
    for value in state["stats"].values():
        value[reset] = 0.0

    # Remember the furthest/highest ascending riser. A turned-away or missing
    # scan cannot clear the latch. Only four loaded wheels beyond the known
    # riser on its upper platform can end a flight.
    edges_valid = scan["ascending_edges_valid"]
    seen_up = edges_valid.any(dim=1)
    heights = torch.where(edges_valid, scan["ascending_edges_upper_z"], -torch.inf)
    top_z, edge_i = heights.max(dim=1)
    start = ~state["engaged"] & seen_up & moving_up
    update_edge = seen_up & (start | (state["engaged"] & (top_z > state["top_z"] + 0.02)))
    scanner_heading = quat_apply(yaw_quat(scanner.data.quat_w), forward)[:, :2]
    edge_x = scan["ascending_edges_x"][edge_i]
    state["edge_xy"][update_edge] = (scanner.data.pos_w[:, :2]
        + scanner_heading * edge_x[:, None])[update_edge]
    state["edge_heading"][update_edge] = scanner_heading[update_edge]
    state["top_z"][update_edge] = top_z[update_edge]
    previous = state["previous_command"]
    reference = state["reference_yaw"]
    mid_yaw = reference + 0.5 * previous[:, 2] * env.step_dt
    tangent = torch.stack((torch.cos(mid_yaw) * previous[:, 0] - torch.sin(mid_yaw) * previous[:, 1],
                           torch.sin(mid_yaw) * previous[:, 0] + torch.cos(mid_yaw) * previous[:, 1]), -1)
    tangent /= torch.linalg.vector_norm(tangent, dim=1).clamp_min(1e-6)[:, None]
    normal = torch.stack((-tangent[:, 1], tangent[:, 0]), -1)
    integrate = state["engaged"] & (previous[:, 0] > TUNING["translation_threshold"]) & ~reset
    state["cross_track"] += ((root_xy - state["previous_xy"]) * normal).sum(-1) * integrate
    reference = reference + previous[:, 2] * env.step_dt * state["engaged"]
    reference = torch.atan2(torch.sin(reference), torch.cos(reference))
    # User changes in travel direction rebase accumulated cross-track error;
    # actual body yaw changes alone never do so. Finishing a commanded in-place
    # turn establishes the new body-relative forward direction.
    old_norm = torch.linalg.vector_norm(previous[:, :2], dim=1)
    new_norm = torch.linalg.vector_norm(command[:, :2], dim=1)
    similarity = (previous[:, :2] * command[:, :2]).sum(-1) / (old_norm * new_norm).clamp_min(1e-6)
    direction_change = (old_norm > 0.1) & (new_norm > 0.1) & (
        similarity < np.cos(TUNING["direction_command_change"]))
    after_turn = (old_norm <= TUNING["translation_threshold"]) & (
        previous[:, 2].abs() > TUNING["yaw_threshold"]) & moving_up
    reference = torch.where(start | after_turn, actual_yaw, reference)
    state["cross_track"][start | direction_change | after_turn] = 0.0
    state["engaged"] |= start
    state["reference_yaw"] = reference

    wheels = robot.data.body_pos_w[:, asset_cfg.body_ids]
    forces = sensor.data.net_forces_w[:, contact_sensor_cfg.body_ids]
    loaded = (forces[..., 2] > 5.0) & (
        forces[..., 2] > 0.5 * torch.linalg.vector_norm(forces, dim=-1))
    loaded &= sensor.data.current_contact_time[:, contact_sensor_cfg.body_ids] >= TUNING["direction_exit_contact_time"]
    past = ((wheels[..., :2] - state["edge_xy"][:, None]) * state["edge_heading"][:, None]).sum(-1)
    on_top = (wheels[..., 2] - state["top_z"][:, None] - TUNING["wheel_radius"]).abs() <= TUNING["direction_exit_height_tolerance"]
    clear = state["engaged"] & ~seen_up & loaded.all(-1) & on_top.all(-1) & (
        past > TUNING["direction_exit_inset"]).all(-1)
    heading_error = torch.atan2(torch.sin(actual_yaw - reference), torch.cos(actual_yaw - reference))
    # A commanded turn leading onto descending terrain also ends ascent shaping;
    # an uncommanded turn away from the scan must retain its cost.
    clear |= state["engaged"] & (scan["down_gate"] > 0) & ~seen_up & (
        heading_error.abs() < TUNING["direction_heading_tolerance"])
    state["engaged"][clear] = False
    path = torch.stack((torch.cos(reference) * command[:, 0] - torch.sin(reference) * command[:, 1],
                        torch.sin(reference) * command[:, 0] + torch.cos(reference) * command[:, 1]), -1)
    path /= new_norm.clamp_min(1e-6)[:, None]
    velocity = robot.data.root_lin_vel_w[:, :2]
    lateral_speed = -velocity[:, 0] * path[:, 1] + velocity[:, 1] * path[:, 0]
    active = state["engaged"] & moving_up
    def excess(value, tolerance, scale):
        x = ((value.abs() - tolerance).clamp_min(0.0) / scale).square()
        return x / (1.0 + x)
    cost = excess(state["cross_track"], TUNING["direction_lateral_tolerance"], TUNING["direction_lateral_scale"])
    cost += TUNING["direction_velocity_mix"] * excess(lateral_speed, TUNING["direction_speed_tolerance"], TUNING["direction_speed_scale"])
    cost += TUNING["direction_heading_mix"] * excess(heading_error, TUNING["direction_heading_tolerance"], TUNING["direction_heading_scale"])
    cost *= active
    stats = state["stats"]
    stats["samples"] += active
    stats["lateral_sum"] += state["cross_track"].abs() * active
    stats["speed_sum"] += lateral_speed.abs() * active
    stats["heading_sum"] += heading_error.abs() * active
    stats["outside_samples"] += (state["cross_track"].abs() > TUNING["direction_lateral_tolerance"]) & active
    stats["cost_sum"] += cost
    stats["max_lateral"] = torch.maximum(stats["max_lateral"], state["cross_track"].abs() * active)
    state["previous_xy"] = root_xy.clone()
    state["previous_command"] = command.clone()
    cached = dict(step=step, cost=cost, active=active, cross_track=state["cross_track"].clone(),
                  lateral_speed=lateral_speed, heading_error=heading_error)
    env._m20_ascent_direction_cache = cached
    return cached


def stair_ascent_direction_cost(env, asset_cfg, sensor_cfg, contact_sensor_cfg,
                                command_name="base_velocity"):
    """Bounded ascent cost for cross-track drift, lateral speed and unwanted yaw."""
    return _ascent_direction_context(env, asset_cfg, sensor_cfg, contact_sensor_cfg, command_name)["cost"]


def ascent_direction_metrics(env, env_ids):
    state = getattr(env, "_m20_ascent_direction_state", None)
    if state is None:
        return {"samples": torch.zeros((), device=env.device)}
    s = {key: value[env_ids].sum() for key, value in state["stats"].items()}
    count = s["samples"].clamp_min(1)
    return dict(samples=s["samples"], lateral_error_mean_m=s["lateral_sum"] / count,
                lateral_error_max_m=state["stats"]["max_lateral"][env_ids].max(),
                lateral_speed_mean_m_s=s["speed_sum"] / count,
                heading_error_mean_deg=torch.rad2deg(s["heading_sum"] / count),
                outside_tolerance_fraction=s["outside_samples"] / count,
                cost_mean=s["cost_sum"] / count)


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
    return (state["front_violation_event"].float() + state["rear_violation_event"].float()) / env.step_dt


def stair_up_same_tread_cost(
    env, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Cost once per riser if both front or both rear wheels share its tread."""
    state = ascent_state(env, asset_cfg, sensor_cfg, contact_sensor_cfg)
    return state["duplicate_event_count"].sum(-1) / env.step_dt


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
    hip_cfg=None, knee_cfg=None, contact_sensor_cfg=None,
) -> torch.Tensor:
    """Retain most native tracking; unsafe posture forfeits a bounded fraction.

    Do not suppress driving throughout an entire swing/target or require a
    periodic gait clock. Legacy saved calls without gait entities stay native.
    """
    from .rewards import track_lin_vel_xy_exp
    native = track_lin_vel_xy_exp(env,std,command_name,asset_cfg)
    if hip_cfg is None or knee_cfg is None or contact_sensor_cfg is None:
        return native
    c = _context(env,asset_cfg,hip_cfg,knee_cfg,sensor_cfg,contact_sensor_cfg,command_name)
    severity = torch.maximum((c["front_fold"]+c["front_swing_fold"]).mean(-1),c["rear_forward"].mean(-1))
    severity = torch.maximum(severity,c["body_gap"])
    return native*(1-TUNING["pose_tracking_discount"]*severity)


def _legacy_step_completion_metrics(env, env_ids) -> dict[str, torch.Tensor]:
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
    down_follow = torch.zeros(env.num_envs,dtype=torch.bool,device=env.device)
    if descent is not None:
        close = torch.linalg.vector_norm(wheels[:,:,:2]-descent["edge_xy"][:,None],dim=-1).amin(-1) < TUNING["pose_phase_radius"]
        down_follow = descent["active"] & ~descent["up"] & (env.episode_length_buf > 1) & close
        down |= down_follow
    down &= command[:, 0] > TUNING["translation_threshold"]
    # All visible upward edges include the one recently passed by the trunk.
    # An active target also keeps the gate through a temporary scan transition.
    rising_edges = scan.get("ascending_edges_valid")
    up = (scan["up_gate"] > 0) if rising_edges is None else rising_edges.any(dim=1)
    ascent = ascent_state(env,asset_cfg,sensor_cfg,contact_sensor_cfg,command_name)
    if ascent is not None:
        up |= ascent["active"] & (env.episode_length_buf > 1)
    up &= (command[:, 0] > TUNING["translation_threshold"]) & ~down
    extension = torch.linalg.vector_norm(wheels - hips, dim=-1)
    reach = torch.linalg.vector_norm((wheels - hips)[..., :2], dim=-1)
    knee = robot.data.joint_pos[:, knee_cfg.joint_ids]
    turn = pure_turn(command)
    size = turn_size_excess(clearance, reach, scan["terrain_gate"])
    size = torch.where(ground_valid & airborne, size, 0.0)
    # Force-supported posture is measurable without a terrain-nearest point.
    # Ground validity remains required only by ground-relative turn/clearance.
    load = (force[...,2] > 5) & (force[...,2] > 0.5*torch.linalg.vector_norm(force,dim=-1)) & ~airborne
    load_gain = (force[...,2] / TUNING["pose_load_force_scale"]).clamp(0,1)
    fold = torch.where(load[:,2:] & down[:,None], fold_excess(extension[:,2:],knee[:,2:]),0)
    rise = scan["edge_delta"].abs()
    if ascent is not None:
        rise = torch.maximum(rise,torch.where(ascent["active"],
            (ascent["target_z"]-ascent["source_z"]).abs(),0))
    if descent is not None:
        rise = torch.maximum(rise,torch.where(down_follow,
            (descent["target_z"]-descent["source_z"]).abs(),0))
    relax = ((rise-TUNING["pose_low_rise"]) / (TUNING["pose_high_rise"]-TUNING["pose_low_rise"])).clamp(0,1)
    a_support = TUNING["front_a_support_low_deg"] + relax*(TUNING["front_a_support_high_deg"]-TUNING["front_a_support_low_deg"])
    a_swing = TUNING["front_a_swing_low_deg"] + relax*(TUNING["front_a_swing_high_deg"]-TUNING["front_a_swing_low_deg"])
    front_a = (torch.pi-knee[:,:2].abs()).clamp_min(0)
    front_support = load[:,:2] & up[:,None]
    # Face brushing cannot pretend to be valid support and escape the wider
    # swing envelope; no initial 0.2s free deep fold and no nearest-map veto.
    front_swing = ~load[:,:2] & up[:,None]
    def bounded(value):
        square = value.clamp_min(0).square()
        return square/(1+square)
    front_loaded_angle = bounded((torch.deg2rad(a_support)[:,None]-front_a)/TUNING["pose_angle_scale"])
    front_air_angle = bounded((torch.deg2rad(a_swing)[:,None]-front_a)/TUNING["pose_angle_scale"])
    support_length = TUNING["front_min_extension"]+relax*(TUNING["front_support_length_high"]-TUNING["front_min_extension"])
    swing_length = TUNING["front_swing_min_extension_v3"]+relax*(TUNING["front_swing_length_high"]-TUNING["front_swing_min_extension_v3"])
    front_loaded_length = bounded((support_length[:,None]-extension[:,:2])/TUNING["pose_length_scale"])
    front_air_length = bounded((swing_length[:,None]-extension[:,:2])/TUNING["pose_length_scale"])
    # Use the worse of angle/length, since both describe the same flexion.
    front_fold = torch.maximum(front_loaded_angle,front_loaded_length)*front_support*load_gain[:,:2]
    front_swing_fold = torch.maximum(front_air_angle,front_air_length)*front_swing
    delta = wheels-hips
    qb = robot.data.root_quat_w[:,None].expand(-1,4,-1).reshape(-1,4)
    body_delta = quat_apply_inverse(qb,delta.reshape(-1,3)).reshape(-1,4,3)
    qy = yaw_quat(robot.data.root_quat_w)[:,None].expand(-1,4,-1).reshape(-1,4)
    gravity_delta = quat_apply_inverse(qy,delta.reshape(-1,3)).reshape(-1,4,3)
    body_angle = torch.atan2(body_delta[...,0],(-body_delta[...,2]).clamp_min(1e-6))
    gravity_angle = torch.atan2(gravity_delta[...,0],(-gravity_delta[...,2]).clamp_min(1e-6))
    rear_limit = relax[:,None]*TUNING["rear_forward_relax"]
    excess_b = (body_angle[:,2:]-TUNING["rear_b_limit"]-rear_limit)/TUNING["pose_angle_scale"]
    excess_g = (gravity_angle[:,2:]-TUNING["rear_g_limit"]-rear_limit)/TUNING["pose_angle_scale"]
    rear_load = load[:,2:] & down[:,None]
    rear_forward = bounded(torch.maximum(excess_b,excess_g))*rear_load*load_gain[:,2:]
    # A gravity-aligned safety lower envelope is a height-map proxy, not an
    # exact collision mesh distance. It alone requires local map validity.
    points = wheels.new_tensor([[x,y,TUNING["body_bottom_z"]] for x in (0.10,0.22,0.34) for y in (-0.16,0,0.16)])
    count = points.shape[0]
    q = robot.data.root_quat_w[:,None].expand(-1,count,-1).reshape(-1,4)
    world = quat_apply(q,points[None].expand(env.num_envs,-1,-1).reshape(-1,3)).reshape(env.num_envs,count,3)+robot.data.root_pos_w[:,None]
    distances = torch.linalg.vector_norm(world[:,:,None,:2]-hits[:,None,:,:2],dim=-1)
    local = valid[:,None] & (distances < TUNING["nearest_valid_distance"])
    floor = torch.where(local,hits[:,None,:,2],-torch.inf).amax(-1)
    gap_valid = local.any(-1).any(-1) & up
    gap = torch.where(local.any(-1),world[...,2]-floor,torch.inf).amin(-1)
    body_gap = torch.where(gap_valid,bounded((TUNING["body_gap_target"]-gap)/TUNING["body_gap_scale"]),0)
    _update_pose_stats(env,up,down,front_a,front_support,front_swing,front_loaded_angle,front_air_angle,
                       body_angle[:,2:],gravity_angle[:,2:],rear_load,rear_forward,gap,gap_valid,load,ground_valid)
    tracking = turn_tracking_quality(robot.data.root_ang_vel_b[:, 2], command[:, 2])
    cached = dict(step=step, turn=turn, tracking=tracking, support=support, airborne=airborne, clearance=clearance,
                  ground_valid=ground_valid, extension=extension, knee=knee, reach=reach,
                  size=size, fold=fold, down=down, up=up,
                  front_fold=front_fold, front_support=front_support,
                  front_swing=front_swing, front_swing_fold=front_swing_fold,
                  rear_forward=rear_forward, body_angle=body_angle, gravity_angle=gravity_angle,
                  body_gap=body_gap, body_gap_m=gap, body_gap_valid=gap_valid, load=load, rear_load=rear_load)
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
    rear_samples = load[:, 2:] & down[:, None]
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

    Support and swing have height-adaptive soft envelopes. Genuine vertical
    load activates posture immediately, independently of nearest-map validity.
    """
    c = _context(env, asset_cfg, hip_cfg, knee_cfg, sensor_cfg, contact_sensor_cfg)
    return (c["front_fold"] + c["front_swing_fold"]).mean(dim=1)


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


def _legacy_gait_quality_metrics(env, env_ids):
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


def descent_rear_forward_cost(env, asset_cfg, hip_cfg, knee_cfg, sensor_cfg, contact_sensor_cfg):
    """Penalize excess rear forward reach while actually bearing vertical load."""
    return _context(env,asset_cfg,hip_cfg,knee_cfg,sensor_cfg,contact_sensor_cfg)["rear_forward"].mean(-1)


def ascent_front_body_clearance_cost(env, asset_cfg, hip_cfg, knee_cfg, sensor_cfg, contact_sensor_cfg):
    """Map-based front belly clearance; invalid cells cannot fabricate a cost."""
    return _context(env,asset_cfg,hip_cfg,knee_cfg,sensor_cfg,contact_sensor_cfg)["body_gap"]


def _update_pose_stats(env,up,down,front_a,front_load,front_air,loaded_cost,air_cost,
                       rear_b,rear_g,rear_load,rear_cost,gap,gap_valid,load,ground_valid):
    names = ("front_loaded_samples","front_loaded_a_sum","front_loaded_bad",
             "front_swing_samples","front_swing_a_sum","front_swing_bad",
             "rear_loaded_samples","rear_b_sum","rear_g_sum","rear_bad",
             "rear_entry_samples","rear_entry_b_sum","rear_entry_g_sum")
    state = getattr(env,"_m20_pose_stats",None)
    if state is None:
        state = {name:torch.zeros(env.num_envs,2,device=env.device) for name in names}
        state.update({name:torch.zeros(env.num_envs,device=env.device) for name in
                      ("gap_samples","gap_sum","gap_bad","up_steps","down_steps","loaded_without_local_map")})
        state["front_loaded_a_min"] = torch.full((env.num_envs,2),torch.pi,device=env.device)
        state["front_swing_a_min"] = torch.full((env.num_envs,2),torch.pi,device=env.device)
        state["rear_b_max"] = torch.full((env.num_envs,2),-torch.pi,device=env.device)
        state["rear_g_max"] = torch.full((env.num_envs,2),-torch.pi,device=env.device)
        env._m20_pose_stats = state
    reset = env.episode_length_buf <= 1
    for key,value in state.items():
        value[reset] = torch.pi if key.endswith("_min") else (-torch.pi if key.endswith("_max") else 0)
    state["up_steps"] += up; state["down_steps"] += down
    for phase,mask,cost in (("loaded",front_load,loaded_cost),("swing",front_air,air_cost)):
        state[f"front_{phase}_samples"] += mask
        state[f"front_{phase}_a_sum"] += front_a*mask
        state[f"front_{phase}_bad"] += (cost>0)&mask
        state[f"front_{phase}_a_min"] = torch.minimum(state[f"front_{phase}_a_min"],torch.where(mask,front_a,torch.pi))
    state["rear_loaded_samples"] += rear_load
    state["rear_b_sum"] += rear_b*rear_load; state["rear_g_sum"] += rear_g*rear_load
    state["rear_bad"] += (rear_cost>0)&rear_load
    state["rear_b_max"] = torch.maximum(state["rear_b_max"],torch.where(rear_load,rear_b,-torch.pi))
    state["rear_g_max"] = torch.maximum(state["rear_g_max"],torch.where(rear_load,rear_g,-torch.pi))
    descent = getattr(env,"_m20_stair_step_state",None)
    entry = torch.zeros(env.num_envs,dtype=torch.bool,device=env.device)
    if descent is not None:
        # Mixed support: at least one front wheel has transferred down but rear
        # support has not completed, rather than time since first seeing stairs.
        entry = descent["active"] & ~descent["up"] & descent["front_reached"].any(-1) & ~descent["rear_support_paid"].all(-1)
    mask = rear_load & entry[:,None]
    state["rear_entry_samples"] += mask
    state["rear_entry_b_sum"] += rear_b*mask; state["rear_entry_g_sum"] += rear_g*mask
    state["gap_samples"] += gap_valid
    state["gap_sum"] += torch.where(gap_valid,gap,0)
    state["gap_bad"] += gap_valid&(gap<TUNING["body_gap_target"])
    state["loaded_without_local_map"] += (load&~ground_valid&(up|down)[:,None]).sum(-1)


def gait_quality_metrics(env,env_ids):
    result = _legacy_gait_quality_metrics(env,env_ids)
    state = getattr(env,"_m20_pose_stats",None)
    if state is None:
        return result
    sums = {key:value[env_ids].sum(0) for key,value in state.items()}
    def average(total,count):
        return torch.rad2deg(total.sum()/count.sum().clamp_min(1))
    for phase in ("loaded","swing"):
        count = sums[f"front_{phase}_samples"]
        result[f"front_{phase}_a_mean_deg"] = average(sums[f"front_{phase}_a_sum"],count)
        result[f"front_{phase}_a_low_fraction"] = sums[f"front_{phase}_bad"].sum()/count.sum().clamp_min(1)
        observed = state[f"front_{phase}_a_min"][env_ids]
        result[f"front_{phase}_a_min_deg"] = torch.rad2deg(observed.amin()) if observed.numel() and count.sum()>0 else count.new_zeros(())
        for leg,i in (("fl",0),("fr",1)):
            result[f"{leg}_{phase}_samples"] = count[i]
            result[f"{leg}_{phase}_a_mean_deg"] = torch.rad2deg(sums[f"front_{phase}_a_sum"][i]/count[i].clamp_min(1))
            result[f"{leg}_{phase}_a_low_fraction"] = sums[f"front_{phase}_bad"][i]/count[i].clamp_min(1)
    rear = sums["rear_loaded_samples"]
    result["rear_forward_samples"] = rear.sum()
    result["rear_forward_excess_fraction"] = sums["rear_bad"].sum()/rear.sum().clamp_min(1)
    result["rear_body_b_mean_deg"] = average(sums["rear_b_sum"],rear)
    result["rear_gravity_g_mean_deg"] = average(sums["rear_g_sum"],rear)
    for coord in ("b","g"):
        peak = state[f"rear_{coord}_max"][env_ids]
        result[f"rear_{coord}_max_deg"] = torch.rad2deg(peak.amax()) if peak.numel() and rear.sum()>0 else rear.new_zeros(())
    for leg,i in (("hl",0),("hr",1)):
        result[f"{leg}_loaded_samples"] = rear[i]
        result[f"{leg}_body_b_mean_deg"] = torch.rad2deg(sums["rear_b_sum"][i]/rear[i].clamp_min(1))
        result[f"{leg}_gravity_g_mean_deg"] = torch.rad2deg(sums["rear_g_sum"][i]/rear[i].clamp_min(1))
        result[f"{leg}_forward_excess_fraction"] = sums["rear_bad"][i]/rear[i].clamp_min(1)
    entry = sums["rear_entry_samples"]
    result["rear_entry_samples"] = entry.sum()
    result["rear_entry_b_mean_deg"] = average(sums["rear_entry_b_sum"],entry)
    result["rear_entry_g_mean_deg"] = average(sums["rear_entry_g_sum"],entry)
    result["front_body_gap_samples"] = sums["gap_samples"]
    result["front_body_gap_mean_m"] = sums["gap_sum"]/sums["gap_samples"].clamp_min(1)
    result["front_body_gap_low_fraction"] = sums["gap_bad"]/sums["gap_samples"].clamp_min(1)
    result["loaded_without_local_map_samples"] = sums["loaded_without_local_map"]
    return result


def stair_step_completion_metrics(env,env_ids):
    result = _legacy_step_completion_metrics(env,env_ids)
    state = getattr(env,"_m20_ascent_state",None)
    if state is None:
        return result
    ids = torch.arange(env.num_envs,device=env.device)[env_ids]
    known = torch.arange(64,device=env.device)[None] < state["map_count"][ids,None]
    intermediate = state["map_intermediate"][ids]&known
    result["registered_risers"] = state["map_count"][ids].float().mean()
    for axle,name in enumerate(("front","rear")):
        paid = state["map_paid"][ids,:,axle]&known
        strict = state["map_correct"][ids,:,axle]&~state["map_duplicate_paid"][ids,:,axle]&intermediate
        result[f"up_{name}_registered_intermediate_risers"] = intermediate.sum().float()
        result[f"up_{name}_landing_coverage"] = paid.sum().float()/known.sum().clamp_min(1)
        result[f"up_{name}_strict_coverage"] = strict.sum().float()/intermediate.sum().clamp_min(1)
        result[f"up_{name}_physical_landings"] = state["physical_total"][ids,axle].float().mean()
        result[f"up_{name}_transition_successes"] = state[name+"_transition_successes"][ids].mean()
        result[f"up_{name}_skipped_riser_events"] = state["skip_total"][ids,axle].float().mean()
        result[f"up_{name}_unsafe_events"] = state["unsafe_total"][ids,axle].float().mean()
        result[f"up_{name}_skip_fraction"] = (
            state["skip_total"][ids,axle].sum()/state["physical_total"][ids,axle].sum().clamp_min(1))
        visited = paid & intermediate
        result[f"up_{name}_visited_intermediate_risers"] = visited.sum().float()
        result[f"up_{name}_same_tread_fraction"] = (
            (state["map_duplicate_paid"][ids,:,axle]&visited).sum().float()/visited.sum().clamp_min(1))
        # Longest surviving strict run. A later duplicate invalidates an earlier
        # tread; future detected-but-unlanded treads contribute no success.
        order = (state["map_flight"][ids].float()*1e6+state["map_z"][ids]).argsort(dim=-1,stable=True)
        good = strict.gather(1,order)
        flight = state["map_flight"][ids].gather(1,order)
        group_start = torch.ones_like(good); group_start[:,1:] = flight[:,1:]!=flight[:,:-1]
        total = good.long().cumsum(-1)
        restart = torch.where(~good,total,torch.where(group_start,total-good.long(),0)).cummax(-1).values
        runs = (total-restart).amax(-1)
        result[f"up_{name}_longest_strict_run"] = runs.float().mean()
        result[f"up_{name}_six_step_strict_fraction"] = (runs>=6).float().mean()
        for side,leg in enumerate((("fl","fr"),("hl","hr"))[axle]):
            result[f"{leg}_stable_treads"] = (state["map_landed"][ids,:,axle*2+side]&known).sum(-1).float().mean()
    result["up_rear_lift_credit"] = state["rear_lift_credit"][ids].mean()
    return result


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
