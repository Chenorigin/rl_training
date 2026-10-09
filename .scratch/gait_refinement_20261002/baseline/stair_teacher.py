# Copyright (c) 2026 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""M20 teacher observations and stair-specific PPO reward terms.

Terrain gates are derived from the same yaw-aligned ray scan exposed to the
teacher actor. Contact and actuator state are used only for training signals
and critic observations.
"""

from __future__ import annotations

import numpy as np
import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse, yaw_quat

from .stair_ascent import ascent_state
from .commands import UniformThresholdVelocityCommand
from .rewards import joint_pos_penalty


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
        cost = cost * (~(turn | side)).to(cost.dtype)
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
