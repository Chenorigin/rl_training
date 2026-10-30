"""Command, reset and completion rules for the straight ascent terrain tiles."""
from dataclasses import dataclass
import math

import numpy as np
import torch

from isaaclab.envs.mdp.events import reset_root_state_uniform
from isaaclab.managers import SceneEntityCfg
from task_support.idle_stop_motion_v1.rewards import (
    StairTeacherVelocityCommand,
    stair_distance_terrain_levels,
)

from .signals import yaw_from_quaternion
from .terrain import AscentStairsTerrainCfg


@dataclass
class StairCourseSettings:
    """Task thresholds in metres, seconds, radians and newtons."""

    speed_range: tuple[float, float] = (0.25, 0.55)
    reset_xy_range: float = 0.10
    reset_yaw_range: float = 0.08
    exit_inset: float = 0.10
    success_hold_time: float = 0.16
    success_lateral_limit: float = 0.35
    success_heading_limit: float = 0.20
    success_tilt_limit: float = 0.30
    success_linear_speed_max: float = 0.65
    success_angular_speed_max: float = 0.80
    support_force_min: float = 15.0
    top_wheel_height_margin: float = 0.02
    top_root_height_margin: float = 0.15
    floor_min_root_height: float = 0.10
    route_lateral_limit: float = 0.80
    end_boundary_margin: float = 0.30
    backwards_limit: float = 0.70


def _settings(env):
    return env.cfg.stair_course


def terrain_column_indices(generator):
    """Exact IsaacLab TerrainGenerator curriculum column selection.

    Upstream uses ``min(where(column / num_cols + .001 < cumsum(p)))``.
    ``searchsorted(..., side="right")`` preserves its strict inequality at
    boundaries. Random-generation mode does not have this column mapping.
    """
    if not generator.curriculum:
        raise ValueError("Straight stair course requires TerrainGenerator curriculum=True.")
    proportions = np.asarray([cfg.proportion for cfg in generator.sub_terrains.values()], dtype=float)
    if (
        len(proportions) == 0 or not np.isfinite(proportions).all()
        or (proportions < 0).any() or proportions.sum() <= 0 or generator.num_cols < 1
    ):
        raise ValueError("Terrain column proportions and num_cols must be valid.")
    cumulative = np.cumsum(proportions / proportions.sum())
    indices = np.searchsorted(cumulative, np.arange(generator.num_cols) / generator.num_cols + 0.001,
                              side="right")
    if (indices >= len(proportions)).any():
        raise ValueError("Terrain proportions do not cover every IsaacLab column.")
    return indices


def course_state(env):
    """Cached course routing and episode state.

    Public ``mask``, ``depth``, ``success`` are per-environment tensors.
    ``goal_x`` is WORLD X at the start of the final landing; ``goal_x_offset``
    is the same point relative to the terrain origin. Origins are refreshed
    on every call because curriculum changes the row before reset events run.
    """
    state = getattr(env, "_stair_trot_course", None)
    if state is None:
        generator = env.scene.terrain.cfg.terrain_generator
        columns = terrain_column_indices(generator)
        names = tuple(generator.sub_terrains)
        count = len(names)
        column_t = torch.as_tensor(columns, dtype=torch.long, device=env.device)
        env_types = column_t[env.scene.terrain.terrain_types]
        mask_by_type = torch.zeros(count, dtype=torch.bool, device=env.device)
        depth_by_type = torch.zeros(count, device=env.device)
        goal_by_type = torch.zeros(count, device=env.device)
        top_by_type = torch.zeros(count, device=env.device)
        end_by_type = torch.zeros(count, device=env.device)
        for index, cfg in enumerate(generator.sub_terrains.values()):
            if not isinstance(cfg, AscentStairsTerrainCfg):
                continue
            if cfg.step_depth_range[0] != cfg.step_depth_range[1]:
                raise ValueError("Course completion requires fixed-depth ascent tiles; use separate depth columns.")
            if cfg.proportion > 0 and index not in columns:
                raise ValueError(f"Ascent terrain {names[index]!r} has no generator column.")
            mask_by_type[index] = True
            depth_by_type[index] = cfg.step_depth_range[0]
            goal_by_type[index] = cfg.start_platform_length / 2 + cfg.num_steps * cfg.step_depth_range[0]
            top_by_type[index] = cfg.num_steps * cfg.step_height_range[0]
            end_by_type[index] = generator.size[0] - cfg.start_platform_length / 2
        n = env.num_envs
        state = {
            "mask": mask_by_type[env_types], "depth": depth_by_type[env_types],
            "goal_x_offset": goal_by_type[env_types], "min_top_z_offset": top_by_type[env_types],
            "end_x_offset": end_by_type[env_types], "type_ids": env_types,
            "type_names": names, "ascent_type_mask": mask_by_type,
            "success": torch.zeros(n, dtype=torch.bool, device=env.device),
            "success_tick": torch.full((n,), -1, dtype=torch.long, device=env.device),
            "stop": torch.zeros(n, dtype=torch.bool, device=env.device),
            "hold_time": torch.zeros(n, device=env.device),
            "speed": torch.zeros(n, device=env.device),
            "last_tick": torch.full((n,), -1, dtype=torch.long, device=env.device),
            "reported_tick": torch.full((n,), -1, dtype=torch.long, device=env.device),
            "completed_count": torch.zeros(count, dtype=torch.long, device=env.device),
            "success_count": torch.zeros(count, dtype=torch.long, device=env.device),
        }
        env._stair_trot_course = state
    state["goal_x"] = env.scene.env_origins[:, 0] + state["goal_x_offset"]
    return state


def _ids(env, env_ids):
    if env_ids is None or isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device)[slice(None) if env_ids is None else env_ids]
    return torch.as_tensor(env_ids, device=env.device, dtype=torch.long)


def _wheel_ids(env, state):
    if "wheel_ids" not in state:
        names = ["fl_wheel", "fr_wheel", "hl_wheel", "hr_wheel"]
        asset_ids, asset_names = env.scene["robot"].find_bodies(names, preserve_order=True)
        sensor_ids, sensor_names = env.scene.sensors["contact_forces"].find_bodies(names, preserve_order=True)
        if list(asset_names) != names or list(sensor_names) != names:
            raise RuntimeError("Stair course requires the four explicitly ordered M20 wheel bodies.")
        state["wheel_ids"] = asset_ids
        state["contact_ids"] = sensor_ids
    return state["wheel_ids"], state["contact_ids"]


def update_course_progress(env):
    """Latch a supported, aligned landing during the episode, once per tick."""
    state = course_state(env)
    tick = env.common_step_counter
    fresh = state["mask"] & (state["last_tick"] != tick) & (env.episode_length_buf > 0)
    if not fresh.any():
        return state
    cfg = _settings(env)
    body_ids, contact_ids = _wheel_ids(env, state)
    data = env.scene["robot"].data
    wheel_pos = data.body_pos_w[:, body_ids]
    forces = env.scene.sensors["contact_forces"].data.net_forces_w[:, contact_ids]
    root = data.root_pos_w
    origin = env.scene.env_origins
    root_delta = root - origin
    minimum_top = origin[:, 2] + state["min_top_z_offset"]
    crossed = (wheel_pos[:, :, 0] >= state["goal_x"][:, None] + cfg.exit_inset).all(-1)
    elevated = (
        (wheel_pos[:, :, 2] >= minimum_top[:, None] + cfg.top_wheel_height_margin).all(-1)
        & (root[:, 2] >= minimum_top + cfg.top_root_height_margin)
    )
    # A large horizontal riser force cannot masquerade as upward load support.
    supported = (
        (forces[:, :, 2] > cfg.support_force_min)
        & (forces[:, :, 2] > torch.linalg.vector_norm(forces[:, :, :2], dim=-1))
    ).sum(-1) >= 2
    heading = yaw_from_quaternion(data.root_quat_w).abs()
    aligned = (
        (root_delta[:, 1].abs() <= cfg.success_lateral_limit)
        & (heading <= cfg.success_heading_limit)
        & (-data.projected_gravity_b[:, 2] >= math.cos(cfg.success_tilt_limit))
    )
    stable = (
        (torch.linalg.vector_norm(data.root_lin_vel_w, dim=-1) <= cfg.success_linear_speed_max)
        & (torch.linalg.vector_norm(data.root_ang_vel_w, dim=-1) <= cfg.success_angular_speed_max)
    )
    valid = (
        torch.isfinite(wheel_pos).all(dim=(1, 2)) & torch.isfinite(forces).all(dim=(1, 2))
        & torch.isfinite(root).all(-1)
    )
    landed = crossed & elevated & aligned & supported & stable & valid
    state["stop"] |= fresh & crossed & elevated & valid
    state["hold_time"][fresh] = torch.where(
        landed[fresh], state["hold_time"][fresh] + env.step_dt, 0.0
    )
    newly_completed = fresh & landed & (state["hold_time"] >= cfg.success_hold_time) & ~state["success"]
    state["success_tick"][newly_completed] = tick
    state["success"] |= newly_completed
    state["last_tick"][fresh] = tick
    return state


class StairAscentVelocityCommand(StairTeacherVelocityCommand):
    """Keep the existing teacher sampler on ordinary terrain.

    Stair tiles start facing world +X and command forward motion with zero
    lateral/yaw rates. Route rewards and failure/completion rules keep that
    world heading; the parent continues accounting commanded episode distance.
    """

    def _apply_course_command(self, env_ids):
        state = course_state(self._env)
        ids = _ids(self._env, env_ids)
        ids = ids[state["mask"][ids]]
        self.vel_command_b[ids] = 0.0
        self.vel_command_b[ids, 0] = torch.where(state["stop"][ids], 0.0, state["speed"][ids])
        self.is_heading_env[ids] = False
        self.is_standing_env[ids] = state["stop"][ids]
        self.heading_target[ids] = 0.0
        # The parent sampler may first choose one of its stop exercises. The
        # ascent override must also clear those flags, not only its velocity.
        for name in ("_stair_zero_transition_env", "_flat_zero_transition_env", "_stair_exit_zero_env",
                     "_entry_pause_pending", "_v4_preview_pending"):
            getattr(self, name)[ids] = False

    def _resample_command(self, env_ids):
        ids = _ids(self._env, env_ids)
        super()._resample_command(ids)
        state = course_state(self._env)
        ascent_ids = ids[state["mask"][ids]]
        state["speed"][ascent_ids] = torch.empty(len(ascent_ids), device=self.device).uniform_(
            *_settings(self._env).speed_range
        )
        self._apply_course_command(ids)

    def _update_command(self):
        super()._update_command()
        self._apply_course_command(None)


def reset_root_on_stair_course(
    env, env_ids, pose_range, velocity_range, asset_cfg=SceneEntityCfg("robot")
):
    """Preserve ordinary resets; start ascent tiles upright, aligned and still."""
    state = course_state(env)
    ids = _ids(env, env_ids)
    normal_ids = ids[~state["mask"][ids]]
    if normal_ids.numel():
        reset_root_state_uniform(env, normal_ids, pose_range, velocity_range, asset_cfg)
    ascent_ids = ids[state["mask"][ids]]
    if ascent_ids.numel():
        cfg = _settings(env)
        asset = env.scene[asset_cfg.name]
        default = asset.data.default_root_state[ascent_ids]
        position = default[:, :3] + env.scene.env_origins[ascent_ids]
        position[:, :2] += torch.empty((len(ascent_ids), 2), device=env.device).uniform_(
            -cfg.reset_xy_range, cfg.reset_xy_range
        )
        yaw = torch.empty(len(ascent_ids), device=env.device).uniform_(-cfg.reset_yaw_range, cfg.reset_yaw_range)
        quaternion = torch.zeros((len(ascent_ids), 4), device=env.device)
        quaternion[:, 0] = torch.cos(yaw / 2)
        quaternion[:, 3] = torch.sin(yaw / 2)
        asset.write_root_pose_to_sim(torch.cat((position, quaternion), -1), env_ids=ascent_ids)
        asset.write_root_velocity_to_sim(torch.zeros((len(ascent_ids), 6), device=env.device), env_ids=ascent_ids)
    state["success"][ids] = False
    state["success_tick"][ids] = -1
    state["stop"][ids] = False
    state["hold_time"][ids] = 0.0
    # Body/sensor data may still describe the old pose until the next physics step.
    state["last_tick"][ids] = env.common_step_counter


def stair_course_failure(env):
    """Non-timeout termination; also observes completion before reset occurs."""
    state = update_course_progress(env)
    cfg = _settings(env)
    delta = env.scene["robot"].data.root_pos_w - env.scene.env_origins
    failed = (
        ~torch.isfinite(delta).all(-1)
        | (delta[:, 2] < cfg.floor_min_root_height)
        | (delta[:, 1].abs() > cfg.route_lateral_limit)
        | (delta[:, 0] > state["end_x_offset"] - cfg.end_boundary_margin)
        | (delta[:, 0] < -cfg.backwards_limit)
    )
    return state["mask"] & failed


def course_completion_reward(env):
    """One completion event per episode; repeated same-tick reads are stable.

    RewardManager multiplies by step_dt, so division here gives a unit event
    before the configured reward weight, independent of the control frequency.
    """
    state = update_course_progress(env)
    return (state["mask"] & (state["success_tick"] == env.common_step_counter)).float() / env.step_dt


def stair_course_terrain_levels(
    env, env_ids, min_commanded_translation_m=0.1, asset_cfg=SceneEntityCfg("robot")
):
    """Promote completed stair ascents; preserve the original mixed-terrain rule."""
    state = update_course_progress(env)
    ids = _ids(env, env_ids)
    normal_ids = ids[~state["mask"][ids]]
    if normal_ids.numel():
        stair_distance_terrain_levels(env, normal_ids, min_commanded_translation_m, asset_cfg)
    ascent_ids = ids[
        state["mask"][ids] & (env.episode_length_buf[ids] > 0)
        & (state["reported_tick"][ids] != env.common_step_counter)
    ]
    if ascent_ids.numel():
        terminated = env.termination_manager.terminated[ascent_ids]
        timed_out = env.termination_manager.time_outs[ascent_ids]
        finished = terminated | timed_out
        move_up = finished & state["success"][ascent_ids] & ~terminated
        move_down = terminated | (timed_out & ~state["success"][ascent_ids])
        types = state["type_ids"][ascent_ids[finished]]
        state["completed_count"].index_add_(0, types, torch.ones_like(types))
        state["success_count"].index_add_(0, types, move_up[finished].long())
        env.scene.terrain.update_env_origins(ascent_ids, move_up, move_down)
        state["reported_tick"][ascent_ids] = env.common_step_counter
    mean_level = env.scene.terrain.terrain_levels.float().mean()
    from task_support.idle_stop_motion_v1.regularizers import update_gait_level_from_terrain_mean
    update_gait_level_from_terrain_mean(mean_level)
    return mean_level


def course_metrics(env, env_ids):
    """Cumulative actual completed-episode counts, safe after origins change."""
    state = course_state(env)
    metrics = {}
    for index, name in enumerate(state["type_names"]):
        if not state["ascent_type_mask"][index]:
            continue
        count = state["completed_count"][index]
        successes = state["success_count"][index]
        metrics[f"{name}/episodes"] = count.float()
        metrics[f"{name}/successes"] = successes.float()
        metrics[f"{name}/success_fraction"] = torch.where(
            count > 0, successes.float() / count.clamp_min(1), count.new_tensor(float("nan"), dtype=torch.float)
        )
    return metrics
