"""Physical row routing and per-width evidence. No virtual safety boundary."""
import torch
from isaaclab.managers import SceneEntityCfg
from task_support.idle_stop_motion_v1.rewards import StairTeacherVelocityCommand
from .settings import cases
from .signals import failure_mask
from .curriculum import WidthCurriculum, record_outcomes


def course_state(env):
    state = getattr(env, "_tail_course", None)
    if state is None:
        table = cases()
        state = {
            key: torch.tensor(values, device=env.device, dtype=dtype)
            for key, values, dtype in (
                ("height_table", [g.height for g, _ in table], torch.float),
                ("depth_table", [g.depth for g, _ in table], torch.float),
                ("width_table", [g.width for g, _ in table], torch.float),
                ("steps_table", [g.steps for g, _ in table], torch.long),
                ("stage_table", [p.stage for _, p in table], torch.long),
            )}
        state["speed"] = torch.full((env.num_envs,), .35, device=env.device)
        state["assign_count"] = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        state["episodes"] = torch.zeros(len(table), dtype=torch.long, device=env.device)
        state["successes"] = torch.zeros(len(table), dtype=torch.long, device=env.device)
        state["stage_episodes"] = torch.zeros(len(table), dtype=torch.long, device=env.device)
        state["stage_successes"] = torch.zeros(len(table), dtype=torch.long, device=env.device)
        state["case_metrics"] = {key: torch.zeros(len(table), device=env.device) for key in (
            "force_peak_sum", "force_peak_max", "impulse_peak_sum", "impulse_peak_max",
            "slip_seconds_sum", "unsafe_seconds_sum", "recovery_events_sum", "edge_failures_sum")}
        env._tail_course = state
        env.tail_stage = env.cfg.stair_tail.initial_stage
    if "width_curriculum" not in state:
        s = env.cfg.stair_tail
        state["width_curriculum"] = WidthCurriculum(env.num_envs, env.device, s)
        state["width_values"] = torch.tensor(s.widths_m, device=env.device)
        state["width_statistics"] = {key: torch.zeros(len(s.widths_m), len(cases()),
            dtype=torch.long, device=env.device) for key in (
                "episodes", "successes", "edge_failures", "recovery_timeouts", "stall_timeouts", "other_failures")}
    types = env.scene.terrain.terrain_types
    for name in ("height", "depth", "steps"):
        state[name] = state[name+"_table"][types]
    levels = env.scene.terrain.terrain_levels
    state["width"] = state["width_values"][levels]
    return state


class WidthAscentCommand(StairTeacherVelocityCommand):
    def _apply_course(self, ids):
        state = course_state(self._env)
        tracker = getattr(self._env, "tail_tracker", None)
        self.vel_command_b[ids] = 0.
        stop = (tracker.last["crossed"] if tracker is not None and hasattr(tracker, "last")
                else torch.zeros(self._env.num_envs, dtype=torch.bool, device=self.device))
        self.vel_command_b[ids, 0] = torch.where(stop[ids], 0., state["speed"][ids])
        self.is_heading_env[ids] = False
        self.is_standing_env[ids] = stop[ids]
        self.heading_target[ids] = 0.
        for name in ("_stair_zero_transition_env", "_flat_zero_transition_env", "_stair_exit_zero_env",
                     "_entry_pause_pending", "_v4_preview_pending"):
            getattr(self, name)[ids] = False

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        state = course_state(self._env)
        state["speed"][env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(
            *self._env.cfg.stair_tail.speed_range)
        self._apply_course(env_ids)

    def _update_command(self):
        super()._update_command()
        self._apply_course(slice(None))

def reset_course(env, env_ids, pose_range, velocity_range, asset_cfg=SceneEntityCfg("robot")):
    state = course_state(env)
    s, terrain = env.cfg.stair_tail, env.scene.terrain
    allowed = torch.where(state["stage_table"] <= env.tail_stage)[0]
    assignment = (env_ids+state["assign_count"][env_ids]*env.num_envs+int(env.cfg.seed)) % len(allowed)
    terrain.terrain_types[env_ids] = allowed[assignment]
    levels = state["width_curriculum"].level[env_ids]
    terrain.terrain_levels[env_ids] = levels
    terrain.env_origins[env_ids] = terrain.terrain_origins[levels, terrain.terrain_types[env_ids]]
    env.scene.env_origins[env_ids] = terrain.env_origins[env_ids]
    state["assign_count"][env_ids] += 1
    robot = env.scene[asset_cfg.name]
    pos = robot.data.default_root_state[env_ids, :3].clone()+env.scene.env_origins[env_ids]
    pos[:, 0] += torch.empty(len(env_ids), device=env.device).uniform_(-.05, .05)
    pos[:, 1] += torch.empty(len(env_ids), device=env.device).uniform_(-s.reset_y_range, s.reset_y_range)
    yaw = torch.empty(len(env_ids), device=env.device).uniform_(-s.reset_yaw_range, s.reset_yaw_range)
    quat = torch.zeros(len(env_ids), 4, device=env.device)
    quat[:, 0], quat[:, 3] = torch.cos(yaw/2), torch.sin(yaw/2)
    robot.write_root_pose_to_sim(torch.cat((pos, quat), -1), env_ids=env_ids)
    robot.write_root_velocity_to_sim(torch.zeros(len(env_ids), 6, device=env.device), env_ids=env_ids)
    course_state(env)


def record_episodes(env, env_ids):
    state, tracker = course_state(env), env.tail_tracker
    finished = env.termination_manager.terminated | env.termination_manager.time_outs
    ids = env_ids[(env.episode_length_buf[env_ids] > 0) & finished[env_ids]]
    types = env.scene.terrain.terrain_types[ids]
    rows = env.scene.terrain.terrain_levels[ids].clone()
    succeeded = tracker.success[ids] & ~failure_mask(env)[ids]
    edge, recovery = tracker.edge_failed[ids], tracker.recovery_failed[ids]
    stall = tracker.stall_age[ids] > env.cfg.stair_tail.no_progress_timeout_s
    record_outcomes(state, types, rows, succeeded, edge, recovery, stall, env.cfg.stair_tail)
    for name in ("force_peak", "impulse_peak", "slip_seconds", "unsafe_seconds", "recovery_events"):
        value = getattr(tracker, name)[ids]
        state["case_metrics"][name+"_sum"].index_add_(0, types, value)
        if name+"_max" in state["case_metrics"]:
            state["case_metrics"][name+"_max"].scatter_reduce_(0, types, value, reduce="amax", include_self=True)
    state["case_metrics"]["edge_failures_sum"].index_add_(0, types, edge.float())
    # Update targets only. Physics, scan and rewards keep the old row until
    # reset_course teleports to the corresponding prebuilt physical staircase.
    state["width_curriculum"].update(ids, actual_levels=rows, succeeded=succeeded)
    s = env.cfg.stair_tail
    active = state["stage_table"] <= env.tail_stage
    ready = ((state["stage_episodes"][active] >= s.stage_min_episodes)
             & (state["stage_successes"][active]/state["stage_episodes"][active].clamp_min(1)
                >= s.stage_success_fraction)).all()
    if s.automatic_stages and env.tail_stage < 2 and bool(ready):
        env.tail_stage += 1
        state["stage_episodes"].zero_()
        state["stage_successes"].zero_()
    result = {"Tail/stage": float(env.tail_stage), "Width/actual_mean_m": state["width"].mean(),
              "Width/target_row_fraction": (env.scene.terrain.terrain_levels == len(s.widths_m)-1).float().mean()}
    for i, width in enumerate(s.widths_m):
        for key, values in state["width_statistics"].items():
            result[f"Width/{width:g}m/{key}"] = values[i].sum().float()
    if len(ids):
        for name in ("force_peak", "impulse_peak", "slip_seconds", "unsafe_seconds", "recovery_events"):
            value = getattr(tracker, name)[ids]
            label = name+"_episode_mean"
            result["Tail/"+label] = value.mean()
            result["Tail/__weighted_total__/"+label] = value.sum()
            result["Tail/__weighted_count__/"+label] = len(ids)
        for name in ("edge_failed", "recovery_failed"):
            value = getattr(tracker, name)[ids].float()
            label = name+"_fraction"
            result["Tail/"+label] = value.mean()
            result["Tail/__weighted_total__/"+label] = value.sum()
            result["Tail/__weighted_count__/"+label] = len(ids)
        for foot, leg in enumerate(("fl", "fr", "hl", "hr")):
            for name in ("unsafe_foot_seconds", "slip_foot_seconds", "support_foot_seconds"):
                value = getattr(tracker, name)[ids, foot]
                label = f"foot/{leg}/{name}_episode_mean"
                result["Tail/"+label] = value.mean()
                result["Tail/__weighted_total__/"+label] = value.sum()
                result["Tail/__weighted_count__/"+label] = len(ids)
    return result
