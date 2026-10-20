"""M20 gait refinement v2: rewards, command sampling and physical lifecycle.

Parameter defaults live in gait_refinement_v2_env_cfg.py. The actor remains244D;
heading state is reward/diagnostic state and never rewrites deployment commands.
"""
import math
from pathlib import Path
import torch

from .quiet_landing_env import M20QuietLandingEnv
from .quiet_landing_env import StairTeacherVelocityCommand

PROJECT_ROOT = Path(__file__).resolve().parents[8]

def idle_mask(policy, command_threshold, flat_range):
    """Drift or missing wheel support cannot disable the requested stopping target."""
    if policy.ndim != 2 or policy.shape[1] != 244:
        raise ValueError("Idle stopping requires the V1 244D policy layout")
    command = policy[:, 6:9]
    scan = policy[:, 57:244]
    known = torch.isfinite(policy).all(-1) & (scan.abs() < 1.0).all(-1)
    flat = torch.nan_to_num(scan).amax(-1) - torch.nan_to_num(scan).amin(-1) <= flat_range
    return known & flat & (command.abs().amax(-1) < command_threshold)


def idle_preservation_loss(predicted, reference, std, mask, settings, uneven_weight=1.0):
    """Preserve legs and moving outputs; fit idle wheel velocity targets to zero."""
    if predicted.shape != reference.shape or predicted.ndim != 2 or predicted.shape[1] != 16:
        raise ValueError("Idle preservation requires [batch,16] actions")
    if mask.shape != (len(predicted),) or mask.dtype != torch.bool:
        raise ValueError("Idle mask must be a matching bool vector")
    difference = (predicted - reference.detach()) / std.detach().clamp_min(0.05)
    keep = torch.ones_like(difference)
    keep[:, 12:] *= (~mask)[:, None]
    preservation = (difference.square() * keep).sum(-1) / keep.sum(-1)
    wheel = (predicted[:, 12:] / settings.wheel_action_reference).square().mean(-1)
    stopping = (wheel * mask).sum() / mask.sum().clamp_min(1)
    return (settings.preservation_weight * (preservation * uneven_weight).mean()
            + settings.wheel_target_weight * stopping)


def wrapped_angle(angle):
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def quaternion_yaw(quaternion):
    w, x, y, z = quaternion.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))


class StairHeadingTracker:
    def __init__(self, num_envs, device, settings):
        self.settings = settings
        self.settings.validate()
        self.reference = torch.zeros(num_envs, device=device)
        self.previous_command = torch.zeros(num_envs, 3, device=device)
        self.segment = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self.hold = torch.zeros(num_envs, device=device)
        self.last_step = -1
        self.current = None
        self.stats = {name: torch.zeros(num_envs, device=device) for name in
                      ("samples", "error_sum", "max_error", "outside", "starts", "up", "down")}

    def begin_step(self, yaw, command, stair, scan_valid, dt):
        """Anchor before physical motion; retain it across stairs/platforms/stops."""
        s = self.settings
        valid = torch.isfinite(yaw) & torch.isfinite(command).all(-1)
        moving = torch.linalg.vector_norm(command[:, :2], dim=-1) > s.translation_threshold
        turning = command[:, 2].abs() >= s.intentional_turn_threshold
        straight = valid & moving & ~turning
        previous_norm = torch.linalg.vector_norm(self.previous_command[:, :2], dim=-1)
        norm = torch.linalg.vector_norm(command[:, :2], dim=-1)
        cosine = (command[:, :2] * self.previous_command[:, :2]).sum(-1) / (norm * previous_norm).clamp_min(1e-6)
        changed = moving & (previous_norm > s.translation_threshold) & (cosine < math.cos(math.radians(s.direction_change_degrees)))
        start = straight & (~self.segment | changed)
        self.reference = torch.where(start, yaw, self.reference)
        self.segment = torch.where(turning | ~valid, False, self.segment | start)
        detected = stair & scan_valid & self.segment & ~turning
        self.hold = torch.where(detected, s.post_stair_seconds, (self.hold - dt).clamp_min(0))
        self.hold = torch.where(turning | ~valid | changed, 0.0, self.hold)
        self.active = self.segment & (self.hold > 0) & scan_valid & valid & ~turning
        self.previous_command = command.clone()
        self.reference = wrapped_angle(self.reference + torch.where(self.segment, command[:, 2] * dt, 0.0))
        self.stats["starts"] += start
        self.current = None

    def sample(self, yaw, yaw_rate, step, direction=None):
        """One update per control step; repeated reward calls cannot double count."""
        if self.current is not None and self.last_step == step:
            return self.current
        s = self.settings
        valid = torch.isfinite(yaw) & torch.isfinite(yaw_rate)
        active = self.active & valid if hasattr(self, "active") else torch.zeros_like(valid)
        error = wrapped_angle(yaw - self.reference)
        error = torch.where(valid, error, 0.0)
        excess = (error.abs() - math.radians(s.soft_target_degrees)).clamp_min(0) / math.radians(s.error_scale_degrees)
        rate = torch.nan_to_num(yaw_rate - self.previous_command[:, 2]) / s.yaw_rate_reference
        cost = (excess.square() + s.yaw_rate_mix * rate.square()).clamp_max(s.cost_cap) * active
        absolute = error.abs() * active
        self.stats["samples"] += active
        self.stats["error_sum"] += absolute
        self.stats["max_error"] = torch.maximum(self.stats["max_error"], absolute)
        self.stats["outside"] += active & (error.abs() > math.radians(s.acceptance_degrees))
        if direction is not None:
            self.stats["up"] += active & (direction > 0)
            self.stats["down"] += active & (direction < 0)
        self.last_step = step
        self.current = dict(active=active, error=error, cost=cost, reference=self.reference.clone())
        return self.current

    def metrics(self, ids):
        stats = {name: value[ids].sum() for name, value in self.stats.items()}
        count = stats["samples"].clamp_min(1)
        return {"GaitV2/heading_samples": stats["samples"],
                "GaitV2/heading_mean_error_deg": torch.rad2deg(stats["error_sum"] / count),
                "GaitV2/heading_max_error_deg": torch.rad2deg(self.stats["max_error"][ids].max()),
                "GaitV2/heading_outside_limit_fraction": stats["outside"] / count,
                "GaitV2/heading_segments": stats["starts"],
                "GaitV2/ascent_samples": stats["up"], "GaitV2/descent_samples": stats["down"]}

    def reset(self, ids):
        self.reference[ids] = 0
        self.previous_command[ids] = 0
        self.segment[ids] = False
        self.hold[ids] = 0
        if hasattr(self, "active"):
            self.active[ids] = False
        for value in self.stats.values():
            value[ids] = 0
        if self.current is not None:
            for value in self.current.values():
                value[ids] = 0


def scan_context(env):
    sensor = env.scene.sensors["height_scanner"]
    hits = sensor.data.ray_hits_w
    valid = torch.isfinite(hits).all(dim=(-1, -2))
    heights = torch.nan_to_num(hits[..., 2])
    span = heights.amax(-1) - heights.amin(-1)
    settings = env.cfg.gait_corrections
    stair = valid & (span > settings.heading.stair_height_range)
    rays = sensor.ray_starts
    rays = rays[0] if rays.ndim == 3 else rays
    front, rear = rays[:, 0] > 0.15, rays[:, 0] < -0.15
    difference = heights[:, front].mean(-1) - heights[:, rear].mean(-1)
    direction = torch.sign(difference) * stair
    return valid, span, stair, direction


def heading_context(env):
    tracker = getattr(env, "gait_heading_tracker", None)
    if tracker is None:
        raise RuntimeError("v2 heading reward requires M20GaitRefinementV2Env")
    robot = env.scene["robot"]
    return tracker.sample(quaternion_yaw(robot.data.root_quat_w),
                          robot.data.root_ang_vel_w[:, 2], env.common_step_counter,
                          getattr(env, "gait_stair_direction", None))


def stair_heading_cost(env):
    if not env.cfg.gait_corrections.stair_heading_enabled:
        return torch.zeros(env.num_envs, device=env.device)
    return heading_context(env)["cost"]


def idle_wheel_target_cost(env):
    """Raw scan and requested command select the target; achieved speed never does."""
    if not env.cfg.gait_corrections.idle_stop_enabled:
        return torch.zeros(env.num_envs, device=env.device)
    settings = env.cfg.gait_corrections.idle
    valid, span, _, _ = scan_context(env)
    command = env.command_manager.get_command("base_velocity")
    idle = valid & (span <= settings.flat_range) & (command.abs().amax(-1) < settings.command_threshold)
    wheel_target = env.action_manager.action[:, 12:]
    return (wheel_target / settings.wheel_action_reference).square().mean(-1).clamp_max(100) * idle


class GaitRefinementV2VelocityCommand(StairTeacherVelocityCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.stop_settings = env.cfg.gait_corrections.idle
        self.focus_sequence = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.sequence_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.sequence_speed = torch.zeros(self.num_envs, device=self.device)

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self.focus_sequence[ids] = (torch.rand(len(ids), device=self.device) < self.stop_settings.stop_sequence_fraction) & self._env.cfg.gait_corrections.idle_stop_enabled
        self.sequence_steps[ids] = 0
        self.sequence_speed[ids] = torch.empty(len(ids), device=self.device).uniform_(*self.stop_settings.move_speed_range)
        focus = ids[self.focus_sequence[ids]]
        self.is_heading_env[focus] = False
        self.is_standing_env[focus] = False
        self.vel_command_b[focus] = 0.0
        self.vel_command_b[focus, 0] = self.sequence_speed[focus]

    def _update_command(self):
        super()._update_command()
        if not self._env.cfg.gait_corrections.idle_stop_enabled:
            return
        focus = self.focus_sequence
        moving = focus & (self.sequence_steps * self._env.step_dt < self.stop_settings.move_seconds)
        stopping = focus & ~moving
        self.vel_command_b[focus] = 0.0
        self.vel_command_b[moving, 0] = self.sequence_speed[moving]
        self.is_standing_env[focus] = stopping[focus]
        self.is_heading_env[focus] = False
        self.sequence_steps += 1


class M20GaitRefinementV2Env(M20QuietLandingEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        cfg.gait_corrections.validate()
        cfg.apply_gait_corrections()
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        self.gait_heading_tracker = StairHeadingTracker(self.num_envs, self.device, cfg.gait_corrections.heading)
        self.gait_stair_direction = torch.zeros(self.num_envs, device=self.device)

    def step(self, action):
        robot = self.scene["robot"]
        valid, _, stair, direction = scan_context(self)
        self.gait_stair_direction = direction
        self.gait_heading_tracker.begin_step(quaternion_yaw(robot.data.root_quat_w),
            self.command_manager.get_command("base_velocity"), stair, valid, self.step_dt)
        # Parent executes the real original substep loop and calls v2 reward
        # terms after physics, before any done environments are reset.
        result = super().step(action)
        return result

    def _reset_idx(self, env_ids):
        tracker = getattr(self, "gait_heading_tracker", None)
        metrics = tracker.metrics(env_ids) if tracker is not None else {}
        super()._reset_idx(env_ids)
        self.extras["log"].update(metrics)
        if tracker is not None:
            tracker.reset(env_ids)
