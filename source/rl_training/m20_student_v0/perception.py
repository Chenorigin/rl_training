"""Bounded synthetic map acquisition, delivery and world-coordinate reprojection.

This samples oracle terrain into delayed map patches. It models map errors, not
LiDAR beams, self-occlusion geometry, multi-frame fusion or a ROS mapping stack.
All three policy channels come from one snapshot. Truth error is never used to
construct confidence. Defaults are engineering starting values, not calibration.
"""

from dataclasses import dataclass, fields
import math

import torch
import torch.nn.functional as F

from .contract import (GRID_ROWS, GRID_COLS, GRID_RESOLUTION, HEIGHT_DIM,
                       HEIGHT_OFFSET, HEIGHT_CLIP)


@dataclass
class StudentV0PerceptionCfg:
    level: float = 0.25
    height_noise_std: float = 0.02
    height_bias_std: float = 0.015
    xy_bias_std: float = 0.015
    yaw_bias_std: float = 0.01
    point_dropout: float = 0.03
    occlusion_probability: float = 0.30
    occlusion_width_m: float = 0.40
    occlusion_height_m: float = 0.30
    occlusion_hold_s: float = 0.50
    map_update_period_s: float = 0.10
    max_delay_s: float = 0.06
    packet_dropout: float = 0.05
    max_map_age_s: float = 0.30
    variance_growth_per_s: float = 0.001
    confidence_reference_std: float = 0.05
    confidence_age_tau_s: float = 0.50

    def validate(self):
        for item in fields(self):
            value = getattr(self, item.name)
            if not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{item.name} must be finite and nonnegative")
        for name in ("level", "point_dropout", "occlusion_probability", "packet_dropout"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in ("max_map_age_s", "confidence_reference_std", "confidence_age_tau_s", "occlusion_hold_s"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


def confidence_from_quality(validity, variance, age_s, cfg):
    """Same finite quality conversion can be used by a deployment adapter."""
    cfg.validate()
    usable = validity.bool() & torch.isfinite(variance) & (variance >= 0)
    usable = usable & torch.isfinite(age_s) & (age_s >= 0) & (age_s <= cfg.max_map_age_s)
    safe_var = torch.where(usable, variance, torch.zeros_like(variance))
    safe_age = torch.where(usable, age_s, torch.zeros_like(age_s))
    score = torch.exp(-safe_age / cfg.confidence_age_tau_s)
    score = score / (1 + safe_var / cfg.confidence_reference_std**2)
    return torch.where(usable, score, torch.zeros_like(score))


class SyntheticMapState:
    """Per-env delayed patches, held in their acquisition coordinates.

    A reset attempts to acquire a new packet; configured dropout/delay still apply. Invalid
    rays stay unknown. Nearest sampling preserves discontinuities and validity.
    """

    def __init__(self, num_envs, device, step_dt, cfg):
        cfg.validate()
        if not math.isfinite(step_dt) or step_dt <= 0:
            raise ValueError("step_dt must be positive")
        self.cfg, self.step_dt, self.num_envs = cfg, step_dt, num_envs
        self.device = device
        self.max_delay_steps = math.ceil(cfg.max_delay_s / step_dt)
        self.queue_length = self.max_delay_steps + 1
        shape = (self.queue_length, num_envs, HEIGHT_DIM)
        self.queue_z = torch.zeros(shape, device=device)
        self.queue_valid = torch.zeros(shape, device=device, dtype=torch.bool)
        self.queue_variance = torch.zeros(shape, device=device)
        self.queue_anchor = torch.zeros(self.queue_length, num_envs, 3, device=device)
        self.queue_capture = torch.full((self.queue_length, num_envs), -1, device=device, dtype=torch.long)
        self.queue_arrival = self.queue_capture.clone()
        self.z = torch.zeros(num_envs, HEIGHT_DIM, device=device)
        self.valid = torch.zeros_like(self.z, dtype=torch.bool)
        self.variance = torch.zeros_like(self.z)
        self.anchor = torch.zeros(num_envs, 3, device=device)
        self.capture = torch.full((num_envs,), -1, device=device, dtype=torch.long)
        self.next_capture = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.next_occlusion = self.next_capture.clone()
        self.bias = torch.zeros(num_envs, 4, device=device)
        self.occlusion_center = torch.zeros(num_envs, 2, device=device)
        self.occluded = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self.needs_reset = torch.ones_like(self.occluded)
        self.last_step = None
        self.channels = None
        x = (torch.arange(GRID_COLS, device=device) - (GRID_COLS - 1)/2) * GRID_RESOLUTION
        y = (torch.arange(GRID_ROWS, device=device) - (GRID_ROWS - 1)/2) * GRID_RESOLUTION
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        self.grid = torch.stack((xx.flatten(), yy.flatten()), -1)

    def reset(self, env_ids):
        ids = slice(None) if env_ids is None else env_ids
        self.queue_capture[:, ids] = -1
        self.queue_arrival[:, ids] = -1
        self.queue_valid[:, ids] = False
        self.capture[ids] = -1
        self.valid[ids] = False
        self.needs_reset[ids] = True
        self.next_capture[ids] = 0
        self.next_occlusion[ids] = 0
        self.last_step = None
        self.channels = None

    def snapshot(self, ray_hits_w, base_pos_w, yaw, step):
        if self.last_step == step and self.channels is not None:
            return tuple(channel.clone() for channel in self.channels)
        if ray_hits_w.shape != (self.num_envs, HEIGHT_DIM, 3):
            raise ValueError("Expected [num_envs,187,3] ordered ray hits")
        if base_pos_w.shape != (self.num_envs, 3) or yaw.shape != (self.num_envs,):
            raise ValueError("Invalid base position/yaw shape")
        if not torch.isfinite(base_pos_w).all() or not torch.isfinite(yaw).all():
            raise ValueError("Nonfinite base pose cannot be reprojected")
        cfg, level = self.cfg, self.cfg.level
        cfg.validate()
        if math.ceil(cfg.max_delay_s / self.step_dt) != self.max_delay_steps:
            raise ValueError("Changing delay capacity requires a new environment")
        new_ids = self.needs_reset.nonzero().flatten()
        if new_ids.numel():
            scale = torch.tensor([cfg.height_bias_std, cfg.xy_bias_std,
                                  cfg.xy_bias_std, cfg.yaw_bias_std], device=self.device)
            self.bias[new_ids] = torch.randn(new_ids.numel(), 4, device=self.device) * scale
            self.needs_reset[new_ids] = False
        redraw = self.next_occlusion <= step
        if redraw.any():
            count = int(redraw.sum())
            self.occluded[redraw] = torch.rand(count, device=self.device) < cfg.occlusion_probability * level
            self.occlusion_center[redraw] = (torch.rand(count, 2, device=self.device) - .5) * torch.tensor([1.6, 1.0], device=self.device)
            self.next_occlusion[redraw] = step + max(1, math.ceil(cfg.occlusion_hold_s / self.step_dt))
        acquire = self.next_capture <= step
        if acquire.any():
            period = max(1, math.ceil(cfg.map_update_period_s * level / self.step_dt))
            self.next_capture[acquire] = step + period
            accepted = acquire & (torch.rand(self.num_envs, device=self.device) >= cfg.packet_dropout * level)
            ids = accepted.nonzero().flatten()
            if ids.numel():
                slot = step % self.queue_length
                hits = ray_hits_w[ids]
                valid = torch.isfinite(hits).all(-1)
                radius = torch.linalg.vector_norm(self.grid, dim=-1)
                std = cfg.height_noise_std * level * (1 + radius)[None, :]
                std = std * (0.5 + torch.rand(ids.numel(), 1, device=self.device))
                z = torch.where(valid, hits[..., 2], torch.zeros_like(hits[..., 2]))
                z = z + torch.randn_like(z) * std + self.bias[ids, 0, None] * level
                valid &= torch.rand_like(z) >= cfg.point_dropout * level
                delta = (self.grid[None] - self.occlusion_center[ids, None]).abs()
                hole = (delta[..., 0] <= cfg.occlusion_width_m * level / 2)
                hole &= delta[..., 1] <= cfg.occlusion_height_m * level / 2
                valid &= ~(hole & self.occluded[ids, None])
                self.queue_z[slot, ids] = z
                self.queue_valid[slot, ids] = valid
                self.queue_variance[slot, ids] = std.expand_as(z).square() + (cfg.height_bias_std * level)**2
                self.queue_anchor[slot, ids, :2] = base_pos_w[ids, :2] + self.bias[ids, 1:3] * level
                self.queue_anchor[slot, ids, 2] = yaw[ids] + self.bias[ids, 3] * level
                delay_steps = math.ceil(cfg.max_delay_s * level / self.step_dt)
                delay = torch.randint(delay_steps + 1, (ids.numel(),), device=self.device)
                self.queue_capture[slot, ids] = step
                self.queue_arrival[slot, ids] = step + delay
        deliverable = (self.queue_capture >= 0) & (self.queue_arrival <= step)
        newest, slots = torch.where(deliverable, self.queue_capture, -1).max(0)
        ids = (newest > self.capture).nonzero().flatten()
        if ids.numel():
            selected = slots[ids]
            self.z[ids] = self.queue_z[selected, ids]
            self.valid[ids] = self.queue_valid[selected, ids]
            self.variance[ids] = self.queue_variance[selected, ids]
            self.anchor[ids] = self.queue_anchor[selected, ids]
            self.capture[ids] = newest[ids]
        # Transform current ego grid -> world -> acquisition grid.
        cos, sin = yaw.cos()[:, None], yaw.sin()[:, None]
        wx = base_pos_w[:, 0, None] + cos*self.grid[:, 0] - sin*self.grid[:, 1]
        wy = base_pos_w[:, 1, None] + sin*self.grid[:, 0] + cos*self.grid[:, 1]
        dx, dy = wx-self.anchor[:, 0, None], wy-self.anchor[:, 1, None]
        cos, sin = self.anchor[:, 2].cos()[:, None], self.anchor[:, 2].sin()[:, None]
        gx, gy = cos*dx + sin*dy, -sin*dx + cos*dy
        query = torch.stack((gx / .8, gy / .5), -1).reshape(self.num_envs, GRID_ROWS, GRID_COLS, 2)
        maps = torch.stack((self.z, self.valid.float(), self.variance), 1)
        maps = maps.reshape(self.num_envs, 3, GRID_ROWS, GRID_COLS)
        sampled = F.grid_sample(maps, query, mode="nearest", padding_mode="zeros", align_corners=True).flatten(2)
        inside = (gx.abs() <= .8 + 1e-5) & (gy.abs() <= .5 + 1e-5)
        age = ((step - self.capture).float() * self.step_dt)[:, None].expand_as(self.z)
        valid = (sampled[:, 1] > .5) & inside & (self.capture[:, None] >= 0)
        valid &= age <= cfg.max_map_age_s
        variance = sampled[:, 2] + age * cfg.variance_growth_per_s * level
        height = (base_pos_w[:, 2, None] - sampled[:, 0] - HEIGHT_OFFSET).clamp(*HEIGHT_CLIP)
        confidence = confidence_from_quality(valid, variance, age, cfg)
        self.channels = (torch.where(valid, height, 0.0), valid.float(), confidence)
        self.last_step = step
        return tuple(channel.clone() for channel in self.channels)


def _state(env, sensor_cfg):
    state = getattr(env, "_m20_student_v0_map", None)
    if state is None:
        state = SyntheticMapState(env.num_envs, env.device, env.step_dt, env.cfg.student_perception)
        env._m20_student_v0_map = state
    sensor = env.scene.sensors[sensor_cfg.name]
    quat = env.scene["robot"].data.root_quat_w
    w, x, y, z = quat.unbind(-1)
    yaw = torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    return state.snapshot(sensor.data.ray_hits_w, sensor.data.pos_w, yaw, env.common_step_counter)


def student_height(env, sensor_cfg):
    return _state(env, sensor_cfg)[0]


def student_validity(env, sensor_cfg):
    return _state(env, sensor_cfg)[1]


def student_confidence(env, sensor_cfg):
    return _state(env, sensor_cfg)[2]


def reset_student_map(env, env_ids):
    state = getattr(env, "_m20_student_v0_map", None)
    if state is not None:
        state.reset(env_ids)
