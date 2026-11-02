"""Versioned training objectives; no deployment controller or actor ABI change.

High-platform lessons explicitly command braking. This is commanded-stop
training, not a claim of autonomous cliff avoidance under a held forward input.
"""
import torch


def enabled(cfg):
    return cfg.get('objective_consistency', {}).get('enabled', False)


def validate(cfg):
    if not isinstance(cfg, dict) or set(cfg) != {'enabled'} or type(cfg['enabled']) is not bool:
        raise ValueError('Invalid objective consistency configuration')


class ObjectiveConsistency:
    def __init__(self, state):
        self.state = state
        n, device = state.env.num_envs, state.env.device
        self.command_seen = torch.zeros(n, 3, device=device)
        self.command_valid = torch.zeros(n, dtype=torch.bool, device=device)
        self.platform_stop = self.command_valid.clone()
        self.high_platform = self.command_valid.clone()
        self.goal_ok = torch.ones_like(self.command_valid)
        self.linear_ok = self.command_valid.clone()
        self.yaw_ok = self.command_valid.clone()
        self.stop_margin = torch.zeros(n, device=device)
        self.starts = torch.tensor([r.get('route_start', 0.) for r in state.atlas.records], device=device)

    def reset(self, ids):
        self.command_valid[ids] = self.platform_stop[ids] = False
        self.high_platform[ids] = self.linear_ok[ids] = self.yaw_ok[ids] = False
        self.goal_ok[ids] = True
        self.stop_margin[ids] = 0.

    def begin_step(self, obs):
        s = self.state
        # Copy the observations that produced this action; never rerun feedback.
        self.command_seen.copy_(torch.where(s.teacher[:, None], obs['teacher'][:, 6:9], obs['policy'][:, 6:9]))
        self.command_valid.fill_(True)
        s.measure_tick = -1
        s.geometry = None

    def command(self, raw):
        return torch.where(self.platform_stop[:, None], torch.zeros_like(raw), raw)

    def advance(self, g, risk):
        s, env = self.state, self.state.env
        p, axis = g['position'], g['axis']
        tail, cfg = env.cfg.stair_tail, env.cfg.cts_platform_practice
        local = p - s.origins[s.tile, None]
        # Full capsule footprint, in world axes, including orientation.
        extent_x = tail.wheel_radius + tail.wheel_half_length * axis[..., 0].abs()
        extent_y = tail.wheel_radius + tail.wheel_half_length * axis[..., 1].abs()
        front = (s.ends[s.tile, None] - local[..., 0] - extent_x).amin(-1)
        back = (local[..., 0] - extent_x - self.starts[s.tile, None]).amin(-1)
        side = (s.widths[s.tile, None] / 2 - local[..., 1].abs() - extent_y).amin(-1)
        d = env.scene['robot'].data
        q, raw = d.root_quat_w, g['command']
        requested_x = (1-2*(q[:, 2].square()+q[:, 3].square()))*raw[:, 0]
        requested_x += 2*(q[:, 1]*q[:, 2]-q[:, 0]*q[:, 3])*raw[:, 1]
        forward = torch.maximum(d.root_lin_vel_w[:, 0], requested_x).clamp_min(0.)
        margin = cfg['edge_margin'] + forward * cfg['prediction_s']
        self.high_platform = (s.course_ids == 9) & (s.mode == 0) & (risk > .5)
        entering = self.high_platform & ~self.platform_stop & ~s.failed & (raw[:, 0] > 1.e-4) & (front <= margin)
        self.stop_margin = torch.where(entering, margin, self.stop_margin)
        self.platform_stop |= entering
        # No side-edge completion, unsupported overhang, lower-floor landing,
        # or retreat far from the attained front stopping region.
        tolerance = s.cfg['hind_clearance']['support_tolerance_m']
        top = (g['height'] - s.origins[s.tile, 2, None]).abs().amax(-1) <= tolerance
        safe = ((front >= 0) & (back >= 0) & (side >= 0) & top
                & g['supported'].all(-1) & torch.isfinite(p).all(dim=(-1, -2)))
        near_front = front <= self.stop_margin + cfg['edge_margin']
        descended = (local[..., 0] > s.ends[s.tile, None]).all(-1)
        spatial = torch.where(self.high_platform, self.platform_stop & safe & near_front, descended)
        self.goal_ok = (s.course_ids != 9) | (s.mode == 8) | spatial

    def stop_phase(self, g, default):
        # The latch-entry physics step still executed forward: never credit it.
        executed_stop = self.platform_stop & (g['command'].abs().amax(-1) <= 1.e-4)
        return torch.where(self.high_platform, executed_stop & self.goal_ok, default)
