"""Batched legacy_v1 heading command feedback shared by v3.5 simulation paths."""
import math
import torch


class LegacyHeadingFeedback:
    def __init__(self, count, device, tuning):
        names = ('translation_threshold', 'yaw_threshold', 'heading_hold_kp', 'heading_hold_kd',
                 'heading_hold_max_yaw_rate', 'heading_hold_deadband', 'stair_heading_hold_kp',
                 'stair_heading_hold_kd', 'stair_heading_hold_max_yaw_rate',
                 'stair_heading_hold_deadband', 'heading_hold_rate_enter')
        if any(k not in tuning or not isinstance(tuning[k], (int, float))
               or not math.isfinite(tuning[k]) or tuning[k] < 0 for k in names):
            raise ValueError('Missing or invalid legacy_v1 heading tuning')
        self.tuning = {k: tuning[k] for k in names}
        self.mode = torch.zeros(count, device=device, dtype=torch.long)
        self.target = torch.zeros(count, device=device)
        self.step = None
        self.before = None

    def reset(self, ids=None):
        ids = slice(None) if ids is None else ids
        self.mode[ids] = 0
        self.target[ids] = 0.
        if self.before is not None:
            self.before[0][ids] = 0
            self.before[1][ids] = 0.

    def update(self, raw, yaw, body_yaw_rate, stairs, step):
        if self.step is not None and step < self.step:
            self.reset()
            self.step = None
        if self.step == step and self.before is not None:
            self.mode.copy_(self.before[0])
            self.target.copy_(self.before[1])
        else:
            self.before = (self.mode.clone(), self.target.clone())
        t = self.tuning
        straight = ((raw[:, 0].abs() > t['translation_threshold'])
                    & (raw[:, 1].abs() <= t['translation_threshold'])
                    & (raw[:, 2].abs() <= t['yaw_threshold']))
        zero = (raw[:, :2].norm(dim=-1) <= t['translation_threshold']) & (raw[:, 2].abs() <= t['yaw_threshold'])
        mode = torch.where(straight, 1, torch.where(zero, 2, 0))
        enter = (mode != self.mode) & (mode != 0)
        self.target[enter] = yaw[enter]
        error = torch.atan2(torch.sin(self.target-yaw), torch.cos(self.target-yaw))
        kp, kd, limit, deadband = [torch.where(stairs, t['stair_'+key], t[key]) for key in
            ('heading_hold_kp', 'heading_hold_kd', 'heading_hold_max_yaw_rate', 'heading_hold_deadband')]
        active = (error.abs() > deadband) | (body_yaw_rate.abs() > t['heading_hold_rate_enter'])
        correction = (kp*error-kd*body_yaw_rate).clamp(-limit, limit)
        result = raw.clone()
        result[:, 2] = torch.where(mode != 0, torch.where(active, correction, 0.), raw[:, 2])
        self.mode.copy_(mode)
        self.step = step
        return result


def stair_scan_mask(hits, grid):
    """Same nearby-edge gate as teacher _stair_scan_context (11x17 XY grid)."""
    z = hits[..., 2].reshape(-1, 11, 17)
    valid = torch.isfinite(z) & (z.abs() < 1.e6)
    lateral = grid[:, 1].reshape(11, 17).abs() <= .35
    valid &= lateral
    profile = torch.nanmedian(torch.where(valid, z, torch.nan), dim=1).values
    columns = valid.sum(dim=1) >= 2
    edge_x = .5*(grid[:17, 0][1:]+grid[:17, 0][:-1])
    nearby = (edge_x >= -.45) & (edge_x <= .70)
    edges = columns[:, 1:] & columns[:, :-1] & (profile[:, 1:]-profile[:, :-1]).abs().gt(.06)
    return (edges & nearby).any(dim=1)


class SensorHeadingFeedback(LegacyHeadingFeedback):
    """Use only delivered student height/validity/confidence and noisy IMU/pose.

    The teacher's legacy PD gains are retained; stair-gain selection is based
    on deployable sensor channels rather than clean rays or reward trackers.
    """
    def __call__(self, raw, estimated_yaw, observed_yaw_rate, channels, step):
        height, valid, confidence = channels
        x = torch.arange(-8, 9, device=height.device, dtype=height.dtype)*.1
        y = torch.arange(-5, 6, device=height.device, dtype=height.dtype)*.1
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        grid = torch.stack((xx.flatten(), yy.flatten()), -1)
        usable = (valid > .5) & (confidence > 0.)
        z = torch.where(usable, -height, torch.nan)
        hits = torch.cat((grid[None].expand(raw.shape[0], -1, -1), z[..., None]), -1)
        stairs = stair_scan_mask(hits, grid)
        return self.update(raw, estimated_yaw, observed_yaw_rate, stairs, step)
