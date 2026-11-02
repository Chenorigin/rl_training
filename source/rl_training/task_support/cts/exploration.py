"""Command-conditioned Gaussian scale shared by collection and replay.

Raw commands are deployable observations; terrain labels and teacher feedback
must not select a different exploration distribution. Missing settings preserve
the legacy fixed scale. The saved model buffer remains the moving scale.
"""
import math
import torch


def validate_scale(value):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 1:
        raise ValueError('stop_exploration_scale must be finite in (0, 1]')


def action_std(models, obs, cfg):
    raw = obs['motion_control']
    scale = cfg.get('stop_exploration_scale', 1.)
    stopped = raw.abs().amax(-1, keepdim=True) <= 1.e-4
    multiplier = torch.where(stopped, scale, 1.)
    return models.log_std.exp() * multiplier
