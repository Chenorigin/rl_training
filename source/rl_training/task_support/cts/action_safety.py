"""Training-only tail risk and one acceptance rule for imitation labels.

Never clips the deployed target or changes the fault checker. Costs use raw
mapped proposals so even outputs outside the normalized ABI clip have gradient.
"""
import math
import torch


class ActionSafety:
    def __init__(self, limits, cfg):
        if limits is None:
            raise ValueError('Safety learning requires the software joint contract')
        self.limits, self.cfg = limits, dict(cfg)

    def raw_targets(self, action):
        return self.limits.default + action[..., :12]*self.limits.scale

    def accept(self, action):
        bounds = self.limits.target
        margin = (bounds[:, 1]-bounds[:, 0])*self.limits.cfg['margin_fraction']
        q = self.raw_targets(action)
        return (torch.isfinite(action).all(-1) &
                (q >= bounds[:, 0]+margin).all(-1) &
                (q <= bounds[:, 1]-margin).all(-1)).detach()

    def tail_cost(self, action):
        if not torch.isfinite(action).all():
            raise RuntimeError('Nonfinite safety learning proposal')
        bounds = self.limits.target
        margin = (bounds[:, 1]-bounds[:, 0])*self.limits.cfg['margin_fraction']
        q = self.raw_targets(action)
        cost = ((bounds[:, 0]+margin-q).clamp_min(0) +
                (q-bounds[:, 1]+margin).clamp_min(0)).square()/margin.square()
        worst = cost.amax(-1).flatten()
        count = max(1, math.ceil(worst.numel()*self.cfg['tail_fraction']))
        return cost.mean() + worst.topk(count).values.mean()
