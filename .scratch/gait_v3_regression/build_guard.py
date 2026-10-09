from pathlib import Path
import inspect
from rsl_rl.algorithms import PPO
src=inspect.getsource(PPO.update).replace('def update(self)', 'def _guarded_update(self)',1)
needle='            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()'
assert src.count(needle)==1
src=src.replace(needle,needle+'\n            if self.reference_actor is not None and self.guard_updates >= self.critic_warmup_iterations:\n                anchor_loss = self._anchor_loss(batch.observations)\n                loss = loss + self.anchor_coefficient * anchor_loss\n                self._anchor_losses.append(float(anchor_loss.detach()))')
header='''# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# SPDX-License-Identifier: BSD-3-Clause
"""RSL-RL 5.0.1 PPO with an explicit frozen-policy preservation loss.

_guarded_update is vendored from the installed official PPO.update, with one
additional loss term. This keeps anchoring inside the same clipped-gradient
optimizer step rather than performing an untracked post-PPO policy update.
"""
from __future__ import annotations
import copy
import math
import torch
from torch import nn
from rsl_rl.algorithms import PPO


class StairGuardedPPO(PPO):
    def __init__(self, *args, anchor_coefficient=0.1, critic_warmup_iterations=50,
                 reference_std_multiplier=1.05, **kwargs):
        super().__init__(*args, **kwargs)
        if self.actor.is_recurrent or self.symmetry or self.rnd or self.is_multi_gpu:
            raise ValueError("StairGuardedPPO currently supports plain single-device MLP PPO only")
        if anchor_coefficient < 0 or critic_warmup_iterations < 0 or reference_std_multiplier < 1:
            raise ValueError("Invalid preservation configuration")
        self.anchor_coefficient = anchor_coefficient
        self.critic_warmup_iterations = critic_warmup_iterations
        self.reference_std_multiplier = reference_std_multiplier
        self.reference_actor = None
        self.guard_updates = 0
        self._anchor_losses = []

    def set_reference(self, actor_state):
        """Freeze a task-capable actor; do not anchor against a damaged checkpoint."""
        self.reference_actor = copy.deepcopy(self.actor)
        self.reference_actor.load_state_dict(actor_state, strict=True)
        self.reference_actor.eval().requires_grad_(False)
        if self.actor.obs_dim != 244:
            raise ValueError("Preservation mask requires the stair teacher 57+187 layout")
        if not hasattr(self.actor.distribution, 'log_std_param'):
            raise ValueError("Preservation requires state-independent log Gaussian std")

    def _anchor_loss(self, observations):
        with torch.no_grad():
            target = self.reference_actor(observations)
            std = self.reference_actor.output_std.detach().clamp_min(0.05)
            # Keep strong preservation on flat ground. Allow larger task-driven
            # changes on uneven terrain, including stairs in either direction.
            height = observations['policy'][:, 57:244]
            uneven = (height.amax(-1)-height.amin(-1)) > 0.07
            weight = torch.where(uneven, 0.1, 1.0)
        predicted = self.actor(observations)
        error = ((predicted-target)/std).square().mean(-1)
        return (weight*error).mean()

    def update(self):
        self._anchor_losses = []
        warmup = self.reference_actor is not None and self.guard_updates < self.critic_warmup_iterations
        parameters = list(self.actor.parameters())
        flags = [p.requires_grad for p in parameters]
        if warmup:
            for p in parameters:
                p.requires_grad_(False)
        try:
            result = self._guarded_update()
        finally:
            for p, flag in zip(parameters, flags):
                p.requires_grad_(flag)
        if self.reference_actor is not None:
            with torch.no_grad():
                limit = self.reference_actor.distribution.log_std_param + math.log(self.reference_std_multiplier)
                self.actor.distribution.log_std_param.copy_(torch.minimum(self.actor.distribution.log_std_param, limit))
        self.guard_updates += 1
        result['anchor'] = sum(self._anchor_losses)/max(len(self._anchor_losses), 1)
        result['critic_warmup'] = float(warmup)
        return result

    def save(self):
        state = super().save()
        if self.reference_actor is not None:
            state['preservation_reference_actor'] = self.reference_actor.state_dict()
        state['preservation_updates'] = self.guard_updates
        return state

    def load(self, loaded_dict, load_cfg=None, strict=True):
        result = super().load(loaded_dict, load_cfg, strict)
        # Restore auxiliary state only for a full resume, never for actor-only initialization.
        if load_cfg is None or load_cfg.get('iteration', False):
            reference = loaded_dict.get('preservation_reference_actor')
            if reference is not None:
                self.set_reference(reference)
                self.guard_updates = int(loaded_dict.get('preservation_updates', 0))
        return result

'''
Path('source/rl_training/rl_training/stair_guarded_ppo.py').write_text(header+src)
