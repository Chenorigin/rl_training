"""Learn small commanded-motion corrections around a frozen M20 baseline.

The original actor is used unchanged on every raw zero-command step. PPO can
only update a bounded residual on commanded motion, preserving the teacher's
idle-stop behavior at the policy-output level.
"""
from __future__ import annotations

import copy

import torch
from torch import nn
from torch.distributions import Normal
from rsl_rl.models import MLPModel


class MotionResidualMLPModel(MLPModel):
    frozen_motion = True
    uses_motion_control = True

    def __init__(self, obs, obs_groups, obs_set, output_dim, residual_hidden_dims=(128, 64), **kwargs):
        groups = copy.deepcopy(obs_groups)
        groups[obs_set] = ["policy"]
        super().__init__(obs, groups, obs_set, output_dim, **kwargs)
        if self.obs_dim != 244 or output_dim != 16 or obs["motion_control"].shape[-1] != 3:
            raise ValueError("MotionResidual requires policy244 + raw_command3 -> action16")
        self.residual_hidden_dims = tuple(residual_hidden_dims)
        self.mlp.requires_grad_(False)
        self.obs_normalizer.requires_grad_(False)
        self.distribution.requires_grad_(False)
        dims = (244 + 3, *self.residual_hidden_dims, 16)
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ELU())
        self.residual_mlp = nn.Sequential(*layers)
        nn.init.zeros_(self.residual_mlp[-1].weight)
        nn.init.zeros_(self.residual_mlp[-1].bias)
        self._last_motion_control = None
        self._last_motion_regularization = None
        self._last_residual_rms = None
        self._rollout_metrics = None

    def load_baseline_state(self, state):
        expected = set(self.state_dict()) - {k for k in self.state_dict() if k.startswith("residual_mlp.")}
        if set(state) != expected:
            raise ValueError("Motion baseline does not match the 244D -> 16D M20 actor")
        result = self.load_state_dict(state, strict=False)
        if result.unexpected_keys or any(not k.startswith("residual_mlp.") for k in result.missing_keys):
            raise ValueError("Unexpected baseline actor state")

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        if masks is not None:
            raise ValueError("MotionResidual is a nonrecurrent single-device policy")
        command = obs["motion_control"]
        moving = command.abs().amax(-1) > 1.e-4
        self._last_motion_control = command.detach()
        baseline_mean = super().forward(obs, stochastic_output=False)
        bounded = torch.tanh(self.residual_mlp(torch.cat((obs["policy"], command), -1)))
        # Leg actions use a 0.25 radian scale and wheels a 5 rad/s scale.
        # These caps allow modest gait corrections while bounding one update.
        correction_limit = baseline_mean.new_tensor([.18, .35, .35] * 4 + [.16] * 4)
        residual = bounded * correction_limit
        self._last_motion_regularization = (bounded.square().mean(-1) * moving).sum() / moving.sum().clamp_min(1)
        self._last_residual_rms = residual.square().mean(-1).sqrt().detach()
        mean = baseline_mean + residual * moving[:, None]
        if not stochastic_output:
            return mean
        self.distribution.update(mean)
        self.distribution._distribution = Normal(mean, self.distribution.std)
        return self.distribution.sample()

    def motion_regularization(self):
        return self._last_motion_regularization

    @torch.no_grad()
    def record_rollout_sample(self, actions):
        command = self._last_motion_control
        moving = command.abs().amax(-1) > 1.e-4
        correction = (self._last_residual_rms * moving).sum()
        values = torch.stack((moving.sum(), (~moving).sum(), correction)).float()
        self._rollout_metrics = values if self._rollout_metrics is None else self._rollout_metrics + values

    def take_rollout_metrics(self):
        if self._rollout_metrics is None:
            return {}
        moving, zero, correction = self._rollout_metrics.cpu().tolist()
        self._rollout_metrics = None
        total = moving + zero
        return {"motion_command_fraction": moving / total if total else 0.,
                "raw_zero_command_fraction": zero / total if total else 0.,
                "motion_residual_rms": correction / moving if moving else 0.}

    def as_jit(self):
        raise NotImplementedError("MotionResidual requires policy + raw motion_control inputs")

    def as_onnx(self, verbose=False):
        raise NotImplementedError("MotionResidual requires explicit raw motion_control inputs")
