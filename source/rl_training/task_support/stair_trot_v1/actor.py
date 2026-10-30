"""Frozen complete motion teacher plus an ascent-only trainable residual."""
import torch
from torch import nn

from task_support.idle_stop_motion_v1.motion_residual_actor import MotionResidualMLPModel


class StairTrotActor(MotionResidualMLPModel):
    uses_trot_control = True

    def __init__(self, *args, trot_hidden_dims=(256, 128), correction_limits=None,
                 command_limits=None, **kwargs):
        super().__init__(*args, **kwargs)
        # Includes the old motion residual, normalizer and action distribution.
        self.requires_grad_(False)
        dims = (244 + 3 + 6, *trot_hidden_dims, 16)
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ELU())
        self.trot_mlp = nn.Sequential(*layers)
        nn.init.zeros_(self.trot_mlp[-1].weight)
        nn.init.zeros_(self.trot_mlp[-1].bias)
        if correction_limits is None or len(correction_limits) != 16:
            raise ValueError("StairTrot requires 16 explicit action correction limits")
        limits = torch.tensor(correction_limits, dtype=torch.float32)
        if not torch.isfinite(limits).all() or (limits <= 0).any():
            raise ValueError("StairTrot correction limits must be positive and finite")
        self.register_buffer("trot_limits", limits)
        if command_limits is None or len(command_limits) != 3:
            raise ValueError("StairTrot requires forward/lateral/yaw command thresholds")
        self.command_limits = tuple(command_limits)
        self._last_gate = None

    def load_teacher_state(self, state):
        expected = {key for key in self.state_dict()
                    if not key.startswith("trot_mlp.") and key != "trot_limits"}
        if set(state) != expected:
            raise ValueError("StairTrot requires the complete motion-residual teacher")
        self.load_state_dict(state, strict=False)

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        if masks is not None:
            raise ValueError("StairTrot supports nonrecurrent policies only")
        control = obs["trot_control"]
        if control.shape[-1] != 6:
            raise ValueError("trot_control requires gate, phase sin/cos, cross-track and heading sin/cos")
        with torch.no_grad():
            baseline = super().forward(obs, stochastic_output=False)
        gate = control[:, :1].clamp(0, 1)
        command = obs["motion_control"]
        moving = ((command[:, 0] > self.command_limits[0])
                  & (command[:, 1].abs() <= self.command_limits[1])
                  & (command[:, 2].abs() <= self.command_limits[2]))
        gate = gate * moving[:, None]
        features = torch.cat((obs["policy"], obs["motion_control"], control), -1)
        bounded = torch.tanh(self.trot_mlp(features))
        correction = bounded * self.trot_limits * gate
        # torch.where makes the inactive path exactly the full teacher.
        mean = torch.where(gate > 0, baseline + correction, baseline)
        self._last_gate = gate[:, 0].detach()
        self._last_residual_rms = correction.square().mean(-1).sqrt().detach()
        self._last_motion_regularization = (
            bounded.square().mean(-1) * gate[:, 0]
        ).sum() / gate.sum().clamp_min(1)
        if not stochastic_output:
            return mean
        self.distribution.update(mean)
        return self.distribution.sample()

    @torch.no_grad()
    def record_rollout_sample(self, actions):
        active = self._last_gate > 0
        values = torch.stack((active.sum(), (~active).sum(),
                              self._last_residual_rms.sum())).float()
        self._rollout_metrics = values if self._rollout_metrics is None else self._rollout_metrics + values

    def take_rollout_metrics(self):
        if self._rollout_metrics is None:
            return {}
        active, inactive, correction = self._rollout_metrics.cpu().tolist()
        self._rollout_metrics = None
        return {"trot_active_fraction": active / max(active + inactive, 1),
                "trot_residual_rms": correction / max(active, 1)}

    def update_normalization(self, obs):
        # The original policy ABI uses no normalization. Do not adapt its buffers.
        return None
