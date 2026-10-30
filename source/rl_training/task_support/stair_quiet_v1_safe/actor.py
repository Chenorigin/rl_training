"""Reuse both width policy weights; add a zero-output safety branch."""
import torch
from torch import nn
from rsl_rl.models import MLPModel
from task_support.idle_stop_motion_v1.motion_residual_actor import MotionResidualMLPModel
from .settings import SAFETY_DIM, SCAN_DIM


class SafeResidualActor(MotionResidualMLPModel):
    def __init__(self, obs, *args, **kwargs):
        super().__init__(obs, *args, **kwargs)
        if obs["safety"].shape[-1] != SAFETY_DIM or obs["locomotion_scan"].shape[-1] != SCAN_DIM:
            raise ValueError("Safe teacher requires raw policy244, longitudinal187 and safety48")
        self.safety_mlp = nn.Sequential(nn.Linear(244+3+SAFETY_DIM, 128), nn.ELU(),
            nn.Linear(128, 64), nn.ELU(), nn.Linear(64, 16))
        nn.init.zeros_(self.safety_mlp[-1].weight)
        nn.init.zeros_(self.safety_mlp[-1].bias)

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        if masks is not None:
            raise ValueError("SafeResidualActor is nonrecurrent")
        command = obs["motion_control"]
        moving = command.abs().amax(-1) > 1.e-4
        self._last_motion_control = command.detach()
        raw = obs["policy"]
        longitudinal = torch.cat((raw[:, :57], obs["locomotion_scan"]), -1)
        base_input = torch.where(moving[:, None], longitudinal, raw)
        baseline = MLPModel.forward(self, {"policy": base_input})
        logits = self.residual_mlp(torch.cat((base_input, command), -1))
        logits = logits+self.safety_mlp(torch.cat((raw, command, obs["safety"]), -1))
        bounded = torch.tanh(logits)
        limit = baseline.new_tensor([.18, .35, .35]*4+[.16]*4)
        residual = bounded*limit
        self._last_motion_regularization = (bounded.square().mean(-1)*moving).sum()/moving.sum().clamp_min(1)
        self._last_residual_rms = residual.square().mean(-1).sqrt().detach()
        mean = baseline+residual*moving[:, None]
        if stochastic_output:
            self.distribution.update(mean)
            return self.distribution.sample()
        return mean

    def load_width_state(self, state):
        expected = {k for k in self.state_dict() if not k.startswith("safety_mlp.")}
        if set(state) != expected:
            raise ValueError("Width actor does not match the safe actor's preserved weights")
        result = self.load_state_dict(state, strict=False)
        if result.unexpected_keys or any(not k.startswith("safety_mlp.") for k in result.missing_keys):
            raise ValueError("Incomplete width actor initialization")
