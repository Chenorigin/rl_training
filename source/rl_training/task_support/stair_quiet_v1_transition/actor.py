"""Keep the stair scan route continuous through zero command; same weights/ABI."""
import torch
from rsl_rl.models import MLPModel
from task_support.stair_quiet_v1_safe.actor import SafeResidualActor


class TransitionResidualActor(SafeResidualActor):
    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        if masks is not None:
            raise ValueError("Transition actor is nonrecurrent")
        raw, command = obs["policy"], obs["motion_control"]
        moving = command.abs().amax(-1) > 1.e-4
        self._last_motion_control = command.detach()
        base_input = torch.cat((raw[:, :57], obs["locomotion_scan"]), -1)
        baseline = MLPModel.forward(self, {"policy": base_input})
        logits = self.residual_mlp(torch.cat((base_input, command), -1))
        logits += self.safety_mlp(torch.cat((raw, command, obs["safety"]), -1))
        bounded = torch.tanh(logits)
        residual = bounded*baseline.new_tensor([.18, .35, .35]*4+[.16]*4)
        self._last_motion_regularization = (bounded.square().mean(-1)*moving).sum()/moving.sum().clamp_min(1)
        self._last_residual_rms = residual.square().mean(-1).sqrt().detach()
        mean = baseline+residual*moving[:, None]
        if stochastic_output:
            self.distribution.update(mean)
            return self.distribution.sample()
        return mean
