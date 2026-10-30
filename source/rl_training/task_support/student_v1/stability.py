"""Training-only stability objectives. The deployed actor ABI is unchanged."""
import torch
import torch.nn.functional as F


def sensor_noise_std(cfg):
    """Convert the configured sensor errors to existing observation units."""
    return {"gyro": cfg.gyro_std_rad_s * .25, "gravity": cfg.roll_pitch_std,
            "joint_position": cfg.joint_position_std_rad,
            "joint_velocity": cfg.joint_velocity_std_rad_s * .05,
            "height": cfg.height_noise_std}


def perturb_sensors(observations, std, scale):
    """Perturb current measurements, preserving commands and recorded actions.

    Past proprio, map validity/confidence, and unknown heights remain intact.
    This is a second measurement of the same state, not a fictitious command
    or action transition. It never adds privileged teacher information.
    """
    result = observations.clone()
    for name, start, end in (("gyro", 0, 3), ("joint_position", 9, 21),
                             ("joint_velocity", 25, 41)):
        result[:, start:end] += torch.randn_like(result[:, start:end]) * std[name] * scale
    gravity = observations[:, 3:6]
    perturbed = gravity + torch.randn_like(gravity) * std["gravity"] * scale
    result[:, 3:6] = F.normalize(perturbed, dim=-1) * gravity.norm(dim=-1, keepdim=True)
    valid = observations[:, 244:431] > .5
    result[:, 57:244] += torch.randn_like(result[:, 57:244]) * std["height"] * scale * valid
    return result


def adjacent_training_pairs(observations, dones, training_mask, regular_phase, commands=None):
    """Return (later time, env) pairs; exclude resets and either held-out end."""
    valid = training_mask[1:] & training_mask[:-1]
    valid &= regular_phase[1:] & regular_phase[:-1]
    valid &= ~dones[:-1].squeeze(-1).bool()
    # Safe's actor command includes noisy heading feedback. Use the original
    # requested command when available: feedback jitter is not a human start/
    # stop and must not silently remove the very pairs we want to supervise.
    commands = observations[:, :, 6:9] if commands is None else commands
    valid &= torch.isclose(commands[1:], commands[:-1],
                           atol=1.e-6, rtol=0.).all(-1)
    time, env = valid.nonzero(as_tuple=True)
    return time + 1, env


def action_group_mse(residual, leg_weight, wheel_weight):
    return (leg_weight * residual[:, :12].square().mean()
            + wheel_weight * residual[:, 12:].square().mean()) / (leg_weight + wheel_weight)


def temporal_residual_loss(current, previous, teacher_current, teacher_previous,
                           leg_weight=1., wheel_weight=1.):
    """Match teacher action changes rather than suppressing required motion."""
    residual = (current - previous) - (teacher_current - teacher_previous).detach()
    return action_group_mse(residual, leg_weight, wheel_weight)
