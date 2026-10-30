"""Training-only flat straight-line course and decoded action supervision."""
import math

import torch


def validate_straight(cfg, motion):
    keys = {'fraction', 'move_s', 'patch_length', 'leg_action_loss_weight', 'batch_size'}
    if not isinstance(cfg, dict) or set(cfg) != keys:
        raise ValueError('Unknown or missing CTS straight practice settings')
    for key in keys - {'batch_size'}:
        v = cfg[key]
        if type(v) not in (int, float) or not math.isfinite(v) or v <= 0:
            raise ValueError('Invalid CTS straight practice '+key)
    if not cfg['fraction'] < motion['fraction'] or cfg['move_s'] <= motion['move_s']:
        raise ValueError('CTS straight practice must be a longer subset of motion')
    if type(cfg['batch_size']) is not int or cfg['batch_size'] < 1:
        raise ValueError('CTS straight batch size must be positive')
    from task_support.stair_quiet_v1_safe.settings import SafeSettings
    s = SafeSettings()
    footprint = math.hypot(s.body_half_length, s.body_half_width)+s.wheel_radius+s.wheel_half_length
    reach = motion['speed_range'][1]*cfg['move_s']+motion['reset_xy_range'][0]+footprint+motion['boundary_margin']
    if reach >= cfg['patch_length']/2:
        raise ValueError('CTS straight patch cannot contain the forward episode')
    lateral = motion['speed_range'][1]*cfg['move_s']*abs(math.sin(motion['reset_yaw_range']))
    if lateral+motion['reset_xy_range'][1]+footprint+motion['boundary_margin'] >= motion['patch_width']/2:
        raise ValueError('CTS straight patch cannot contain lateral reset heading')
    return cfg


def straight_patch_mesh(generator, motion, cfg):
    """A separate real flat mesh, after all ordinary motion patches."""
    import numpy as np
    import trimesh
    from .motion import variants
    from task_support.stair_quiet_v1_width.settings import PROFILES
    validate_straight(cfg, motion)
    start = (generator.num_rows*generator.size[0]/2+motion['patch_gap']
             +len(variants(motion))*(motion['patch_length']+motion['patch_gap']))
    x = start+cfg['patch_length']/2
    origins = np.zeros((len(PROFILES), 3), dtype=np.float32)
    parts = []
    for profile in range(len(PROFILES)):
        y = -generator.num_cols*generator.size[1]/2+(profile+.5)*generator.size[1]
        origins[profile] = (x, y, 0.)
        vertices = [[x+dx, y+dy, 0.] for dx, dy in
                    [(-cfg['patch_length']/2, -motion['patch_width']/2),
                     (cfg['patch_length']/2, -motion['patch_width']/2),
                     (cfg['patch_length']/2, motion['patch_width']/2),
                     (-cfg['patch_length']/2, motion['patch_width']/2)]]
        parts.append(trimesh.Trimesh(vertices=vertices, faces=[[0, 1, 2], [0, 2, 3]], process=False))
    return trimesh.util.concatenate(parts), origins


@torch.no_grad()
def straight_teacher_actions(models, obs, teacher_latent=None):
    """Privileged encoder AND clean decoder common input at the same state."""
    latent = models.teacher_latent(obs) if teacher_latent is None else teacher_latent
    return models.decoder(obs['teacher'][:, :57],latent,obs['motion_control']).detach()


def straight_action_loss(models, data, target_actions, cfg):
    """Same student-occupied state, current clean teacher, frozen decoder.

    Only the student encoder is optimized by the caller. Wheels are measured,
    never included in this new loss. No teacher-cohort trajectories are paired
    with unrelated student trajectories.
    """
    zero = target_actions.new_zeros(())
    metrics = dict(straight_leg_action_mse=0., straight_front_left_action_mse=0.,
                   straight_wheel_action_mse=0., straight_action_samples=0)
    if cfg is None:
        return zero, metrics
    command = data['obs']['motion_control']
    selected = data['straight'] & ~data['teacher']
    selected &= (command[:, 0] > 1.e-4) & (command[:, 1:].abs().amax(-1) <= 1.e-4)
    ids = selected.nonzero().flatten()
    if not len(ids):
        return zero, metrics
    ids = ids[torch.randperm(len(ids), device=ids.device)[:cfg['batch_size']]]
    action = models.student_action(data['obs']['policy'][ids], command[ids])
    residual = action-target_actions[ids].detach()
    leg = residual[:, :12].square().mean()
    metrics.update(straight_leg_action_mse=float(leg.detach()),
                   straight_front_left_action_mse=float(residual[:, :3].square().mean().detach()),
                   straight_wheel_action_mse=float(residual[:, 12:].square().mean().detach()),
                   straight_action_samples=len(ids))
    return cfg['leg_action_loss_weight']*leg, metrics
