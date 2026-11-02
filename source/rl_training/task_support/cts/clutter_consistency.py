"""Training-only counterfactual tall returns on known level support.

Never edits rollout observations, commands, valid/confidence or deployed inputs.
A transient front patch, one side, a doorway, and a narrow post are alternate
measurements of a flat-motion state, not simulated physical pedestrians.
"""
import math
import torch


def validate(c):
    keys = {'enabled', 'weight', 'batch_size', 'probability', 'height_min_m', 'height_max_m',
            'flat_tolerance_m', 'protected_half_length_m', 'protected_half_width_m', 'min_valid_fraction'}
    if not isinstance(c, dict) or set(c) != keys or type(c['enabled']) is not bool:
        raise ValueError('Invalid clutter consistency configuration')
    for k in keys - {'enabled'}:
        if type(c[k]) not in (float, int) or not math.isfinite(c[k]) or c[k] <= 0:
            raise ValueError('Invalid clutter consistency value: ' + k)
    if type(c['batch_size']) is not int or not 0 < c['probability'] <= 1 or not 0 < c['min_valid_fraction'] <= 1:
        raise ValueError('Invalid clutter consistency sampling')
    if not 0 < c['height_min_m'] < c['height_max_m'] or not .3 <= c['protected_half_length_m'] < .8 or not .2 <= c['protected_half_width_m'] < .5:
        raise ValueError('Invalid clutter consistency geometry')


def grid(policy):
    y, x = torch.meshgrid(torch.arange(-5, 6, device=policy.device, dtype=policy.dtype)*.1,
                          torch.arange(-8, 9, device=policy.device, dtype=policy.dtype)*.1, indexing='ij')
    return x.flatten(), y.flatten()


def eligible(data, regular, c):
    p = data['obs']['policy']
    x, y = grid(p)
    # Course IDs are privileged collection tags, never inferred from noisy maps.
    # No staircase (including its landing), pit, block or platform supervision.
    level_course = (data['course'] == 3) | (data['course'] == 8)
    centre = y.abs() <= c['protected_half_width_m']
    clean = data['obs']['teacher'][:, 57:244][:, centre]
    flat = torch.isfinite(clean).all(-1) & (clean.amax(-1)-clean.amin(-1) <= c['flat_tolerance_m'])
    # Count only the forward preview used by this augmentation. Rear cells
    # are intentionally absent with the deployed front-only LiDAR mask.
    preview = centre & (x > c['protected_half_length_m'])
    known = (p[:, 244:431][:, preview] > .5).float().mean(-1) >= c['min_valid_fraction']
    return data['clutter'] & regular & level_course & flat & known & torch.isfinite(p).all(-1)


def perturb(policy, c, *, kinds=None, heights=None):
    """Change known peripheral height cells only; a no-op is not a sample."""
    n = len(policy); x, y = grid(policy)
    if kinds is None: kinds = torch.randint(4, (n,), device=policy.device)
    if heights is None:
        heights = torch.empty(n, 1, device=policy.device).uniform_(c['height_min_m'], c['height_max_m'])
    side = torch.where(torch.rand(n, 1, device=policy.device) < .5, -1., 1.)
    centre = (torch.rand(n, 1, device=policy.device)-.5)*.6
    front = (x[None] > c['protected_half_length_m']) & ((y[None]-centre).abs() <= .20)
    wall = y[None]*side > c['protected_half_width_m']
    doorway = y[None].abs() > c['protected_half_width_m']
    post = (x[None] > c['protected_half_length_m']) & ((y[None]-centre).abs() <= .10)
    mask = torch.where(kinds[:, None] == 0, front,
           torch.where(kinds[:, None] == 1, wall, torch.where(kinds[:, None] == 2, doorway, post)))
    mask &= (policy[:,244:431] > .5) & (policy[:,431:618] > 0.)
    result = policy.clone()
    height = policy[:,57:244]
    # Observation is base_z - surface_z - offset: taller return -> smaller value.
    changed = (height-heights).clamp(-1.,1.)
    mask &= changed != height
    result[:,57:244] = torch.where(mask, changed, height)
    return result, mask.any(-1), kinds


def loss(models, data, ids, regular, c, safety=None):
    zero = data['obs']['policy'].new_zeros(())
    metrics = dict(clutter_leg_mse=0., clutter_samples=0., clutter_eligible=0.)
    if not c or not c['enabled']: return zero, metrics
    candidates = ids[eligible(data, regular, c)[ids]]
    metrics['clutter_eligible'] = len(candidates)
    candidates = candidates[torch.rand(len(candidates), device=ids.device) < c['probability']][:c['batch_size']]
    if not len(candidates): return zero, metrics
    policy, command = (data['obs'][k][candidates] for k in ('policy','motion_control'))
    with torch.no_grad():
        reference = models.student_action(policy, command).detach()
    perturbed, changed, kinds = perturb(policy, c)
    accepted = changed & torch.isfinite(reference).all(-1)
    if safety is not None:
        accepted &= safety.accept(reference)
        # Reject unsafe teacher states as well as unsafe clean student anchors.
        with torch.no_grad():
            accepted &= safety.accept(models.teacher_action({k:v[candidates] for k,v in data['obs'].items()}))
    if not accepted.any(): return zero, metrics
    actions = models.student_action(perturbed[accepted], command[accepted])
    value = (actions[:,:12]-reference[accepted,:12]).square().mean()
    # No wheel invariance: navigation remains free to request slowing/stopping.
    penalty = zero if safety is None else safety.cfg['student_tail_weight']*safety.tail_cost(actions)
    metrics.update(clutter_leg_mse=float(value.detach()), clutter_samples=int(accepted.sum()))
    for k, name in enumerate(('front','side','doorway','post')):
        metrics['clutter_'+name+'_samples'] = int((kinds[accepted] == k).sum())
    return c['weight']*value + penalty, metrics
