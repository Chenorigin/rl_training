"""Versioned training coverage for peripheral map quality and supported stops.

Never fills unknown terrain or changes deployment observations. Collection tags
describe the pre-action physical state; raw motion_control owns the idle gate.
"""
import math
import torch


def settings(cfg):
    c = (cfg or {}).get('wall_stability')
    return c if c and c['enabled'] else None


def validate(c):
    keys = {'enabled', 'quality_weight', 'wheel_consistency_weight', 'idle_output_weight',
            'batch_size', 'probability', 'protected_half_width_m', 'flat_tolerance_m',
            'min_valid_fraction', 'minimum_supports', 'minimum_upright',
            'idle_speed_m_s', 'idle_yaw_rad_s', 'stair_initial_hold_s', 'preview_start_m'}
    if not isinstance(c, dict) or set(c) != keys or type(c['enabled']) is not bool:
        raise ValueError('Invalid CTS wall stability configuration')
    for key in keys - {'enabled'}:
        x = c[key]
        if type(x) not in (int, float) or not math.isfinite(x) or x <= 0:
            raise ValueError('Invalid wall stability value: ' + key)
    if (type(c['batch_size']) is not int or type(c['minimum_supports']) is not int
            or not 3 <= c['minimum_supports'] <= 4
            or not .2 <= c['protected_half_width_m'] < .5
            or not .3 <= c['preview_start_m'] < .8
            or not 0 < c['minimum_upright'] <= 1
            or not 0 < c['min_valid_fraction'] <= 1 or not 0 < c['probability'] <= 1):
        raise ValueError('Invalid wall stability support/sampling bounds')


def initial_hold(state):
    old = state.cfg['commands']['initial_hold_s']
    c = settings(state.cfg)
    if c is None:
        return old
    # Match double-precision episode ages, including the exact legacy hold boundary.
    return torch.where(state.stairs,
        torch.full(state.stairs.shape, c['stair_initial_hold_s'], device=state.stairs.device, dtype=torch.float64), old)


def level_support(state, g):
    c = settings(state.cfg)
    if c is None:
        return torch.zeros_like(state.stairs)
    height, normal = g['height'], g['normal']
    q = state.env.scene['robot'].data.root_quat_w
    upright = 1 - 2 * (q[:, 1].square() + q[:, 2].square())
    return ((g['supported'].sum(-1) >= c['minimum_supports'])
            & torch.isfinite(height).all(-1) & torch.isfinite(normal).all(dim=(-1, -2))
            & ((height.amax(-1) - height.amin(-1)) <= c['flat_tolerance_m'])
            & (normal[..., 2].amin(-1) >= c['minimum_upright'])
            & (upright >= c['minimum_upright']) & ~state.failed)


def collection_tags(state, g, raw):
    c = settings(state.cfg)
    supported = level_support(state, g)
    allowed = state.stairs | (state.course_ids == 3) | (state.course_ids == 8)
    velocity = g['measured_velocity']
    idle = ((raw.abs().amax(-1) <= 1.e-4)
            & (velocity[:, :2].norm(dim=-1) <= c['idle_speed_m_s'])
            & (velocity[:, 2].abs() <= c['idle_yaw_rad_s']))
    return {'cts_wall_quality': supported & allowed,
            'cts_wall_idle': supported & allowed & idle}


def eligible(data, ids, c, tag):
    from .clutter_consistency import grid
    p = data['obs']['policy'][ids]
    x, y = grid(p)
    centre = y.abs() <= c['protected_half_width_m']
    clean = data['obs']['teacher'][ids, 57:244][:, centre]
    flat = torch.isfinite(clean).all(-1) & ((clean.amax(-1)-clean.amin(-1)) <= c['flat_tolerance_m'])
    preview = centre & (x > c['preview_start_m'])
    known = ((p[:, 244:431][:, preview] > .5) & (p[:, 431:618][:, preview] > 0.)).float().mean(-1)
    raw_idle = data['obs']['motion_control'][ids].abs().amax(-1) <= 1.e-4
    return (data[tag][ids] & flat & (known >= c['min_valid_fraction'])
            & torch.isfinite(p).all(-1) & (raw_idle if tag == 'wall_idle' else True))


def perturb(policy, c, *, kinds=None):
    """Occlude existing peripheral returns, or change only their confidence.

    Known and degraded views share a detached clean anchor. A return becoming
    visible again uses that original measurement; original unknowns stay unknown.
    """
    from .clutter_consistency import grid
    n = len(policy)
    _, y = grid(policy)
    if kinds is None:
        kinds = torch.randint(3, (n,), device=policy.device)
    side = torch.where(torch.rand(n, 1, device=policy.device) < .5, -1., 1.)
    known = (policy[:, 244:431] > .5) & (policy[:, 431:618] > 0.)
    mask = known & (y[None]*side > c['protected_half_width_m'])
    sparse = torch.rand_like(policy[:, 57:244]) < .5
    mask &= (kinds[:, None] != 0) | sparse
    result = policy.clone()
    drop = mask & (kinds[:, None] < 2)
    for start in (57, 244, 431):
        result[:, start:start+187] = torch.where(drop, 0., result[:, start:start+187])
    quality = mask & (kinds[:, None] == 2)
    # [0,1) factor never invents a higher-confidence or newly known surface.
    factor = torch.rand(n, 1, device=policy.device)
    result[:, 431:618] = torch.where(quality, policy[:, 431:618]*factor, result[:, 431:618])
    return result, (result[:, 57:618] != policy[:, 57:618]).any(-1), kinds


def idle_loss(actions, selected, c):
    values = actions[:, 12:].square().mean(-1)
    return c['idle_output_weight']*(values*selected).sum()/selected.sum().clamp_min(1)


def student_loss(models, data, ids, c, safety=None):
    zero = data['obs']['policy'].new_zeros(())
    metrics = dict(wall_quality_samples=0, wall_quality_leg_mse=0., wall_quality_wheel_mse=0.,
                   wall_idle_samples=0, wall_idle_output_loss=0.)
    if c is None:
        return zero, metrics
    idle = eligible(data, ids, c, 'wall_idle')
    actions = models.student_action(data['obs']['policy'][ids], data['obs']['motion_control'][ids])
    stopping = idle_loss(actions, idle, c)
    metrics.update(wall_idle_samples=int(idle.sum()), wall_idle_output_loss=float(stopping.detach()))
    selected = ids[eligible(data, ids, c, 'wall_quality')]
    selected = selected[torch.rand(len(selected), device=ids.device) < c['probability']][:c['batch_size']]
    if not len(selected):
        return stopping, metrics
    p, raw = (data['obs'][key][selected] for key in ('policy', 'motion_control'))
    with torch.no_grad():
        reference = models.student_action(p, raw).detach()
        accepted = torch.isfinite(reference).all(-1)
        if safety is not None:
            accepted &= safety.accept(reference)
            accepted &= safety.accept(models.teacher_action({key:v[selected] for key,v in data['obs'].items()}))
    changed, active, kinds = perturb(p, c)
    accepted &= active
    if not accepted.any():
        return stopping, metrics
    predicted = models.student_action(changed[accepted], raw[accepted])
    error = (predicted-reference[accepted]).square()
    legs = error[:, :12].mean()
    # Same raw zero request: navigation remains free during every moving command.
    idle_pair = raw[accepted].abs().amax(-1) <= 1.e-4
    wheels = (error[:, 12:].mean(-1)*idle_pair).sum()/idle_pair.sum().clamp_min(1)
    metrics.update(wall_quality_samples=int(accepted.sum()), wall_quality_leg_mse=float(legs.detach()),
                   wall_quality_wheel_mse=float(wheels.detach()))
    for index, name in enumerate(('sparse', 'side', 'confidence')):
        metrics['wall_quality_'+name+'_samples'] = int((kinds[accepted] == index).sum())
    return stopping+c['quality_weight']*legs+c['wheel_consistency_weight']*wheels, metrics
