"""Optional v15 command coverage and axiswise tracking, with legacy fallback."""
import math
import torch

# Preserve mode 0..10 identities. Rows 11..18 add xy and lateral/yaw pairs.
SIGNS = ((1,0,0),(-1,0,0),(0,0,1),(0,0,-1),
         (1,0,1),(1,0,-1),(-1,0,1),(-1,0,-1),(0,0,0),(0,1,0),(0,-1,0),
         (1,1,0),(1,-1,0),(-1,1,0),(-1,-1,0),
         (0,1,1),(0,1,-1),(0,-1,1),(0,-1,-1))


def validate_compound_commands(all_terrain):
    cfg = all_terrain['compound_commands']
    keys = {'enabled', 'forward_speed', 'backward_speed', 'lateral_speed', 'yaw_rate',
            'travel_m', 'transition_probability'}
    if not isinstance(cfg, dict) or set(cfg) != keys or type(cfg['enabled']) is not bool:
        raise ValueError('Invalid CTS compound command settings')
    for name in ('forward_speed', 'backward_speed', 'lateral_speed', 'yaw_rate'):
        v = cfg[name]
        if (not isinstance(v, (list, tuple)) or len(v) != 2
                or any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in v)
                or v[0] > v[1]):
            raise ValueError('Invalid CTS compound command interval: '+name)
    for name in ('travel_m', 'transition_probability'):
        v = cfg[name]
        if type(v) not in (int, float) or not math.isfinite(v) or v <= 0:
            raise ValueError('Invalid CTS compound command value: '+name)
    if cfg['transition_probability'] >= 1:
        raise ValueError('Compound commands must retain fixed and transitioning pairs')
    if cfg['enabled'] and not all_terrain.get('motion_stability'):
        raise ValueError('Compound commands require motion stability')
    # Actual footprint is checked against each collision tile at reset.
    if 2*cfg['travel_m'] >= all_terrain['geometry']['tile_width']:
        raise ValueError('Compound travel cannot fit the terrain field')


def sample_commands(mode, cfg):
    signs = torch.tensor(SIGNS, device=mode.device, dtype=torch.float32)[mode]
    def draw(key):
        return torch.empty(len(mode), device=mode.device).uniform_(*cfg[key])
    x = torch.where(signs[:, 0] > 0, draw('forward_speed'), draw('backward_speed'))
    return signs*torch.stack((x, draw('lateral_speed'), draw('yaw_rate')), -1)


def compound_sequence(command, elapsed, duration, ramp, transition, secondary_last):
    """First axis -> pair -> first axis, retaining an optional constant pair.

    Both translation and yaw are body-frame commands. Each episode independently
    samples its constituent axis order, separately from all command signs.
    Smooth rise/fall remains inside the allotted move phase.
    """
    from .command_sequences import _step, command_sequence
    t = elapsed.to(command.dtype)
    result = command_sequence(command, t, duration, ramp, torch.zeros_like(transition))
    active = command.abs() > 0
    first = active.long().argmax(-1)
    last = (active.long()*torch.arange(3, device=command.device)).argmax(-1)
    secondary = torch.where(secondary_last, last, first)
    envelope = (_step(t-duration/4, ramp)-_step(t-3*duration/4, ramp)).clamp(0, 1)
    factor = torch.where(transition, envelope, torch.ones_like(envelope))
    result.scatter_(1, secondary[:, None], result.gather(1, secondary[:, None])*factor[:, None])
    return result


def reserve_compound_path(xy, bounds, travel, tail, boundary_margin):
    """Any yaw history fits: speed integral <= travel, plus rotating footprint."""
    footprint = math.hypot(tail.body_half_length, tail.body_half_width)
    footprint += tail.wheel_radius+tail.wheel_half_length+boundary_margin
    margin = footprint+travel
    low = bounds[:, (0, 2)]+margin
    high = bounds[:, (1, 3)]-margin
    if (low > high).any():
        raise ValueError('Terrain field cannot contain compound path and support footprint')
    return torch.maximum(low, torch.minimum(high, xy))


def tracking_rewards(state, geometry, legacy_linear, legacy_yaw):
    """Worst active normalized axis gates BOTH reward budgets on motion courses.

    This prevents perfect forward tracking from hiding an ignored yaw/side
    command. Inactive axes retain the legacy zero-target drift variance.
    Bounds are [0,1], total reward weights remain unchanged; idle/other courses
    keep their previous objectives. This function does not read key names.
    """
    command = geometry['command']
    measured = geometry['measured_velocity']
    cfg = state.cfg['commands']
    floor = command.new_tensor([cfg['tracking_floor_m_s'], cfg['tracking_floor_m_s'],
                                cfg['yaw_tracking_floor_rad_s']])
    active = command.abs() > 1.e-4
    sigma = torch.maximum(command.abs()*cfg['relative_tracking_tolerance'], floor)
    sigma = torch.where(active, sigma, .25**.5)
    target = geometry['tracking_command'] if getattr(state, 'objectives', None) is not None else command
    normalized = ((measured-target)/sigma).square()
    score = torch.exp(-normalized.amax(-1))
    use = ((state.course_ids >= 3) & (state.course_ids <= 7) & active.any(-1))
    return torch.where(use, score, legacy_linear), torch.where(use, score, legacy_yaw)


def axis_tracking_ok(error_sum, command_energy, frames, cfg):
    """Per-axis RMS acceptance using the actual ramped/partial command sequence."""
    count = frames.clamp_min(1)[:, None]
    rms = (command_energy/count).sqrt()
    floor = rms.new_tensor([cfg['tracking_floor_m_s']]*2+[cfg['yaw_tracking_floor_rad_s']])
    ceiling = rms.new_tensor([cfg['tracking_rms_m_s']]*2+[cfg['yaw_tracking_rms_rad_s']])
    tolerance = torch.minimum(torch.maximum(rms*cfg['relative_tracking_tolerance'], floor), ceiling)
    tolerance = torch.where(command_energy > 0, tolerance, ceiling)
    return ((error_sum/count).sqrt() <= tolerance).all(-1)
