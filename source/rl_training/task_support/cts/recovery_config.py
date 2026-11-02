"""Strict optional v17 extensions; absence preserves older checkpoints."""
import math


KEYS = {
    'safety_learning': {'enabled','teacher_weight','student_tail_weight','tail_fraction'},
    'stair_recovery': {'enabled','near_start_probability','front_extent_m','start_gap_m','speed_m_s',
        'pause_probability','pause_step','settle_s','pause_timeout_s','heading_weight','lateral_weight',
        'heading_scale_rad','lateral_scale_m','stall_s','progress_epsilon_m','stall_weight',
        'perturb_probability','perturb_yaw_rate_rad_s','perturb_lateral_m_s'},
    'obstacle_response': {'enabled','finite_wall_half_length_m','post_half_width_m',
        'leg_delta_weight','leg_jerk_weight'},
}


def validate_extensions(cfg):
    for name, keys in KEYS.items():
        if name not in cfg:
            continue
        c = cfg[name]
        if not isinstance(c,dict) or set(c) != keys or type(c['enabled']) is not bool:
            raise ValueError('Invalid recovery extension: '+name)
        for key, value in c.items():
            if key == 'enabled': continue
            interval = key in ('start_gap_m','speed_m_s','pause_step')
            if interval != isinstance(value,list):
                raise ValueError('Invalid recovery field shape: '+key)
            values = value if interval else [value]
            if any(type(v) not in (int,float) or not math.isfinite(v) or v < 0 for v in values):
                raise ValueError('Invalid recovery value: '+key)
            if isinstance(value, list):
                if len(value) != 2:
                    raise ValueError('Invalid recovery vector: '+key)
                if value[0] > value[1]:
                    raise ValueError('Unordered recovery interval: '+key)
            if (key.endswith('probability') or key == 'tail_fraction') and not 0 < value <= 1:
                raise ValueError('Invalid recovery probability: '+key)
            if key != 'pause_step' and any(v <= 0 for v in values):
                raise ValueError('Recovery values must be positive: '+key)
    c = cfg.get('stair_recovery')
    if c:
        if any(type(v) is not int for v in c['pause_step']) or not 0 <= c['pause_step'][0] <= c['pause_step'][1] <= 2:
            raise ValueError('Restart course must exercise the first three steps')
        if c['settle_s'] >= c['pause_timeout_s'] or c['speed_m_s'][0] <= .1:
            raise ValueError('Restart needs a settling budget and commands above heading deadband')
        if c['front_extent_m']+c['start_gap_m'][1] >= cfg['geometry']['route_start']:
            raise ValueError('Near-start footprint must fit before first stair')
    c = cfg.get('obstacle_response')
    if c and c['enabled']:
        if not cfg.get('wall_practice',{}).get('enabled'):
            raise ValueError('Obstacle response requires physical wall courses')
        if c['post_half_width_m'] > c['finite_wall_half_length_m']:
            raise ValueError('Post must be shorter than the finite wall')
    return cfg
