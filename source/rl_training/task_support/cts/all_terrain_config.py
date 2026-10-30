"""Validation for the CTS all-terrain numerical ledger (no simulator import)."""
import math

from .all_terrain_curriculum import COURSE_NAMES, validate_curriculum_config
from .all_terrain_geometry import DEFAULTS as GEOMETRY_DEFAULTS

COURSES = COURSE_NAMES
DOMAIN_KEYS = {'static_friction', 'dynamic_ratio', 'mass_scale', 'kp_scale', 'kd_scale',
               'com_offset_m', 'initial_scale'}
COMMAND_KEYS = {'forward_speed', 'backward_speed', 'yaw_rate', 'initial_hold_s', 'move_s',
                'straight_move_s', 'stop_s', 'success_hold_s', 'route_time_margin_s',
                'tracking_rms_m_s', 'yaw_tracking_rms_rad_s', 'stop_speed_m_s', 'stop_yaw_rad_s',
                'terrain_exposure_m', 'relative_tracking_tolerance', 'tracking_floor_m_s',
                'yaw_tracking_floor_rad_s'}
REWARD_KEYS = {'linear', 'yaw', 'progress', 'success', 'tilt', 'action_rate', 'effort',
               'failure', 'slip', 'impact', 'stop', 'min_root_clearance', 'min_upright',
               'boundary_margin', 'flat_air_cost', 'flat_support_cost', 'flat_pose_cost',
               'flat_leg_rate_cost', 'flat_clearance_free_m', 'flat_clearance_scale_m'}
ACTOR_KEYS = {'idle_leg_scale', 'new_domain_anchor_scale'}


def _keys(value, expected, name):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f'Unknown or missing CTS all-terrain {name} settings')


def _number(value, name, *, minimum=0., maximum=None, positive=False, integer=False):
    numeric = type(value) is int if integer else type(value) in (int, float)
    if (not numeric or not math.isfinite(value) or value < minimum
            or (positive and value <= 0) or (maximum is not None and value > maximum)):
        raise ValueError('Invalid CTS all-terrain numeric setting: ' + name)
    return value


def _interval(value, name, *, integer=False, strict=False, minimum=0., maximum=None):
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError('CTS all-terrain interval requires two endpoints: ' + name)
    for endpoint in value:
        _number(endpoint, name, minimum=minimum, maximum=maximum,
                positive=True, integer=integer)
    if value[0] > value[1] or (strict and value[0] == value[1]):
        raise ValueError('CTS all-terrain interval endpoints are not ordered: ' + name)
    return value


def _geometry(cfg):
    _keys(cfg, GEOMETRY_DEFAULTS, 'geometry')
    counts = {'steps_range', 'obstacle_count_range', 'pit_count_range'}
    for key, value in cfg.items():
        if key == 'raised_platform_back_edge':
            _number(value, 'geometry.' + key, minimum=-math.inf)
        elif key.endswith('_range') and isinstance(GEOMETRY_DEFAULTS[key], list):
            _interval(value, 'geometry.' + key, integer=key in counts, strict=True)
        elif type(GEOMETRY_DEFAULTS[key]) is int:
            _number(value, 'geometry.' + key, integer=True,
                    minimum=0 if key == 'seed' else 1)
        else:
            _number(value, 'geometry.' + key, positive=True)
    if cfg['geometry_levels'] > 7:
        raise ValueError('The atlas defines at most seven geometry levels')
    if (not cfg['height_range'][0] < cfg['initial_height_max'] <= cfg['intermediate_height_max'] <= cfg['height_range'][1]
            or not cfg['depth_range'][0] <= cfg['intermediate_depth_min'] <= cfg['initial_depth_min'] < cfg['depth_range'][1]
            or not cfg['width_range'][0] <= cfg['initial_width_min'] < cfg['width_range'][1]
            or not cfg['steps_range'][0] <= cfg['initial_steps_max'] <= cfg['steps_range'][1]):
        raise ValueError('Geometry curriculum intervals must remain inside their full ranges')
    if cfg['tile_length'] < 24. or cfg['tile_width'] < 6.:
        raise ValueError('The current terrain atlas requires a tile of at least 24 m by 6 m')
    if not .025 <= cfg['grid_resolution'] <= .25:
        raise ValueError('Invalid actual terrain mesh resolution')
    if max(cfg['width_range'][1], cfg['platform_width_range'][1]) > cfg['tile_width']:
        raise ValueError('A route cannot be wider than its physical tile')
    xmax = cfg['tile_length'] - cfg['start_backward']
    longest = (cfg['route_start'] + 2 * cfg['steps_range'][1] * cfg['depth_range'][1]
               + cfg['platform_length_range'][1] + cfg['goal_landing'] + cfg['goal_clearance'])
    if longest >= xmax:
        raise ValueError('The longest round trip requires a complete landing inside its physical tile')
    half_length, half_width = cfg['field_half_length'], cfg['tile_width'] / 2
    back = cfg['raised_platform_back_edge']
    if (not -half_length <= back < -cfg['spawn_flat_half_length']
            or back + cfg['raised_platform_length_range'][0] <= cfg['spawn_flat_half_length']
            or back + cfg['raised_platform_length_range'][1] + cfg['goal_landing'] + cfg['goal_clearance'] >= half_length):
        raise ValueError('Raised platform must contain the spawn and provide a complete exit landing')
    if cfg['field_goal'] + cfg['goal_clearance'] >= half_length:
        raise ValueError('A field goal requires a complete landing inside its physical tile')
    if cfg['spawn_flat_half_length'] + cfg['spawn_transition_length'] >= half_length:
        raise ValueError('The flat spawn and transition cannot occupy the whole terrain field')
    for key in ('obstacle_x_range', 'mixed_obstacle_x_range'):
        if cfg[key][1] + cfg['obstacle_length_range'][1] / 2 > half_length:
            raise ValueError('An obstacle extends beyond its physical field')
    if (cfg['obstacle_y_half_range'] + cfg['obstacle_width_range'][1] / 2 > half_width
            or cfg['pit_x_range'][1] + cfg['pit_radius_range'][1] > half_length
            or cfg['pit_y_half_range'] + cfg['pit_radius_range'][1] > half_width):
        raise ValueError('An obstacle or pit extends beyond its physical field')
    if (cfg['obstacle_x_range'][0] - cfg['obstacle_length_range'][1] / 2 <= cfg['spawn_flat_half_length']
            or cfg['pit_x_range'][0] - cfg['pit_radius_range'][1] <= cfg['spawn_flat_half_length']
            or cfg['route_start'] <= cfg['spawn_flat_half_length']):
        raise ValueError('Terrain obstacles must leave the configured flat spawn support area clear')
    if cfg['mixed_ramp_start'] + cfg['mixed_ramp_length'] > cfg['mixed_obstacle_x_range'][0] - cfg['obstacle_length_range'][1] / 2:
        raise ValueError('Mixed-terrain obstacles must start after the ramp plateau')


def validate_all_terrain(cfg):
    if not isinstance(cfg, dict):
        raise ValueError('CTS all-terrain settings must be a dictionary')
    sections = {'enabled', 'fractions', 'geometry', 'curriculum', 'domain', 'commands', 'rewards', 'actor', 'simulation'}
    stability = {'motion_stability', 'startup_stability'} & set(cfg)
    optional = stability | ({'hind_clearance', 'compound_commands', 'wall_practice'} & set(cfg))
    if stability and stability != {'motion_stability', 'startup_stability'}:
        raise ValueError('CTS stability curriculum and startup objectives must be versioned together')
    _keys(cfg, sections | optional, 'sections')
    if 'hind_clearance' in cfg:
        from .hind_clearance import validate_hind_clearance
        validate_hind_clearance(cfg['hind_clearance'])
    if type(cfg['enabled']) is not bool:
        raise ValueError('CTS all-terrain enabled must be a boolean')
    fractions = cfg['fractions']
    if not isinstance(fractions, (tuple, list)) or len(fractions) != len(COURSES):
        raise ValueError('CTS all-terrain requires a fraction for each course')
    for value in fractions:
        _number(value, 'fractions', positive=True, maximum=1.)
    if not math.isclose(math.fsum(fractions), 1., rel_tol=0., abs_tol=1e-9):
        raise ValueError('All-terrain course fractions must sum to one')
    _geometry(cfg['geometry'])
    if 'compound_commands' in cfg:
        from .compound_commands import validate_compound_commands
        validate_compound_commands(cfg)
    if 'wall_practice' in cfg:
        from .wall_practice import validate
        validate(cfg['wall_practice'])
        if cfg['wall_practice']['enabled'] and 'compound_commands' not in cfg:
            raise ValueError('Wall practice requires the v15 command ledger')
    validate_curriculum_config(cfg['curriculum'])
    if cfg['geometry']['geometry_levels'] != cfg['curriculum']['geometry_levels']:
        raise ValueError('All-terrain geometry/curriculum dimensions differ')
    if cfg['curriculum']['dr_levels'] != 3:
        raise ValueError('Current continuous physics interpolation defines three DR levels')
    domain = cfg['domain']
    _keys(domain, DOMAIN_KEYS, 'domain')
    for key in DOMAIN_KEYS - {'com_offset_m', 'initial_scale'}:
        _interval(domain[key], 'domain.' + key)
    if domain['dynamic_ratio'][1] > 1:
        raise ValueError('Dynamic friction cannot exceed static friction')
    _number(domain['com_offset_m'], 'domain.com_offset_m')
    _number(domain['initial_scale'], 'domain.initial_scale', maximum=1.)
    commands = cfg['commands']
    _keys(commands, COMMAND_KEYS, 'commands')
    for key in ('forward_speed', 'backward_speed', 'yaw_rate'):
        _interval(commands[key], 'commands.' + key)
    for key in COMMAND_KEYS - {'forward_speed', 'backward_speed', 'yaw_rate'}:
        _number(commands[key], 'commands.' + key, positive=True)
    if not 0 < commands['relative_tracking_tolerance'] < 1:
        raise ValueError('Relative tracking tolerance must distinguish no motion from commanded motion')
    if commands['success_hold_s'] > commands['stop_s']:
        raise ValueError('The stop phase must allow the success stability hold')
    if commands['terrain_exposure_m'] >= cfg['geometry']['field_half_length']:
        raise ValueError('Required motion exposure exceeds the physical terrain field')
    rewards = cfg['rewards']
    _keys(rewards, REWARD_KEYS, 'rewards')
    for key, value in rewards.items():
        _number(value, 'rewards.' + key,
                positive=key in ('min_root_clearance', 'min_upright', 'flat_clearance_scale_m'))
    if rewards['min_upright'] > 1:
        raise ValueError('Minimum upright projection cannot exceed one')
    if rewards['boundary_margin'] >= cfg['geometry']['tile_width'] / 2:
        raise ValueError('Boundary margin leaves no traversable tile width')
    actor = cfg['actor']
    _keys(actor, ACTOR_KEYS, 'actor')
    for key, value in actor.items():
        _number(value, 'actor.' + key, maximum=1., positive=key == 'idle_leg_scale')
    simulation = cfg['simulation']
    _keys(simulation, {'collision_stack_mib'}, 'simulation')
    _number(simulation['collision_stack_mib'], 'simulation.collision_stack_mib',
            integer=True, minimum=64, maximum=2048)
    if stability:
        from .stability import validate_stability
        validate_stability(cfg)
    return cfg
