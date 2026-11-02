"""CTS identity and one numerical ledger, independent of legacy distillation."""
from copy import deepcopy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
TASK = "Rough-Deeprobotics-M20-CTS"
CONFIG_PATH = ROOT / "configs/distillation/cts.json"
VERSION = "m20_cts_action_latent_v26"
LEGACY_VERSION = "m20_cts_action_latent_v2"
LEGACY_VERSIONS = (LEGACY_VERSION, "m20_cts_action_latent_v3", "m20_cts_action_latent_v4", "m20_cts_action_latent_v5", "m20_cts_action_latent_v6", "m20_cts_action_latent_v7", "m20_cts_action_latent_v8", "m20_cts_action_latent_v9", "m20_cts_action_latent_v10", "m20_cts_action_latent_v11", "m20_cts_action_latent_v12", "m20_cts_action_latent_v13", "m20_cts_action_latent_v14", "m20_cts_action_latent_v15", "m20_cts_action_latent_v16", "m20_cts_action_latent_v17", "m20_cts_action_latent_v18", "m20_cts_action_latent_v19", "m20_cts_action_latent_v20", "m20_cts_action_latent_v21", "m20_cts_action_latent_v22", "m20_cts_action_latent_v23", "m20_cts_action_latent_v24", "m20_cts_action_latent_v25")


TEACHER_TRAINING_KEYS = {'train_teacher_base', 'teacher_learning_rate'}


def train_teacher_base(cfg):
    """Missing paired settings retain the pre-v16 frozen-base contract."""
    import math
    present = TEACHER_TRAINING_KEYS.intersection(cfg)
    if not present:
        return False
    if present != TEACHER_TRAINING_KEYS or type(cfg['train_teacher_base']) is not bool:
        raise ValueError('CTS teacher training requires a boolean mode and learning rate')
    rate = cfg['teacher_learning_rate']
    if type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0:
        raise ValueError('CTS teacher_learning_rate must be finite and positive')
    return cfg['train_teacher_base']


def algorithm_version(metadata_version):
    return {LEGACY_VERSION: 2, **{version:3 for version in (*LEGACY_VERSIONS[1:],VERSION)}}[metadata_version]


def default_settings():
    return json.loads(CONFIG_PATH.read_text())


class CTSConfig:
    def to_dict(self):
        return deepcopy(default_settings())


def validate_config(config, *, legacy=False):
    from .auxiliary import validate_config as validate_auxiliary
    expected = default_settings()
    if not isinstance(config, dict):
        raise ValueError('CTS configuration must be a dictionary')
    keys = set(expected)-({key for key in ('backward_practice', 'platform_practice','sample_stairs','legacy_stairs','motion_practice','map_delay_mixture','map_capture','straight_practice','all_terrain','joint_limits') if key not in config} if legacy else set())
    # Additive v15 training feature: pre-feature v15 checkpoints remain readable.
    # Resume still requires exact saved source hashes; --init-cts adopts defaults.
    if 'map_recovery' not in config:
        keys.discard('map_recovery')
    if legacy and 'backward_practice' in config:
        keys.add('backward_practice')
    algorithm_keys = set(expected['algorithm'])
    if legacy and isinstance(config.get('algorithm'), dict) and not TEACHER_TRAINING_KEYS.intersection(config['algorithm']):
        algorithm_keys -= TEACHER_TRAINING_KEYS
    if legacy and 'stop_exploration_scale' not in config.get('algorithm', {}):
        algorithm_keys.discard('stop_exploration_scale')
    if (not isinstance(config, dict) or set(config) != keys
            or not isinstance(config['algorithm'], dict)
            or set(config['algorithm']) != algorithm_keys):
        raise ValueError('Unknown or missing CTS configuration keys')
    from .algorithm import validate_config as validate_algorithm
    validate_algorithm(config['algorithm'])
    validate_auxiliary(config['auxiliary'])
    if 'map_recovery' in config:
        from .map_recovery import validate_recovery
        validate_recovery(config['map_recovery'])
        if config['map_recovery']['enabled'] and not config.get('all_terrain', {}).get('enabled'):
            raise ValueError('CTS map recovery requires the all-terrain environment')
    if 'joint_limits' in config:
        from .joint_limits import validate_limits
        validate_limits(config['joint_limits'])
        if not config.get('all_terrain', {}).get('enabled'):
            raise ValueError('CTS software joint limits require the all-terrain environment')
    if 'all_terrain' in config:
        from .all_terrain_config import validate_all_terrain
        validate_all_terrain(config['all_terrain'])
        if not legacy and config['all_terrain']['enabled'] and not all(
                key in config['all_terrain'] for key in ('motion_stability', 'startup_stability')):
            raise ValueError('CTS v14 requires online motion and startup stability settings')
    if 'straight_practice' in config:
        if 'motion_practice' not in config:raise ValueError('CTS straight course requires motion settings')
        from .straight import validate_straight
        validate_straight(config['straight_practice'], config['motion_practice'])
    if 'map_capture' in config:
        from .perception import validate_capture
        validate_capture(config['map_capture'])
    if 'map_delay_mixture' in config:
        from .perception import validate_mixture
        validate_mixture(config['map_delay_mixture'])
    if 'motion_practice' in config:
        from .motion import validate_motion
        validate_motion(config['motion_practice'])
        if sum(config[k]['fraction'] for k in ('backward_practice','platform_practice','motion_practice') if k in config) >= 1:
            raise ValueError('CTS practice must leave staircase environments')
    if 'legacy_stairs' in config:
        from .terrain import validate_legacy
        validate_legacy(config['legacy_stairs'])
    if 'sample_stairs' in config:
        from .sample_stairs import validate_sample
        validate_sample(config['sample_stairs'])
    if 'backward_practice' in config:
        from .practice import validate_practice
        validate_practice(config['backward_practice'])
    if 'platform_practice' in config:
        from .platform import validate_platform
        validate_platform(config['platform_practice'],legacy=legacy)
    for name in ('num_steps_per_env', 'max_iterations', 'save_interval'):
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError(f'CTS {name} must be a positive integer')
    if type(config['seed']) is not int or config['seed'] < 0:
        raise ValueError('CTS seed must be a nonnegative integer')
    return config
