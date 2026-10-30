"""CTS observation extension without changing the frozen source contracts."""
from copy import deepcopy

from isaaclab.utils import configclass

from .stair_quiet_v1_safe_env_cfg import DeeproboticsM20StairQuietV1SafeEnvCfg
from .student_v3_safe_env_cfg import DeeproboticsM20StudentV3SafeEnvCfg


@configclass
class DeeproboticsM20CTSEnvCfg(DeeproboticsM20StudentV3SafeEnvCfg):
    """Student physics and sensors with a separate privileged value input."""
    cts_platform_practice: dict = {}
    cts_sample_stairs: dict = {}
    cts_legacy_stairs: dict = {}
    cts_motion_practice: dict = {}
    cts_straight_practice: dict = {}
    cts_sample_automatic_stages: bool = True
    cts_cohort_config: dict = {}
    cts_all_terrain: dict = {}

    def __post_init__(self):
        super().__post_init__()
        # StudentV3Safe replaces critic with its packed sensor input. Recover
        # the complete original group, including ordered joints and friction,
        # from a fresh Safe configuration rather than copying its implementation.
        teacher_cfg = DeeproboticsM20StairQuietV1SafeEnvCfg()
        self.observations.teacher_critic = deepcopy(teacher_cfg.observations.critic)
        from task_support.cts.configuration import default_settings
        from task_support.cts.practice_isaac import configure_practice
        settings = default_settings()
        configure_practice(self, settings['algorithm'], settings['platform_practice'],settings['sample_stairs'],settings['legacy_stairs'],settings['motion_practice'],settings['map_delay_mixture'],settings['straight_practice'])
