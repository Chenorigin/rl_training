"""Same ten physical widths and four-wheel safety, new teacher input routing."""
from isaaclab.utils import configclass
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg, RewardTermCfg
from .stair_quiet_v1_width_env_cfg import DeeproboticsM20StairQuietV1WidthEnvCfg
from task_support.stair_quiet_v1_safe.settings import SafeSettings
from task_support.stair_quiet_v1_safe.observations import locomotion_scan, teacher_safety
from task_support.stair_quiet_v1_safe.course import reset_course
from task_support.stair_quiet_v1_width.signals import tail_cost


@configclass
class LongitudinalScanCfg(ObservationGroupCfg):
    scan = ObservationTermCfg(func=locomotion_scan)
    enable_corruption = False
    concatenate_terms = True


@configclass
class TeacherSafetyCfg(ObservationGroupCfg):
    safety = ObservationTermCfg(func=teacher_safety)
    enable_corruption = False
    concatenate_terms = True


@configclass
class DeeproboticsM20StairQuietV1SafeEnvCfg(DeeproboticsM20StairQuietV1WidthEnvCfg):
    stair_tail: SafeSettings = SafeSettings()

    def __post_init__(self):
        super().__post_init__()
        pattern = self.scene.height_scanner.pattern_cfg
        if tuple(pattern.size) != (1.6, 1.0) or pattern.resolution != .1:
            raise ValueError("Safe longitudinal routing requires the teacher's 17 by 11 scan grid")
        self.observations.locomotion_scan = LongitudinalScanCfg()
        self.observations.safety = TeacherSafetyCfg()
        self.events.randomize_reset_base.func = reset_course
        self.rewards.tail_landing_risk = RewardTermCfg(func=tail_cost,
            weight=self.stair_tail.landing_risk_weight, params={"name": "landing_risk"})
        self.m20_refinement_profile = "stair_quiet_v1_safe"
