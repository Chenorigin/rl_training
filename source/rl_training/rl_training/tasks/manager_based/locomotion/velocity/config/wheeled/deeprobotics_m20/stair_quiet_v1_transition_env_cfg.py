"""v1 follow-up: consistent stop scan plus physical entry diversity."""
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg
from .stair_quiet_v1_safe_env_cfg import DeeproboticsM20StairQuietV1SafeEnvCfg
from task_support.stair_quiet_v1_transition.settings import TransitionSettings, GEOMETRIES, PROFILES
from task_support.stair_quiet_v1_transition.terrain import TransitionStairsCfg, TransitionTerrainImporter
from task_support.stair_quiet_v1_transition.course import reset_course
from task_support.stair_quiet_v1_transition.commands import TransitionAscentCommand
from task_support.stair_quiet_v1_transition.observations import locomotion_scan, teacher_safety
from task_support.stair_quiet_v1_transition.signals import transition_cost


@configclass
class DeeproboticsM20StairQuietV1TransitionEnvCfg(DeeproboticsM20StairQuietV1SafeEnvCfg):
    stair_tail: TransitionSettings = TransitionSettings()

    def __post_init__(self):
        super().__post_init__()
        s = self.stair_tail
        self.scene.terrain.class_type = TransitionTerrainImporter
        gen = self.scene.terrain.terrain_generator
        gen.num_cols = len(GEOMETRIES)*len(PROFILES)
        gen.sub_terrains = {f"{g.name}__{p.name}": TransitionStairsCfg(
            proportion=1., geometry_index=gi, profile_index=pi, settings=s)
            for gi, g in enumerate(GEOMETRIES) for pi, p in enumerate(PROFILES)}
        self.commands.base_velocity.class_type = TransitionAscentCommand
        self.events.randomize_reset_base.func = reset_course
        self.observations.locomotion_scan.scan.func = locomotion_scan
        self.observations.safety.safety.func = teacher_safety
        for name in ("riser_work", "riser_descent"):
            setattr(self.rewards, "transition_"+name, RewardTermCfg(func=transition_cost,
                weight=getattr(s, name+"_weight"), params={"name": name}))
        self.m20_refinement_profile = "stair_quiet_v1_transition"
