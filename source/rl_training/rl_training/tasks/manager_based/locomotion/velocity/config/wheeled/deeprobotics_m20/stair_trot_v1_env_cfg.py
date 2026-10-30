"""Independent ascent trot task; retain the complete teacher observation ABI."""
from isaaclab.managers import (
    ObservationGroupCfg, ObservationTermCfg, RewardTermCfg, CurriculumTermCfg, TerminationTermCfg,
)
from isaaclab.utils import configclass

from rl_training.tasks.manager_based.locomotion.velocity.mdp import stair_trot_v1_env as trot
from task_support.stair_trot_v1.signals import TrotSettings
from task_support.stair_trot_v1.terrain import AscentStairsTerrainCfg
from task_support.stair_trot_v1 import course
from .idle_stop_motion_v1_env_cfg import DeeproboticsM20IdleStopMotionV1EnvCfg, _gait_entities


def _entities():
    return {key: value for key, value in _gait_entities().items() if key not in ("hip_cfg", "knee_cfg")}


@configclass
class TrotControlCfg(ObservationGroupCfg):
    state = ObservationTermCfg(func=trot.trot_control, params=_entities())
    concatenate_terms = True
    enable_corruption = False


@configclass
class DeeproboticsM20StairTrotV1EnvCfg(DeeproboticsM20IdleStopMotionV1EnvCfg):
    trot: TrotSettings = TrotSettings()
    stair_course: course.StairCourseSettings = course.StairCourseSettings()

    def __post_init__(self):
        super().__post_init__()
        self.observations.trot_control = TrotControlCfg()
        # These old terms reward a particular front/rear landing sequence,
        # rather than simultaneous FL+HR / FR+HL swings.
        for name in (
            "stair_front_wheel_lift", "stair_front_wheel_touchdown",
            "stair_front_single_support", "stair_diagonal_support_symmetry",
            "stair_front_alternation", "stair_up_step_completion",
            "stair_up_rear_step_completion", "stair_rear_wheel_lift",
            "stair_rear_transition_recovery", "stair_rear_style_refund",
            "stair_up_order_violation_cost", "stair_up_same_tread_cost",
            "stair_up_same_tread_dwell_cost",
            # Front/rear default joint signs differ; diagonal phase is not
            # raw joint-angle equality. Replace these conflicting/weak costs.
            "joint_mirror", "stair_orientation_cost", "stair_impact_cost",
            "stair_mechanical_power_cost", "action_smooth_l2",
            "stair_ascent_direction_cost", "stair_ascent_heading_cost",
            "stair_ascent_straight_motion_cost",
        ):
            setattr(self.rewards, name, None)
        for metric, weight in (
            ("phase", 3.0), ("diagonal", 2.0), ("events", 0.15),
            ("pair_mismatch", -1.0), ("support_deficit", -2.0),
            ("stance_missing", -0.5),
            ("rear_slip", -1.0), ("front_slip", -0.4),
            ("posture", -0.5), ("impact", -0.2), ("edge_touch", -0.1),
            ("jerk", -0.025), ("route", -1.0), ("power", -0.1),
        ):
            setattr(self.rewards, f"trot_{metric}", RewardTermCfg(
                func=trot.trot_reward, weight=weight, params={**_entities(), "metric": metric}))
        self.curriculum.trot = CurriculumTermCfg(func=trot.trot_metrics)
        # Learn easy risers first. Keep flat, descent, slopes and rough surfaces.
        self.scene.terrain.max_init_terrain_level = 0
        # 40 columns: 24 dedicated ascent, 8 flat, and 8 legacy terrain.
        generator = self.scene.terrain.terrain_generator
        generator.num_cols = 40
        generator.curriculum = True
        proportions = {"pyramid_stairs": .05, "pyramid_stairs_inv": .05, "flat": .20,
                       "boxes": .025, "random_rough": .025,
                       "hf_pyramid_slope": .025, "hf_pyramid_slope_inv": .025}
        for name, proportion in proportions.items():
            generator.sub_terrains[name].proportion = proportion
        for riser, prefix in ((True, "solid"), (False, "open")):
            for label, depth in (("short", .25), ("medium", .35), ("long", .50)):
                generator.sub_terrains[f"ascent_{prefix}_{label}"] = AscentStairsTerrainCfg(
                    proportion=.10, step_height_range=(.08, .24),
                    step_depth_range=(depth, depth), riser=riser)
        self.commands.base_velocity.class_type = course.StairAscentVelocityCommand
        self.events.randomize_reset_base.func = course.reset_root_on_stair_course
        self.curriculum.terrain_levels = CurriculumTermCfg(func=course.stair_course_terrain_levels)
        self.curriculum.stair_course = CurriculumTermCfg(func=course.course_metrics)
        self.terminations.stair_course_failure = TerminationTermCfg(func=course.stair_course_failure)
        self.rewards.stair_course_complete = RewardTermCfg(func=course.course_completion_reward, weight=5.0)
        # Randomise wheel-ground friction even at the easy geometric level.
        self.events.randomize_rigid_body_material.params.update(
            static_friction_range=(.6, 1.2), dynamic_friction_range=(.5, 1.0),
            make_consistent=True)
        self.m20_refinement_profile = "stair_trot_v1"
