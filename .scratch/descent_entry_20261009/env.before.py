# Copyright (c) 2026 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause

"""Teacher PPO configuration for M20Pro stair and mixed-terrain training."""

from copy import deepcopy
import os

from isaaclab.envs.mdp import height_scan
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains.trimesh import MeshPlaneTerrainCfg
from isaaclab.utils import configclass

from rl_training.tasks.manager_based.locomotion.velocity.mdp import stair_teacher
from rl_training.tasks.manager_based.locomotion.velocity.velocity_env_cfg import CurriculumCfg, ObservationsCfg

from .rough_env_cfg import DeeproboticsM20RewardsCfg, DeeproboticsM20RoughEnvCfg


DEFAULT_M20_USD_PATH = (
    "/home/ubuntu/桌面/m20_1/M20_perception_Lidar_rl_2/"
    "M20_perception_Lidar_rl/deep_robotics_model/M20/usd/M20.usd"
)


def _gait_entities():
    """Keep wheel/hip/knee order identical: FL, FR, HL, HR."""
    legs = ("fl", "fr", "hl", "hr")
    return {
        "asset_cfg": SceneEntityCfg("robot", body_names=[f"{leg}_wheel" for leg in legs], preserve_order=True),
        "hip_cfg": SceneEntityCfg("robot", body_names=[f"{leg}_hipy" for leg in legs], preserve_order=True),
        "knee_cfg": SceneEntityCfg("robot", joint_names=[f"{leg}_knee_joint" for leg in legs], preserve_order=True),
        "sensor_cfg": SceneEntityCfg("height_scanner"),
        "contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names=[f"{leg}_wheel" for leg in legs], preserve_order=True),
    }


@configclass
class M20StairTeacherObservationsCfg(ObservationsCfg):
    """Teacher receives the M20 proprioceptive vector and the height scan."""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        height_scan = ObsTerm(
            func=height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=None,
            clip=(-1.0, 1.0),
            scale=1.0,
        )

    @configclass
    class CriticCfg(ObservationsCfg.CriticCfg):
        joint_acceleration = ObsTerm(
            func=stair_teacher.critic_joint_acceleration,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=0.05,
        )
        applied_torque = ObsTerm(
            func=stair_teacher.critic_applied_torque,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            clip=(-1000.0, 1000.0),
            scale=0.02,
        )
        wheel_contact_force_magnitude = ObsTerm(
            func=stair_teacher.critic_wheel_contact_force_magnitude,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=["fl_wheel", "fr_wheel", "hl_wheel", "hr_wheel"],
                    preserve_order=True,
                )
            },
            clip=(0.0, 2000.0),
            scale=0.01,
        )
        material_friction = ObsTerm(
            func=stair_teacher.critic_material_friction,
            params={"asset_cfg": SceneEntityCfg("robot")},
            clip=(0.0, 3.0),
            scale=1.0,
        )

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class M20StairTeacherRewardsCfg(DeeproboticsM20RewardsCfg):
    """M20-specific terrain-gated stair rewards and effort penalties."""

    stair_orientation_cost = RewTerm(
        func=stair_teacher.stair_terrain_aware_orientation_cost,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "flat_weight": 50.0,
            "stair_weight": 5.0,
        },
    )
    stair_forward_progress = RewTerm(
        func=stair_teacher.stair_forward_progress,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
        },
    )
    stair_front_wheel_lift = RewTerm(
        func=stair_teacher.stair_front_wheel_lift,
        weight=stair_teacher.TUNING["prep_weight"],
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["fl_wheel", "fr_wheel", "hl_wheel", "hr_wheel"], preserve_order=True),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "contact_sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["fl_wheel", "fr_wheel", "hl_wheel", "hr_wheel"], preserve_order=True
            ),
        },
    )
    stair_forward_motion_cost = RewTerm(
        func=stair_teacher.stair_forward_motion_cost,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "sensor_cfg": SceneEntityCfg("height_scanner")},
    )
    stair_ascent_direction_cost = RewTerm(
        func=stair_teacher.stair_ascent_direction_cost,
        weight=stair_teacher.TUNING["ascent_direction_weight"],
        params={k: v for k, v in _gait_entities().items() if k not in ("hip_cfg", "knee_cfg")},
    )
    stair_front_wheel_touchdown = RewTerm(
        func=stair_teacher.stair_front_wheel_touchdown,
        weight=stair_teacher.TUNING["front_touchdown_weight"],
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["fl_wheel", "fr_wheel", "hl_wheel", "hr_wheel"], preserve_order=True),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "contact_sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["fl_wheel", "fr_wheel", "hl_wheel", "hr_wheel"], preserve_order=True
            ),
        },
    )
    stair_front_single_support = RewTerm(
        func=stair_teacher.stair_front_single_support,
        weight=0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "contact_sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["fl_wheel", "fr_wheel"], preserve_order=True
            ),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
        },
    )
    stair_diagonal_support_symmetry = RewTerm(
        func=stair_teacher.stair_diagonal_support_symmetry,
        weight=0.05,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "contact_sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["fl_wheel", "fr_wheel", "hl_wheel", "hr_wheel"],
                preserve_order=True,
            ),
        },
    )
    stair_front_alternation = RewTerm(
        func=stair_teacher.stair_front_alternation,
        weight=0.25,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["fl_wheel", "fr_wheel", "hl_wheel", "hr_wheel"], preserve_order=True),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "contact_sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["fl_wheel", "fr_wheel", "hl_wheel", "hr_wheel"], preserve_order=True
            ),
        },
    )
    stair_up_step_completion = RewTerm(
        func=stair_teacher.stair_up_step_completion,
        weight=3.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["fl_wheel", "fr_wheel", "hl_wheel", "hr_wheel"], preserve_order=True
            ),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "contact_sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["fl_wheel", "fr_wheel", "hl_wheel", "hr_wheel"], preserve_order=True,
            ),
        },
    )

    stair_up_task_crossing = RewTerm(
        func=stair_teacher.stair_up_task_crossing,
        weight=stair_teacher.TUNING["task_crossing_weight"],
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["fl_wheel", "fr_wheel", "hl_wheel", "hr_wheel"], preserve_order=True),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names=["fl_wheel", "fr_wheel", "hl_wheel", "hr_wheel"], preserve_order=True),
        },
    )
    stair_up_rear_step_completion = RewTerm(
        func=stair_teacher.stair_up_rear_step_completion,
        weight=stair_teacher.TUNING["rear_completion_weight"],
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["fl_wheel", "fr_wheel", "hl_wheel", "hr_wheel"], preserve_order=True
            ),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "contact_sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["fl_wheel", "fr_wheel", "hl_wheel", "hr_wheel"], preserve_order=True,
            ),
        },
    )
    stair_rear_wheel_lift = RewTerm(
        func=stair_teacher.stair_rear_wheel_lift,
        weight=stair_teacher.TUNING["prep_weight"],
        params={k:v for k,v in _gait_entities().items() if k not in ("hip_cfg","knee_cfg")},
    )
    stair_descent_rear_forward_cost = RewTerm(
        func=stair_teacher.descent_rear_forward_cost,
        weight=stair_teacher.TUNING["rear_forward_weight"], params=_gait_entities(),
    )
    stair_ascent_front_body_clearance_cost = RewTerm(
        func=stair_teacher.ascent_front_body_clearance_cost,
        weight=stair_teacher.TUNING["body_gap_weight"], params=_gait_entities(),
    )
    stair_up_order_violation_cost = RewTerm(
        func=stair_teacher.stair_up_order_violation_cost,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["fl_wheel", "fr_wheel", "hl_wheel", "hr_wheel"], preserve_order=True
            ),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "contact_sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["fl_wheel", "fr_wheel", "hl_wheel", "hr_wheel"], preserve_order=True,
            ),
        },
    )
    stair_up_same_tread_cost = RewTerm(
        func=stair_teacher.stair_up_same_tread_cost,
        weight=stair_teacher.TUNING["same_tread_event_weight"],
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["fl_wheel", "fr_wheel", "hl_wheel", "hr_wheel"], preserve_order=True
            ),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "contact_sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["fl_wheel", "fr_wheel", "hl_wheel", "hr_wheel"], preserve_order=True,
            ),
        },
    )
    stair_up_same_tread_dwell_cost = RewTerm(
        func=stair_teacher.stair_up_same_tread_dwell_cost,
        weight=stair_teacher.TUNING["same_tread_dwell_weight"],
        params={k: v for k, v in _gait_entities().items() if k not in ("hip_cfg", "knee_cfg")},
    )
    stair_descent_rear_fold_cost = RewTerm(
        func=stair_teacher.descent_rear_fold_cost,
        weight=stair_teacher.TUNING["rear_fold_weight"], params=_gait_entities(),
    )
    stair_ascent_front_fold_cost = RewTerm(
        func=stair_teacher.ascent_front_fold_cost,
        weight=stair_teacher.TUNING["front_fold_weight"], params=_gait_entities(),
    )
    turn_swing_size_cost = RewTerm(
        func=stair_teacher.turn_swing_size_cost,
        weight=stair_teacher.TUNING["turn_size_weight"], params=_gait_entities(),
    )
    stair_down_step_completion = RewTerm(
        func=stair_teacher.stair_down_step_completion,
        weight=3.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["fl_wheel", "fr_wheel", "hl_wheel", "hr_wheel"], preserve_order=True
            ),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "contact_sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["fl_wheel", "fr_wheel", "hl_wheel", "hr_wheel"], preserve_order=True,
            ),
        },
    )
    stair_front_riser_face_cost = RewTerm(
        func=stair_teacher.stair_front_riser_face_cost,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["fl_wheel", "fr_wheel", "hl_wheel", "hr_wheel"], preserve_order=True
            ),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "contact_sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["fl_wheel", "fr_wheel", "hl_wheel", "hr_wheel"],
                preserve_order=True,
            ),
        },
    )
    stair_descent_rear_support = RewTerm(
        func=stair_teacher.stair_descent_rear_support,
        weight=0.35,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["fl_wheel", "fr_wheel", "hl_wheel", "hr_wheel"], preserve_order=True
            ),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "contact_sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["fl_wheel", "fr_wheel", "hl_wheel", "hr_wheel"],
                preserve_order=True,
            ),
        },
    )
    stair_impact_cost = RewTerm(
        func=stair_teacher.stair_impact_cost,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["fl_wheel", "fr_wheel", "hl_wheel", "hr_wheel"], preserve_order=True
            ),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "contact_sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["fl_wheel", "fr_wheel", "hl_wheel", "hr_wheel"],
                preserve_order=True,
            ),
            "free_fall_speed": 0.10,
        },
    )
    stair_non_wheel_contact_cost = RewTerm(
        func=stair_teacher.stair_non_wheel_contact_cost,
        weight=-0.2,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["^(?!(base_link|.*_wheel)$).*"], preserve_order=True
            ),
            "stair_sensor_cfg": SceneEntityCfg("height_scanner"),
        },
    )
    stair_non_wheel_scrape_cost = RewTerm(
        func=stair_teacher.stair_non_wheel_scrape_cost,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["^(?!(base_link|.*_wheel)$).*"], preserve_order=True
            ),
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["^(?!(base_link|.*_wheel)$).*"], preserve_order=True
            ),
            "stair_sensor_cfg": SceneEntityCfg("height_scanner"),
        },
    )
    stair_joint_effort_cost = RewTerm(
        func=stair_teacher.stair_joint_effort_cost,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True),
                "sensor_cfg": SceneEntityCfg("height_scanner")},
    )
    stair_joint_acceleration_cost = RewTerm(
        func=stair_teacher.stair_joint_acceleration_cost,
        weight=-0.02,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="^(?!.*_wheel_joint$).*", preserve_order=True),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "normalization_acceleration": 120.0,
        },
    )
    stair_mechanical_power_cost = RewTerm(
        func=stair_teacher.stair_mechanical_power_cost,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True),
                "sensor_cfg": SceneEntityCfg("height_scanner"),
                "normalization_power": 1800.0},
    )


@configclass
class M20StairTeacherCurriculumCfg(CurriculumCfg):
    """Log each terrain type without changing level promotion or rewards."""

    step_completion = CurrTerm(func=stair_teacher.stair_step_completion_metrics, params={})
    gait_quality = CurrTerm(func=stair_teacher.gait_quality_metrics, params={})
    ascent_direction = CurrTerm(func=stair_teacher.ascent_direction_metrics, params={})

    terrain_level_overall = CurrTerm(
        func=stair_teacher.stair_terrain_level_mean, params={"terrain_name": "overall"}
    )
    terrain_level_pyramid_stairs = CurrTerm(
        func=stair_teacher.stair_terrain_level_mean, params={"terrain_name": "pyramid_stairs"}
    )
    terrain_level_pyramid_stairs_inv = CurrTerm(
        func=stair_teacher.stair_terrain_level_mean, params={"terrain_name": "pyramid_stairs_inv"}
    )
    terrain_level_boxes = CurrTerm(
        func=stair_teacher.stair_terrain_level_mean, params={"terrain_name": "boxes"}
    )
    terrain_level_random_rough = CurrTerm(
        func=stair_teacher.stair_terrain_level_mean, params={"terrain_name": "random_rough"}
    )
    terrain_level_hf_pyramid_slope = CurrTerm(
        func=stair_teacher.stair_terrain_level_mean, params={"terrain_name": "hf_pyramid_slope"}
    )
    terrain_level_hf_pyramid_slope_inv = CurrTerm(
        func=stair_teacher.stair_terrain_level_mean, params={"terrain_name": "hf_pyramid_slope_inv"}
    )
    terrain_level_flat = CurrTerm(
        func=stair_teacher.stair_terrain_level_mean, params={"terrain_name": "flat"}
    )

@configclass
class DeeproboticsM20StairTeacherEnvCfg(DeeproboticsM20RoughEnvCfg):
    """Teacher PPO task: proprioception + height map, privileged critic."""

    observations: M20StairTeacherObservationsCfg = M20StairTeacherObservationsCfg()
    rewards: M20StairTeacherRewardsCfg = M20StairTeacherRewardsCfg()
    curriculum: M20StairTeacherCurriculumCfg = M20StairTeacherCurriculumCfg()
    domain_randomization_stage: int = 0

    DOMAIN_RANDOMIZATION_STAGE_SCALES = (0.25, 0.50, 0.75, 1.0)

    def __post_init__(self):
        super().__post_init__()

        # The inherited base scanner only looks beneath the trunk. Relax posture
        # when the actor's wider height scan sees a stair edge, including one
        # recently passed by the trunk but not yet cleared by the rear wheels.
        for term_name, scale, disable_turn_side in (
            ("hipx_joint_pos_penalty", 0.5, False),
            ("hipy_joint_pos_penalty", 0.1, True),
            ("knee_joint_pos_penalty", 0.1, True),
        ):
            term = getattr(self.rewards, term_name)
            term.func = stair_teacher.stair_joint_pos_cost
            for stale_param in ("sensor_cfg", "terrain_height_threshold", "high_terrain_penalty_scale"):
                term.params.pop(stale_param, None)
            term.params["stair_sensor_cfg"] = SceneEntityCfg("height_scanner")
            term.params["stair_penalty_scale"] = scale
            term.params["disable_turn_side_cmd"] = disable_turn_side
            term.params["turn_penalty_scale"] = stair_teacher.TUNING["turn_posture_scale"]

        self.rewards.feet_air_time_ang_z_M20.func = stair_teacher.compact_turn_air_time
        self.rewards.feet_air_time_ang_z_M20.params["stair_sensor_cfg"] = SceneEntityCfg("height_scanner")
        self.rewards.feet_air_time_ang_z_M20.params["threshold"] = stair_teacher.TUNING["air_target"]
        self.rewards.rotation_gait_status.func = stair_teacher.compact_turn_status
        self.rewards.rotation_gait_status.params = _gait_entities()
        self.rewards.rotation_gait_status.weight = stair_teacher.TUNING["turn_status_weight"]
        # Narrow the existing duty-cycle shaping to actual in-place turns,
        # and reduce its pressure relative to command tracking/completion.
        self.rewards.rotation_gait_symmetry.params["lin_vel_threshold"] = stair_teacher.TUNING["translation_threshold"]
        self.rewards.rotation_gait_symmetry.params["ang_vel_threshold"] = stair_teacher.TUNING["yaw_threshold"]
        self.rewards.rotation_gait_symmetry.weight = stair_teacher.TUNING["turn_symmetry_weight"]

        # Keep basic standing/turning practice while favoring traversal commands.
        self.commands.base_velocity.class_type = stair_teacher.StairTeacherVelocityCommand
        self.commands.base_velocity.rel_zero_vel_envs = 0.05
        self.commands.base_velocity.rel_only_ang_z_envs = 0.05

        # This project's known M20 USD is the teacher default. An environment
        # override remains available when the asset is moved to another host.
        model_usd = os.path.realpath(os.environ.get("RL_TRAINING_M20_USD_PATH", DEFAULT_M20_USD_PATH))
        if not os.path.isfile(model_usd):
            raise FileNotFoundError(
                f"M20 teacher USD does not exist: {model_usd}. "
                "Set RL_TRAINING_M20_USD_PATH if the model was moved."
            )
        self.scene.robot.spawn.usd_path = model_usd

        # The M20 rough parent disables the actor scan; restore a clean,
        # noise-free map channel for the teacher actor.
        self.observations.policy.height_scan = ObsTerm(
            func=height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=None,
            clip=(-1.0, 1.0),
            scale=1.0,
        )
        for term_name in ("joint_pos", "joint_vel"):
            self.observations.critic.__getattribute__(term_name).params["asset_cfg"].joint_names = self.joint_names

        # Keep the official rough terrain set, but bias the generator toward
        # 10-25 cm ascending and descending stairs with a small generalization mix.
        terrain_generator = deepcopy(self.scene.terrain.terrain_generator)
        sub_terrains = terrain_generator.sub_terrains
        stair_names = ("pyramid_stairs", "pyramid_stairs_inv")
        for name in stair_names:
            if name not in sub_terrains:
                raise RuntimeError(f"Official rough terrain config is missing required sub-terrain: {name}")
            sub_terrains[name].step_height_range = (0.10, 0.25)
        proportions = {
            "pyramid_stairs": 0.35,
            "pyramid_stairs_inv": 0.35,
            "boxes": 0.05,
            "random_rough": 0.05,
            "hf_pyramid_slope": 0.05,
            "hf_pyramid_slope_inv": 0.05,
        }
        for name, proportion in proportions.items():
            if name not in sub_terrains:
                raise RuntimeError(f"Official rough terrain config is missing required sub-terrain: {name}")
            sub_terrains[name].proportion = proportion
        sub_terrains["flat"] = MeshPlaneTerrainCfg(proportion=0.10)
        sub_terrains["boxes"].grid_height_range = (0.025, 0.12)
        sub_terrains["random_rough"].noise_range = (0.01, 0.08)
        # Keep the small slope mix gentler than the official 0.4 gradient.
        for name in ("hf_pyramid_slope", "hf_pyramid_slope_inv"):
            sub_terrains[name].slope_range = (0.0, 0.20)
        self.scene.terrain.terrain_generator = terrain_generator
        self.scene.terrain.max_init_terrain_level = 0

        # Distance-based promotion/demotion; ignore episodes with almost no translation command.
        self.curriculum.terrain_levels = CurrTerm(
            func=stair_teacher.stair_distance_terrain_levels,
            params={"min_commanded_translation_m": 0.1},
        )

        # The user's preference is to keep every randomization source active
        # from the beginning and widen its range in four explicit run stages.
        scales = self.DOMAIN_RANDOMIZATION_STAGE_SCALES
        stage = int(self.domain_randomization_stage)
        if stage < 0 or stage >= len(scales):
            raise ValueError(f"domain_randomization_stage must be in [0, {len(scales) - 1}], got {stage}")
        scale = scales[stage]
        self._scale_event_range(self.events.randomize_rigid_body_material.params, "static_friction_range", scale)
        self._scale_event_range(self.events.randomize_rigid_body_material.params, "dynamic_friction_range", scale)
        self._scale_event_range(self.events.randomize_rigid_body_material.params, "restitution_range", scale)
        self._scale_event_range(
            self.events.randomize_rigid_body_mass.params, "mass_distribution_params", scale, center=1.0
        )
        self._scale_event_range(
            self.events.randomize_rigid_body_mass_base.params, "mass_distribution_params", scale
        )
        self._scale_event_range(
            self.events.randomize_rigid_body_inertia.params, "inertia_distribution_params", scale, center=1.0
        )
        self._scale_event_range(
            self.events.randomize_actuator_gains.params, "stiffness_distribution_params", scale, center=1.0
        )
        self._scale_event_range(
            self.events.randomize_actuator_gains.params, "damping_distribution_params", scale, center=1.0
        )
        self._scale_mapping(self.events.randomize_com_positions.params["com_range"], scale)
        self._scale_mapping(
            self.events.randomize_apply_external_force_torque.params, scale, ("force_range", "torque_range")
        )
        self._scale_mapping(self.events.randomize_reset_base.params["pose_range"], scale)
        self._scale_mapping(self.events.randomize_reset_base.params["velocity_range"], scale)
        self._scale_mapping(self.events.randomize_push_robot.params["velocity_range"], scale)
        # Introduce small reset jitter immediately, then expand it with later stages.
        self.events.randomize_reset_joints.params["position_range"] = (0.98, 1.02)
        self.events.randomize_reset_joints.params["velocity_range"] = (-0.1, 0.1)
        self._scale_event_range(self.events.randomize_reset_joints.params, "position_range", scale, center=1.0)
        self._scale_event_range(self.events.randomize_reset_joints.params, "velocity_range", scale, center=0.0)

        # Keep the official velocity reward but prevent sideways bypassing a
        # detected forward stair route from earning its full dense reward.
        self.rewards.track_lin_vel_xy_exp.func = stair_teacher.stair_route_track_lin_vel_xy_exp
        self.rewards.track_lin_vel_xy_exp.params.update(_gait_entities())

        # Replace the flat orientation term with a map-gated L2 cost.
        self.rewards.flat_orientation_l2.weight = 0.0
        # Remove unused zero-weight inherited terms only after custom terms exist.
        self.disable_zero_weight_rewards()

    @classmethod
    def _scale_event_range(cls, params, key: str, scale: float, center: float | None = None):
        values = params[key]
        if center is None:
            center = 0.5 * (float(values[0]) + float(values[1]))
        scaled = tuple(center + (float(value) - center) * scale for value in values)
        params[key] = list(scaled) if isinstance(values, list) else scaled

    @classmethod
    def _scale_mapping(cls, mapping, scale: float, only_keys: tuple[str, ...] | None = None):
        keys = tuple(mapping) if only_keys is None else only_keys
        for key in keys:
            value = mapping[key]
            if isinstance(value, dict):
                cls._scale_mapping(value, scale)
            elif isinstance(value, (tuple, list)) and len(value) == 2:
                cls._scale_event_range(mapping, key, scale, center=0.0)
