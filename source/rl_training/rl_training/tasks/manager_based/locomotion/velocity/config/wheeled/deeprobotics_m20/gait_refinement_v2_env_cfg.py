"""M20 gait refinement v2 parameter ledger and environment configuration.

Scope: idle stance, move-to-stop, ascent/descent heading. The5degree limit is
a behavior acceptance target; rewards do not certify a trained model.
"""
from copy import deepcopy
from dataclasses import dataclass, field
import math

from isaaclab.managers import RewardTermCfg
from isaaclab.utils import configclass
from rl_training.tasks.manager_based.locomotion.velocity.mdp.quiet_landing_signals import idle_stance_cost
from rl_training.tasks.manager_based.locomotion.velocity.mdp.gait_refinement_v2_env import (
    GaitRefinementV2VelocityCommand, idle_wheel_target_cost, stair_heading_cost,
)
from rl_training.tasks.manager_based.locomotion.velocity.mdp.rewards import track_ang_vel_z_world_exp
from .quiet_landing_env_cfg import DeeproboticsM20QuietLandingEnvCfg

@dataclass
class IdleStopSettings:
    command_threshold: float = 0.05
    flat_range: float = 0.025
    move_seconds: float = 4.0
    stop_sequence_fraction: float = 0.5
    move_speed_range: tuple = (0.35, 0.8)
    linear_reference: float = 0.03
    yaw_reference: float = 0.05
    wheel_action_reference: float = 0.10
    wheel_target_weight: float = 0.5
    preservation_weight: float = 5.0
    motion_reward_weight: float = -2.0
    pose_reward_weight: float = -0.5

    def validate(self):
        scalars = [v for k, v in vars(self).items() if k != "move_speed_range"]
        if not all(math.isfinite(v) for v in scalars + list(self.move_speed_range)):
            raise ValueError("Idle/stop settings must be finite")
        if min(self.command_threshold, self.flat_range, self.move_seconds,
               self.linear_reference, self.yaw_reference, self.wheel_action_reference) <= 0:
            raise ValueError("Idle/stop thresholds, time and reference scales must be positive")
        if len(self.move_speed_range) != 2 or not 0 < self.move_speed_range[0] <= self.move_speed_range[1]:
            raise ValueError("Idle/stop speed range must be positive and ordered")
        if not 0 <= self.stop_sequence_fraction <= 1:
            raise ValueError("Invalid move-to-stop sampling fraction")
        if min(self.wheel_target_weight, self.preservation_weight) < 0:
            raise ValueError("Idle/stop loss weights must be nonnegative")
        if max(self.motion_reward_weight, self.pose_reward_weight) > 0:
            raise ValueError("Idle/stop cost reward weights must be nonpositive")


@dataclass
class HeadingSettings:
    acceptance_degrees: float = 5.0
    soft_target_degrees: float = 1.0
    error_scale_degrees: float = 2.0
    stair_height_range: float = 0.06
    post_stair_seconds: float = 2.0
    translation_threshold: float = 0.10
    intentional_turn_threshold: float = 0.05
    direction_change_degrees: float = 20.0
    yaw_rate_reference: float = 0.25
    yaw_rate_mix: float = 0.10
    cost_cap: float = 16.0
    reward_weight: float = -1.5

    def validate(self):
        if not 0 < self.soft_target_degrees < self.acceptance_degrees < 90:
            raise ValueError("Heading target must be stricter than the acceptance angle")
        if min(self.error_scale_degrees, self.stair_height_range, self.post_stair_seconds,
               self.translation_threshold, self.intentional_turn_threshold,
               self.direction_change_degrees, self.yaw_rate_reference, self.cost_cap) <= 0:
            raise ValueError("Heading scales, thresholds and hold time must be positive")
        if not all(math.isfinite(x) for x in vars(self).values()):
            raise ValueError("Heading settings must be finite")
        if self.yaw_rate_mix < 0 or self.reward_weight > 0 or self.direction_change_degrees >= 180:
            raise ValueError("Heading cost weights or direction threshold are invalid")


@dataclass
class GaitCorrectionsSettings:
    revision: str = "v2"
    idle_stop_enabled: bool = True
    stair_heading_enabled: bool = True
    idle: IdleStopSettings = field(default_factory=IdleStopSettings)
    heading: HeadingSettings = field(default_factory=HeadingSettings)
    flat_training_fraction: float = 0.30
    terrain_rows: int = 4
    terrain_columns: int = 8

    def validate(self):
        self.idle.validate()
        self.heading.validate()
        if not 0 < self.flat_training_fraction < 1:
            raise ValueError("v2 must retain both flat and uneven terrain exposure")
        if min(self.terrain_rows, self.terrain_columns) <= 0:
            raise ValueError("Terrain dimensions must be positive")


@configclass
class DeeproboticsM20GaitRefinementV2EnvCfg(DeeproboticsM20QuietLandingEnvCfg):
    gait_corrections: GaitCorrectionsSettings = GaitCorrectionsSettings()

    def __post_init__(self):
        super().__post_init__()
        settings = self.gait_corrections
        settings.validate()
        self.apply_gait_corrections()
        self.commands.base_velocity.class_type = GaitRefinementV2VelocityCommand
        self.commands.base_velocity.rel_zero_vel_envs = 0.15
        self.commands.base_velocity.rel_only_lin_x_envs = 0.30
        self.commands.base_velocity.resampling_time_range = (8.0, 12.0)
        generator = deepcopy(self.scene.terrain.terrain_generator)
        previous_flat = generator.sub_terrains["flat"].proportion
        for name, terrain in generator.sub_terrains.items():
            if name != "flat":
                terrain.proportion *= (1 - settings.flat_training_fraction) / (1 - previous_flat)
        generator.sub_terrains["flat"].proportion = settings.flat_training_fraction
        generator.num_rows = settings.terrain_rows
        generator.num_cols = settings.terrain_columns
        self.scene.terrain.terrain_generator = generator
        self.commands.base_velocity.debug_vis = False

    def apply_gait_corrections(self):
        """Apply final values after Hydra overrides and before managers exist."""
        settings = self.gait_corrections
        idle = settings.idle
        if settings.idle_stop_enabled:
            self.rewards.quiet_idle_motion = RewardTermCfg(
                func=idle_stance_cost, weight=idle.motion_reward_weight,
                params={"component": "motion", "linear_reference": idle.linear_reference,
                        "yaw_reference": idle.yaw_reference, "command_threshold": idle.command_threshold,
                        "flat_range": idle.flat_range})
            self.rewards.quiet_idle_pose = RewardTermCfg(
                func=idle_stance_cost, weight=idle.pose_reward_weight,
                params={"component": "pose", "command_threshold": idle.command_threshold,
                        "flat_range": idle.flat_range})
            self.rewards.v2_idle_wheel_target = RewardTermCfg(
                func=idle_wheel_target_cost, weight=-idle.wheel_target_weight)
        else:
            for name in ("quiet_idle_motion", "quiet_idle_pose", "v2_idle_wheel_target"):
                setattr(self.rewards, name, None)
        if settings.stair_heading_enabled:
            self.rewards.v2_stair_heading = RewardTermCfg(
                func=stair_heading_cost, weight=settings.heading.reward_weight)
            # v2 replaces the old ascent-only direction objective. The new
            # persistent objective covers descent and the flat transition too.
            self.rewards.stair_ascent_direction_cost.weight = 0.0
            self.rewards.track_ang_vel_z_exp.func = track_ang_vel_z_world_exp
        else:
            self.rewards.v2_stair_heading = None
            baseline = DeeproboticsM20QuietLandingEnvCfg()
            self.rewards.stair_ascent_direction_cost = deepcopy(baseline.rewards.stair_ascent_direction_cost)
            self.rewards.track_ang_vel_z_exp = deepcopy(baseline.rewards.track_ang_vel_z_exp)

