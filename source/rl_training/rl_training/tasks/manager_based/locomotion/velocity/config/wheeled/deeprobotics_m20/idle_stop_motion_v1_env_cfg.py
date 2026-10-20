# Copyright (c) 2026 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause

"""Teacher PPO configuration for M20Pro stair and mixed-terrain training."""
from copy import deepcopy
import os
from isaaclab.envs.mdp import height_scan
from isaaclab.managers import ObservationGroupCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains.trimesh import MeshPlaneTerrainCfg
from isaaclab.utils import configclass
from task_support.idle_stop_motion_v1 import rewards as stair_teacher
from task_support.idle_stop_motion_v1 import regularizers
from task_support.idle_stop_motion_v1.profile import PROFILE, CONTROL_PROFILE, REWARD_WEIGHTS, POLICY_NOISE
from rl_training.tasks.manager_based.locomotion.velocity.mdp import rewards as shared_rewards
from rl_training.tasks.manager_based.locomotion.velocity.velocity_env_cfg import CurriculumCfg, ObservationsCfg
from .rough_env_cfg import DeeproboticsM20RewardsCfg, DeeproboticsM20RoughEnvCfg
from .stair_teacher_env_cfg import DEFAULT_M20_USD_PATH

def _gait_entities():
    """Keep wheel/hip/knee order identical: FL, FR, HL, HR."""
    legs = ('fl', 'fr', 'hl', 'hr')
    return {'asset_cfg': SceneEntityCfg('robot', body_names=[f'{leg}_wheel' for leg in legs], preserve_order=True), 'hip_cfg': SceneEntityCfg('robot', body_names=[f'{leg}_hipy' for leg in legs], preserve_order=True), 'knee_cfg': SceneEntityCfg('robot', joint_names=[f'{leg}_knee_joint' for leg in legs], preserve_order=True), 'sensor_cfg': SceneEntityCfg('height_scanner'), 'contact_sensor_cfg': SceneEntityCfg('contact_forces', body_names=[f'{leg}_wheel' for leg in legs], preserve_order=True)}

@configclass
class M20IdleStopMotionV1ObservationsCfg(ObservationsCfg):
    """Teacher receives the M20 proprioceptive vector and the height scan."""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        height_scan = ObsTerm(func=height_scan, params={'sensor_cfg': SceneEntityCfg('height_scanner')}, noise=None, clip=(-1.0, 1.0), scale=1.0)
        velocity_commands = ObsTerm(func=stair_teacher.heading_hold_velocity_commands, params={'command_name': 'base_velocity'}, clip=(-100.0, 100.0), scale=1.0)

    @configclass
    class CriticCfg(ObservationsCfg.CriticCfg):
        velocity_commands = ObsTerm(func=stair_teacher.heading_hold_velocity_commands, params={'command_name': 'base_velocity'}, clip=(-100.0, 100.0), scale=1.0)
        joint_acceleration = ObsTerm(func=stair_teacher.critic_joint_acceleration, params={'asset_cfg': SceneEntityCfg('robot', joint_names='.*', preserve_order=True)}, clip=(-100.0, 100.0), scale=0.05)
        applied_torque = ObsTerm(func=stair_teacher.critic_applied_torque, params={'asset_cfg': SceneEntityCfg('robot', joint_names='.*', preserve_order=True)}, clip=(-1000.0, 1000.0), scale=0.02)
        wheel_contact_force_magnitude = ObsTerm(func=stair_teacher.critic_wheel_contact_force_magnitude, params={'sensor_cfg': SceneEntityCfg('contact_forces', body_names=['fl_wheel', 'fr_wheel', 'hl_wheel', 'hr_wheel'], preserve_order=True)}, clip=(0.0, 2000.0), scale=0.01)
        material_friction = ObsTerm(func=stair_teacher.critic_material_friction, params={'asset_cfg': SceneEntityCfg('robot')}, clip=(0.0, 3.0), scale=1.0)
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()

@configclass
class M20IdleStopMotionV1RewardsCfg(DeeproboticsM20RewardsCfg):
    """M20-specific terrain-gated stair rewards and effort penalties."""
    stair_orientation_cost = RewTerm(func=stair_teacher.stair_terrain_aware_orientation_cost, weight=-1.0, params={'asset_cfg': SceneEntityCfg('robot'), 'sensor_cfg': SceneEntityCfg('height_scanner'), 'flat_weight': 50.0, 'stair_weight': 5.0})
    stair_forward_progress = RewTerm(func=stair_teacher.stair_forward_progress, weight=1.0, params={'asset_cfg': SceneEntityCfg('robot', body_names='base_link'), 'sensor_cfg': SceneEntityCfg('height_scanner')})
    stair_front_wheel_lift = RewTerm(func=stair_teacher.stair_front_wheel_lift, weight=stair_teacher.TUNING['prep_weight'], params={'asset_cfg': SceneEntityCfg('robot', body_names=['fl_wheel', 'fr_wheel', 'hl_wheel', 'hr_wheel'], preserve_order=True), 'sensor_cfg': SceneEntityCfg('height_scanner'), 'contact_sensor_cfg': SceneEntityCfg('contact_forces', body_names=['fl_wheel', 'fr_wheel', 'hl_wheel', 'hr_wheel'], preserve_order=True)})
    stair_forward_motion_cost = RewTerm(func=stair_teacher.stair_forward_motion_cost, weight=-2.0, params={'asset_cfg': SceneEntityCfg('robot'), 'sensor_cfg': SceneEntityCfg('height_scanner')})
    stair_ascent_direction_cost = RewTerm(func=stair_teacher.stair_ascent_direction_cost, weight=stair_teacher.TUNING['ascent_direction_weight'], params={k: v for (k, v) in _gait_entities().items() if k not in ('hip_cfg', 'knee_cfg')})
    stair_ascent_heading_cost = RewTerm(func=stair_teacher.stair_ascent_heading_cost, weight=-3.0, params={k: v for (k, v) in _gait_entities().items() if k not in ('hip_cfg', 'knee_cfg')})
    stair_ascent_yaw_tracking_cost = RewTerm(func=stair_teacher.stair_ascent_yaw_tracking_cost, weight=-1.0, params={k: v for (k, v) in _gait_entities().items() if k not in ('hip_cfg', 'knee_cfg')})
    stair_ascent_straight_motion_cost = RewTerm(func=stair_teacher.stair_ascent_straight_motion_cost, weight=stair_teacher.TUNING['ascent_straight_motion_weight'], params={'asset_cfg': SceneEntityCfg('robot'), 'command_name': 'base_velocity'})
    straight_flat_heading_cost = RewTerm(func=stair_teacher.straight_flat_heading_cost, weight=-1.0, params={'command_name': 'base_velocity'})
    stair_descent_direction_cost = RewTerm(func=stair_teacher.stair_descent_direction_cost, weight=-0.5, params={'command_name': 'base_velocity'})
    stair_front_wheel_touchdown = RewTerm(func=stair_teacher.stair_front_wheel_touchdown, weight=stair_teacher.TUNING['front_touchdown_weight'], params={'asset_cfg': SceneEntityCfg('robot', body_names=['fl_wheel', 'fr_wheel', 'hl_wheel', 'hr_wheel'], preserve_order=True), 'sensor_cfg': SceneEntityCfg('height_scanner'), 'contact_sensor_cfg': SceneEntityCfg('contact_forces', body_names=['fl_wheel', 'fr_wheel', 'hl_wheel', 'hr_wheel'], preserve_order=True)})
    stair_front_single_support = RewTerm(func=stair_teacher.stair_front_single_support, weight=0.1, params={'asset_cfg': SceneEntityCfg('robot'), 'contact_sensor_cfg': SceneEntityCfg('contact_forces', body_names=['fl_wheel', 'fr_wheel'], preserve_order=True), 'sensor_cfg': SceneEntityCfg('height_scanner')})
    stair_diagonal_support_symmetry = RewTerm(func=stair_teacher.stair_diagonal_support_symmetry, weight=0.05, params={'asset_cfg': SceneEntityCfg('robot'), 'sensor_cfg': SceneEntityCfg('height_scanner'), 'contact_sensor_cfg': SceneEntityCfg('contact_forces', body_names=['fl_wheel', 'fr_wheel', 'hl_wheel', 'hr_wheel'], preserve_order=True)})
    stair_front_alternation = RewTerm(func=stair_teacher.stair_front_alternation, weight=0.25, params={'asset_cfg': SceneEntityCfg('robot', body_names=['fl_wheel', 'fr_wheel', 'hl_wheel', 'hr_wheel'], preserve_order=True), 'sensor_cfg': SceneEntityCfg('height_scanner'), 'contact_sensor_cfg': SceneEntityCfg('contact_forces', body_names=['fl_wheel', 'fr_wheel', 'hl_wheel', 'hr_wheel'], preserve_order=True)})
    stair_up_step_completion = RewTerm(func=stair_teacher.stair_up_step_completion, weight=3.0, params={'asset_cfg': SceneEntityCfg('robot', body_names=['fl_wheel', 'fr_wheel', 'hl_wheel', 'hr_wheel'], preserve_order=True), 'sensor_cfg': SceneEntityCfg('height_scanner'), 'contact_sensor_cfg': SceneEntityCfg('contact_forces', body_names=['fl_wheel', 'fr_wheel', 'hl_wheel', 'hr_wheel'], preserve_order=True)})
    stair_up_task_crossing = RewTerm(func=stair_teacher.stair_up_task_crossing, weight=stair_teacher.TUNING['task_crossing_weight'], params={'asset_cfg': SceneEntityCfg('robot', body_names=['fl_wheel', 'fr_wheel', 'hl_wheel', 'hr_wheel'], preserve_order=True), 'sensor_cfg': SceneEntityCfg('height_scanner'), 'contact_sensor_cfg': SceneEntityCfg('contact_forces', body_names=['fl_wheel', 'fr_wheel', 'hl_wheel', 'hr_wheel'], preserve_order=True)})
    stair_up_rear_step_completion = RewTerm(func=stair_teacher.stair_up_rear_step_completion, weight=stair_teacher.TUNING['rear_completion_weight'], params={'asset_cfg': SceneEntityCfg('robot', body_names=['fl_wheel', 'fr_wheel', 'hl_wheel', 'hr_wheel'], preserve_order=True), 'sensor_cfg': SceneEntityCfg('height_scanner'), 'contact_sensor_cfg': SceneEntityCfg('contact_forces', body_names=['fl_wheel', 'fr_wheel', 'hl_wheel', 'hr_wheel'], preserve_order=True)})
    stair_rear_wheel_lift = RewTerm(func=stair_teacher.stair_rear_wheel_lift, weight=stair_teacher.TUNING['prep_weight'], params={k: v for (k, v) in _gait_entities().items() if k not in ('hip_cfg', 'knee_cfg')})
    stair_rear_transition_recovery = RewTerm(func=stair_teacher.stair_rear_transition_recovery, weight=stair_teacher.TUNING['rear_recovery_weight'], params={k: v for (k, v) in _gait_entities().items() if k not in ('hip_cfg', 'knee_cfg')})
    stair_rear_style_refund = RewTerm(func=stair_teacher.stair_rear_style_refund, weight=-1.0, params={k: v for (k, v) in _gait_entities().items() if k not in ('hip_cfg', 'knee_cfg')})
    stair_descent_rear_forward_cost = RewTerm(func=stair_teacher.descent_rear_forward_cost, weight=stair_teacher.TUNING['rear_forward_weight'], params=_gait_entities())
    stair_descent_entry_retract = RewTerm(func=stair_teacher.stair_descent_entry_retract, weight=stair_teacher.TUNING['entry_retract_weight'], params=_gait_entities())
    stair_descent_entry_position_cost = RewTerm(func=stair_teacher.stair_descent_entry_position_cost, weight=stair_teacher.TUNING['entry_range_cost_weight'], params=_gait_entities())
    stair_descent_entry_front_lead_reward = RewTerm(func=stair_teacher.stair_descent_entry_front_lead_reward, weight=stair_teacher.TUNING['entry_front_lead_weight'], params=_gait_entities())
    stair_descent_entry_front_progress_reward = RewTerm(func=stair_teacher.stair_descent_entry_front_progress_reward, weight=stair_teacher.TUNING['entry_front_progress_weight'], params=_gait_entities())
    stair_descent_entry_sequence_reward = RewTerm(func=stair_teacher.stair_descent_entry_sequence_reward, weight=stair_teacher.TUNING['entry_sequence_weight'], params=_gait_entities())
    stair_descent_entry_wheel_spacing_left_cost = RewTerm(func=stair_teacher.stair_descent_entry_wheel_spacing_left_cost, weight=stair_teacher.TUNING['entry_wheel_spacing_left_weight'], params=_gait_entities())
    stair_descent_entry_wheel_spacing_right_cost = RewTerm(func=stair_teacher.stair_descent_entry_wheel_spacing_right_cost, weight=stair_teacher.TUNING['entry_wheel_spacing_right_weight'], params=_gait_entities())
    stair_descent_entry_rear_early_transfer_cost = RewTerm(func=stair_teacher.stair_descent_entry_rear_early_transfer_cost, weight=stair_teacher.TUNING['entry_rear_early_transfer_weight'], params=_gait_entities())
    stair_descent_wheel_shank_cost = RewTerm(func=stair_teacher.stair_descent_wheel_shank_cost, weight=-5.0, params=_gait_entities())
    stair_ascent_front_body_clearance_cost = RewTerm(func=stair_teacher.ascent_front_body_clearance_cost, weight=stair_teacher.TUNING['body_gap_weight'], params=_gait_entities())
    stair_up_order_violation_cost = RewTerm(func=stair_teacher.stair_up_order_violation_cost, weight=-0.5, params={'asset_cfg': SceneEntityCfg('robot', body_names=['fl_wheel', 'fr_wheel', 'hl_wheel', 'hr_wheel'], preserve_order=True), 'sensor_cfg': SceneEntityCfg('height_scanner'), 'contact_sensor_cfg': SceneEntityCfg('contact_forces', body_names=['fl_wheel', 'fr_wheel', 'hl_wheel', 'hr_wheel'], preserve_order=True)})
    stair_up_same_tread_cost = RewTerm(func=stair_teacher.stair_up_same_tread_cost, weight=stair_teacher.TUNING['same_tread_event_weight'], params={'asset_cfg': SceneEntityCfg('robot', body_names=['fl_wheel', 'fr_wheel', 'hl_wheel', 'hr_wheel'], preserve_order=True), 'sensor_cfg': SceneEntityCfg('height_scanner'), 'contact_sensor_cfg': SceneEntityCfg('contact_forces', body_names=['fl_wheel', 'fr_wheel', 'hl_wheel', 'hr_wheel'], preserve_order=True)})
    stair_up_same_tread_dwell_cost = RewTerm(func=stair_teacher.stair_up_same_tread_dwell_cost, weight=stair_teacher.TUNING['same_tread_dwell_weight'], params={k: v for (k, v) in _gait_entities().items() if k not in ('hip_cfg', 'knee_cfg')})
    stair_descent_rear_fold_cost = RewTerm(func=stair_teacher.descent_rear_fold_cost, weight=stair_teacher.TUNING['rear_fold_weight'], params=_gait_entities())
    stair_ascent_front_fold_cost = RewTerm(func=stair_teacher.ascent_front_fold_cost, weight=stair_teacher.TUNING['front_fold_weight'], params=_gait_entities())
    stair_descent_front_posture_cost = RewTerm(func=stair_teacher.descent_front_posture_cost, weight=stair_teacher.TUNING['descent_front_posture_weight'], params=_gait_entities())
    turn_swing_size_cost = RewTerm(func=stair_teacher.turn_swing_size_cost, weight=stair_teacher.TUNING['turn_size_weight'], params=_gait_entities())
    stair_down_step_completion = RewTerm(func=stair_teacher.stair_down_step_completion, weight=3.0, params={'asset_cfg': SceneEntityCfg('robot', body_names=['fl_wheel', 'fr_wheel', 'hl_wheel', 'hr_wheel'], preserve_order=True), 'sensor_cfg': SceneEntityCfg('height_scanner'), 'contact_sensor_cfg': SceneEntityCfg('contact_forces', body_names=['fl_wheel', 'fr_wheel', 'hl_wheel', 'hr_wheel'], preserve_order=True)})
    stair_front_riser_face_cost = RewTerm(func=stair_teacher.stair_front_riser_face_cost, weight=-0.5, params={'asset_cfg': SceneEntityCfg('robot', body_names=['fl_wheel', 'fr_wheel', 'hl_wheel', 'hr_wheel'], preserve_order=True), 'sensor_cfg': SceneEntityCfg('height_scanner'), 'contact_sensor_cfg': SceneEntityCfg('contact_forces', body_names=['fl_wheel', 'fr_wheel', 'hl_wheel', 'hr_wheel'], preserve_order=True)})
    stair_descent_rear_support = RewTerm(func=stair_teacher.stair_descent_rear_support, weight=0.35, params={'asset_cfg': SceneEntityCfg('robot', body_names=['fl_wheel', 'fr_wheel', 'hl_wheel', 'hr_wheel'], preserve_order=True), 'sensor_cfg': SceneEntityCfg('height_scanner'), 'contact_sensor_cfg': SceneEntityCfg('contact_forces', body_names=['fl_wheel', 'fr_wheel', 'hl_wheel', 'hr_wheel'], preserve_order=True)})
    stair_descent_entry_rear_pair_swing_cost = RewTerm(func=stair_teacher.stair_descent_entry_rear_pair_swing_cost, weight=stair_teacher.TUNING['entry_rear_pair_swing_cost_weight'], params={'asset_cfg': SceneEntityCfg('robot', body_names=['fl_wheel', 'fr_wheel', 'hl_wheel', 'hr_wheel'], preserve_order=True), 'sensor_cfg': SceneEntityCfg('height_scanner'), 'contact_sensor_cfg': SceneEntityCfg('contact_forces', body_names=['fl_wheel', 'fr_wheel', 'hl_wheel', 'hr_wheel'], preserve_order=True)})
    stair_impact_cost = RewTerm(func=stair_teacher.stair_impact_cost, weight=-0.2, params={'asset_cfg': SceneEntityCfg('robot', body_names=['fl_wheel', 'fr_wheel', 'hl_wheel', 'hr_wheel'], preserve_order=True), 'sensor_cfg': SceneEntityCfg('height_scanner'), 'contact_sensor_cfg': SceneEntityCfg('contact_forces', body_names=['fl_wheel', 'fr_wheel', 'hl_wheel', 'hr_wheel'], preserve_order=True), 'free_fall_speed': 0.1})
    stair_non_wheel_contact_cost = RewTerm(func=stair_teacher.stair_non_wheel_contact_cost, weight=-0.2, params={'sensor_cfg': SceneEntityCfg('contact_forces', body_names=['^(?!(base_link|.*_wheel)$).*'], preserve_order=True), 'stair_sensor_cfg': SceneEntityCfg('height_scanner')})
    stair_non_wheel_scrape_cost = RewTerm(func=stair_teacher.stair_non_wheel_scrape_cost, weight=-0.1, params={'sensor_cfg': SceneEntityCfg('contact_forces', body_names=['^(?!(base_link|.*_wheel)$).*'], preserve_order=True), 'asset_cfg': SceneEntityCfg('robot', body_names=['^(?!(base_link|.*_wheel)$).*'], preserve_order=True), 'stair_sensor_cfg': SceneEntityCfg('height_scanner')})
    stair_joint_effort_cost = RewTerm(func=stair_teacher.stair_joint_effort_cost, weight=-0.1, params={'asset_cfg': SceneEntityCfg('robot', joint_names='.*', preserve_order=True), 'sensor_cfg': SceneEntityCfg('height_scanner')})
    stair_joint_acceleration_cost = RewTerm(func=stair_teacher.stair_joint_acceleration_cost, weight=-0.02, params={'asset_cfg': SceneEntityCfg('robot', joint_names='^(?!.*_wheel_joint$).*', preserve_order=True), 'sensor_cfg': SceneEntityCfg('height_scanner'), 'normalization_acceleration': 120.0})
    stair_mechanical_power_cost = RewTerm(func=stair_teacher.stair_mechanical_power_cost, weight=-0.1, params={'asset_cfg': SceneEntityCfg('robot', joint_names='.*', preserve_order=True), 'sensor_cfg': SceneEntityCfg('height_scanner'), 'normalization_power': 1800.0})

@configclass
class M20IdleStopMotionV1CurriculumCfg(CurriculumCfg):
    """Log core diagnostics; optional detail never changes level promotion."""
    step_completion = CurrTerm(func=stair_teacher.stair_step_completion_metrics, params={})
    gait_quality = CurrTerm(func=stair_teacher.gait_quality_metrics, params={})
    ascent_direction = CurrTerm(func=stair_teacher.ascent_direction_metrics, params={})
    descent_direction = CurrTerm(func=stair_teacher.descent_direction_metrics, params={})
    terrain_level_overall = None
    terrain_level_pyramid_stairs = CurrTerm(func=stair_teacher.stair_terrain_level_mean, params={'terrain_name': 'pyramid_stairs'})
    terrain_level_pyramid_stairs_inv = CurrTerm(func=stair_teacher.stair_terrain_level_mean, params={'terrain_name': 'pyramid_stairs_inv'})
    terrain_level_boxes = CurrTerm(func=stair_teacher.stair_terrain_level_mean, params={'terrain_name': 'boxes', 'detailed_only': True})
    terrain_level_random_rough = CurrTerm(func=stair_teacher.stair_terrain_level_mean, params={'terrain_name': 'random_rough', 'detailed_only': True})
    terrain_level_hf_pyramid_slope = CurrTerm(func=stair_teacher.stair_terrain_level_mean, params={'terrain_name': 'hf_pyramid_slope', 'detailed_only': True})
    terrain_level_hf_pyramid_slope_inv = CurrTerm(func=stair_teacher.stair_terrain_level_mean, params={'terrain_name': 'hf_pyramid_slope_inv', 'detailed_only': True})
    terrain_level_flat = CurrTerm(func=stair_teacher.stair_terrain_level_mean, params={'terrain_name': 'flat'})

@configclass
class DeeproboticsM20IdleStopMotionV1EnvCfg(DeeproboticsM20RoughEnvCfg):
    """Teacher PPO task: proprioception + height map, privileged critic."""
    observations: M20IdleStopMotionV1ObservationsCfg = M20IdleStopMotionV1ObservationsCfg()
    rewards: M20IdleStopMotionV1RewardsCfg = M20IdleStopMotionV1RewardsCfg()
    curriculum: M20IdleStopMotionV1CurriculumCfg = M20IdleStopMotionV1CurriculumCfg()
    domain_randomization_stage: int = 0
    teacher_detailed_diagnostics: bool = False
    DOMAIN_RANDOMIZATION_STAGE_SCALES = (0.25, 0.5, 0.75, 1.0)

    def __post_init__(self):
        super().__post_init__()
        self.rewards.track_ang_vel_z_exp.func = stair_teacher.heading_aware_track_ang_vel_z_exp
        self.rewards.action_smooth_l2.func = stair_teacher.teacher_action_smooth_l2
        for (term_name, scale, disable_turn_side) in (('hipx_joint_pos_penalty', 0.5, False), ('hipy_joint_pos_penalty', 0.1, True), ('knee_joint_pos_penalty', 0.1, True)):
            term = getattr(self.rewards, term_name)
            term.func = stair_teacher.stair_joint_pos_cost
            for stale_param in ('sensor_cfg', 'terrain_height_threshold', 'high_terrain_penalty_scale'):
                term.params.pop(stale_param, None)
            term.params['stair_sensor_cfg'] = SceneEntityCfg('height_scanner')
            term.params['stair_penalty_scale'] = scale
            term.params['disable_turn_side_cmd'] = disable_turn_side
            term.params['turn_penalty_scale'] = stair_teacher.TUNING['turn_posture_scale']
        self.rewards.feet_air_time_ang_z_M20.func = stair_teacher.compact_turn_air_time
        self.rewards.feet_air_time_ang_z_M20.params['stair_sensor_cfg'] = SceneEntityCfg('height_scanner')
        self.rewards.feet_air_time_ang_z_M20.params['threshold'] = stair_teacher.TUNING['air_target']
        self.rewards.rotation_gait_status.func = stair_teacher.compact_turn_status
        self.rewards.rotation_gait_status.params = _gait_entities()
        self.rewards.rotation_gait_status.weight = stair_teacher.TUNING['turn_status_weight']
        self.rewards.rotation_gait_symmetry.params['lin_vel_threshold'] = stair_teacher.TUNING['translation_threshold']
        self.rewards.rotation_gait_symmetry.params['ang_vel_threshold'] = stair_teacher.TUNING['yaw_threshold']
        self.rewards.rotation_gait_symmetry.weight = stair_teacher.TUNING['turn_symmetry_weight']
        self.commands.base_velocity.class_type = stair_teacher.StairTeacherVelocityCommand
        self.commands.base_velocity.rel_zero_vel_envs = 0.15
        self.commands.base_velocity.rel_only_lin_x_envs = 0.3
        self.commands.base_velocity.rel_only_ang_z_envs = 0.05
        model_usd = os.path.realpath(os.environ.get('RL_TRAINING_M20_USD_PATH', DEFAULT_M20_USD_PATH))
        if not os.path.isfile(model_usd):
            raise FileNotFoundError(f'M20 teacher USD does not exist: {model_usd}. Set RL_TRAINING_M20_USD_PATH if the model was moved.')
        self.scene.robot.spawn.usd_path = model_usd
        self.observations.policy.height_scan = ObsTerm(func=height_scan, params={'sensor_cfg': SceneEntityCfg('height_scanner')}, noise=None, clip=(-1.0, 1.0), scale=1.0)
        for (term_name, magnitude) in (('base_ang_vel', stair_teacher.TUNING['teacher_angular_velocity_noise']), ('joint_vel', stair_teacher.TUNING['teacher_joint_velocity_noise']), ('joint_pos', stair_teacher.TUNING['teacher_joint_position_noise']), ('projected_gravity', stair_teacher.TUNING['teacher_projected_gravity_noise'])):
            noise = getattr(self.observations.policy, term_name).noise
            (noise.n_min, noise.n_max) = (-magnitude, magnitude)
        for term_name in ('joint_pos', 'joint_vel'):
            self.observations.critic.__getattribute__(term_name).params['asset_cfg'].joint_names = self.joint_names
        terrain_generator = deepcopy(self.scene.terrain.terrain_generator)
        sub_terrains = terrain_generator.sub_terrains
        stair_names = ('pyramid_stairs', 'pyramid_stairs_inv')
        for name in stair_names:
            if name not in sub_terrains:
                raise RuntimeError(f'Official rough terrain config is missing required sub-terrain: {name}')
            sub_terrains[name].step_height_range = (0.1, 0.25)
        proportions = {'pyramid_stairs': 0.3, 'pyramid_stairs_inv': 0.3, 'boxes': 0.05, 'random_rough': 0.05, 'hf_pyramid_slope': 0.05, 'hf_pyramid_slope_inv': 0.05}
        for (name, proportion) in proportions.items():
            if name not in sub_terrains:
                raise RuntimeError(f'Official rough terrain config is missing required sub-terrain: {name}')
            sub_terrains[name].proportion = proportion
        sub_terrains['flat'] = MeshPlaneTerrainCfg(proportion=0.2)
        sub_terrains['boxes'].grid_height_range = (0.025, 0.12)
        sub_terrains['random_rough'].noise_range = (0.01, 0.08)
        for name in ('hf_pyramid_slope', 'hf_pyramid_slope_inv'):
            sub_terrains[name].slope_range = (0.0, 0.2)
        self.scene.terrain.terrain_generator = terrain_generator
        self.scene.terrain.max_init_terrain_level = 0
        self.curriculum.terrain_levels = CurrTerm(func=stair_teacher.stair_distance_terrain_levels, params={'min_commanded_translation_m': 0.1})
        scales = self.DOMAIN_RANDOMIZATION_STAGE_SCALES
        stage = int(self.domain_randomization_stage)
        if stage < 0 or stage >= len(scales):
            raise ValueError(f'domain_randomization_stage must be in [0, {len(scales) - 1}], got {stage}')
        scale = scales[stage]
        self._scale_event_range(self.events.randomize_rigid_body_material.params, 'static_friction_range', scale)
        self._scale_event_range(self.events.randomize_rigid_body_material.params, 'dynamic_friction_range', scale)
        self._scale_event_range(self.events.randomize_rigid_body_material.params, 'restitution_range', scale)
        self._scale_event_range(self.events.randomize_rigid_body_mass.params, 'mass_distribution_params', scale, center=1.0)
        self._scale_event_range(self.events.randomize_rigid_body_mass_base.params, 'mass_distribution_params', scale)
        self._scale_event_range(self.events.randomize_rigid_body_inertia.params, 'inertia_distribution_params', scale, center=1.0)
        self._scale_event_range(self.events.randomize_actuator_gains.params, 'stiffness_distribution_params', scale, center=1.0)
        self._scale_event_range(self.events.randomize_actuator_gains.params, 'damping_distribution_params', scale, center=1.0)
        self._scale_mapping(self.events.randomize_com_positions.params['com_range'], scale)
        self._scale_mapping(self.events.randomize_apply_external_force_torque.params, scale, ('force_range', 'torque_range'))
        self._scale_mapping(self.events.randomize_reset_base.params['pose_range'], scale)
        self._scale_mapping(self.events.randomize_reset_base.params['velocity_range'], scale)
        self._scale_mapping(self.events.randomize_push_robot.params['velocity_range'], scale)
        self.events.randomize_reset_joints.params['position_range'] = (0.98, 1.02)
        self.events.randomize_reset_joints.params['velocity_range'] = (-0.1, 0.1)
        self._scale_event_range(self.events.randomize_reset_joints.params, 'position_range', scale, center=1.0)
        self._scale_event_range(self.events.randomize_reset_joints.params, 'velocity_range', scale, center=0.0)
        self.rewards.track_lin_vel_xy_exp.func = stair_teacher.stair_route_track_lin_vel_xy_exp
        self.rewards.track_lin_vel_xy_exp.params.update(_gait_entities())
        self.rewards.flat_orientation_l2.weight = 0.0
        self.disable_zero_weight_rewards()
        apply_environment_profile(self)

    @classmethod
    def _scale_event_range(cls, params, key: str, scale: float, center: float | None=None):
        values = params[key]
        if center is None:
            center = 0.5 * (float(values[0]) + float(values[1]))
        scaled = tuple((center + (float(value) - center) * scale for value in values))
        params[key] = list(scaled) if isinstance(values, list) else scaled

    @classmethod
    def _scale_mapping(cls, mapping, scale: float, only_keys: tuple[str, ...] | None=None):
        keys = tuple(mapping) if only_keys is None else only_keys
        for key in keys:
            value = mapping[key]
            if isinstance(value, dict):
                cls._scale_mapping(value, scale)
            elif isinstance(value, (tuple, list)) and len(value) == 2:
                cls._scale_event_range(mapping, key, scale, center=0.0)


@configclass
class MotionControlCfg(ObservationGroupCfg):
    raw_command = ObsTerm(func=stair_teacher.raw_velocity_command_observation)
    concatenate_terms = True
    enable_corruption = False


def apply_environment_profile(env_cfg):
    """Bind the actual checkpoint objective and unchanged legacy-v1 inputs."""
    for name, term in vars(env_cfg.rewards).items():
        if term is not None and hasattr(term, "weight"):
            if name not in REWARD_WEIGHTS:
                setattr(env_cfg.rewards, name, None)
            else:
                term.weight = REWARD_WEIGHTS[name]
    for name, weight in REWARD_WEIGHTS.items():
        term = getattr(env_cfg.rewards, name, None)
        if term is None:
            if name == "stair_descent_lateral_cost":
                env_cfg.rewards.stair_descent_lateral_cost = RewTerm(
                    func=stair_teacher.stair_descent_lateral_cost, weight=weight,
                    params={"command_name": "base_velocity"})
            else:
                raise ValueError(f"Idle-stop-motion-v1 reward missing: {name}")
    for term in vars(env_cfg.rewards).values():
        if term is not None and hasattr(term, "func"):
            func = term.func
            name = getattr(func, "__name__", "")
            if (getattr(func, "__module__", "") == shared_rewards.__name__
                    and hasattr(regularizers, name)):
                term.func = getattr(regularizers, name)
    env_cfg.rewards.action_smooth_l2.func = regularizers.action_smooth_l2
    env_cfg.rewards.stand_still.func = stair_teacher.zero_stop_flat_stand_still
    env_cfg.scene.terrain.max_init_terrain_level = 3
    env_cfg.observations.motion_control = MotionControlCfg()
    env_cfg.curriculum.zero_command_hold = None
    env_cfg.curriculum.zero_stop = None
    for name, noise in POLICY_NOISE.items():
        actual = getattr(env_cfg.observations.policy, name).noise
        actual.n_min, actual.n_max = noise["n_min"], noise["n_max"]
    env_cfg.m20_refinement_profile = PROFILE
    env_cfg.m20_control_profile = CONTROL_PROFILE
