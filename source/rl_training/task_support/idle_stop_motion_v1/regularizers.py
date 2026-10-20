# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# This curriculum scalar belongs only to idle_stop_motion_v1.
gait_level: float = 0.0

def update_gait_level_from_terrain_mean(terrain_level_mean: float | torch.Tensor) -> float:
    """Update global gait_level from mean terrain level.

    Mapping rule:
    - mean <= 0.0 -> 0.0
    - 0.0 < mean < 3.0 -> 使用 exp 函数映射
    - mean == 3.0 -> 1.0
    - mean >= 3.0 -> 1.0
    """
    global gait_level
    mean_tensor = torch.as_tensor(terrain_level_mean, dtype=torch.float32)
    if mean_tensor.numel() == 0:
        mean_val = 0.0
    else:
        mean_val = float(torch.mean(mean_tensor).item())
    if math.isnan(mean_val) or math.isinf(mean_val):
        mean_val = 0.0
    if mean_val <= 0.0:
        gait_level = 0.0
    elif mean_val < 3.0:
        gait_level = math.exp(mean_val - 3.0)
    else:
        gait_level = 1.0
    return gait_level

def get_gait_level_tensor(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return global gait_level as tensor matching environment batch size."""
    return torch.full((env.num_envs,), gait_level, device=env.device)

def joint_torques_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg=SceneEntityCfg('robot')) -> torch.Tensor:
    """Penalize joint torques (curriculum-scaled by gait_level)."""
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)
    return reward * get_gait_level_tensor(env)

def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize action rate (curriculum-scaled by gait_level)."""
    reward = torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)
    return reward * get_gait_level_tensor(env)

def action_smooth_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize second-order action changes using an env-local action history."""
    cache_name = '_action_smooth_prev_prev_action'
    if not hasattr(env, cache_name):
        setattr(env, cache_name, torch.zeros_like(env.action_manager.action))
    prev_prev_action = getattr(env, cache_name)
    diff = torch.square(env.action_manager.action + prev_prev_action - 2 * env.action_manager.prev_action)
    diff = diff * (env.action_manager.prev_action != 0)
    diff = diff * (prev_prev_action != 0)
    reward = torch.sum(diff, dim=1)
    setattr(env, cache_name, env.action_manager.prev_action.clone())
    return reward * get_gait_level_tensor(env)

def contact_forces(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize contact force violations (curriculum-scaled by gait_level)."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    violation = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] - threshold
    reward = torch.sum(violation.clip(min=0.0), dim=1)
    return reward * get_gait_level_tensor(env)

def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, 'joint_mirror_joints_cache') or env.joint_mirror_joints_cache is None:
        env.joint_mirror_joints_cache = [[asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints]
    reward = torch.zeros(env.num_envs, device=env.device)
    for joint_pair in env.joint_mirror_joints_cache:
        diff = torch.sum(torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]), dim=-1)
        reward += diff
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    reward *= torch.clamp(-env.scene['robot'].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward * get_gait_level_tensor(env)

def feet_slide(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg=SceneEntityCfg('robot')) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[:, :].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(asset.data.root_quat_w, cur_footvel_translated[:, i, :])
    foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(env.num_envs, -1)
    reward = torch.sum(foot_leteral_vel * contacts, dim=1)
    return reward

def feet_slide_ang_z_cmd(env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg=SceneEntityCfg('robot'), cmd_lin_threshold: float=0.1, cmd_ang_threshold: float=0.1) -> torch.Tensor:
    """Penalize feet sliding, only active when linear xy command is small and angular z command is large.

    This is a variant of ``feet_slide`` that is gated by the velocity command: the penalty is only applied
    when ``norm(cmd_xy) < cmd_lin_threshold`` AND ``abs(cmd_z) > cmd_ang_threshold``, i.e. during pure
    yaw-rotation commands where foot sliding is most problematic.
    """
    reward = feet_slide(env, sensor_cfg=sensor_cfg, asset_cfg=asset_cfg)
    command = env.command_manager.get_command(command_name)
    lin_xy_norm = torch.norm(command[:, :2], dim=1)
    ang_z_abs = torch.abs(command[:, 2])
    gate = (lin_xy_norm < cmd_lin_threshold) & (ang_z_abs > cmd_ang_threshold)
    reward = reward * gate.float()
    return reward * get_gait_level_tensor(env)

class RotationGaitSymmetry(ManagerTermBase):
    """Reward symmetric contact duty cycle during rotation.

    Tracks contact history over a 5-second window and rewards each diagonal leg group
    for spending roughly equal time in contact vs air (50/50 duty cycle).

    Only active when linear velocity command is small and angular velocity command
    is significant (rotating in place).
    """

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.command_name: str = cfg.params['command_name']
        self.lin_vel_threshold: float = cfg.params['lin_vel_threshold']
        self.ang_vel_threshold: float = cfg.params['ang_vel_threshold']
        self.target_duty: float = cfg.params['target_duty']
        self.std: float = cfg.params['std']
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params['sensor_cfg'].name]
        group_a_body_names = cfg.params['group_a_body_names']
        group_b_body_names = cfg.params['group_b_body_names']
        self.group_a_ids = [self.contact_sensor.find_bodies(n)[0][0] for n in group_a_body_names]
        self.group_b_ids = [self.contact_sensor.find_bodies(n)[0][0] for n in group_b_body_names]
        self.buffer_size = int(5.0 / env.step_dt)
        num_feet = len(self.group_a_ids) + len(self.group_b_ids)
        self.all_ids = self.group_a_ids + self.group_b_ids
        self.contact_buffer = torch.zeros(env.num_envs, num_feet, self.buffer_size, device=env.device)
        self.buffer_idx = 0
        self.buffer_filled = False

    def reset(self, env_ids: torch.Tensor | None=None) -> None:
        """Clear buffer for reset envs."""
        if env_ids is not None and len(env_ids) > 0:
            self.contact_buffer[env_ids] = 0.0

    def __call__(self, env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, group_a_body_names: list[str], group_b_body_names: list[str], target_duty: float, std: float, lin_vel_threshold: float, ang_vel_threshold: float) -> torch.Tensor:
        contact_time = self.contact_sensor.data.current_contact_time
        in_contact = (contact_time[:, self.all_ids] > 0).float()
        self.contact_buffer[:, :, self.buffer_idx] = in_contact
        self.buffer_idx = (self.buffer_idx + 1) % self.buffer_size
        if not self.buffer_filled and self.buffer_idx == 0:
            self.buffer_filled = True
        command = env.command_manager.get_command(self.command_name)
        lin_vel_cmd = torch.linalg.norm(command[:, :2], dim=1)
        ang_vel_cmd = torch.abs(command[:, 2])
        gate = (lin_vel_cmd < self.lin_vel_threshold) & (ang_vel_cmd > self.ang_vel_threshold)
        num_a = len(self.group_a_ids)
        duty_a = self.contact_buffer[:, :num_a, :].mean(dim=(1, 2))
        duty_b = self.contact_buffer[:, num_a:, :].mean(dim=(1, 2))
        reward_a = torch.exp(-torch.square(duty_a - self.target_duty) / self.std ** 2)
        reward_b = torch.exp(-torch.square(duty_b - self.target_duty) / self.std ** 2)
        reward = reward_a * reward_b
        if not self.buffer_filled:
            reward = reward * (self.buffer_idx / self.buffer_size)
        reward *= gate.float()
        return reward * get_gait_level_tensor(env)

def track_lin_vel_xy_exp(env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg=SceneEntityCfg('robot')) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    asset: RigidObject = env.scene[asset_cfg.name]
    lin_vel_error = torch.sum(torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]), dim=1)
    reward = torch.exp(-lin_vel_error / std ** 2)
    return reward
