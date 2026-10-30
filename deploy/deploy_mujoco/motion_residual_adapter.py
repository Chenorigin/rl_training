"""Inference compatibility for the colleague's idle_stop_motion_v1 checkpoint.

Actor source: frider635-beep/Deeprobotics_M20_stair, commit fad7528.
The shared actor file is copied byte-for-byte and checked against its manifest.
Legacy heading feedback follows that commit's HeadingHoldController(legacy=True).
"""
from __future__ import annotations

import hashlib
import importlib.util
import math
from pathlib import Path
import sys

import mujoco
import numpy as np
import torch
from tensordict import TensorDict

ROOT = Path(__file__).resolve().parents[2]
ACTOR_SOURCE = ROOT / 'source/rl_training/task_support/idle_stop_motion_v1/motion_residual_actor.py'
ACTOR_CLASS = 'rl_training.motion_residual_actor:MotionResidualMLPModel'
CANONICAL_ACTOR_CLASS = 'task_support.idle_stop_motion_v1.motion_residual_actor:MotionResidualMLPModel'
MOTION_VERSION = 'm20_raw_command_gate_20261017'


def load_motion_actor(obs, state, manifest):
    config = manifest.get('actor_model_config', {})
    if manifest.get('m20_profile') != 'idle_stop_motion_v1' or config.get('class_name') not in (ACTOR_CLASS, CANONICAL_ACTOR_CLASS):
        raise ValueError('Residual weights require the idle_stop_motion_v1 training manifest')
    if manifest.get('motion_control_version') != MOTION_VERSION:
        raise ValueError('Unsupported motion_control version; raw-command semantics must match training')
    if manifest.get('actor_input_groups') != {'policy': 244, 'motion_control': 3}:
        raise ValueError('Expected policy244 + raw motion_control3')
    if config.get('hidden_dims') != [512, 256, 128] or config.get('activation') != 'elu' or config.get('obs_normalization') is not False:
        raise ValueError('Residual actor base architecture differs from the 244D teacher contract')
    if manifest.get('control_profile') != 'legacy_v1':
        raise ValueError('Unsupported residual checkpoint command control profile')
    expected = manifest.get('source_sha256', {}).get('motion_residual_actor.py')
    source_hash = hashlib.sha256(ACTOR_SOURCE.read_bytes()).hexdigest()
    if not isinstance(expected, str) or len(expected) != 64 or source_hash != expected:
        raise ValueError('Shared motion_residual_actor.py does not match checkpoint source hash')
    name = 'm20_deploy_motion_residual_actor_' + source_hash[:16]
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ACTOR_SOURCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[name] = module
    observations = obs.clone()
    observations['motion_control'] = torch.zeros(1, 3)
    actor = sys.modules[name].MotionResidualMLPModel(
        observations, {'actor': ['policy']}, 'actor', 16,
        hidden_dims=config['hidden_dims'], activation=config['activation'],
        obs_normalization=False, distribution_cfg=config['distribution_cfg'],
        residual_hidden_dims=config.get('residual_hidden_dims', [128, 64]))
    actor.load_state_dict(state, strict=True)
    actor.eval()
    actor.m20_training_manifest = manifest
    return actor


def policy_inputs(observation, requested_command, actor, trot_control=None):
    values = {'policy': torch.from_numpy(observation).unsqueeze(0)}
    if getattr(actor, 'uses_motion_control', False):
        raw = np.asarray(requested_command, dtype=np.float32)
        if raw.shape != (3,) or not np.isfinite(raw).all():
            raise ValueError('motion_control requires a finite 3D raw requested command')
        values['motion_control'] = torch.from_numpy(raw.copy()).reshape(1, 3)
    if getattr(actor, 'uses_trot_control', False):
        control = np.asarray(trot_control, dtype=np.float32)
        if control.shape != (6,) or not np.isfinite(control).all():
            raise ValueError('StairTrot requires six finite sensor-derived control values')
        values['trot_control'] = torch.from_numpy(control.copy()).reshape(1, 6)
    return TensorDict(values, batch_size=[1])


class LegacyHeadingHold:
    """Checkpoint-specific direct body-z PD; ordinary MLP control is unchanged."""
    def __init__(self, contract, tuning):
        self.contract = contract
        self.threshold = float(np.float32(tuning['translation_threshold']))
        self.yaw_threshold = float(np.float32(tuning['yaw_threshold']))
        self.flat = tuple(tuning[k] for k in ('heading_hold_kp', 'heading_hold_kd', 'heading_hold_max_yaw_rate', 'heading_hold_deadband'))
        self.stairs = tuple(tuning[k] for k in ('stair_heading_hold_kp', 'stair_heading_hold_kd', 'stair_heading_hold_max_yaw_rate', 'stair_heading_hold_deadband'))
        self.rate_enter = tuning['heading_hold_rate_enter']
        self.reset()

    def reset(self):
        self.mode = 'manual'
        self.target_yaw = 0.
        self.last_time = None
        self.before = None

    def update(self, data, requested):
        now = float(data.time)
        if self.last_time is not None and now < self.last_time:
            self.reset()
        if now == self.last_time and self.before is not None:
            self.mode, self.target_yaw = self.before
        else:
            self.before = (self.mode, self.target_yaw)
        raw = np.asarray(requested, dtype=np.float32)
        rotation = data.xmat[self.contract.base_id].reshape(3, 3)
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.contract.model, data, mujoco.mjtObj.mjOBJ_BODY,
                                self.contract.base_id, velocity, 1)
        straight = abs(float(raw[0])) > self.threshold and abs(float(raw[1])) <= self.threshold and abs(float(raw[2])) <= self.yaw_threshold
        stationary = float(np.linalg.norm(raw[:2])) <= self.threshold and abs(float(raw[2])) <= self.yaw_threshold
        mode = 'straight' if straight else 'zero' if stationary else 'manual'
        if mode != self.mode and mode != 'manual':
            self.target_yaw = yaw
        command = raw.copy()
        if mode != 'manual':
            terrain = self.contract.terrain
            on_stairs = terrain.kind in ('ascent', 'descent') and terrain.parameters['start_x'][0] - .70 <= data.xpos[self.contract.base_id, 0] <= terrain.goal_x + .50
            kp, kd, limit, deadband = self.stairs if on_stairs else self.flat
            error = math.atan2(math.sin(self.target_yaw-yaw), math.cos(self.target_yaw-yaw))
            active = abs(error) > deadband or abs(velocity[2]) > self.rate_enter
            command[2] = np.clip(kp*error-kd*velocity[2], -limit, limit) if active else 0.
        self.mode, self.last_time = mode, now
        return command


def heading_controller(contract, actor, disabled=False):
    if not getattr(actor, 'uses_motion_control', False) or disabled:
        return None
    manifest = actor.m20_training_manifest
    if getattr(actor, 'uses_trot_control', False):
        manifest = manifest['source_teacher_manifest']
    return LegacyHeadingHold(contract, manifest['stair_tuning'])
