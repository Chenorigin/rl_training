"""V3.5 frozen residual teacher; student and checkpoint ABI remain v1/v3.

Load the pinned actor by file so CPU preflight never imports the Isaac task tree.
State keys remain identical to the original teacher, including residual_mlp.*.
"""
from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path

import torch
from tensordict import TensorDict

TASK = 'Rough-Deeprobotics-M20-Student-v3.5'
TEACHER_CLASS = 'm20_student_v1.v3_5_teacher.MotionTeacherModel'
ORIGINAL_CLASS = 'rl_training.motion_residual_actor:MotionResidualMLPModel'
CANONICAL_CLASS = 'task_support.idle_stop_motion_v1.motion_residual_actor:MotionResidualMLPModel'
SOURCE = Path(__file__).resolve().parents[1] / 'idle_stop_motion_v1/motion_residual_actor.py'
SOURCE_SHA256 = 'c15973be1f5ddfd4c1888f402d7f70a5387fc866b57f237306ee9a7f5fb05129'


def validate_manifest(manifest):
    if manifest.get('m20_profile') != 'idle_stop_motion_v1' or manifest.get('control_profile') != 'legacy_v1':
        raise ValueError('V3.5 requires the idle_stop_motion_v1 / legacy_v1 teacher')
    if manifest.get('motion_control_version') != 'm20_raw_command_gate_20261017':
        raise ValueError('Unsupported residual teacher raw-command semantics')
    if manifest.get('actor_input_groups') != {'policy': 244, 'motion_control': 3}:
        raise ValueError('V3.5 teacher requires policy244 + raw motion_control3')
    expected = manifest.get('source_sha256', {}).get('motion_residual_actor.py')
    if expected != SOURCE_SHA256 or hashlib.sha256(SOURCE.read_bytes()).hexdigest() != expected:
        raise ValueError('Residual teacher source hash is missing or differs from the pinned source')
    config = manifest.get('actor_model_config', {})
    if (config.get('class_name') not in (ORIGINAL_CLASS, CANONICAL_CLASS) or config.get('hidden_dims') != [512, 256, 128]
            or config.get('residual_hidden_dims') != [128, 64] or config.get('activation') != 'elu'
            or config.get('obs_normalization') is not False
            or config.get('distribution_cfg') != {'class_name': 'GaussianDistribution', 'init_std': 1.0, 'std_type': 'log'}):
        raise ValueError('Unsupported V3.5 residual teacher architecture')
    # Check every field consumed by command feedback before simulator startup.
    from .v3_5_feedback import LegacyHeadingFeedback
    LegacyHeadingFeedback(1, 'cpu', manifest.get('stair_tuning', {}))
    return deepcopy(manifest)


def teacher_manifest(saved, resume=False, *, allow_quiet=False):
    if not allow_quiet:
        from task_support.stair_quiet_v1.deployment import reject_unsupported_quiet_checkpoint
        reject_unsupported_quiet_checkpoint(saved, "Legacy v3.5 teacher adapter")
    manifest = (saved.get('student_v1', {}).get('provenance', {}).get('teacher_manifest')
                if resume else saved.get('m20_training_manifest'))
    return validate_manifest(manifest or {})


def teacher_config(manifest):
    config = deepcopy(manifest['actor_model_config'])
    config['class_name'] = TEACHER_CLASS
    return config


def initialization_source(saved, explicit_student=None):
    if explicit_student is not None:
        state = explicit_student.get('student_state_dict', explicit_student.get('actor_state_dict', {}))
        if any(k.startswith('residual_mlp.') for k in state):
            raise ValueError('--init-student must be a plain student/MLP checkpoint, not a residual teacher')
        return explicit_student
    # Only the automatic student warm start copies the frozen baseline. Teacher
    # labels always come from the complete actor, never this extracted baseline.
    state = saved['actor_state_dict']
    if not any(k.startswith('residual_mlp.') for k in state):
        raise ValueError('V3.5 teacher checkpoint has no residual weights')
    return {'actor_state_dict': {k: v for k, v in state.items() if not k.startswith('residual_mlp.')}}


if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != SOURCE_SHA256:
    raise ValueError('Pinned motion residual actor source changed')
_spec = importlib.util.spec_from_file_location('m20_v3_5_pinned_motion_actor', SOURCE)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


class MotionTeacherModel(_module.MotionResidualMLPModel):
    """Map distillation's teacher244 group to the unchanged original actor."""
    def __init__(self, obs, obs_groups, obs_set, output_dim, **kwargs):
        if obs_set != 'teacher' or obs['teacher'].shape[-1] != 244:
            raise ValueError('MotionTeacherModel is only a clean 244D distillation teacher')
        packed = self._pack(obs)
        super().__init__(packed, {'teacher': ['policy']}, 'teacher', output_dim, **kwargs)
        self.requires_grad_(False)
        self.eval()

    @staticmethod
    def _pack(obs):
        raw = obs['motion_control']
        if raw.shape[:-1] != obs['teacher'].shape[:-1] or raw.shape[-1] != 3 or not torch.isfinite(raw).all():
            raise ValueError('motion_control must be the finite raw requested command, shape [N,3]')
        return TensorDict({'policy': obs['teacher'], 'motion_control': raw}, batch_size=obs.batch_size)

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        return super().forward(self._pack(obs), masks, hidden_state, stochastic_output)
