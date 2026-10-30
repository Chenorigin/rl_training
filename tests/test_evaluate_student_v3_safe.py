"""Verdict regression for the independent closed-loop evaluator."""
import importlib.util
from pathlib import Path

path = Path(__file__).resolve().parents[1]/'scripts/tools/evaluate_student_v3_safe.py'
spec = importlib.util.spec_from_file_location('safe_eval', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_success_requires_completion_without_failure_or_timeout():
    assert module.outcome(True, False, False)
    for success, failed, timeout in ((True, True, False), (True, False, True),
                                     (False, False, False), (False, True, True)):
        assert not module.outcome(success, failed, timeout)


def test_tracker_reset_must_share_inference_lifetime():
    """The real tracker replaces tensors during physics; reset must stay in mode."""
    import pytest
    import torch
    from task_support.stair_quiet_v1_safe.settings import SafeSettings
    from task_support.stair_quiet_v1_safe.signals import SafeTailTracker
    pos = torch.tensor([[[1.1,.25,.24],[1.1,-.25,.24],[1.1,.25,.24],[1.1,-.25,.24]]])
    axis = torch.zeros_like(pos); axis[...,1] = 1
    force = torch.zeros_like(pos); force[...,2] = 100
    sample = dict(position=pos, axis=axis, velocity=torch.zeros_like(pos), omega=torch.zeros_like(pos), forces=force,
                  root=torch.zeros(1,3), root_velocity=torch.zeros(1,3), yaw=torch.zeros(1), upright=torch.ones(1),
                  height=torch.tensor([.15]), depth=torch.tensor([.3]), width=torch.ones(1), steps=torch.tensor([6]), dt=.005)
    tracker = SafeTailTracker(1, 'cpu', SafeSettings())
    with torch.inference_mode():
        tracker.update(**sample)
    assert torch.is_inference(tracker.support_age)
    with pytest.raises(RuntimeError, match='InferenceMode|inference tensor'):
        tracker.reset(torch.tensor([0]), torch.tensor([0.]))
    with torch.inference_mode():
        tracker.reset(torch.tensor([0]), torch.tensor([0.]))
    assert tracker.elapsed_s.item() == 0


def test_recreated_sensor_requires_heading_feedback_reinstallation():
    from types import SimpleNamespace
    from task_support.student_v1.configuration import StudentV1PerceptionCfg
    from task_support.student_v3_5.v3_5_environment import feedback
    from task_support.student_v1.environment import sensor_state
    keys = ('translation_threshold','yaw_threshold','heading_hold_kp','heading_hold_kd',
            'heading_hold_max_yaw_rate','heading_hold_deadband','stair_heading_hold_kp',
            'stair_heading_hold_kd','stair_heading_hold_max_yaw_rate','stair_heading_hold_deadband',
            'heading_hold_rate_enter')
    e = SimpleNamespace(num_envs=1,device='cpu',step_dt=.02,
                        cfg=SimpleNamespace(student_perception=StudentV1PerceptionCfg(),
                                            student_history_steps=4,motion_teacher_tuning=dict.fromkeys(keys,1.)))
    first = feedback(e)
    assert sensor_state(e).command_transform is first
    del e._m20_student_v1_state
    assert feedback(e) is first
    assert sensor_state(e).command_transform is None  # Actual failure counterexample.
    del e._v3_5_heading
    reinstalled = feedback(e)
    assert sensor_state(e).command_transform is reinstalled
