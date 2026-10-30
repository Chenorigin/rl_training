"""SDK fault regressions, boundary precision, substep latching and gradients."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from task_support.cts.configuration import default_settings, validate_config, VERSION, LEGACY_VERSIONS
from task_support.cts.joint_limits import TorchJointLimits, JointLimitMonitor, validate_limits, fault_damping


def config():
    return default_settings()['joint_limits']


@pytest.mark.parametrize('side', [0, 1])
def test_exact_sdk_float32_boundary_accepted_next_float_rejected(side):
    limits = TorchJointLimits(config())
    edge = limits.target[:, side].clone()
    assert not limits.violations(edge, limits.target).any()
    outward = torch.nextafter(edge, torch.full_like(edge, -torch.inf if side == 0 else torch.inf))
    assert limits.violations(outward, limits.target).all()
    inward = torch.nextafter(edge, torch.zeros_like(edge))
    assert not limits.violations(inward, limits.target).any()


def test_real_five_faults_rejected_while_feedback_is_still_legal():
    source = json.loads((Path(__file__).parent/'fixtures/cts1500_position_faults.json').read_text())
    limits = TorchJointLimits(config())
    sdk_legs = [i for i in range(16) if i % 4 != 3]
    for frame in source['frames']:
        target = limits.targets(torch.tensor(frame['action']))
        bad = limits.violations(target, limits.target)
        assert [sdk_legs[i] for i in bad.nonzero().flatten().tolist()] == [v['sdk_joint'] for v in frame['violations']]
        for v in frame['violations']:
            assert float(target[sdk_legs.index(v['sdk_joint'])]) == pytest.approx(v['target_rad'], abs=1.e-7)
        assert not limits.violations(torch.tensor(frame['q_sdk'])[sdk_legs], limits.feedback).any()
        assert frame['published'] is False


def test_wheel_positions_are_excluded_and_leg_margin_has_inward_gradient():
    limits = TorchJointLimits(config())
    action = torch.zeros(2, 16)
    action[0, 0], action[1, 0] = 5., -5.
    action[:, 12:] = 99.
    action.requires_grad_()
    loss = limits.student_loss(action)
    loss.backward()
    assert action.grad[0, 0] > 0 and action.grad[1, 0] < 0
    assert (action.grad[:, 12:] == 0).all()
    assert limits.student_loss(torch.zeros(2, 16)) == 0


def test_feedback_excursion_is_latched_even_when_it_returns_inside_before_tick_end():
    monitor = JointLimitMonitor(config(), 3, 'cpu')
    q = monitor.limits.default.repeat(3, 1)
    monitor.begin(torch.zeros(3, 16), q)
    q[1, 0] = .612
    monitor.sample(q)
    cost = monitor.feedback_cost.clone()
    q[1, 0] = 0
    monitor.sample(q)
    assert monitor.feedback_fault.tolist() == [False, True, False]
    assert torch.equal(cost, monitor.feedback_cost)
    monitor.reset(torch.tensor([0, 2]))
    assert monitor.faulted.tolist() == [False, True, False]
    monitor.reset(torch.tensor([1]))
    assert not monitor.faulted.any()


def test_target_and_corrected_failures_latch_per_env_and_cost_survives_safe_substep():
    monitor = JointLimitMonitor(config(), 3, 'cpu')
    q = monitor.limits.default.repeat(3, 1)
    action = torch.zeros(3, 16); action[0, 0] = 5.
    monitor.begin(action, q)
    corrected = q.clone(); corrected[1, 6] = -.5
    monitor.sample(q, corrected)
    monitor.sample(q, q)
    assert monitor.target_fault.tolist() == [True, False, False]
    assert monitor.corrected_fault.tolist() == [False, True, False]
    assert (monitor.target_cost[:2] > 0).all()
    assert monitor.reward_cost()[2] == 0


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_requests_fail_closed(value):
    monitor = JointLimitMonitor(config(), 2, 'cpu')
    q = monitor.limits.default.repeat(2, 1)
    action = torch.zeros(2, 16); action[0, 0] = value; action[1, 15] = value
    monitor.begin(action, q)
    assert monitor.faulted.all()
    assert torch.isfinite(monitor.reward_cost()).all()


def test_emergency_output_overrides_queued_effort_without_touching_healthy_envs():
    effort = torch.tensor([[50., -50.], [7., 8.]])
    velocity = torch.tensor([[3., -100.], [2., 2.]])
    damping = torch.full_like(effort, 2.)
    result = fault_damping(effort, velocity, damping, torch.full_like(effort, 20.), torch.tensor([True, False]))
    torch.testing.assert_close(result, torch.tensor([[-6., 20.], [7., 8.]]))


@pytest.mark.parametrize('change', ['name', 'nan', 'reversed', 'scale', 'margin', 'default', 'feedback', 'clip'])
def test_bad_limit_contracts_are_rejected(change):
    cfg = config()
    if change == 'name': cfg['joint_names'].reverse()
    if change == 'nan': cfg['target_bounds_rad'][0][0] = float('nan')
    if change == 'reversed': cfg['target_bounds_rad'][0].reverse()
    if change == 'scale': cfg['action_scale'][0] = -1
    if change == 'margin': cfg['margin_fraction'] = .5
    if change == 'default': cfg['default_position_rad'][0] = 2
    if change == 'clip': cfg['normalized_action_clip'] = [-1., 1.]
    if change == 'feedback': cfg['feedback_bounds_rad'][0][0] = 0
    with pytest.raises(ValueError): validate_limits(cfg)


def test_old_v14_remains_readable_but_new_training_requires_limits():
    cfg = default_settings()
    assert VERSION.endswith('_v16') and 'm20_cts_action_latent_v14' in LEGACY_VERSIONS
    validate_config(cfg)
    cfg.pop('joint_limits')
    validate_config(cfg, legacy=True)
    with pytest.raises(ValueError): validate_config(cfg)


def test_student_loss_reaches_encoder_through_frozen_decoder():
    from test_cts_algorithm import fixture, collect
    models, algorithm, obs, *_ = fixture()
    algorithm.joint_limits = TorchJointLimits(config())
    collect(algorithm, obs)
    original = models.student_action
    # Force an out-of-range final target while retaining the actual gradient path.
    models.student_action = lambda policy, command: original(policy, command) + 10.
    decoder_before = {k: v.clone() for k, v in models.decoder.state_dict().items()}
    student_before = {k: v.clone() for k, v in models.student.state_dict().items()}
    result = algorithm.distill_update()
    assert result['joint_limit_loss'] > 0
    assert any(not torch.equal(v, models.student.state_dict()[k]) for k, v in student_before.items())
    assert all(torch.equal(v, models.decoder.state_dict()[k]) for k, v in decoder_before.items())
    assert all(p.grad is None for p in algorithm.ppo_parameters)


def test_limit_failure_on_timeout_tick_is_terminal_not_bootstrapped(monkeypatch):
    from types import SimpleNamespace
    from task_support.cts import all_terrain_terms as terms
    state = SimpleNamespace(timeout=torch.tensor([1., 1.]), advance=lambda: None)
    monkeypatch.setattr(terms, 'course', lambda env: state)
    env = SimpleNamespace(episode_length_buf=torch.tensor([50, 50]), step_dt=.02,
                          cts_limit_monitor=SimpleNamespace(faulted=torch.tensor([True, False])))
    assert terms.timed_out(env).tolist() == [False, True]
