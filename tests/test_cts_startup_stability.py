"""CPU contracts for startup coverage and measured stability objectives."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from task_support.cts.configuration import default_settings, validate_config, LEGACY_VERSIONS
from task_support.cts.stability import reset_supported_start_joints, stability_costs, linear_tracking_reward


def fake(n=256):
    cfg = default_settings()['all_terrain']
    cfg.pop('wall_stability')  # This fixture exercises the pre-v18 objective.
    data = SimpleNamespace(default_joint_pos=torch.zeros(n, 16), joint_pos=torch.zeros(n, 16),
        joint_vel=torch.zeros(n, 16), root_ang_vel_b=torch.zeros(n, 3),
        soft_joint_pos_limits=torch.tensor([-1., 1.]).expand(n, 16, 2).clone())
    robot = SimpleNamespace(data=data, write_joint_state_to_sim=Mock(),
        find_joints=lambda names, preserve_order: (names, names))
    state = SimpleNamespace(cfg=cfg, course_ids=torch.full((n,), 3), stairs=torch.zeros(n, dtype=torch.bool))
    env = SimpleNamespace(scene={'robot': robot}, cts_all_terrain=state,
        cfg=SimpleNamespace(leg_joint_names=list(range(12)), wheel_joint_names=list(range(12, 16))),
        action_manager=SimpleNamespace(action=torch.zeros(n, 16), prev_action=torch.zeros(n, 16)))
    return env, state, {'command': torch.zeros(n, 3)}


def test_versioned_config_keeps_v13_readable_but_requires_v14_coverage():
    cfg = default_settings()
    validate_config(cfg)
    assert 'm20_cts_action_latent_v13' in LEGACY_VERSIONS
    old = deepcopy(cfg)
    for key in ('wall_hold_practice','stair_hold_practice','motion_stability','startup_stability','compound_commands','wall_practice','hind_clearance',
                'safety_learning','stair_recovery','obstacle_response','stair_walls','hind_ascent',
                'front_descent','clutter_consistency','wall_stability','objective_consistency','stair_stop','motion_quality'):
        old['all_terrain'].pop(key)
    validate_config(old, legacy=True)
    with pytest.raises(ValueError, match='v14'):
        validate_config(old)
    cfg['all_terrain'].pop('motion_stability')
    with pytest.raises(ValueError):
        validate_config(cfg, legacy=True)


def test_additive_start_covers_zero_nominal_both_signs_and_preserves_stairs():
    torch.manual_seed(32)
    env, state, _ = fake()
    state.course_ids[:64] = 0
    reset_supported_start_joints(env, torch.arange(256))
    args, kwargs = env.scene['robot'].write_joint_state_to_sim.call_args
    q, dq = args
    assert kwargs['env_ids'].min() == 64
    assert kwargs['joint_ids'] == list(range(12))
    assert (q.min(0).values < -.1).all() and (q.max(0).values > .1).all()
    assert q.abs().max() <= state.cfg['startup_stability']['joint_offset_rad']
    assert dq.abs().max() <= state.cfg['startup_stability']['joint_velocity_rad_s']


def test_start_limits_are_preserved_and_invalid_limits_fail_closed():
    env, state, _ = fake()
    data = env.scene['robot'].data
    data.default_joint_pos.fill_(.99)
    reset_supported_start_joints(env, torch.arange(256))
    q = env.scene['robot'].write_joint_state_to_sim.call_args.args[0]
    assert q.max() <= .8 + 1.e-6
    data.soft_joint_pos_limits[:, 0, 1] = float('nan')
    with pytest.raises(ValueError, match='finite ordered'):
        reset_supported_start_joints(env, torch.arange(256))


def test_legacy_has_no_startup_writes_or_added_costs():
    env, state, geometry = fake()
    state.cfg.pop('startup_stability')
    reset_supported_start_joints(env, torch.arange(256))
    env.scene['robot'].write_joint_state_to_sim.assert_not_called()
    assert stability_costs(env, state, geometry) == {}


def test_idle_cost_uses_measured_wheels_and_does_not_penalize_lateral_stepping():
    env, state, geometry = fake(4)
    data = env.scene['robot'].data
    data.joint_vel[0, 12:] = 10.
    data.joint_vel[1] = 10.
    geometry['command'][1, 1] = .2
    data.root_ang_vel_b[2, 0] = 2.
    state.stairs[3] = True
    data.joint_vel[3] = 10.
    data.root_ang_vel_b[3] = 2.
    costs = stability_costs(env, state, geometry)
    assert costs['idle_wheel_speed'][0] > 0
    assert costs['idle_joint_speed'][0] == 0
    assert costs['idle_wheel_speed'][1] == costs['idle_joint_speed'][1] == 0
    assert costs['roll_pitch_rate'][2] > 0
    assert all(v[3] == 0 for v in costs.values())


def test_near_limit_cost_is_symmetric_and_action_jump_is_idle_only():
    env, state, geometry = fake(4)
    env.scene['robot'].data.joint_pos[0, :12] = .95
    env.scene['robot'].data.joint_pos[1, :12] = -.95
    env.action_manager.action[2:] = 1.
    geometry['command'][3, 1] = -.2
    costs = stability_costs(env, state, geometry)
    torch.testing.assert_close(costs['joint_limit'][0], costs['joint_limit'][1])
    assert costs['joint_limit'][0] > costs['joint_limit'][2]
    assert costs['idle_action_rate'][2] > costs['idle_action_rate'][3] == 0


def test_small_lateral_command_cannot_receive_almost_full_credit_for_standing():
    _, state, geometry = fake(4)
    state.mode = torch.tensor([9, 10, 9, 0])
    geometry['command'][:, 1] = torch.tensor([.08, -.08, .08, 0.])
    geometry['error'] = torch.tensor([.08**2, .08**2, 0., .08**2])
    reward = linear_tracking_reward(state, geometry)
    assert reward[0] == reward[1] < .1
    assert reward[2] == 1
    assert reward[3] > .9
    state.cfg.pop('motion_stability')
    torch.testing.assert_close(linear_tracking_reward(state, geometry), torch.exp(-geometry['error']/.25))
