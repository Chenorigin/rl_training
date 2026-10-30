"""Online reverse course: allocation, physical geometry and reward isolation."""
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'source/rl_training'))
from task_support.cts.configuration import default_settings
from task_support.cts.practice import (
    cohort_mask, practice_mask, validate_practice, requested_command,
    flat_patch_mesh, BackwardPractice, FlatCourseState, stair_only, practice_boundary, practice_timeout, explicit_pd_gains,
)
from task_support.stair_quiet_v1_width.geometry import face_profile_ids


def legacy_backward_settings():
    # Frozen historical schema; no active backward-only cohort exists in CTS v10.
    return {'fraction': 0.05, 'speed_range': [-0.7, -0.2], 'initial_hold_s': 0.4, 'reverse_s': 3.0, 'stop_s': 0.6, 'patch_length': 10.0, 'boundary_margin': 0.6, 'reset_xy_range': [0.15, 0.15], 'reset_yaw_range': 0.2, 'reset_velocity_range': 0.05, 'push_interval_range_s': [0.6, 1.2], 'push_velocity_range': [-0.12, 0.12], 'tracking_std': 0.5, 'linear_reward': 2.0, 'yaw_reward': 0.5, 'tilt_cost': 1.0, 'action_rate_cost': 0.02, 'effort_cost': 1e-05, 'failure_cost': 10.0}


def fake_env(n=20):
    settings = default_settings()
    cfg = SimpleNamespace(scene=SimpleNamespace(num_envs=n), sim=SimpleNamespace(device='cpu'),
                          stair_tail=SimpleNamespace(tile_width=5.))
    env = SimpleNamespace(cfg=cfg, device='cpu', num_envs=n, step_dt=.02,
                          episode_length_buf=torch.zeros(n, dtype=torch.long))
    env.cts_backward = BackwardPractice(env, legacy_backward_settings(), settings['algorithm'])
    env.scene = SimpleNamespace(env_origins=torch.zeros(n, 3))
    robot = SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.zeros(n,3)))
    class Scene(SimpleNamespace):
        def __getitem__(self, key):
            assert key == 'robot'
            return robot
    env.scene = Scene(env_origins=torch.zeros(n,3))
    return env


def test_fraction_is_stratified_by_actual_ppo_cohorts():
    teacher = cohort_mask(2048, .75, 42)
    selected = practice_mask(teacher, .1, 42)
    assert int((selected & teacher).sum()) == 154
    assert int((selected & ~teacher).sum()) == 51
    assert abs(float(selected.float().mean())-.1) < 1/2048
    small = fake_env().cts_backward
    assert int(small.mask.sum()) == 2  # Each nonempty policy cohort gets a practice environment.
    assert (small.mask & small.teacher).any() and (small.mask & ~small.teacher).any()
    assert not practice_mask(teacher, 0., 42).any()


def test_commands_cover_wait_reverse_and_stop_without_forward_leak():
    cfg = legacy_backward_settings()
    age = torch.tensor([0., cfg['initial_hold_s'], cfg['initial_hold_s']+cfg['reverse_s']])
    command = requested_command(age, torch.full((3,), -.5), cfg)
    torch.testing.assert_close(command, torch.tensor([0., -.5, 0.]))
    for key, value in (('fraction', .5), ('speed_range', [0., .5]), ('patch_length', 1.)):
        wrong = deepcopy(cfg);wrong[key] = value
        with pytest.raises(ValueError):validate_practice(wrong)


def test_flat_mesh_has_distinct_rows_and_real_original_contact_domains():
    generator = SimpleNamespace(num_rows=2, num_cols=5, size=(22., 5.))
    mesh, origins = flat_patch_mesh(generator, 10.)
    assert len(np.unique(origins.reshape(-1,3), axis=0)) == 10
    assert np.all(origins[:, :, 0] < -22.)
    np.testing.assert_array_equal(origins[:, :, 2], 0.)
    profiles = face_profile_ids(mesh, 5, 5., 5)
    top = mesh.face_normals[:,2] > .9
    assert np.bincount(profiles[top], minlength=5).tolist() == [4]*5
    for row in range(2):
        for col in range(5):
            v = mesh.vertices[(row*5+col)*8:(row*5+col+1)*8]
            assert abs(v[:,2].max()) < 1.e-9
            assert np.isclose(np.ptp(v[:,0]), 10.)


def test_every_lookup_reapplies_flat_geometry_without_mutating_tables():
    env = fake_env()
    adapter = env.cts_backward
    state = dict(height=torch.ones(20), steps=torch.ones(20, dtype=torch.long),
                 width=torch.ones(20), goal_height=torch.ones(20), spans=torch.ones(20,3,3),
                 round_trip=torch.ones(20,dtype=torch.bool), goal=torch.ones(20), ascent_end=torch.ones(20))
    for _ in range(3):
        state['height'].fill_(.15);state['width'].fill_(1.)
        adapter.flat_state(state)
        assert (state['height'][adapter.mask] == 0).all()
        assert (state['width'][adapter.mask] == 5).all()
        assert (state['height'][~adapter.mask] == .15).all()
        assert (state['spans'][adapter.mask, 0, 0] == -5).all()


def test_course_view_survives_legacy_reassignment_and_initial_backing_dict_writes():
    env = fake_env();practice = env.cts_backward
    backing = {'height': torch.full((20,), .15), 'width': torch.ones(20)}
    view = FlatCourseState(backing, practice)
    # Original cache constructor continues adding keys after assignment.
    backing['height_table'] = torch.full((20,), .3)
    for _ in range(3):
        view['height'] = view['height_table'].clone()
        view['width'] = torch.ones(20)
        assert (view['height'][practice.mask] == 0).all()
        assert (view['width'][practice.mask] == 5).all()
        assert (view['height_table'] == .3).all()
    assert 'height_table' in view and view.data is backing


def test_actual_region_kernel_uses_the_flat_spans_behind_original_start():
    from task_support.student_v3_safe.terrain_geometry import regions
    from task_support.stair_quiet_v1_safe.settings import SafeSettings
    env = fake_env();p = env.cts_backward
    state = dict(height=torch.ones(20), depth=torch.ones(20), steps=torch.ones(20,dtype=torch.long),
                 width=torch.ones(20), goal_height=torch.ones(20), spans=torch.zeros(20,3,3),
                 round_trip=torch.zeros(20,dtype=torch.bool), goal=torch.ones(20), ascent_end=torch.ones(20))
    p.flat_state(state)
    position = torch.zeros(20,4,3);position[:,:,0] = -2.;position[:,:,2] = .09
    axis = torch.zeros_like(position);axis[:,:,1] = 1.
    forward, _, top = regions(position, axis, state, SafeSettings())
    assert (forward[p.mask] > 2.).all()
    assert (top[p.mask] == 0.).all()


def test_stair_reward_and_termination_are_masked_only_on_practice():
    env = fake_env()
    values = torch.arange(20, dtype=torch.float32)
    wrapped = stair_only(lambda env, offset: values+offset)
    actual = wrapped(env, offset=1.)
    assert (actual[env.cts_backward.mask] == 0).all()
    torch.testing.assert_close(actual[~env.cts_backward.mask], (values+1)[~env.cts_backward.mask])
    boolean = stair_only(lambda env: torch.ones(20,dtype=torch.bool))(env)
    assert not boolean[env.cts_backward.mask].any()
    assert boolean[~env.cts_backward.mask].all()
    class Stateful:
        reset = Mock()
        def __call__(self, env):
            return values
    original = Stateful()
    wrapped = stair_only(original)
    wrapped.reset(env_ids=torch.tensor([1]))
    original.reset.assert_called_once()


def test_independent_boundary_and_timeout_do_not_relax_real_failure():
    env = fake_env()
    selected = env.cts_backward.mask.nonzero().flatten()
    env.scene['robot'].data.root_pos_w[:,2] = .5
    env.scene['robot'].data.root_pos_w[selected[0],0] = -4.6
    assert practice_boundary(env)[selected[0]]
    assert not practice_timeout(env).any()
    env.episode_length_buf[:] = 201
    assert torch.equal(practice_timeout(env), env.cts_backward.mask)
    env.scene['robot'].data.root_pos_w[selected[1],2] = .05
    assert practice_boundary(env)[selected[1]]


def test_accounting_restore_rejects_changed_allocation_and_negative_counts():
    env = fake_env();practice = env.cts_backward
    saved = practice.snapshot()
    practice.restore(saved)
    bad = deepcopy(saved);bad['mask'] = ~bad['mask']
    with pytest.raises(ValueError):practice.restore(bad)
    bad = deepcopy(saved);bad['pushes'][0] = -1
    with pytest.raises(ValueError):practice.restore(bad)


def test_gain_instrument_reads_actual_explicit_controller_not_zero_physx_drives():
    env = fake_env();mask = env.cts_backward.mask
    kp = torch.arange(20,dtype=torch.float32)[:,None].expand(-1,3)
    kd = kp*.1
    env.stair_quiet_landing = SimpleNamespace(_gains=[{'base_kp':kp,'base_kd':kd}])
    stiffness, damping = explicit_pd_gains(env, mask)
    torch.testing.assert_close(stiffness, kp[mask])
    torch.testing.assert_close(damping, kd[mask])
    assert (stiffness.amax(0)-stiffness.amin(0)).max() > 0
