"""Fake-physics regressions for course accounting and measured task outcomes.

The geometry and command code are real. Only the simulator articulation and
domain-parameter writes are replaced, so these tests perform no optimizer or
Isaac updates and never claim closed-loop robot capability.
"""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from task_support.cts.all_terrain_course import AllTerrainCourse
from task_support.cts.all_terrain_geometry import build_atlas
from task_support.cts.all_terrain_curriculum import CURRENT, CHALLENGE


@pytest.fixture(scope='module')
def source():
    settings = json.loads((Path(__file__).resolve().parents[1]/'configs/distillation/cts.json').read_text())
    for key in ('wall_hold_practice','stair_hold_practice','safety_learning','stair_recovery','motion_quality','stair_stop','obstacle_response','stair_walls','hind_ascent','wall_stability','clutter_consistency','front_descent','objective_consistency'):
        settings['all_terrain'].pop(key,None)
    settings['all_terrain'].pop('compound_commands', None)
    settings['all_terrain'].pop('wall_practice', None)
    settings['all_terrain']['fractions'] = [.28,.28,.14,.10,.03,.03,.02,.02,.05,.05]
    # Keep these established tests as the v13 compatibility contract.
    settings['all_terrain'].pop('motion_stability', None)
    settings['all_terrain'].pop('startup_stability', None)
    geometry = dict(settings['all_terrain']['geometry'], samples_per_level=3, grid_resolution=.25)
    return settings, build_atlas(geometry)


class Scene(SimpleNamespace):
    def __getitem__(self, name):
        return self.robot if name == 'robot' else self.sensors[name]


class Robot:
    def __init__(self, n):
        self.data = SimpleNamespace(default_root_state=torch.zeros(n, 13), root_pos_w=torch.zeros(n, 3),
            root_quat_w=torch.zeros(n, 4), root_lin_vel_b=torch.zeros(n, 3), root_ang_vel_b=torch.zeros(n, 3),
            projected_gravity_b=torch.zeros(n, 3), body_link_pos_w=torch.zeros(n, 4, 3),
            body_link_quat_w=torch.zeros(n, 4, 4), body_com_pos_w=torch.zeros(n, 4, 3),
            body_lin_vel_w=torch.zeros(n, 4, 3), body_ang_vel_w=torch.zeros(n, 4, 3))
        self.data.default_root_state[:, 2] = .5
        self.data.default_root_state[:, 3] = 1.
        self.data.root_quat_w[:, 0] = 1.
        self.data.body_link_quat_w[..., 0] = 1.
        self.data.projected_gravity_b[:, 2] = -1.

    def find_bodies(self, names, preserve_order=False):
        return list(range(4)), names

    def write_root_pose_to_sim(self, value, env_ids):
        self.data.root_pos_w[env_ids] = value[:, :3]
        self.data.root_quat_w[env_ids] = value[:, 3:]

    def write_root_velocity_to_sim(self, value, env_ids):
        self.data.root_lin_vel_b[env_ids] = value[:, :3]
        self.data.root_ang_vel_b[env_ids] = value[:, 3:]


def fake(source, n=400):
    settings, atlas = source
    cfg = deepcopy(settings['all_terrain'])
    env = SimpleNamespace(num_envs=n, device='cpu', step_dt=.02, common_step_counter=0,
        episode_length_buf=torch.zeros(n, dtype=torch.long),
        cfg=SimpleNamespace(cts_all_terrain=cfg, cts_cohort_config=settings['algorithm'],
            cts_platform_practice=settings['platform_practice'],
            stair_tail=SimpleNamespace(wheel_radius=.09, wheel_half_length=.027, support_force_N=10.,
                                       body_half_length=.25, body_half_width=.20),
            stair_quiet=SimpleNamespace(wheel_axis_local=(0., 1., 0.))))
    robot = Robot(n)
    sensor = SimpleNamespace(data=SimpleNamespace(net_forces_w=torch.zeros(n, 4, 3)),
                             find_bodies=lambda names, preserve_order=False: (list(range(4)), names))
    env.scene = Scene(robot=robot, terrain=SimpleNamespace(atlas=atlas,
        initial_tiles=torch.zeros(n, dtype=torch.long), terrain_levels=torch.zeros(n, dtype=torch.long),
        terrain_types=torch.zeros(n, dtype=torch.long), env_origins=torch.zeros(n, 3)),
        env_origins=torch.zeros(n, 3), sensors={'contact_forces':sensor})
    env.termination_manager = SimpleNamespace(terminated=torch.zeros(n, dtype=torch.bool),
                                              time_outs=torch.zeros(n, dtype=torch.bool))
    env._m20_student_v1_state = SimpleNamespace(zero_delivery=torch.zeros(n, dtype=torch.bool))
    state = env.cts_all_terrain = AllTerrainCourse(env)
    state.physics = SimpleNamespace(apply=Mock(), diagnostics=lambda: {})
    env.command_manager = SimpleNamespace(get_command=lambda name: state.raw_command())
    state.reset(torch.arange(n))
    support(env, state)
    return env, state


def support(env, state):
    """Place true four wheel capsules on each selected triangle surface."""
    d = env.scene.robot.data
    offsets = torch.tensor([[.25, .20], [.25, -.20], [-.25, .20], [-.25, -.20]])
    d.body_link_pos_w[..., :2] = d.root_pos_w[:, None, :2]+offsets
    height, normal = state.surface.sample(state.tile, d.body_link_pos_w[..., :2])
    extent = env.cfg.stair_tail.wheel_radius+env.cfg.stair_tail.wheel_half_length*normal[..., 1].abs()
    d.body_link_pos_w[..., 2] = height+extent/normal[..., 2]
    d.body_com_pos_w.copy_(d.body_link_pos_w)
    env.scene.sensors['contact_forces'].data.net_forces_w.copy_(normal*100.)
    state.measure_tick = -1


def frame(env, state, age, velocity=None):
    env.common_step_counter += 1
    if isinstance(age, torch.Tensor):
        env.episode_length_buf.copy_((age/env.step_dt).round().long())
    else:
        env.episode_length_buf.fill_(round(age/env.step_dt))
    if velocity == 'command':
        cmd = state.raw_command()
        env.scene.robot.data.root_lin_vel_b[:, :2] = cmd[:, :2]
        env.scene.robot.data.root_ang_vel_b[:, 2] = cmd[:, 2]
    elif velocity == 'zero':
        env.scene.robot.data.root_lin_vel_b.zero_()
        env.scene.robot.data.root_ang_vel_b.zero_()
    state.advance()


def selected(state, course=3, teacher=False):
    return ((state.course_ids == course) & (state.teacher == teacher)).nonzero().flatten()


def test_reset_balances_cohorts_and_exercises_each_command_mode(source):
    env, state = fake(source)
    expected = torch.tensor([28, 28, 14, 10, 3, 3, 2, 2, 5, 5])
    torch.testing.assert_close(torch.bincount(state.course_ids[~state.teacher], minlength=10), expected)
    torch.testing.assert_close(torch.bincount(state.course_ids[state.teacher], minlength=10), expected*3)
    seen = [set() for _ in range(env.num_envs)]
    for _ in range(9):
        for i, mode in enumerate(state.mode.tolist()):
            seen[i].add(mode)
        state.reset(torch.arange(env.num_envs))
    for i, modes in enumerate(seen):
        course = int(state.course_ids[i])
        assert modes == ({0} if course < 3 or course == 8 else {0, 8} if course == 9 else set(range(9)))
    assert state.physics.apply.call_count == 10


def test_observe_is_pure_even_on_a_new_tick_and_advance_counts_once(source):
    env, state = fake(source)
    env.episode_length_buf.fill_(50)
    before = {name:getattr(state, name).clone() for name in ('frames', 'tracking_frames', 'success', 'failed', 'stable_age')}
    first = state.observe()
    assert state.observe() is first
    env.common_step_counter += 1
    state.observe()
    for name, value in before.items():
        torch.testing.assert_close(getattr(state, name), value)
    state.advance()
    assert int(state.frames.sum()) == env.num_envs
    saved = {name:getattr(state, name).clone() for name in before}
    for _ in range(5):
        state.observe()
        state.advance()
    for name, value in saved.items():
        torch.testing.assert_close(getattr(state, name), value)
    env.common_step_counter += 1
    state.advance()
    assert int(state.frames.sum()) == 2*env.num_envs


def test_partial_reset_does_not_recount_unchanged_physics_tick(source):
    env, state = fake(source)
    frame(env, state, 1., 'command')
    frames = state.frames.clone()
    tracking = state.tracking_frames.clone()
    ids = torch.tensor([0])
    state.reset(ids)
    support(env, state)
    state.observe()
    state.advance()
    torch.testing.assert_close(state.frames, frames)
    torch.testing.assert_close(state.tracking_frames[1:], tracking[1:])


def test_slow_reverse_standing_still_cannot_pass_tracking_and_stop(source):
    env, state = fake(source)
    ids = selected(state)[:2]
    good, bad = ids.tolist()
    state.mode[ids] = 1
    state.command_value[ids] = torch.tensor([-.15, 0., 0.])
    state.move_duration[ids] = 5.
    env.scene.robot.data.root_lin_vel_b[good, 0] = -.15
    frame(env, state, 1.)
    assert state.tracking_frames[good] == 1 and state.tracking_error[good].sum() == 0
    assert state.tracking_error[bad, 0] > 0
    for i in range(12):
        frame(env, state, 5.5+i*.02, 'zero')
    assert state.success[good]
    assert not state.success[bad]


def test_backward_duration_reaches_beyond_spawn_and_raw_phase_boundaries(source):
    env, state = fake(source)
    ids = selected(state)
    state.resets[ids] = 1-state.rank[ids]
    state.reset(ids)
    assert (state.mode[ids] == 1).all()
    assert (state.command_value[ids, 0] < 0).all()
    assert (state.move_duration[ids]*state.command_value[ids, 0].abs() >= 2.5-1e-5).all()
    env.episode_length_buf.fill_(19)
    assert (state.raw_command() == 0).all()
    env.episode_length_buf.fill_(20)
    assert (state.raw_command()[ids, 0] < 0).all()
    end = state.cfg['commands']['initial_hold_s']+state.move_duration
    env.episode_length_buf.copy_((end/env.step_dt).ceil().long())
    assert (state.raw_command()[ids] == 0).all()
    assert (state.timeout[ids] > end[ids]).all()


def test_stop_requires_actual_rest_and_timeout_without_completion_is_failure(source):
    env, state = fake(source)
    ids = selected(state)[:2]
    good, moving = ids.tolist()
    state.mode[ids] = 8
    state.command_value[ids] = 0
    state.move_duration[ids] = 1.
    frame(env, state, .6, 'zero')
    for i in range(12):
        env.scene.robot.data.root_lin_vel_b[moving, 0] = .12
        frame(env, state, 1.5+i*.02)
    assert state.success[good] and not state.success[moving]
    env.termination_manager.time_outs[ids] = True
    state.record(ids)
    assert int(state.successes[1, 3, 8]) == 1
    assert int(state.failures[1, 3, 8]) == 1
    state.record(ids)
    assert int(state.episodes[1, 3, 8]) == 2


def test_stair_success_needs_all_wheels_past_goal_and_stable_support(source):
    env, state = fake(source)
    i = int(selected(state, course=0)[0])
    robot = env.scene.robot.data
    goal = state.goals[state.tile[i], 0]
    robot.root_pos_w[i, 0] = goal+.1
    robot.root_pos_w[i, 2] = state.goals[state.tile[i], 2]+.5
    support(env, state)
    frame(env, state, 10., 'zero')
    assert not state.reached[i] and not state.success[i]
    robot.root_pos_w[i, 0] = goal+.5
    support(env, state)
    frame(env, state, 10.02, 'zero')
    assert state.reached[i] and not state.success[i]
    for tick in range(12):
        frame(env, state, 10.04+tick*.02, 'zero')
    assert state.success[i]


def test_failure_overrides_latched_success_and_records_exactly_once(source):
    env, state = fake(source)
    ids = selected(state)[:3]
    success, failed, initial = ids.tolist()
    env.episode_length_buf[ids] = 10
    env.episode_length_buf[initial] = 0
    state.success[ids] = True
    state.failed[failed] = True
    env.termination_manager.terminated[ids] = True
    state.mode[ids] = 0
    before = int(state.episodes.sum())
    state.record(ids)
    state.record(ids)
    assert int(state.episodes.sum()) == before+2
    assert int(state.successes[1, 3, 0]) == 1
    assert int(state.failures[1, 3, 0]) == 1
    assert state.finished[success] and state.finished[failed] and not state.finished[initial]


def test_teacher_and_stale_or_challenge_student_results_do_not_promote(source):
    env, state = fake(source)
    group = 15
    teacher = int(selected(state, teacher=True)[0])
    student, stale, challenge = selected(state)[:3].tolist()
    ids = torch.tensor([teacher, student, stale, challenge])
    # Advance one grade through real student evidence, retaining stale episode 0.
    for _ in range(200):
        state.curriculum.record(group, True, False, CURRENT, 0, 0, boundary=True, delayed=True)
    assert state.curriculum.current(group)['version'] == 1
    state.group[ids] = group
    state.role[ids] = CURRENT
    state.role[challenge] = CHALLENGE
    state.version[ids] = 1
    state.version[stale] = 0
    state.success[ids] = True
    env.episode_length_buf[ids] = 10
    env.termination_manager.terminated[ids] = True
    state.record(ids)
    info = state.curriculum.diagnostics()[group]
    assert info['window']['episodes'] == 1
    assert info['teacher_episodes'] == 1
    assert info['ignored_version'] == 1 and info['ignored_role'] == 1
    assert state.curriculum.current(group)['version'] == 1


def test_snapshot_restores_sampling_and_counters_without_counting_inflight_episodes(source):
    env, state = fake(source)
    frame(env, state, 1., 'command')
    saved = deepcopy(state.snapshot())
    ids = torch.arange(env.num_envs)
    state.reset(ids)
    expected_tiles, expected_roles = state.tile.clone(), state.role.clone()
    state.restore(saved)
    assert state.finished.all()
    env.termination_manager.time_outs.fill_(True)
    before = state.episodes.clone()
    state.record(ids)
    torch.testing.assert_close(state.episodes, before)
    state.reset(ids)
    torch.testing.assert_close(state.tile, expected_tiles)
    torch.testing.assert_close(state.role, expected_roles)
    bad = deepcopy(saved)
    bad['course_ids'][0] = (bad['course_ids'][0]+1) % 10
    with pytest.raises(ValueError, match='allocation'):
        state.restore(bad)


@pytest.mark.parametrize('mode', [2, 3, 4, 5, 6, 7, 8],
                         ids=['turn_left', 'turn_right', 'forward_left', 'forward_right',
                              'backward_left', 'backward_right', 'idle'])
@pytest.mark.parametrize('course_id,variant', [(3, 'slope'), (3, 'wave'), (4, 'rough'),
                                            (5, 'blocks'), (6, 'pits'), (7, 'mixed')])
def test_turning_and_stationary_modes_spawn_on_actual_terrain_above_all_four_supports(source, mode, course_id, variant):
    env, state = fake(source)
    # Pin only the motion subtype; difficulty and continuous geometry remain
    # selected by the real sampler. Exercise both teacher and student resets.
    if course_id == 3:
        for grade in range(state.cfg['geometry']['geometry_levels']):
            pool = state.pools[course_id, grade]
            state.pools[course_id, grade] = pool[[state.atlas.records[i]['variant'] == variant for i in pool]]
    ids = (state.course_ids == course_id).nonzero().flatten()
    assert state.teacher[ids].any() and (~state.teacher[ids]).any()
    state.resets[ids] = mode-state.rank[ids]
    state.reset(ids)
    assert (state.mode[ids] == mode).all()
    root = env.scene.robot.data.root_pos_w[ids]
    local = root-state.origins[state.tile[ids]]
    assert (local[:, 0].abs() > state.cfg['geometry']['spawn_flat_half_length']).all()

    hx, hy = env.cfg.stair_tail.body_half_length, env.cfg.stair_tail.body_half_width
    offsets = torch.tensor([[-hx, -hy], [-hx, hy], [hx, -hy], [hx, hy]])
    height, normal = state.surface.sample(state.tile[ids], root[:, None, :2]+offsets)
    center_height, _ = state.surface.sample(state.tile[ids], root[:, :2])
    assert torch.isfinite(height).all()
    # The supports must actually see varied geometry, not just translated XY
    # coordinates over another flat launch pad. Pits remain below ground.
    assert ((height-height[:, :1]).abs().amax(-1) > 1e-5).all()
    if variant in ('slope', 'wave', 'rough', 'pits', 'mixed'):
        assert (normal[..., :2].abs().amax((1, 2)) > 1e-5).all()
    required_z = torch.maximum(center_height, height.amax(-1))+env.scene.robot.data.default_root_state[ids, 2]
    assert (root[:, 2] >= required_z-1e-5).all()
    if variant == 'pits':
        assert (height.amin(-1) < -1e-5).all()


def test_curriculum_diagnostics_report_exact_partial_window_and_version_counters(source):
    _, state = fake(source)
    group = 4*5+1  # rough/backward; adjacent direction/course must stay empty.
    for i in range(200):
        state.curriculum.record(group, i % 100 < 90, False, CURRENT, 0, 0,
                                boundary=i % 100 < 10, delayed=i % 100 < 30)
    assert state.curriculum.current(group)['version'] == 1
    state.curriculum.record(group, True, True, CURRENT, 1, 1)
    state.curriculum.record(group, True, False, CURRENT, 0, 1)
    state.curriculum.record(group, True, False, CHALLENGE, 1, 1)
    for success, boundary, delayed in ((True, True, True), (False, False, True), (True, False, False)):
        state.curriculum.record(group, success, False, CURRENT, 1, 1, boundary=boundary, delayed=delayed)
    diag = state.diagnostics()
    expected = {'version': 1, 'promotions': 1, 'demotions': 0, 'ignored_role': 1,
                'ignored_version': 1, 'expired_windows': 0, 'completed_windows': 2,
                'window_episodes': 3, 'window_successes': 2, 'window_boundary': 1,
                'window_delayed': 2, 'last_success_rate': .9, 'consecutive_windows': 0}
    for key, value in expected.items():
        assert diag['curriculum/rough/backward/'+key] == value
    assert diag['level/rough/backward/geometry'] == 1
    assert diag['level/rough/backward/physics'] == 0
    for prefix in ('curriculum/rough/forward/', 'curriculum/blocks/backward/'):
        assert diag[prefix+'window_episodes'] == 0
        assert diag[prefix+'completed_windows'] == 0
        assert diag[prefix+'last_success_rate'] == -1.

    # An unfinished stale window is visible as expired, and a fresh result
    # starts a new window rather than silently preserving its old counts.
    update = state.cfg['curriculum']['max_window_updates']+2
    state.curriculum.record(group, False, False, CURRENT, 1, update, boundary=False, delayed=True)
    diag = state.diagnostics()
    assert diag['curriculum/rough/backward/expired_windows'] == 1
    assert diag['curriculum/rough/backward/window_episodes'] == 1
    assert diag['curriculum/rough/backward/window_successes'] == 0
    assert diag['curriculum/rough/backward/window_boundary'] == 0
    assert diag['curriculum/rough/backward/window_delayed'] == 1


def test_contact_validation_rejects_visible_floor_without_physical_support(source):
    env, state = fake(source)
    # Correct ground geometry and positive ray counts do not establish contact.
    state.ray_checks = 1000
    env.scene.sensors['contact_forces'].data.net_forces_w.zero_()
    frame(env, state, .1)
    with pytest.raises(RuntimeError, match='No measured wheel/terrain support'):
        state.validate_actual_contacts()
    assert not state.contact_frames.any()
    support(env, state)
    frame(env, state, .12)
    state.validate_actual_contacts()
    assert (state.contact_frames > 0).all()
    saved = state.snapshot()
    state.contact_frames.zero_()
    state.restore(saved)
    state.validate_actual_contacts()
