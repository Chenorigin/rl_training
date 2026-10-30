"""v14 real course/surface logic with bounded CPU articulation fixtures.

No simulator step, optimizer update or robot-capability claim is made here.
"""
from copy import deepcopy
import json
import math
from pathlib import Path

import pytest
import torch

from task_support.cts.all_terrain_geometry import build_atlas
from task_support.cts.command_sequences import command_sequence, lateral_spawn
from test_cts_all_terrain_course import fake, frame, selected, support


STABILITY = dict(lateral_speed=[.08, .35], lateral_travel_m=.6,
                 command_ramp_s=[.15, .45], abrupt_command_probability=.25,
                 arc_transition_probability=.5)


@pytest.fixture(scope='module')
def sources():
    settings = json.loads((Path(__file__).resolve().parents[1]/'configs/distillation/cts.json').read_text())
    settings['all_terrain'].pop('compound_commands', None)
    settings['all_terrain'].pop('wall_practice', None)  # Freeze pre-wall mode/geometry contract.
    current = deepcopy(settings)
    settings['all_terrain'].pop('motion_stability', None)
    settings['all_terrain'].pop('startup_stability', None)
    geometry = dict(settings['all_terrain']['geometry'], samples_per_level=3, grid_resolution=.25)
    atlas = build_atlas(geometry)
    current['all_terrain']['motion_stability'] = deepcopy(STABILITY)
    return (settings, atlas), (current, atlas)


def force_modes(env, state, ids, modes):
    state.resets[ids] = modes-state.rank[ids]
    state.reset(ids)
    support(env, state)


def test_missing_stability_preserves_nine_modes_five_groups_and_snapshot(sources):
    env, state = fake(sources[0])
    assert state.frames.shape == (2, 10, 9)
    assert state.num_groups == 5
    assert 'motion_stability' not in state.snapshot()
    for _ in range(9):
        assert not state.command_value[:, 1].any()
        env.episode_length_buf.fill_(20)
        torch.testing.assert_close(state.raw_command(), state.command_value)
        state.reset(torch.arange(env.num_envs))
    # Legacy snapshots still restore with exactly their original key set.
    state.restore(deepcopy(state.snapshot()))


def test_both_cohorts_visit_left_and_right_on_every_motion_terrain(sources):
    env, state = fake(sources[1])
    seen = [set() for _ in range(env.num_envs)]
    for _ in range(11):
        for i, mode in enumerate(state.mode.tolist()):
            seen[i].add(mode)
        side = state.mode >= 9
        assert (state.command_value[side, 0] == 0).all()
        assert (state.command_value[side, 2] == 0).all()
        speed = state.command_value[side, 1].abs()
        assert ((speed >= .08) & (speed <= .35)).all()
        assert (state.command_value[state.mode == 9, 1] > 0).all()
        assert (state.command_value[state.mode == 10, 1] < 0).all()
        assert (state.group[side] == state.course_ids[side]*7+state.mode[side]-4).all()
        state.reset(torch.arange(env.num_envs))
    for i, modes in enumerate(seen):
        c = int(state.course_ids[i])
        assert modes == ({0} if c < 3 or c == 8 else {0, 8} if c == 9 else set(range(11)))
    assert state.frames.shape == (2, 10, 11)


def test_lateral_path_stays_in_support_corridor_and_has_requested_travel(sources):
    env, state = fake(sources[1])
    ids = ((state.course_ids >= 3) & (state.course_ids <= 7)).nonzero().flatten()
    modes = torch.where(torch.arange(len(ids)) % 2 == 0, 9, 10)
    force_modes(env, state, ids, modes)
    root = env.scene.robot.data.root_pos_w[ids]
    bounds = state.bounds[state.tile[ids]]
    footprint = math.hypot(.25, .20)+.09+.027+state.cfg['rewards']['boundary_margin']
    signed = torch.where(modes == 9, 1., -1.)
    left = torch.stack((-state.initial_yaw[ids].sin(), state.initial_yaw[ids].cos()), -1)
    end = root[:, :2]+left*signed[:, None]*.6
    for points in (root[:, :2], end):
        assert (points[:, 0] >= bounds[:, 0]+footprint-1e-5).all()
        assert (points[:, 0] <= bounds[:, 1]-footprint+1e-5).all()
        assert (points[:, 1] >= bounds[:, 2]+footprint-1e-5).all()
        assert (points[:, 1] <= bounds[:, 3]-footprint+1e-5).all()
    distance = state.command_value[ids, 1].abs()*(state.move_duration[ids]-state.command_ramp[ids])
    torch.testing.assert_close(distance, torch.full_like(distance, .6))
    # Actual terrain is still queried after the corridor clamp.
    height, _ = state.surface.sample(state.tile[ids], root[:, :2])
    assert (root[:, 2] >= height+env.scene.robot.data.default_root_state[ids, 2]-1e-5).all()
    assert ((root[:, 0]-state.origins[state.tile[ids], 0]).abs() > .9).all()


def test_lateral_spawn_rejects_a_field_that_cannot_hold_the_robot_and_path():
    with pytest.raises(ValueError, match='cannot contain'):
        lateral_spawn(torch.zeros(1, 2), torch.tensor([[-1., 1., -.5, .5]]),
                      torch.tensor([9]), .6, .25, .2, .09, .027, .2)


@pytest.mark.parametrize('ramp', [0., .4])
def test_real_raw_command_turns_during_translation_and_returns_to_straight(sources, ramp):
    env, state = fake(sources[1])
    ids = selected(state)[:2]
    force_modes(env, state, ids, torch.tensor([4, 5]))
    state.command_value[ids] = torch.tensor([[.4, 0., .2], [.4, 0., -.2]])
    state.move_duration[ids] = 8.
    state.command_ramp[ids] = ramp
    state.arc_transition[ids] = True
    for t, yaw_factor in ((1., 0.), (3., 1.), (5., -1.), (7., 0.)):
        env.episode_length_buf.fill_(round((t+.4)/env.step_dt))
        cmd = state.raw_command()[ids]
        torch.testing.assert_close(cmd[:, 0], torch.full((2,), .4))
        torch.testing.assert_close(cmd[:, 2], torch.tensor([.2, -.2])*yaw_factor)
    # Zero and ramped schedules include both command onset and stopping.
    env.episode_length_buf.fill_(round(8.4/env.step_dt))
    assert not state.raw_command()[ids].any()
    state.arc_transition[ids] = False
    env.episode_length_buf.fill_(round(3.4/env.step_dt))
    torch.testing.assert_close(state.raw_command()[ids, 2], torch.tensor([.2, -.2]))


def test_sequence_smooths_the_sign_change_without_modifying_forward_velocity():
    t = torch.linspace(3.9, 4.5, 301)
    command = torch.tensor([[.4, 0., .2]]).repeat(len(t), 1)
    smooth = command_sequence(command, t, torch.full_like(t, 8.), torch.full_like(t, .4),
                              torch.ones(len(t), dtype=torch.bool))
    torch.testing.assert_close(smooth[:, 0], torch.full_like(t, .4))
    assert (smooth[1:, 2]-smooth[:-1, 2]).abs().max() < .0031
    abrupt = command_sequence(command, t, torch.full_like(t, 8.), torch.zeros_like(t),
                              torch.ones(len(t), dtype=torch.bool))
    assert (abrupt[1:, 2]-abrupt[:-1, 2]).abs().max() > .39


def test_new_sequences_do_not_time_out_stair_goal_commands(sources):
    env, state = fake(sources[1])
    stairs = state.stairs
    env.episode_length_buf.fill_(round(15/env.step_dt))
    assert (state.raw_command()[stairs, 0] > 0).all()
    state.reached[stairs] = True
    assert not state.raw_command()[stairs].any()


def test_transition_spawn_bounds_the_actual_nominal_sequence_on_each_terrain(sources):
    env, state = fake(sources[1])
    state.motion_stability['arc_transition_probability'] = 1.
    ids = ((state.course_ids >= 3) & (state.course_ids <= 7)).nonzero().flatten()
    modes = 4+torch.arange(len(ids)) % 4
    force_modes(env, state, ids, modes)
    assert state.arc_transition[ids].all()
    # Integrate commands only, not dynamics, to test the geometric admission
    # bound for every sampled speed, turn direction and randomized initial yaw.
    dt = .01
    time = torch.arange(0., float(state.move_duration[ids].max())+dt, dt)
    n, steps = len(ids), len(time)
    commands = command_sequence(state.command_value[ids].repeat(steps, 1),
        time[:,None].expand(-1,n).reshape(-1), state.move_duration[ids].repeat(steps),
        state.command_ramp[ids].repeat(steps), state.arc_transition[ids].repeat(steps)).reshape(steps,n,3)
    heading = state.initial_yaw[ids]+commands[:,:,2].cumsum(0)*dt
    displacement = torch.stack((commands[:,:,0]*heading.cos(),commands[:,:,0]*heading.sin()),-1).cumsum(0)*dt
    path = env.scene.robot.data.root_pos_w[ids,:2]+displacement
    b = state.bounds[state.tile[ids]]
    footprint = math.hypot(.25,.20)+.09+.027+state.cfg['rewards']['boundary_margin']
    assert (path[:,:,0] >= b[:,0]+footprint-1e-3).all()
    assert (path[:,:,0] <= b[:,1]-footprint+1e-3).all()
    assert (path[:,:,1] >= b[:,2]+footprint-1e-3).all()
    assert (path[:,:,1] <= b[:,3]-footprint+1e-3).all()


def test_lateral_success_requires_world_displacement_not_tilt_or_local_velocity(sources):
    env, state = fake(sources[1], n=120)
    ids = selected(state)[:3]
    assert len(ids) == 3
    force_modes(env, state, ids, torch.full((3,), 9))
    good, stationary, tilt_only = ids.tolist()
    state.command_value[ids] = torch.tensor([0., .15, 0.])
    state.move_duration[ids] = 4.
    state.command_ramp[ids] = 0.
    state.initial_yaw[ids] = 0.
    planted = env.scene.robot.data.body_link_pos_w[tilt_only, :, :2].clone()
    # Local velocities deliberately look perfect for all three: this negative
    # fixture proves the independent world-position guard is actually wired.
    for tick in range(1, 241):
        age = tick*env.step_dt
        env.episode_length_buf.fill_(tick)
        cmd = state.raw_command()
        env.scene.robot.data.root_pos_w[good, 1] += cmd[good, 1]*env.step_dt
        # Lean far enough that the root-only displacement check would pass.
        env.scene.robot.data.root_pos_w[tilt_only, 1] += cmd[tilt_only, 1]*env.step_dt*.5
        env.scene.robot.data.root_quat_w[tilt_only] = torch.tensor([math.cos(.1), math.sin(.1), 0., 0.])
        env.scene.robot.data.projected_gravity_b[tilt_only, 2] = -math.cos(.2)
        support(env, state)
        env.scene.robot.data.body_link_pos_w[tilt_only, :, :2] = planted
        frame(env, state, age, 'command')
    assert state.tracking_ok[ids].all()
    assert state.success[good]
    assert not state.success[stationary] and not state.success[tilt_only]
    assert state.lateral_actual[good] > .59
    assert state.lateral_actual[stationary] == 0
    assert state.lateral_actual[tilt_only] > .29
    assert state.lateral_support_actual[tilt_only] == 0
    assert state.lateral_expected[ids].min() > .59
    env.termination_manager.time_outs[ids] = True
    state.record(ids)
    d = state.diagnostics()
    prefix = 'student/motion/command/left/'
    assert d[prefix+'episodes'] == 3
    assert d[prefix+'success_rate'] == pytest.approx(1/3)
    assert d[prefix+'failure_displacement'] == 2
    assert d[prefix+'failure_timeout'] == 2
    assert d[prefix+'failure_tracking'] == 0


def test_straight_phases_cannot_hide_ignored_turns_in_sequence_success(sources):
    env, state = fake(sources[1], n=120)
    ids = selected(state)[:2]
    force_modes(env, state, ids, torch.tensor([4, 4]))
    good, never_turns = ids.tolist()
    state.command_value[ids] = torch.tensor([.3, 0., .1])
    state.move_duration[ids] = 8.
    state.command_ramp[ids] = .4
    state.arc_transition[ids] = True
    for tick in range(1, 451):
        env.episode_length_buf.fill_(tick)
        cmd = state.raw_command()
        env.scene.robot.data.root_lin_vel_b[:, :2] = cmd[:, :2]
        env.scene.robot.data.root_ang_vel_b[:, 2] = cmd[:, 2]
        env.scene.robot.data.root_ang_vel_b[never_turns, 2] = 0.
        frame(env, state, tick*env.step_dt)
    assert state.success[good]
    assert not state.success[never_turns]
    assert state.tracking_ok[good] and not state.tracking_ok[never_turns]


def test_diagnostics_separate_failure_causes_and_restore_completed_motion_ledger(sources):
    env, state = fake(sources[1], n=120)
    ids = selected(state)[:3]
    force_modes(env, state, ids, torch.tensor([9, 10, 4]))
    env.episode_length_buf[ids] = 40
    state.tracking_frames[ids] = 20
    state.tracking_error[ids] = .5
    state.failed[ids[0]] = True
    state.tracking_ok[ids[1]] = False
    state.stop_ok[ids[2]] = False
    state.arc_transition[ids[2]] = True
    state.lateral_actual[ids[1]] = -.1  # Signed backtracking is valid evidence.
    env.termination_manager.terminated[ids[0]] = True
    env.termination_manager.time_outs[ids[1:]] = True
    state.record(ids)
    saved = deepcopy(state.snapshot())
    d = state.diagnostics()
    assert d['student/motion/command/left/failure_physical'] == 1
    assert d['student/motion/command/right/failure_tracking'] == 1
    assert d['student/motion/command/forward_left/failure_stop'] == 1
    assert d['student/motion/command/forward_left/failure_timeout'] == 1
    assert d['student/motion/command/forward_left/transition_episodes'] == 1
    for values in (*state.failure_counts.values(), *state.motion_totals.values(), *state.sequence_counts.values()):
        values.zero_()
    state.restore(saved)
    assert state.diagnostics() == d
    assert state.finished.all()
    # Old/new ledgers cannot silently restore into the wrong mode dimension.
    legacy = fake(sources[0], n=120)[1]
    with pytest.raises(ValueError, match='snapshot keys'):
        legacy.restore(saved)
    bad = deepcopy(saved)
    bad['motion_stability']['motion_totals']['linear_tracking_rms_sum'][0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match='motion snapshot'):
        state.restore(bad)
