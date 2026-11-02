"""Stair-stop counterexamples use the actual course, reward and restore paths."""
from copy import deepcopy
import math
import pytest
import torch

from task_support.cts.configuration import default_settings, validate_config, LEGACY_VERSIONS
from task_support.cts.stair_stop import validate
from task_support.cts.all_terrain_terms import reward
from test_cts_all_terrain_course import source, fake, support, frame
from test_cts_objective_consistency import reward_env


def setup(source, enabled=True):
    cfg, atlas = source
    cfg = deepcopy(cfg)
    for key in ('motion_stability', 'startup_stability', 'objective_consistency', 'stair_recovery', 'stair_stop'):
        cfg['all_terrain'][key] = default_settings()['all_terrain'][key]
    cfg['all_terrain']['stair_stop']['enabled'] = enabled
    env, s = fake((cfg, atlas), n=100)
    d = env.scene.robot.data
    d.root_lin_vel_w = d.root_lin_vel_b  # identity orientation in these fixtures
    d.root_quat_w[:] = torch.tensor([1., 0., 0., 0.])
    reward_env(env)
    r = s.recovery
    r.phase.zero_();r.perturb_pending.zero_()
    i = int(((s.course_ids == 0) & ~s.teacher).nonzero()[0])
    r.phase[i] = 1
    d.root_pos_w[i, 0] = r.trigger_x[i]-.2
    support(env, s)
    return env, s, i


def enter(env, s, i, speed=0.):
    env.scene.robot.data.root_lin_vel_w[i, 0] = speed
    frame(env, s, 3.)
    assert s.recovery.phase[i] == 2


def settle(env, s, count=23):
    for _ in range(count):
        frame(env, s, float(env.episode_length_buf[0]*env.step_dt)+env.step_dt, 'zero')


def test_entering_step_not_credited_and_resume_step_keeps_executed_zero_reward(source):
    env, s, i = setup(source);h = s.recovery.hold
    enter(env, s, i)
    assert not h.active[i] and s.recovery.settled_age[i] == 0
    anchor = h.anchor[i].clone()
    for _ in range(30):
        frame(env, s, float(env.episode_length_buf[0]*env.step_dt)+env.step_dt, 'zero')
        if s.recovery.phase[i] == 3:break
    assert s.recovery.phase[i] == 3 and h.active[i] and h.ready[i]
    assert h.totals['resumes'][1, 0] == 1
    assert s.observe()['command'][i, 0] == 0
    assert s.raw_command()[i, 0] > 0
    # The first actually forward step is no longer position constrained.
    env.scene.robot.data.root_pos_w[i, 0] += .4
    support(env, s);frame(env, s, 3.8, 'command')
    assert not h.active[i]
    torch.testing.assert_close(h.anchor[i], anchor)
    x = torch.rand(env.num_envs);b = torch.ones(env.num_envs, dtype=torch.bool)
    lin, ang, stopped, cost = h.reward_terms(x, x, b)
    assert lin[i] == ang[i] == x[i] and stopped[i] and cost[i] == 0


def test_retreat_then_stationary_is_rejected_but_legacy_accepts(source):
    for enabled in (True, False):
        env, s, i = setup(source, enabled);enter(env, s, i)
        env.scene.robot.data.root_pos_w[i, 0] -= .66
        support(env, s);settle(env, s)
        if enabled:
            h = s.recovery.hold
            assert not h.region[i] and not h.ready[i]
            assert s.recovery.phase[i] == 2 and not h.reward_terms(torch.ones(100), torch.ones(100), torch.ones(100,dtype=torch.bool))[2][i]
            assert h.max_back[i] == pytest.approx(.66, abs=1e-4)
        else:
            assert s.recovery.phase[i] == 3  # known-bad v19 counterexample


def test_finite_braking_envelope_uses_real_speed_and_tread_depth(source):
    env, s, i = setup(source);h = s.recovery.hold
    enter(env, s, i, speed=.5)
    expected = min(h.cfg['forward_base_m']+.5**2/(2*h.cfg['braking_m_s2']),
                   h.cfg['forward_max_m'],float(h.depth[s.tile[i]])*h.cfg['tread_fraction'])
    assert h.forward[i] == pytest.approx(expected)
    env.scene.robot.data.root_pos_w[i, 0] += expected*.5
    support(env, s);settle(env, s)
    assert s.recovery.phase[i] == 3


@pytest.mark.parametrize('axis,amount', [(0, .3), (1, .3)])
def test_outside_forward_or_side_region_cannot_resume(source, axis, amount):
    env, s, i = setup(source);enter(env, s, i)
    env.scene.robot.data.root_pos_w[i, axis] += amount
    support(env, s);settle(env, s)
    assert s.recovery.phase[i] == 2 and not s.recovery.hold.ready[i]


def test_slow_drift_and_sliding_are_worse_than_supported_stationary_in_real_reward(source):
    env, s, i = setup(source);enter(env, s, i);frame(env, s, 3.02, 'zero')
    stationary = float(reward(env)[i]);h = s.recovery.hold
    env.scene.robot.data.root_lin_vel_w[i, 0] = -.15
    frame(env, s, 3.04)
    assert not h.ready[i] and float(reward(env)[i]) < stationary-1.
    env.scene.robot.data.root_lin_vel_w.zero_()
    env.scene.robot.data.body_lin_vel_w[i, 2:, 0] = -.2
    frame(env, s, 3.06)
    assert not h.ready[i] and float(reward(env)[i]) < stationary
    assert h.totals['slip_distance_m'][1, 0] > 0


def test_footprint_retreat_cannot_hide_behind_root_pose(source):
    env, s, i = setup(source);enter(env, s, i)
    d = env.scene.robot.data
    d.body_link_pos_w[i, :, 0] -= .4
    d.body_com_pos_w.copy_(d.body_link_pos_w)
    settle(env, s)
    assert not s.recovery.hold.region[i] and s.recovery.phase[i] == 2


def test_heading_wrap_and_yaw_region(source):
    env, s, i = setup(source);d = env.scene.robot.data
    d.root_quat_w[i] = torch.tensor([math.cos((math.pi-.01)/2), 0., 0., math.sin((math.pi-.01)/2)])
    enter(env, s, i)
    d.root_quat_w[i] = torch.tensor([math.cos((-math.pi+.01)/2), 0., 0., math.sin((-math.pi+.01)/2)])
    frame(env, s, 3.02, 'zero');assert s.recovery.hold.region[i]
    d.root_quat_w[i] = torch.tensor([1., 0., 0., 0.])
    frame(env, s, 3.04, 'zero');assert not s.recovery.hold.ready[i]


def test_two_diagonal_supports_allowed_but_front_only_cannot_resume(source):
    for contacts, accepted in (([0, 3], True), ([0, 1], False)):
        env, s, i = setup(source);enter(env, s, i)
        f = env.scene.sensors['contact_forces'].data.net_forces_w
        f[i] = 0;f[i, contacts, 2] = 100
        settle(env, s)
        assert (s.recovery.phase[i] == 3) == accepted


def test_more_contact_force_does_not_buy_more_stop_reward(source):
    env, s, i = setup(source);enter(env, s, i);frame(env, s, 3.02, 'zero')
    before = float(reward(env)[i])
    env.scene.sensors['contact_forces'].data.net_forces_w[i] *= 10
    frame(env, s, 3.04, 'zero')
    assert float(reward(env)[i]) == pytest.approx(before)


@pytest.mark.parametrize('field', ['body_lin_vel_w', 'root_pos_w'])
def test_nonfinite_measurement_fails_closed_and_counters_stay_finite(source, field):
    env, s, i = setup(source);enter(env, s, i)
    getattr(env.scene.robot.data, field)[i] = float('nan')
    frame(env, s, 3.02, 'zero')
    assert s.failed[i] and not s.recovery.hold.ready[i]
    assert torch.isfinite(reward(env)[i])
    assert all(torch.isfinite(v).all() for v in s.recovery.hold.totals.values())


def test_record_reset_snapshot_and_repeat_read_are_consistent(source):
    env, s, i = setup(source);h=s.recovery.hold;enter(env,s,i)
    env.scene.robot.data.root_pos_w[i,0] -= .4;support(env,s)
    settle(env,s,count=153)
    assert s.failed[i] and h.totals['timeouts'][1,0] == 1
    previous=h.snapshot();reward(env);reward(env);s.advance();h.diagnostics()
    for k,v in previous.items():torch.testing.assert_close(v,h.totals[k])
    env.termination_manager.terminated[i]=True;s.record(torch.tensor([i]));s.record(torch.tensor([i]))
    assert h.totals['completed_pauses'][1,0] == 1
    saved=s.snapshot();s.restore(saved)
    for k,v in saved['stair_stop'].items():torch.testing.assert_close(v,h.totals[k])
    bad=deepcopy(saved);bad['stair_stop']['entries'][1,0]=float('nan')
    with pytest.raises(ValueError,match='stair_stop'):s.restore(bad)
    s.reset(torch.tensor([i]))
    assert not h.seen[i] and not h.active[i] and h.anchor[i].abs().sum()==0


def test_config_legacy_and_scope(source):
    c=default_settings();validate_config(c)
    assert 'm20_cts_action_latent_v19' in LEGACY_VERSIONS
    c['all_terrain'].pop('wall_hold_practice');c['all_terrain'].pop('stair_stop');c['all_terrain'].pop('stair_hold_practice');validate_config(c,legacy=True)
    for k,v in [('backward_m',float('nan')),('braking_m_s2',0),('tread_fraction',2),('enabled',1)]:
        bad=deepcopy(default_settings()['all_terrain']['stair_stop']);bad[k]=v
        with pytest.raises(ValueError):validate(bad)
    env,s,i=setup(source);frame(env,s,.1,'zero')
    assert not s.recovery.hold.active.any()  # initial hold unaffected
    enter(env,s,i);frame(env,s,3.02,'zero')
    assert not s.recovery.hold.active[s.course_ids != 0].any()


def test_terminal_low_body_frame_still_records_retreat_without_stop_credit(source):
    env,s,i=setup(source);h=s.recovery.hold;enter(env,s,i)
    frame(env,s,3.02,'zero')
    d=env.scene.robot.data
    d.root_pos_w[i,0]-=.2;d.root_pos_w[i,2]=-1.
    support(env,s);frame(env,s,3.04,'zero')
    assert s.failed[i] and not h.ready[i] and not h.active[i]
    assert h.max_back[i] == pytest.approx(.2,abs=1e-5)
    assert h.totals['outside_frames'][1,0] == 1


def test_resume_success_ledger_and_world_velocity_are_not_body_velocity(source):
    env,s,i=setup(source);h=s.recovery.hold;enter(env,s,i)
    d=env.scene.robot.data
    d.root_lin_vel_w=d.root_lin_vel_b.clone();d.root_lin_vel_w[i,0]=-.2
    frame(env,s,3.02,'zero');assert not h.ready[i]
    d.root_lin_vel_w.zero_();settle(env,s)
    assert h.resumed[i]
    s.success[i]=True;s.failed[i]=False;env.termination_manager.terminated[i]=True
    s.record(torch.tensor([i]));s.record(torch.tensor([i]))
    saved=s.snapshot();s.restore(saved)
    assert h.totals['completed_resumed'][1,0] == h.totals['success_after_resume'][1,0] == 1
