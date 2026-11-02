"""Executed-frame, reward and success-gate counterexamples for v21."""
from copy import deepcopy
import math
import pytest
import torch
from task_support.cts.configuration import default_settings,validate_config
from task_support.cts.motion_quality import validate
from task_support.cts.all_terrain_terms import reward
from test_cts_all_terrain_course import source,fake,support
from test_cts_objective_consistency import reward_env


def setup(source,course=8,mode=0,high=False):
    c,a=source;c=deepcopy(c)
    for k in ('motion_stability','startup_stability','hind_clearance','objective_consistency','motion_quality'):
        c['all_terrain'][k]=default_settings()['all_terrain'][k]
    env,s=fake((c,a),n=100);d=env.scene.robot.data
    d.root_lin_vel_w=torch.zeros(100,3)
    i=int((s.course_ids==course).nonzero()[0]);s.mode[i]=mode
    s.command_value[i]=torch.tensor([.5 if mode==0 else 0.,0.,0.])
    if course==9:
        tiles=[j for j,r in enumerate(a.records) if r['course_id']==9 and (r['height']>.4 if high else r['height']<.12)]
        s.tile[i]=tiles[0]
    d.root_pos_w[i]=s.origins[s.tile[i]]+torch.tensor([0.,0.,.5])
    d.root_quat_w[i]=torch.tensor([1.,0.,0.,0.]);support(env,s);reward_env(env)
    return env,s,i


def step(env,s,age,*,angle=None,i=None,dx=0.,speed=None,noise=.0):
    d=env.scene.robot.data
    env.episode_length_buf.fill_(round(age/env.step_dt)-1)
    s.measure_tick=-1;s.observe();s.quality.begin_step()
    obs={'teacher':torch.zeros(env.num_envs,244),'policy':torch.zeros(env.num_envs,789)}
    for v in obs.values():v[:,8]=noise
    s.objectives.begin_step(obs)
    env.common_step_counter+=1;env.episode_length_buf+=1
    if i is not None:
        if angle is not None:d.root_quat_w[i]=torch.tensor([math.cos(angle/2),0.,0.,math.sin(angle/2)])
        d.root_pos_w[i,0]+=dx
        if speed is not None:d.root_lin_vel_w[i,0]=speed;d.root_lin_vel_b[i,0]=speed
    support(env,s);s.advance()


def test_noise_in_actor_yaw_cannot_reject_exact_world_heading(source):
    env,s,i=setup(source)
    for k in range(30):step(env,s,.5+k*.02,i=i,speed=.5,noise=.6)
    assert s.objectives.yaw_ok[i]==False  # original diagnostic preserved
    assert s.quality.heading_ok[i] and s.tracking_ok[i]
    assert float(s.tracking_error[i,1])>0


@pytest.mark.parametrize('pattern',['constant','oscillate','spike'])
def test_drift_and_zero_mean_oscillation_fail_heading(source,pattern):
    env,s,i=setup(source)
    for k in range(40):
        angle=.3 if pattern=='constant' else (.2*(-1)**k if pattern=='oscillate' else (.35 if k==10 else 0.))
        step(env,s,.5+k*.02,i=i,angle=angle,speed=.5)
    assert not s.quality.heading_ok[i] and not s.tracking_ok[i]


def test_wrap_and_pitch_do_not_create_false_yaw(source):
    env,s,i=setup(source);d=env.scene.robot.data
    d.root_quat_w[i]=torch.tensor([math.cos(3.14/2),0,0,math.sin(3.14/2)])
    step(env,s,.5,i=i,angle=-3.14,speed=.5)
    assert s.quality.heading_ok[i]
    # Full quaternion pitch with unchanged yaw must not be yaw rate.
    env,s,i=setup(source);d=env.scene.robot.data
    s.observe();env.episode_length_buf.fill_(24);s.quality.begin_step()
    d.root_quat_w[i]=torch.tensor([math.cos(.2),0.,math.sin(.2),0.])
    env.common_step_counter+=1;env.episode_length_buf+=1;support(env,s);s.advance()
    assert s.quality.heading_ok[i] and s.quality.filtered_rate[i]==0


def test_hold_first_step_drift_cannot_reanchor_and_restart_releases(source):
    env,s,i=setup(source,9,8)
    step(env,s,.5,i=i,dx=.2,speed=0.)
    anchor=s.quality.anchor[i].clone()
    for k in range(15):step(env,s,.52+k*.02,i=i)
    assert not s.quality.hold_ok[i] and not s.success[i]
    torch.testing.assert_close(anchor,s.quality.anchor[i])
    assert s.quality.totals['hold_drift_max_m'].max()>=.19
    s.mode[i]=0;s.command_value[i,0]=.5
    step(env,s,1.,i=i,speed=.5)
    assert not s.quality.holding[i]


def test_micro_motion_gets_less_reward_than_anchored_stop(source):
    values=[]
    for v in (0.,.036):
        env,s,i=setup(source,9,8)
        step(env,s,.5,i=i,speed=v)
        values.append(float(reward(env)[i]))
        assert bool(s.quality.hold_ok[i])==(v==0)
    assert values[0]-values[1]>2.  # previously nearly full reward for both


def test_hold_requires_front_and_rear_support(source):
    env,s,i=setup(source,9,8)
    step(env,s,.5,i=i)
    assert s.quality.hold_ok[i]
    env.common_step_counter+=1;s.quality.begin_tick=env.common_step_counter-1
    env.scene.sensors['contact_forces'].data.net_forces_w[i,2:]=0;s.measure_tick=-1;s.advance()
    assert not s.quality.hold_ok[i]


def test_braking_allowance_bounded_and_fixed(source):
    env,s,i=setup(source,9,8);d=env.scene.robot.data
    d.root_lin_vel_w[i,0]=.5
    step(env,s,.5,i=i,dx=.1,speed=0.)
    assert s.quality.hold_ok[i]
    assert float(s.quality.braking[i])==pytest.approx(.125)
    step(env,s,.52,i=i,dx=.2,speed=0.)
    assert not s.quality.hold_ok[i]
    assert float(s.quality.braking[i])==pytest.approx(.125)


def test_transition_covers_rear_wheels_until_supported_landing(source):
    env,s,i=setup(source,9);d=env.scene.robot.data
    edge=float(s.ends[s.tile[i]]);d.root_pos_w[i,0]=s.origins[s.tile[i],0]+edge+.1
    step(env,s,.5,i=i,speed=.5)
    assert s.quality.transition[i] and not s.quality.landed[i]
    d.root_pos_w[i,0]+=1.;d.root_pos_w[i,2]=.5
    for k in range(8):step(env,s,.52+k*.02,i=i,speed=.5)
    assert s.quality.landed[i] and not s.quality.transition[i]
    d.root_pos_w[i,0]-=1.;d.root_pos_w[i,2]=s.origins[s.tile[i],2]+.5
    step(env,s,.8,i=i,speed=.5)
    assert not s.quality.landed[i] and s.quality.transition[i]


def test_new_hold_cannot_override_high_platform_spatial_gate(source):
    env,s,i=setup(source,9,high=True);s.command_value[i]=0
    for k in range(20):step(env,s,5.5+k*.02,i=i)
    assert s.quality.hold_ok[i]
    assert not s.objectives.goal_ok[i] and not s.success[i]


def test_counts_once_snapshot_reset_and_nonfinite_fail_closed(source):
    env,s,i=setup(source,9,8);step(env,s,.5,i=i)
    saved=s.snapshot();s.advance();assert torch.equal(saved['motion_quality']['hold_frames'],s.snapshot()['motion_quality']['hold_frames'])
    s.restore(saved);s.reset(torch.tensor([i]));assert not s.quality.seen_hold[i]
    s.mode[i]=8;s.command_value[i]=0
    step(env,s,.5,i=i,dx=float('nan'))
    assert s.failed[i] and not s.quality.hold_ok[i]
    assert torch.isfinite(reward(env)).all()


def test_schema_optional_and_disabled_preserve_legacy(source):
    c=default_settings();validate_config(c)
    c['all_terrain'].pop('motion_quality');validate_config(c,legacy=True)
    for bad in ({},{'enabled':True},dict(default_settings()['all_terrain']['motion_quality'],rate_filter_s=0)):
        with pytest.raises(ValueError):validate(bad)
    c,a=source;c=deepcopy(c);c['all_terrain']['motion_quality']=dict(default_settings()['all_terrain']['motion_quality'],enabled=False)
    env,s=fake((c,a),n=100)
    assert s.quality is None and 'motion_quality' not in s.snapshot()


def test_failure_counts_survive_later_physics_and_restore(source):
    env,s,i=setup(source)
    step(env,s,.5,i=i,angle=.3,speed=.5)
    env.termination_manager.time_outs[i]=True;s.record(torch.tensor([i]))
    cell=int(not s.teacher[i]),8,0
    assert s.failure_counts['heading'][cell]==1
    assert s.failure_counts['yaw_tracking'][cell]==0
    step(env,s,.52,i=i)
    assert s.failure_counts['heading'][cell]==1
    snap=s.snapshot();s.restore(snap)
    assert s.failure_counts['heading'][cell]==1


def test_last_forward_step_cannot_count_as_platform_stop(source):
    env,s,i=setup(source,9);d=env.scene.robot.data
    d.root_pos_w[i,0]=s.origins[s.tile[i],0]+s.ends[s.tile[i]]+1.
    d.root_pos_w[i,2]=.5
    step(env,s,5.4,i=i,speed=0.)
    assert s.quality.moving[i] and not s.quality.holding[i]
    assert s.stable_age[i]==0
    step(env,s,5.42,i=i,speed=0.)
    if not s.quality.holding[i]:
        assert s.stable_age[i]==0
        step(env,s,5.44,i=i,speed=0.)
    assert s.quality.holding[i] and s.stable_age[i] <= env.step_dt+1.e-6
