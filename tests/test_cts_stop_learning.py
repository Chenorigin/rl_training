"""v22 regression: on-policy noise, strict hold scoring, frame counts and rolling retreat."""
from copy import deepcopy
import math
import pytest
import torch
from task_support.cts.algorithm import CTSAlgorithm
from task_support.cts.configuration import default_settings, validate_config
from task_support.cts.timing import required_steps, advance_steps
from test_cts_algorithm import fixture, collect
from test_cts_all_terrain_course import source, frame, support
from test_cts_stair_stop import setup as stairs, enter
from test_cts_motion_quality import setup as platform, step
from task_support.cts.all_terrain_terms import reward


def test_noise_uses_raw_zero_only_and_does_not_change_export():
    models, legacy, obs, *_ = fixture()
    obs['motion_control'].zero_();obs['motion_control'][2,1]=.1;obs['motion_control'][3,2]=.1
    obs['policy'][:,8]=10.;obs['teacher'][:,8]=-10.
    cfg={**legacy.cfg,'stop_exploration_scale':.2}
    alg=CTSAlgorithm(models,cfg,8,4)
    before=models.student_export()(obs['policy'],obs['motion_control'])
    sd=alg.action_std(obs);base=models.log_std.exp()
    torch.testing.assert_close(sd[[0,1,4,5,6,7]],base.expand(6,-1)*.2)
    torch.testing.assert_close(sd[[2,3]],base.expand(2,-1))
    torch.testing.assert_close(legacy.action_std(obs),base.expand(8,-1))
    torch.testing.assert_close(before,models.student_export()(obs['policy'],obs['motion_control']),rtol=0,atol=0)
    torch.manual_seed(9);action=alg.act(obs)
    dist=torch.distributions.Normal(alg.pending['mean'],sd)
    torch.testing.assert_close(alg.pending['log_prob'],dist.log_prob(action).sum(-1),rtol=0,atol=0)
    wrong=torch.distributions.Normal(alg.pending['mean'],base).log_prob(action).sum(-1)
    assert (wrong-alg.pending['log_prob']).abs().max()>1


def test_stored_noise_replays_ppo_and_roundtrips_state():
    models, legacy, obs, *_=fixture();obs['motion_control'][:4]=0.
    alg=CTSAlgorithm(models,{**legacy.cfg,'stop_exploration_scale':.2},8,4)
    collect(alg,obs);data=alg._data()
    std=alg.action_std(data['obs']);assert torch.equal(std,data['std'])
    mean,_=models.mean(data['obs'],data['teacher'])
    logp=torch.distributions.Normal(mean,std).log_prob(data['actions']).sum(-1)
    torch.testing.assert_close(logp,data['log_prob'],atol=1e-5,rtol=1e-5)
    metrics=alg.ppo_update();assert metrics['ppo_minibatches']>0
    for key in ('ppo_stop_kl_max','ppo_move_kl_max','ppo_kl_early_stop'):assert math.isfinite(metrics[key])
    # load requires a rollout boundary, exactly like the production runner.
    alg.rollout.clear();alg.returns=alg.advantages=None
    saved=deepcopy(alg.save());alg.load(saved)
    assert alg.cfg['stop_exploration_scale']==.2


@pytest.mark.parametrize('value',[0.,-1.,1.01,float('nan'),float('inf'),True])
def test_invalid_exploration_is_rejected(value):
    c=default_settings();c['algorithm']['stop_exploration_scale']=value
    with pytest.raises(ValueError):validate_config(c)


def test_legacy_schema_keeps_old_noise_and_idle_scoring():
    c=default_settings();c['algorithm'].pop('stop_exploration_scale')
    c['all_terrain']['motion_quality'].pop('idle_terminal_hold');c['all_terrain']['stair_stop'].pop('retreat_weight')
    validate_config(c,legacy=True)
    with pytest.raises(ValueError):validate_config(c)


@pytest.mark.parametrize('duration,expected',[(.2,10),(.201,11),(.4,20),(.12,6),(.06,.06/.02),(3.,150)])
def test_exact_completed_steps(duration,expected):
    assert required_steps(duration,.02)==expected
    n=torch.zeros(1,dtype=torch.long)
    for _ in range(int(expected)-1):n=advance_steps(n,torch.tensor([True]))
    assert not (n>=required_steps(duration,.02)).item()
    n=advance_steps(n,torch.tensor([True]));assert (n>=required_steps(duration,.02)).item()
    n=advance_steps(n,torch.tensor([False]));assert n.item()==0


def test_platform_idle_histories_are_diagnostics_but_drift_still_fails(source):
    env,s,i=platform(source,9,8)
    for k in range(5):step(env,s,.5+k*.02,i=i,noise=.6)
    start=float(s.move_duration[i])+s.cfg['commands']['initial_hold_s']
    for k in range(10):
        step(env,s,start+k*.02,i=i,noise=.6)
        if k<9:assert not s.success[i]
    assert s.success[i] and s.stop_ok[i]
    assert s.stable_steps[i]==10 and not s.objectives.yaw_ok[i]
    # A fresh rollout with the same historical yaw but 9cm drift is still rejected.
    env,s,i=platform(source,9,8)
    for k in range(20):step(env,s,5.+k*.02,i=i,dx=.09 if k==0 else 0.,noise=.6)
    assert not s.success[i] and not s.quality.hold_ok[i]
    env.termination_manager.time_outs[i]=True;s.record(torch.tensor([i]))
    cell=(int(not s.teacher[i]),9,8)
    assert s.failure_counts['hold_quality'][cell]==1
    assert s.failure_counts['yaw_tracking'][cell]==0 and s.failure_counts['linear_tracking'][cell]==0
    assert s.motion_totals['yaw_rms'][cell]>0 if 'yaw_rms' in s.motion_totals else s.tracking_error[i,1]>0


def test_legacy_idle_still_uses_yaw_history(source):
    env,s,i=platform(source,9,8);s.quality.cfg.pop('idle_terminal_hold')
    for k in range(5):step(env,s,.5+k*.02,i=i,noise=.6)
    start=float(s.move_duration[i])+s.cfg['commands']['initial_hold_s']
    for k in range(20):step(env,s,start+k*.02,i=i,noise=.6)
    assert s.quality.hold_ok[i] and s.stop_ok[i] and not s.success[i]


def test_stair_resume_exactly_twenty_zero_frames_and_reset(source):
    env,s,i=stairs(source);enter(env,s,i)
    for k in range(19):frame(env,s,3.02+k*.02,'zero')
    assert s.recovery.phase[i]==2 and s.recovery.settled_steps[i]==19
    frame(env,s,3.4,'zero');assert s.recovery.phase[i]==3
    s.reset(torch.tensor([i]));assert s.recovery.settled_steps[i]==s.recovery.pause_steps[i]==0


def test_stair_timeout_exactly_150_unstable_frames(source):
    env,s,i=stairs(source);enter(env,s,i)
    env.scene.robot.data.root_pos_w[i,0]-=.3;support(env,s)
    for k in range(149):frame(env,s,3.02+k*.02,'zero')
    assert not s.failed[i] and s.recovery.pause_steps[i]==149
    frame(env,s,6.,'zero');assert s.failed[i] and s.recovery.events['pause_failure'][i]==1


@pytest.mark.parametrize('body_moves',[False,True])
def test_early_pure_rolling_retreat_is_penalized_inside_position_tolerance(source,body_moves):
    values=[]
    for weight in (0.,.5):
        torch.manual_seed(314)
        env,s,i=stairs(source);h=s.recovery.hold;h.cfg['retreat_weight']=weight
        enter(env,s,i);frame(env,s,3.02,'zero');d=env.scene.robot.data
        if body_moves:d.root_lin_vel_w[i,0]=-.1;d.root_pos_w[i,0]-=.002
        # Whole footprint moves back without contact-material slip; root can remain still.
        d.body_link_pos_w[i,:,0]-=.002;d.body_com_pos_w[i,:,0]-=.002
        footprint_vx=(d.body_link_pos_w[i,:,:2].mean(0)[0]-h.previous_feet[i,0])/env.step_dt
        expected=.5*(max(.1 if body_moves else 0.,float(-footprint_vx))/.1)**2
        frame(env,s,3.04)
        assert h.active[i] and h.region[i] and s.observe()['slip'][i].max()==0
        values.append(float(reward(env)[i]))
    assert values[0]-values[1]==pytest.approx(expected,abs=1e-5)
