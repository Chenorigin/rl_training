"""Real course/reward counterexamples for v19; no optimizer or physics claims."""
from copy import deepcopy
from types import SimpleNamespace
import math
import pytest
import torch
from task_support.cts.configuration import default_settings, validate_config, VERSION, LEGACY_VERSIONS
from task_support.cts.objective_consistency import validate
from test_cts_all_terrain_course import source, fake, support, frame


def setup(source):
    cfg,atlas=source;cfg=deepcopy(cfg)
    for key in ('motion_stability','startup_stability','compound_commands','objective_consistency'):
        cfg['all_terrain'][key]=default_settings()['all_terrain'][key]
    env,s=fake((cfg,atlas),n=100)
    env.scene.robot.data.root_lin_vel_w=torch.zeros(100,3)
    return env,s


def platform(source, high=True):
    env,s=setup(source)
    i=int((s.course_ids==9).nonzero()[0])
    tiles=[j for j,r in enumerate(s.atlas.records) if r['course_id']==9 and ((r['height']>.4) if high else (r['height']<.12))]
    s.tile[i]=tiles[0];s.mode[i]=0;s.command_value[i]=torch.tensor([.5,0.,0.])
    d=env.scene.robot.data;d.root_quat_w[i]=torch.tensor([1.,0.,0.,0.])
    d.root_pos_w[i]=s.origins[s.tile[i]]+torch.tensor([float(s.ends[s.tile[i]])-.45,0.,.5])
    support(env,s)
    return env,s,i


def test_pause_and_initial_hold_progress_not_paid_or_repaid(source):
    env,s=setup(source);i=int((s.course_ids==0).nonzero()[0]);d=env.scene.robot.data
    start=d.root_pos_w[i,0].clone();s.command_value[i]=0
    d.root_pos_w[i,0]+=.1;support(env,s);frame(env,s,1.,'zero')
    assert s.progress[i]==0
    assert s.best_x[i]==start+.1
    s.command_value[i,0]=.5;frame(env,s,1.02,'command');assert s.progress[i]==0
    d.root_pos_w[i,0]+=.01;support(env,s);frame(env,s,1.04,'command')
    assert float(s.progress[i])==pytest.approx(.01,abs=1e-5)


def test_actor_yaw_cache_precedes_tags_and_reset_invalidates(source):
    env,s=setup(source);o=s.objectives;ids=(s.course_ids==9).nonzero().flatten()
    env.episode_length_buf.fill_(50);s.observe()  # stale cached geometry exists
    teacher=torch.zeros(100,244);policy=torch.zeros(100,789)
    teacher[:,8]=.16;policy[:,8]=.08
    actual=torch.where(s.teacher,.16,.08)
    env.scene.robot.data.root_ang_vel_b[:,2]=actual
    o.begin_step({'teacher':teacher,'policy':policy});s.tags();g=s.observe()
    assert (g['yaw_error'][ids]==0).all()
    assert (g['command'][ids,2]==0).all()
    torch.testing.assert_close(g['tracking_command'][:,2],actual)
    env.common_step_counter+=1;s.advance()
    assert (s.tracking_error[ids,1]==0).all()
    assert (s.axis_tracking_error[ids,2]==0).all()
    s.reset(ids);assert not o.command_valid[ids].any()


def test_high_platform_brakes_then_requires_executed_zero_supported_stop(source):
    env,s,i=platform(source);o=s.objectives
    frame(env,s,1.,'zero')
    assert o.platform_stop[i] and s.raw_command()[i].abs().sum()==0
    assert s.stable_age[i]==0 and not s.success[i]  # trigger step executed +.5
    frames=s.tracking_frames[i].clone()
    for j in range(12):frame(env,s,1.02+j*.02,'zero')
    assert s.success[i] and not s.failed[i]
    assert s.tracking_frames[i]==frames  # early braking excluded from moving RMS
    assert o.goal_ok[i]
    s.reset(torch.tensor([i]));assert not o.platform_stop[i]


@pytest.mark.parametrize('bad',['side','overhang','floor','retreat','unsupported'])
def test_high_platform_rejects_false_completion(source,bad):
    env,s,i=platform(source);o=s.objectives;d=env.scene.robot.data
    if bad=='side':
        d.root_pos_w[i,0]-=1.;d.root_pos_w[i,1]+=float(s.widths[s.tile[i]])/2-.3
        support(env,s);frame(env,s,1.,'zero');assert not o.platform_stop[i]
    else:
        frame(env,s,1.,'zero');assert o.platform_stop[i]
        if bad=='overhang':d.root_pos_w[i,0]+=.15
        if bad=='floor':d.root_pos_w[i,0]+=1.;d.root_pos_w[i,2]-=s.heights[s.tile[i]]
        if bad=='retreat':d.root_pos_w[i,0]-=1.
        support(env,s)
        if bad=='unsupported':env.scene.sensors['contact_forces'].data.net_forces_w[i,0]=0
    for j in range(15):frame(env,s,1.02+j*.02,'zero')
    assert not s.success[i]
    assert not o.goal_ok[i]


def test_low_platform_keeps_forward_goal_and_cannot_win_by_standing(source):
    env,s,i=platform(source,False)
    frame(env,s,1.,'zero')
    assert not s.objectives.platform_stop[i] and s.raw_command()[i,0]>.1
    for j in range(15):frame(env,s,5.5+j*.02,'zero')
    assert not s.success[i] and not s.objectives.goal_ok[i]


def test_footprint_prediction_uses_world_x_not_body_speed(source):
    env,s,i=platform(source);d=env.scene.robot.data
    # At margin .16, body velocity +2 toward world -X must not predict a +X advance.
    d.root_quat_w[i]=torch.tensor([0.,0.,0.,1.]);d.root_lin_vel_b[i,0]=2.
    d.root_lin_vel_w[i,0]=-2.;frame(env,s,1.)
    assert not s.objectives.platform_stop[i]


def test_version_optional_legacy_and_schema():
    assert VERSION.endswith('_v26') and 'm20_cts_action_latent_v18' in LEGACY_VERSIONS
    cfg=default_settings();validate_config(cfg)
    cfg['all_terrain'].pop('stair_stop', None)
    cfg['all_terrain'].pop('wall_hold_practice', None)
    cfg['all_terrain'].pop('stair_hold_practice', None)
    cfg['all_terrain'].pop('motion_quality', None)
    cfg['all_terrain'].pop('objective_consistency');validate_config(cfg,legacy=True)
    for c in ({},{'enabled':1},{'enabled':True,'ignored':True}):
        with pytest.raises(ValueError):validate(c)


def test_snapshot_counters_roundtrip_and_episode_state_reset(source):
    env,s,i=platform(source);frame(env,s,1.,'zero')
    saved=s.snapshot();assert 'goal' in saved['motion_stability']['failure_counts']
    saved['motion_stability']['failure_counts']['goal'][0,9,0]=7
    s.restore(saved);s.reset(torch.arange(100))
    assert s.failure_counts['goal'][0,9,0]==7
    assert not s.objectives.platform_stop.any() and not s.objectives.command_valid.any()


def reward_env(env):
    n=env.num_envs;robot=env.scene.robot;d=robot.data
    d.joint_pos=torch.zeros(n,16);d.default_joint_pos=torch.zeros(n,16)
    d.joint_vel=torch.zeros(n,16);d.applied_torque=torch.zeros(n,16)
    d.soft_joint_pos_limits=torch.tensor([-1.,1.]).expand(n,16,2).clone()
    robot.find_joints=lambda names,preserve_order: (names,names)
    env.cfg.leg_joint_names=list(range(12));env.cfg.wheel_joint_names=list(range(12,16))
    env.action_manager=SimpleNamespace(action=torch.zeros(n,16),prev_action=torch.zeros(n,16))


def test_actual_pause_reward_no_longer_prefers_small_forward_drift(source):
    from task_support.cts.all_terrain_terms import reward
    values={}
    for legacy in (False,True):
        env,s=setup(source);reward_env(env);i=int((s.course_ids==0).nonzero()[0])
        if legacy:s.objectives=None
        s.command_value[i]=0
        d=env.scene.robot.data;d.root_pos_w[i,0]+=.0016;d.root_lin_vel_b[i,0]=.08
        support(env,s);frame(env,s,1.)
        drifting=reward(env)[i].item()
        s.progress[i]=0;d.root_lin_vel_b[i,0]=0;s.measure_tick=-1
        stationary=reward(env)[i].item()
        values[legacy]=drifting-stationary
    assert values[True]>0  # known bad v18 counterexample must be red
    assert values[False]<0


def test_actual_high_stop_reward_uses_zero_target_not_max_of_competing_targets(source):
    from task_support.cts.all_terrain_terms import reward
    env,s,i=platform(source);reward_env(env)
    frame(env,s,1.,'zero');frame(env,s,1.02,'zero')
    stopped=reward(env)[i].item()
    env.scene.robot.data.root_lin_vel_b[i,0]=.5;s.measure_tick=-1
    forward=reward(env)[i].item()
    assert stopped>forward
    assert s.observe()['command'][i].abs().sum()==0


def test_failure_metrics_separate_goal_linear_and_actor_yaw(source):
    env,s,i=platform(source,False)
    # Stand still on a low slab: spatial goal and linear tracking fail, yaw passes.
    frame(env,s,1.,'zero');frame(env,s,6.42,'zero')
    env.termination_manager.time_outs[i]=True;s.record(torch.tensor([i]))
    cell=int(not s.teacher[i]),9,0
    assert s.failure_counts['goal'][cell]==1
    assert s.failure_counts['linear_tracking'][cell]==1
    assert s.failure_counts['yaw_tracking'][cell]==0
    assert s.failure_counts['tracking'][cell]==1


def test_high_stop_requires_continuous_safe_support_not_one_last_touch(source):
    env,s,i=platform(source);frame(env,s,1.,'zero')
    env.scene.sensors['contact_forces'].data.net_forces_w[i,0]=0
    for j in range(12):frame(env,s,1.02+j*.02,'zero')
    assert s.stable_age[i]==0 and not s.success[i]
    support(env,s);frame(env,s,1.28,'zero')
    assert not s.success[i] and s.stable_age[i]<s.cfg['commands']['success_hold_s']
    for j in range(12):frame(env,s,1.30+j*.02,'zero')
    assert s.success[i]


def test_compound_reward_cannot_prefer_ignoring_actor_heading_feedback():
    from task_support.cts.compound_commands import tracking_rewards
    state=SimpleNamespace(cfg=default_settings()['all_terrain'],course_ids=torch.tensor([3]),objectives=object())
    raw=torch.tensor([[.5,0.,0.]]);seen=torch.tensor([[.5,0.,.16]])
    g=dict(command=raw,tracking_command=seen,measured_velocity=seen.clone())
    good=tracking_rewards(state,g,torch.ones(1),torch.ones(1))
    g['measured_velocity']=raw.clone()
    bad=tracking_rewards(state,g,torch.ones(1),torch.ones(1))
    assert good[0]==good[1]==1 and bad[0]<good[0] and bad[1]<good[1]
    state.objectives=None
    legacy=tracking_rewards(state,g,torch.ones(1),torch.ones(1))
    assert legacy[0]==legacy[1]==1


@pytest.mark.parametrize('field,value,match', [('map_capture_padding_m', 99., 'capture'), ('continuous_delay', False, 'continuous')])
def test_v18_migration_still_rejects_broken_perception(field,value,match):
    from test_cts_entry import CTSMetadataTests
    from task_support.cts.entry import validate_metadata
    meta=CTSMetadataTests().meta();meta['version']='m20_cts_action_latent_v18'
    meta['config']['all_terrain'].pop('stair_stop', None)
    meta['config']['all_terrain'].pop('wall_hold_practice', None)
    meta['config']['all_terrain'].pop('stair_hold_practice', None)
    meta['config']['all_terrain'].pop('motion_quality', None)
    meta['config']['all_terrain'].pop('objective_consistency')
    validate_metadata(meta,resume=False)
    meta['perception'][field]=value;meta['environment']['perception'][field]=value
    with pytest.raises(ValueError,match=match):validate_metadata(meta,resume=False)
