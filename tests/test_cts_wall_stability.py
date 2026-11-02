"""Wall-quality regression contracts; bounded CPU gradients, no training run."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch
import pytest
import torch
from task_support.cts.configuration import default_settings
from task_support.cts import wall_stability as wall
from task_support.cts.model import SharedDecoder, build_models, transfer_state
from task_support.cts.algorithm import CTSAlgorithm
from test_cts_algorithm import fixture
from test_cts_clutter_consistency import fixture as data_fixture
from test_cts_startup_stability import fake


def config():
    c=default_settings()['all_terrain']['wall_stability'];c['probability']=1.
    return c


def data():
    alg,d,_=data_fixture();c=config()
    d['wall_quality']=torch.ones(32,dtype=torch.bool);d['wall_idle']=torch.ones(32,dtype=torch.bool)
    d['obs']['motion_control'].zero_()
    return alg,d,c


def test_quality_drop_is_coherent_and_preserves_unknown_centre_and_history():
    _,d,c=data();p=d['obs']['policy'];p[:,57:244]=.3
    p[:,244:249]=0.;p[:,431:436]=0.
    original=p.clone();kinds=torch.arange(32)%3
    q,active,_=wall.perturb(p,c,kinds=kinds)
    assert active.all()
    from task_support.cts.clutter_consistency import grid
    _,y=grid(p);centre=y.abs()<=c['protected_half_width_m']
    torch.testing.assert_close(q[:,57:244][:,centre],p[:,57:244][:,centre],rtol=0,atol=0)
    torch.testing.assert_close(q[:,:57],p[:,:57],rtol=0,atol=0)
    torch.testing.assert_close(q[:,618:],p[:,618:],rtol=0,atol=0)
    torch.testing.assert_close(q[:,57:62],p[:,57:62],rtol=0,atol=0)
    dropped=(q[:,244:431]!=p[:,244:431])
    assert dropped.any() and (q[:,57:244][dropped]==0).all() and (q[:,431:618][dropped]==0).all()
    torch.testing.assert_close(q[kinds==2,57:431],p[kinds==2,57:431])
    assert (q[kinds==2,431:618] < p[kinds==2,431:618]).any()
    torch.testing.assert_close(p,original,rtol=0,atol=0)
    p[:,244:431]=0.
    q,active,_=wall.perturb(p,c);assert not active.any();torch.testing.assert_close(q,p)


@pytest.mark.parametrize('bad',['step','pit','blackout','unsupported','moving','nan'])
def test_hard_idle_supervision_rejects_unsafe_or_moving_context(bad):
    _,d,c=data();ids=torch.arange(32)
    assert wall.eligible(d,ids,c,'wall_idle').all()
    if bad in ('step','pit'):d['obs']['teacher'][:,150]=.2 if bad=='step' else -.2
    if bad=='blackout':d['obs']['policy'][:,244:431]=0.
    if bad=='unsupported':d['wall_idle'].zero_()
    if bad=='moving':d['obs']['motion_control'][:,0]=.01
    if bad=='nan':d['obs']['policy'][:,58]=float('nan')
    assert not wall.eligible(d,ids,c,'wall_idle').any()


def physical():
    env,s,g=fake(8);s.cfg['wall_stability']=config();s.env=env
    s.stairs.fill_(True);s.course_ids.zero_();s.failed=torch.zeros(8,dtype=torch.bool)
    env.scene['robot'].data.root_quat_w=torch.tensor([1.,0.,0.,0.]).repeat(8,1)
    g.update(height=torch.zeros(8,4),normal=torch.tensor([0.,0.,1.]).repeat(8,4,1),
        supported=torch.ones(8,4,dtype=torch.bool),measured_velocity=torch.zeros(8,3))
    return env,s,g


def test_physical_tags_and_measured_rate_cost_do_not_reward_runaway():
    from task_support.cts.stability import stability_costs
    env,s,g=physical();raw=g['command']
    g['height'][1,0]=.15;g['normal'][2,:,2]=.9;g['supported'][3,:2]=False
    env.scene['robot'].data.root_quat_w[4]=torch.tensor([.9,.43589,0.,0.])
    s.failed[5]=True;g['measured_velocity'][6,0]=1.;raw[7,0]=.3
    tags=wall.collection_tags(s,g,raw)
    assert tags['cts_wall_idle'].tolist()==[True,False,False,False,False,False,False,False]
    assert tags['cts_wall_quality'].tolist()==[True,False,False,False,False,False,True,True]
    env.scene['robot'].data.joint_vel.fill_(10.)
    costs=stability_costs(env,s,g)
    assert costs['idle_wheel_speed'][0]>0 and costs['idle_wheel_speed'][6]>0
    assert costs['idle_wheel_speed'][7]==0 and costs['idle_wheel_speed'][1:6].sum()==0


def test_gain_identity_and_attenuation_never_changes_moving_commands():
    decoder=SharedDecoder([8], [.2]*16,.25,True)
    p=torch.randn(8,57);latent=torch.randn(8,16)*100;raw=torch.zeros(8,3);raw[4:,0]=.001
    torch.testing.assert_close(decoder(p,latent,raw),latent,rtol=0,atol=0)
    with torch.no_grad():decoder.idle_wheel_mlp[-1].bias.fill_(-6.)
    out=decoder(p,latent,raw)
    assert out[:4,12:].abs().max()<latent[:4,12:].abs().max()*.01
    torch.testing.assert_close(out[4:],latent[4:],rtol=0,atol=0)
    assert decoder(p,torch.zeros_like(latent),raw).count_nonzero()==0
    exported=torch.jit.script(decoder)
    torch.testing.assert_close(exported(p,latent,raw),out,rtol=0,atol=0)


def new_algorithm():
    _,old,obs,_,_,_,t,s=fixture()
    cfg={'actor':{'idle_leg_scale':.25,'new_domain_anchor_scale':.1},'wall_stability':config()}
    models=build_models(t,s,old.cfg,all_terrain_cfg=cfg)
    alg=CTSAlgorithm(models,old.cfg,8,4,all_terrain_cfg=cfg)
    obs['policy'][:,57:244]=0.;obs['policy'][:,244:618]=1.;obs['teacher'][:,57:244]=0.;obs['motion_control'].zero_()
    return alg,obs,t,s,cfg


def test_v17_migration_identity_and_partial_or_trained_new_head_rejected():
    alg,obs,t,s,cfg=new_algorithm();oldcfg=deepcopy(cfg);oldcfg.pop('wall_stability')
    old=build_models(t,s,alg.cfg,all_terrain_cfg=oldcfg)
    source=deepcopy(old.state_dict());merged=transfer_state(alg.models,source)
    alg.models.load_state_dict(merged,strict=True)
    for fn in ('student_action','teacher_action'):
        if fn=='student_action':args=(obs['policy'],obs['motion_control'])
        else:args=(obs,)
        torch.testing.assert_close(getattr(alg.models,fn)(*args),getattr(old,fn)(*args),rtol=0,atol=0)
    partial=deepcopy(merged);partial.pop('decoder.idle_wheel_mlp.0.weight')
    with pytest.raises(ValueError,match='complete'):transfer_state(alg.models,partial)
    with torch.no_grad():alg.models.decoder.idle_wheel_mlp[-1].bias.fill_(.1)
    with pytest.raises(ValueError,match='zero-output'):transfer_state(alg.models,source)


def collect(alg,obs):
    for k in range(4):
        alg.act(obs)
        alg.process_env_step(obs,torch.arange(8).float()*.1,torch.zeros(8,dtype=torch.bool),
          {'cts_anchor':torch.zeros(8,dtype=torch.bool),'cts_course':torch.zeros(8,dtype=torch.long),
           'cts_wall_quality':torch.ones(8,dtype=torch.bool),'cts_wall_idle':torch.ones(8,dtype=torch.bool),
           'student_v1':{'phase':torch.zeros(8,dtype=torch.long)}})
    alg.compute_returns(obs)


def test_production_ppo_reaches_wheel_head_and_student_loss_freezes_decoder():
    alg,obs,*_=new_algorithm();collect(alg,obs)
    before=deepcopy(alg.models.state_dict());grads=[]
    def capture():grads.append(alg.models.decoder.idle_wheel_mlp[-1].weight.grad.abs().sum().item())
    with patch.object(alg.optimizer,'step',capture):m=alg.ppo_update()
    assert m['ppo_wall_idle_samples']>0 and m['ppo_wall_idle_output_loss']>0 and max(grads)>0
    with patch.object(alg.student_optimizer,'step',lambda:None):m=alg.distill_update()
    assert m['wall_quality_samples']>0 and m['wall_idle_samples']>0
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in alg.student_parameters)
    for k,v in before.items():torch.testing.assert_close(alg.models.state_dict()[k],v,rtol=0,atol=0)


def test_new_tags_are_required_and_recovery_is_excluded():
    alg,obs,*_=new_algorithm();alg.act(obs)
    extra={'cts_anchor':torch.zeros(8,dtype=torch.bool),'cts_course':torch.zeros(8,dtype=torch.long),
           'student_v1':{'phase':torch.ones(8,dtype=torch.long)}}
    with pytest.raises(ValueError,match='cts_wall_quality'):
        alg.process_env_step(obs,torch.zeros(8),torch.zeros(8,dtype=torch.bool),extra)
    extra.update(cts_wall_quality=torch.ones(8,dtype=torch.bool),cts_wall_idle=torch.ones(8,dtype=torch.bool))
    alg.process_env_step(obs,torch.zeros(8),torch.zeros(8,dtype=torch.bool),extra)
    assert not alg.rollout[-1]['wall_idle'].any() and not alg.rollout[-1]['wall_quality'].any()


def test_hold_changes_only_stairs_and_legacy_preserves_scalar():
    _,s,_=physical();s.stairs[4:]=False
    hold=wall.initial_hold(s)
    assert (hold[:4]==config()['stair_initial_hold_s']).all()
    assert (hold[4:]==s.cfg['commands']['initial_hold_s']).all()
    s.cfg.pop('wall_stability');assert isinstance(wall.initial_hold(s),(int,float))


@pytest.mark.parametrize('key,value',[('minimum_supports',2),('probability',1.1),('quality_weight',float('nan')),
 ('protected_half_width_m',.5),('batch_size',1.5)])
def test_invalid_configuration_fails_closed(key,value):
    c=config();c[key]=value
    with pytest.raises(ValueError):wall.validate(c)


def test_live_course_hold_boundary_timeout_and_recovery_use_same_duration(source):
    from test_cts_all_terrain_course import fake as course_fake
    from test_cts_recovery import recovering
    torch.manual_seed(1234)
    env,old=course_fake(source,n=100)
    cfg,atlas=source;newcfg=deepcopy(cfg);newcfg['all_terrain']['wall_stability']=config()
    torch.manual_seed(1234)
    newenv,new=course_fake((newcfg,atlas),n=100)
    extra=config()['stair_initial_hold_s']-new.cfg['commands']['initial_hold_s']
    torch.testing.assert_close(new.timeout-old.timeout,new.stairs.float()*extra,atol=1e-5,rtol=0)
    newenv.episode_length_buf.fill_(20);env.episode_length_buf.fill_(20)
    torch.testing.assert_close(new.raw_command()[~new.stairs],old.raw_command()[~old.stairs],atol=0,rtol=0)
    assert not new.raw_command()[new.stairs].any()
    newenv.episode_length_buf.fill_(100)
    torch.testing.assert_close(new.raw_command()[new.stairs],new.command_value[new.stairs])
    env,s=recovering((newcfg,atlas));r=s.recovery
    g=s.observe().copy();ids=r.active.nonzero().flatten()
    g['position']=g['position'].clone();g['position'][ids,:2,0]=r.trigger_x[ids,None]+.1
    g['supported']=torch.ones(env.num_envs,4,dtype=torch.bool)
    env.episode_length_buf.fill_(99)
    r.advance(g,torch.ones(env.num_envs,dtype=torch.bool));assert not (r.phase[ids]==2).any()
    env.episode_length_buf.fill_(100)
    r.advance(g,torch.ones(env.num_envs,dtype=torch.bool));assert (r.phase[ids]==2).all()


from test_cts_all_terrain_course import source
