from copy import deepcopy
from unittest.mock import patch
import pytest
import torch
from task_support.cts.configuration import default_settings
from task_support.cts.clutter_consistency import eligible,perturb,loss,grid,validate
from test_cts_auxiliary import make_algorithm,collect,aux_config


def fixture():
    alg,obs=make_algorithm(aux_config())
    obs['teacher'][:,57:244]=0.;obs['policy'][:,57:244]=0.
    obs['policy'][:,244:618]=1.
    collect(alg,obs)
    data=alg._data();data['course'].fill_(3);data['clutter'].fill_(True)
    cfg=default_settings()['all_terrain']['clutter_consistency'];cfg['probability']=1.
    return alg,data,cfg


def test_structured_returns_preserve_abi_commands_history_quality_and_central_support():
    _,d,c=fixture();p=d['obs']['policy'];old=p.clone();kinds=torch.arange(len(p))%4
    q,changed,k=perturb(p,c,kinds=kinds,heights=torch.full((len(p),1),.8))
    assert changed.all() and q.shape==(32,789)
    torch.testing.assert_close(p,old,rtol=0,atol=0)
    torch.testing.assert_close(q[:,:57],p[:,:57],rtol=0,atol=0)
    torch.testing.assert_close(q[:,244:],p[:,244:],rtol=0,atol=0)
    x,y=grid(p);centre=(x.abs()<=c['protected_half_length_m']) & (y.abs()<=c['protected_half_width_m'])
    torch.testing.assert_close(q[:,57:244][:,centre],p[:,57:244][:,centre])
    assert (q[:,57:244]<=p[:,57:244]).all() and (q[:,57:244]<p[:,57:244]).any()


def test_no_unknown_or_zero_confidence_fill_and_noop_is_not_sample():
    _,d,c=fixture();p=d['obs']['policy'];p[:,244:431]=0.
    q,changed,_=perturb(p,c);assert not changed.any();torch.testing.assert_close(p,q)
    p[:,244:431]=1.;p[:,431:618]=0.
    q,changed,_=perturb(p,c);assert not changed.any();torch.testing.assert_close(p,q)


@pytest.mark.parametrize('kind',['up','down','roundtrip','pit','platform','rough','slope','invalid','unsupported','recovery'])
def test_actual_terrain_and_invalid_states_never_receive_invariance(kind):
    _,d,c=fixture();regular=torch.ones(32,dtype=torch.bool)
    codes={'up':0,'down':1,'roundtrip':2,'pit':6,'platform':9,'rough':4}
    if kind in codes:d['course'].fill_(codes[kind])
    if kind=='slope':d['obs']['teacher'][:,57:244]=torch.linspace(0.,.2,187)
    if kind=='invalid':d['obs']['policy'][:,244:431]=0.
    if kind=='unsupported':d['clutter'].zero_()
    if kind=='recovery':regular.zero_()
    assert not eligible(d,regular,c).any()


def test_real_student_gradient_detached_anchor_and_unchanged_rollout():
    alg,d,c=fixture();ids=torch.arange(32);reg=torch.ones(32,dtype=torch.bool)
    original={k:v.clone() for k,v in d['obs'].items()}
    alg.models.decoder.requires_grad_(False)
    calls=[];real=alg.models.student_action
    def instrument(p,cmd):
        calls.append(torch.is_grad_enabled());return real(p,cmd)
    with patch.object(alg.models,'student_action',instrument):v,m=loss(alg.models,d,ids,reg,c)
    assert v>0 and m['clutter_samples']==32 and calls==[False,True]
    v.backward();assert any(p.grad is not None and p.grad.abs().sum()>0 for p in alg.student_parameters)
    assert all(p.grad is None for p in alg.models.decoder.parameters())
    for k,v in original.items():torch.testing.assert_close(d['obs'][k],v,rtol=0,atol=0)


def test_production_distillation_calls_new_loss_without_optimizer_updates():
    alg,d,c=fixture()
    for step in alg.rollout:step['course'].fill_(3);step['clutter'].fill_(True)
    alg.all_terrain_cfg={'clutter_consistency':c}
    before={k:v.clone() for k,v in alg.models.state_dict().items()}
    with patch.object(alg.student_optimizer,'step',lambda:None):metrics=alg.distill_update()
    assert metrics['clutter_samples']>0 and metrics['clutter_leg_mse']>0
    for k,v in before.items():torch.testing.assert_close(alg.models.state_dict()[k],v,rtol=0,atol=0)


def test_unsafe_reference_not_imitated():
    alg,d,c=fixture()
    class Safety:
        def accept(self,a):return torch.zeros(len(a),dtype=torch.bool)
    v,m=loss(alg.models,d,torch.arange(32),torch.ones(32,dtype=torch.bool),c,Safety())
    assert v==0 and m['clutter_samples']==0


def test_missing_production_tag_is_error():
    alg,d,c=fixture();alg.all_terrain_cfg={'clutter_consistency':c}
    alg.rollout.clear();obs={k:v[:8] for k,v in d['obs'].items()};alg.act(obs)
    extra={'cts_anchor':torch.ones(8,dtype=torch.bool),'cts_course':torch.full((8,),3,dtype=torch.long),'student_v1':{'phase':torch.zeros(8,dtype=torch.long)}}
    with pytest.raises(ValueError,match='cts_clutter'):alg.process_env_step(obs,torch.zeros(8),torch.zeros(8,dtype=torch.bool),extra)


@pytest.mark.parametrize('k,v',[('weight',float('nan')),('probability',1.1),('batch_size',2.5),('protected_half_length_m',.1)])
def test_config_rejects_invalid_values(k,v):
    _,_,c=fixture();c[k]=v
    with pytest.raises(ValueError):validate(c)


def test_actual_front_only_output_mask_admits_samples_but_blackout_does_not():
    from task_support.student_v1.configuration import StudentV1PerceptionCfg
    from task_support.student_v1.perception import grid_points,map_output_mask
    alg,d,c=fixture();sensor=StudentV1PerceptionCfg(front_only=True)
    mask=map_output_mask(grid_points('cpu'),sensor)
    d['obs']['policy'][:,244:431]=mask.float()
    assert eligible(d,torch.ones(32,dtype=torch.bool),c).all()
    _,m=loss(alg.models,d,torch.arange(32),torch.ones(32,dtype=torch.bool),c)
    assert m['clutter_samples']>0
    d['obs']['policy'][:,244:431]=0.
    assert not eligible(d,torch.ones(32,dtype=torch.bool),c).any()


def test_clutter_checks_phase_even_if_old_temporal_spatial_objectives_disabled():
    alg,d,c=fixture();alg.all_terrain_cfg={'clutter_consistency':c}
    alg.rollout.clear();obs={k:v[:8] for k,v in d['obs'].items()};alg.act(obs)
    extra={'cts_anchor':torch.ones(8,dtype=torch.bool),'cts_course':torch.full((8,),3,dtype=torch.long),
           'cts_clutter':torch.ones(8,dtype=torch.bool),'student_v1':{'phase':torch.ones(8,dtype=torch.long)}}
    alg.process_env_step(obs,torch.zeros(8),torch.zeros(8,dtype=torch.bool),extra)
    assert not alg.rollout[-1]['regular_phase'].any()
