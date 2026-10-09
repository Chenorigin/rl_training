#!/usr/bin/env python3
"""CPU checks for task-vs-style rewards and PPO preservation using real RSL-RL."""
import argparse
import copy
import json
from pathlib import Path
import sys

import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'source/rl_training'))
import importlib.util
spec = importlib.util.spec_from_file_location("stair_guarded_ppo", ROOT/"source/rl_training/rl_training/stair_guarded_ppo.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
StairGuardedPPO = module.StairGuardedPPO
from check_gait_safety import load


def models(obs):
    groups={'actor':['policy'],'critic':['critic']}
    actor=MLPModel(obs,groups,'actor',16,hidden_dims=[32,32],activation='elu',
        distribution_cfg={'class_name':'GaussianDistribution','init_std':.35,'std_type':'log'})
    critic=MLPModel(obs,groups,'critic',1,hidden_dims=[32,32],activation='elu')
    return actor,critic


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    torch.set_num_threads(1);torch.manual_seed(42)
    obs=TensorDict({'policy':torch.randn(8,244)*.1,'critic':torch.randn(8,285)*.1},batch_size=[8])
    actor,critic=models(obs)
    storage=RolloutStorage('rl',8,4,obs,[16])
    alg=StairGuardedPPO(actor,critic,storage,critic_warmup_iterations=1,num_learning_epochs=1,
        num_mini_batches=1,learning_rate=1e-4,schedule='fixed')
    reference=copy.deepcopy(actor.state_dict());alg.set_reference(reference)
    def rollout():
        for _ in range(4):
            with torch.no_grad():
                alg.act(obs)
                alg.process_env_step(obs,torch.ones(8),torch.zeros(8),{})
        alg.compute_returns(obs)
    before=copy.deepcopy(actor.state_dict());cbefore=copy.deepcopy(critic.state_dict())
    rollout();warmup=alg.update()
    assert warmup['critic_warmup'] == 1.0
    assert all(torch.equal(before[k],v) for k,v in actor.state_dict().items())
    assert any(not torch.equal(cbefore[k],v) for k,v in critic.state_dict().items())
    with torch.no_grad():actor.mlp[-1].bias.add_(.1)
    positive=float(alg._anchor_loss(obs));assert positive>0
    rollout();losses=alg.update();assert losses['anchor']>0 and losses['critic_warmup']==0
    assert all(p.grad is None for p in alg.reference_actor.parameters())
    assert all(torch.equal(reference[k],v) for k,v in alg.reference_actor.state_dict().items())
    saved=alg.save();a2,c2=models(obs);restored=StairGuardedPPO(a2,c2,RolloutStorage('rl',8,4,obs,[16]))
    restored.load(saved,None,True)
    assert restored.guard_updates==2 and restored.reference_actor is not None
    legacy=copy.deepcopy(saved)
    del legacy['preservation_reference_actor'];del legacy['preservation_updates']
    for group in legacy['optimizer_state_dict']['param_groups']:group['lr']=0.000170859375
    a3,c3=models(obs)
    legacy_resume=StairGuardedPPO(a3,c3,RolloutStorage('rl',8,4,obs,[16]),learning_rate=1e-5,schedule='fixed')
    legacy_resume.load(legacy,None,True)
    assert legacy_resume.reference_actor is not None and legacy_resume.guard_updates==0
    assert all(g['lr']==1e-5 for g in legacy_resume.optimizer.param_groups)
    with torch.no_grad():alg.actor.distribution.log_std_param.add_(2)
    rollout();alg.update()
    limit=alg.reference_actor.distribution.log_std_param.exp()*alg.reference_std_multiplier
    assert (alg.actor.distribution.log_std_param.exp()<=limit+1e-7).all()
    ns=load(ROOT/'source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/stair_teacher.py')
    n=1;m=2
    state={'map_landed':torch.ones(n,m,4,dtype=torch.bool),'map_safe':torch.ones(n,m,4,dtype=torch.bool),
        'map_half_width':torch.ones(n,m)*.05,'map_z':torch.tensor([[.1,.2]]),
        'map_task_paid':torch.zeros(n,m,dtype=torch.bool),'radius':torch.ones(n,4)*.09}
    known=torch.ones(n,m,dtype=torch.bool);x=torch.ones(n,m,4)*.2;cross=torch.zeros(n,m,4)
    wheels=torch.zeros(n,4,3);wheels[...,2]=.4;loaded=torch.ones(n,4,dtype=torch.bool)
    root_pos=torch.tensor([[.7,0,.6]]);root_quat=torch.tensor([[1.,0,0,0]])
    fn=lambda s,k,x,c,w,l:ns['ascent_task_crossing_events'](s,k,x,c,w,l,root_pos,root_quat)
    valid=fn(state,known,x,cross,wheels,loaded)
    assert valid.sum()==2
    state['map_task_paid'] |= valid
    assert not fn(state,known,x,cross,wheels,loaded).any()
    state['map_task_paid'].zero_()
    assert not fn(state,known,x,cross+2,wheels,loaded).any()
    assert not fn(state,known,-x,cross,wheels,loaded).any()
    state['map_landed'][:,:,2:]=False
    assert not fn(state,known,x,cross,wheels,loaded).any()
    state['map_landed'].fill_(True);state['map_safe'].zero_()
    assert fn(state,known,x,cross,wheels,loaded).all()  # physical task is not a style reward
    root_pos[:,2]=.2
    assert not fn(state,known,x,cross,wheels,loaded).any()
    root_pos[:,2]=.6;root_quat[:]=torch.tensor([[0.,1.,0.,0.]])
    assert not fn(state,known,x,cross,wheels,loaded).any()
    out={'warmup_actor_exactly_frozen':True,'warmup_critic_changed':True,'anchor_detects_drift':positive,
         'actual_ppo_losses':losses,'reference_frozen':True,'checkpoint_auxiliary_state_restored':True,
         'std_growth_capped':True,'crossing_no_repeat_bypass_retreat_front_only_collapse_or_inversion':True}
    out['legacy_resume_restores_guards_and_honors_fixed_lr']=True
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))


if __name__=='__main__':main()
