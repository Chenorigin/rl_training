#!/usr/bin/env python3
"""CPU-only direct TB/checkpoint and frozen MuJoCo summary review."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT/'logs/rsl_rl/deeprobotics_m20_stair_teacher'
base = LOG/'2026-10-06_22-06-56_stair_resume'
latest = LOG/'2026-10-08_16-51-12_gait_v3'
out = {}
for label, directory in (('baseline', base), ('latest', latest)):
    ea = EventAccumulator(str(directory), size_guidance={'scalars': 0})
    ea.Reload()
    curves = {}
    for tag in ea.Tags()['scalars']:
        values = ea.Scalars(tag)
        arr = np.array([v.value for v in values])
        n = min(50, len(values))
        curves[tag] = dict(first_step=values[0].step, last_step=values[-1].step,
            first=float(arr[0]), early=float(arr[:n].mean()), late=float(arr[-n:].mean()),
            last=float(arr[-1]), minimum=float(arr.min()), maximum=float(arr.max()),
            n=len(values), checkpoints={str(v.step): v.value for v in values if v.step in (0,1,10,50,100,500,1000,1500,1700)})
    rewards = {k:v for k,v in curves.items() if k.startswith('Episode_Reward/')}
    sums = {}
    for stage in ('early', 'late'):
        positive = sum(v[stage] for v in rewards.values() if v[stage]>0)
        negative = sum(v[stage] for v in rewards.values() if v[stage]<0)
        sums[stage] = dict(positive=positive, negative=negative, total=positive+negative,
            top_positive=sorted(((k,v[stage]) for k,v in rewards.items() if v[stage]>0),key=lambda kv:-kv[1])[:8],
            top_negative=sorted(((k,v[stage]) for k,v in rewards.items() if v[stage]<0),key=lambda kv:kv[1])[:10])
    params = {name: yaml.load((directory/'params'/f'{name}.yaml').read_text(), Loader=yaml.BaseLoader) for name in ('agent','env')}
    out[label] = dict(curves=curves, reward_sums=sums, agent=params['agent'],
        episode_length_s=params['env'].get('episode_length_s'),
        reward_weights={k:v.get('weight') for k,v in params['env']['rewards'].items() if isinstance(v,dict)},
        physics=params['env'].get('sim'), decimation=params['env'].get('decimation'))

paths = [base/'model_199998.pt', latest/'model_0.pt', latest/'model_100.pt', latest/'model_500.pt',latest/'model_1700.pt']
states = {}
for path in paths:
    data = torch.load(path,map_location='cpu',weights_only=True)
    actor, critic = data['actor_state_dict'], data['critic_state_dict']
    states[path.name if path.parent==latest else 'baseline199998'] = (actor,critic)
    std = {k:torch.exp(v).flatten().tolist() if 'log_std' in k else v.flatten().tolist()
           for k,v in actor.items() if 'std' in k}
    optimizer = data.get('optimizer_state_dict',{})
    pg = optimizer.get('param_groups',[])
    out.setdefault('checkpoints',{})[str(path.relative_to(ROOT))] = dict(iteration=data.get('iter'),std=std,
        actual_learning_rates=[group.get('lr') for group in pg], keys=list(data),
        actor_dims=list(actor['mlp.0.weight'].shape),critic_dims=list(critic['mlp.0.weight'].shape),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest())
base_actor, base_critic = states['baseline199998']
for label, (actor,critic) in states.items():
    if label=='baseline199998':continue
    def relative(target,source):
        keys=[k for k in source if k.startswith('mlp.')]
        diff=sum(float((target[k]-source[k]).square().sum()) for k in keys)
        norm=sum(float(source[k].square().sum()) for k in keys)
        return (diff/norm)**.5
    out.setdefault('relative_network_drift',{})[label] = dict(actor=relative(actor,base_actor),critic=relative(critic,base_critic))

comparison={}
for label in ('baseline','latest'):
    summary=json.loads((ROOT/f'docs/gait_v3_regression/{label}/summary.json').read_text())
    comparison[label] = {'checkpoint':summary['checkpoint'], 'cases':{}}
    for name,c in summary['cases'].items():
        fields=['cleared_terrain','crossed_finish','inside_course','fell','translation_velocity_error_mean_m_s',
                'turn_yaw_error_mean_rad_s','mean_absolute_mechanical_power_w','turn_reach_p95_m','turn_clearance_p95_m',
                'front_same_tread_events','rear_same_tread_events','front_alternation_rate','rear_alternation_rate',
                'alternation_checks','final_xyz']
        comparison[label]['cases'][name] = {k:c.get(k) for k in fields}
        landings=c.get('landings',[])
        comparison[label]['cases'][name]['highest_landing_level'] = max([v['level'] for v in landings],default=None)
out['actual_mujoco_summaries_read']=comparison
out['source_sha256']=hashlib.sha256((ROOT/'source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/stair_teacher.py').read_bytes()).hexdigest()
Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2)+'\n')
for label in ('baseline','latest'):
    print(label, json.dumps(out[label]['reward_sums'],ensure_ascii=False))
    print(label,'non_reward_curves',json.dumps({k:v for k,v in out[label]['curves'].items() if not k.startswith('Episode_Reward/') and any(word in k.lower() for word in ('loss','std','learning','episode','level','success','stationary','retreat','loaded_a','rear_entry','strict','skip','same_tread'))},ensure_ascii=False))
print('checkpoint_summary',json.dumps(out['checkpoints'],ensure_ascii=False))
print('relative_network_drift',json.dumps(out['relative_network_drift'],ensure_ascii=False))
print('mujoco',json.dumps(comparison,ensure_ascii=False))
