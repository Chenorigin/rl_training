import sys,json
from pathlib import Path
sys.path.insert(0,'scripts/tools')
import torch,numpy as np
from check_gait_safety import load
from check_stair_rewards import MDP,make_env,context,Entity
ns=load(MDP/'stair_teacher.py'); torch.set_num_threads(1)
p=dict(asset_cfg=Entity('robot'),hip_cfg=Entity('robot',body_ids=[4,5,6,7]),knee_cfg=Entity('robot',joint_ids=[0,1,2,3]),sensor_cfg=Entity('height_scanner'),contact_sensor_cfg=Entity('contact_forces'))
def make(angle,force,down=True):
 e=make_env();r=e.scene['robot'].data
 hips=r.body_pos_w.clone();hips[...,2]+=.45
 r.body_pos_w=torch.cat([r.body_pos_w,hips],1);r.joint_pos=torch.ones(1,4);r.root_ang_vel_b=torch.zeros(1,3)
 r.body_pos_w[:,2:4,0]=hips[:,2:4,0]+.45*np.sin(np.deg2rad(angle))
 r.body_pos_w[:,2:4,2]=hips[:,2:4,2]-.45*np.cos(np.deg2rad(angle))
 c=e.scene['contact_forces'].data;c.net_forces_w[:,2:,2]=force;c.current_contact_time[:,2:]=.02 if force else 0
 context(e);q=e._m20_stair_scan_context['context'];q['down_gate'].fill_(float(down));q['up_gate'].fill_(float(not down));return e
out={}
for f in (0,6,100):
 for a in (24,30,49,70):
  e=make(a,f);out[f'angle{a}_force{f}']=float(ns['descent_rear_forward_cost'](e,**p))
assert out['angle49_force0']>0 and out['angle30_force6']>0
for f in (0,6,100):
 e=make(70,f,False);out[f'uphill_force{f}']=float(ns['descent_rear_forward_cost'](e,**p));assert out[f'uphill_force{f}']==0
# Cached stats should update once.
e=make(49,0);ns['descent_rear_forward_cost'](e,**p);s=e._m20_rear_swing_stats['samples'].clone();ns['descent_rear_forward_cost'](e,**p);assert torch.equal(s,e._m20_rear_swing_stats['samples']);out['cached_swing_stats_once']=True
# Explicitly break all live tensor batches into two envs; no production mocks beyond sensor data.
e=make(49,0)
for obj in [e.scene['robot'].data,e.scene['height_scanner'].data,e.scene['contact_forces'].data]:
 for k,v in vars(obj).items():
  if isinstance(v,torch.Tensor):setattr(obj,k,v.repeat((2,)+(1,)*(v.ndim-1)))
e.num_envs=2;e.episode_length_buf=torch.tensor([2,2]);cmd=torch.tensor([[.5,0,0],[.5,0,0.]])
e.command_manager.get_command=lambda _:cmd
q=e._m20_stair_scan_context['context'];e._m20_stair_scan_context['context']={k:v.repeat((2,)+(1,)*(v.ndim-1)) for k,v in q.items()}
y=ns['descent_rear_forward_cost'](e,**p);assert y.shape==(2,) and torch.allclose(y[0],y[1]);out['vectorized_cost']=y.tolist()
# Drop scanner edge while retaining close physical phase.
e.common_step_counter+=1;e.episode_length_buf+=1;e._m20_stair_scan_context['counter']=e.common_step_counter;e._m20_stair_scan_context['context']['down_gate'].zero_();e._m20_stair_scan_context['context']['up_gate'].zero_()
y=ns['descent_rear_forward_cost'](e,**p);out['no_scan_latched_cost']=y.tolist();assert (y>0).all()
# One env reset cannot leak phase into a new flat episode.
e.common_step_counter+=1;e.episode_length_buf=torch.tensor([1,4]);e._m20_stair_scan_context['counter']=e.common_step_counter
y=ns['descent_rear_forward_cost'](e,**p);out['one_env_reset_cost']=y.tolist();assert y[0]==0 and y[1]>0
# Score rises toward next tread but declines for overshoot and excess lift.
f=ns['rear_target_score'];tx=torch.tensor([.1]);tz=torch.tensor([.2]);near=float(f(torch.tensor([.1]),torch.tensor([.2]),tx,tz));overshoot=float(f(torch.tensor([.7]),torch.tensor([.2]),tx,tz));overlift=float(f(torch.tensor([.1]),torch.tensor([.7]),tx,tz));out['target_scores']=[near,overshoot,overlift];assert near>overshoot and near>overlift
print(json.dumps(out,indent=2))
Path('docs/gait_fix_20261009/reviewer_probe.json').write_text(json.dumps(out,indent=2)+'\n')
import ast
module=ast.parse(Path('scripts/tools/check_gait_v3.py').read_text())
checkfn=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='check')
makefn=next(n for n in checkfn.body if isinstance(n,ast.FunctionDef) and n.name=='make')
local=dict(torch=torch,np=np,make_env=make_env,context=context,ns=ns,p={k:v for k,v in p.items() if k not in ('hip_cfg','knee_cfg')})
exec(compile(ast.Module(body=[makefn],type_ignores=[]),'<actual test fixture>','exec'),local)
e,tick,land=local['make']()
values=[]
for xpos in (-.25,-.15,-.25,-.15,-.35,-.15):
 st=tick(2,xpos,.12,False)
 values.append(float(st['rear_lift_gain']))
out['rear_oscillation_gains']=values
assert sum(values[2:])==0,values
cmd=e.command_manager.get_command('base_velocity');cmd[:,0]=0;st=tick(2,-.3,.12,False);cmd[:,0]=.5;st=tick(2,-.15,.12,False)
out['rear_pause_restart_gain']=float(st['rear_lift_gain']);assert out['rear_pause_restart_gain']==0
# A correct first rear landing determines an opposite-side next target.
e,tick,land=local['make']();st=land(2,0)
lead=int(st['rear_expected'][0]);assert lead==1
before=float(st['rear_lift_credit']);st=tick(2,.4,.29,False)
out['wrong_leg_target_gain']=float(st['rear_lift_gain']);assert out['wrong_leg_target_gain']==0
print(json.dumps({k:out[k] for k in ('rear_oscillation_gains','rear_pause_restart_gain','wrong_leg_target_gain')},indent=2))
Path('docs/gait_fix_20261009/reviewer_probe.json').write_text(json.dumps(out,indent=2)+'\n')
# Controlled historical-target revisit: force selection away and back without
# discarding its ledger. This tests reset/revisit bookkeeping, not a gait track.
e,tick,land=local['make']();st=tick(2,-.15,.12,False)
slot=int(st['axis_target_slot'][0,1]);oldmax=float(st['map_prep_max'][0,slot,1]);oldinit=bool(st['rear_prep_initialized'][0,slot]);assert oldinit
st['map_paid'][0,slot,1]=True;tick(2,-.3,.12,False)
st['map_paid'][0,slot,1]=False;st=tick(2,-.15,.12,False)
out['historical_target_revisit_gain']=float(st['rear_lift_gain']);out['historical_target_preserved_max']=float(st['map_prep_max'][0,slot,1]);assert out['historical_target_revisit_gain']==0 and out['historical_target_preserved_max']>=oldmax
# Descent phase cannot survive a scan-free 90-degree turn to another direction.
e=make(49,0);ns['descent_rear_forward_cost'](e,**p);e.common_step_counter+=1;e.episode_length_buf+=1;e._m20_stair_scan_context['counter']=e.common_step_counter;e._m20_stair_scan_context['context']['down_gate'].zero_();e._m20_stair_scan_context['context']['up_gate'].zero_();e.scene['height_scanner'].data.quat_w[:]=torch.tensor([[np.sqrt(.5),0,0,np.sqrt(.5)]])
y=ns['descent_rear_forward_cost'](e,**p);out['turned_away_phase_cost']=float(y);assert y==0 and not bool(e._m20_descent_pose_phase['active'])
print(json.dumps({k:out[k] for k in ('historical_target_revisit_gain','historical_target_preserved_max','turned_away_phase_cost')},indent=2))
Path('docs/gait_fix_20261009/reviewer_probe.json').write_text(json.dumps(out,indent=2)+'\n')
