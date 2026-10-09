import sys,json
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,'scripts/tools')
from check_gait_safety import load
from check_stair_rewards import Entity,MDP,make_env,context
ns=load(MDP/'stair_teacher.py'); torch.set_num_threads(1)
out={}
for angle in (20,25,30,35,40,49):
 for force in (0,6,15,60,100):
  env=make_env();w=env.scene['robot'].data.body_pos_w;hips=w.clone();hips[...,2]+=.42
  env.scene['robot'].data.body_pos_w=torch.cat([w,hips],1)
  r=np.deg2rad(angle)
  env.scene['robot'].data.body_pos_w[:,2:4,0]=hips[:,2:4,0]+.45*np.sin(r)
  env.scene['robot'].data.body_pos_w[:,2:4,2]=hips[:,2:4,2]-.45*np.cos(r)
  env.scene['robot'].data.joint_pos=torch.zeros(1,4);env.scene['robot'].data.root_ang_vel_b=torch.zeros(1,3)
  env.scene['contact_forces'].data.net_forces_w[:,:,2]=force
  env.scene['contact_forces'].data.current_contact_time.fill_(.02 if force else 0)
  context(env);s=env._m20_stair_scan_context['context'];s['down_gate'].fill_(1);s['up_gate'].fill_(0)
  c=ns['descent_rear_forward_cost'](env,Entity('robot'),Entity('robot',body_ids=[4,5,6,7]),Entity('robot',joint_ids=[0,1,2,3]),Entity('height_scanner'),Entity('contact_forces'))
  out[f'angle{angle}_force{force}']=float(c)
Path('docs/gait_audit_20261009/numeric.json').write_text(json.dumps(out,indent=2))
results={}
for name in ('latest','baseline'):
 for case in ('ascent','descent'):
  p=Path(f'docs/gait_audit_20261009/{name}_{case}')
  summary=json.loads((p/'summary.json').read_text())['cases'][case];d=np.load(p/f'{case}.npz')
  q=d['root_quat'];v=d['wheel_pos']-d['hip_pos']
  # Inverse full quaternion: validate vertical=0 and forward45=45 separately below.
  vv=v.copy();qq=np.repeat(q[:,None,:],4,axis=1);qi=qq.copy();qi[:,:,1:]*=-1
  t=2*np.cross(qi[:,:,1:],vv);body=vv+qi[:,:,:1]*t+np.cross(qi[:,:,1:],t)
  yaw=np.arctan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2]**2+q[:,3]**2))
  gx=np.cos(yaw)[:,None]*v[:,:,0]+np.sin(yaw)[:,None]*v[:,:,1]
  b=np.degrees(np.arctan2(body[:,:,0],np.maximum(-body[:,:,2],1e-6)))[:,2:]
  g=np.degrees(np.arctan2(gx,np.maximum(-v[:,:,2],1e-6)))[:,2:]
  force=d['contact_force'];loaded=(force[:,:,2]>5)&(force[:,:,2]>.5*np.linalg.norm(force,axis=2));rl=loaded[:,2:]
  onstairs=(d['root_pos'][:,0]>.0)&(d['root_pos'][:,0]< (5.8 if case=='ascent' else 2.8))
  def stats(mask):
   mask=np.broadcast_to(mask,b.shape)
   return dict(samples=int(mask.sum()),b_p95=float(np.percentile(b[mask],95)) if mask.any() else None,b_max=float(b[mask].max()) if mask.any() else None,g_p95=float(np.percentile(g[mask],95)) if mask.any() else None,excess_fraction=float(((b>35)|(g>25))[mask].mean()) if mask.any() else None)
  land=summary['landings'];seen=[{},{}]; seq=[[],[]]
  for e in land:
   level=e['level'];leg=e['leg'];a=0 if leg in ('fl','fr') else 1
   if level<=0 or level>=(20 if case=='ascent' else 5):continue
   if leg not in seen[a].setdefault(level,set()):seq[a].append(f'{leg}{level}');seen[a][level].add(leg)
  results[name+'_'+case]=dict(final_xyz=summary['final_xyz'],fell=summary['fell'],crossing={k:v for k,v in summary.items() if 'cross' in k or 'pass' in k},duration=float(d['time'][-1]),max_abs_y=float(abs(d['root_pos'][:,1]).max()),onstairs_heading_abs_p95_deg=float(np.percentile(abs(np.degrees(yaw[onstairs])),95)) if onstairs.any() else None,loaded_rear_onstairs=stats(rl&onstairs[:,None]),unloaded_rear_onstairs=stats(~rl&onstairs[:,None]),front_sequence=seq[0],rear_sequence=seq[1],historical_dual_treads=[sum(len(x)==2 for x in axle.values()) for axle in seen])
assert np.degrees(np.arctan2(0.,1.))==0
assert abs(np.degrees(np.arctan2(1.,1.))-45)<1e-6
Path('docs/gait_audit_20261009/mujoco_analysis.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
print('numeric',out)
