import sys,json
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,str(Path.cwd()/'scripts/tools'))
from check_gait_safety import load
from check_stair_rewards import Entity,make_env,MDP
ns=load(MDP/'stair_teacher.py');torch.set_num_threads(1);N=8
p=dict(asset_cfg=Entity('robot'),sensor_cfg=Entity('height_scanner'),contact_sensor_cfg=Entity('contact_forces'))
rng=np.random.default_rng(901338)
def new(n):
 e=make_env();e.num_envs=n
 for o in e.scene.values():
  for k,v in vars(o.data).items():
   if isinstance(v,torch.Tensor):setattr(o.data,k,v.repeat((n,)+(1,)*(v.ndim-1)))
 e.episode_length_buf=e.episode_length_buf.repeat(n)
 cmd=torch.tensor([[.5,0,0]]).repeat(n,1);e.command_manager.get_command=lambda _:cmd
 return e
bat=new(N);ss=[new(1) for _ in range(N)]
maxmaps=0
for t in range(120):
 yaw=torch.tensor(rng.uniform(-.35,.35,N),dtype=torch.float32);q=torch.zeros(N,4);q[:,0]=torch.cos(yaw/2);q[:,3]=torch.sin(yaw/2)
 shift=torch.tensor(rng.uniform(-.02,.04,(N,4,2)),dtype=torch.float32)
 heights=torch.tensor(rng.choice([.09,.19,.29,.39,.49,.59],(N,4)),dtype=torch.float32)
 touch=torch.tensor(rng.uniform(0,1,(N,4))>.35)
 contactforces=torch.zeros(N,4,3);contactforces[:,:,2]=touch.float()*torch.tensor(rng.uniform(30,150,(N,4)),dtype=torch.float32)
 resets=torch.tensor(rng.uniform(0,1,N)<.06)
 commanded=torch.tensor(rng.choice([0.,.5,-.5],N),dtype=torch.float32)
 seen=torch.tensor(rng.uniform(0,1,(N,3))>.3)
 for i,e in enumerate([bat]+ss):
  idx=slice(None) if i==0 else slice(i-1,i)
  e.common_step_counter+=1;e.episode_length_buf+=1;e.episode_length_buf[resets[idx]]=0
  e.scene['robot'].data.body_pos_w[:,:,:2]+=shift[idx]
  e.scene['robot'].data.body_pos_w[:,:,2]=heights[idx]
  e.scene['height_scanner'].data.quat_w=q[idx].clone();e.scene['robot'].data.root_quat_w=q[idx].clone()
  e.scene['contact_forces'].data.net_forces_w=contactforces[idx].clone()
  e.scene['contact_forces'].data.current_contact_time=torch.where(touch[idx],torch.full_like(heights[idx],.2),0)
  e.command_manager.get_command('base_velocity')[:,0]=commanded[idx]
  e._m20_stair_scan_context=dict(counter=e.common_step_counter,sensor_name='height_scanner',context=dict(
   ascending_edges_valid=seen[idx],ascending_edges_upper_z=torch.tensor([[.1,.2,.3]]).repeat(e.num_envs,1),
   ascending_edges_lower_z=torch.tensor([[0.,.1,.2]]).repeat(e.num_envs,1),ascending_edges_x=torch.tensor([.15,.45,.75]),
   edge_x=torch.full((e.num_envs,),.15),up_gate=seen[idx].any(-1).float(),command_x=commanded[idx]))
  ns['ascent_state'](e,**p)
 for key,a in bat._m20_ascent_state.items():
  b=torch.cat([e._m20_ascent_state[key] for e in ss])
  if a.is_floating_point():assert torch.allclose(a,b,atol=1e-6,rtol=1e-6,equal_nan=True),(t,key,torch.max(torch.nan_to_num((a-b).abs())))
  else:assert torch.equal(a,b),(t,key)
 maxmaps=max(maxmaps,int(bat._m20_ascent_state['map_count'].max()))
result=dict(envs=N,ticks=120,all_state_fields_batched_equal_individual=True,independent_resets=True,max_registered=maxmaps)
print(json.dumps(result));Path('.scratch/gait_v3_20261008/reviewer_vector_v3.json').write_text(json.dumps(result,indent=2)+'\n')
