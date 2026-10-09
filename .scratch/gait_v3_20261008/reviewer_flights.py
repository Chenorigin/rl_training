import sys,json
from pathlib import Path
import torch
sys.path.insert(0,str(Path.cwd()/'scripts/tools'))
from check_gait_safety import load
from check_stair_rewards import Entity,make_env,MDP
ns=load(MDP/'stair_teacher.py');torch.set_num_threads(1)
p=dict(asset_cfg=Entity('robot'),sensor_cfg=Entity('height_scanner'),contact_sensor_cfg=Entity('contact_forces'))
e=make_env();wheel=e.scene['robot'].data.body_pos_w;heading=torch.tensor([1.,0.]);normal=torch.tensor([0.,1.]);origin=torch.tensor([0.,0.]);bottom=0.;flight=0

def tick(side=None,along=None,z=None,touch=True,visible=True):
 global wheel
 e.common_step_counter+=1;e.episode_length_buf+=1
 if side is not None:
  lateral=.22 if side%2==0 else -.22;wheel[0,side,:2]=origin+heading*along+normal*lateral;wheel[0,side,2]=z
  e.scene['contact_forces'].data.current_contact_time[0,side]=.2 if touch else 0
  e.scene['contact_forces'].data.net_forces_w[0,side]=torch.tensor([0.,0.,100. if touch else 0.])
 sensor=e.scene['height_scanner'];root=e.scene['robot'].data.root_pos_w
 mean=((wheel[0,:2,:2]-origin)*heading).sum(-1).mean()-.35;root[0,:2]=origin+heading*mean;sensor.data.pos_w[0,:2]=root[0,:2]
 xs=torch.arange(8)*.3-mean
 ctx=dict(edge_x=xs[:1],upper_z=torch.tensor([bottom+.1]),lower_z=torch.tensor([bottom]),up_gate=torch.ones(1),command_x=torch.tensor([.5]),
  ascending_edges_x=xs,ascending_edges_valid=((xs>=-.45)&(xs<=.70)&visible)[None],ascending_edges_upper_z=(torch.arange(8)*.1+.1+bottom)[None],ascending_edges_lower_z=(torch.arange(8)*.1+bottom)[None])
 e._m20_stair_scan_context=dict(counter=e.common_step_counter,sensor_name='height_scanner',context=ctx)
 return ns['ascent_state'](e,**p)

def land(side,i):
 x=i*.3;z=bottom+.1*(i+1)
 tick(side,x-.12,float(wheel[0,side,2]),True);tick(side,x-.04,z+.15,False);tick(side,x+.08,z+.15,False);tick(side,x+.10,z+.09,True);return tick(side,x+.10,z+.09,True)

tick()
for i in range(8):land(i%2,i)
for i in range(8):land(2+i%2,i)
s=tick();assert s['front_successes']==s['rear_successes']==8
# All four on the top platform; two feet on a plateau are not duplicates.
for side in range(4):tick(side,2.5,.89,True)
assert s['same_tread_events_total'].sum()==0
# A new flight turns 90 degrees and switches first-side, without episode reset.
origin=torch.tensor([3.,0.]);heading=torch.tensor([0.,1.]);normal=torch.tensor([-1.,0.]);bottom=.8
q=torch.tensor([[2**-.5,0.,0.,2**-.5]]);e.scene['robot'].data.root_quat_w=q;e.scene['height_scanner'].data.quat_w=q
for side in range(4):wheel[0,side,:2]=origin+heading*(-.25 if side<2 else -.95)+normal*(.22 if side%2==0 else -.22);wheel[0,side,2]=.89
for i in range(8):land(1-i%2,i)
for i in range(8):land(3-i%2,i)
s=tick()
result={k:float(s[k]) for k in ('front_successes','rear_successes','front_count','rear_count','flight_count')};result['same_tread_events']=s['same_tread_events_total'][0].tolist();result['longest_run']=s['axis_max_run'][0].tolist()
assert s['front_successes']==s['rear_successes']==16,result
assert s['flight_count']==2 and s['same_tread_events_total'].sum()==0,result
print(json.dumps(result));Path('.scratch/gait_v3_20261008/reviewer_flights.json').write_text(json.dumps(result,indent=2)+'\n')
