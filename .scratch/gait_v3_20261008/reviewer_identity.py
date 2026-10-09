import sys,json
from pathlib import Path
import torch
sys.path.insert(0,str(Path.cwd()/'scripts/tools'))
from check_gait_safety import load
from check_stair_rewards import Entity,make_env,MDP,context
ns=load(MDP/'stair_teacher.py');torch.set_num_threads(1)
p=dict(asset_cfg=Entity('robot'),sensor_cfg=Entity('height_scanner'),contact_sensor_cfg=Entity('contact_forces'))
e=make_env()
def step(x=None,z=None,touch=True,shift_y=0.):
 e.common_step_counter+=1;e.episode_length_buf+=1
 e.scene['height_scanner'].data.pos_w[0,1]=shift_y
 if x is not None:
  e.scene['robot'].data.body_pos_w[0,0,:]=torch.tensor([x,.22+shift_y,z])
  e.scene['contact_forces'].data.current_contact_time[0,0]=.2 if touch else 0
  e.scene['contact_forces'].data.net_forces_w[0,0]=torch.tensor([0.,0.,100. if touch else 0.])
 context(e,0.,.1)
 return ns['ascent_state'](e,**p)
def land(y=0.):
 step(-.12,.09,True,y);step(-.04,.26,False,y);step(.08,.26,False,y);step(.1,.19,True,y);return step(.1,.19,True,y)
step();s=land();first=float(s['front_successes']);s=land(.6)
r=dict(first_physical_tread_successes=first,after_same_tread_lateral_scan_shift_successes=float(s['front_successes']),map_count=int(s['map_count']),map_z=s['map_z'][0,:int(s['map_count'])].tolist(),map_xy=s['map_xy'][0,:int(s['map_count'])].tolist(),flights=int(s['flight_count']))
assert r['after_same_tread_lateral_scan_shift_successes']==1 and r['map_count']==1,r
print(json.dumps(r));Path('.scratch/gait_v3_20261008/reviewer_identity.json').write_text(json.dumps(r,indent=2)+'\n')
