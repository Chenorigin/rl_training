from pathlib import Path
import sys,json
sys.path.insert(0,str(Path.cwd()/'scripts/tools'))
from check_gait_safety import load
from check_stair_rewards import make_env,context,Entity,MDP
import torch
torch.set_num_threads(1);ns=load(MDP/'stair_teacher.py');e=make_env();p=dict(asset_cfg=Entity('robot'),sensor_cfg=Entity('height_scanner'),contact_sensor_cfg=Entity('contact_forces'))
def step(x,z,touch=True):
 e.common_step_counter+=1;e.episode_length_buf+=1;e.scene['robot'].data.body_pos_w[0,0,:]=torch.tensor([x,.22,z]);e.scene['contact_forces'].data.current_contact_time[0,0]=.2 if touch else 0;e.scene['contact_forces'].data.net_forces_w[0,0]=torch.tensor([0,0,100 if touch else 0.]);context(e,0.,.1);return ns['ascent_state'](e,**p)
step(-.25,.09);step(-.12,.09);step(-.04,.26,False);step(.03,.26,False);step(.03,.19);s=step(.03,.19)
r={'within_interval_physical_count':float(s['front_count']),'within_interval_style_success':float(s['front_successes'])}
assert r['within_interval_physical_count']==0 and r['within_interval_style_success']==0,r
step(.1,.19);s=step(.1,.19);r.update(after_exiting_interval_physical_count=float(s['front_count']),after_exiting_interval_style_success=float(s['front_successes']))
assert r['after_exiting_interval_style_success']==1,r
print(json.dumps(r));Path('.scratch/gait_v3_20261008/reviewer_early_landing.json').write_text(json.dumps(r,indent=2)+'\n')
