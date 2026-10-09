import sys,json
from pathlib import Path
sys.path.insert(0,'scripts/tools')
from check_ascent_direction import load
from check_stair_rewards import Entity,make_env
import torch
ns=load();torch.set_num_threads(1);results={}
for offset in (0,.1,.2,.4,1,1.18,2,4):
 env=make_env();env.scene['robot'].data.root_lin_vel_w=torch.tensor([[.5,0,0]])
 def tick():
  env.common_step_counter+=1;env.episode_length_buf+=1
  env._m20_stair_scan_context=dict(counter=env.common_step_counter,sensor_name='height_scanner',context=dict(ascending_edges_valid=torch.tensor([[True]]),ascending_edges_upper_z=torch.tensor([[.15]]),ascending_edges_x=torch.tensor([.25]),down_gate=torch.tensor([0.])))
  return ns['_ascent_direction_context'](env,Entity('robot'),Entity('height_scanner'),Entity('contact_forces'))
 tick();env.scene['robot'].data.root_pos_w[0,1]=offset
 c=tick();results[str(offset)]={'active':bool(c['active']),'cost':float(c['cost']),'weighted':-2*float(c['cost'])}
assert results['0']['cost']==0 and results['0.1']['cost']==0
assert results['0.2']['cost']>0 and results['1']['cost']>results['0.4']['cost']
Path('docs/gait_audit_20261009/direction_numeric.json').write_text(json.dumps(results,indent=2))
print(results)
