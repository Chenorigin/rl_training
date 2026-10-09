from pathlib import Path
import sys,json
import torch
sys.path.insert(0,str(Path.cwd()/'scripts/tools'))
from check_gait_safety import load
from check_stair_rewards import make_env,Entity,MDP,context
ns=load(MDP/'stair_teacher.py');torch.set_num_threads(1);e=make_env();p=dict(asset_cfg=Entity('robot'),hip_cfg=Entity('robot',body_ids=[4,5,6,7]),knee_cfg=Entity('robot',joint_ids=[0,1,2,3]),sensor_cfg=Entity('height_scanner'),contact_sensor_cfg=Entity('contact_forces'))
r=e.scene['robot'].data;hip=r.body_pos_w.clone();hip[:,:,2]+=.30;r.body_pos_w=torch.cat((r.body_pos_w,hip),1);r.joint_pos=torch.full((1,4),2.2);r.root_ang_vel_b=torch.zeros(1,3);r.root_lin_vel_w=torch.zeros(1,3)
context(e,0.,.1);a=ns['_context'](e,**p);result={'stair_fold_cost':float(ns['ascent_front_fold_cost'](e,**p))}
# Reproduce a controlled lateral exit onto a flat 3m away. Keep the old ledger,
# but actor scan and force-supported body state are now wholly on flat ground.
e.common_step_counter+=1;e.episode_length_buf+=1;r.body_pos_w[:,:,1]+=3.;r.root_pos_w[:,1]+=3
sc=e.scene['height_scanner'];sc.data.pos_w[:,1]+=3;sc.data.ray_hits_w[:,:,1]+=3;sc.data.ray_hits_w[:,:,2]=0
fresh=ns['_stair_scan_context'](e,p['sensor_cfg']);b=ns['_context'](e,**p)
result.update(flat_scan_up_gate=float(fresh['up_gate']),pending_risers=int(e._m20_ascent_state['map_count']),flat_active=bool(e._m20_ascent_state['active']),flat_posture_up_gate=bool(b['up']),flat_fold_cost=float(ns['ascent_front_fold_cost'](e,**p)))
assert result['flat_fold_cost']==0 and not result['flat_active'] and not result['flat_posture_up_gate'],result
print(json.dumps(result));Path('.scratch/gait_v3_20261008/reviewer_exit_gate.json').write_text(json.dumps(result,indent=2)+'\n')
