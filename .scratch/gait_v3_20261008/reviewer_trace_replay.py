from pathlib import Path
import sys,json,argparse
from types import SimpleNamespace
import numpy as np,torch,mujoco
ROOT=Path.cwd();sys.path.insert(0,str(ROOT/'scripts/tools'));sys.path.insert(0,str(ROOT/'deploy/deploy_mujoco'))
from check_stair_rewards import Entity,make_env
from check_gait_safety import load
import deploy_mujoco as deploy

torch.set_num_threads(1)
model=Path('/home/ubuntu/桌面/m20_1/M20_perception_Lidar_rl_2/M20_perception_Lidar_rl/deep_robotics_model/M20/mjcf/M20.xml')
a=argparse.Namespace(terrain=None,terrain_xml=ROOT/'deploy/deploy_mujoco/terrains/stairs_ascent.xml',stair_height=.15,tread_depth=.3,stair_count=8,stair_start=.25)
m=deploy.load_model(model,a);c=deploy.M20Contract(m,a.terrain_config);d=mujoco.MjData(m)
rows=np.load(ROOT/'docs/latest_gait_20261008/latest/up30/telemetry.npz')
configs={'pre_oct7':ROOT/'.scratch/gait_v3_20261008/before/stair_teacher.py','oct7':ROOT/'.scratch/ascent_direction_20261008/before/stair_teacher.py'}
result={}
p=dict(asset_cfg=Entity('robot'),hip_cfg=Entity('robot',body_ids=[4,5,6,7]),knee_cfg=Entity('robot',joint_ids=[0,1,2,3]),sensor_cfg=Entity('height_scanner'),contact_sensor_cfg=Entity('contact_forces'))
for label,path in configs.items():
 ns=load(path);e=make_env();e.scene['height_scanner'].ray_starts=torch.tensor(np.column_stack((c.grid,np.zeros(187))),dtype=torch.float32)
 every=[];events=[];cost=[];radius=[];supports=[]
 for i,qpos in enumerate(rows['qpos']):
  d.qpos[:]=qpos;mujoco.mj_kinematics(m,d);c.height_scan(d)
  r=e.scene['robot'].data;r.body_pos_w=torch.tensor(np.concatenate((rows['wheel_pos'][i],rows['hip_pos'][i]))[None],dtype=torch.float32)
  r.body_lin_vel_w=torch.zeros(1,8,3)
  if i:r.body_lin_vel_w[:,:4]=torch.tensor((rows['wheel_pos'][i]-rows['wheel_pos'][i-1])[None]/.02,dtype=torch.float32)
  r.joint_pos=torch.tensor(rows['knee'][i][None],dtype=torch.float32);r.root_quat_w=torch.tensor(qpos[3:7][None],dtype=torch.float32)
  r.root_pos_w=torch.tensor(rows['root_pos'][i][None],dtype=torch.float32);r.root_ang_vel_b=torch.zeros(1,3)
  sc=e.scene['height_scanner'].data;sc.ray_hits_w=torch.tensor(c.last_scan_hits[None],dtype=torch.float32);sc.pos_w=r.root_pos_w.clone();sc.quat_w=r.root_quat_w.clone()
  co=e.scene['contact_forces'].data;co.net_forces_w=torch.tensor(rows['forces'][i][None],dtype=torch.float32);co.current_contact_time=torch.tensor(rows['contact_time'][i][None],dtype=torch.float32)
  e.common_step_counter=i+3;e.episode_length_buf.fill_(i+3);e.command_manager.get_command('base_velocity')[:]=torch.tensor(rows['command'][i][None])
  state=ns['ascent_state'](e,p['asset_cfg'],p['sensor_cfg'],p['contact_sensor_cfg'])
  ctx=ns['_context'](e,**p)
  radius.append(torch.isfinite(state['radius']).numpy()[0]);supports.append(ctx['front_support'].numpy()[0])
  cost.append(float(ns['ascent_front_fold_cost'](e,**p)))
  every.append([int(state['front_count']),int(state['rear_count']),bool(state['active']),float(state['target_z'])])
  if state['front_event'].any() or state['rear_event'].any() or state['front_violation_event'].any() or state['rear_violation_event'].any():
   events.append(dict(time=float(rows['time'][i]),front=int(state['front_count']),rear=int(state['rear_count']),fe=state['front_event'][0].tolist(),re=state['rear_event'][0].tolist(),fv=bool(state['front_violation_event']),rv=bool(state['rear_violation_event']),edge_xy=state['edge_xy'][0].tolist(),target_z=float(state['target_z']),wheel_xyz=r.body_pos_w[0,:4].tolist(),front_safe=state['front_safe'][0].tolist(),front_failed=state['front_failed'][0].tolist(),front_contact_age=state['front_physical_time'][0].tolist()))
 vals=np.array(every);valid=rows['phase'];radius=np.array(radius)
 result[label]={'front_count':int(state['front_count']),'front_successes':float(state['front_successes']),'rear_count':int(state['rear_count']),'rear_successes':float(state['rear_successes']),
 'unresolved_target_active':bool(state['active']),'radius_finite_fraction':radius[valid].mean(0).tolist(),'same_tread_events':state['same_tread_events_total'][0].tolist(),
 'actual_saved_front_first_landing_events':12,'actual_saved_rear_first_landing_events':14,'event_trace':events,'front_fold_condition_mean':float(np.array(cost)[valid].mean()),
 'front_gated_samples':int(np.array(supports)[valid].sum())}
print(json.dumps(result,indent=2));(ROOT/'.scratch/gait_v3_20261008/reviewer_trace_replay.json').write_text(json.dumps(result,indent=2)+'\n')
