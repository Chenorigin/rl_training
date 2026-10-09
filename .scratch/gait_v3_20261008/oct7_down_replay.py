from pathlib import Path
import argparse,sys,json
import numpy as np,torch,mujoco
ROOT=Path.cwd();sys.path.insert(0,str(ROOT/'scripts/tools'));sys.path.insert(0,str(ROOT/'deploy/deploy_mujoco'))
from check_gait_safety import load
from check_stair_rewards import make_env,Entity,MDP
import deploy_mujoco as deploy

torch.set_num_threads(1)
robot=Path('/home/ubuntu/桌面/m20_1/M20_perception_Lidar_rl_2/M20_perception_Lidar_rl/deep_robotics_model/M20/mjcf/M20.xml')
a=argparse.Namespace(terrain=None,terrain_xml=ROOT/'deploy/deploy_mujoco/terrains/stairs_descent.xml',stair_height=.15,tread_depth=.3,stair_count=8,stair_start=.5)
m=deploy.load_model(robot,a);c=deploy.M20Contract(m,a.terrain_config);d=mujoco.MjData(m)
p=dict(asset_cfg=Entity('robot'),hip_cfg=Entity('robot',body_ids=[4,5,6,7]),knee_cfg=Entity('robot',joint_ids=[0,1,2,3]),sensor_cfg=Entity('height_scanner'),contact_sensor_cfg=Entity('contact_forces'))
output={}
for label in ('baseline','latest'):
 rows=np.load(ROOT/f'docs/latest_gait_20261008/{label}/down30/telemetry.npz');summary=json.loads((ROOT/f'docs/latest_gait_20261008/{label}/down30/summary.json').read_text())
 e=make_env();e.scene['height_scanner'].ray_starts=torch.tensor(np.column_stack((c.grid,np.zeros(187))),dtype=torch.float32);ns=load(ROOT/'.scratch/ascent_direction_20261008/before/stair_teacher.py')
 costs=[];b=[];g=[];gates=[];loaded=[];force_support=[]
 for i,qpos in enumerate(rows['qpos']):
  d.qpos[:]=qpos;mujoco.mj_kinematics(m,d);c.height_scan(d)
  r=e.scene['robot'].data;r.body_pos_w=torch.tensor(np.concatenate((rows['wheel_pos'][i],rows['hip_pos'][i]))[None],dtype=torch.float32);r.body_lin_vel_w=torch.zeros(1,8,3)
  r.joint_pos=torch.tensor(rows['knee'][i][None],dtype=torch.float32);r.root_quat_w=torch.tensor(qpos[3:7][None],dtype=torch.float32);r.root_pos_w=torch.tensor(rows['root_pos'][i][None],dtype=torch.float32);r.root_ang_vel_b=torch.zeros(1,3)
  r.projected_gravity_b=ns['quat_apply_inverse'](r.root_quat_w,torch.tensor([[0.,0.,-1.]]))
  sc=e.scene['height_scanner'].data;sc.ray_hits_w=torch.tensor(c.last_scan_hits[None],dtype=torch.float32);sc.pos_w=r.root_pos_w.clone();sc.quat_w=r.root_quat_w.clone()
  co=e.scene['contact_forces'].data;co.net_forces_w=torch.tensor(rows['forces'][i][None],dtype=torch.float32);co.current_contact_time=torch.tensor(rows['contact_time'][i][None],dtype=torch.float32)
  e.common_step_counter=i+3;e.episode_length_buf.fill_(i+3);e.command_manager.get_command('base_velocity')[:]=torch.tensor(rows['command'][i][None])
  ns['ascent_state'](e,p['asset_cfg'],p['sensor_cfg'],p['contact_sensor_cfg']);ctx=ns['_context'](e,**p)
  costs.append(float(ns['descent_rear_forward_cost'](e,**p)));b.append(torch.rad2deg(ctx['body_angle'][0,2:]).numpy());g.append(torch.rad2deg(ctx['gravity_angle'][0,2:]).numpy());gates.append(bool(ctx['down']));loaded.append(((co.current_contact_time>0)&(co.net_forces_w[...,2]>5)&(co.net_forces_w[...,2]>.5*torch.linalg.vector_norm(co.net_forces_w,dim=-1))&ctx['ground_valid']&(ctx['clearance'].abs()<=.05)&ctx['down'][:,None])[0,2:].numpy())
  force_support.append(((co.current_contact_time>0)&(co.net_forces_w[...,2]>5)&(co.net_forces_w[...,2]>.5*torch.linalg.vector_norm(co.net_forces_w,dim=-1)))[0,2:].numpy())
  # Production config calls the descending stage after the posture terms.
  ns['_stair_step_completion_events'](e,p['asset_cfg'],p['sensor_cfg'],p['contact_sensor_cfg'])
 costs=np.array(costs);b=np.array(b);g=np.array(g);gates=np.array(gates);loaded=np.array(loaded);force_support=np.array(force_support)
 window=(rows['time']>=summary['descent_entry']['first_front_down_s'])&(rows['time']<=summary['descent_entry']['first_rear_down_s']+1)
 q=window[:,None]&rows['support'][:,2:]&(rows['contact_time'][:,2:]>=.12)
 selected=rows['phase'];bad=((b>35)|(g>25))&force_support&window[:,None]
 metrics={k:float(v) for k,v in ns['gait_quality_metrics'](e,slice(None)).items() if k.startswith(('rear_entry','rear_forward','rear_body','rear_gravity'))}
 output[label]=dict(recorded_frames=len(rows['time']),phase_frames=int(selected.sum()),bad_rear_loaded_samples=int(bad.sum()),bad_samples_with_reward_gate=int((bad&loaded).sum()),
  entry_b_p95_deg=float(np.quantile(b[q],.95)),entry_g_p95_deg=float(np.quantile(g[q],.95)),
  measured_angle_b_max_abs_error_deg=float(abs(b-rows['tilt_body_deg'][:,2:]).max()),measured_angle_g_max_abs_error_deg=float(abs(g-rows['tilt_yaw_deg'][:,2:]).max()),
  rear_forward_cost_rate_phase_mean=float((-ns['TUNING']['rear_forward_weight']*costs[selected]).mean()),rear_forward_cost_rate_entry_mean=float((-ns['TUNING']['rear_forward_weight']*costs[window]).mean()),
  rear_forward_cost_rate_entry_peak=float((-ns['TUNING']['rear_forward_weight']*costs[window]).max()),cost_integral_entry=float((-ns['TUNING']['rear_forward_weight']*costs[window]).sum()*.02),
  invalid_phase_gate_fraction=float((~gates[window]).mean()),metrics=metrics)
print(json.dumps(output,indent=2));(ROOT/'docs/gait_v3_20261008/oct7_down_replay.json').write_text(json.dumps(output,indent=2)+'\n')
