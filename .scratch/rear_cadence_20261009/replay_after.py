import sys,json,argparse
from pathlib import Path
import numpy as np,torch,mujoco
from tensordict import TensorDict
sys.path.insert(0,'scripts/tools');sys.path.insert(0,'deploy/deploy_mujoco')
from check_gait_safety import load
from check_stair_rewards import MDP,make_env,Entity
import deploy_mujoco as deploy
from gait_metrics import GaitMetrics
ns=load(MDP/'stair_teacher.py');torch.set_num_threads(1)
p=argparse.Namespace(terrain='ascent',terrain_xml=None,stair_height=.20,tread_depth=.25,stair_count=20)
model=deploy.load_model(Path('/home/ubuntu/桌面/m20_1/M20_perception_Lidar_rl_2/M20_perception_Lidar_rl/deep_robotics_model/M20/mjcf/M20.xml'),p)
c=deploy.M20Contract(model,p.terrain_config);data=mujoco.MjData(model);snap=mujoco.MjData(model);c.reset(data)
metrics=GaitMetrics(c);actor=deploy.load_actor(Path('logs/rsl_rl/deeprobotics_m20_stair_teacher/2026-10-09_01-55-35_gait_fix_descent_entry/model_214000.pt'))
e=make_env();params=dict(asset_cfg=Entity('robot'),hip_cfg=Entity('robot',body_ids=[4,5,6,7]),knee_cfg=Entity('robot',joint_ids=[0,1,2,3]),sensor_cfg=Entity('height_scanner'),contact_sensor_cfg=Entity('contact_forces'))
e.scene['height_scanner'].ray_starts=torch.tensor(np.c_[c.grid,np.zeros(187)],dtype=torch.float32)
command=np.array([.5,0,0],dtype=np.float32);prev=np.zeros(16,dtype=np.float32);times=np.zeros(4);rows=[]
def t(x):return torch.tensor(np.asarray(x),dtype=torch.float32)[None]
for step in range(700):
 obs=c.observe(data,command,prev)
 with torch.inference_mode():act=actor(TensorDict({'policy':torch.from_numpy(obs)[None]},batch_size=[1])).squeeze(0).numpy()
 for _ in range(4):c.apply_pd(data,act);mujoco.mj_step(model,data)
 prev=act.astype(np.float32)
 mujoco.mj_copyData(snap,model,data);mujoco.mj_kinematics(model,snap)
 c.height_scan(snap)
 r=e.scene['robot'].data;r.body_pos_w=t(np.r_[snap.xpos[c.wheel_body_ids],snap.xpos[metrics.hips]])
 vel=np.zeros((8,3));rv=np.zeros(6)
 for i,bid in enumerate(np.r_[c.wheel_body_ids,metrics.hips]):
  mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,int(bid),rv,0);vel[i]=rv[3:]
 r.body_lin_vel_w=t(vel);r.root_pos_w=t(snap.xpos[c.base_id]);r.root_quat_w=t(snap.xquat[c.base_id])
 mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,c.base_id,rv,1)
 r.root_lin_vel_b=t(rv[3:]);r.root_ang_vel_b=t(rv[:3])
 mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,c.base_id,rv,0);r.root_lin_vel_w=t(rv[3:])
 r.joint_pos=t(snap.qpos[metrics.knees])
 force=metrics.forces(data);support=(force[:,2]>5)&(force[:,2]>.5*np.linalg.norm(force,axis=1))
 times=np.where(support,times+.02,0)
 e.scene['contact_forces'].data.net_forces_w=t(force);e.scene['contact_forces'].data.current_contact_time=t(times)
 e.scene['height_scanner'].data.ray_hits_w=t(c.last_scan_hits)
 e.scene['height_scanner'].data.pos_w=t(snap.xpos[c.base_id]+[0,0,20]);e.scene['height_scanner'].data.quat_w=r.root_quat_w
 e.common_step_counter+=1;e.episode_length_buf+=1
 s=ns['ascent_state'](e,Entity('robot'),Entity('height_scanner'),Entity('contact_forces'))
 slot=int(s['axis_target_slot'][0,1]);last=int(s['axis_last_slot'][0,1])
 rows.append(dict(t=data.time,x=float(snap.xpos[c.base_id,0]),support=support.tolist(),
  gain=float(s['rear_lift_gain']),front=float(s['front_transition_successes']),rear=float(s['rear_transition_successes']),
  expected=int(s['rear_expected']),target=slot,target_z=float(s['map_z'][0,slot]) if slot>=0 else None,
  last_z=float(s['map_z'][0,last]) if last>=0 else None,
  lead=int(s['map_lead'][0,max(slot,0),1]),radius=s['radius'].tolist()))
 if snap.xpos[c.base_id,2]<c.ground_height(snap,*snap.xpos[c.base_id,:2])+.22:break
result=dict(scope='real MuJoCo trajectory, counterfactual production ledger, adapter contact times differ from Isaac',records=rows,
 gain_total=sum(x['gain'] for x in rows),gain_loaded=sum(x['gain'] for x in rows if x['lead']>=0 and x['support'][2+x['lead']]),
 metrics={k:float(v) for k,v in ns['stair_step_completion_metrics'](e,slice(None)).items()})
Path('docs/rear_cadence_20261009/reward_replay_after.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
