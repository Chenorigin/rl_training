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
p=argparse.Namespace(terrain='descent',terrain_xml=None,stair_height=.15,tread_depth=.30,stair_count=5)
model=deploy.load_model(Path('/home/ubuntu/桌面/m20_1/M20_perception_Lidar_rl_2/M20_perception_Lidar_rl/deep_robotics_model/M20/mjcf/M20.xml'),p)
c=deploy.M20Contract(model,p.terrain_config);data=mujoco.MjData(model);snap=mujoco.MjData(model);c.reset(data)
metrics=GaitMetrics(c);actor=deploy.load_actor(Path('logs/rsl_rl/deeprobotics_m20_stair_teacher/2026-10-09_01-55-35_gait_fix_descent_entry/model_214400.pt'))
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
 reward=float(ns['stair_descent_entry_retract'](e,**params));cost=float(ns['stair_descent_entry_position_cost'](e,**params))
 s=e._m20_descent_entry_state
 ctx=e._m20_gait_context
 rows.append(dict(t=data.time, root_x=float(snap.xpos[c.base_id,0]), root_v=float(r.root_lin_vel_b[0,0]),
  b=torch.rad2deg(ctx['body_angle'][0,2:]).tolist(), g=torch.rad2deg(ctx['gravity_angle'][0,2:]).tolist(),
  body_x=(ctx['body_angle'][0,2:]*0).tolist(), force=force[:,2].tolist(),
  rear_cost=float(ns['descent_rear_forward_cost'](e,**params)),entry_cost=cost,gain=reward*.02,
  entry_active=bool(s['active']),phase_active=bool(e._m20_descent_pose_phase['active']),
  rear_load=ctx['rear_load'][0].tolist(),ground_valid=ctx['ground_valid'][0].tolist(),
  wheel_ground=[c.ground_height(snap,*v[:2]) for v in snap.xpos[c.wheel_body_ids]]))
result=dict(scope='counterfactual production rewards over actual MuJoCo descent; different contact-time adapter',records=rows,
 metrics={k:float(v) for k,v in ns['gait_quality_metrics'](e,slice(None)).items()},
 gain_total=sum(x['gain'] for x in rows), rear_cost_integral=sum(x['rear_cost']*.02 for x in rows))
Path('docs/descent_followup_20261009/replay_after.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
