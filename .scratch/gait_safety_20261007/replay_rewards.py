"""Evaluate changed rewards on recorded physical trajectories, no policy training.

mj_kinematics reconstructs geometry only. Forces come from saved physical runs.
This attributes rewards to old behavior; it cannot predict new learned behavior.
"""
import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

ROOT=Path.cwd()
sys.path.insert(0,str(ROOT/'deploy/deploy_mujoco'))
sys.path.insert(0,str(ROOT/'scripts/tools'))
import deploy_mujoco as deploy
from check_gait_safety import load
from check_stair_rewards import Entity, make_env, MDP

torch.set_num_threads(1)
robot=Path('/home/ubuntu/桌面/m20_1/M20_perception_Lidar_rl_2/M20_perception_Lidar_rl/deep_robotics_model/M20/mjcf/M20.xml')
ns=load(MDP/'stair_teacher.py')
entities=dict(asset_cfg=Entity('robot'),hip_cfg=Entity('robot',body_ids=[4,5,6,7]),
    knee_cfg=Entity('robot',joint_ids=[0,1,2,3]),sensor_cfg=Entity('height_scanner'),contact_sensor_cfg=Entity('contact_forces'))
results={}
for case in ('stairs_ascent_xml','down30'):
    a=argparse.Namespace(terrain=None if case=='stairs_ascent_xml' else 'descent',
        terrain_xml=ROOT/'deploy/deploy_mujoco/terrains/stairs_ascent.xml' if case=='stairs_ascent_xml' else None,
        stair_height=.15,tread_depth=.3,stair_count=8,stair_start=None)
    # XML values must not be overridden by defaults.
    if case=='stairs_ascent_xml': a.stair_height=a.tread_depth=a.stair_count=None
    m=deploy.load_model(robot,a); c=deploy.M20Contract(m,a.terrain_config); d=mujoco.MjData(m)
    path=ROOT/('docs/stairs_xml_alignment_20261007/telemetry.npz' if case=='stairs_ascent_xml'
               else 'docs/gait_quality_20261007/latest/down30/telemetry.npz')
    rows=np.load(path); env=make_env(); timers=np.zeros(4); weighted=[]; gaps=[]; knee_deg=[]
    ids=np.array([m.body(f'{leg}_{part}').id for part in ('wheel','hipy') for leg in ('fl','fr','hl','hr')])
    qids=np.array([m.joint(f'{leg}_knee_joint').qposadr[0] for leg in ('fl','fr','hl','hr')])
    # No extra free-riser inference: scalar posture attribution uses exact scan.
    env.scene['height_scanner'].ray_starts=torch.tensor(np.column_stack((c.grid,np.zeros(187))),dtype=torch.float32)
    for i,qpos in enumerate(rows['qpos']):
        d.qpos[:]=qpos; mujoco.mj_kinematics(m,d); c.height_scan(d)
        f=rows['force' if case=='stairs_ascent_xml' else 'forces'][i]
        timers=np.where(np.linalg.norm(f,axis=-1)>5,timers+.02,0)
        r=env.scene['robot'].data
        r.body_pos_w=torch.tensor(d.xpos[ids][None],dtype=torch.float32)
        r.body_lin_vel_w=torch.zeros(1,8,3)
        r.joint_pos=torch.tensor(qpos[qids][None],dtype=torch.float32)
        r.root_quat_w=torch.tensor(qpos[3:7][None],dtype=torch.float32)
        r.root_pos_w=torch.tensor(d.xpos[c.base_id][None],dtype=torch.float32)
        r.root_ang_vel_b=torch.zeros(1,3)
        scan=env.scene['height_scanner'].data
        scan.ray_hits_w=torch.tensor(c.last_scan_hits[None],dtype=torch.float32)
        scan.pos_w=r.root_pos_w.clone(); scan.quat_w=r.root_quat_w.clone()
        contact=env.scene['contact_forces'].data
        contact.net_forces_w=torch.tensor(f[None],dtype=torch.float32)
        contact.current_contact_time=torch.tensor(timers[None],dtype=torch.float32)
        env.common_step_counter=i+3;env.episode_length_buf.fill_(i+3)
        env.command_manager.get_command('base_velocity')[:,0]=.7 if case=='stairs_ascent_xml' else .5
        ns['ascent_state'](env,entities['asset_cfg'],entities['sensor_cfg'],entities['contact_sensor_cfg'])
        ns['_stair_step_completion_events'](env,entities['asset_cfg'],entities['sensor_cfg'],entities['contact_sensor_cfg'])
        ctx=ns['_context'](env,**entities)
        vals=[-ns['TUNING']['front_fold_weight']*float(ns['ascent_front_fold_cost'](env,**entities)),
              -ns['TUNING']['body_gap_weight']*float(ns['ascent_front_body_clearance_cost'](env,**entities)),
              -ns['TUNING']['rear_forward_weight']*float(ns['descent_rear_forward_cost'](env,**entities))]
        weighted.append(vals);gaps.append(float(ctx['body_gap'])); knee_deg.append((180.-np.degrees(abs(qpos[qids]))).tolist())
    w=np.array(weighted); valid=rows['phase'].astype(bool)
    r=dict(recorded_frames=len(w),selected_frames=int(valid.sum()),training=False,
           definition='hypothetical changed posture costs on unchanged recorded poses/physical forces; no new policy rollout',
           posture_cost_rate_mean=w[valid].mean(0).tolist(),posture_cost_rate_p95=np.quantile(w[valid],.95,axis=0).tolist(),
           posture_cost_rate_max=w[valid].max(0).tolist(),cost_order=['front_fold','front_body_gap','rear_forward'])
    if case=='stairs_ascent_xml':
        gap=np.array(gaps)[valid]
        r.update(front_body_gap_min_m=float(gap.min()),front_body_gap_p05_m=float(np.quantile(gap,.05)),
                 front_body_gap_below_target_fraction=float((gap<ns['TUNING']['body_gap_target']).mean()))
    results[case]=r
out=ROOT/'docs/gait_safety_20261007/recorded_posture_costs.json'
out.write_text(json.dumps(results,indent=2)+'\n'); print(json.dumps(results,indent=2))
