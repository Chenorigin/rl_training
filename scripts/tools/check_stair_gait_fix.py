#!/usr/bin/env python3
"""Execute the canonical reward bodies against counterexamples; no gait success claim."""
import argparse,json
from pathlib import Path
import torch
import numpy as np
from check_gait_safety import load
from check_stair_rewards import MDP,Entity,make_env,context

def check(ns):
    result={}
    params=dict(asset_cfg=Entity('robot'),hip_cfg=Entity('robot',body_ids=[4,5,6,7]),
                knee_cfg=Entity('robot',joint_ids=[0,1,2,3]),sensor_cfg=Entity('height_scanner'),
                contact_sensor_cfg=Entity('contact_forces'))
    def pose(angle,force,down=True):
        e=make_env();r=e.scene['robot'].data
        hips=r.body_pos_w.clone();hips[...,2]+=.45
        r.body_pos_w=torch.cat([r.body_pos_w,hips],1);r.joint_pos=torch.zeros(1,4);r.root_ang_vel_b=torch.zeros(1,3)
        r.body_pos_w[:,2:4,0]=hips[:,2:4,0]+.45*np.sin(np.deg2rad(angle))
        r.body_pos_w[:,2:4,2]=hips[:,2:4,2]-.45*np.cos(np.deg2rad(angle))
        sensor=e.scene['contact_forces'].data
        sensor.net_forces_w[:,2:,2]=force;sensor.current_contact_time[:,2:]=.02 if force else 0
        context(e);q=e._m20_stair_scan_context['context'];q['down_gate'].fill_(float(down));q['up_gate'].fill_(0)
        return e
    for angle in (-49,0,24,30,49,70):
        for force in (0,6,15,100):
            result[f'rear_{angle}deg_{force}N']=float(ns['descent_rear_forward_cost'](pose(angle,force),**params))
    for label,cmd in [('flat',(.5,0,0)),('yaw',(0,0,.6)),('reverse',(-.5,0,0))]:
        e=pose(70,100,down=False if label=='flat' else True)
        e.command_manager.get_command('base_velocity')[:]=torch.tensor([cmd])
        result[label]=float(ns['descent_rear_forward_cost'](e,**params));assert result[label]==0
    assert result['rear_-49deg_100N']==result['rear_0deg_100N']==0
    assert result['rear_49deg_0N']>0 and result['rear_30deg_6N']>0
    # Missing scan cannot excuse the last rear transfer. Real lower platform support clears it.
    e=pose(49,100);ns['descent_rear_forward_cost'](e,**params)
    e.common_step_counter+=1;e.episode_length_buf+=1;context(e)
    e._m20_stair_scan_context['context']['down_gate'].fill_(0);e._m20_stair_scan_context['context']['up_gate'].fill_(0)
    result['rear_transfer_scan_missing']=float(ns['descent_rear_forward_cost'](e,**params));assert result['rear_transfer_scan_missing']>0
    e.scene['robot'].data.body_pos_w[:,2:4,0]=.3;e.scene['robot'].data.body_pos_w[:,2:4,2]=.09
    e.common_step_counter+=1;e.episode_length_buf+=1;context(e)
    e._m20_stair_scan_context['context']['down_gate'].fill_(0);e._m20_stair_scan_context['context']['up_gate'].fill_(0)
    ns['descent_rear_forward_cost'](e,**params)
    result['lower_platform_clears_latch']=not bool(e._m20_descent_pose_phase['active']);assert result['lower_platform_clears_latch']
    metrics=ns['gait_quality_metrics'](e,slice(None))
    assert 'hl_loaded_b_p95_upper_deg' in metrics and 'hr_swing_g_p95_upper_deg' in metrics
    result['histogram_metrics_wired']=True
    # Degree-bin quantile cannot exceed physical angle upper bound.
    assert all(0<=float(v)<=90 for k,v in metrics.items() if 'p95_upper' in k)
    score=ns['rear_target_score'];tx=torch.tensor([.1]);tz=torch.tensor([.2])
    scores=[float(score(torch.tensor([x]),torch.tensor([z]),tx,tz)) for x,z in [(-.4,0.),(-.2,.1),(.1,.2),(.7,.2),(.1,.7)]]
    result['rear_target_scores']=scores
    assert scores[0]<scores[1]<scores[2] and scores[3]<scores[2] and scores[4]<scores[2]
    for offset in (.2,1.,2.,4.):
        e=make_env();e.scene['robot'].data.root_lin_vel_w=torch.tensor([[.5,0,0.]])
        def tick():
            e.common_step_counter+=1;e.episode_length_buf+=1
            e._m20_stair_scan_context=dict(counter=e.common_step_counter,sensor_name='height_scanner',context=dict(
                ascending_edges_valid=torch.tensor([[True]]),ascending_edges_upper_z=torch.tensor([[.15]]),
                ascending_edges_x=torch.tensor([.25]),down_gate=torch.tensor([0.])))
            return ns['_ascent_direction_context'](e,Entity('robot'),Entity('height_scanner'),Entity('contact_forces'))
        tick();e.scene['robot'].data.root_pos_w[0,1]=offset;c=tick()
        result[f'direction_{offset}m']=float(c['cost'])
    assert result['direction_2.0m']-result['direction_1.0m']>.5
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    torch.set_num_threads(1);r=check(load(MDP/'stair_teacher.py'))
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
