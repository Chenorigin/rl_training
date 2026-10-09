#!/usr/bin/env python3
"""Exercise actual ascent-direction reward bodies with adversarial command tracks.

These CPU checks test attribution, coordinate frames and state lifecycles;
they do not claim a newly trained policy follows its commanded route.
"""
import argparse
import ast
import json
from pathlib import Path

import numpy as np
import torch
from check_stair_rewards import Entity, MDP, load_functions, make_env, quat_apply, yaw_quat


def load():
    ns = dict(torch=torch, np=np, SceneEntityCfg=Entity, quat_apply=quat_apply, yaw_quat=yaw_quat)
    load_functions(MDP/'stair_teacher.py',ns)
    return ns


def check(ns):
    entities=dict(asset_cfg=Entity('robot'),sensor_cfg=Entity('height_scanner'),contact_sensor_cfg=Entity('contact_forces'))
    results={}
    def make(cmd=(.5,0,0),yaw=0.):
        env=make_env(); env.scene['robot'].data.root_lin_vel_w=torch.zeros(1,3)
        env.command_manager.get_command('base_velocity')[:]=torch.tensor([cmd])
        pose(env,yaw=yaw)
        return env
    def pose(env,yaw=None,velocity=(.5,0)):
        env.scene['robot'].data.root_lin_vel_w[0,:2]=torch.tensor(velocity)
        if yaw is not None:
            q=torch.tensor([[np.cos(yaw/2),0,0,np.sin(yaw/2)]],dtype=torch.float32)
            env.scene['robot'].data.root_quat_w=q
            env.scene['height_scanner'].data.quat_w=q
    def tick(env,seen=True,down=False,advance=(0,0)):
        env.common_step_counter+=1
        env.scene['robot'].data.root_pos_w[0,:2]+=torch.tensor(advance)
        env.episode_length_buf+=1
        env._m20_stair_scan_context={'counter':env.common_step_counter,'sensor_name':'height_scanner','context':dict(
            ascending_edges_valid=torch.tensor([[seen]]),ascending_edges_upper_z=torch.tensor([[.15]]),ascending_edges_x=torch.tensor([.25]),down_gate=torch.tensor([float(down)]))}
        return ns['_ascent_direction_context'](env,**entities)
    def value(c): return float(c['cost'])
    # Correct forward, side-command and arbitrary yaw all share zero cost.
    for yaw in (0.,np.pi/2,-np.pi/3):
        for vy in (0.,.2,-.2):
            env=make((.5,vy,0),yaw);velocity=(np.cos(yaw)*.5-np.sin(yaw)*vy,np.sin(yaw)*.5+np.cos(yaw)*vy)
            pose(env,velocity=velocity);tick(env)
            v=value(tick(env,advance=tuple(v*.02 for v in velocity)))
            results[f'correct_yaw{yaw:.3f}_vy{vy}']=v;assert v<1e-6
    for sign in (-1,1):
        env=make();tick(env);pose(env,velocity=(.5,sign*.3));c=tick(env,advance=(.01,sign*.15))
        results[f'drift_sign{sign}']=value(c);assert value(c)>0
        env=make();tick(env);pose(env,yaw=sign*np.deg2rad(30),velocity=(.5*np.cos(np.deg2rad(30)),sign*.5*np.sin(np.deg2rad(30))))
        c=tick(env,seen=False,advance=(.01,sign*.15));results[f'co_rotated_body_sign{sign}']=value(c);assert value(c)>0 and bool(c['active'])
    assert abs(results['drift_sign-1']-results['drift_sign1'])<1e-6
    assert abs(results['co_rotated_body_sign-1']-results['co_rotated_body_sign1'])<1e-6
    # A curved commanded route at full/slow speed has no timed along-track error.
    for speed in (.5,.2):
        env=make((.5,.1,.6));pose(env,velocity=(speed,.2*speed));tick(env)
        peak=0.
        for i in range(1,101):
            theta=i*.6*.02;mid=theta-.6*.01
            v=(speed*np.cos(theta)-.2*speed*np.sin(theta),speed*np.sin(theta)+.2*speed*np.cos(theta))
            delta=((speed*np.cos(mid)-.2*speed*np.sin(mid))*.02,(speed*np.sin(mid)+.2*speed*np.cos(mid))*.02)
            pose(env,yaw=theta,velocity=v);peak=max(peak,value(tick(env,advance=delta)))
        results[f'curved_speed{speed}_peak']=peak;assert peak<1e-6
    # Gate controls: unconditionally applying the error would be positive.
    for label,cmd,seen,down in [('flat',(.5,0,0),False,False),('descent',(.5,0,0),False,True),('standing',(0,0,0),True,False),('pure_yaw',(0,0,.6),True,False),('reverse',(-.5,0,0),True,False)]:
        env=make(cmd);pose(env,velocity=(0,.8));v=value(tick(env,seen,down))
        results[label]=v;assert v==0
    env=make();tick(env);pose(env,yaw=.6,velocity=(.4,.3));c=tick(env,seen=False,down=True)
    results['turn_away_scan_not_free']=value(c);assert value(c)>0 and bool(c['active'])
    # Once-per-step cache and scalar statistics must not double-integrate.
    before=env._m20_ascent_direction_state['cross_track'].clone()
    again=ns['_ascent_direction_context'](env,**entities)
    assert torch.equal(before,env._m20_ascent_direction_state['cross_track']) and again is c
    results['cache_preserves_error']=True
    # Reset removes route/metrics; one env must not retain a prior drift.
    env.episode_length_buf[:]=0;pose(env,yaw=.6,velocity=(.5*np.cos(.6),.5*np.sin(.6)))
    c=tick(env);results['reset_cost']=value(c);assert value(c)<1e-6
    # All wheels on upper platform beyond remembered edge end the latch.
    env=make();tick(env);state=env._m20_ascent_direction_state
    wheels=env.scene['robot'].data.body_pos_w
    wheels[:,:,0]=state['edge_xy'][0,0]+.2;wheels[:,:,2]=.24
    c=tick(env,seen=False);results['top_platform_inactive']=not bool(c['active']);assert not bool(c['active'])
    c=tick(env,seen=True);results['next_flight_restarts']=bool(c['active']);assert bool(c['active'])
    # Command change resets cross-track, body rotation does not.
    env=make();tick(env);tick(env,advance=(.01,.2))
    env.command_manager.get_command('base_velocity')[:]=torch.tensor([[.5,.5,0.]])
    pose(env,velocity=(.5,.5));c=tick(env,advance=(.01,.01));results['new_side_command_error']=float(c['cross_track']);assert abs(float(c['cross_track']))<1e-6
    env=make();tick(env);pose(env,velocity=(0,0));env.command_manager.get_command('base_velocity')[:]=torch.tensor([[0.,0.,.6]])
    tick(env);pose(env,yaw=1.,velocity=(0,0));tick(env)
    env.command_manager.get_command('base_velocity')[:]=torch.tensor([[.5,0.,0.]])
    pose(env,velocity=(.5*np.cos(1.),.5*np.sin(1.)));c=tick(env)
    results['after_commanded_turn']=value(c);assert value(c)<1e-6
    costs=[]
    for offset in (.05,.1,.2,.4,2.):
        env=make();tick(env);c=tick(env,advance=(.01,offset));costs.append(value(c))
    results['offset_monotonic_costs']=costs;assert costs[0]==costs[1]==0 and all(b>a for a,b in zip(costs[1:],costs[2:]))
    env=make();tick(env);pose(env,yaw=2.,velocity=(.5,100.));c=tick(env,advance=(0,100.))
    results['extreme_cost']=value(c);assert np.isfinite(value(c)) and value(c)<3*ns["TUNING"]["direction_soft_cap"]+.001
    metrics=ns['ascent_direction_metrics'](env,slice(None));results['metrics']={k:float(v) for k,v in metrics.items()};assert metrics['samples']>0
    # Production binding uses this new cost and keeps pre_teacher independent.
    cfg=MDP.parent/'config/wheeled/deeprobotics_m20/stair_teacher_env_cfg.py'
    tree=ast.parse(cfg.read_text());bindings={target.id:node.value for node in ast.walk(tree) if isinstance(node,ast.Assign) and isinstance(node.value,ast.Call) for target in node.targets if isinstance(target,ast.Name)}
    for name,func in [('stair_ascent_direction_cost','stair_ascent_direction_cost'),('ascent_direction','ascent_direction_metrics')]:
        assert isinstance(bindings[name],ast.Call)
        actual=next(k.value for k in bindings[name].keywords if k.arg=='func');assert actual.attr==func
    results['reward_and_metric_binding']=True
    assert ns['TUNING']['front_fold_weight']==-1 and ns['TUNING']['same_tread_event_weight']==-1.5
    results['task_preservation_weights_retained']=True
    return results


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);args=parser.parse_args()
    torch.set_num_threads(1)
    results=check(load())
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))
