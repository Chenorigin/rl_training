#!/usr/bin/env python3
"""Adversarial reward attribution tests; no learned-gait success claim.

Run old and new canonical function bodies with only Isaac imports replaced.
Tests cover pose monotonicity, temporal landing histories and reward wiring.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from check_stair_rewards import Entity, MDP, context, load_functions, make_env, quat_apply, yaw_quat


def load(path):
    ns = dict(torch=torch, np=np, SceneEntityCfg=Entity, quat_apply=quat_apply,
              quat_apply_inverse=lambda q,v: quat_apply(q * torch.tensor([1.,-1.,-1.,-1.]),v),
              yaw_quat=yaw_quat)
    load_functions(path, ns)
    return ns


def check(ns):
    wheels, scanner, contact = Entity('robot'), Entity('height_scanner'), Entity('contact_forces')
    hips, knees = Entity('robot', body_ids=[4,5,6,7]), Entity('robot',joint_ids=[0,1,2,3])
    params=dict(asset_cfg=wheels,hip_cfg=hips,knee_cfg=knees,sensor_cfg=scanner,contact_sensor_cfg=contact)
    results={}
    progress_env=make_env(); context(progress_env)
    results['progress_actual_call']=float(ns['stair_forward_progress'](progress_env,wheels,scanner))
    results['route_actual_call']=float(ns['_stair_route_quality'](progress_env,scanner))
    def pose(a=110., down=False, swing=False, force=100., body_height=.4):
        env=make_env()
        w=env.scene['robot'].data.body_pos_w
        h=w.clone(); h[...,2]+=.42
        env.scene['robot'].data.body_pos_w=torch.cat([w,h],1)
        env.scene['robot'].data.joint_pos=torch.full((1,4),np.deg2rad(180.-a),dtype=torch.float32)
        env.scene['robot'].data.root_ang_vel_b=torch.zeros(1,3)
        env.scene['robot'].data.root_pos_w[:,2]=body_height
        env.scene['height_scanner'].data.ray_hits_w[...,2]=0
        env.scene['contact_forces'].data.current_contact_time.fill_(0. if swing else .02)
        env.scene['contact_forces'].data.net_forces_w[:,:,2]=0. if swing else force
        context(env)
        c=env._m20_stair_scan_context['context']
        c['up_gate'].fill_(float(not down)); c['down_gate'].fill_(float(down))
        return env
    def cost(name,env):
        f=ns.get(name)
        return float(f(env,**params)) if f else None
    for a in (110.,90.,80.,60.4,55.,22.8):
        for swing in (False,True):
            results[f'front_a{a}_{"swing" if swing else "loaded"}']=cost('ascent_front_fold_cost',pose(a=a,swing=swing))
    env=pose(a=60.4,force=61.2)
    results['screenshot_contact_cost']=cost('ascent_front_fold_cost',env)
    for z in (.4,.13,.08):
        results[f'body_root_z{z}_cost']=cost('ascent_front_body_clearance_cost',pose(a=110.,body_height=z))
    env=pose(a=110.,down=True)
    # .45m extended rear legs tilted49deg: old length/knee cost is zero.
    h=env.scene['robot'].data.body_pos_w[:,6:8]
    env.scene['robot'].data.body_pos_w[:,2:4,0]=h[:,:,0]+.45*np.sin(np.deg2rad(49.))
    env.scene['robot'].data.body_pos_w[:,2:4,2]=h[:,:,2]-.45*np.cos(np.deg2rad(49.))
    hits=env.scene['height_scanner'].data.ray_hits_w
    hits[...,2]=env.scene['robot'].data.body_pos_w[0,2,2]-.09
    results['rear_extended_old_fold']=cost('descent_rear_fold_cost',env)
    results['rear_forward49_cost']=cost('descent_rear_forward_cost',env)
    results['rear_vertical_cost']=cost('descent_rear_forward_cost',pose(a=110.,down=True))
    for label,cmd,invalid in [('flat',(0,0,0),False),('yaw',(0,0,.6),False),('reverse',(-.5,0,0),False),('invalid',(.5,0,0),True)]:
        env=pose(a=22.8,swing=True,body_height=.08)
        env.command_manager.get_command('base_velocity')[:]=torch.tensor([cmd])
        if invalid: env.scene['height_scanner'].data.ray_hits_w.fill_(torch.inf)
        for name in ('ascent_front_fold_cost','ascent_front_body_clearance_cost','descent_rear_forward_cost'):
            results[f'{label}_{name}']=cost(name,env)

    def track():
        env=make_env()
        def step(side=None,x=None,z=None,touch=True,edge=0.,height=.1):
            env.common_step_counter+=1
            env.episode_length_buf+=1
            if side is not None:
                env.scene['robot'].data.body_pos_w[0,side,0]=x
                env.scene['robot'].data.body_pos_w[0,side,2]=z
                env.scene['robot'].data.body_lin_vel_w[0,side,2]=0 if touch else .3
                env.scene['contact_forces'].data.current_contact_time[0,side]=float(touch)
                env.scene['contact_forces'].data.net_forces_w[0,side]=torch.tensor([0.,0.,100. if touch else 0.])
            context(env,edge,height)
            return ns['ascent_state'](env,wheels,scanner,contact)
        def land(side,edge,height,scan_edge=None,scan_height=None):
            se=edge if scan_edge is None else scan_edge
            sh=height if scan_height is None else scan_height
            step(side,edge-.12,env.scene['robot'].data.body_pos_w[0,side,2],True,se,sh)
            step(side,edge-.04,height+.15,False,se,sh)
            step(side,edge+.08,height+.15,False,se,sh)
            step(side,edge+.10,height+.09,True,se,sh)
            return step(side,edge+.10,height+.09,True,se,sh)
        step()
        return env,step,land
    for axle in (0,1):
        env,step,land=track()
        land(0,0.,.1); land(1,.3,.2)
        if axle: land(2,0.,.1,scan_edge=.6,scan_height=.3)
        side=2*axle
        step(side,.7,.55,False,.6,.3)
        events=[]
        for i in range(8):
            s=step(side+1,.10,.19,True,.6,.3)
            events.append(bool(s[('front','rear')[axle]+'_same_tread_event']))
        results[f'{("front","rear")[axle]}_delayed_same_tread_events']=sum(events)
        results[f'{("front","rear")[axle]}_delayed_same_tread_per_frame']=events
        # Falling back to a paid old tread cannot get a new event.
        step(side+1,.1,.3,False,.6,.3)
        repeats=[]
        for i in range(4):
            s=step(side+1,.1,.19,True,.6,.3)
            repeats.append(bool(s[('front','rear')[axle]+'_same_tread_event']))
        results[f'{("front","rear")[axle]}_bounce_extra_events']=sum(repeats)
    env,step,land=track(); land(0,0.,.1); land(1,.3,.2)
    step(0,.7,.55,False,.6,.3)
    env.scene['height_scanner'].data.quat_w[0]=torch.tensor([np.cos(np.pi/8),0.,0.,np.sin(np.pi/8)])
    turned=[]
    for _ in range(4):
        s=step(1,.1,.19,True,.6,.3)
        turned.append(bool(s['front_same_tread_event']))
    results['body_turn_cannot_hide_duplicate']=sum(turned)
    env,step,land=track()
    for i,side in enumerate((0,1,0)): land(side,.3*i,.1*(i+1))
    for i,side in enumerate((2,3,2)): land(side,.3*i,.1*(i+1),scan_edge=.9,scan_height=.4)
    s=step(edge=.9,height=.4)
    results['correct_front_successes']=float(s['front_successes'])
    results['correct_rear_successes']=float(s['rear_successes'])
    results['correct_duplicate_events']=float(s['same_tread_events_total'].sum())
    env,step,land=track(); land(0,0.,.1)
    s=land(0,.3,.2)
    results['same_leg_next_step_successes']=float(s['front_successes'])
    results['same_leg_next_step_violations']=float(s['front_violations'])
    env,step,land=track(); land(0,0.,.1)
    # Lift FL out, then FR touches tread1 for one frame only.
    step(0,.5,.4,False,.3,.2)
    s=step(1,.1,.19,True,.3,.2)
    results['one_frame_not_landing']=bool(s['front_same_tread_event'])
    # Face contact without vertical support at identical height must not count.
    env.scene['contact_forces'].data.net_forces_w[0,1]=torch.tensor([100.,0.,1.])
    s=step(edge=.3,height=.2)
    results['horizontal_face_not_landing']=bool(s['front_same_tread_event'])
    # Terminal platform: no higher edge, both legs allowed.
    env,step,land=track(); land(0,0.,.1); step(0,.5,.4,False)
    for _ in range(5): s=step(1,.1,.19,True,0.,.1)
    results['terminal_platform_events']=float(s['same_tread_events_total'].sum())
    # Direct simultaneous-axle event cost should add, not OR collapse.
    state=ns['_new_state'](env)
    state['front_same_tread_event'].fill_(True); state['rear_same_tread_event'].fill_(True)
    env._m20_ascent_cache=dict(step=env.common_step_counter,state=state)
    results['both_axles_event_units']=float(ns['stair_up_same_tread_cost'](env,wheels,scanner,contact)*env.step_dt)
    ns['_reset'](state,torch.tensor([True]))
    if 'tread_seen' in state:
        results['reset_history_empty']=not bool(state['tread_seen'].any())
    return results


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,default=MDP/'stair_teacher.py')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--assert-new',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(1)
    r=check(load(args.source))
    if args.assert_new:
        for phase in ('loaded','swing'):
            vals=[r[f'front_a{a}_{phase}'] for a in (110.,90.,80.,60.4,55.,22.8)]
            assert all(np.isfinite(vals)) and all(x<=y for x,y in zip(vals,vals[1:])), vals
            assert vals[-1]>vals[-2]>vals[0]
        assert r['screenshot_contact_cost']>0
        assert r['body_root_z0.4_cost']==0<r['body_root_z0.13_cost']<r['body_root_z0.08_cost']
        assert r['rear_extended_old_fold']==0 and r['rear_forward49_cost']>0 and r['rear_vertical_cost']==0
        assert r['front_delayed_same_tread_events']==r['rear_delayed_same_tread_events']==1
        assert r['front_bounce_extra_events']==r['rear_bounce_extra_events']==0
        assert r['body_turn_cannot_hide_duplicate']==1
        assert r['correct_front_successes']==r['correct_rear_successes']==3 and r['correct_duplicate_events']==0
        assert r['same_leg_next_step_successes']==1 and r['same_leg_next_step_violations']==1
        assert not r['one_frame_not_landing'] and not r['horizontal_face_not_landing']
        assert r['terminal_platform_events']==0 and r['both_axles_event_units']==2 and r['reset_history_empty']
        for k,v in r.items():
            if k.startswith(('flat_','yaw_','reverse_','invalid_')): assert v==0,(k,v)
    print(json.dumps(r,indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(r,indent=2)+'\n')


if __name__=='__main__': main()
