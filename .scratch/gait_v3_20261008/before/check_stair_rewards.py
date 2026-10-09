#!/usr/bin/env python3
"""CPU behavioral checks of the production stair reward code, without Isaac startup.

Only simulator imports are replaced. The reward/state functions themselves are
compiled from the selected source files. Synthetic tracks test credit assignment,
not whether a learned policy can physically climb stairs.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
MDP = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp"


class Entity:
    def __init__(self, name, body_ids=None, joint_ids=slice(None)):
        self.name, self.body_ids, self.joint_ids = name, list(range(4)) if body_ids is None else body_ids, joint_ids


def quat_apply(q, v):
    t = 2 * torch.cross(q[:, 1:], v, dim=-1)
    return v + q[:, :1] * t + torch.cross(q[:, 1:], t, dim=-1)


def yaw_quat(q):
    yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                      1 - 2 * (q[:, 2].square() + q[:, 3].square()))
    out = torch.zeros_like(q)
    out[:, 0], out[:, 3] = torch.cos(yaw / 2), torch.sin(yaw / 2)
    return out


def load_functions(path, namespace):
    tree = ast.parse(path.read_text())
    constants = [n for n in tree.body if isinstance(n, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == 'TUNING' for target in n.targets)]
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    # Use injected simulator dependencies for the CPU checks. Execute the
    # production parameter table and function bodies from the unified module.
    for fn in functions:
        fn.body = [n for n in fn.body if not (isinstance(n, ast.ImportFrom)
                   and (n.module == 'stair_teacher' or
                        (n.module == 'rewards' and all(alias.name in namespace for alias in n.names))))]
    exec(compile(ast.Module(body=constants + functions, type_ignores=[]), str(path), 'exec'), namespace)


class Scene(dict):
    @property
    def sensors(self):
        return self


def make_env():
    wheel = torch.tensor([[[-.25, .22, .09], [-.25, -.22, .09],
                           [-.95, .22, .09], [-.95, -.22, .09]]])
    robot = SimpleNamespace(data=SimpleNamespace(
        body_pos_w=wheel, body_lin_vel_w=torch.zeros_like(wheel),
        root_pos_w=torch.tensor([[-.45, 0., .4]]), root_quat_w=torch.tensor([[1.,0.,0.,0.]]), root_lin_vel_b=torch.zeros(1, 3),
        projected_gravity_b=torch.tensor([[0., 0., -1.]]),
        joint_acc=torch.zeros(1, 16)))
    xy = torch.cartesian_prod(torch.linspace(-.8, .8, 17), torch.linspace(-.5, .5, 11))
    hits = torch.cat((xy[None] + torch.tensor([[[-.45, 0.]]]), torch.zeros(1,187,1)), -1)
    hits[..., 2] = torch.where(hits[..., 0] >= 0, .1, 0.)
    scanner = SimpleNamespace(ray_starts=torch.cat((xy, torch.zeros(187,1)), -1),
        data=SimpleNamespace(ray_hits_w=hits, pos_w=torch.tensor([[-.45,0.,20.4]]),
                             quat_w=torch.tensor([[1.,0.,0.,0.]])))
    sensor = SimpleNamespace(data=SimpleNamespace(
        net_forces_w=torch.tensor([[[0.,0.,100.]]*4]),
        current_contact_time=torch.ones(1,4)))
    command=torch.tensor([[.5,0.,0.]])
    env=SimpleNamespace(num_envs=1, device='cpu', common_step_counter=2, step_dt=.02,
        episode_length_buf=torch.tensor([2]), scene=Scene(robot=robot,height_scanner=scanner,contact_forces=sensor),
        command_manager=SimpleNamespace(get_command=lambda _:command))
    return env


def context(env, edge=0., height=.1):
    pos=env.scene['height_scanner'].data.pos_w[0,0].item()
    c={k:torch.tensor([v]) for k,v in dict(gate=1.,terrain_gate=1.,up_gate=1.,down_gate=0.,
        edge_x=edge-pos,edge_delta=.1,lower_z=height-.1,upper_z=height,command_x=.5).items()}
    env._m20_stair_scan_context={'counter':env.common_step_counter,'sensor_name':'height_scanner','context':c}


def check(namespace):
    robot, scanner, contacts=Entity('robot'), Entity('height_scanner'), Entity('contact_forces')
    results={}
    def step(env, side=None, x=None, z=None, contact=True, edge=0., height=.1, face=False):
        env.common_step_counter+=1
        if side is not None:
            env.scene['robot'].data.body_pos_w[0,side,0]=x
            env.scene['robot'].data.body_pos_w[0,side,2]=z
            env.scene['robot'].data.body_lin_vel_w[0,side,2]=0 if contact else .3
            env.scene['contact_forces'].data.current_contact_time[0,side]=float(contact)
            env.scene['contact_forces'].data.net_forces_w[0,side]=torch.tensor([-100.,0.,10.] if face else [0.,0.,100. if contact else 0.])
        context(env,edge,height)
        return namespace['ascent_state'](env,robot,scanner,contacts)
    def land(env,side,edge,height):
        step(env,side,edge-.12,env.scene['robot'].data.body_pos_w[0,side,2].item(),edge=edge,height=height)
        step(env,side,edge-.04,height+.15,False,edge,height)
        step(env,side,edge+.08,height+.15,False,edge,height)
        step(env,side,edge+.10,height+.09,True,edge,height)
        return step(env,side,edge+.10,height+.09,True,edge,height)
    env=make_env(); step(env)
    s=step(env,0,-.08,.15,False)
    results['lift_before_landing']=float(namespace['stair_front_wheel_lift'](env,robot,scanner,contacts).item()*env.step_dt)
    # No second payment for the identical lift, or for lowering and lifting again.
    rewards=[]
    for z in [.15,.09,.15]:
        step(env,0,-.08,z,False)
        rewards.append(float(namespace['stair_front_wheel_lift'](env,robot,scanner,contacts).item()*env.step_dt))
    results['repeat_lift_credit']=sum(rewards)
    env=make_env(); step(env)
    step(env,0,.03,.16,False); step(env,0,-.12,.09,True)
    results['retry_after_low_crossing']=float(land(env,0,0.,.1)['front_successes'].item())
    env=make_env(); step(env); step(env,0,-.08,.09,True,face=True)
    step(env,0,-.12,.09,True)
    results['retry_after_face_contact']=float(land(env,0,0.,.1)['front_successes'].item())
    env=make_env(); step(env)
    land(env,0,0.,.1); land(env,1,.3,.2)
    results['third_front_target_active']=bool(step(env,edge=.6,height=.3)['active'].item())
    results['three_alternating_front_landings']=float(land(env,0,.6,.3)['front_successes'].item())
    land(env,2,0.,.1); land(env,3,.3,.2)
    results['alternating_rear_landings']=float(land(env,2,.6,.3)['rear_successes'].item())
    env=make_env(); step(env); land(env,0,0.,.1)
    results['wrong_front_side_no_credit']=float(land(env,0,.3,.2)['front_successes'].item())
    env=make_env(); step(env); land(env,0,0.,.1)
    step(env,1,.10,.19,True,edge=.3,height=.2)
    s=step(env,1,.10,.19,True,edge=.3,height=.2)
    results['same_tread_cost']=bool(s['front_same_tread_event'].item())
    s=step(env,1,.10,.19,True,edge=.3,height=.2)
    results['same_tread_repeat_cost']=bool(s['front_same_tread_event'].item())
    results['same_tread_completion_count']=float(s['front_successes'].item())
    # Old collision/clearance path is not required for the legitimate first lift.
    env=make_env()
    c=namespace['_stair_scan_context'](env,scanner)
    results['real_height_profile_up_gate']=float(c['up_gate'].item())
    for tilt in ['flat','stair']:
        env=make_env(); env.scene['robot'].data.projected_gravity_b[:]=torch.tensor([[.2,0.,-.98]])
        if tilt=='flat': env.scene['height_scanner'].data.ray_hits_w[...,2]=0.
        results[f'{tilt}_orientation_coefficient']=float(namespace['stair_terrain_aware_orientation_cost'](env,robot,scanner).item()/.04)
    if 'stair_forward_motion_cost' in namespace:
        env=make_env(); step(env)
        for vx in [-.2,0.,.5]:
            env.scene['robot'].data.root_lin_vel_b[0,0]=vx
            results[f'motion_cost_vx_{vx}']=float(namespace['stair_forward_motion_cost'](env,robot,scanner).item())
        env.command_manager.get_command('base_velocity')[0,0]=-.5
        results['backward_command_cost']=float(namespace['stair_forward_motion_cost'](env,robot,scanner).item())
        env=make_env(); env.scene['height_scanner'].data.ray_hits_w[...,2]=0
        results['flat_motion_cost']=float(namespace['stair_forward_motion_cost'](env,robot,scanner).item())
        env=make_env(); step(env); env.scene['robot'].data.root_pos_w[0,0]=-5.
        env.common_step_counter+=1; context(env)
        for k in ['gate','terrain_gate','up_gate']: env._m20_stair_scan_context['context'][k].zero_()
        namespace['ascent_state'](env,robot,scanner,contacts)
        results['abandoned_target_cost']=float(namespace['stair_forward_motion_cost'](env,robot,scanner).item())
        env=make_env(); step(env); land(env,0,0.,.1)
        for _ in range(510): step(env)
        results['rear_wait_cursor']=int(env._m20_ascent_state['rear_count'].item())
        env=make_env(); env.scene['robot'].data.body_pos_w[0,0]=torch.tensor([-.01,.22,.19])
        s=step(env)
        results['edge_radius_not_frozen']=bool(torch.isnan(s['radius'][0,0]))
        s=step(env,0,.25,.19,True)
        results['flat_tread_radius']=float(s['radius'][0,0])
        # Exercise A -> abandon -> B -> abandon -> A, without any landing.
        # A target must retain its lift budget even after another target.
        env=make_env(); step(env); step(env,0,-.08,.25,False)
        namespace['stair_front_wheel_lift'](env,robot,scanner,contacts)
        def abandon(e):
            e.scene['robot'].data.root_pos_w[0,0]=-5.
            e.common_step_counter+=1; context(e)
            for k in ['gate','terrain_gate','up_gate']: e._m20_stair_scan_context['context'][k].zero_()
            namespace['ascent_state'](e,robot,scanner,contacts)
        abandon(env)
        for edge in [3.,0.]:
            env.scene['robot'].data.root_pos_w[0,0]=edge-.45
            env.scene['height_scanner'].data.pos_w[0,0]=edge-.45
            env.scene['robot'].data.body_pos_w[0,:2,0]=edge-.25
            env.scene['robot'].data.body_pos_w[0,:2,2]=.09
            env.scene['contact_forces'].data.current_contact_time.fill_(1)
            env.scene['contact_forces'].data.net_forces_w[0,:,:]=torch.tensor([0.,0.,100.])
            step(env,edge=edge); step(env,0,edge-.08,.25,False,edge,.1)
            namespace['stair_front_wheel_lift'](env,robot,scanner,contacts)
            if edge==3.: abandon(env)
        results['two_targets_lift_credit_after_return']=float(env._m20_ascent_state['front_lift_credit'])
        abandon(env)
        angle=torch.tensor(.7)
        heading=torch.tensor([torch.cos(angle),torch.sin(angle)])
        lateral=torch.tensor([-heading[1],heading[0]])
        env.scene['robot'].data.root_pos_w[0,:2]=-heading*.45
        env.scene['height_scanner'].data.pos_w[0,:2]=-heading*.45
        env.scene['height_scanner'].data.quat_w[0]=torch.tensor([torch.cos(angle/2),0.,0.,torch.sin(angle/2)])
        env.scene['robot'].data.body_pos_w[0,0,:2]=-heading*.08+lateral*.22
        env.scene['robot'].data.body_pos_w[0,0,2]=.25
        env.scene['contact_forces'].data.current_contact_time[0,0]=0
        env.scene['contact_forces'].data.net_forces_w[0,0]=0
        env.common_step_counter+=1; context(env)
        env._m20_stair_scan_context['context']['edge_x'].fill_(.45)
        s=namespace['ascent_state'](env,robot,scanner,contacts)
        results['lift_credit_after_turned_return']=float(s['front_lift_credit'])
    # Both axles can occupy different lower treads in a narrow descending
    # flight. Completion must not demand they all park on the same tread.
    env=make_env(); env.scene['robot'].data.body_pos_w[...,2]=.39
    for tick in range(10):
        env.common_step_counter+=1; context(env,0.,.2)
        c=env._m20_stair_scan_context['context']; c['up_gate'].zero_(); c['down_gate'].fill_(1)
        c['lower_z'].fill_(.3); c['edge_delta'].fill_(-.1)
        if tick>=1: env.scene['robot'].data.body_pos_w[0,0]=torch.tensor([.1,.22,.29])
        if tick>=2: env.scene['robot'].data.body_pos_w[0,1]=torch.tensor([.4,-.22,.19])
        if tick>=4: env.scene['robot'].data.body_pos_w[0,2]=torch.tensor([.1,.22,.29])
        if tick>=5: env.scene['robot'].data.body_pos_w[0,3]=torch.tensor([.4,-.22,.19])
        namespace['_stair_step_completion_events'](env,robot,scanner,contacts)
    results['descent_different_treads_completed']=float(env._m20_stair_step_state['down_successes'].item())
    return results


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source-dir',type=Path,default=MDP)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--assert-fixed',action='store_true')
    args=parser.parse_args()
    namespace=dict(torch=torch,np=np,SceneEntityCfg=Entity,quat_apply=quat_apply,
                   yaw_quat=yaw_quat,quat_apply_inverse=lambda q,v:quat_apply(q*torch.tensor([1.,-1.,-1.,-1.]),v))
    load_functions(args.source_dir/'stair_teacher.py',namespace)
    result=check(namespace)
    if args.assert_fixed:
        assert result['lift_before_landing']>0, result
        assert result['repeat_lift_credit']==0, result
        assert result['retry_after_low_crossing']==1, result
        assert result['retry_after_face_contact']==1, result
        assert result['third_front_target_active'], result
        assert result['three_alternating_front_landings']==3, result
        assert result['alternating_rear_landings']==3, result
        assert result['wrong_front_side_no_credit']==1, result
        assert result['same_tread_cost'] and not result['same_tread_repeat_cost'], result
        assert result['same_tread_completion_count']==1, result
        assert abs(result['stair_orientation_coefficient']-5)<1e-4, result
        assert abs(result['flat_orientation_coefficient']-50)<1e-4, result
        assert result['motion_cost_vx_-0.2']>result['motion_cost_vx_0.0']>result['motion_cost_vx_0.5'], result
        assert result['flat_motion_cost']==0 and result['backward_command_cost']==0, result
        assert result['descent_different_treads_completed']==1, result
        assert result['abandoned_target_cost']==0 and result['rear_wait_cursor']==0, result
        assert result['edge_radius_not_frozen'] and abs(result['flat_tread_radius']-.09)<1e-5, result
        assert result['two_targets_lift_credit_after_return']<=2, result
        assert result['lift_credit_after_turned_return']<=1, result
    print(json.dumps(result,indent=2))
    if args.output: args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__': main()
