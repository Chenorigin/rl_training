#!/usr/bin/env python3
"""Counterexamples for actual quiet idle and world-route reward functions."""
from pathlib import Path
from types import SimpleNamespace as NS
import ast,sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp'))
from quiet_landing_signals import idle_stance_cost

def actual_functions(path,names,ns):
 tree=ast.parse(path.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
 assert len(nodes)==len(names)
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
 return ns

def yaw_quat(q):
 w,x,y,z=q.unbind(-1);angle=torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
 out=torch.zeros_like(q);out[:,0]=torch.cos(angle/2);out[:,3]=torch.sin(angle/2);return out

def quat_apply(q,v):
 axis=q[:,1:];twice=2*torch.cross(axis,v,dim=-1)
 return v+q[:,:1]*twice+torch.cross(axis,twice,dim=-1)

def robot_env(angle=0):
 data=NS(root_pos_w=torch.zeros(1,3),root_quat_w=torch.tensor([[np.cos(angle/2),0,0,np.sin(angle/2)]],dtype=torch.float32),root_lin_vel_w=torch.zeros(1,3),root_ang_vel_w=torch.zeros(1,3),joint_pos=torch.zeros(1,16),default_joint_pos=torch.zeros(1,16),joint_vel=torch.zeros(1,16))
 command=torch.zeros(1,3);hits=torch.zeros(1,187,3)
 scene={'robot':NS(data=data)}
 class Scene(dict):pass
 scene=Scene(scene);scene.sensors={'height_scanner':NS(data=NS(ray_hits_w=hits))}
 env=NS(scene=scene,command_manager=NS(get_command=lambda _:command),quiet_extended_tracker=NS(last={'normal_force_N':torch.full((1,4),85.)}),quiet_tracker=NS(age=torch.ones(1)),cfg=NS(quiet_landing=NS(reset_grace_time=.1)),_quiet_leg_joint_ids=list(range(12)),_quiet_wheel_joint_ids=list(range(12,16)),common_step_counter=0,episode_length_buf=torch.tensor([10]),num_envs=1,device='cpu',step_dt=.02)
 return env,data,command,hits

def main():
 torch.set_num_threads(1)
 env,data,cmd,hits=robot_env()
 assert all(idle_stance_cost(env,k).item()==0 for k in ['motion','pose','load'])
 data.root_lin_vel_w[:,0]=.1
 assert idle_stance_cost(env,'motion').item()>0,'Moving must not disable idle objective'
 data.joint_vel[:,12:]=1
 assert idle_stance_cost(env,'pose').item()>0
 env.quiet_extended_tracker.last['normal_force_N'][:]=torch.tensor([[100,70,70,100]])
 assert idle_stance_cost(env,'load').item()>0
 cmd[:,0]=.7
 assert all(idle_stance_cost(env,k).item()==0 for k in ['motion','pose','load']),'Do not constrain traversal'
 cmd.zero_();hits[:,0,2]=.2
 assert all(idle_stance_cost(env,k).item()==0 for k in ['motion','pose','load']),'No flat stance target on stairs'
 hits.zero_();hits[:,0,2]=float('nan')
 assert idle_stance_cost(env,'motion').item()==0,'Unknown terrain fails closed'
 hits.zero_();env.quiet_extended_tracker.last['normal_force_N'][:,0]=0
 assert idle_stance_cost(env,'load').item()>0,'Unloading one wheel must not erase standing load cost'
 assert idle_stance_cost(env,'motion').item()>0,'Lifting a wheel must not evade stopping'
 cfg=lambda name:NS(name=name)
 source=ROOT/'source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/quiet_landing_traversal.py'
 tuning=ast.parse(source.read_text());literal=next(n.value for n in tuning.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='TUNING' for t in n.targets))
 tuning=eval(compile(ast.Expression(literal),str(source),'eval'),{'np':np})
 ns=actual_functions(source,{'straight_route_context','straight_route_cost'},{'torch':torch,'np':np,'SceneEntityCfg':cfg,'quat_apply':quat_apply,'yaw_quat':yaw_quat,'TUNING':tuning})
 probe=ns['straight_route_context']
 costs=[]
 for angle in [0,.6,-1.5]:
  env,data,cmd,hits=robot_env(angle);cmd[:,0]=.7
  assert probe(env)['cost'].item()==0
  env.common_step_counter+=1
  # Rotate the whole scene/reference: cost is invariant to its world heading.
  data.root_pos_w[0,:2]=torch.tensor([-.3*np.sin(angle),.3*np.cos(angle)])
  data.root_lin_vel_w[0,:2]=torch.tensor([-.2*np.sin(angle),.2*np.cos(angle)])
  new=angle+.3;data.root_quat_w[0]=torch.tensor([np.cos(new/2),0,0,np.sin(new/2)])
  value=probe(env);costs.append(value['cost'].item())
  assert value['cross_track'].item()>.29 and value['heading_error'].item()>.29
  assert probe(env) is value,'No double integration in one physics-policy step'
  before=value['cost'].item();env.common_step_counter+=1;data.root_pos_w[:,2]=-.9
  assert np.isclose(probe(env)['cost'].item(),before),'Descent height does not disable route cost'
  env.common_step_counter+=1;cmd[:,2]=.6
  assert probe(env)['cost'].item()==0,'Intentional turn is allowed'
  env.common_step_counter+=1;cmd[:,2]=0
  assert probe(env)['cost'].item()==0,'After explicit turn establish new straight reference'
  env.common_step_counter+=1;env.episode_length_buf[:]=0;data.root_pos_w.zero_()
  assert probe(env)['cost'].item()==0,'Reset cannot inherit route offset'
 assert np.allclose(costs,costs[0],atol=1e-6),costs
 reward_source=ROOT/'source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/rewards.py'
 ns=actual_functions(reward_source,{'track_ang_vel_z_exp','track_ang_vel_z_world_exp'},{'torch':torch,'SceneEntityCfg':cfg,'ManagerBasedRLEnv':object})
 env,data,cmd,_=robot_env();data.projected_gravity_b=torch.tensor([[.5,0,-np.sqrt(.75)]],dtype=torch.float32)
 data.root_ang_vel_b=torch.tensor([[.2,0,0.]])
 # A pitched body's x rotation has a world-z component despite zero body-z.
 data.root_ang_vel_w[:,2]=.1
 a=ns['track_ang_vel_z_exp'](env,.7,'base_velocity');b=ns['track_ang_vel_z_world_exp'](env,'base_velocity',.7)
 assert a.item()==1 and b.item()<1,'Body-z reward misses world-z rotation under pitch'
 print('PASS: idle zero-motion/force bias; stairs/invalid-map gates; rotated world references; descent; turns/resets; body-vs-world angular counterexample')
if __name__=='__main__':main()
