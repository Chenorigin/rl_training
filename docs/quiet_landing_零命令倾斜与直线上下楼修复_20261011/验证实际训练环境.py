from pathlib import Path
import sys,argparse,json,faulthandler,traceback,os
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'source/rl_training'))
sys.path.insert(0,str(ROOT/'scripts/reinforcement_learning/rsl_rl'))
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);AppLauncher.add_app_launcher_args(p);args=p.parse_args()
args.output.unlink(missing_ok=True)
app=AppLauncher(args).app
import gymnasium as gym
import torch
import rl_training.tasks
from isaaclab_tasks.utils import parse_env_cfg,load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
from cli_args import convert_rsl_rl_cfg_dict
from rl_training.tasks.manager_based.locomotion.velocity.mdp.quiet_landing_signals import idle_stance_cost

def main():
 faulthandler.dump_traceback_later(180,repeat=False);torch.set_num_threads(1)
 task='Rough-Deeprobotics-M20-QuietLanding-V1_5-v0';cfg=parse_env_cfg(task,device='cuda:0',num_envs=16)
 cfg.seed=42;cfg.scene.terrain.terrain_generator.num_rows=2;cfg.scene.terrain.terrain_generator.num_cols=4
 for name,terrain in cfg.scene.terrain.terrain_generator.sub_terrains.items():terrain.proportion=1.0 if name=='flat' else 0.0
 cfg.scene.terrain.visual_material=None;cfg.scene.sky_light=None;cfg.commands.base_velocity.debug_vis=False
 env=gym.make(task,cfg=cfg);wrapped=RslRlVecEnvWrapper(env,clip_actions=100);base=env.unwrapped
 agent=load_cfg_from_registry(task,'rsl_rl_cfg_entry_point');runner=OnPolicyRunner(wrapped,convert_rsl_rl_cfg_dict(agent.to_dict()),log_dir=None,device='cuda:0')
 saved=torch.load(args.checkpoint,map_location='cpu',weights_only=True);runner.alg.actor.load_state_dict(saved['actor_state_dict'],strict=True)
 policy=runner.get_inference_policy(device='cuda:0');obs,_=wrapped.reset();cmd=base.command_manager.get_term('base_velocity')
 peaks={k:0. for k in ['quiet_idle_motion','quiet_idle_pose','quiet_idle_load','quiet_straight_route']};seenlogs=set()
 # A short binding check, not a locomotion performance evaluation.
 for step in range(120):
  cmd.vel_command_b.zero_()
  if step>=60:cmd.vel_command_b[:,0]=.7
  obs=wrapped.get_observations()
  with torch.inference_mode():action=policy(obs);obs,reward,done,info=wrapped.step(action)
  assert torch.isfinite(reward).all()
  for name in peaks:
   term=base.reward_manager.get_term_cfg(name);value=term.func(base,**term.params);assert torch.isfinite(value).all();peaks[name]=max(peaks[name],float(value.max()))
  seenlogs.update(info.get('log',{}))
  if step in [59,119]:
   with torch.inference_mode():
    ids=torch.arange(base.num_envs,device=base.device);base._reset_idx(ids);seenlogs.update(base.extras.get('log',{}))
  if step%30==0:print('STEP',step,'peaks',peaks,flush=True)
 assert all(peaks[k]>0 for k in peaks),peaks
 for key in ['QuietReward/quiet_idle_motion','QuietReward/quiet_idle_pose','QuietReward/quiet_idle_load','QuietReward/quiet_straight_route','QuietStand/idle_flat_fraction','QuietStand/diagonal_abs_mean','QuietRoute/cross_track_abs_mean_m']:
  assert key in seenlogs,key
 # Logging remains connected during a zero-weight route ablation.
 term=base.reward_manager.get_term_cfg('quiet_straight_route');term.weight=0.0;base.reward_manager.set_term_cfg('quiet_straight_route',term)
 with torch.inference_mode():
  cmd.vel_command_b.zero_();cmd.vel_command_b[:,0]=.7;obs=wrapped.get_observations()
  wrapped.step(policy(obs))
 assert base._quiet_straight_route_cache['step']==base.common_step_counter
 result={'actual_task':task,'actor_dim':base.observation_manager.group_obs_dim['policy'],'action_dim':base.action_manager.total_action_dim,'reward_binding':{k:base.reward_manager.get_term_cfg(k).func.__module__+'.'+base.reward_manager.get_term_cfg(k).func.__name__ for k in peaks},'yaw_binding':base.reward_manager.get_term_cfg('track_ang_vel_z_exp').func.__name__,'old_ascent_weight':base.reward_manager.get_term_cfg('stair_ascent_direction_cost').weight if 'stair_ascent_direction_cost' in base.reward_manager.active_terms else 0,'cost_peaks':peaks,'new_logs':sorted(k for k in seenlogs if k.startswith(('QuietStand/','QuietRoute/'))),'command_sampling':{'zero':cfg.commands.base_velocity.rel_zero_vel_envs,'only_linear_x':cfg.commands.base_velocity.rel_only_lin_x_envs},'zero_route_weight_logging':True,'steps':121,'training':False,'performance_acceptance':False}
 args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2));print('ACTUAL_QUIET_STABILITY_WIRING_PASS',flush=True);env.close()
try:main()
except BaseException:
 traceback.print_exc()
 sys.stderr.flush();sys.stdout.flush();os._exit(1)
finally:app.close(wait_for_replicator=False)
