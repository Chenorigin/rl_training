#!/usr/bin/env python3
"""Bounded Isaac evaluation of reward wiring. Does not train or save weights."""
import argparse
import json
import sys
import faulthandler
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'source/rl_training'))
sys.path.insert(0, str(ROOT / 'scripts/reinforcement_learning/rsl_rl'))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=Path, required=True)
parser.add_argument('--num_envs', type=int, default=16)
parser.add_argument('--steps', type=int, default=250)
parser.add_argument('--output', type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch
import rl_training.tasks  # register the actual checkout's task
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
import cli_args


def main():
    faulthandler.dump_traceback_later(120, repeat=True)
    torch.set_num_threads(1)
    task = 'Rough-Deeprobotics-M20-StairTeacher-v0'
    cfg = parse_env_cfg(task, device=args.device or 'cuda:0', num_envs=args.num_envs)
    cfg.seed = 42
    # Small atlas, same terrain types and physical robot. Avoid network-only
    # visual assets in this headless interface check.
    cfg.scene.terrain.terrain_generator.num_rows = 2
    cfg.scene.terrain.visual_material = None
    cfg.scene.sky_light = None
    cfg.commands.base_velocity.debug_vis = False
    env = gym.make(task, cfg=cfg)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=100)
    agent = load_cfg_from_registry(task, 'rsl_rl_cfg_entry_point')
    runner = OnPolicyRunner(wrapped, cli_args.convert_rsl_rl_cfg_dict(agent.to_dict()),
                            log_dir=None, device=cfg.sim.device)
    print('STAIR_SMOKE_RUNNER_OK', flush=True)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    runner.alg.actor.load_state_dict(saved['actor_state_dict'], strict=True)
    print('STAIR_SMOKE_ACTOR_LOADED', flush=True)
    policy = runner.get_inference_policy(device=cfg.sim.device)
    obs, _ = wrapped.reset()
    print('STAIR_SMOKE_RESET_OK', flush=True)
    base = env.unwrapped
    term_peak = torch.zeros(len(base.reward_manager.active_terms), device=base.device)
    binding = {}
    for name in ('stair_descent_rear_fold_cost', 'turn_swing_size_cost', 'rotation_gait_status'):
        term = base.reward_manager.get_term_cfg(name)
        binding[name] = {key: {'bodies': cfg.body_ids if cfg.body_names else None,
                              'joints': cfg.joint_ids if cfg.joint_names else None}
                         for key, cfg in term.params.items() if hasattr(cfg, 'body_ids')}
    command_term = base.command_manager.get_term('base_velocity')
    reward_sum = 0.0
    for step in range(args.steps):
        # Exercise both translation and pure-yaw gates without changing PPO or
        # this task's training sampler. This is only a bounded wiring probe.
        command_term.vel_command_b[:] = torch.tensor(
            [0.,0.,.6] if step < args.steps//2 else [.5,0.,0.], device=base.device)
        command_term.is_heading_env[:] = False
        command_term.is_standing_env[:] = False
        with torch.inference_mode():
            action = policy(obs)
            obs, reward, _, _ = wrapped.step(action)
        if not torch.isfinite(action).all() or not torch.isfinite(reward).all():
            raise RuntimeError('Nonfinite action or reward')
        for name, value in obs.items():
            if not torch.isfinite(value).all():
                raise RuntimeError(f'Nonfinite observation group {name}')
        reward_sum += float(reward.mean())
        term_peak = torch.maximum(term_peak, base.reward_manager._step_reward.abs().amax(dim=0))
        if (step + 1) % 25 == 0:
            print(f'STAIR_SMOKE_PROGRESS={step+1}/{args.steps}', flush=True)
    weights = {name: base.reward_manager.get_term_cfg(name).weight for name in base.reward_manager.active_terms}
    state = base._m20_ascent_state
    result = dict(steps=args.steps, num_envs=args.num_envs, trained=False,
        actor_checkpoint=str(args.checkpoint), observation_shapes={k:list(v.shape) for k,v in obs.items()},
        finite=True, mean_return=reward_sum,
        targets_started=float(state['targets_started'].sum()),
        front_lift_credit=float(state['front_lift_credit'].sum()),
        unique_lift_targets=int(state['lift_history_count'].sum()),
        calibrated_wheel_fraction=float(torch.isfinite(state['radius']).float().mean()),
        active_reward_weights=weights, resolved_gait_entities=binding,
        reward_abs_peaks={name:float(term_peak[i]) for i,name in enumerate(base.reward_manager.active_terms)})
    from rl_training.tasks.manager_based.locomotion.velocity.mdp.gait_refinement import gait_quality_metrics
    result['gait_metrics'] = {key:float(value) for key,value in gait_quality_metrics(base, slice(None)).items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print('STAIR_SMOKE_RESULT='+json.dumps(result), flush=True)
    wrapped.close()
    faulthandler.cancel_dump_traceback_later()


try:
    main()
finally:
    app.close()
