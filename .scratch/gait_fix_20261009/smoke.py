#!/usr/bin/env python3
"""Bounded Isaac evaluation of reward wiring. Does not train or save weights."""
import argparse
import hashlib
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
__import__('rl_training.tasks')  # register the actual checkout's task
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
import cli_args
from rl_training.tasks.manager_based.locomotion.velocity.mdp import stair_teacher, stair_ascent, gait_refinement


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
    # Old saved YAML callable paths must resolve to the canonical objects.
    for legacy in (stair_ascent, gait_refinement):
        for name in legacy.__all__:
            assert getattr(legacy, name) is getattr(stair_teacher, name), (legacy.__name__, name)
    canonical_module = stair_teacher.__name__
    unified_names = ('stair_up_step_completion', 'stair_up_rear_step_completion',
                     'stair_descent_rear_fold_cost', 'stair_ascent_front_fold_cost',
                     'stair_ascent_direction_cost',
                     'stair_rear_wheel_lift', 'stair_descent_rear_forward_cost',
                     'stair_ascent_front_body_clearance_cost',
                     'turn_swing_size_cost', 'rotation_gait_status', 'feet_air_time_ang_z_M20')
    unified_bindings = {name: base.reward_manager.get_term_cfg(name).func.__module__ for name in unified_names}
    assert all(module == canonical_module for module in unified_bindings.values()), unified_bindings
    term_peak = torch.zeros(len(base.reward_manager.active_terms), device=base.device)
    binding = {}
    for name in ('stair_descent_rear_fold_cost', 'stair_ascent_front_fold_cost', 'stair_ascent_direction_cost', 'stair_rear_wheel_lift', 'stair_descent_rear_forward_cost', 'stair_ascent_front_body_clearance_cost', 'turn_swing_size_cost', 'rotation_gait_status', 'track_lin_vel_xy_exp'):
        term = base.reward_manager.get_term_cfg(name)
        binding[name] = {}
        for key, entity in term.params.items():
            if not hasattr(entity, 'body_ids'):
                continue
            target = base.scene[entity.name]
            body_ids = entity.body_ids if entity.body_names else None
            joint_ids = entity.joint_ids if entity.joint_names else None
            bodies = ([target.body_names[i] for i in body_ids] if isinstance(body_ids, list) else None)
            joints = ([target.joint_names[i] for i in joint_ids] if isinstance(joint_ids, list) else None)
            binding[name][key] = dict(bodies=body_ids,joints=joint_ids,
                                     resolved_body_names=bodies,resolved_joint_names=joints)
            legs = ('fl','fr','hl','hr')
            expected = ({'asset_cfg':[f'{x}_wheel' for x in legs],
                         'hip_cfg':[f'{x}_hipy' for x in legs],
                         'knee_cfg':[f'{x}_knee_joint' for x in legs],
                         'contact_sensor_cfg':[f'{x}_wheel' for x in legs]}).get(key)
            if expected is not None:
                assert (joints if key == 'knee_cfg' else bodies) == expected, (name,key,binding[name][key])
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
        observed_risers=int(state['map_count'].sum()),
        nominal_radius_finite_fraction=float(torch.isfinite(state['radius']).float().mean()),
        calibrated_wheel_fraction=float(state['radius_calibrated'].float().mean()),
        unified_reward_modules=unified_bindings, legacy_aliases_identical=True,
        active_reward_weights=weights, resolved_gait_entities=binding,
        reward_abs_peaks={name:float(term_peak[i]) for i,name in enumerate(base.reward_manager.active_terms)})
    from rl_training.tasks.manager_based.locomotion.velocity.mdp.stair_teacher import gait_quality_metrics
    result['gait_metrics'] = {key:float(value) for key,value in gait_quality_metrics(base, slice(None)).items()}
    result['ascent_direction_metrics'] = {
        key:float(value) for key,value in stair_teacher.ascent_direction_metrics(base, slice(None)).items()}
    result['step_metrics'] = {key:float(value) for key,value in stair_teacher.stair_step_completion_metrics(base, slice(None)).items()}
    result['gait_tuning'] = stair_teacher.TUNING
    result['canonical_source_sha256'] = hashlib.sha256(Path(stair_teacher.__file__).read_bytes()).hexdigest()
    assert result['observed_risers'] > 0, 'Smoke never observed a riser'
    for tag in ('up_front_strict_coverage','up_rear_strict_coverage','up_rear_lift_credit',
                'up_front_skip_fraction','up_rear_skip_fraction',
                'up_front_same_tread_fraction','up_rear_same_tread_fraction',
                'up_front_transition_successes','up_rear_transition_successes'):
        assert tag in result['step_metrics'],tag
    for tag in ('front_loaded_a_low_fraction','rear_forward_excess_fraction','rear_entry_samples'):
        assert tag in result['gait_metrics'],tag
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print('STAIR_SMOKE_RESULT='+json.dumps(result), flush=True)
    wrapped.close()
    faulthandler.cancel_dump_traceback_later()


try:
    main()
except Exception:
    import traceback, os
    traceback.print_exc()
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(1)
else:
    app.close()
