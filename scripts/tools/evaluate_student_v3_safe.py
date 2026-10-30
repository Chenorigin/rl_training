#!/usr/bin/env python3
"""Bounded student-only Isaac evaluation; never constructs a teacher or optimizer.

Cases stay on the original 10x30 physical terrain. Terminal records are captured
before Isaac's automatic reset. Changes to fixed-case assignment are process-local.
"""
import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'source/rl_training'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def outcome(success, failed, timed_out):
    return bool(success and not failed and not timed_out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--training-pid', type=int, required=True)
    p.add_argument('--wall-seconds', type=int, default=2400)
    p.add_argument('--reuse-baseline-seed42', type=Path)
    a = p.parse_args()
    import numpy as np
    import torch
    from task_support.student_v3_safe.deployment import validate_resume, source_hashes
    from task_support.student_v3_safe.configuration import TASK
    from task_support.student_v1.configuration import StudentV1PerceptionCfg
    from task_support.student_v1.model_io import actor_from_state
    from task_support.stair_quiet_v1.deployment import execution_contract
    from task_support.stair_quiet_v1.landing import LandingSettings
    from task_support.stair_quiet_v1_safe.settings import SafeSettings
    from task_support.stair_quiet_v1_width.settings import GEOMETRIES, PROFILES

    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'plan.json').exists():
        raise RuntimeError('Use a new output directory; never overwrite a frozen plan')
    os.kill(a.training_pid, 0)
    stat = a.checkpoint.stat()
    saved = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    provenance = validate_resume(saved)
    if stat.st_size != a.checkpoint.stat().st_size or stat.st_mtime_ns != a.checkpoint.stat().st_mtime_ns:
        raise RuntimeError('Checkpoint changed while loading')
    checkpoint_sha = digest(a.checkpoint)
    actor = actor_from_state(saved['student_state_dict'], 789, provenance['training_config']['student'])
    weights = {k: v.clone() for k, v in actor.state_dict().items()}
    control = execution_contract(provenance, required=True)
    manifest = provenance['teacher_manifest']
    base = StudentV1PerceptionCfg(**provenance['perception'])
    settings = SafeSettings(**manifest['noise_refinement']['contract']['environment']['stair_tail'])
    settings.width_curriculum_enabled = False
    settings.automatic_stages = False
    settings.initial_stage = 2
    variants = {
        'baseline': base,
        'low_latency': replace(base, min_delay_s=.04, max_delay_s=.08),
        'clean_height': replace(base, height_noise_std=0., height_bias_std=0., correlated_height_std=0.,
                                map_artifact_probability=0., map_artifact_height_std=0.),
    }
    for cfg in variants.values():
        cfg.validate()
    conditions = [{'env': j*30+t, 'row': row, 'width_m': settings.widths_m[row], 'type': t,
                   'geometry': GEOMETRIES[t//5].name, 'profile': PROFILES[t%5].name}
                  for j, row in enumerate((9, 8, 7, 6)) for t in range(30)]
    reused = []
    reuse_receipt = None
    if a.reuse_baseline_seed42:
        prev = a.reuse_baseline_seed42.resolve()
        prev_plan = json.loads((prev/'plan.json').read_text())
        if prev_plan['checkpoint_sha256'] != checkpoint_sha or prev_plan['source_hashes'] != source_hashes():
            raise RuntimeError('Cannot reuse results from a different checkpoint/source')
        reused = [json.loads(x) for x in (prev/'episodes.jsonl').read_text().splitlines()]
        expected = {(i,j) for i in range(120) for j in range(5)}
        if (len(reused) != 600 or {(x['env'],x['episode']) for x in reused} != expected
                or any(x['variant']!='baseline' or x['seed']!=42 or x.get('incomplete') for x in reused)):
            raise RuntimeError('Only a complete, unique baseline seed42 may be reused')
        reuse_receipt = {'source':str(prev),'episodes_sha256':digest(prev/'episodes.jsonl'),
                         'plan_sha256':digest(prev/'plan.json'),'episodes':600,
                         'limitation':'Baseline seed42 from a separate process; its trajectory was not saved'}
    free, total = torch.cuda.mem_get_info(0)
    available = int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
    estimate = 10*1024**3  # Isaac base <=8GiB +120*4MiB +<1GiB sensor buffers, conservative allowance
    if total-free+estimate > .85*total or available < 24*1024**3:
        raise RuntimeError('Insufficient resource headroom for independent simulation')
    plan = {'checkpoint': str(a.checkpoint.resolve()), 'checkpoint_sha256': checkpoint_sha,
            'checkpoint_iteration': saved['iter'], 'tool_sha256': digest(__file__),
            'source_hashes': source_hashes(), 'task': TASK, 'conditions': conditions,
            'seeds': [42, 43], 'episodes_per_condition_seed': 5, 'max_policy_steps_per_seed_variant': 5500,
            'variants': {k: asdict(v) for k, v in variants.items()},
            'variant_selection': 'baseline all120; ablations reference_15_30_open only20',
            'success': 'original tracker success AND no failure AND no timeout; incomplete quota remains failure',
            'settings': asdict(settings), 'wall_seconds': a.wall_seconds, 'training_pid': a.training_pid,
            'resource': {'gpu_free_bytes': free, 'gpu_total_bytes': total, 'increment_estimate_bytes': estimate,
                         'host_available_bytes': available, 'max_process_rss_bytes': 24*1024**3},
            'teacher_forward_calls': 0, 'optimizer_updates': 0,
            'reuse_baseline_seed42': reuse_receipt,
            'limits': ['six-step training geometry; not a three-step real-scene reconstruction',
                       'same seed does not guarantee paired sensor noise or joint reset RNG',
                       'ablations affect both student and shared landing controller',
                       '10 episodes per condition is screening, not hardware reliability certification']}
    (out/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    (out/'plan.json').chmod(0o444)
    del saved
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, device='cuda:0', device_explicit=True).app
    env = None
    started = time.monotonic()
    records = list(reused)
    trajectories = []
    state = {'collect': False, 'variant': None, 'seed': 42, 'counts': [0]*120, 'selected': set(), 'step': 0}
    inference_context = torch.inference_mode()
    inference_context.__enter__()
    try:
        import gymnasium as gym
        import rl_training.tasks  # noqa: F401
        from isaaclab_tasks.utils import parse_env_cfg
        from task_support.stair_quiet_v1_safe.course import course_state
        from task_support.stair_quiet_v1_width.signals import failure_mask
        cfg = parse_env_cfg(TASK, device='cuda:0', num_envs=120)
        cfg.seed = 42
        cfg.student_perception = base
        cfg.student_history_steps = 4
        cfg.motion_teacher_tuning = deepcopy(manifest['stair_tuning'])
        cfg.landing_control = LandingSettings(**control['settings'])
        cfg.stair_tail = settings
        for case in cfg.scene.terrain.terrain_generator.sub_terrains.values():
            case.settings = settings
        cfg.scene.terrain.visual_material = None
        cfg.scene.sky_light = None
        cfg.commands.base_velocity.debug_vis = False

        def fixed_reset(e, env_ids, pose_range, velocity_range, asset_cfg=None):
            cs = course_state(e)
            cs['material_practice'].zero_()
            e.tail_stage = 2
            ids = env_ids.tolist()
            rows = torch.tensor([conditions[i]['row'] for i in ids], device=e.device)
            tt = torch.tensor([conditions[i]['type'] for i in ids], device=e.device)
            cs['width_curriculum'].level[env_ids] = rows
            ter = e.scene.terrain
            ter.terrain_levels[env_ids], ter.terrain_types[env_ids] = rows, tt
            ter.env_origins[env_ids] = ter.terrain_origins[rows, tt]
            e.scene.env_origins[env_ids] = ter.env_origins[env_ids]
            robot = e.scene['robot' if asset_cfg is None else asset_cfg.name]
            pos = robot.data.default_root_state[env_ids, :3].clone()+ter.env_origins[env_ids]
            # Paired base pose across variants, preserving original reset ranges.
            samples = [np.random.default_rng(np.random.SeedSequence([state['seed'], i, state['counts'][i]])).uniform(-1, 1, 4) for i in ids]
            rnd = torch.tensor(np.asarray(samples), device=e.device, dtype=pos.dtype)
            pos[:, 0] += .05*rnd[:, 0]
            pos[:, 1] += settings.reset_y_range*rnd[:, 1]
            yaw = settings.reset_yaw_range*rnd[:, 2]
            quat = torch.zeros(len(ids), 4, device=e.device)
            quat[:, 0], quat[:, 3] = (yaw/2).cos(), (yaw/2).sin()
            robot.write_root_pose_to_sim(torch.cat((pos, quat), -1), env_ids=env_ids)
            robot.write_root_velocity_to_sim(torch.zeros(len(ids), 6, device=e.device), env_ids=env_ids)
            cs['assign_count'][env_ids] += 1
            course_state(e)

        cfg.events.randomize_reset_base.func = fixed_reset
        env = gym.make(TASK, cfg=cfg).unwrapped
        actor = actor.to(env.device)
        original_reset = env._reset_idx
        fp = (out/'episodes.jsonl').open('w', buffering=1)
        for record in reused:
            fp.write(json.dumps(record)+'\n')
        dims = None
        valid_sum = torch.zeros(120, device=env.device)
        obs_count = torch.zeros(120, device=env.device)
        max_y = torch.zeros(120, device=env.device)

        def assert_cases(e):
            ter, cs = e.scene.terrain, course_state(e)
            rows = torch.tensor([c['row'] for c in conditions], device=e.device)
            tt = torch.tensor([c['type'] for c in conditions], device=e.device)
            assert torch.equal(ter.terrain_levels, rows) and torch.equal(ter.terrain_types, tt), 'Physical case drift'
            assert torch.equal(e.scene.env_origins, ter.terrain_origins[rows, tt]), 'Physical origin drift'
            for name, expected in [('width', [settings.widths_m[c['row']] for c in conditions]),
                                   ('height', [GEOMETRIES[c['type']//5].height for c in conditions]),
                                   ('depth', [GEOMETRIES[c['type']//5].depth for c in conditions]),
                                   ('steps', [GEOMETRIES[c['type']//5].steps for c in conditions])]:
                assert torch.allclose(cs[name].float(), torch.tensor(expected, device=e.device).float()), name+' cache drift'

        def capture(e, env_ids):
            if state['collect']:
                failed = failure_mask(e)
                done = e.termination_manager.terminated | e.termination_manager.time_outs
                local = e.scene['robot'].data.root_pos_w-e.scene.env_origins
                for i in env_ids.tolist():
                    if i not in state['selected'] or state['counts'][i] >= 5 or not bool(done[i]) or e.episode_length_buf[i] <= 0:
                        continue
                    tr = e.tail_tracker
                    timeout = bool(e.termination_manager.time_outs[i])
                    rec = dict(conditions[i], variant=state['variant'], seed=state['seed'],
                               episode=state['counts'][i], success=outcome(bool(tr.success[i]), bool(failed[i]), timeout),
                               tracker_success=bool(tr.success[i]), failed=bool(failed[i]), timeout=timeout,
                               policy_steps=int(e.episode_length_buf[i]), elapsed_s=float(tr.elapsed_s[i]),
                               edge_failed=bool(tr.edge_failed[i]), recovery_failed=bool(tr.recovery_failed[i]),
                               stall=bool(tr.stall_age[i]>settings.no_progress_timeout_s),
                               end_local_xyz=local[i].tolist(), max_abs_y_m=max(float(max_y[i]), abs(float(local[i, 1]))),
                               valid_fraction=float(valid_sum[i]/obs_count[i].clamp_min(1)),
                               termination_terms={k: bool(e.termination_manager.get_term(k)[i]) for k in e.termination_manager.active_terms},
                               **{k: float(getattr(tr, k)[i]) for k in ('force_peak','impulse_peak','slip_seconds','unsafe_seconds','recovery_events','backward_distance_m')})
                    records.append(rec)
                    fp.write(json.dumps(rec)+'\n')
                    state['counts'][i] += 1
            original_reset(env_ids)
            valid_sum[env_ids] = obs_count[env_ids] = max_y[env_ids] = 0
            # The command manager resamples after event reset. Reapply a paired
            # uniform speed within the original [.4,.8] range after all resets.
            cs = course_state(e)
            for i in env_ids.tolist():
                rng = np.random.default_rng(np.random.SeedSequence([state['seed'], i, state['counts'][i]]))
                speed_u = rng.uniform(-1, 1, 4)[3]
                lo, hi = settings.speed_range
                cs['speed'][i] = lo+(speed_u+1)*.5*(hi-lo)
            e.command_manager.get_term('base_velocity')._apply_course(env_ids)
            assert_cases(e)

        env._reset_idx = types.MethodType(capture, env)
        physics_before = env._sim_step_counter
        landing_before = env.stair_quiet_landing.applied_substeps
        receipts = []
        for variant, perception in variants.items():
            for seed in (42, 43):
                if reused and variant == 'baseline' and seed == 42:
                    receipts.append({'variant':variant,'seed':seed,'completed':600,'expected':600,'reused':True})
                    continue
                state.update(collect=False, variant=variant, seed=seed, counts=[0]*120, step=0,
                             selected={c['env'] for c in conditions if variant=='baseline' or c['type']<5})
                env.cfg.student_perception = perception
                for name in ('_v3_5_heading', '_m20_student_v1_state', '_m20_student_v1_delay'):
                    if hasattr(env, name):
                        delattr(env, name)
                env._tail_skip_episode_record = True
                try:
                    obs, _ = env.reset(seed=seed)
                finally:
                    env._tail_skip_episode_record = False
                assert_cases(env)
                from m20_student_v1.environment import sensor_state
                assert sensor_state(env).command_transform is env._v3_5_heading, 'Heading feedback disconnected'
                dims = {k: list(v.shape) for k, v in obs.items()}
                assert dims['policy'] == [120, 789]
                state['collect'] = True
                last_progress = time.monotonic()
                with torch.inference_mode():
                    for step in range(5500):
                        state['step'] = step
                        if step%50 == 0:
                            os.kill(a.training_pid, 0)
                            rss = int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmRSS:')))*1024
                            if time.monotonic()-started > a.wall_seconds or rss > 24*1024**3:
                                raise RuntimeError('Evaluation wall/RSS ceiling reached')
                        action = actor(obs['policy']).clamp(-provenance['training_config']['clip_actions'], provenance['training_config']['clip_actions'])
                        assert torch.isfinite(action).all(), 'Nonfinite student action'
                        valid_sum += (obs['policy'][:,244:431]>.5).float().mean(-1)
                        obs_count += 1
                        local = env.scene['robot'].data.root_pos_w-env.scene.env_origins
                        max_y = torch.maximum(max_y, local[:,1].abs())
                        if step%5 == 0:
                            for i in (0, 4, 60, 64):
                                if i in state['selected'] and state['counts'][i]<5:
                                    trajectories.append([variant, seed, i, state['counts'][i], step,
                                                         *local[i].tolist(), *env.scene['robot'].data.root_quat_w[i].tolist(),
                                                         float((obs['policy'][i,244:431]>.5).float().mean()), *action[i].tolist()])
                        obs, reward, term, timeout, extras = env.step(action)
                        if not torch.isfinite(obs['policy']).all():
                            raise RuntimeError('Nonfinite student observation')
                        if all(state['counts'][i]>=5 for i in state['selected']):
                            break
                        if time.monotonic()-last_progress > 20:
                            print(json.dumps({'variant':variant,'seed':seed,'step':step,'completed':sum(min(state['counts'][i],5) for i in state['selected']),
                                              'expected':5*len(state['selected']),'wall_s':round(time.monotonic()-started)}), flush=True)
                            last_progress = time.monotonic()
                state['collect'] = False
                for i in sorted(state['selected']):
                    for episode in range(state['counts'][i], 5):
                        rec = dict(conditions[i], variant=variant, seed=seed, episode=episode, success=False,
                                   incomplete=True, reason='policy_step_ceiling')
                        records.append(rec); fp.write(json.dumps(rec)+'\n')
                receipts.append({'variant':variant,'seed':seed,'policy_steps':step+1,
                                 'completed':sum(state['counts'][i] for i in state['selected']), 'expected':5*len(state['selected'])})
                np.savez_compressed(out/'trajectories.npz', rows=np.asarray(trajectories, dtype=object))
                (out/'stage_receipts.json').write_text(json.dumps(receipts,indent=2)+'\n')
                print('STAGE_DONE='+json.dumps(receipts[-1]), flush=True)
        fp.close()
        assert digest(a.checkpoint) == checkpoint_sha and source_hashes() == plan['source_hashes']
        assert all(torch.equal(weights[k], v.cpu()) for k,v in actor.state_dict().items())
        assert env.stair_quiet_landing.applied_substeps-landing_before == env._sim_step_counter-physics_before
        np.savez_compressed(out/'trajectories.npz', rows=np.asarray(trajectories, dtype=object))
        verification = {'status':'complete','teacher_forward_calls':0,'optimizer_updates':0,'student_weights_unchanged':True,
                        'checkpoint_unchanged':True,'source_hashes_unchanged':True,'observation_shapes':dims,
                        'physics_substeps':env._sim_step_counter-physics_before,
                        'landing_substeps':env.stair_quiet_landing.applied_substeps-landing_before,
                        'receipts':receipts,'landing_contract':env.stair_quiet_landing.contract(),
                        'landing_summary':env.stair_quiet_landing.summary(),'wall_s':time.monotonic()-started}
        (out/'verification.json').write_text(json.dumps(verification, indent=2)+'\n')
        print('EVALUATION_COMPLETE='+json.dumps({k:v for k,v in verification.items() if k not in ('landing_contract','landing_summary')}), flush=True)
    except BaseException as exc:
        import traceback
        traceback.print_exc()
        (out/'failure.json').write_text(json.dumps({'error':repr(exc),'traceback':traceback.format_exc(),
                                                   'variant':state['variant'],'seed':state['seed'],
                                                   'step':state['step'],'records':len(records)},indent=2)+'\n')
        raise
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == '__main__':
    main()
