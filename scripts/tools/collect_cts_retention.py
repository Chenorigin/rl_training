#!/usr/bin/env python3
"""Collect complete healthy expert rollouts through the existing MuJoCo deployment.

Inference only. Seeds 42/43 are the training/held-out data split; performance
evaluation must use different seeds. Failed rollouts never produce training data.
"""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'source/rl_training'), str(ROOT / 'deploy/deploy_mujoco')]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-checkpoint', type=Path, required=True)
    parser.add_argument('--profile-checkpoint', type=Path, required=True, help='CTS sensor/control provenance')
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--report-dir', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=600)
    parser.add_argument('--startup-steps', type=int, default=50, help='Per-rollout command-onset samples to balance against later motion')
    args = parser.parse_args()
    if not 100 <= args.steps <= 1000:
        parser.error('Collection is bounded to 100..1000 policy steps per case')
    if not 1 <= args.startup_steps < min(300, args.steps):
        parser.error('startup-steps must leave later-motion samples in both source datasets')
    if args.output_dir.exists():
        parser.error('Use a new output directory; existing runtime inputs are never overwritten')

    import numpy as np
    import psutil
    import torch
    import deploy_cts_student as deploy
    from task_support.cts.model import build_models
    from task_support.cts.auxiliary import command_contract
    from task_support.student_v1.retention import CLASSES, checked_manifest

    # Observed CPU deployment peak < 1 GiB. Reserve 2 GiB, including the full
    # 10*steps*(789+16)*4 dataset and serialization copies. No CUDA allocation.
    torch.set_num_threads(1)
    host = psutil.virtual_memory()
    if host.total-host.available+2*1024**3 > .85*host.total:
        raise RuntimeError('Insufficient whole-host headroom for serial CPU collection')
    args.report_dir.mkdir(parents=True, exist_ok=True)
    if psutil.disk_usage(args.report_dir).free < 256*1024**2:
        raise RuntimeError('Insufficient output disk headroom')
    plan_path = args.report_dir / 'collection_plan.json'
    if plan_path.exists():
        raise RuntimeError('Use a new report directory; never overwrite a frozen plan')
    profile = torch.load(args.profile_checkpoint, map_location='cpu', weights_only=True)
    # Use the base actually bound to this profile checkpoint, so changing the
    # current training default cannot silently change a repeated collection.
    cfg = profile['cts']['config']['auxiliary']
    _, prior_payload, prior = checked_manifest(cfg['retention_manifest'], cfg['retention_sha256'])
    if digest(args.reference_checkpoint) != prior['source_checkpoint_sha256']:
        raise ValueError('Reference must be the previously screened healthy retention expert')
    commands, _ = command_contract(cfg)
    meta = deploy.validate_metadata(profile['cts'], resume=False)
    reference = torch.load(args.reference_checkpoint, map_location='cpu', weights_only=True)
    models = build_models(profile['initialization']['teacher'], reference, meta['config']['algorithm'])
    actor = deploy.CTSStudentActor(models.student_export()).eval().requires_grad_(False)
    actor.student_provenance = meta
    control = meta['quiet_execution_control']
    plan = dict(reference_checkpoint=str(args.reference_checkpoint.resolve()),
                reference_sha256=digest(args.reference_checkpoint),
                profile_checkpoint=str(args.profile_checkpoint.resolve()),
                profile_sha256=digest(args.profile_checkpoint),
                model=str(args.model.resolve()), model_sha256=digest(args.model),
                collector_sha256=digest(__file__), steps=args.steps, commands=commands.tolist(),
                original_manifest_sha256=cfg['retention_sha256'],
                original_payload_sha256=prior['payload_sha256'],
                startup_steps=args.startup_steps,
                seeds={'train': 42, 'heldout': 43}, classes=list(CLASSES),
                noise_profile='student-v1', physics=True, optimizer_updates=0,
                acceptance='Every case completes all steps without the original deployment fall; finite observations/actions; every policy step retained, including startup.',
                resources=dict(estimated_peak_bytes=2*1024**3, host_available_bytes=host.available,
                               cpu_threads=1, gpu=False, serial=True))
    plan_path.write_text(json.dumps(plan, indent=2)+'\n')
    plan_path.chmod(0o444)
    rows, labels, classes, heldout, results = [], [], [], [], []

    class Recorder(torch.nn.Module):
        uses_motion_control = True

        def forward(self, obs):
            action = actor(obs)
            rows.append(obs['policy'][0].detach().cpu().numpy().copy())
            labels.append(action[0].detach().cpu().numpy().copy())
            return action

    recording_actor = Recorder().eval()
    for index, name in enumerate(CLASSES):
        for seed in (42, 43):
            start = len(rows)
            case = SimpleNamespace(checkpoint=args.profile_checkpoint.resolve(), model=args.model.resolve(),
                terrain='flat', terrain_xml=None, stair_height=None, tread_depth=None, stair_count=None,
                stair_start=None, noise_profile='student-v1', profile_file=None, noise_level=None, seed=seed,
                steps=args.steps, command_x=float(commands[index, 0]), command_y=float(commands[index, 1]),
                command_yaw=float(commands[index, 2]), viewer=False, live_plot=False, output=None,
                snapshot=None, stop_on_fall=True)
            torch.manual_seed(seed)
            np.random.seed(seed)
            factory = lambda a, c: deploy.StudentPerception(c, a, meta)
            with patch.object(deploy.quiet, 'load_policy', return_value=(recording_actor, control)), \
                    patch.object(deploy.quiet, 'make_noise_adapter', side_effect=factory), \
                    patch.object(deploy.quiet.baseline, 'heading_controller', deploy.sensor_heading_controller), \
                    contextlib.redirect_stdout(io.StringIO()):
                result = deploy.quiet.main(case)
            record = dict(case=name, seed=seed, samples=len(rows)-start, fell=result['fell'],
                          duration_s=result['duration_s'], controller=result['controller_execution'])
            results.append(record)
            (args.report_dir/'collection_results.json').write_text(json.dumps(results, indent=2)+'\n')
            print(json.dumps(record), flush=True)
            if result['fell'] or len(rows)-start != args.steps:
                raise RuntimeError('Rejected expert rollout; no dataset published')
            classes.extend([index]*args.steps)
            heldout.extend([seed == 43]*args.steps)

    arrays = dict(observations=np.asarray(rows, dtype=np.float32), labels=np.asarray(labels, dtype=np.float32),
                  classes=np.asarray(classes, dtype=np.int64), holdout=np.asarray(heldout, dtype=np.bool_))
    # Original data already cover command onset after standing and two seconds
    # of policy history. Keep them; direct-entry history-zero rollouts extend
    # coverage rather than replacing that different, useful handover path.
    with np.load(prior_payload, allow_pickle=False) as original:
        original_count = len(original['classes'])
        startup = list(prior.get('sampling', {}).get('startup_indices', []))
        if not startup:
            for index in range(len(CLASSES)):
                for split in (False, True):
                    indices = np.flatnonzero((original['classes'] == index) & (original['holdout'] == split))
                    if len(indices) <= args.startup_steps:
                        raise RuntimeError('Original retention split has no later-motion remainder')
                    startup.extend(indices[:args.startup_steps].tolist())
        for start in range(original_count, original_count+len(rows), args.steps):
            startup.extend(range(start, start+args.startup_steps))
        arrays = {key: np.concatenate((original[key], value)) for key, value in arrays.items()}
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise RuntimeError('Nonfinite expert data; no dataset published')
    args.output_dir.mkdir(parents=True)
    payload = args.output_dir/'samples.npz'
    np.savez_compressed(payload, **arrays)
    manifest = args.output_dir/'manifest.json'
    manifest.write_text(json.dumps(dict(schema='m20_nonforward_retention_v1', obs_dim=789, action_dim=16,
        classes=list(CLASSES), sample_count=len(arrays['classes']), payload=payload.name, payload_sha256=digest(payload),
        source_checkpoint=str(args.reference_checkpoint.resolve()), source_checkpoint_sha256=digest(args.reference_checkpoint),
        label_semantics='healthy expert nominal actions before clipping or actuator delay; original stood/handover history plus complete direct-entry history-zero episodes',
        collection='original interactive deployment data retained, extended through deploy_quiet_landing.main physics, student-v1 sensing and mandatory landing execution',
        original_manifest_sha256=cfg['retention_sha256'], original_sample_count=prior['sample_count'],
        sampling=dict(mode='class_startup_balanced_v1', startup_fraction=.5,
                      startup_policy_steps=args.startup_steps, startup_indices=startup),
        split='entire seed42 train; entire seed43 held out; neither split contains performance seeds44/45',
        plan=str(plan_path.resolve()), plan_sha256=digest(plan_path),
        model=plan['model'], model_sha256=plan['model_sha256'],
        maximum_absolute_label=float(np.max(np.abs(arrays['labels'])))), indent=2)+'\n')
    command_path = args.output_dir/'commands.json'
    command_path.write_text(json.dumps(dict(schema='m20_cts_retention_commands_v1', classes=list(CLASSES),
        commands=commands.tolist(), retention_manifest_sha256=digest(manifest), payload_sha256=digest(payload),
        source_checkpoint_sha256=digest(args.reference_checkpoint), source_plan=str(plan_path.resolve()),
        source_plan_sha256=digest(plan_path)), indent=2)+'\n')
    print(json.dumps(dict(manifest=str(manifest), retention_sha256=digest(manifest),
        retention_command_manifest=str(command_path), retention_command_sha256=digest(command_path),
        samples=len(arrays['classes']), payload_bytes=payload.stat().st_size)), flush=True)


if __name__ == '__main__':
    main()
