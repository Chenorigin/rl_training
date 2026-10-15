#!/usr/bin/env python3
"""Train/validate independent M20 quiet V1 from an explicit checkpoint path.

An explicit path permits full optimizer resume across experiment directories.
No latest-run matching and no teacher log directory reuse.
"""
import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import shutil
import os

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "source/rl_training"))
BASELINE = ROOT / "logs/rsl_rl/deeprobotics_m20_stair_teacher/2026-10-09_01-55-35_gait_fix_descent_entry/model_214700.pt"
TASK = "Rough-Deeprobotics-M20-QuietLanding-v0"

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, default=BASELINE)
parser.add_argument("--reference-checkpoint", type=Path, help="Frozen anchor; defaults to the loaded checkpoint")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--max_iterations", type=int, default=5000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--run_name", default="quiet_landing_v1")
parser.add_argument("--validate-only", action="store_true", help="Run inference and check metrics; no PPO updates")
parser.add_argument("--probe-steps", type=int, default=240)
parser.add_argument("--warmup-iterations", type=int, help="Override critic warmup (use 1 for a bounded two-update smoke)")
parser.add_argument("--output-root", type=Path, default=ROOT / "logs/rsl_rl/deeprobotics_m20_quiet_landing_v1")
parser.add_argument("--small-atlas", action="store_true", help="Small offline atlas for integration smoke only")
parser.add_argument("--smoke-episode-seconds", type=float, help="Short episodes for logger/reset integration testing only")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.checkpoint.is_file() or args.num_envs < 1 or args.max_iterations < 1 or args.probe_steps < 1:
    parser.error("Require an existing checkpoint and positive env/update/probe counts")
if args.warmup_iterations is not None and args.warmup_iterations < 0:
    parser.error("Warmup iterations cannot be negative")
if args.smoke_episode_seconds is not None and (not args.small_atlas or args.smoke_episode_seconds <= 0):
    parser.error("Short smoke episodes require --small-atlas and a positive duration")
app = AppLauncher(args).app

import gymnasium as gym
import torch
import rl_training.tasks  # noqa: F401: registers this checkout's task
from isaaclab_tasks.utils import parse_env_cfg, load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab.utils.io import dump_yaml
from rl_training.quiet_landing_runner import QuietLandingRunner
import cli_args


def main():
    torch.set_num_threads(1)
    cfg = parse_env_cfg(TASK, device=args.device or "cuda:0", num_envs=args.num_envs)
    cfg.seed = args.seed
    agent = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
    agent.seed, agent.device = args.seed, cfg.sim.device
    agent.resume = True
    agent.load_run = str(args.checkpoint.resolve().parent)
    agent.load_checkpoint = args.checkpoint.name
    agent.max_iterations = args.max_iterations
    if args.warmup_iterations is not None:
        agent.algorithm.critic_warmup_iterations = args.warmup_iterations
    if args.small_atlas:
        cfg.scene.terrain.terrain_generator.num_rows = 2
        cfg.scene.terrain.visual_material = None
        cfg.scene.sky_light = None
        cfg.commands.base_velocity.debug_vis = False
    if args.smoke_episode_seconds is not None:
        cfg.episode_length_s = args.smoke_episode_seconds
    if any(name in cfg.curriculum.__dict__ for name in ("gait_quality", "ascent_direction", "step_completion")):
        raise RuntimeError("Old gait debug curriculum leaked into quiet task")
    log_dir = args.output_root / (datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + args.run_name)
    log_dir.mkdir(parents=True, exist_ok=False)
    env = gym.make(TASK, cfg=cfg)
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
        runner = QuietLandingRunner(wrapped, cli_args.convert_rsl_rl_cfg_dict(agent.to_dict()),
                                log_dir=str(log_dir), device=agent.device)
        runner.load(str(args.checkpoint.resolve()))
        source_manifest = args.checkpoint.parent / "launch.json"
        quiet_resume = source_manifest.is_file() and json.loads(source_manifest.read_text()).get("task") == TASK
        reference_path = args.reference_checkpoint or args.checkpoint
        if args.reference_checkpoint or not quiet_resume:
            reference = torch.load(reference_path, map_location="cpu", weights_only=True)
            # First teacher-to-quiet initialization replaces the old 199998 anchor.
            runner.alg.set_reference(reference["actor_state_dict"])
        else:
            # Subsequent quiet resume preserves the frozen reference and warmup count.
            reference_path = Path(json.loads(source_manifest.read_text())["reference"])
        dump_yaml(str(log_dir / "params/env.yaml"), cfg)
        dump_yaml(str(log_dir / "params/agent.yaml"), agent)
        manifest = {"task": TASK, "checkpoint": str(args.checkpoint.resolve()),
                    "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                    "reference": str(reference_path.resolve()),
                    "settings": asdict(cfg.quiet_landing), "extended_settings": asdict(cfg.quiet_extended),
                    "reward_revision": "expanded_20261010", "small_atlas": args.small_atlas,
                    "mode": "inference_probe" if args.validate_only else "ppo_resume"}
        sources = [ROOT / "source/rl_training/m20_quiet/signals.py",
                   ROOT / "source/rl_training/m20_quiet/extended.py",
                   ROOT / "source/rl_training/rl_training/quiet_landing_env.py",
                   ROOT / "source/rl_training/rl_training/quiet_landing_runner.py",
                   ROOT / "source/rl_training/rl_training/stair_guarded_ppo.py",
                   Path(__file__).resolve()]
        config_dir = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20"
        sources += [config_dir / "quiet_landing_env_cfg.py", config_dir / "agents/quiet_landing_ppo_cfg.py"]
        sources += list((ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp").glob("*.py"))
        manifest["source_sha256"] = {}
        for source in sources:
            relative = source.relative_to(ROOT)
            destination = log_dir / "source_snapshot" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            manifest["source_sha256"][str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
        (log_dir / "launch.json").write_text(json.dumps(manifest, indent=2))
        base = env.unwrapped
        obs, _ = wrapped.reset()
        assert obs['policy'].shape == (args.num_envs, 244)
        assert obs['critic'].shape == (args.num_envs, 285)
        assert "stair_impact_cost" not in base.reward_manager.active_terms
        assert "quiet_touchdown" in base.reward_manager.active_terms
        for name, weight in cfg.quiet_extended.weights.items():
            assert ("quiet_"+name in base.reward_manager.active_terms) == (weight != 0)
        if args.validate_only:
            policy = runner.get_inference_policy(device=agent.device)
            total_events, seen_logs = 0, set()
            for _ in range(args.probe_steps):
                with torch.inference_mode():
                    obs, rewards, _, extras = wrapped.step(policy(obs))
                assert torch.isfinite(rewards).all()
                total_events += int(base.quiet_step_events.sum())
                seen_logs.update(extras.get("log", {}))
                assert not any(key.startswith(("Curriculum/", "Episode_Reward/")) for key in extras.get("log", {}))
            report = {"events_seen": total_events, "log_keys": sorted(seen_logs),
                      "actor_shape": list(obs['policy'].shape), "critic_shape": list(obs['critic'].shape),
                      "quiet": {key: float(value) for key, value in base.quiet_tracker.metrics(slice(None)).items()}}
            (log_dir / "probe.json").write_text(json.dumps(report, indent=2))
        else:
            runner.add_git_repo_to_log(__file__)
            runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
        print("QUIET_LANDING_ISAAC_PASS " + str(log_dir), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # Isaac fast shutdown can exit 0 before Python prints an unwinding
        # exception: make the failure visible BEFORE closing the app.
        import traceback
        traceback.print_exc()
        sys.stderr.flush()
        sys.stdout.flush()
        # Kit fast shutdown can terminate the process with status 0. On failure
        # exit this process explicitly; OS teardown releases its GPU resources.
        os._exit(1)
    finally:
        app.close()
