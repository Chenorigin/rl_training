#!/usr/bin/env python3
"""Fixed MuJoCo gait evaluation and reference-clip capture; no training."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "deploy/deploy_mujoco"))
import deploy_mujoco as deploy
from gait_metrics import GaitMetrics
import mujoco
import numpy as np
import torch
from tensordict import TensorDict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=750)
    parser.add_argument("--cases", nargs="+", choices=("ascent", "descent", "turn_left", "turn_right", "flat"),
                        default=["ascent", "descent", "turn_left", "turn_right", "flat"])
    parser.add_argument("--stair-height", type=float, default=0.15)
    parser.add_argument("--tread-depth", type=float, default=0.30)
    parser.add_argument("--stair-count", type=int, default=5)
    parser.add_argument("--min-extension", type=float, default=0.30)
    parser.add_argument("--command-vx", type=float, default=0.5,
                        help="Fixed forward command in m/s for ascent/descent/flat")
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("steps must be positive")
    torch.set_num_threads(1)
    actor = deploy.load_actor(args.checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = dict(checkpoint=str(args.checkpoint.resolve()),
                   checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(), cases={})
    for case in args.cases:
        terrain = case if case in {"ascent", "descent", "flat"} else "flat"
        cfg = argparse.Namespace(terrain=terrain, terrain_xml=None,
            stair_height=args.stair_height if terrain in {"ascent", "descent"} else None,
            tread_depth=args.tread_depth if terrain in {"ascent", "descent"} else None,
            stair_count=args.stair_count if terrain in {"ascent", "descent"} else None)
        model = deploy.load_model(args.model, cfg)
        contract = deploy.M20Contract(model, cfg.terrain_config)
        data = mujoco.MjData(model)
        contract.reset(data)
        metrics = GaitMetrics(contract, min_extension=args.min_extension)
        command = np.array([args.command_vx, 0, 0] if case in {"ascent", "descent", "flat"}
                           else [0, 0, 0.6 if case == "turn_left" else -0.6], dtype=np.float32)
        previous = np.zeros(16, dtype=np.float32)
        records = {k: [] for k in ("time", "root_pos", "root_quat", "joint_pos", "joint_vel", "command", "action", "wheel_pos", "hip_pos", "wheel_ground", "contact_force")}
        fell = False
        for _ in range(args.steps):
            obs = contract.observe(data, command, previous)
            with torch.inference_mode():
                action = actor(TensorDict({"policy": torch.from_numpy(obs)[None]}, batch_size=[1])).squeeze(0).numpy()
            if not np.isfinite(action).all():
                raise RuntimeError("Nonfinite actor output")
            action = np.clip(action, -100, 100)
            for _ in range(4):
                contract.apply_pd(data, action)
                mujoco.mj_step(model, data)
            metrics.update(data, command)
            sample = metrics.samples[-1]
            values = dict(time=data.time, root_pos=data.xpos[contract.base_id].copy(),
                root_quat=data.xquat[contract.base_id].copy(), joint_pos=data.qpos[contract.qpos_ids].copy(),
                joint_vel=data.qvel[contract.qvel_ids].copy(), command=command.copy(), action=action.copy(),
                wheel_pos=data.xpos[contract.wheel_body_ids].copy(), hip_pos=data.xpos[metrics.hips].copy(),
                wheel_ground=sample["wheel_ground"], contact_force=sample["forces"])
            for key, value in values.items():
                records[key].append(value)
            previous = action.astype(np.float32)
            position = data.xpos[contract.base_id]
            ground = contract.ground_height(data, *position[:2])
            fell |= bool(position[2] < ground + 0.22 or data.xmat[contract.base_id].reshape(3, 3)[2, 2] < 0.35)
            if fell:
                break
        result = {**cfg.terrain_config.summary(), **metrics.summary(),
                  **contract.crossing_summary(data, fell), "fell": fell, "physics_enabled": True,
                  "final_xyz": data.xpos[contract.base_id].tolist(), "landings": metrics.landings,
                  "reference_clip_status": "unreviewed; not an AMP expert dataset"}
        np.savez_compressed(args.output_dir / f"{case}.npz", **{k: np.asarray(v) for k, v in records.items()}, summary=json.dumps(result))
        results["cases"][case] = result
        print(case, json.dumps({k: result[k] for k in ("sample_count", "fell", "rear_supported_extension_p05_m", "turn_clearance_p95_m", "front_same_tread_events")}), flush=True)
    (args.output_dir / "summary.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
