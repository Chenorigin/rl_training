"""Fit the existing M20 actor to stop idle wheels and preserve baseline outputs.

Inputs are NPZ files containing actual 244D policy observations. No deployment
branch or action clamp is added. The first two actor layers remain frozen.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

import numpy as np
import torch
from tensordict import TensorDict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "deploy/deploy_mujoco"))
sys.path.insert(0, str(ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp"))
from deploy_mujoco import load_actor
from quiet_landing_signals import idle_mask


def fit(args):
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    checkpoint = args.checkpoint.resolve()
    reference = load_actor(checkpoint).requires_grad_(False)
    actor = load_actor(checkpoint)
    files = args.inputs
    if args.input_manifest:
        files = [Path(p) for p in json.loads(args.input_manifest.read_text())["inputs"]]
    observations = []
    input_hashes = {}
    for path in files:
        path = path.resolve()
        with np.load(path) as record:
            observation = torch.from_numpy(record["policy"].copy())
        if observation.dtype != torch.float32 or observation.ndim != 2 or observation.shape[1] != 244:
            raise ValueError(f"Expected float32 [N,244] policy observations: {path}")
        if not torch.isfinite(observation).all():
            raise ValueError(f"Nonfinite actor inputs: {path}")
        observations.append(observation)
        input_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    inputs = torch.cat(observations)
    idle = inputs[idle_mask(inputs)].clone()
    augmentations = []
    # Cover feedback as the previous wheel actions/velocities settle to zero.
    # These augmentations are supervised examples, not physical rollouts.
    for fraction in (0.25, 0.5, 0.75, 1.0):
        sample = idle.clone()
        sample[:, 53:57] *= 1 - fraction
        sample[:, 37:41] *= 1 - fraction
        sample[:, :3] *= 1 - fraction
        augmentations.append(sample)
    for speed in (0.35, 0.5, 0.8, -0.7):
        sample = idle[::4].clone()
        sample[:, 6] = speed
        augmentations.append(sample)
    sample = inputs[::4].clone()
    sample[:, 6:9] = 0
    augmentations.append(sample)
    inputs = torch.cat([inputs, *augmentations])
    mask = idle_mask(inputs)
    with torch.no_grad():
        targets = reference(TensorDict({"policy": inputs}, batch_size=[len(inputs)])).clone()
        targets[mask, 12:] = 0
        # _modules retains all three uses of the shared ELU instance; children()
        # deduplicates that instance and would produce incorrect features.
        features = torch.nn.Sequential(*list(actor.mlp._modules.values())[:4])(inputs).detach()
    actor.requires_grad_(False)
    for name in ("4", "6"):
        actor.mlp._modules[name].requires_grad_(True)
    head = torch.nn.Sequential(*list(actor.mlp._modules.values())[4:])
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-4)
    idle_ids, moving_ids = torch.where(mask)[0], torch.where(~mask)[0]
    if not len(idle_ids) or not len(moving_ids):
        raise ValueError("Both idle and preservation samples are required")
    started = time.time()
    for step in range(args.steps):
        ids = torch.cat([idle_ids[torch.randint(len(idle_ids), (512,))],
                         moving_ids[torch.randint(len(moving_ids), (512,))]])
        error = (head(features[ids]) - targets[ids]).square()
        loss = args.idle_weight * error[:512, 12:].mean() + 20 * error[:512, :12].mean() + 100 * error[512:].mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        if step % 1000 == 0:
            print(f"step={step} loss={float(loss):.6f}", flush=True)
    with torch.no_grad():
        error = head(features) - targets
        original = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = actor.state_dict()
        for name, value in original["actor_state_dict"].items():
            if not name.startswith(("mlp.4.", "mlp.6.")):
                assert torch.equal(value, state[name]), name
        original["actor_state_dict"] = state
        original["optimizer_state_dict"]["state"] = {}
        metadata = {
            "method": "supervised baseline distillation with idle wheel-zero target; last two layers only",
            "steps": args.steps, "idle_weight": args.idle_weight, "seed": args.seed,
            "seconds": time.time() - started, "samples": len(inputs),
            "idle_samples": len(idle_ids), "moving_samples": len(moving_ids),
            "idle_wheel_rms": float(error[mask, 12:].square().mean().sqrt()),
            "idle_leg_rms_delta": float(error[mask, :12].square().mean().sqrt()),
            "moving_all_rms_delta": float(error[~mask].square().mean().sqrt()),
            "runtime_clamp": False, "inputs": list(input_hashes), "input_sha256": input_hashes,
            "first_two_layers_unchanged": True,
            "original_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "future_training": "Use --init_actor_from; the original critic/reference do not certify a new PPO run",
        }
        original["idle_stop_calibration"] = metadata
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            raise FileExistsError(f"Refusing to overwrite checkpoint: {args.output}")
        torch.save(original, args.output)
        args.output.with_suffix(".fit.json").write_text(json.dumps(metadata, indent=2))
        print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    data = parser.add_mutually_exclusive_group(required=True)
    data.add_argument("--inputs", type=Path, nargs="+")
    data.add_argument("--input_manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--idle_weight", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.steps <= 0 or args.idle_weight <= 0:
        parser.error("steps and idle_weight must be positive")
    fit(args)
