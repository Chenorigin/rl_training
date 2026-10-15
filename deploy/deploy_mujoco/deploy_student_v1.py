#!/usr/bin/env python3
"""M20 student v1 manual MuJoCo deployment and CPU actor export.

Reuses v0's pinned PD/physics and keyboard lifecycle in a separate module
namespace; v0 defaults are never modified. V1 supplies its own sensor adapter.
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path

import torch

from student_v1_adapter import load_student, profile_config, StudentMapAdapter, sha256, observation_contract, PROFILE_NAMES, canonical_profile_name


def runtime():
    spec = importlib.util.spec_from_file_location("m20_student_v1_mujoco_runtime", Path(__file__).with_name("deploy_student_v0.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.load_student = load_student
    module.profile_config = profile_config
    module.StudentMapAdapter = StudentMapAdapter
    module.export_student = export_student
    return module


def parse_args(argv=None):
    engine = runtime()
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--policy-jit", type=Path)
    parser.add_argument("--model", type=Path)
    terrain = parser.add_mutually_exclusive_group()
    terrain.add_argument("--terrain", choices=tuple(engine.baseline.PRESETS))
    terrain.add_argument("--terrain-xml", type=Path)
    parser.add_argument("--noise-profile", type=canonical_profile_name, choices=PROFILE_NAMES, default="student-v1")
    parser.add_argument("--profile-file", type=Path)
    parser.add_argument("--noise-level", type=float)
    parser.add_argument("--command-x", type=float, default=.5)
    parser.add_argument("--command-y", type=float, default=0.)
    parser.add_argument("--command-yaw", type=float, default=0.)
    parser.add_argument("--warmup-s", type=float, default=0.)
    parser.add_argument("--track-course", action="store_true")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--viewer", action="store_true", help="Keyboard: Z stand, C student, W/S/A/D/Q/E move, R damping")
    parser.add_argument("--autoplay", action="store_true")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--continue-after-fall", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--export-dir", type=Path)
    parser.add_argument("--suite", action="store_true")
    parser.add_argument("--suite-profiles", nargs="+", type=canonical_profile_name, choices=PROFILE_NAMES, default=["teacher", "lidar_fov", "student-v0"])
    parser.add_argument("--suite-terrains", nargs="+", choices=tuple(engine.baseline.PRESETS), default=["flat", "ascent", "descent", "rough"])
    parser.add_argument("--suite-seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--friction-scale", type=float, default=1.)
    for name in ("stair-height", "tread-depth", "stair-start"):
        parser.add_argument("--"+name, type=float)
    parser.add_argument("--stair-count", type=int)
    args = parser.parse_args(argv)
    if (args.steps is not None and args.steps < 1) or not math.isfinite(args.warmup_s) or args.warmup_s < 0:
        parser.error("Invalid steps/warmup")
    if not all(math.isfinite(value) for value in (args.command_x, args.command_y, args.command_yaw, args.friction_scale)) or args.friction_scale <= 0:
        parser.error("Invalid command/friction")
    if args.noise_level is not None and not 0 <= args.noise_level <= 1:
        parser.error("noise-level must be in [0,1]")
    if args.export_only:
        if args.export_dir is None or args.policy_jit is not None:
            parser.error("export-only requires checkpoint/export-dir")
    elif args.model is None:
        parser.error("MuJoCo deployment requires model")
    if args.suite and (args.viewer or args.output_dir is None or args.terrain_xml or args.terrain or args.snapshot):
        parser.error("suite requires output-dir, without viewer/custom terrain/snapshot")
    if args.suite and any(getattr(args, name) is not None for name in engine.baseline.STAIR_OVERRIDES):
        parser.error("stair overrides apply to an individual rollout")
    if args.autoplay and not args.viewer:
        parser.error("autoplay requires viewer")
    if args.viewer and not args.autoplay and args.track_course:
        parser.error("Keyboard control supplies the commands")
    return args


def export_student(args, policy, provenance, identity):
    engine = runtime()
    engine.prepare_output(args.export_dir)
    dim = provenance["contract"]["student_observation_dim"]
    target = args.export_dir/"student_v1.pt"
    torch.jit.trace(policy, torch.zeros(1, dim)).save(str(target))
    reloaded = torch.jit.load(str(target), map_location="cpu")
    inputs = torch.randn(32, dim)
    with torch.inference_mode():
        error = float((policy(inputs)-reloaded(inputs)).abs().max())
    if error > 1e-6:
        raise RuntimeError("V1 export parity failed")
    engine.write_json(args.export_dir/"observation_contract.json", observation_contract(provenance["contract"]["history_steps"]))
    result = {**identity, "training_provenance": provenance, "export_sha256": sha256(target),
              "export_max_action_error": error, "weights_updated": False,
              "dimensions": {"input": [1, dim], "output": [1, 16]}}
    engine.write_json(args.export_dir/"deployment.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def main(argv=None):
    args = parse_args(argv)
    return runtime().main(args)


if __name__ == "__main__":
    main()
