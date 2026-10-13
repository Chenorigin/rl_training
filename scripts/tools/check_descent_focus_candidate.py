#!/usr/bin/env python3
"""Frozen baseline-versus-candidate checks; geometry, not reward, decides success."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from tensordict import TensorDict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "deploy/deploy_mujoco"))
from deploy_mujoco import load_actor


def compare(baseline, candidate, descent):
    results = {}
    for case in ("ascent", "flat", "turn_left", "turn_right", "slope_up", "slope_down", "rough", "obstacles", "hurdles"):
        old, new = baseline["cases"][case], candidate["cases"][case]
        checks = {"no_new_fall": not new["fell"] if not old["fell"] else False,
                  "baseline_completion_preserved": not old["crossed_finish"] or new["crossed_finish"],
                  "supported_extension_not_lower": new["rear_supported_extension_p05_m"] >= old["rear_supported_extension_p05_m"]}
        if case == "ascent":
            checks["front_same_tread_not_more"] = new["front_same_tread_events"] <= old["front_same_tread_events"]
        if case.startswith("turn_"):
            checks["turn_clearance_not_higher"] = new["turn_clearance_p95_m"] <= old["turn_clearance_p95_m"]
        results[case] = {"checks": checks, "passed": all(checks.values()),
                         "baseline": {k: old[k] for k in ("fell", "crossed_finish", "rear_supported_extension_p05_m", "turn_clearance_p95_m", "front_same_tread_events")},
                         "candidate": {k: new[k] for k in ("fell", "crossed_finish", "rear_supported_extension_p05_m", "turn_clearance_p95_m", "front_same_tread_events")}}
    descent_checks = {
        "left_no_collision": descent["left_front_rear_clearance"]["min_m"] >= 0,
        "right_no_collision": descent["right_front_rear_clearance"]["min_m"] >= 0,
        "finished_stairs": descent["cleared_stairs"],
        "no_fall": not descent["fell"],
        "inside_all_samples": descent["inside_course_all_samples"],
        "no_cost_outside_gate": descent["full_rollout_gate_audit"]["inactive_nonzero_cost_frames"] == 0,
        "single_entry_interval": len(descent["full_rollout_gate_audit"]["active_intervals_s"]) == 1,
    }
    return {"descent_checks": descent_checks, "preservation": results,
            "accepted": all(descent_checks.values()) and all(v["passed"] for v in results.values()),
            "unmeasured_terrains": ["terrain/command/seed combinations outside the fixed suite"],
            "scope": "fixed nine-case preservation only; not proof of all-terrain identity"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-dir", type=Path, required=True)
    p.add_argument("--candidate-dir", type=Path, required=True)
    p.add_argument("--descent-probe", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    baseline = json.loads((args.baseline_dir / "summary.json").read_text())
    candidate = json.loads((args.candidate_dir / "summary.json").read_text())
    descent = json.loads(args.descent_probe.read_text())
    if candidate["checkpoint_sha256"] != descent["checkpoint_sha256"]:
        raise ValueError("Preservation and descent probes must measure the same checkpoint")
    result = compare(baseline, candidate, descent)
    old_actor, new_actor = load_actor(Path(baseline["checkpoint"])), load_actor(Path(candidate["checkpoint"]))
    drift = {}
    for case in result["preservation"]:
        with np.load(args.baseline_dir / f"{case}.npz") as clip:
            obs = TensorDict({"policy": torch.from_numpy(clip["policy_obs"]).float()}, batch_size=[len(clip["policy_obs"])])
        with torch.inference_mode():
            error = new_actor(obs) - old_actor(obs)
        drift[case] = {"same_observation_action_max_abs": float(error.abs().max()),
                       "same_observation_action_rms": float(error.square().mean().sqrt())}
    result["action_drift_diagnostic"] = drift
    result["baseline_sha256"] = baseline["checkpoint_sha256"]
    result["candidate_sha256"] = candidate["checkpoint_sha256"]
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"accepted": result["accepted"], "descent_checks": result["descent_checks"],
                      "preservation": {k: v["passed"] for k, v in result["preservation"].items()}}))


if __name__ == "__main__":
    main()
