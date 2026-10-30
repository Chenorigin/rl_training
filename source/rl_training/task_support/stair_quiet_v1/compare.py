"""Compare real MuJoCo trajectories through shared, explicitly assumed acoustics.

This ranks hypotheses and exposes sensitivity. It never certifies actual M20
sound reduction. Calibrated recordings use audio.py and acoustics.py separately.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .acoustics import read_json, STANDARD_PATH
from .audio import measure_pressure
from .benchmark import write_json, digest
from .sound_model import load_profile, render_sound, paired_profile_sweep, write_sound
from .stress import canonical_hash
from .mujoco_landing import validate_role


def control_reasons(item):
    """Require an enabled controller's identity and evidence of actual use."""
    reasons = []
    role = item.get("role", "")
    mode = item.get("landing_control", "off")
    try:
        validate_role(role, mode)
    except ValueError as exc:
        reasons.append(str(exc))
    document = item.get("control_contract")
    if document is not None:
        from .landing import contract_hash
        if (not isinstance(document, dict) or document.get("settings", {}).get("mode") != mode
                or item.get("control_contract_sha256") != contract_hash(document)):
            reasons.append("invalid control contract or hash")
        sources = item.get("source_sha256", {})
        for field, filename in (("source_sha256", "landing.py"), ("memory_source_sha256", "terrain_memory.py")):
            if field in document:
                path = "source/rl_training/task_support/stair_quiet_v1/" + filename
                if sources.get(path) != document[field]:
                    reasons.append(f"control contract differs from recorded {filename} source")
    if mode != "off" or role.startswith("control-"):
        execution = item.get("controller_execution", {})
        modified = execution.get("modified_wheel_samples")
        samples = execution.get("wheel_samples")
        if document is None or execution.get("enabled") is not True:
            reasons.append("enabled controller lacks contract or execution evidence")
        if (type(samples) is not int or type(modified) is not int
                or not 0 < modified <= samples):
            reasons.append("controller did not demonstrably modify any wheel sample")
    elif "controller_execution" in item:
        execution = item["controller_execution"]
        if (execution.get("enabled") is not False
                or execution.get("wheel_samples") != 0
                or execution.get("modified_wheel_samples") != 0):
            reasons.append("off controller has inconsistent execution evidence")
    return reasons


def read_rollout(folder):
    folder = Path(folder)
    summary = read_json(folder / "summary.json")
    events = read_json(folder / "events.json")
    if summary.get("schema") != "stair_quiet_v1.mujoco_impact_rollout.v1":
        raise ValueError("Unsupported rollout schema")
    if events.get("provenance", {}).get("checkpoint_sha256") != summary.get("checkpoint_sha256"):
        raise ValueError("Events and rollout checkpoint identity differ")
    if "control_contract" in summary:
        from .landing import contract_hash
        if summary.get("control_contract_sha256") != contract_hash(summary["control_contract"]):
            raise ValueError("Control contract hash mismatch")
        if events.get("provenance", {}).get("control_contract_sha256") != summary["control_contract_sha256"]:
            raise ValueError("Events and controller identity differ")
        sources = summary.get("source_sha256", {})
        for field, filename in (("source_sha256", "landing.py"), ("memory_source_sha256", "terrain_memory.py")):
            if field in summary["control_contract"]:
                if summary["control_contract"][field] != sources.get(
                        "source/rl_training/task_support/stair_quiet_v1/" + filename):
                    raise ValueError(f"Controller and recorded {filename} source differ")
    if "stress" in summary:
        stress = summary["stress"]
        if (stress.get("profile_sha256") != canonical_hash(stress.get("profile"))
                or events.get("provenance", {}).get("stress_profile_sha256") != stress["profile_sha256"]):
            raise ValueError("Stress profile hash mismatch")
    if events.get("duration_s") != summary.get("duration_s"):
        raise ValueError("Event duration differs from raw rollout")
    if events.get("measurement_window_s") != [summary.get("measurement_start_s"), summary.get("measurement_end_s")]:
        raise ValueError("Event measurement window differs from rollout")
    start, end, duration, total = [summary.get(k) for k in (
        "measurement_start_s", "measurement_end_s", "measurement_duration_s", "duration_s")]
    if (any(type(x) not in (int, float) or not math.isfinite(x) for x in (start, end, duration, total))
            or not 0 <= start < end <= total
            or not math.isclose(end - start, duration, abs_tol=1e-8, rel_tol=1e-8)):
        raise ValueError("Rollout measurement duration is inconsistent")
    evidence_names = ("summary.json", "events.json", "contacts.npz", "contact_points.npz")
    if any(not (folder / name).is_file() for name in evidence_names):
        raise ValueError("Raw contact evidence is required")
    if summary.get("landing_control", "off") != "off":
        import numpy as np
        with np.load(folder / "contacts.npz", allow_pickle=False) as recorded:
            required = ("time_s", "landing_leg_target", "landing_kp_scale", "landing_kd_scale",
                        "landing_phase", "landing_modified", "landing_invalid", "landing_contact_valid",
                        "effort_requested_Nm", "effort_applied_Nm")
            if any(key not in recorded for key in required):
                raise ValueError("Enabled controller is missing raw execution arrays")
            n = len(recorded["time_s"])
            if any(len(recorded[key]) != n or not np.isfinite(recorded[key]).all() for key in required):
                raise ValueError("Controller evidence has inconsistent length or non-finite values")
            shapes = {"time_s": (n,), "landing_leg_target": (n, 4, 3),
                      "effort_requested_Nm": (n, 16), "effort_applied_Nm": (n, 16)}
            if n == 0 or any(recorded[key].shape != shapes.get(key, (n, 4)) for key in required):
                raise ValueError("Controller execution evidence has invalid shape")
            dt = summary.get("physics_dt_s")
            if (type(dt) not in (int, float) or not math.isfinite(dt) or dt <= 0
                    or not np.allclose(recorded["time_s"], np.arange(n) * dt, rtol=0., atol=1e-8)
                    or not math.isclose(n * dt, total, rel_tol=0., abs_tol=1e-8)):
                raise ValueError("Controller evidence does not cover the entire rollout")
            for key in ("landing_modified", "landing_invalid", "landing_contact_valid"):
                if recorded[key].dtype.kind != "b":
                    raise ValueError("Controller masks must be Boolean")
            if (recorded["landing_phase"].dtype.kind not in "iu"
                    or not np.isin(recorded["landing_phase"], np.arange(6)).all()):
                raise ValueError("Unknown controller phase code")
            count = int(np.count_nonzero(recorded["landing_modified"]))
            execution = summary.get("controller_execution", {})
            if execution.get("modified_wheel_samples") != count or execution.get("wheel_samples") != n * 4:
                raise ValueError("Controller execution summary differs from raw samples")
    return {"folder": str(folder.resolve()), "summary": summary, "events": events,
            "files_sha256": {name: digest(folder / name) for name in evidence_names}}


def comparison_reasons(baseline, candidate, standard):
    """Reject quiet-by-stopping and incompatible physical trials before ranking."""
    reasons = []
    for label, item in (("baseline", baseline), ("candidate", candidate)):
        # Legacy off-only fixtures and reports have no controller metadata.
        if "landing_control" in item or item.get("role", "").startswith("control-"):
            reasons.extend(f"{label}: {reason}" for reason in control_reasons(item))
        if item.get("completed") is not True or item.get("window_valid") is not True:
            reasons.append(f"{label}: incomplete stair traversal")
        if item.get("fell") is not False or item.get("ever_outside_course") is not False:
            reasons.append(f"{label}: fall or lateral bypass")
        duration = item.get("measurement_duration_s")
        if type(duration) not in (int, float) or not math.isfinite(duration) or duration <= 0:
            reasons.append(f"{label}: invalid measurement duration")
        pause = item.get("max_stationary_interval_s")
        if type(pause) not in (int, float) or not math.isfinite(pause) or not 0 <= pause <= standard["max_stationary_interval_s"]:
            reasons.append(f"{label}: pause limit exceeded or missing")
        if type(item.get("optimizer_updates")) is not int or item["optimizer_updates"] != 0 or item.get("weights_updated") is not False:
            reasons.append(f"{label}: not a fixed-policy inference run")
    for key in ("compiled_model_sha256", "command", "physics_dt_s", "policy_dt_s", "seed",
                "event_extraction", "traversal_settings", "source_sha256"):
        if key not in baseline or key not in candidate or baseline[key] != candidate[key]:
            reasons.append(f"incompatible {key}")
    if "stress" in baseline or "stress" in candidate:
        for key in ("profile", "profile_sha256", "contact_extra_delay_steps",
                    "contact_inherent_age_s", "contact_total_age_s", "actuator_extra_delay_steps"):
            if (key not in baseline.get("stress", {}) or key not in candidate.get("stress", {})
                    or baseline["stress"][key] != candidate["stress"][key]):
                reasons.append(f"incompatible paired stress {key}")
        for label, item in (("baseline", baseline), ("candidate", candidate)):
            stress = item.get("stress", {})
            if stress.get("profile_sha256") != canonical_hash(stress.get("profile")):
                reasons.append(f"{label}: invalid stress profile hash")
    if baseline.get("role", "").endswith("student") and candidate.get("role", "").endswith("student"):
        for key in ("action_delay_steps", "student_sensor"):
            if key not in baseline or key not in candidate or baseline[key] != candidate[key]:
                reasons.append(f"incompatible student {key}")
    b, c = baseline.get("measurement_duration_s", 0), candidate.get("measurement_duration_s", 0)
    if type(b) in (int, float) and type(c) in (int, float) and b > 0:
        if not math.isfinite(c) or abs(c / b - 1) > standard["max_completion_duration_relative_difference"]:
            reasons.append("completion time differs beyond the fixed tolerance")
    return reasons


def validate_candidate(summary, expected):
    role = summary.get("role")
    allowed = ("candidate-teacher", "candidate-student", "control-teacher", "control-student")
    if role not in allowed:
        raise ValueError("Candidate role is missing")
    if role.startswith("control-"):
        reference = expected[1 if role.endswith("student") else 0]["sha256"]
        if (summary.get("checkpoint_sha256") != reference
                or summary.get("pinned_actor_identity_checked") is not True
                or summary.get("candidate_kind") != "baseline_actor_plus_controller"):
            raise ValueError("Control role must identify its unchanged pinned actor")
        if summary.get("landing_control", "off") == "off":
            raise ValueError("Control role cannot silently disable its controller")
    elif summary.get("checkpoint_sha256") in {r["sha256"] for r in expected}:
        raise ValueError("Candidate is one of the unchanged baselines")


def metric_comparison(baseline, candidate, fraction):
    if (type(baseline) not in (int, float) or type(candidate) not in (int, float)
            or not math.isfinite(baseline) or not math.isfinite(candidate) or baseline <= 0):
        return {"status": "undefined", "hypothetical_threshold_met": False}
    delta = candidate - baseline
    return {
        "status": "model_prediction_only", "baseline_db": baseline, "candidate_db": candidate,
        "delta_db": delta, "reading_reduction_percent": 100 * (1 - candidate / baseline),
        "energy_ratio": 10 ** (delta / 10),
        "hypothetical_threshold_met": candidate <= fraction * baseline,
    }


def evaluate(teacher_dir, student_dir, candidate_dir, output, profile_name):
    standard = read_json(STANDARD_PATH)
    trials = {"baseline_teacher": read_rollout(teacher_dir), "baseline_student": read_rollout(student_dir)}
    expected = list(standard["baselines"].values())
    for label, reference, role in zip(
            trials, expected, ("baseline-teacher", "baseline-student")):
        summary = trials[label]["summary"]
        if summary["checkpoint_sha256"] != reference["sha256"] or summary.get("role") != role:
            raise ValueError(f"{label} is not the requested pinned baseline")
        if summary.get("landing_control", "off") != "off":
            raise ValueError("Baseline rollout has an enabled landing controller")
    if candidate_dir is not None:
        candidate = read_rollout(candidate_dir)
        validate_candidate(candidate["summary"], expected)
        trials["candidate"] = candidate
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    profile = load_profile(profile_name)
    report = {
        "schema": "stair_quiet_v1.computer_sound_comparison.v1",
        "status": "uncalibrated_simulation", "actual_acoustic_pass": False,
        "real_world_reduction_percent": None,
        "target_reading_reduction_percent": 100 * (1 - standard["max_db_reading_fraction"]),
        "criterion": "both LAeq and LAFmax readings at most half of EACH pinned baseline; model results are hypotheses",
        "candidate_available": candidate_dir is not None,
        "candidate_kind": trials["candidate"]["summary"].get("candidate_kind", "trained_actor") if candidate_dir else None,
        "trials": {}, "comparisons": {},
        "limitations": [
            "Uncalibrated equivalent radiator, not a validated M20/stair/room sound field.",
            "Rigid-body impulses and assumed contact-pulse duration do not recover actual impact audio.",
            "Only selected dynamic impact excitation is modeled; motor, rolling and sliding noise are omitted.",
            "One seed per input directory; this is not the repeated measured-sound acceptance protocol.",
            "No actual sound uncertainty bound is available; a predicted 50% cannot certify real reduction.",
        ],
    }
    for label, trial in trials.items():
        pressure, metadata = render_sound(trial["events"], profile)
        metrics = measure_pressure(pressure, profile["sample_rate_hz"],
                                   start_s=metadata["measurement_start_s"], end_s=metadata["measurement_end_s"])
        write_sound(output / label, pressure, metadata)
        report["trials"][label] = {
            "source_directory": trial["folder"], "files_sha256": trial["files_sha256"],
            "rollout": trial["summary"], "prediction_metrics": metrics,
            "profile_sha256": metadata["profile_sha256"],
            "claim_scope": metadata["claim_scope"], "actual_acoustic_pass": False,
        }
    if candidate_dir is not None:
        for label in ("baseline_teacher", "baseline_student"):
            reasons = comparison_reasons(trials[label]["summary"], trials["candidate"]["summary"], standard)
            entry = {"eligible_for_model_comparison": not reasons, "reasons": reasons,
                     "actual_acoustic_pass": False, "all_assumptions_meet_hypothetical_target": False}
            # Preserve the failed trial and predicted sound; do not rank an
            # incomplete traversal or short/slow episode as a quieter candidate.
            if not reasons:
                sweep = paired_profile_sweep(trials[label]["events"], trials["candidate"]["events"], profile)
                minimum = {"laeq_db": math.inf, "lafmax_db": math.inf}
                passed = True
                for row in sweep["rows"]:
                    row["comparison"] = {}
                    for metric in minimum:
                        diff = metric_comparison(
                            row["baseline"]["metrics"][metric], row["candidate"]["metrics"][metric],
                            standard["max_db_reading_fraction"])
                        row["comparison"][metric] = diff
                        passed &= diff["hypothetical_threshold_met"]
                        minimum[metric] = min(minimum[metric], diff.get("reading_reduction_percent", -math.inf))
                entry["sensitivity"] = sweep
                entry["minimum_predicted_reading_reduction_percent"] = {
                    k: v if math.isfinite(v) else None for k, v in minimum.items()}
                entry["all_assumptions_meet_hypothetical_target"] = bool(passed)
            report["comparisons"][label] = entry
    write_json(output / "comparison.json", report)
    lines = [
        "# quiet_v1 电脑声学假设模型结果",
        "",
        "**以下为未标定模型预测，不是机器狗实际分贝测量；实际 50% 目标尚未验收。**",
        "",
        "| 策略/控制器 | 类型 | 完整上楼 | 预测 LAeq (dBA) | 预测 LAFmax (dBA) |",
        "|---|---|---|---:|---:|",
    ]
    for label, item in report["trials"].items():
        m = item["prediction_metrics"]
        fmt = lambda x: "未知" if x is None else f"{x:.3f}"
        kind = item["rollout"].get("candidate_kind", item["rollout"].get("role"))
        lines.append(f"| {label} | {kind} | {item['rollout']['completed']} | {fmt(m['laeq_db'])} | {fmt(m['lafmax_db'])} |")
    lines += ["", "参数来源、原始 Pa、接触事件、模型与策略 SHA256、限制见同目录 JSON/NPZ。",
              "未提供候选时，表内只有基线；不能计算或宣称改进降幅。" if candidate_dir is None
              else "候选只有通过相同物理条件、完整上楼及速度检查后才进行成对参数敏感性比较。",
              ""]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-teacher", type=Path, required=True)
    parser.add_argument("--baseline-student", type=Path, required=True)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--profile", default="illustrative_compact_panel")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = evaluate(args.baseline_teacher, args.baseline_student, args.candidate, args.output, args.profile)
    print(json.dumps({"status": report["status"], "candidate_available": report["candidate_available"],
                      "actual_acoustic_pass": False, "report": str(args.output / "comparison.json")}, indent=2))


if __name__ == "__main__":
    main()
