"""Fail-closed acceptance of calibrated sound-level-meter measurements.

Decibels are logarithmic: halving sound energy changes a reading by about
-3.0103 dB; it does NOT halve its dB number. This task deliberately implements
the user's literal, much stricter 50% reduction in positive dB readings.
Contact-force, impulse and touchdown-speed proxies are never converted to dB.
This module does not collect sound, control a robot, or prove data authenticity.

Only Python's standard library is needed. See ACOUSTICS.md for the workflow.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any


STANDARD_PATH = Path(__file__).with_name("acoustic_protocol.json")
POLICY_CANDIDATE = "candidate"


class InvalidEvidence(ValueError):
    """Missing or incomparable evidence, rather than proof of success."""


def read_json(path: str | Path) -> Any:
    def reject_constant(value: str) -> None:
        raise InvalidEvidence(f"non-finite JSON number: {value}")
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject_constant)


def fingerprint(value: Any) -> str:
    try:
        raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise InvalidEvidence(f"not finite JSON: {exc}") from exc
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def checkpoint_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InvalidEvidence(message)


def _number(value: Any, name: str, minimum: float | None = None) -> float:
    _require(type(value) in (int, float) and math.isfinite(value), f"{name}: finite number required")
    _require(minimum is None or value >= minimum, f"{name}: must be >= {minimum}")
    return float(value)


def _text(value: Any, name: str) -> str:
    _require(isinstance(value, str) and bool(value.strip()), f"{name}: nonempty text required")
    _require(not value.startswith("FILL_"), f"{name}: template placeholder remains")
    return value


def _sha(value: Any, name: str) -> str:
    _require(isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None,
             f"{name}: lowercase SHA256 required")
    return value


def _time(value: Any, name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(_text(value, name).replace("Z", "+00:00"))
    except ValueError as exc:
        raise InvalidEvidence(f"{name}: ISO8601 datetime required") from exc
    _require(parsed.tzinfo is not None, f"{name}: timezone required")
    return parsed


def setup_template() -> dict:
    """No template value is a sound measurement or a successful trial."""
    return {
        "robot_id": None,
        "staircase_id": None,
        "stair_count": None,
        "staircase_geometry_and_material_record": None,
        "wheel_surface_and_condition_record": None,
        "payload_kg": None,
        "microphone_position_record": None,
        "microphone_distance_m": None,
        "microphone_height_m": None,
        "command_forward_velocity_m_s": None,
        "command_yaw_rate_rad_s": 0.0,
        "route_distance_m": None,
        "meter": {
            "serial": None,
            "microphone_serial": None,
            "iec_61672_class": None,
            "certificate_record": None,
            "certificate_valid_until": None,
            "calibrator_serial": None,
            "calibrator_certificate_record": None,
            "calibrator_certificate_valid_until": None,
            "calibrator_reference_db": None,
            "expanded_uncertainty_db": None,
            "lower_measurement_limit_db": None,
            "upper_measurement_limit_db": None,
        },
    }


def _validate_setup(setup: dict) -> None:
    _require(isinstance(setup, dict), "setup: object required")
    for key in ("robot_id", "staircase_id", "staircase_geometry_and_material_record",
                "wheel_surface_and_condition_record", "microphone_position_record"):
        _text(setup[key], key)
    _require(type(setup["stair_count"]) is int and setup["stair_count"] > 0,
             "stair_count: positive integer required")
    _number(setup["payload_kg"], "payload_kg", 0)
    for key in ("microphone_distance_m", "microphone_height_m",
                "command_forward_velocity_m_s", "route_distance_m"):
        _require(_number(setup[key], key) > 0, f"{key}: must be positive")
    _require(_number(setup["command_yaw_rate_rad_s"], "command_yaw_rate_rad_s") == 0,
             "this protocol measures straight ascent; yaw command must be zero")
    meter = setup["meter"]
    for key in ("serial", "microphone_serial", "certificate_record", "calibrator_serial",
                "calibrator_certificate_record"):
        _text(meter[key], f"meter.{key}")
    _require(meter["iec_61672_class"] in ("1", "2"), "meter: IEC 61672 class 1 or 2 required")
    for key in ("certificate_valid_until", "calibrator_certificate_valid_until"):
        _time(meter[key], f"meter.{key}")
    _number(meter["calibrator_reference_db"], "calibrator_reference_db")
    _require(_number(meter["expanded_uncertainty_db"], "expanded_uncertainty_db") > 0,
             "expanded_uncertainty_db: a documented positive uncertainty is required")
    lower = _number(meter["lower_measurement_limit_db"], "lower_measurement_limit_db")
    upper = _number(meter["upper_measurement_limit_db"], "upper_measurement_limit_db")
    _require(-100 <= lower < upper <= 250, "invalid meter measurement limits")


def freeze_protocol(setup: dict, candidate_sha256: str, *, frozen_at: str | None = None) -> dict:
    _validate_setup(setup)
    protocol = {
        "schema": "stair_quiet_v1.frozen_acoustic_protocol.v1",
        "standard": read_json(STANDARD_PATH),
        "setup": deepcopy(setup),
        "candidate_sha256": _sha(candidate_sha256, "candidate_sha256"),
        "frozen_at": frozen_at or datetime.now(timezone.utc).isoformat(),
    }
    _validate_protocol(protocol)
    return protocol


def _validate_protocol(protocol: dict) -> None:
    fingerprint(protocol)
    _require(protocol["schema"] == "stair_quiet_v1.frozen_acoustic_protocol.v1",
             "unsupported frozen protocol schema")
    _require(protocol["standard"] == read_json(STANDARD_PATH),
             "acceptance standard changed; do not relax it after measurements")
    _validate_setup(protocol["setup"])
    _sha(protocol["candidate_sha256"], "candidate_sha256")
    frozen = _time(protocol["frozen_at"], "frozen_at")
    meter = protocol["setup"]["meter"]
    for key in ("certificate_valid_until", "calibrator_certificate_valid_until"):
        _require(_time(meter[key], key) > frozen, f"meter.{key}: expired before protocol freeze")


def measurement_template(protocol: dict) -> dict:
    _validate_protocol(protocol)
    standard = protocol["standard"]
    policies = {key: value["sha256"] for key, value in standard["baselines"].items()}
    policies[POLICY_CANDIDATE] = protocol["candidate_sha256"]
    records = []
    # Rotate the three policies in successive blocks so each occupies each order.
    names = list(policies)
    for repeat in range(standard["min_repeats_per_policy"]):
        order = names[repeat % len(names):] + names[:repeat % len(names)]
        for policy in order:
            records.append({
                "trial_id": f"block_{repeat + 1}_{policy}",
                "block_id": repeat + 1,
                "policy_id": policy,
                "checkpoint_sha256": policies[policy],
                "started_at": None,
                "conditions": deepcopy(protocol["setup"]),
                "measurement_window": standard["measurement_window"],
                "instrument_export_record": None,
                "observation_record": None,
                "duration_s": None,
                "distance_m": None,
                "stairs_completed": None,
                "completed": None,
                "fall_count": None,
                "emergency_stop": None,
                "max_stationary_interval_s": None,
                "laeq_db": None,
                "lafmax_db": None,
                "overload": None,
                "under_range": None,
                "external_noise_event": None,
                "background_before": {
                    "started_at": None, "duration_s": None, "laeq_db": None, "lafmax_db": None,
                    "source_inactive": None, "instrument_export_record": None,
                },
                "background_after": {
                    "started_at": None, "duration_s": None, "laeq_db": None, "lafmax_db": None,
                    "source_inactive": None, "instrument_export_record": None,
                },
            })
    return {
        "schema": "stair_quiet_v1.acoustic_measurements.v1",
        "protocol_sha256": fingerprint(protocol),
        "all_attempts_included": None,
        "calibration": {
            "before_at": None, "after_at": None, "before_db": None, "after_db": None,
            "instrument_export_record": None,
        },
        "trials": records,
    }


def energy_mean_db(levels_db: list[float], durations_s: list[float]) -> float:
    """Equivalent level across concatenated trials; never average dB arithmetically."""
    _require(bool(levels_db) and len(levels_db) == len(durations_s), "empty/mismatched energy samples")
    for level, duration in zip(levels_db, durations_s):
        _number(level, "level")
        _require(_number(duration, "duration") > 0, "duration must be positive")
    anchor = max(levels_db)
    weighted = sum(t * 10 ** ((level - anchor) / 10) for level, t in zip(levels_db, durations_s))
    return anchor + 10 * math.log10(weighted / sum(durations_s))


def _validate_trial(trial: dict, protocol: dict, calibration_times: tuple) -> tuple[float, float]:
    """Validate one complete before/ascent/after bundle and return its time span."""
    setup, standard = protocol["setup"], protocol["standard"]
    meter = setup["meter"]
    _validate_setup(trial["conditions"])
    _text(trial["trial_id"], "trial_id")
    for key in ("instrument_export_record", "observation_record"):
        _text(trial[key], key)
    _require(trial["conditions"] == setup, "conditions differ from frozen staircase/microphone/command setup")
    _require(trial["measurement_window"] == standard["measurement_window"], "wrong acoustic measurement window")
    duration = _number(trial["duration_s"], "duration_s")
    _require(duration > 0, "duration_s must be positive")
    started = _time(trial["started_at"], "started_at")
    before, after = calibration_times
    _require(before <= started and started.timestamp() + duration <= after.timestamp(),
             "trial is not bracketed by field calibration")
    for key in ("certificate_valid_until", "calibrator_certificate_valid_until"):
        _require(after <= _time(meter[key], key), f"meter.{key}: certificate expired during collection")
    _require(trial["completed"] is True, "incomplete ascent cannot prove quiet successful ascent")
    _require(type(trial["stairs_completed"]) is int and trial["stairs_completed"] == setup["stair_count"],
             "not all stairs completed")
    _require(type(trial["fall_count"]) is int and trial["fall_count"] == 0, "fall or missing fall observation")
    for key in ("emergency_stop", "overload", "under_range", "external_noise_event"):
        _require(trial[key] is False, f"{key}: must be explicitly false")
    _require(math.isclose(_number(trial["distance_m"], "distance_m"), setup["route_distance_m"],
                          rel_tol=0.01, abs_tol=0.01), "wrong completed route distance")
    _require(_number(trial["max_stationary_interval_s"], "max_stationary_interval_s", 0)
             <= standard["max_stationary_interval_s"], "stopping cannot substitute for quiet ascent")
    background_intervals = {}
    for key in ("background_before", "background_after"):
        background = trial[key]
        _require(background["source_inactive"] is True, f"{key}: source must be inactive")
        _text(background["instrument_export_record"], f"{key}.instrument_export_record")
        background_duration = _number(background["duration_s"], f"{key}.duration_s")
        _require(background_duration >= duration, f"{key}: background interval shorter than ascent")
        background_start = _time(background["started_at"], f"{key}.started_at").timestamp()
        background_end = background_start + background_duration
        _require(before.timestamp() <= background_start and background_end <= after.timestamp(),
                 f"{key}: complete background interval must be bracketed by field calibration")
        background_intervals[key] = (background_start, background_end)
    _require(background_intervals["background_before"][1] <= started.timestamp(),
             "background_before must end no later than ascent starts")
    _require(background_intervals["background_after"][0] >= started.timestamp() + duration,
             "background_after must start no earlier than ascent ends")
    uncertainty = meter["expanded_uncertainty_db"]
    for metric in ("laeq_db", "lafmax_db"):
        level = _number(trial[metric], metric)
        _require(level - uncertainty > meter["lower_measurement_limit_db"]
                 and level + uncertainty < meter["upper_measurement_limit_db"],
                 f"{metric}: outside calibrated measurement range including uncertainty")
        for key in ("background_before", "background_after"):
            background = trial[key]
            background_level = _number(background[metric], f"{key}.{metric}")
            _require(background_level - uncertainty > meter["lower_measurement_limit_db"]
                     and background_level + uncertainty < meter["upper_measurement_limit_db"],
                     f"{key}.{metric}: outside calibrated range")
            _require(level - background_level - 2 * uncertainty >= standard["min_background_margin_db"],
                     f"{metric}: background masks source; use quieter site, not background subtraction")
    return background_intervals["background_before"][0], background_intervals["background_after"][1]


def evaluate(measurements: dict, protocol: dict) -> dict:
    """Return pass/fail/inconclusive. Invalid or incomplete evidence never passes."""
    result: dict = {
        "status": "inconclusive",
        "claim_scope": "supplied calibrated instrument readings; authenticity requires retained raw exports and observation",
        "target": "50% reduction of dB readings for both LAeq and LAFmax against each baseline",
        "energy_halving_delta_db": 10 * math.log10(0.5),
        "reasons": [],
    }
    try:
        _validate_protocol(protocol)
        result["protocol_sha256"] = fingerprint(protocol)
        result["measurements_sha256"] = fingerprint(measurements)
        _require(measurements["schema"] == "stair_quiet_v1.acoustic_measurements.v1",
                 "unsupported measurements schema")
        _require(measurements["protocol_sha256"] == result["protocol_sha256"], "protocol SHA256 mismatch")
        _require(measurements["all_attempts_included"] is True,
                 "all attempts, including failed ascents, must be included")
        standard = protocol["standard"]
        meter = protocol["setup"]["meter"]
        calibration = measurements["calibration"]
        _text(calibration["instrument_export_record"], "calibration.instrument_export_record")
        before, after = (_time(calibration[key], key) for key in ("before_at", "after_at"))
        _require(_time(protocol["frozen_at"], "frozen_at") <= before < after,
                 "freeze protocol before calibration and collection")
        readings = [_number(calibration[key], key) for key in ("before_db", "after_db")]
        _require(all(abs(x - meter["calibrator_reference_db"]) <= standard["max_field_calibration_error_db"]
                     for x in readings), "field calibration outside reference tolerance")
        _require(abs(readings[0] - readings[1]) <= standard["max_field_calibration_drift_db"],
                 "field calibration drift too large")
        policies = {key: item["sha256"] for key, item in standard["baselines"].items()}
        policies[POLICY_CANDIDATE] = protocol["candidate_sha256"]
        groups: dict[str, list] = {key: [] for key in policies}
        trials = measurements["trials"]
        _require(isinstance(trials, list) and bool(trials), "zero trials")
        seen, blocks = set(), {}
        previous_ascent_end = previous_bundle_end = None
        for trial in trials:
            bundle_start, bundle_end = _validate_trial(trial, protocol, (before, after))
            policy = trial["policy_id"]
            _require(policy in policies, f"unknown policy: {policy}")
            _require(trial["checkpoint_sha256"] == policies[policy], f"{policy}: wrong checkpoint SHA256")
            _require(trial["trial_id"] not in seen, "duplicate trial_id")
            seen.add(trial["trial_id"])
            block = trial["block_id"]
            _require(type(block) is int and block > 0, "block_id: positive integer required")
            blocks.setdefault(block, []).append(policy)
            started = _time(trial["started_at"], "started_at").timestamp()
            _require(previous_ascent_end is None or started >= previous_ascent_end,
                     "ascent intervals must be chronological and must not overlap")
            _require(previous_bundle_end is None or bundle_start >= previous_bundle_end,
                     "adjacent before/ascent/after acquisition bundles must not overlap or reuse background intervals")
            previous_ascent_end = started + trial["duration_s"]
            previous_bundle_end = bundle_end
            groups[policy].append(trial)
        _require(all(Counter(items) == Counter(policies.keys()) for items in blocks.values()),
                 "each interleaved block must include each baseline and candidate exactly once")
        names = list(policies)
        expected_order = []
        for index in range(len(blocks)):
            order = names[index % len(names):] + names[:index % len(names)]
            expected_order.extend((index + 1, name) for name in order)
        _require([(t["block_id"], t["policy_id"]) for t in trials] == expected_order,
                 "collection must follow the frozen rotating interleaved block order")
        for name, rows in groups.items():
            _require(len(rows) >= standard["min_repeats_per_policy"], f"{name}: insufficient repeats")
        candidate = groups[POLICY_CANDIDATE]
        for name, rows in groups.items():
            if name == POLICY_CANDIDATE:
                continue
            baseline_duration = statistics.median(t["duration_s"] for t in rows)
            tolerance = standard["max_completion_duration_relative_difference"]
            _require(all(abs(t["duration_s"] / baseline_duration - 1) <= tolerance + 1e-12
                         for t in candidate), f"candidate completion duration differs from {name}")
        aggregates = {
            name: {
                "n": len(rows),
                "laeq_db": energy_mean_db([t["laeq_db"] for t in rows], [t["duration_s"] for t in rows]),
                "lafmax_db": max(t["lafmax_db"] for t in rows),
                "median_duration_s": statistics.median(t["duration_s"] for t in rows),
            } for name, rows in groups.items()
        }
        result["aggregates"] = aggregates
        result["comparisons"] = {}
        uncertainty, fraction = meter["expanded_uncertainty_db"], standard["max_db_reading_fraction"]
        checks = []
        for name in standard["baselines"]:
            comparison = {}
            for metric in ("laeq_db", "lafmax_db"):
                baseline, value = aggregates[name][metric], aggregates[POLICY_CANDIDATE][metric]
                _require(baseline - uncertainty > 0, "literal percentage of nonpositive baseline dB is undefined")
                limit = fraction * (baseline - uncertainty)
                passed = value + uncertainty <= limit
                comparison[metric] = {
                    "baseline_db": baseline, "candidate_db": value,
                    "reading_reduction_percent": 100 * (1 - value / baseline),
                    "delta_db": value - baseline,
                    "energy_ratio": 10 ** ((value - baseline) / 10),
                    "candidate_plus_uncertainty_db": value + uncertainty,
                    "allowed_db_including_uncertainty_guard": limit,
                    "passed": passed,
                }
                checks.append(passed)
            result["comparisons"][name] = comparison
        result["status"] = "pass" if all(checks) else "fail"
        if result["status"] == "fail":
            result["reasons"].append("valid measurements do not meet both literal dB-half targets against both baselines")
    except (InvalidEvidence, KeyError, TypeError, ValueError, OverflowError) as exc:
        result["reasons"].append(str(exc))
    return result


def _write_new(path: str | Path, value: dict) -> None:
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser("setup-template", help="write an unfilled setup; it is not measurement evidence")
    setup.add_argument("--output", required=True)
    freeze = commands.add_parser("freeze", help="freeze conditions and candidate identity before collecting data")
    freeze.add_argument("setup")
    freeze.add_argument("--candidate-checkpoint", required=True)
    freeze.add_argument("--output", required=True)
    template = commands.add_parser("measurement-template", help="write empty trials in interleaved order")
    template.add_argument("--protocol", required=True)
    template.add_argument("--output", required=True)
    check = commands.add_parser("evaluate", help="evaluate actual meter exports, with no DSP or force conversion")
    check.add_argument("measurements")
    check.add_argument("--protocol", required=True)
    check.add_argument("--output")
    args = parser.parse_args(argv)
    try:
        if args.command == "setup-template":
            _write_new(args.output, setup_template())
        elif args.command == "freeze":
            _write_new(args.output, freeze_protocol(read_json(args.setup),
                       checkpoint_sha256(args.candidate_checkpoint)))
        elif args.command == "measurement-template":
            _write_new(args.output, measurement_template(read_json(args.protocol)))
        else:
            report = evaluate(read_json(args.measurements), read_json(args.protocol))
            if args.output:
                _write_new(args.output, report)
            print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
            return {"pass": 0, "fail": 1, "inconclusive": 2}[report["status"]]
        return 0
    except (OSError, InvalidEvidence, KeyError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "inconclusive", "reasons": [str(exc)]}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
