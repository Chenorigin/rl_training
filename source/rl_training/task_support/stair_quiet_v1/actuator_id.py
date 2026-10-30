"""Offline kinematic actuator-response identification; never controls a robot.

CLI:
  python -m task_support.stair_quiet_v1.actuator_id recording.npz metadata.json out.json

NPZ arrays (SI): state_time_s[N], command_time_s[M], command[M,J],
q_rad[N,J], dq_rad_s[N,J]. Commands must be final issued joint position (rad)
or wheel velocity (rad/s), NOT normalized policy actions. Separate recordings
are required for suspended and loaded_contact conditions. Metadata declares
schema="m20_actuator_recording_v1", origin="hardware_recording", joint_names,
command_kinds ("position"/"velocity"), condition, common_clock, and
state_timestamp_kind ("capture"/"host_receive"). Capture timestamps require
clock alignment before this tool; receive timestamps only identify the
combined transport/sensing/actuation response.

This estimates acceleration response, not inertia, torque, stiffness, contact
parameters, firmware safety limits, or a calibrated actuator. A hardware origin
declaration is provenance supplied by the caller, not independent attestation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA = "m20_actuator_recording_v1"


def _array(value, name, ndim):
    a = np.asarray(value)
    if a.dtype.kind not in "fi" or a.ndim != ndim or not np.isfinite(a).all():
        raise ValueError(f"{name} must be a finite numeric {ndim}D array")
    return a.astype(np.float64)


def validate_recording(arrays, metadata, *, allow_synthetic=False):
    if metadata.get("schema") != SCHEMA:
        raise ValueError("Unknown actuator recording schema")
    origins = ("hardware_recording", "synthetic_test_fixture") if allow_synthetic else ("hardware_recording",)
    if metadata.get("origin") not in origins:
        raise ValueError("Identification requires a hardware recording; no fabricated calibration")
    if metadata.get("condition") not in ("suspended", "loaded_contact"):
        raise ValueError("Separate suspended and loaded_contact recordings; mixed/unknown is invalid")
    if not isinstance(metadata.get("common_clock"), str) or not metadata["common_clock"].strip():
        raise ValueError("A common aligned timestamp clock must be declared")
    if metadata.get("state_timestamp_kind") not in ("capture", "host_receive"):
        raise ValueError("Declare capture or host_receive state timestamps")
    t = _array(arrays["state_time_s"], "state_time_s", 1)
    tc = _array(arrays["command_time_s"], "command_time_s", 1)
    q = _array(arrays["q_rad"], "q_rad", 2)
    dq = _array(arrays["dq_rad_s"], "dq_rad_s", 2)
    cmd = _array(arrays["command"], "command", 2)
    if not 150 <= len(t) <= 1_000_000 or not 2 <= len(tc) <= 1_000_000:
        raise ValueError("Insufficient or oversized timestamped recording")
    if q.shape != dq.shape or q.shape[0] != len(t) or not 1 <= q.shape[1] <= 16:
        raise ValueError("Joint state shapes disagree or joint count is invalid")
    if cmd.shape != (len(tc), q.shape[1]):
        raise ValueError("Command shape differs from declared joint states")
    for clock, name in ((t, "state"), (tc, "command")):
        if np.any(np.diff(clock) <= 0):
            raise ValueError(f"{name} timestamps must be strictly increasing; no resets/duplicates")
    if t[-1] - t[0] < 2.0 or np.max(np.diff(t)) > .05:
        raise ValueError("Need >=2 seconds with state gaps <=50 ms; split/reset gaps first")
    names, kinds = metadata.get("joint_names"), metadata.get("command_kinds")
    if (not isinstance(names, list) or len(names) != q.shape[1] or len(set(names)) != len(names)
            or any(not isinstance(x, str) or not x for x in names)):
        raise ValueError("Provide unique joint names in array order")
    if not isinstance(kinds, list) or len(kinds) != len(names) or any(x not in ("position", "velocity") for x in kinds):
        raise ValueError("Each command must declare physical position or velocity units")
    return t, tc, q, dq, cmd


def _features(q, dq, command, kind):
    error = command - (q if kind == "position" else dq)
    terms = [error]
    if kind == "position":
        terms.append(-dq)
    terms.extend((-np.tanh(dq / .05), np.ones_like(dq)))
    return np.column_stack(terms)


def _fit(x, y):
    scale = np.linalg.norm(x, axis=0)
    if np.any(scale <= 1e-10):
        raise ValueError("Insufficient excitation: a response feature is constant zero")
    normalized = x / scale
    singular = np.linalg.svd(normalized, compute_uv=False)
    condition = float(singular[0] / max(singular[-1], 1e-30))
    if condition > 1e6:
        raise ValueError("Insufficient independent excitation; response parameters are not identifiable")
    theta = np.linalg.lstsq(normalized, y, rcond=None)[0] / scale
    return theta, condition


def fit_recording(arrays, metadata, *, delay_candidates_s=None, allow_synthetic=False):
    """Fit on first 60%, select delay on next 20%, report untouched final 20%.

    The one-step forward-velocity model uses actual variable sample intervals.
    All delay candidates use the same sample support. Commands use zero-order
    hold at t-delay, never future interpolation. Fitting is ordinary least
    squares on measured acceleration; no controller or policy is optimized.
    """
    t, tc, q, dq, cmd = validate_recording(arrays, metadata, allow_synthetic=allow_synthetic)
    delays = _array(np.arange(0., .081, .005) if delay_candidates_s is None else delay_candidates_s,
                    "delay_candidates_s", 1)
    if not 2 <= len(delays) <= 101 or np.any(delays < 0) or np.any(delays > .5) or np.any(np.diff(delays) <= 0):
        raise ValueError("Delay grid must contain 2..101 increasing values in [0,0.5] seconds")
    valid = (t[:-1] - delays[-1] >= tc[0]) & (t[:-1] <= tc[-1])
    ids = np.flatnonzero(valid)
    if len(ids) < 150 or np.any(np.diff(ids) != 1):
        raise ValueError("Insufficient continuous command/state overlap across the entire delay grid")
    # Avoid treating an indefinitely held final command after recording stopped
    # as valid evidence; overlap above ends at the final command timestamp.
    dt = np.diff(t)[ids]
    target = (dq[ids + 1] - dq[ids]) / dt[:, None]
    n = len(ids)
    train_end, validation_end = int(.6 * n), int(.8 * n)
    train, validation, test = slice(0, train_end), slice(train_end, validation_end), slice(validation_end, n)
    if min(train_end, validation_end - train_end, n - validation_end) < 30:
        raise ValueError("Each chronological split needs at least 30 samples")
    result = {
        "schema": "m20_actuator_response_fit_v1",
        "status": "offline_kinematic_fit_not_calibration",
        "hardware_calibrated": False,
        "hardware_authenticity_verified": False,
        "recording_origin_claim": metadata["origin"],
        "condition": metadata["condition"],
        "state_timestamp_kind": metadata["state_timestamp_kind"],
        "delay_meaning": "command publication to observed state; includes unseparated sensing/transport/actuation",
        "fit_equation": "ddq = response_gain*(delayed_command-q_or_dq) - damping*dq(position only) - friction_accel*tanh(dq/0.05) + bias_accel",
        "parameter_limit": "Kinematic coefficients only: inertia/torque/contact stiffness and physical Kp/Kd are not identified",
        "split": {"method": "chronological 60/20/20; final test never selects delay",
                  "train_samples": train_end, "validation_samples": validation_end - train_end,
                  "test_samples": n - validation_end,
                  "train_end_state_time_s": float(t[ids[train_end - 1]]),
                  "test_start_state_time_s": float(t[ids[validation_end]])},
        "state_dt_s": {"min": float(dt.min()), "median": float(np.median(dt)), "max": float(dt.max())},
        "joints": [],
        "limitations": [
            "No automatically approved deployment parameters or policy changes",
            "Loaded-contact response is condition dependent; not a separated free actuator model",
            "Suspended response does not identify wheel-ground contact or acoustic radiation",
            "Finite-difference acceleration amplifies encoder velocity noise",
            "One-step held-out fit does not establish closed-loop stability or safe total PD limits",
        ],
    }
    for j, (name, kind) in enumerate(zip(metadata["joint_names"], metadata["command_kinds"])):
        candidates = []
        for delay in delays:
            ci = np.searchsorted(tc, t[ids] - delay, side="right") - 1
            x = _features(q[ids, j], dq[ids, j], cmd[ci, j], kind)
            theta, condition = _fit(x[train], target[train, j])
            residual = (x @ theta - target[:, j]) * dt
            candidates.append((float(np.mean(residual[validation] ** 2)), theta, residual, condition))
        selected = min(range(len(candidates)), key=lambda k: candidates[k][0])
        score, theta, residual, condition = candidates[selected]
        # An unphysical fit is retained as a rejected fit, not silently clipped
        # to appealing control gains. No selection uses the final test split.
        physical_signs = bool(theta[0] > 0 and np.all(theta[1:-1] >= 0))
        near = [float(delay) for delay, candidate in zip(delays, candidates)
                if candidate[0] <= score * 1.05 + 1e-12]
        names = (["response_gain_s2", "damping_s1", "friction_accel_rad_s2", "bias_accel_rad_s2"]
                 if kind == "position" else ["response_gain_s1", "friction_accel_rad_s2", "bias_accel_rad_s2"])
        result["joints"].append({
            "joint": name, "command_kind": kind, "selected_delay_s": float(delays[selected]),
            "delay_grid_boundary_selected": selected in (0, len(delays) - 1),
            "near_optimal_validation_delays_s": near,
            "coefficients": dict(zip(names, map(float, theta))),
            "physically_expected_coefficient_signs": physical_signs,
            "training_design_condition_number": condition,
            "train_velocity_rmse_rad_s": float(np.sqrt(np.mean(residual[train] ** 2))),
            "validation_velocity_rmse_rad_s": float(np.sqrt(np.mean(residual[validation] ** 2))),
            "heldout_velocity_rmse_rad_s": float(np.sqrt(np.mean(residual[test] ** 2))),
            "heldout_velocity_p95_abs_error_rad_s": float(np.quantile(np.abs(residual[test]), .95)),
            "validation_delay_sweep": [{"delay_s": float(d), "velocity_mse": item[0]} for d, item in zip(delays, candidates)],
        })
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("metadata", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("Output exists; preserve the previous identification report")
    if args.recording.stat().st_size > 256 * 1024 ** 2:
        raise ValueError("Recording exceeds 256 MiB offline input limit; split it first")
    metadata = json.loads(args.metadata.read_text())
    with np.load(args.recording, allow_pickle=False) as archive:
        report = fit_recording(archive, metadata)
    report["inputs"] = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in (args.recording, args.metadata)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "status": report["status"], "hardware_calibrated": False}))


if __name__ == "__main__":
    main()
