"""Uncalibrated SI-unit impact -> modal vibration -> monopole pressure model.

Never converts a force ratio into a sound-pressure ratio. No parameter in this
module is inferred from low-rate forces or fitted to the requested reduction.
See SOUND_MODEL.md for derivation, assumptions, sources and unsupported physics.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import wavfile


PROFILES_PATH = Path(__file__).with_name("sound_profiles.json")
EVENT_SCHEMA = "stair_quiet_v1.impact_events.v1"
STATUS = "uncalibrated_simulation"
WHEELS = ("fl", "fr", "hl", "hr")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value, name, *, positive=False):
    _require(type(value) in (float, int) and math.isfinite(value), f"{name}: finite number required")
    _require(not positive or value > 0, f"{name}: must be positive")
    return float(value)


def _vector(value, name):
    _require(isinstance(value, (list, tuple)) and len(value) == 3, f"{name}: three SI coordinates required")
    return np.array([_number(x, name) for x in value], dtype=np.float64)


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def load_profile(name="illustrative_compact_panel"):
    document = json.loads(PROFILES_PATH.read_text())
    _require(document["schema"] == "stair_quiet_v1.sound_profiles.v1", "unknown profile schema")
    result = deepcopy(document["profiles"][name])
    validate_profile(result)
    return result


def _validate_mode(mode):
    frequency = _number(mode["frequency_hz"], "frequency_hz", positive=True)
    _number(mode["modal_mass_kg"], "modal_mass_kg", positive=True)
    damping = _number(mode["damping_ratio"], "damping_ratio", positive=True)
    _require(damping < 1, "this modal model requires underdamped modes: 0 < damping_ratio < 1")
    _number(mode["force_participation"], "force_participation")
    _number(mode["radiating_area_m2"], "radiating_area_m2")
    return frequency, damping


def validate_profile(profile):
    """Reject false calibration claims, invalid physics and undersampling."""
    _require(profile.get("calibration_status") == STATUS,
             "this model only accepts explicitly uncalibrated_simulation profiles")
    _require(profile.get("actual_pass") in (None, False), "a profile cannot certify actual_pass")
    _require(isinstance(profile.get("parameter_basis"), str) and profile["parameter_basis"].strip(),
             "document the parameter assumptions")
    fs = profile["sample_rate_hz"]
    _require(type(fs) is int and 8000 <= fs <= 192000, "sample_rate_hz must be integer 8000..192000")
    for key in ("pulse_duration_s", "air_density_kg_m3", "sound_speed_m_s",
                "equivalent_source_radius_m", "nominal_band_limit_hz", "max_nominal_ka",
                "minimum_distance_radii", "tail_envelope_fraction",
                "max_source_duration_s", "max_render_duration_s",
                "max_source_receiver_distance_m"):
        _number(profile[key], key, positive=True)
    _require(_number(profile["pre_roll_s"], "pre_roll_s") >= 1,
             "pre_roll_s must be >= 1 for zero-state acoustic meter filters")
    _require(profile["tail_envelope_fraction"] < 1, "tail_envelope_fraction must be below 1")
    _require(profile["max_nominal_ka"] <= 0.5, "nominal compact-source ka limit cannot exceed 0.5")
    _require(profile["minimum_distance_radii"] >= 10, "receiver must be at least ten source radii away")
    for key, lower, upper in (("minimum_samples_per_pulse", 16, 10000),
                             ("minimum_samples_per_band_period", 10, 10000),
                             ("max_events", 1, 100000), ("lowpass_order", 2, 12)):
        _require(type(profile[key]) is int and lower <= profile[key] <= upper, f"invalid {key}")
    band = profile["nominal_band_limit_hz"]
    _require(fs * profile["pulse_duration_s"] >= profile["minimum_samples_per_pulse"],
             "contact pulse is undersampled; do not reconstruct it from 200 Hz forces")
    _require(fs / band >= profile["minimum_samples_per_band_period"], "acoustic band is undersampled")
    _vector(profile["receiver_position_m"], "receiver_position_m")
    radius = profile["equivalent_source_radius_m"]
    _require(profile["max_source_receiver_distance_m"] >= profile["minimum_distance_radii"] * radius,
             "maximum source distance must include the minimum receiver distance")
    ka = 2 * math.pi * band * radius / profile["sound_speed_m_s"]
    _require(ka <= profile["max_nominal_ka"], "nominal band exceeds compact-monopole validity")
    _require(isinstance(profile["modes"], list) and 1 <= len(profile["modes"]) <= 32, "expected 1..32 modes")
    for mode in profile["modes"]:
        frequency, _ = _validate_mode(mode)
        _require(frequency < band, "mode frequency must be below the nominal acoustic band limit")
        _require(abs(mode["radiating_area_m2"]) <= 4 * math.pi * radius ** 2,
                 "effective radiating area exceeds the equivalent sphere surface area")
    return profile


def half_sine_force(impulse_n_s, duration_s, sample_rate_hz):
    """Bin-average force [N]; its discrete integral is exactly impulse [N s].

    The sub-bin shape is an EXPLICIT assumed half sine, not reconstructed data.
    """
    impulse = _number(impulse_n_s, "impulse_n_s")
    duration = _number(duration_s, "duration_s", positive=True)
    _require(impulse >= 0, "negative normal impulse is invalid")
    _require(type(sample_rate_hz) is int and sample_rate_hz > 0, "positive integer sample rate required")
    n = math.ceil(duration * sample_rate_hz)
    edges = np.minimum(np.arange(n + 1, dtype=np.float64) / sample_rate_hz, duration)
    cumulative = 0.5 * impulse * (1 - np.cos(math.pi * edges / duration))
    return np.diff(cumulative) * sample_rate_hz


def modal_acceleration(force_n, sample_rate_hz, mode, *, equilibrium_force_n=0.0):
    """Exact ZOH discretization of the continuous modal ODE, output [m/s²].

    A nonzero equilibrium_force_n initializes the *static displaced state*,
    so a constant support load emits no acceleration. Starting from rest and
    suddenly applying a force is a transient and can radiate.
    """
    frequency, damping = _validate_mode(mode)
    force = np.asarray(force_n, dtype=np.float64)
    _require(force.ndim == 1 and force.size and np.isfinite(force).all(), "finite 1-D force required")
    _require(type(sample_rate_hz) is int and sample_rate_hz >= 10 * frequency,
             "modal integration needs at least ten samples per period")
    equilibrium = _number(equilibrium_force_n, "equilibrium_force_n")
    omega = 2 * math.pi * frequency
    gain = mode["force_participation"] / mode["modal_mass_kg"]
    # State is [q, qdot]; C x + D F is qddot, NOT displacement or velocity.
    a = np.array([[0.0, 1.0], [-omega ** 2, -2 * damping * omega]])
    b = np.array([[0.0], [gain]])
    c = np.array([[-omega ** 2, -2 * damping * omega]])
    d = np.array([[gain]])
    ad, bd, cd, dd, _ = signal.cont2discrete((a, b, c, d), 1 / sample_rate_hz, method="zoh")
    numerator, denominator = signal.ss2tf(ad, bd, cd, dd)
    # Stable C implementation of the exact discrete second-order system.
    zi = signal.lfilter_zi(numerator[0], denominator) * equilibrium
    acceleration, _ = signal.lfilter(numerator[0], denominator, force, zi=zi)
    _require(np.isfinite(acceleration).all(), "nonfinite modal solution")
    return acceleration


def _validate_events(document, profile):
    _require(document.get("schema") == EVENT_SCHEMA, "unsupported impact event schema")
    _require(document.get("actual_pass") in (None, False), "simulation events cannot claim actual_pass")
    duration = _number(document["duration_s"], "duration_s", positive=True)
    _require(duration <= profile["max_source_duration_s"], "source recording is too long for bounded rendering")
    window = document.get("measurement_window_s", [0.0, duration])
    _require(isinstance(window, (list, tuple)) and len(window) == 2,
             "measurement_window_s must contain [start, end] relative to source time zero")
    start, end = (_number(value, "measurement_window_s") for value in window)
    _require(0 <= start < end <= duration, "measurement window must lie inside the full source recording")
    events = document["events"]
    _require(isinstance(events, list) and len(events) <= profile["max_events"], "invalid/too many impact events")
    receiver = _vector(profile["receiver_position_m"], "receiver_position_m")
    seen, distances = set(), []
    previous = -math.inf
    for event in events:
        _require(not ({"force_N", "static_force_N", "contact_force_N"} & set(event)),
                 "input must contain dynamic event impulses, not stationary support force")
        t = _number(event["time_s"], "time_s")
        _require(0 <= t < duration and t >= previous, "events must be chronological inside duration_s")
        previous = t
        _require(t + profile["pulse_duration_s"] <= duration + 1e-12,
                 "contact pulse extends beyond source observation window")
        impulse = _number(event["normal_impulse_N_s"], "normal_impulse_N_s")
        _require(impulse >= 0, "negative normal impulse is invalid")
        _require(event["wheel"] in WHEELS, "wheel must be fl/fr/hl/hr")
        contact = event["contact_id"]
        _require(isinstance(contact, str) and contact.strip() and contact not in seen,
                 "contact_id must identify a unique dynamic loading episode; duplicate/replayed contact")
        seen.add(contact)
        # Different contacts may overlap, including tread+riser on one wheel.
        distance = float(np.linalg.norm(receiver - _vector(event["position_m"], "position_m")))
        _require(distance >= profile["minimum_distance_radii"] * profile["equivalent_source_radius_m"],
                 "receiver is too close to the equivalent source")
        _require(distance <= profile["max_source_receiver_distance_m"],
                 "event exceeds the profile maximum source-receiver distance")
        distances.append(distance)
    return duration, events, distances, (start, end)


def _unit_volume_acceleration(profile):
    fs = profile["sample_rate_hz"]
    # Bounding the exponential envelope is a truncation criterion, not a claim
    # that the integrated response error equals this small fraction.
    decay_rate = min(2 * math.pi * m["frequency_hz"] * m["damping_ratio"] for m in profile["modes"])
    decay_tail = -math.log(profile["tail_envelope_fraction"]) / decay_rate
    filter_tail = 16 / profile["nominal_band_limit_hz"]
    kernel_duration = profile["pulse_duration_s"] + decay_tail + filter_tail
    _require(kernel_duration < profile["max_render_duration_s"], "modal ringdown exceeds rendering budget")
    force = np.zeros(math.ceil(kernel_duration * fs) + 1, dtype=np.float64)
    pulse = half_sine_force(1.0, profile["pulse_duration_s"], fs)
    force[:len(pulse)] = pulse
    volume_acceleration = np.zeros_like(force)
    for mode in profile["modes"]:
        volume_acceleration += mode["radiating_area_m2"] * modal_acceleration(force, fs, mode)
    # Keep the assumed radiator inside its nominal compact-source band. This
    # causal filter has a documented roll-off, not an ideal brick-wall cutoff.
    sos = signal.butter(profile["lowpass_order"], profile["nominal_band_limit_hz"], fs=fs, output="sos")
    volume_acceleration = signal.sosfilt(sos, volume_acceleration)
    return volume_acceleration, kernel_duration


def render_sound(events_document, profile):
    """Return (float64 pressure_pa, metadata); never an actual acoustic pass.

    One common modal response is translated to each event source and propagated
    to the same fixed receiver. This is a local equivalent-source assumption,
    not a solved global staircase mode shape.
    """
    profile = deepcopy(profile)
    validate_profile(profile)
    duration, events, distances, window = _validate_events(events_document, profile)
    kernel, kernel_duration = _unit_volume_acceleration(profile)
    fs, pre_roll = profile["sample_rate_hz"], profile["pre_roll_s"]
    # Use the profile domain, never the observed event positions, to set the
    # averaging window. Otherwise a zero-impulse distant event can lower LAeq
    # solely by increasing its denominator.
    propagation_margin = profile["max_source_receiver_distance_m"] / profile["sound_speed_m_s"]
    observed_max_delay = max(distances, default=0) / profile["sound_speed_m_s"]
    length_s = pre_roll + duration + propagation_margin + kernel_duration
    _require(length_s <= profile["max_render_duration_s"], "total render duration exceeds bounded budget")
    n = math.ceil(length_s * fs) + 1
    # Output+kernel+time/interpolation scratch remain below this conservative
    # estimate. No GPU or parallel worker is ever used.
    estimated_bytes = 8 * (2 * n + 8 * len(kernel))
    _require(estimated_bytes <= 512 * 1024 ** 2, "estimated sound render workspace exceeds 512 MiB")
    pressure = np.zeros(n, dtype=np.float64)
    kernel_index = np.arange(len(kernel), dtype=np.float64)
    for event, distance in zip(events, distances):
        onset = (pre_roll + event["time_s"] + distance / profile["sound_speed_m_s"]) * fs
        start, stop = math.ceil(onset), min(n, math.ceil(onset + len(kernel)))
        times = np.arange(start, stop, dtype=np.float64) - onset
        delayed = np.interp(times, kernel_index, kernel, left=0.0, right=0.0)
        coefficient = profile["air_density_kg_m3"] * event["normal_impulse_N_s"] / (4 * math.pi * distance)
        pressure[start:stop] += coefficient * delayed
    _require(np.isfinite(pressure).all(), "nonfinite radiated pressure")
    metadata = {
        "schema": "stair_quiet_v1.sound_prediction.v1",
        "status": STATUS, "evidence_kind": STATUS, "actual_pass": False,
        "calibrated": False, "claim_scope": "assumed equivalent radiator only; not real M20/stair noise",
        "pressure_unit": "Pa", "sample_rate_hz": fs,
        "profile": profile, "profile_sha256": _hash(profile),
        "event_document_sha256": _hash(events_document),
        "event_provenance": deepcopy(events_document.get("provenance", {})),
        "event_count": len(events), "source_duration_s": duration,
        "pre_roll_s": pre_roll, "max_propagation_delay_s": propagation_margin,
        "propagation_margin_s": propagation_margin,
        "observed_max_propagation_delay_s": observed_max_delay,
        "kernel_duration_s": kernel_duration, "render_duration_s": len(pressure) / fs,
        "source_measurement_window_s": list(window),
        "source_measurement_duration_s": window[1] - window[0],
        "measurement_start_s": pre_roll + window[0],
        "measurement_end_s": pre_roll + window[1] + propagation_margin,
        "measurement_window_note": "full source history retained; fixed profile max_source_receiver_distance_m/sound_speed_m_s extends the end identically for all policies, independent of events; extra modal tail never averaged in",
        "dynamic_normal_impulse_sum_n_s": sum(e["normal_impulse_N_s"] for e in events),
        "estimated_workspace_bytes": estimated_bytes,
        "nominal_band_limit_hz": profile["nominal_band_limit_hz"],
        "radiation": "free-field compact equivalent monopole; rho/(4*pi*r) * volume_acceleration(t-r/c)",
        "source_motion": "each event position held fixed through its ringdown",
        "numerics": "bin-integrated half sine; exact ZOH modal dynamics; causal Butterworth; linear fractional propagation delay",
        "limitations": [
            "Shared assumed pulse width cannot represent a policy's actual longer contact; equal impulses can yield equal predictions.",
            "Low-rate event timing does not resolve high-frequency modal phase; the pulse/damping grid excludes substep timing and modal-frequency uncertainty.",
            "Not a complete acoustic training reward and not evidence for actual robot promotion or a real-world reduction pass.",
        ],
    }
    return pressure, metadata


def paired_profile_sweep(baseline_events, candidate_events, profile, *, metric_fn=None):
    """Cross pulse and damping assumptions, applying EACH profile to BOTH runs.

    No policy-specific parameter override is accepted. The result contains
    predictions and sensitivity only; actual_pass is always false.
    """
    validate_profile(profile)
    if metric_fn is None:
        from .audio import measure_pressure
        metric_fn = measure_pressure
    sensitivity = profile["sensitivity"]
    pulse_factors = sensitivity["pulse_duration_factors"]
    damping_factors = sensitivity["damping_factors"]
    _require(isinstance(pulse_factors, list) and isinstance(damping_factors, list)
             and 1 <= len(pulse_factors) * len(damping_factors) <= 25, "invalid sensitivity grid")
    rows = []
    for pulse_factor in pulse_factors:
        _number(pulse_factor, "pulse_factor", positive=True)
        for damping_factor in damping_factors:
            _number(damping_factor, "damping_factor", positive=True)
            shared = deepcopy(profile)
            shared["pulse_duration_s"] *= pulse_factor
            for mode in shared["modes"]:
                mode["damping_ratio"] *= damping_factor
            validate_profile(shared)
            row = {"pulse_duration_factor": pulse_factor, "damping_factor": damping_factor,
                   "shared_profile_sha256": _hash(shared)}
            for label, document in (("baseline", baseline_events), ("candidate", candidate_events)):
                pressure, metadata = render_sound(document, shared)
                measured = metric_fn(pressure, shared["sample_rate_hz"],
                                     start_s=metadata["measurement_start_s"], end_s=metadata["measurement_end_s"])
                row[label] = {"metrics": measured, "event_document_sha256": metadata["event_document_sha256"],
                              "profile_sha256": metadata["profile_sha256"],
                              "source_duration_s": metadata["source_duration_s"],
                              "status": STATUS, "actual_pass": False}
            rows.append(row)
    return {"status": STATUS, "actual_pass": False, "rows": rows,
            "scope": "paired sensitivity grid; no best-profile selection and no real-world acceptance"}


def write_sound(prefix, pressure_pa, metadata):
    """Write raw float64 Pa NPZ, float32 Pa WAV and a mandatory JSON sidecar.

    WAV samples are physical Pa, NOT normalized PCM. Generic players are not
    calibrated pressure renderers and may clip values outside [-1, 1].
    """
    _require(metadata.get("status") == STATUS and metadata.get("actual_pass") is False,
             "only explicitly uncalibrated predictions may be written")
    pressure = np.asarray(pressure_pa, dtype=np.float64)
    _require(pressure.ndim == 1 and pressure.size and np.isfinite(pressure).all(), "finite pressure required")
    _require(np.abs(pressure).max() <= np.finfo(np.float32).max,
             "pressure cannot be represented in float32 WAV without overflow")
    prefix = Path(prefix)
    paths = [Path(str(prefix) + suffix) for suffix in (".npz", ".wav", ".json")]
    _require(not any(p.exists() for p in paths), "output exists; preserve source predictions")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    stored = deepcopy(metadata)
    stored["wav_encoding"] = "IEEE float32; one sample unit = 1 Pa; no normalization"
    stored["wav_requires_metadata_sidecar"] = True
    stored["wav_generic_playback_warning"] = "not calibrated audio playback; values above 1 Pa can clip in players"
    text = json.dumps(stored, sort_keys=True, allow_nan=False)
    fs = metadata["sample_rate_hz"]
    np.savez_compressed(paths[0], pressure_pa=pressure, time_s=np.arange(len(pressure)) / fs,
                        sample_rate_hz=np.asarray(fs), metadata_json=np.asarray(text))
    wavfile.write(paths[1], fs, pressure.astype(np.float32))
    paths[2].write_text(json.dumps(stored, indent=2, allow_nan=False) + "\n")
    return [str(p) for p in paths]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("events", type=Path)
    parser.add_argument("--profile", default="illustrative_compact_panel")
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args(argv)
    document = json.loads(args.events.read_text())
    pressure, metadata = render_sound(document, load_profile(args.profile))
    paths = write_sound(args.output_prefix, pressure, metadata)
    print(json.dumps({"status": STATUS, "actual_pass": False, "outputs": paths,
                      "event_count": metadata["event_count"], "profile_sha256": metadata["profile_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
