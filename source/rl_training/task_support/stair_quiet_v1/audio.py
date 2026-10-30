"""Compute sound-pressure metrics from Pa samples or calibrated, uncompressed WAV.

This is a numerical instrument, not an IEC-certified meter or a robot acceptance
decision. Generic Pa arrays may be simulated; ``measure_pressure`` never assigns
real-world provenance. WAV results remain provisional when acquisition evidence
is missing. No force-to-dB conversion, certificate fabrication or auto-normalizing
recordings is performed. See AUDIO.md for equations, sources and limitations.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import struct
import warnings

import numpy as np
from scipy import signal
from scipy.io import wavfile


REFERENCE_PRESSURE_PA = 20e-6
FAST_TIME_CONSTANT_S = 0.125
OVERSAMPLE = 4
MIN_HISTORY_S = 1.0
MAX_INPUT_BYTES = 64 * 1024 * 1024
MAX_TOTAL_SAMPLES = 3_000_000
SUPPORTED_RATES = (48000, 96000)
MAX_CALIBRATION_DRIFT_DB = 0.5
MIN_BACKGROUND_MARGIN_DB = 10.0


class InvalidAudio(ValueError):
    pass


def _require(ok, message):
    if not ok:
        raise InvalidAudio(message)


def _finite(value, name):
    _require(type(value) in (int, float) and math.isfinite(value), f"{name}: finite number required")
    return float(value)


def _sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _level(amplitude):
    return None if amplitude <= 0 else 20 * math.log10(amplitude / REFERENCE_PRESSURE_PA)


def a_weighting_sos(sample_rate_hz):
    """Analog A poles/zeros -> bilinear SOS, normalized at digital 1 kHz.

Call at the oversampled rate. The analog poles are given in Hz below, hence 2π.
Four zeros at DC and six poles form the A weighting transfer function.
"""
    poles = -2 * np.pi * np.array([20.598997, 20.598997, 107.65265, 737.86223,
                                  12194.217, 12194.217])
    z, p, k = signal.bilinear_zpk(np.zeros(4), poles, 1.0, fs=sample_rate_hz)
    sos = signal.zpk2sos(z, p, k)
    _, response = signal.sosfreqz(sos, worN=[1000.0], fs=sample_rate_hz)
    sos[0, :3] /= abs(response[0])
    return sos


def _fast_average_squared(squared_pressure_pa2, sample_rate_hz):
    alpha = math.exp(-1 / (sample_rate_hz * FAST_TIME_CONSTANT_S))
    return signal.lfilter([1 - alpha], [1, -alpha], squared_pressure_pa2)


def measure_pressure(pressure_pa: np.ndarray, sample_rate_hz: int, *,
                     start_s: float = 0.0, end_s: float | None = None) -> dict:
    """Compute LAeq, LAFmax and unweighted sample peak; never an acceptance pass.

    Supply the complete prehistory and select the measurement interval with
    start_s/end_s. Filtering starts from zero at sample zero. Raw pressure input
    is already acoustic pressure, so this low-level function does not remove DC.
    """
    _require(type(sample_rate_hz) is int and sample_rate_hz in SUPPORTED_RATES,
             f"sample_rate_hz must be one of {SUPPORTED_RATES}")
    pressure = np.asarray(pressure_pa, dtype=np.float64)
    _require(pressure.ndim == 1 and 0 < pressure.size <= MAX_TOTAL_SAMPLES,
             "pressure must be a nonempty bounded mono sequence")
    _require(np.isfinite(pressure).all(), "non-finite pressure")
    _require(np.max(np.abs(pressure)) <= 1e6, "pressure exceeds numerical instrument range")
    start = _finite(start_s, "start_s")
    end = pressure.size / sample_rate_hz if end_s is None else _finite(end_s, "end_s")
    _require(0 <= start < end <= pressure.size / sample_rate_hz, "invalid measurement interval")
    left, right = round(start * sample_rate_hz), round(end * sample_rate_hz)
    _require(right > left, "measurement interval contains zero samples")
    segment = pressure[left:right]
    rms = float(np.sqrt(np.mean(segment ** 2)))
    peak = float(np.max(np.abs(segment)))
    # Fourfold upsampling reduces bilinear warping; A weighting remains causal.
    interpolation = signal.firwin(64 * OVERSAMPLE + 1, 1 / OVERSAMPLE, window=("kaiser", 8.6))
    upsampled = signal.resample_poly(pressure, OVERSAMPLE, 1, window=interpolation)
    weighted = signal.sosfilt(a_weighting_sos(sample_rate_hz * OVERSAMPLE), upsampled)
    del upsampled
    weighted **= 2
    a_energy = float(np.mean(weighted[left * OVERSAMPLE:right * OVERSAMPLE]))
    fast_energy = _fast_average_squared(weighted, sample_rate_hz * OVERSAMPLE)
    fast_max = float(np.max(fast_energy[left * OVERSAMPLE:right * OVERSAMPLE]))
    cautions = []
    if start < MIN_HISTORY_S:
        cautions.append("less_than_one_second_of_filter_history; zero-initialized Fast/A startup is included")
    if peak == 0:
        cautions.append("silent_measurement; logarithmic levels are undefined, not zero dB")
    return {
        "evidence_kind": "pressure_array_numerical_metrics",
        "sample_rate_hz": sample_rate_hz,
        "start_s": left / sample_rate_hz,
        "end_s": right / sample_rate_hz,
        "duration_s": (right - left) / sample_rate_hz,
        "laeq_db": _level(math.sqrt(max(0.0, a_energy))),
        "lafmax_db": _level(math.sqrt(max(0.0, fast_max))),
        "lzpeak_db": _level(peak),
        "peak_pressure_pa": peak,
        "rms_pressure_pa": rms,
        "a_weighted_rms_pressure_pa": math.sqrt(max(0.0, a_energy)),
        "warnings": cautions,
        "method": {
            "reference_pressure_pa": REFERENCE_PRESSURE_PA,
            "a_weighting": "4x polyphase interpolation, analog A poles/zeros bilinear SOS, 1kHz normalization",
            "fast": "exponential mean of squared A-weighted pressure, tau=0.125s, initial energy zero",
            "peak": "max(abs(unweighted input pressure samples)); not Fast max or continuous-time intersample peak",
            "iec_certified": False,
        },
    }


def _wav_header(path):
    """Read PCM format/valid bits before allocating; necessary for 24-bit rails."""
    size = Path(path).stat().st_size
    _require(12 <= size <= MAX_INPUT_BYTES, "WAV empty, truncated or exceeds 64 MiB resource bound")
    with Path(path).open("rb") as stream:
        header = stream.read(12)
        _require(header[:4] in (b"RIFF", b"RIFX") and header[8:] == b"WAVE",
                 "only RIFF/RIFX uncompressed WAVE is supported")
        endian = "<" if header[:4] == b"RIFF" else ">"
        fmt, data_bytes = None, None
        while stream.tell() + 8 <= size:
            name, length = struct.unpack(endian + "4sI", stream.read(8))
            _require(stream.tell() + length <= size, "truncated WAV chunk")
            if name == b"fmt ":
                _require(16 <= length <= 4096, "invalid WAV format chunk")
                value = stream.read(length)
                code, channels, rate, _, alignment, bits = struct.unpack(endian + "HHIIHH", value[:16])
                valid_bits = bits
                if code == 65534:
                    _require(length >= 40, "truncated WAVE_FORMAT_EXTENSIBLE")
                    valid_bits = struct.unpack(endian + "H", value[18:20])[0] or bits
                    code = struct.unpack(endian + "I", value[24:28])[0]
                _require(code in (1, 3), "compressed WAV is not calibrated PCM")
                _require(bits in (8, 16, 24, 32, 64) and 0 < valid_bits <= bits, "unsupported PCM bit depth")
                _require(code != 3 or bits in (32, 64), "unsupported float PCM bit depth")
                _require(1 <= channels <= 8 and alignment == channels * (bits // 8), "invalid channel/block alignment")
                fmt = {"encoding": code, "channels": channels, "sample_rate_hz": rate,
                       "container_bits": bits, "valid_bits": valid_bits, "alignment": alignment}
            elif name == b"data":
                _require(data_bytes is None, "multiple WAV data chunks are not supported")
                data_bytes = length
                stream.seek(length, 1)
            else:
                stream.seek(length, 1)
            if length % 2:
                stream.seek(1, 1)
    _require(fmt is not None and data_bytes is not None, "missing PCM format/data")
    _require(data_bytes > 0 and data_bytes % fmt["alignment"] == 0, "empty or incomplete PCM frames")
    frames = data_bytes // fmt["alignment"]
    _require(frames * fmt["channels"] <= MAX_TOTAL_SAMPLES, "WAV exceeds total-sample resource bound")
    fmt["frames"] = frames
    return fmt


def load_pcm_wav(path, channel=0):
    """Return normalized signed PCM samples and provenance, without peak normalization."""
    path = Path(path)
    header = _wav_header(path)
    _require(header["sample_rate_hz"] in SUPPORTED_RATES, f"WAV rate must be one of {SUPPORTED_RATES}")
    _require(type(channel) is int and 0 <= channel < header["channels"], "select an existing channel explicitly")
    with warnings.catch_warnings():
        warnings.simplefilter("error", wavfile.WavFileWarning)
        rate, raw = wavfile.read(path)
    _require(raw.shape[0] == header["frames"] and rate == header["sample_rate_hz"], "PCM header/decoded length mismatch")
    selected = raw if raw.ndim == 1 else raw[:, channel]
    if selected.dtype.kind == "u":
        _require(selected.dtype.itemsize == 1, "unsupported unsigned PCM")
        samples = (selected.astype(np.float64) - 128.0) / 128.0
    elif selected.dtype.kind == "i":
        # SciPy returns 24-bit packed PCM LEFT justified in int32.
        samples = selected.astype(np.float64) / 2 ** (selected.dtype.itemsize * 8 - 1)
    elif selected.dtype.kind == "f":
        samples = selected.astype(np.float64)
    else:
        raise InvalidAudio("unsupported decoded PCM dtype")
    _require(np.isfinite(samples).all(), "non-finite PCM samples")
    _require(np.max(np.abs(samples)) <= 16, "PCM far outside normalized full scale")
    # Float PCM has relative precision, not a fixed full-scale quantization step.
    quantum = 2.0 ** (1 - header["valid_bits"]) if header["encoding"] == 1 else 0.0
    rail = 1 - quantum if header["encoding"] == 1 else 1.0
    clipped = int(np.count_nonzero((samples <= -1.0) | (samples >= rail)))
    dc = float(np.mean(samples))
    samples -= dc
    rms = float(np.sqrt(np.mean(samples ** 2)))
    issues = []
    if clipped:
        issues.append("digital_rail_or_out_of_range_samples")
    if rms <= 16 * quantum:
        issues.append("silent_or_underresolved_recording")
    if abs(dc) > 0.1:
        issues.append("large_ADC_DC_offset")
    return samples, {
        "path": str(path.resolve()), "sha256": _sha256(path), **header, "selected_channel": channel,
        "duration_s": len(samples) / rate, "decoded_dtype": str(selected.dtype),
        "normalized_rms": rms, "removed_DC_full_scale": dc,
        "quantization_step": quantum, "rail_sample_count": clipped, "issues": issues,
    }


def calibrate(path, reference_level_db, frequency_hz=1000.0, channel=0):
    samples, metadata = load_pcm_wav(path, channel)
    rate = metadata["sample_rate_hz"]
    reference = _finite(reference_level_db, "reference_level_db")
    frequency = _finite(frequency_hz, "calibration_frequency_hz")
    _require(-40 <= reference <= 180 and 20 <= frequency <= 20000, "invalid calibration reference")
    _require(metadata["duration_s"] >= 1.0, "calibration needs at least one second of steady tone")
    _require(not metadata["issues"], "invalid calibration: " + ", ".join(metadata["issues"]))
    frequencies, powers = signal.periodogram(samples, rate, window="hann", scaling="spectrum")
    peak_index = int(np.argmax(powers))
    detected = float(frequencies[peak_index])
    _require(abs(detected - frequency) <= max(2.0, frequency * 0.002),
             "calibration tone frequency mismatch")
    tone_fraction = float(np.sum(powers[max(0, peak_index - 3):peak_index + 4]) / np.sum(powers))
    _require(tone_fraction >= 0.99, "calibration is not a sufficiently pure reference tone")
    block = round(0.25 * rate)
    block_rms = np.array([np.sqrt(np.mean(samples[i:i + block] ** 2))
                         for i in range(0, len(samples) - block + 1, block)])
    stability_db = float(20 * np.log10(block_rms.max() / block_rms.min()))
    _require(math.isfinite(stability_db) and stability_db <= 0.2, "unstable calibration tone")
    reference_pa = REFERENCE_PRESSURE_PA * 10 ** (reference / 20)
    return {
        **metadata, "reference_level_db": reference, "reference_rms_pa": reference_pa,
        "expected_frequency_hz": frequency, "detected_frequency_hz": detected,
        "tone_energy_fraction": tone_fraction, "within_record_level_span_db": stability_db,
        "pa_per_normalized_sample": reference_pa / metadata["normalized_rms"],
    }


def _timestamp(value):
    _require(isinstance(value, str), "timestamp missing")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    _require(parsed.tzinfo is not None, "timestamp needs timezone")
    return parsed.timestamp()


def _manifest_issues(manifest, files):
    """Validate claimed provenance without turning it into fabricated certification."""
    issues = []
    if not isinstance(manifest, dict):
        return ["acquisition_manifest_missing"]
    for key in ("recorder_id", "microphone_id", "gain_setting", "calibrator_id",
                "calibrator_certificate_record", "microphone_response_record",
                "measurement_window_record"):
        if not isinstance(manifest.get(key), str) or not manifest[key].strip():
            issues.append(f"missing_evidence:{key}")
    for key in ("same_gain_all_files", "agc_disabled", "processing_disabled", "background_source_inactive"):
        if manifest.get(key) is not True:
            issues.append(f"not_confirmed:{key}")
    try:
        uncertainty = _finite(manifest["expanded_uncertainty_db"], "expanded_uncertainty_db")
        _require(uncertainty > 0, "positive uncertainty required")
    except (KeyError, InvalidAudio):
        issues.append("documented_expanded_uncertainty_missing")
    previous_end = None
    for role in ("calibration_before", "background_before", "recording", "background_after", "calibration_after"):
        try:
            record, actual = manifest["files"][role], files[role]
            _require(record["sha256"] == actual["sha256"], f"{role}: manifest SHA256 mismatch")
            _require(record["gain_setting"] == manifest["gain_setting"], f"{role}: gain changed")
            start = _timestamp(record["started_at"])
            _require(previous_end is None or start >= previous_end, f"{role}: overlapping/out-of-order acquisition")
            previous_end = start + actual["duration_s"]
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            issues.append(f"provenance:{role}:{exc}")
    try:
        _require(previous_end is not None and
                 _timestamp(manifest["calibrator_certificate_valid_until"]) >= previous_end,
                 "calibrator certificate expired")
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        issues.append(f"calibration_certificate_validity:{exc}")
    return issues


def analyze_recording(recording_path, calibration_before_path, *, reference_level_db,
                      start_s, end_s=None, calibration_after_path=None,
                      background_before_path=None, background_after_path=None,
                      calibration_frequency_hz=1000.0, channel=0, manifest=None):
    """Return raw calculated metrics plus evidence quality; never accept 50% reduction."""
    result = {
        "schema": "stair_quiet_v1.calibrated_pcm_metrics.v1",
        "status": "inconclusive", "evidence_kind": "calibrated_wav_numerical_metrics",
        "physical_measurement_status": "not_independently_verified",
        "claim_scope": "numerical metrics from supplied WAV and declared calibration; source authenticity is not established",
        "raw_metrics_are_provisional": True,
        "robot_noise_reduction_claim": "not_evaluated",
        "iec_certified": False, "reasons": [], "files": {},
    }
    try:
        if manifest is not None:
            json.dumps(manifest, allow_nan=False)
        before = calibrate(calibration_before_path, reference_level_db, calibration_frequency_hz, channel)
        result["files"]["calibration_before"] = before
        scale, rate = before["pa_per_normalized_sample"], before["sample_rate_hz"]
        if calibration_after_path is None:
            result["reasons"].append("calibration_after_missing")
        else:
            after = calibrate(calibration_after_path, reference_level_db, calibration_frequency_hz, channel)
            result["files"]["calibration_after"] = after
            _require(after["sample_rate_hz"] == rate, "calibration sample rate changed")
            drift = 20 * math.log10(after["normalized_rms"] / before["normalized_rms"])
            result["calibration_drift_db"] = drift
            if abs(drift) > MAX_CALIBRATION_DRIFT_DB:
                result["reasons"].append("calibration_drift_exceeds_limit")
        samples, metadata = load_pcm_wav(recording_path, channel)
        result["files"]["recording"] = metadata
        _require(metadata["sample_rate_hz"] == rate, "recording/calibration sample rate mismatch")
        _require(_finite(start_s, "start_s") >= MIN_HISTORY_S,
                 "recording measurement needs at least one second of recorded filter prehistory")
        result["reasons"].extend(f"recording:{issue}" for issue in metadata["issues"])
        samples *= scale
        metrics = measure_pressure(samples, rate, start_s=start_s, end_s=end_s)
        result["metrics"] = metrics
        if metrics["rms_pressure_pa"] == 0:
            result["reasons"].append("measurement_window_silent")
        del samples
        duration = metrics["duration_s"]
        for role, path in (("background_before", background_before_path), ("background_after", background_after_path)):
            if path is None:
                result["reasons"].append(f"{role}_missing")
                continue
            samples, metadata = load_pcm_wav(path, channel)
            result["files"][role] = metadata
            _require(metadata["sample_rate_hz"] == rate, f"{role}: sample rate mismatch")
            result["reasons"].extend(f"{role}:{issue}" for issue in metadata["issues"])
            _require(metadata["duration_s"] >= MIN_HISTORY_S + duration,
                     f"{role}: insufficient background duration plus filter history")
            samples *= scale
            background = measure_pressure(samples, rate, start_s=MIN_HISTORY_S, end_s=MIN_HISTORY_S + duration)
            result[role] = background
            del samples
            uncertainty = manifest.get("expanded_uncertainty_db") if isinstance(manifest, dict) else None
            uncertainty_valid = type(uncertainty) in (int, float) and math.isfinite(uncertainty) and uncertainty > 0
            for metric in ("laeq_db", "lafmax_db"):
                if metrics[metric] is None or background[metric] is None:
                    result["reasons"].append(f"{role}:{metric}:undefined_level")
                elif metrics[metric] - background[metric] < MIN_BACKGROUND_MARGIN_DB + (2 * uncertainty if uncertainty_valid else 0):
                    result["reasons"].append(f"{role}:{metric}:background_masks_source")
        result["reasons"].extend(_manifest_issues(manifest, result["files"]))
        result["status"] = "computed" if not result["reasons"] else "inconclusive"
        result["raw_metrics_are_provisional"] = result["status"] != "computed"
        result["calibration_scale_pa_per_normalized_sample"] = scale
        result["acquisition_manifest"] = manifest
    except (OSError, ValueError, KeyError, TypeError, OverflowError, wavfile.WavFileWarning) as exc:
        result["reasons"].append(str(exc))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", required=True)
    parser.add_argument("--cal-before", required=True)
    parser.add_argument("--cal-after")
    parser.add_argument("--background-before")
    parser.add_argument("--background-after")
    parser.add_argument("--reference-db", type=float, required=True)
    parser.add_argument("--calibration-frequency-hz", type=float, default=1000.0)
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--start-s", type=float, required=True)
    parser.add_argument("--end-s", type=float)
    parser.add_argument("--manifest")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    try:
        manifest = None if args.manifest is None else json.loads(Path(args.manifest).read_text())
        report = analyze_recording(
            args.recording, args.cal_before, reference_level_db=args.reference_db,
            start_s=args.start_s, end_s=args.end_s, calibration_after_path=args.cal_after,
            background_before_path=args.background_before, background_after_path=args.background_after,
            calibration_frequency_hz=args.calibration_frequency_hz, channel=args.channel, manifest=manifest)
        encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
        if args.output:
            with Path(args.output).open("x", encoding="utf-8") as stream:
                stream.write(encoded + "\n")
        print(encoded)
        return 0 if report["status"] == "computed" else 2
    except (OSError, ValueError, TypeError) as exc:
        print(json.dumps({"status": "inconclusive", "reasons": [str(exc)]}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
