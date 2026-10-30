"""Analytic and synthetic WAV tests of computation, NOT real robot measurements."""
from datetime import datetime, timedelta, timezone
import importlib.util
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
import wave
from contextlib import redirect_stdout

import numpy as np
from scipy.io import wavfile


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "stair_quiet_audio", ROOT / "source/rl_training/task_support/stair_quiet_v1/audio.py")
audio = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audio)
RATE = 48000
ONE_PA_DB = 20 * math.log10(1 / 20e-6)


def tone(rms=1., frequency=1000., duration=2., rate=RATE):
    return math.sqrt(2) * rms * np.sin(2 * np.pi * frequency * np.arange(round(duration * rate)) / rate)


class PressureMetricTests(unittest.TestCase):
    def test_one_pa_rms_one_khz_and_instantaneous_peak_are_distinct(self):
        result = audio.measure_pressure(tone(), RATE, start_s=1)
        self.assertAlmostEqual(result["laeq_db"], ONE_PA_DB, delta=.002)
        self.assertAlmostEqual(result["lafmax_db"], ONE_PA_DB, delta=.01)
        self.assertAlmostEqual(result["rms_pressure_pa"], 1, places=10)
        self.assertAlmostEqual(result["peak_pressure_pa"], math.sqrt(2), places=10)
        self.assertAlmostEqual(result["lzpeak_db"], ONE_PA_DB + 10 * math.log10(2), places=10)
        self.assertNotIn("actual_pass", result)

    def test_half_pressure_amplitude_means_minus_six_db_not_half_db_reading(self):
        full = audio.measure_pressure(tone(), RATE, start_s=1)
        half = audio.measure_pressure(tone(.5), RATE, start_s=1)
        for key in ("laeq_db", "lafmax_db", "lzpeak_db"):
            self.assertAlmostEqual(half[key] - full[key], 20 * math.log10(.5), places=10)

    def test_fast_step_and_decay_follow_125_ms_exponential_of_squared_pressure(self):
        n = round(.125 * RATE)
        response = audio._fast_average_squared(np.r_[np.ones(n), np.zeros(n)], RATE)
        self.assertAlmostEqual(response[n - 1], 1 - math.exp(-1), places=10)
        self.assertAlmostEqual(response[-1], (1 - math.exp(-1)) * math.exp(-1), places=10)

    def test_a_weighting_matches_independent_analog_reference_over_audio_band(self):
        # Independent scalar frequency-response equation, not the IIR coefficients.
        def magnitude(f):
            f1, f2, f3, f4 = 20.598997, 107.65265, 737.86223, 12194.217
            return f4 ** 2 * f ** 4 / ((f ** 2 + f1 ** 2) *
                   math.sqrt((f ** 2 + f2 ** 2) * (f ** 2 + f3 ** 2)) * (f ** 2 + f4 ** 2))
        for frequency in (31.5, 100, 1000, 4000, 8000, 16000, 20000):
            with self.subTest(frequency=frequency):
                result = audio.measure_pressure(tone(frequency=frequency), RATE, start_s=1)
                ideal_db = 20 * math.log10(magnitude(frequency) / magnitude(1000))
                # Engineering approximation, deliberately NOT an IEC class claim.
                self.assertAlmostEqual(result["laeq_db"] - ONE_PA_DB, ideal_db, delta=.55)

    def test_silence_is_null_db_and_empty_nan_bad_rates_are_rejected(self):
        silent = audio.measure_pressure(np.zeros(RATE), RATE)
        self.assertIsNone(silent["laeq_db"])
        self.assertIsNone(silent["lzpeak_db"])
        for values, rate in (([], RATE), ([float("nan")], RATE), (tone(), 16000),
                             (np.zeros((2, 2)), RATE), ([float("inf")], RATE)):
            with self.subTest(rate=rate):
                with self.assertRaises(audio.InvalidAudio):
                    audio.measure_pressure(values, rate)

    def test_selected_interval_does_not_dilute_level_with_trailing_silence(self):
        full = tone()
        padded = np.r_[full, np.zeros(RATE)]
        a = audio.measure_pressure(full, RATE, start_s=1)
        b = audio.measure_pressure(padded, RATE, start_s=1, end_s=2)
        self.assertAlmostEqual(a["laeq_db"], b["laeq_db"], delta=.002)


class PcmCalibrationTests(unittest.TestCase):
    def setUp(self):
        # Keep transient fixtures inside the assigned worktree, cleaned on exit.
        self.directory = tempfile.TemporaryDirectory(prefix=".audio_test_", dir=ROOT / "tests")
        self.addCleanup(self.directory.cleanup)
        self.folder = Path(self.directory.name)

    def write(self, name, samples, *, rate=RATE, dtype=np.int32):
        path = self.folder / name
        if np.dtype(dtype).kind == "i":
            scale = 2 ** (np.dtype(dtype).itemsize * 8 - 1)
            samples = np.clip(np.rint(samples * scale), -scale, scale - 1).astype(dtype)
        else:
            samples = np.asarray(samples, dtype=dtype)
        wavfile.write(path, rate, samples)
        return path

    def fixture(self, gain=1.0):
        paths = {
            "calibration_before": self.write("cal_before.wav", tone(.1 * gain)),
            "calibration_after": self.write("cal_after.wav", tone(.1 * gain)),
            "recording": self.write("recording.wav", tone(.01 * gain)),
            "background_before": self.write("background_before.wav", tone(.0001 * gain)),
            "background_after": self.write("background_after.wav", tone(.0001 * gain)),
        }
        start = datetime(2030, 1, 1, tzinfo=timezone.utc)
        manifest = {
            "recorder_id": "synthetic-recorder", "microphone_id": "synthetic-mic",
            "gain_setting": "synthetic-fixed-gain", "calibrator_id": "synthetic-calibrator",
            "calibrator_certificate_record": "SYNTHETIC_FIXTURE_NOT_A_CERTIFICATE",
            "calibrator_certificate_valid_until": "2031-01-01T00:00:00Z",
            "microphone_response_record": "SYNTHETIC_FIXTURE_NOT_A_SENSOR",
            "measurement_window_record": "synthetic analytic sine, not a robot",
            "expanded_uncertainty_db": .5,
            "same_gain_all_files": True, "agc_disabled": True, "processing_disabled": True,
            "background_source_inactive": True, "files": {},
        }
        order = ("calibration_before", "background_before", "recording", "background_after", "calibration_after")
        for i, role in enumerate(order):
            manifest["files"][role] = {
                "sha256": audio._sha256(paths[role]),
                "started_at": (start + timedelta(seconds=i * 3)).isoformat(),
                "gain_setting": manifest["gain_setting"],
            }
        return paths, manifest

    def analyze(self, paths, manifest=None):
        return audio.analyze_recording(
            paths["recording"], paths["calibration_before"], reference_level_db=ONE_PA_DB,
            start_s=1, end_s=2, calibration_after_path=paths.get("calibration_after"),
            background_before_path=paths.get("background_before"),
            background_after_path=paths.get("background_after"), manifest=manifest)

    def test_48k_calibration_computes_known_pa_and_complete_fixture_metrics(self):
        paths, manifest = self.fixture()
        report = self.analyze(paths, manifest)
        self.assertEqual(report["status"], "computed", report["reasons"])
        self.assertAlmostEqual(report["calibration_scale_pa_per_normalized_sample"], 10, places=6)
        self.assertAlmostEqual(report["metrics"]["rms_pressure_pa"], .1, places=7)
        self.assertAlmostEqual(report["metrics"]["laeq_db"], ONE_PA_DB - 20, delta=.002)
        self.assertEqual(report["robot_noise_reduction_claim"], "not_evaluated")
        self.assertNotIn("actual_pass", report)

    def test_common_gain_scaling_cancels_but_recording_only_half_amplitude_drops_six_db(self):
        paths, _ = self.fixture()
        original = self.analyze(paths)
        paths, _ = self.fixture(gain=2)
        scaled = self.analyze(paths)
        self.assertAlmostEqual(original["metrics"]["laeq_db"], scaled["metrics"]["laeq_db"], places=6)
        paths["recording"] = self.write("half.wav", tone(.01))
        half = self.analyze(paths)
        self.assertAlmostEqual(half["metrics"]["laeq_db"] - scaled["metrics"]["laeq_db"],
                               20 * math.log10(.5), places=6)

    def test_94_db_is_not_rounded_to_exactly_one_pa(self):
        path = self.write("cal.wav", tone(.1))
        result = audio.calibrate(path, 94)
        self.assertAlmostEqual(result["reference_rms_pa"], 20e-6 * 10 ** (94 / 20), places=12)
        self.assertNotEqual(result["reference_rms_pa"], 1)

    def test_packed_24bit_is_left_justified_and_scaled_correctly(self):
        values = np.array([-.5, -.25, 0, .25, .5])
        packed = np.rint(values * 2 ** 23).astype(np.int32)
        octets = np.column_stack([packed & 255, (packed >> 8) & 255, (packed >> 16) & 255]).astype(np.uint8)
        path = self.folder / "packed24.wav"
        with wave.open(str(path), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(3)
            stream.setframerate(RATE)
            stream.writeframes(octets.tobytes())
        samples, metadata = audio.load_pcm_wav(path)
        np.testing.assert_allclose(samples, values, atol=1e-12)
        self.assertEqual(metadata["container_bits"], 24)
        self.assertEqual(metadata["decoded_dtype"], "int32")

    def test_24bit_positive_rail_is_detected_below_int32_max(self):
        path = self.folder / "rail24.wav"
        with wave.open(str(path), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(3)
            stream.setframerate(RATE)
            stream.writeframes(bytes([255, 255, 127, 0, 0, 0, 0, 0, 128]))
        _, metadata = audio.load_pcm_wav(path)
        self.assertEqual(metadata["rail_sample_count"], 2)
        self.assertIn("digital_rail_or_out_of_range_samples", metadata["issues"])

    def test_pcm8_unsigned_centering_and_explicit_stereo_channel(self):
        path = self.folder / "u8.wav"
        wavfile.write(path, RATE, np.array([64, 128, 192], dtype=np.uint8))
        samples, _ = audio.load_pcm_wav(path)
        np.testing.assert_array_equal(samples, [-.5, 0, .5])
        stereo = self.write("stereo.wav", np.column_stack([tone(.1), tone(.2)]))
        first, _ = audio.load_pcm_wav(stereo, 0)
        second, _ = audio.load_pcm_wav(stereo, 1)
        np.testing.assert_allclose(second, 2 * first, atol=1e-9)

    def test_calibration_rejects_clipping_wrong_frequency_and_non_tone(self):
        bad = {
            "clipped": np.clip(tone(1.0), -.999999999, .999999999),
            "frequency": tone(.1, frequency=500),
            "square": np.sign(tone(.1)) * .1,
            "silence": np.zeros(RATE * 2),
        }
        for name, values in bad.items():
            with self.subTest(name=name):
                path = self.write(name + ".wav", values)
                with self.assertRaises(audio.InvalidAudio):
                    audio.calibrate(path, ONE_PA_DB)

    def test_empty_nan_unsupported_rate_and_truncated_pcm_fail_closed(self):
        paths = [
            self.write("empty.wav", np.array([])),
            self.write("nan.wav", [0, float("nan")], dtype=np.float32),
            self.write("rate.wav", tone(.1, rate=16000), rate=16000),
        ]
        truncated = self.write("truncated.wav", tone(.1))
        truncated.write_bytes(truncated.read_bytes()[:-4])
        paths.append(truncated)
        for path in paths:
            with self.subTest(path=path):
                with self.assertRaises((audio.InvalidAudio, wavfile.WavFileWarning)):
                    audio.load_pcm_wav(path)

    def test_missing_evidence_clipping_drift_and_background_never_computed(self):
        for problem in ("manifest", "cal_after", "background", "clip", "drift", "mask", "gain", "overlap", "sha", "agc"):
            with self.subTest(problem=problem):
                paths, manifest = self.fixture()
                if problem == "manifest":
                    manifest = None
                elif problem == "cal_after":
                    del paths["calibration_after"]
                elif problem == "background":
                    del paths["background_after"]
                elif problem == "clip":
                    paths["recording"] = self.write("clip.wav", tone(1))
                elif problem == "drift":
                    paths["calibration_after"] = self.write("drift.wav", tone(.1 * 10 ** (.6 / 20)))
                elif problem == "mask":
                    paths["background_after"] = self.write("mask.wav", tone(.008))
                elif problem == "gain":
                    manifest["files"]["recording"]["gain_setting"] = "other-gain"
                elif problem == "overlap":
                    manifest["files"]["recording"]["started_at"] = manifest["files"]["background_before"]["started_at"]
                elif problem == "sha":
                    manifest["files"]["recording"]["sha256"] = "0" * 64
                else:
                    manifest["agc_disabled"] = False
                report = self.analyze(paths, manifest)
                self.assertEqual(report["status"], "inconclusive", report)
                self.assertTrue(report["reasons"])

    def test_cli_emits_raw_metrics_without_fabricating_missing_manifest(self):
        paths, _ = self.fixture()
        output = self.folder / "output.json"
        with redirect_stdout(io.StringIO()):
            result = audio.main(["--recording", str(paths["recording"]),
                                 "--cal-before", str(paths["calibration_before"]),
                                 "--reference-db", str(ONE_PA_DB),
                                 "--start-s", "1", "--end-s", "2", "--output", str(output)])
        self.assertEqual(result, 2)
        report = json.loads(output.read_text())
        self.assertEqual(report["status"], "inconclusive")
        self.assertAlmostEqual(report["metrics"]["rms_pressure_pa"], .1, places=7)
        self.assertIsNone(report["acquisition_manifest"])


if __name__ == "__main__":
    unittest.main()
