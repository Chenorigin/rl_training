"""Short CPU physics/numerics counterexamples; no robot, training or simulation."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/rl_training"))
from task_support.stair_quiet_v1.sound_model import (
    EVENT_SCHEMA, STATUS, half_sine_force, load_profile, modal_acceleration,
    paired_profile_sweep, render_sound, validate_profile, write_sound,
)


def events(impulse=0.8):
    return {
        "schema": EVENT_SCHEMA, "duration_s": 0.6,
        "provenance": {"kind": "synthetic unit test; not robot evidence"},
        "events": [{"time_s": 0.1, "normal_impulse_N_s": impulse,
                    "position_m": [0.0, 0.0, 0.0], "wheel": "fl",
                    "contact_id": "fl:tread:1", "surface": "tread"}],
    }


class SoundModelTests(unittest.TestCase):
    def setUp(self):
        self.profile = load_profile()

    def test_pulse_integrates_to_declared_impulse_with_partial_final_bin(self):
        for duration in (0.004, 0.004003):
            force = half_sine_force(1.7, duration, 48000)
            self.assertAlmostEqual(float(force.sum() / 48000), 1.7, places=13)
            self.assertTrue((force >= 0).all())

    def test_impulse_scaling_is_linear_in_pressure_not_log_of_force(self):
        a, meta = render_sound(events(), self.profile)
        b, _ = render_sound(events(1.6), self.profile)
        np.testing.assert_allclose(b, 2 * a, rtol=1e-12, atol=1e-14)
        self.assertGreater(float(np.abs(a).max()), 0)
        self.assertEqual(meta["status"], STATUS)
        self.assertFalse(meta["actual_pass"])
        self.assertFalse(meta["calibrated"])

    def test_distance_decay_and_causal_travel_time(self):
        one, two = deepcopy(self.profile), deepcopy(self.profile)
        # 2 ms and 4 ms delays are integral sample offsets.
        one["receiver_position_m"] = [0.0, -0.686, 0.0]
        two["receiver_position_m"] = [0.0, -1.372, 0.0]
        p, pm = render_sound(events(), one)
        q, qm = render_sound(events(), two)
        shift = 96
        length = min(len(p), len(q) - shift)
        np.testing.assert_allclose(q[shift:shift + length], 0.5 * p[:length], atol=1e-10, rtol=1e-8)
        self.assertEqual(float(p[:int((1.0 + 0.1 + 0.002) * 48000)].max()), 0)
        self.assertAlmostEqual(qm["observed_max_propagation_delay_s"],
                               2 * pm["observed_max_propagation_delay_s"])
        self.assertEqual(qm["propagation_margin_s"], pm["propagation_margin_s"])
        self.assertEqual(qm["measurement_end_s"], pm["measurement_end_s"])

    def test_same_wheel_distinct_contacts_superpose_but_replayed_contact_fails(self):
        data = events()
        second = deepcopy(data["events"][0])
        second.update(contact_id="fl:riser:1", surface="riser")
        a, _ = render_sound(data, self.profile)
        data["events"].append(second)
        b, _ = render_sound(data, self.profile)
        np.testing.assert_allclose(b, 2 * a, atol=1e-14)
        data["events"][1]["contact_id"] = data["events"][0]["contact_id"]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            render_sound(data, self.profile)

    def test_static_equilibrium_load_does_not_radiate(self):
        mode = self.profile["modes"][0]
        force = np.full(24000, 100.0)
        acceleration = modal_acceleration(force, 48000, mode, equilibrium_force_n=100)
        self.assertLess(float(np.abs(acceleration).max()), 1e-7)
        silence = events()
        silence["events"] = []
        pressure, _ = render_sound(silence, self.profile)
        self.assertEqual(float(np.abs(pressure).max()), 0)
        invalid = events()
        invalid["events"][0]["static_force_N"] = 100
        with self.assertRaisesRegex(ValueError, "stationary"):
            render_sound(invalid, self.profile)

    def test_more_damping_reduces_late_ringdown(self):
        fs = 48000
        force = np.zeros(fs // 2)
        pulse = half_sine_force(1, 0.004, fs)
        force[:len(pulse)] = pulse
        low = deepcopy(self.profile["modes"][0])
        high = deepcopy(low)
        high["damping_ratio"] *= 2
        a = modal_acceleration(force, fs, low)
        b = modal_acceleration(force, fs, high)
        tail = slice(int(0.2 * fs), int(0.4 * fs))
        self.assertLess(float(np.mean(b[tail] ** 2)), 0.01 * float(np.mean(a[tail] ** 2)))
        self.assertLess(float(np.mean(a[-4800:] ** 2)), float(np.mean(a[4800:9600] ** 2)))

    def test_sample_rate_convergence(self):
        waveforms = []
        for fs in (48000, 96000, 192000):
            profile = deepcopy(self.profile)
            profile["sample_rate_hz"] = fs
            pressure, _ = render_sound(events(), profile)
            waveforms.append(pressure[::fs // 48000])
        n = min(map(len, waveforms))
        coarse = np.linalg.norm(waveforms[0][:n] - waveforms[2][:n])
        fine = np.linalg.norm(waveforms[1][:n] - waveforms[2][:n])
        self.assertLess(fine, coarse * 0.6)
        self.assertLess(coarse / np.linalg.norm(waveforms[2][:n]), 0.03)

    def test_bad_impulses_geometry_duration_and_sampling_fail_closed(self):
        for bad in (-1, float("nan"), float("inf"), True):
            with self.subTest(impulse=bad), self.assertRaises(ValueError):
                render_sound(events(bad), self.profile)
        for key, value in (("duration_s", 0), ("duration_s", 10000)):
            data = events()
            data[key] = value
            with self.assertRaises(ValueError):
                render_sound(data, self.profile)
        data = events()
        data["events"][0]["time_s"] = 0.599
        with self.assertRaisesRegex(ValueError, "extends"):
            render_sound(data, self.profile)
        for key, value in (("sample_rate_hz", 8000), ("pulse_duration_s", 0.000001),
                           ("receiver_position_m", [0, 0, 0]),
                           ("equivalent_source_radius_m", 1),
                           ("calibration_status", "measured"), ("actual_pass", True)):
            profile = deepcopy(self.profile)
            profile[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                render_sound(events(), profile)

    def test_unstable_or_nonphysical_modal_parameters_are_rejected(self):
        for key, value in (("damping_ratio", 0), ("damping_ratio", 1),
                           ("modal_mass_kg", -1), ("frequency_hz", 10000),
                           ("radiating_area_m2", 1)):
            profile = deepcopy(self.profile)
            profile["modes"][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_profile(profile)

    def test_tail_is_saved_but_not_averaged_into_measurement_window(self):
        pressure, metadata = render_sound(events(), self.profile)
        self.assertGreater(metadata["render_duration_s"], metadata["measurement_end_s"])
        self.assertEqual(metadata["measurement_start_s"], 1)
        self.assertAlmostEqual(metadata["measurement_end_s"] - 1,
                               0.6 + metadata["max_propagation_delay_s"])
        self.assertEqual(len(pressure), round(metadata["render_duration_s"] * 48000))

    def test_measurement_subwindow_retains_preceding_excitation_and_rejects_bad_bounds(self):
        data = events()
        whole, _ = render_sound(data, self.profile)
        # Impact at 0.1 s precedes the measured interval. Keep its physical
        # ringdown and the acoustic meter's entire prehistory, never crop events.
        data["measurement_window_s"] = [0.12, 0.4]
        pressure, metadata = render_sound(data, self.profile)
        np.testing.assert_array_equal(pressure, whole)
        self.assertAlmostEqual(metadata["measurement_start_s"], 1.12)
        self.assertAlmostEqual(metadata["measurement_end_s"], 1.4 + metadata["max_propagation_delay_s"])
        self.assertAlmostEqual(metadata["source_measurement_duration_s"], 0.28)
        start = round(metadata["measurement_start_s"] * 48000)
        self.assertGreater(float(np.abs(pressure[start:start + 480]).max()), 0)
        for window in ([-1, 0.4], [0.4, 0.2], [0.1, 0.7], [True, 0.3], [0.1]):
            data["measurement_window_s"] = window
            with self.subTest(window=window), self.assertRaises(ValueError):
                render_sound(data, self.profile)

    def test_remote_zero_or_outside_window_event_cannot_dilute_measurement(self):
        data = events()
        data["measurement_window_s"] = [0.1, 0.3]
        pressure, metadata = render_sound(data, self.profile)
        receiver = np.asarray(self.profile["receiver_position_m"])
        for impulse, time in ((0.0, 0.2), (0.8, 0.5)):
            changed = deepcopy(data)
            changed["events"].append({
                "time_s": time, "normal_impulse_N_s": impulse,
                "position_m": (receiver + [15.0, 0.0, 0.0]).tolist(),
                "wheel": "fr", "contact_id": "fr:remote:1",
            })
            other, other_metadata = render_sound(changed, self.profile)
            self.assertEqual(other_metadata["measurement_start_s"], metadata["measurement_start_s"])
            self.assertEqual(other_metadata["measurement_end_s"], metadata["measurement_end_s"])
            self.assertEqual(other.shape, pressure.shape)
            end = int(metadata["measurement_end_s"] * self.profile["sample_rate_hz"])
            np.testing.assert_array_equal(other[:end], pressure[:end])
        self.assertAlmostEqual(metadata["propagation_margin_s"],
                               self.profile["max_source_receiver_distance_m"] / self.profile["sound_speed_m_s"])
        # Out-of-domain locations are invalid even when their impulse is zero.
        for impulse in (0.0, 0.8):
            changed = events(impulse)
            changed["events"][0]["position_m"] = (receiver + [20.01, 0.0, 0.0]).tolist()
            with self.assertRaisesRegex(ValueError, "maximum source-receiver distance"):
                render_sound(changed, self.profile)

    def test_sensitivity_applies_identical_parameters_to_both_policies(self):
        def simple_meter(pressure, fs, *, start_s, end_s):
            portion = pressure[int(start_s * fs):int(end_s * fs)]
            return {"rms_pressure_pa": float(np.sqrt(np.mean(portion ** 2)))}
        report = paired_profile_sweep(events(), events(0.4), self.profile, metric_fn=simple_meter)
        self.assertEqual(len(report["rows"]), 9)
        self.assertFalse(report["actual_pass"])
        for row in report["rows"]:
            self.assertEqual(row["baseline"]["profile_sha256"], row["candidate"]["profile_sha256"])
            self.assertEqual(row["shared_profile_sha256"], row["baseline"]["profile_sha256"])
            self.assertAlmostEqual(row["candidate"]["metrics"]["rms_pressure_pa"],
                                   0.5 * row["baseline"]["metrics"]["rms_pressure_pa"], places=12)
            self.assertFalse(row["candidate"]["actual_pass"])
        self.assertEqual(self.profile["pulse_duration_s"], 0.004)

    def test_raw_pa_npz_and_float_wav_roundtrip_without_normalization(self):
        # Deliberately oversized synthetic impulse exercises WAV values > 1;
        # it is not a plausible robot event or a profile fit.
        pressure, metadata = render_sound(events(800), self.profile)
        self.assertGreater(float(np.abs(pressure).max()), 1)
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / "prediction"
            paths = write_sound(prefix, pressure, metadata)
            with np.load(paths[0], allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["pressure_pa"], pressure)
                self.assertFalse(json.loads(str(saved["metadata_json"]))["actual_pass"])
            fs, wav = wavfile.read(paths[1])
            self.assertEqual(fs, 48000)
            self.assertEqual(wav.dtype, np.float32)
            np.testing.assert_array_equal(wav, pressure.astype(np.float32))
            self.assertFalse(json.loads(Path(paths[2]).read_text())["actual_pass"])
            with self.assertRaisesRegex(ValueError, "exists"):
                write_sound(prefix, pressure, metadata)


if __name__ == "__main__":
    unittest.main()
