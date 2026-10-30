"""Synthetic instrument fixtures test the acceptance tool, never robot quietness."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import importlib.util
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "source/rl_training/task_support/stair_quiet_v1/acoustics.py"
SPEC = importlib.util.spec_from_file_location("stair_quiet_acoustics", MODULE_PATH)
acoustics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(acoustics)


def evidence():
    """Entirely artificial readings that satisfy a physical-measurement schema."""
    setup = acoustics.setup_template()
    setup.update({
        "robot_id": "synthetic-robot",
        "staircase_id": "synthetic-staircase",
        "stair_count": 6,
        "staircase_geometry_and_material_record": "fixture: six concrete steps",
        "wheel_surface_and_condition_record": "fixture: same dry wheels",
        "payload_kg": 0,
        "microphone_position_record": "fixture: fixed marked stand",
        "microphone_distance_m": 1,
        "microphone_height_m": 1.2,
        "command_forward_velocity_m_s": 0.3,
        "route_distance_m": 3,
    })
    setup["meter"].update({
        "serial": "synthetic-meter",
        "microphone_serial": "synthetic-microphone",
        "iec_61672_class": "1",
        "certificate_record": "fixture-only",
        "certificate_valid_until": "2031-01-01T00:00:00Z",
        "calibrator_serial": "synthetic-calibrator",
        "calibrator_certificate_record": "fixture-only",
        "calibrator_certificate_valid_until": "2031-01-01T00:00:00Z",
        "calibrator_reference_db": 94,
        "expanded_uncertainty_db": 1,
        "lower_measurement_limit_db": 0,
        "upper_measurement_limit_db": 120,
    })
    protocol = acoustics.freeze_protocol(setup, "c" * 64, frozen_at="2030-01-01T00:00:00Z")
    data = acoustics.measurement_template(protocol)
    data["all_attempts_included"] = True
    data["calibration"].update({
        "before_at": "2030-01-01T01:00:00Z", "after_at": "2030-01-01T02:00:00Z",
        "before_db": 94, "after_db": 94.1, "instrument_export_record": "fixture-only",
    })
    start = datetime(2030, 1, 1, 1, 1, tzinfo=timezone.utc)
    for index, trial in enumerate(data["trials"]):
        candidate = trial["policy_id"] == "candidate"
        # Both baselines are identical only to make expected numbers explicit.
        trial.update({
            "started_at": (start + timedelta(seconds=60 * index)).isoformat(),
            "instrument_export_record": "fixture-only",
            "observation_record": "fixture-only",
            "duration_s": 10, "distance_m": 3, "stairs_completed": 6,
            "completed": True, "fall_count": 0, "emergency_stop": False,
            "max_stationary_interval_s": 0.1,
            "laeq_db": 35 if candidate else 80,
            "lafmax_db": 40 if candidate else 90,
            "overload": False, "under_range": False, "external_noise_event": False,
        })
        for key in ("background_before", "background_after"):
            offset = -15 if key == "background_before" else 15
            trial[key].update({
                "started_at": (start + timedelta(seconds=60 * index + offset)).isoformat(),
                "duration_s": 10, "laeq_db": 10, "lafmax_db": 15,
                "source_inactive": True, "instrument_export_record": "fixture-only",
            })
    return protocol, data


class AcousticAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.protocol, self.data = evidence()

    def run_check(self):
        return acoustics.evaluate(self.data, self.protocol)

    def assert_inconclusive(self):
        report = self.run_check()
        self.assertEqual(report["status"], "inconclusive", report)
        self.assertTrue(report["reasons"])

    def test_positive_synthetic_control_passes_both_metrics_and_baselines(self):
        report = self.run_check()
        self.assertEqual(report["status"], "pass", report)
        self.assertEqual(len(report["comparisons"]), 2)
        for comparison in report["comparisons"].values():
            self.assertTrue(comparison["laeq_db"]["passed"])
            self.assertTrue(comparison["lafmax_db"]["passed"])
        self.assertEqual(report["aggregates"]["candidate"]["laeq_db"], 35)
        self.assertEqual(report["aggregates"]["candidate"]["lafmax_db"], 40)

    def test_energy_halving_is_not_dB_reading_halving(self):
        for row in self.data["trials"]:
            if row["policy_id"] == "candidate":
                row["laeq_db"] = 80 + 10 * math.log10(0.5)
                row["lafmax_db"] = 90 + 10 * math.log10(0.5)
        report = self.run_check()
        self.assertEqual(report["status"], "fail")
        comparison = next(iter(report["comparisons"].values()))
        self.assertAlmostEqual(comparison["laeq_db"]["energy_ratio"], 0.5)
        self.assertFalse(comparison["laeq_db"]["passed"])

    def test_failing_one_baseline_is_overall_failure(self):
        baseline = list(self.protocol["standard"]["baselines"])[1]
        for row in self.data["trials"]:
            if row["policy_id"] == baseline:
                row["laeq_db"], row["lafmax_db"] = 60, 70
        self.assertEqual(self.run_check()["status"], "fail")

    def test_peak_is_maximum_not_average_and_cannot_hide_one_loud_impact(self):
        candidate = next(t for t in self.data["trials"] if t["policy_id"] == "candidate")
        candidate["lafmax_db"] = 70
        report = self.run_check()
        self.assertEqual(report["status"], "fail")
        self.assertEqual(report["aggregates"]["candidate"]["lafmax_db"], 70)

    def test_uncertainty_guard_prevents_rounded_boundary_pass(self):
        for row in self.data["trials"]:
            if row["policy_id"] == "candidate":
                row["laeq_db"], row["lafmax_db"] = 40, 45
        self.assertEqual(self.run_check()["status"], "fail")

    def test_missing_field_empty_repeats_and_duplicate_ids_fail_closed(self):
        original = deepcopy(self.data)
        variants = []
        missing = deepcopy(original)
        del missing["trials"][0]["laeq_db"]
        variants.append(missing)
        empty = deepcopy(original)
        empty["trials"] = []
        variants.append(empty)
        few = deepcopy(original)
        few["trials"] = few["trials"][:3]
        variants.append(few)
        duplicate = deepcopy(original)
        duplicate["trials"][1]["trial_id"] = duplicate["trials"][0]["trial_id"]
        variants.append(duplicate)
        for data in variants:
            with self.subTest(data=data["trials"][:1]):
                self.data = data
                self.assert_inconclusive()

    def test_nan_infinity_and_boolean_numbers_fail_closed(self):
        for value in (float("nan"), float("inf"), float("-inf"), True, "35"):
            with self.subTest(value=value):
                self.protocol, self.data = evidence()
                self.data["trials"][0]["laeq_db"] = value
                self.assert_inconclusive()

    def test_stopping_falling_short_route_and_incomplete_ascent_cannot_pass(self):
        for field, value in (
            ("completed", False), ("stairs_completed", 0), ("fall_count", 1),
            ("max_stationary_interval_s", 4), ("distance_m", 0),
            ("emergency_stop", True), ("duration_s", 0),
        ):
            with self.subTest(field=field):
                self.protocol, self.data = evidence()
                candidate = next(t for t in self.data["trials"] if t["policy_id"] == "candidate")
                candidate[field] = value
                self.assert_inconclusive()

    def test_slow_candidate_cannot_pass_by_spreading_energy_over_time(self):
        for row in self.data["trials"]:
            if row["policy_id"] == "candidate":
                row["duration_s"] = 15
                for key in ("background_before", "background_after"):
                    row[key]["duration_s"] = 15
        self.assert_inconclusive()
        self.assertIn("duration differs", self.run_check()["reasons"][0])

    def test_background_masking_under_range_and_external_noise_are_inconclusive(self):
        for field in ("under_range", "overload", "external_noise_event"):
            with self.subTest(field=field):
                self.protocol, self.data = evidence()
                self.data["trials"][0][field] = True
                self.assert_inconclusive()
        self.protocol, self.data = evidence()
        candidate = next(t for t in self.data["trials"] if t["policy_id"] == "candidate")
        candidate["background_after"]["laeq_db"] = 30
        self.assert_inconclusive()

    def test_uncalibrated_drift_expired_and_wrong_time_are_inconclusive(self):
        for field, value in (("before_db", None), ("before_db", 96),
                             ("before_at", "2030-01-01T01:15:00Z"),
                             ("after_at", "2030-01-01T01:02:00Z"),
                             ("instrument_export_record", "")):
            with self.subTest(field=field):
                self.protocol, self.data = evidence()
                self.data["calibration"][field] = value
                self.assert_inconclusive()
        self.protocol, self.data = evidence()
        self.data["calibration"].update(before_db=93.6, after_db=94.4)
        self.assert_inconclusive()
        self.protocol, self.data = evidence()
        self.protocol["setup"]["meter"]["certificate_valid_until"] = "2029-01-01T00:00:00Z"
        self.assert_inconclusive()

    def test_changed_conditions_hashes_and_acceptance_threshold_cannot_pass(self):
        for change in ("conditions", "checkpoint", "protocol", "threshold", "attempts"):
            with self.subTest(change=change):
                self.protocol, self.data = evidence()
                if change == "conditions":
                    self.data["trials"][0]["conditions"] = deepcopy(self.protocol["setup"])
                    self.data["trials"][0]["conditions"]["microphone_distance_m"] = 2
                elif change == "checkpoint":
                    self.data["trials"][0]["checkpoint_sha256"] = "d" * 64
                elif change == "protocol":
                    self.data["protocol_sha256"] = "d" * 64
                elif change == "threshold":
                    self.protocol["standard"]["max_db_reading_fraction"] = 0.99
                else:
                    self.data["all_attempts_included"] = False
                self.assert_inconclusive()

    def test_noninterleaved_or_missing_policy_block_is_inconclusive(self):
        self.data["trials"][1]["block_id"] = 2
        self.assert_inconclusive()
        self.protocol, self.data = evidence()
        self.data["trials"].sort(key=lambda row: row["policy_id"])
        self.assert_inconclusive()

    def test_increasing_start_times_cannot_hide_overlapping_ten_second_ascents(self):
        start = datetime(2030, 1, 1, 1, 1, tzinfo=timezone.utc)
        for index, trial in enumerate(self.data["trials"]):
            trial["started_at"] = (start + timedelta(seconds=index)).isoformat()
            for key, offset in (("background_before", -15), ("background_after", 15)):
                trial[key]["started_at"] = (start + timedelta(seconds=index + offset)).isoformat()
        self.assert_inconclusive()
        self.assertIn("ascent intervals", self.run_check()["reasons"][0])

    def test_background_must_have_a_complete_nonoverlapping_time_window(self):
        first_start = datetime(2030, 1, 1, 1, 1, tzinfo=timezone.utc)
        cases = (
            ("background_before", None, "started_at"),
            ("background_before", "2030-01-01T00:59:55Z", "bracketed"),
            ("background_before", (first_start - timedelta(seconds=9)).isoformat(), "must end"),
            ("background_after", (first_start + timedelta(seconds=9)).isoformat(), "must start"),
            ("background_after", "2030-01-01T01:59:55Z", "bracketed"),
        )
        for key, value, expected in cases:
            with self.subTest(key=key, value=value):
                self.protocol, self.data = evidence()
                self.data["trials"][0][key]["started_at"] = value
                self.assert_inconclusive()
                self.assertIn(expected, self.run_check()["reasons"][0])

    def test_neighbor_background_cannot_overlap_or_reuse_previous_bundle(self):
        first = self.data["trials"][0]
        second = self.data["trials"][1]
        # Ascents themselves do not overlap, but this reuses the previous after
        # measurement as the next before measurement.
        second["background_before"]["started_at"] = first["background_after"]["started_at"]
        self.assert_inconclusive()
        self.assertIn("acquisition bundles", self.run_check()["reasons"][0])

    def test_exactly_adjacent_complete_acquisition_windows_are_valid(self):
        start = datetime(2030, 1, 1, 1, 1, tzinfo=timezone.utc)
        for index, trial in enumerate(self.data["trials"]):
            trial["started_at"] = (start + timedelta(seconds=30 * index)).isoformat()
            for key, offset in (("background_before", -10), ("background_after", 10)):
                trial[key]["started_at"] = (start + timedelta(seconds=30 * index + offset)).isoformat()
        self.assertEqual(self.run_check()["status"], "pass", self.run_check())

    def test_empty_template_cannot_pass(self):
        self.data = acoustics.measurement_template(self.protocol)
        self.assert_inconclusive()

    def test_template_conditions_do_not_mutate_frozen_protocol_and_reject_booleans(self):
        self.data["trials"][0]["conditions"]["microphone_distance_m"] = True
        self.assertEqual(type(self.protocol["setup"]["microphone_distance_m"]), int)
        self.assertEqual(self.data["trials"][1]["conditions"]["microphone_distance_m"], 1)
        self.assert_inconclusive()

    def test_energy_average_is_duration_weighted_and_handles_wide_dynamic_range(self):
        actual = acoustics.energy_mean_db([40, 50], [1, 3])
        expected = 10 * math.log10((10 ** 4 + 3 * 10 ** 5) / 4)
        self.assertAlmostEqual(actual, expected)
        self.assertNotAlmostEqual(actual, 47.5)
        self.assertTrue(math.isfinite(acoustics.energy_mean_db([0, 4000], [1, 1])))
        for levels, durations in (([], []), ([1], [0]), ([float("nan")], [1]), ([1], [])):
            with self.assertRaises(acoustics.InvalidEvidence):
                acoustics.energy_mean_db(levels, durations)

    def test_cli_writes_report_exit_codes_and_rejects_nonfinite_json(self):
        with tempfile.TemporaryDirectory() as directory:
            protocol_path = Path(directory) / "protocol.json"
            data_path = Path(directory) / "measurements.json"
            output_path = Path(directory) / "report.json"
            protocol_path.write_text(json.dumps(self.protocol))
            data_path.write_text(json.dumps(self.data))
            with redirect_stdout(io.StringIO()):
                code = acoustics.main(["evaluate", str(data_path), "--protocol", str(protocol_path),
                                       "--output", str(output_path)])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(output_path.read_text())["status"], "pass")
            data_path.write_text('{"trials": [NaN]}')
            with redirect_stdout(io.StringIO()) as stdout:
                code = acoustics.main(["evaluate", str(data_path), "--protocol", str(protocol_path)])
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(stdout.getvalue())["status"], "inconclusive")

    def test_freeze_hashes_checkpoint_and_refuses_to_overwrite_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            setup_path = Path(directory) / "setup.json"
            checkpoint = Path(directory) / "candidate.pt"
            output = Path(directory) / "protocol.json"
            setup_path.write_text(json.dumps(self.protocol["setup"]))
            checkpoint.write_bytes(b"synthetic checkpoint bytes")
            arguments = ["freeze", str(setup_path), "--candidate-checkpoint", str(checkpoint),
                         "--output", str(output)]
            with redirect_stdout(io.StringIO()):
                self.assertEqual(acoustics.main(arguments), 0)
                self.assertEqual(acoustics.main(arguments), 2)
            frozen = json.loads(output.read_text())
            self.assertEqual(frozen["candidate_sha256"], acoustics.checkpoint_sha256(checkpoint))


if __name__ == "__main__":
    unittest.main()
