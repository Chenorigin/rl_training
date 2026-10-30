from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source/rl_training"))
from task_support.stair_quiet_v1.compare import comparison_reasons, metric_comparison
from task_support.stair_quiet_v1.acoustics import read_json, STANDARD_PATH


def trial():
    return dict(completed=True, window_valid=True, fell=False, ever_outside_course=False,
                measurement_duration_s=8., max_stationary_interval_s=.1,
                optimizer_updates=0, weights_updated=False, compiled_model_sha256="a" * 64,
                command=[.5, 0, 0], physics_dt_s=.005, policy_dt_s=.02, seed=42, event_extraction={},
                traversal_settings={}, source_sha256={"controller": "a"})


def test_paired_conditions_and_stopping_cannot_win():
    good = trial()
    standard = read_json(STANDARD_PATH)
    assert comparison_reasons(good, deepcopy(good), standard) == []
    for key, value in (("completed", False), ("ever_outside_course", True),
                       ("max_stationary_interval_s", 1.), ("measurement_duration_s", 12.),
                       ("compiled_model_sha256", "b" * 64), ("command", [0., 0., 0.]),
                       ("source_sha256", {"controller": "b"})):
        bad = {**good, key: value}
        assert comparison_reasons(good, bad, standard)


@pytest.mark.parametrize("key,value", [
    ("completed", "false"), ("window_valid", "false"), ("fell", None),
    ("ever_outside_course", None), ("max_stationary_interval_s", -9),
    ("optimizer_updates", False), ("measurement_duration_s", True),
])
def test_malformed_summary_cannot_pass(key, value):
    good = trial()
    assert comparison_reasons(good, {**good, key: value}, read_json(STANDARD_PATH))


def test_literal_half_is_not_half_energy():
    half = metric_comparison(80., 40., .5)
    assert half["reading_reduction_percent"] == 50
    assert half["energy_ratio"] == pytest.approx(.0001)
    assert half["hypothetical_threshold_met"]
    assert not metric_comparison(80., 76.9897, .5)["hypothetical_threshold_met"]
    for baseline, candidate in ((0., -10.), (None, 40.), (80., None), (80., float("nan"))):
        assert not metric_comparison(baseline, candidate, .5)["hypothetical_threshold_met"]
