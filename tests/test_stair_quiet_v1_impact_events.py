from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source/rl_training"))
from task_support.stair_quiet_v1.impact_events import ImpactEvents, ExtractionSettings, select_window


def sample(force, wheel="fl", surface="tread"):
    return {"normal_force_N": force, "position_m": [1., 0., .2], "wheel": wheel, "surface": surface}


def test_impulse_is_integral_and_support_does_not_retrigger():
    tracker = ImpactEvents(.005, ExtractionSettings(reset_grace_s=0))
    for i in range(40):
        tracker.update(i * .005, {"fl/tread": sample(100.)})
    events = tracker.finish()
    assert len(events) == 1
    assert events[0]["normal_impulse_N_s"] == pytest.approx(4.)
    assert events[0]["position_m"] == pytest.approx([1., 0., .2])


def test_reload_and_two_surfaces_remain_distinct():
    tracker = ImpactEvents(.005, ExtractionSettings(reset_grace_s=0))
    for i in range(30):
        tracker.update(i * .005, {
            "fl/tread": sample(100 if i < 20 else 200),
            "fl/riser": sample(50, surface="riser"),
        })
    events = tracker.finish()
    assert len(events) == 3
    assert sum(e["trigger"] == "rapid_reload" for e in events) == 1
    assert len({e["contact_id"] for e in events}) == 3


def test_grace_and_separation():
    tracker = ImpactEvents(.005)
    for i in range(40):
        tracker.update(i * .005, {"fl/tread": sample(100.)} if i < 10 or i >= 30 else {})
    assert len(tracker.finish()) == 1


def test_invalid_samples_and_window_boundary():
    tracker = ImpactEvents(.005)
    tracker.update(0., {})
    with pytest.raises(ValueError, match="contiguous"):
        tracker.update(.01, {})
    with pytest.raises(ValueError, match="normal force"):
        ImpactEvents(.005).update(0., {"bad": sample(np.nan)})
    event = {"time_s": 1., "integrated_duration_s": .04, "contact_id": "x"}
    assert select_window([event], 1., 2.) == ([{**event, "time_s": 0.}], [])
    assert select_window([event], 1.02, 2.) == ([], ["x"])
