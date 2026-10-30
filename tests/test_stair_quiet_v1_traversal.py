import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source/rl_training"))
from task_support.stair_quiet_v1.traversal import TopLanding


def test_flying_above_top_is_not_completion():
    landing = TopLanding()
    for i in range(100):
        assert not landing.update(i * .02, [True] * 4, [False] * 4)


def test_each_wheel_lands_then_fixed_capture_is_required():
    landing = TopLanding()
    for i in range(4):
        assert not landing.update(i * .02, [True] * 4, [j == i for j in range(4)])
    assert not landing.update(.30, [True] * 4, [False] * 4)
    assert landing.update(.32, [True] * 4, [False] * 4)
    assert not landing.update(.34, [True] * 4, [False] * 4, invalid=True)


def test_return_below_edge_invalidates_previous_touchdown():
    landing = TopLanding()
    landing.update(0., [True] * 4, [True] * 4)
    assert not landing.update(.1, [True, True, True, False], [False] * 4)
    assert not landing.update(1., [True] * 4, [False] * 4)
    with pytest.raises(ValueError, match="monotonic"):
        landing.update(.5, [True] * 4, [True] * 4)
