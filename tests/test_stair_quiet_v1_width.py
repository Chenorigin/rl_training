"""Width curriculum must change real geometry and count only target-width success."""
from dataclasses import replace
from pathlib import Path
import sys
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source/rl_training"))
from task_support.stair_quiet_v1_width.settings import WidthSettings, width_for_difficulty
from task_support.stair_quiet_v1_width.curriculum import (
    WidthCurriculum, final_width_mask, record_outcomes, validate_outcome_counts)
from task_support.stair_quiet_v1_width.settings import GEOMETRIES
from task_support.stair_quiet_v1_width.geometry import stair_mesh, foothold_regions


def test_width_owns_safety_sources_and_retired_task_is_absent():
    import ast
    from task_support.stair_quiet_v1_width.runner import sources
    root = Path(__file__).resolve().parents[1]
    retired = "stair_quiet_v1"+"_tail"
    base = root / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity"
    cfg = base / "config/wheeled/deeprobotics_m20"
    tree = ast.parse((cfg / "__init__.py").read_text())
    table = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "_TASKS" for t in n.targets))
    tasks = ast.literal_eval(table)
    assert any(t[1] == "stair_quiet_v1_width" for t in tasks)
    assert not any(t[1] == retired for t in tasks)
    assert not (root / "source/rl_training/task_support" / retired).exists()
    source_hashes = sources()
    for name in ("geometry.py", "signals.py", "observations.py", "terrain.py", "course.py"):
        assert f"source/rl_training/task_support/stair_quiet_v1_width/{name}" in source_hashes
    assert not any(p.endswith("stair_quiet_v1_width/validate.py") for p in source_hashes)
    assert not any(retired in p for p in source_hashes)


def test_two_successes_then_narrow_failure_widens_partial_environment_only():
    s = WidthSettings()
    s.validate()
    c = WidthCurriculum(3, "cpu", s)
    ids = torch.tensor([1])
    for expected in (0, 1):
        c.update(ids, actual_levels=c.level[ids].clone(), succeeded=torch.tensor([True]))
        assert c.level.tolist() == [0, expected, 0]
    c.update(ids, actual_levels=torch.tensor([1]), succeeded=torch.tensor([False]))
    assert c.level.tolist() == [0, 0, 0] and c.success_streak.sum() == 0
    last = len(s.widths_m)-1
    c.level[1] = last
    for _ in range(6):
        c.update(ids, actual_levels=torch.tensor([last]), succeeded=torch.tensor([True]))
    assert c.level.tolist() == [0, last, 0]
    c.update(torch.tensor([0]), actual_levels=torch.tensor([0]), succeeded=torch.tensor([False]))
    assert c.level[0] == 0


def test_resume_streak_and_disabled_fixed_width_evaluation():
    s = WidthSettings()
    c = WidthCurriculum(2, "cpu", s)
    c.update(torch.tensor([0]), actual_levels=torch.tensor([0]), succeeded=torch.tensor([True]))
    resumed = WidthCurriculum(2, "cpu", s)
    resumed.load_state_dict(c.state_dict())
    resumed.update(torch.tensor([0]), actual_levels=torch.tensor([0]), succeeded=torch.tensor([True]))
    assert resumed.level.tolist() == [1, 0]
    last = len(s.widths_m)-1
    for malformed in ({"level": [0, last+1], "success_streak": [0, 0]},
                      {"level": [0, 0], "success_streak": [0, 2]},
                      {"level": [0.5, 0], "success_streak": [0, 0]},
                      {"level": [0], "success_streak": [0]}):
        with pytest.raises(ValueError):
            resumed.load_state_dict(malformed)
    fixed = WidthCurriculum(2, "cpu", replace(s, width_curriculum_enabled=False, initial_width_level=last))
    fixed.update(torch.tensor([0, 1]), actual_levels=fixed.level.clone(), succeeded=torch.tensor([False, True]))
    assert fixed.level.tolist() == [last, last]


def test_wide_success_cannot_open_contact_stage_and_failure_tags_are_preserved():
    s = WidthSettings()
    last = len(s.widths_m)-1
    state = {k: torch.zeros(2, dtype=torch.long) for k in (
        "episodes", "successes", "stage_episodes", "stage_successes")}
    state["width_statistics"] = {k: torch.zeros(len(s.widths_m), 2, dtype=torch.long) for k in (
        "episodes", "successes", "edge_failures", "recovery_timeouts", "stall_timeouts", "other_failures")}
    types, rows = torch.tensor([0, 1]), torch.tensor([0, 2])
    yes, no = torch.ones(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool)
    record_outcomes(state, types, rows, yes, no, no, no, WidthSettings())
    assert state["successes"].tolist() == [1, 1]
    assert state["stage_episodes"].sum() == state["stage_successes"].sum() == 0
    record_outcomes(state, types, torch.tensor([last, last]), torch.tensor([True, False]),
                    torch.tensor([False, True]), torch.tensor([False, True]), no, WidthSettings())
    assert state["stage_episodes"].tolist() == [1, 1]
    assert state["stage_successes"].tolist() == [1, 0]
    stats = state["width_statistics"]
    assert stats["episodes"].sum(0).tolist() == state["episodes"].tolist()
    assert stats["edge_failures"][last, 1] == stats["recovery_timeouts"][last, 1] == 1
    validate_outcome_counts(state, stats)
    # Fabricated stage successes must not unlock new materials after resume.
    state["stage_successes"][1] = 1  # That case has zero lifetime successes.
    with pytest.raises(ValueError, match="Inconsistent"):
        validate_outcome_counts(state, stats)
    state["stage_successes"][1] = 0
    state["stage_episodes"][0] = 2  # Also cannot invent stage failures.
    with pytest.raises(ValueError, match="Inconsistent"):
        validate_outcome_counts(state, stats)


@pytest.mark.parametrize("widths", [(1., 1.3), (2., 1.5), (5., 1.), (2., float("nan"), 1.)])
def test_invalid_width_ledger(widths):
    with pytest.raises(ValueError):
        replace(WidthSettings(), widths_m=widths).validate()


@pytest.mark.parametrize("values", [
    {"recovery_timeout_s": 0.}, {"recovery_support_loss_s": -1.},
    {"forward_progress_m": 0.}, {"speed_range": (.8, .4)},
    {"speed_range": (.4, float("nan"))}, {"speed_range": (0., .8)},
])
def test_invalid_recovery_and_motion_settings_are_rejected(values):
    with pytest.raises(ValueError):
        replace(WidthSettings(), **values).validate()


def test_stair_commands_match_source_teacher_practice_range():
    assert WidthSettings().speed_range == (.4, .8)


def test_each_generator_row_changes_physical_treads_and_all_four_safety_regions():
    s = WidthSettings(front_lateral_margin=.04, rear_lateral_margin=.06)
    g = GEOMETRIES[0]
    position = torch.tensor([[[1.15, .55, .24]]*4])
    axis = torch.tensor([[[0., 1., 0.]]*4])
    previous = None
    for row, width in enumerate(s.widths_m):
        assert width_for_difficulty((row+.001)/len(s.widths_m), s) == width
        assert width_for_difficulty((row+.999)/len(s.widths_m), s) == width
        mesh = stair_mesh(replace(g, width=width), s)
        # Ignore the bottom catch floor; actual tread boards have this width.
        boards = mesh.vertices[mesh.vertices[:, 2] > -.1]
        assert np.isclose(np.ptp(boards[:, 1]), width)
        _, side, _ = foothold_regions(position, axis, torch.tensor([g.height]),
            torch.tensor([g.depth]), torch.tensor([width]), torch.tensor([g.steps]), s)
        assert side[0, 0] == side[0, 1] and side[0, 2] == side[0, 3]
        assert torch.allclose(side[0, :2]-side[0, 2:], torch.tensor([.02, .02]))
        if previous is not None:
            assert (side < previous).all()
        previous = side
    assert (previous < 0).all()  # Same wheel placement is unsafe at the 1 m edge.
    assert width_for_difficulty(1., s) == 1.
    assert len(s.widths_m) == 10 and s.widths_m[0] == 4. and s.widths_m[-1] == 1.
    assert final_width_mask(torch.arange(10), s).tolist() == [False]*9+[True]
