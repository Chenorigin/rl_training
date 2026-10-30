"""Student evidence, reversible progression and fixed all-terrain populations."""
from copy import deepcopy
import math
import random

import pytest
import torch

from task_support.cts.all_terrain_curriculum import (
    CHALLENGE, COURSE_FRACTIONS, COURSE_NAMES, CURRENT, DEFAULT_CURRICULUM_CONFIG,
    REPLAY, CourseCurriculum, allocate_courses, validate_curriculum_config,
)


def settings(**changes):
    cfg = deepcopy(DEFAULT_CURRICULUM_CONFIG)
    cfg.update(window_episodes=4, min_boundary_episodes=1, min_delayed_episodes=2)
    cfg.update(changes)
    return cfg


def window(curriculum, successes, course=0, update=0, teacher=False,
           role=CURRENT, version=None, coverage=True):
    version = curriculum.current(course)['version'] if version is None else version
    event = None
    for i in range(curriculum.cfg['window_episodes']):
        event = curriculum.record(
            course, i < successes, teacher, role, version, update,
            boundary=coverage and i < curriculum.cfg['min_boundary_episodes'],
            delayed=coverage and i < curriculum.cfg['min_delayed_episodes'],
        )
    return event


def promote(curriculum, course=0, update=0):
    result = None
    for _ in range(curriculum.cfg['consecutive_windows']):
        result = window(curriculum, curriculum.cfg['window_episodes'], course, update)
    return result


def test_allocation_preserves_each_cohort_and_exact_large_population_fractions():
    teacher = torch.arange(400) % 4 != 0
    before = teacher.clone()
    ids = allocate_courses(teacher, COURSE_FRACTIONS, seed=42)
    assert COURSE_NAMES == ('up', 'down', 'round_trip', 'motion', 'rough', 'blocks',
                            'pits', 'mixed', 'straight', 'platform')
    assert ids.dtype == torch.long and ids.device == teacher.device
    assert torch.equal(teacher, before)
    for mask in (teacher, ~teacher):
        counts = torch.bincount(ids[mask], minlength=len(COURSE_NAMES))
        assert counts.tolist() == [round(int(mask.sum()) * f) for f in COURSE_FRACTIONS]
    assert torch.equal(ids, allocate_courses(teacher, COURSE_FRACTIONS, seed=42))
    assert not torch.equal(ids, allocate_courses(teacher, COURSE_FRACTIONS, seed=43))


@pytest.mark.parametrize('n', [1, 7, 40, 73, 2048])
def test_largest_remainder_uses_every_environment_with_at_most_one_rounding_error(n):
    teacher = torch.arange(n) % 4 != 0
    ids = allocate_courses(teacher)
    for mask in (teacher, ~teacher):
        count = int(mask.sum())
        actual = torch.bincount(ids[mask], minlength=10).tolist()
        assert sum(actual) == count
        assert all(math.floor(count * f) <= x <= math.ceil(count * f)
                   for x, f in zip(actual, COURSE_FRACTIONS))
    assert ids.min() >= 0 and ids.max() < 10


def test_allocation_and_sampling_do_not_change_global_random_states():
    torch_state, python_state = torch.get_rng_state().clone(), random.getstate()
    allocate_courses(torch.arange(80) % 4 != 0)
    curriculum = CourseCurriculum(settings())
    for _ in range(50):
        curriculum.sample(0)
    assert torch.equal(torch_state, torch.get_rng_state())
    assert python_state == random.getstate()


@pytest.mark.parametrize('mask,fractions,seed', [
    (torch.ones(4), COURSE_FRACTIONS, 42),
    (torch.ones(2, 2, dtype=torch.bool), COURSE_FRACTIONS, 42),
    (torch.zeros(0, dtype=torch.bool), COURSE_FRACTIONS, 42),
    (torch.ones(4, dtype=torch.bool), (.5, .5), 42),
    (torch.ones(4, dtype=torch.bool), (.3,) * 10, 42),
    (torch.ones(4, dtype=torch.bool), COURSE_FRACTIONS, True),
])
def test_invalid_allocation_is_rejected(mask, fractions, seed):
    with pytest.raises(ValueError):
        allocate_courses(mask, fractions, seed)


def test_teacher_success_cannot_promote_a_student_with_forty_percent_success():
    curriculum = CourseCurriculum(deepcopy(DEFAULT_CURRICULUM_CONFIG))
    for _ in range(150):
        curriculum.record(0, True, True, CURRENT, 0, 0, boundary=True, delayed=True)
    for i in range(50):
        curriculum.record(0, i < 20, False, CURRENT, 0, 0, boundary=True, delayed=True)
    assert curriculum.current(0)['version'] == 0
    data = curriculum.diagnostics()[0]
    assert data['teacher_successes'] == 150
    assert data['student_successes'] == 20
    assert data['window']['episodes'] == 50
    # The old pooled gate would see 170 / 200 == .85; this gate sees 20 / 50.
    assert data['consecutive_windows'] == 0


def test_two_disjoint_student_windows_are_required_and_courses_are_independent():
    curriculum = CourseCurriculum(settings())
    for _ in range(3):
        assert curriculum.record(0, True, False, CURRENT, 0, 0, True, True) is None
    event = curriculum.record(0, True, False, CURRENT, 0, 0, True, True)
    assert event['event'] == 'window' and event['window']['episodes'] == 4
    assert curriculum.current(0)['geometry_level'] == 0
    event = window(curriculum, 4)
    assert event['event'] == 'promote'
    assert curriculum.current(0) == dict(geometry_level=1, dr_level=0, version=1, next_axis='dr')
    assert curriculum.current(1) == dict(geometry_level=0, dr_level=0, version=0, next_axis='geometry')
    assert curriculum.diagnostics()[0]['window']['episodes'] == 0
    assert curriculum.diagnostics()[0]['consecutive_windows'] == 0


def test_replay_challenge_teacher_and_stale_episodes_do_not_supply_promotion_evidence():
    curriculum = CourseCurriculum(settings())
    promote(curriculum)
    for _ in range(3):
        window(curriculum, 4, teacher=True)
        window(curriculum, 4, role=REPLAY)
        window(curriculum, 4, role=CHALLENGE)
        window(curriculum, 4, version=0)
    current = curriculum.current(0)
    assert current['version'] == 1 and current['geometry_level'] == 1 and current['dr_level'] == 0
    data = curriculum.diagnostics()[0]
    assert data['window']['episodes'] == 0
    assert data['ignored_role'] == 24 and data['ignored_version'] == 12
    assert data['teacher_episodes'] == 12


def test_a_hold_or_opposite_window_breaks_the_consecutive_streak():
    curriculum = CourseCurriculum(settings())
    window(curriculum, 4)
    held = window(curriculum, 3)
    assert held['event'] == 'hold' and held['reason'] == 'between_thresholds'
    assert window(curriculum, 4)['event'] == 'window'
    assert window(curriculum, 0)['event'] == 'window'
    assert window(curriculum, 4)['event'] == 'window'
    assert curriculum.current(0)['version'] == 0
    assert window(curriculum, 4)['event'] == 'promote'


def test_coverage_is_required_for_each_promoting_window_but_not_for_failure_rollback():
    curriculum = CourseCurriculum(settings())
    assert window(curriculum, 4)['event'] == 'window'
    held = window(curriculum, 4, coverage=False)
    assert held['event'] == 'hold' and held['reason'] == 'insufficient_coverage'
    assert window(curriculum, 4)['event'] == 'window'
    assert window(curriculum, 4)['event'] == 'promote'
    assert window(curriculum, 0, coverage=False)['event'] == 'window'
    assert window(curriculum, 0, coverage=False)['event'] == 'demote'
    assert curriculum.current(0)['geometry_level'] == 0


def test_threshold_boundaries_ninety_promotes_seventy_holds_and_sixty_nine_demotes():
    curriculum = CourseCurriculum(deepcopy(DEFAULT_CURRICULUM_CONFIG))
    assert window(curriculum, 90)['event'] == 'window'
    assert window(curriculum, 90)['event'] == 'promote'
    assert window(curriculum, 70)['event'] == 'hold'
    assert window(curriculum, 70)['event'] == 'hold'
    assert curriculum.current(0)['geometry_level'] == 1
    assert window(curriculum, 69)['event'] == 'window'
    assert window(curriculum, 69)['event'] == 'demote'


def test_progression_alternates_axes_and_failure_rolls_back_only_latest_axis():
    curriculum = CourseCurriculum(settings())
    grades = []
    for _ in range(4):
        assert promote(curriculum)['event'] == 'promote'
        current = curriculum.current(0)
        grades.append((current['geometry_level'], current['dr_level']))
    assert grades == [(1, 0), (1, 1), (2, 1), (2, 2)]
    assert window(curriculum, 0)['event'] == 'window'
    assert window(curriculum, 0)['event'] == 'demote'
    assert curriculum.current(0) == dict(geometry_level=2, dr_level=1, version=5, next_axis='dr')
    assert promote(curriculum)['event'] == 'promote'
    assert curriculum.current(0)['dr_level'] == 2
    assert curriculum.current(0)['geometry_level'] == 2


def test_upper_and_lower_bound_hold_without_version_churn_or_nonexistent_roles():
    curriculum = CourseCurriculum(settings(geometry_levels=2, dr_levels=1,
                                          role_fractions=[.01, 0., .99]))
    assert window(curriculum, 0)['event'] == 'window'
    held = window(curriculum, 0)
    assert held['event'] == 'hold' and held['reason'] == 'lower_bound'
    assert curriculum.current(0)['version'] == 0
    assert promote(curriculum)['event'] == 'promote'
    held = promote(curriculum)
    assert held['event'] == 'hold' and held['reason'] == 'upper_bound'
    assert curriculum.current(0)['version'] == 1
    assert all(curriculum.sample(0)['role'] == CURRENT for _ in range(100))
    no_progression = CourseCurriculum(settings(geometry_levels=1, dr_levels=1))
    assert all(no_progression.sample(0)['role'] == CURRENT for _ in range(100))


def test_empty_replay_pool_returns_current_and_challenge_only_advances_one_axis():
    curriculum = CourseCurriculum(settings(role_fractions=[.01, .49, .50]))
    first = [curriculum.sample(0) for _ in range(200)]
    assert {s['role'] for s in first} == {CURRENT, CHALLENGE}
    assert all((s['geometry_level'], s['dr_level']) == (1, 0)
               for s in first if s['role'] == CHALLENGE)
    promote(curriculum)
    second = [curriculum.sample(0) for _ in range(200)]
    assert {s['role'] for s in second} == {CURRENT, REPLAY, CHALLENGE}
    assert all((s['geometry_level'], s['dr_level']) == (0, 0)
               for s in second if s['role'] == REPLAY)
    assert all((s['geometry_level'], s['dr_level']) == (1, 1)
               for s in second if s['role'] == CHALLENGE)
    assert all(s['version'] == 1 for s in second)
    assert curriculum.current(0)['dr_level'] == 0


def test_old_partial_windows_and_old_completed_evidence_expire_by_policy_updates():
    curriculum = CourseCurriculum(settings(max_window_updates=10))
    for _ in range(3):
        curriculum.record(0, True, False, CURRENT, 0, 0, True, True)
    curriculum.record(0, True, False, CURRENT, 0, 11, True, True)
    assert curriculum.diagnostics()[0]['window']['episodes'] == 1
    assert curriculum.diagnostics()[0]['expired_windows'] == 1
    for _ in range(3):
        event = curriculum.record(0, True, False, CURRENT, 0, 11, True, True)
    assert event['event'] == 'window'
    assert window(curriculum, 4, update=22)['event'] == 'window'
    assert curriculum.current(0)['version'] == 0
    assert curriculum.diagnostics()[0]['expired_streaks'] == 1
    assert window(curriculum, 4, update=22)['event'] == 'promote'


def test_expired_partial_window_breaks_a_previous_completed_window_streak():
    curriculum = CourseCurriculum(settings(max_window_updates=10))
    window(curriculum, 4)
    curriculum.record(0, True, False, CURRENT, 0, 1, True, True)
    curriculum.record(0, True, False, CURRENT, 0, 12, True, True)
    data = curriculum.diagnostics()[0]
    assert data['consecutive_windows'] == 0 and data['window']['episodes'] == 1
    assert data['expired_windows'] == 1


def test_snapshot_restores_partial_evidence_reversible_history_and_sampling_rng():
    original = CourseCurriculum(settings())
    promote(original)
    promote(original)
    window(original, 4, course=1, update=3)
    original.record(1, True, False, CURRENT, 0, 4, True, True)
    for _ in range(17):
        original.sample(0)
    saved = original.snapshot()
    restored = CourseCurriculum(settings())
    restored.restore(saved)
    assert restored.state_dict() == original.state_dict()
    for i in range(100):
        assert restored.sample(i % 2) == original.sample(i % 2)
    for i in range(3):
        a = restored.record(1, True, False, CURRENT, 0, 5, True, True)
        b = original.record(1, True, False, CURRENT, 0, 5, True, True)
        assert a == b
    assert restored.current(1)['version'] == 1
    for curriculum in (original, restored):
        window(curriculum, 0, update=5)
        window(curriculum, 0, update=5)
    assert restored.current(0) == original.current(0)
    assert restored.current(0)['dr_level'] == 0
    # Caller-owned copies cannot mutate a running curriculum.
    saved['states'][0]['geometry_level'] = 999
    assert restored.current(0)['geometry_level'] == 1


def test_fifty_terrain_command_groups_remain_independent_and_restore_exactly():
    original = CourseCurriculum(settings(), num_courses=50)
    promote(original, course=49)
    window(original, 4, course=48)
    original.record(47, False, False, CURRENT, 0, 0, True, True)
    assert original.current(49)['version'] == 1
    assert original.current(48)['version'] == 0
    assert original.diagnostics()[48]['consecutive_windows'] == 1
    assert original.diagnostics()[47]['window']['episodes'] == 1
    restored = CourseCurriculum(settings(), num_courses=50)
    restored.load_state_dict(original.state_dict())
    assert restored.state_dict() == original.state_dict()
    for course in range(50):
        assert original.sample(course) == restored.sample(course)
    assert window(restored, 4, course=48)['event'] == 'promote'
    assert restored.current(47)['version'] == 0
    with pytest.raises(ValueError, match='contract'):
        CourseCurriculum(settings()).load_state_dict(original.state_dict())


@pytest.mark.parametrize('mutation', [
    lambda s: s.update(extra=True),
    lambda s: s.update(schema_version=True),
    lambda s: s.update(num_courses=1),
    lambda s: s['config'].update(promote_success=.95),
    lambda s: s.update(rng_state=('broken',)),
    lambda s: s['states'][0].update(geometry_level=6),
    lambda s: s['states'][0].update(version=100),
    lambda s: s['states'][0]['history'][0].update(dr_level=1),
    lambda s: s['states'][0]['statistics'].update(teacher_successes=100),
    lambda s: s['states'][0]['window'].update(episodes=4),
    lambda s: s['states'][0]['last_window'].update(outcome='demote'),
])
def test_bad_checkpoint_is_rejected_atomically(mutation):
    curriculum = CourseCurriculum(settings())
    promote(curriculum)
    before = curriculum.state_dict()
    bad = deepcopy(before)
    mutation(bad)
    with pytest.raises(ValueError):
        curriculum.load_state_dict(bad)
    assert curriculum.state_dict() == before


@pytest.mark.parametrize('change', [
    dict(extra=1), dict(window_episodes=True), dict(consecutive_windows=0),
    dict(max_window_updates=0), dict(min_boundary_episodes=5),
    dict(min_delayed_episodes=-1), dict(role_fractions=[0., .5, .5]),
    dict(role_fractions=[.6, .25, .16]), dict(promote_success=float('nan')),
    dict(demote_success=.95), dict(geometry_levels=0), dict(seed=True),
])
def test_invalid_contract_is_rejected(change):
    with pytest.raises(ValueError):
        validate_curriculum_config(settings(**change))


def test_invalid_episode_metadata_and_backwards_update_do_not_mutate_state():
    curriculum = CourseCurriculum(settings())
    curriculum.record(0, True, False, CURRENT, 0, 10, True, True)
    before = curriculum.state_dict()
    for args in [(0, 1, False, CURRENT, 0, 11),
                 (0, True, False, 99, 0, 11),
                 (0, True, False, CURRENT, 1, 11),
                 (0, True, False, CURRENT, 0, 9),
                 (10, True, False, CURRENT, 0, 11)]:
        with pytest.raises(ValueError):
            curriculum.record(*args)
        assert curriculum.state_dict() == before
