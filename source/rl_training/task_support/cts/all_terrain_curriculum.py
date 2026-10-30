"""CPU-only allocation and student-gated curriculum for CTS all-terrain training.

An episode keeps the role and version returned by :meth:`sample` until it ends.
The environment owns episode identity and must call ``record`` exactly once per
completed episode. Parameters sampled continuously *within* a difficulty range
are deliberately not keys: evidence belongs to a course and range version.
"""
from copy import deepcopy
import math
from numbers import Integral
import random

import torch


COURSE_NAMES = ('up', 'down', 'round_trip', 'motion', 'rough', 'blocks', 'pits',
                'mixed', 'straight', 'platform')
COURSE_FRACTIONS = (.28, .28, .14, .10, .03, .03, .02, .02, .05, .05)
CURRENT, REPLAY, CHALLENGE = 0, 1, 2
DEFAULT_CURRICULUM_CONFIG = dict(
    window_episodes=100, promote_success=.90, demote_success=.70,
    consecutive_windows=2, max_window_updates=1000,
    min_boundary_episodes=10, min_delayed_episodes=30,
    role_fractions=[.60, .25, .15], geometry_levels=7, dr_levels=3, seed=42,
)


def _integer(value, name, minimum=0, maximum=None):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f'{name} must be an integer')
    value = int(value)
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f'{name} is outside its valid range')
    return value


def _fractions(values, count, name):
    if not isinstance(values, (tuple, list)) or len(values) != count:
        raise ValueError(f'{name} requires {count} fractions')
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0
           for v in values):
        raise ValueError(f'{name} requires finite nonnegative fractions')
    total = math.fsum(values)
    if not math.isclose(total, 1., rel_tol=0., abs_tol=1e-9):
        raise ValueError(f'{name} must sum to one')
    return [float(v) / total for v in values]


def validate_curriculum_config(cfg):
    """Validate the entire versioned contract; do not silently fill old fields."""
    if not isinstance(cfg, dict) or set(cfg) != set(DEFAULT_CURRICULUM_CONFIG):
        raise ValueError('Unknown or missing CTS all-terrain curriculum settings')
    for name in ('window_episodes', 'consecutive_windows', 'max_window_updates',
                 'geometry_levels', 'dr_levels'):
        _integer(cfg[name], name, 1)
    for name in ('min_boundary_episodes', 'min_delayed_episodes'):
        _integer(cfg[name], name, 0, cfg['window_episodes'])
    _integer(cfg['seed'], 'seed', 0, 2**63 - 1)
    thresholds = (cfg['demote_success'], cfg['promote_success'])
    if (any(type(v) not in (int, float) or not math.isfinite(v) for v in thresholds)
            or not 0 <= thresholds[0] < thresholds[1] <= 1):
        raise ValueError('Curriculum requires 0 <= demote < promote <= 1')
    fractions = _fractions(cfg['role_fractions'], 3, 'role_fractions')
    if fractions[CURRENT] <= 0:
        raise ValueError('The current curriculum role must receive samples')
    return cfg


def allocate_courses(teacher_mask, fractions=COURSE_FRACTIONS, seed=42):
    """Largest-remainder counts independently in the two fixed cohorts.

    Cohort membership is never changed. Small populations can leave a course
    empty; callers must validate their requested environment count separately.
    The returned tensor has the input mask's device and integer course IDs.
    """
    if (not isinstance(teacher_mask, torch.Tensor) or teacher_mask.dtype != torch.bool
            or teacher_mask.ndim != 1 or teacher_mask.numel() == 0):
        raise ValueError('teacher_mask must be a nonempty one-dimensional bool tensor')
    fractions = _fractions(fractions, len(COURSE_NAMES), 'course fractions')
    seed = _integer(seed, 'seed', 0, 2**63 - 1)
    mask = teacher_mask.detach().cpu()
    result = torch.empty(len(mask), dtype=torch.long)
    for cohort, selected in enumerate((mask, ~mask)):
        ids = selected.nonzero(as_tuple=False).flatten()
        n = len(ids)
        exact = [n * v for v in fractions]
        counts = [math.floor(v) for v in exact]
        order = sorted(range(len(counts)), key=lambda i: (-(exact[i] - counts[i]), i))
        for i in order[:n - sum(counts)]:
            counts[i] += 1
        labels = torch.repeat_interleave(torch.arange(len(counts)), torch.tensor(counts))
        generator = torch.Generator(device='cpu').manual_seed((seed + cohort) % (2**63))
        result[ids[torch.randperm(n, generator=generator)]] = labels
    return result.to(device=teacher_mask.device)


def _empty_window():
    return dict(episodes=0, successes=0, boundary=0, delayed=0,
                start_update=None, end_update=None)


def _empty_statistics():
    return dict(teacher_episodes=0, teacher_successes=0, student_episodes=0,
                student_successes=0, ignored_role=0, ignored_version=0,
                completed_windows=0, expired_windows=0, expired_streaks=0,
                promotions=0, demotions=0, bound_holds=0)


def _initial_state():
    return dict(geometry_level=0, dr_level=0, next_axis='geometry', version=0,
                history=[], window=_empty_window(), recent_windows=[],
                last_window=None, statistics=_empty_statistics())


class CourseCurriculum:
    """Independent course grades with teacher/student sharing one distribution.

    Successful current-student windows alternate a geometry and a domain-
    randomization increment, skipping an axis at its upper bound. Failure rolls
    back the latest increment, preserving the other axis. Coverage is required
    for promotion only; failure may downgrade even if difficult cases cannot be
    completed. All state, including the private sampling RNG, is checkpointed.
    """
    SCHEMA_VERSION = 1

    def __init__(self, cfg, num_courses=len(COURSE_NAMES)):
        self.cfg = deepcopy(validate_curriculum_config(cfg))
        self.num_courses = _integer(num_courses, 'num_courses', 1)
        self._states = [_initial_state() for _ in range(self.num_courses)]
        self._rng = random.Random(self.cfg['seed'])
        self._last_update_id = -1

    def _state(self, course_id):
        return self._states[_integer(course_id, 'course_id', 0, self.num_courses - 1)]

    def _advance(self, state):
        """Return a prospective one-axis grade without mutating the course."""
        preferred = state['next_axis']
        for axis in (preferred, 'dr' if preferred == 'geometry' else 'geometry'):
            key = axis + '_level'
            if state[key] + 1 < self.cfg[axis + '_levels']:
                result = {k: state[k] for k in ('geometry_level', 'dr_level', 'next_axis')}
                result[key] += 1
                result['next_axis'] = 'dr' if axis == 'geometry' else 'geometry'
                return result
        return None

    def sample(self, course_id):
        """Sample current/replay/challenge; empty pools return current samples."""
        state = self._state(course_id)
        fractions = self.cfg['role_fractions']
        draw = self._rng.random()
        role = (CURRENT if draw < fractions[0] else
                REPLAY if draw < fractions[0] + fractions[1] else CHALLENGE)
        grade = state
        if role == REPLAY:
            if state['history']:
                grade = self._rng.choice(state['history'])
            else:
                role = CURRENT
        elif role == CHALLENGE:
            grade = self._advance(state)
            if grade is None:
                role, grade = CURRENT, state
        return dict(geometry_level=grade['geometry_level'], dr_level=grade['dr_level'],
                    role=role, version=state['version'])

    def current(self, course_id):
        """Return an immutable-by-copy view for diagnostics or environment setup."""
        state = self._state(course_id)
        return {key: state[key] for key in ('geometry_level', 'dr_level', 'version', 'next_axis')}

    def _expire(self, state, update_id):
        limit = self.cfg['max_window_updates']
        window = state['window']
        if window['start_update'] is not None and update_id - window['start_update'] > limit:
            state['window'] = _empty_window()
            state['statistics']['expired_windows'] += 1
            if state['recent_windows']:
                state['statistics']['expired_streaks'] += 1
                state['recent_windows'].clear()
        recent = state['recent_windows']
        fresh = [w for w in recent if update_id - w['start_update'] <= limit]
        if len(fresh) != len(recent):
            state['statistics']['expired_streaks'] += 1
            state['recent_windows'] = fresh

    def record(self, course_id, success, teacher, role, version, update_id,
               boundary=False, delayed=False):
        """Record one completed episode and return a completed-window event.

        The caller must retain the episode's start-time ``role`` and ``version``.
        A stale episode or replay/challenge result is diagnostic evidence only.
        ``update_id`` counts policy updates, must be globally nondecreasing, and
        cannot be replaced by per-environment episode counts.
        """
        state = self._state(course_id)
        for name, value in (('success', success), ('teacher', teacher),
                            ('boundary', boundary), ('delayed', delayed)):
            if type(value) is not bool:
                raise ValueError(f'{name} must be a bool')
        role = _integer(role, 'role', CURRENT, CHALLENGE)
        version = _integer(version, 'version')
        if version > state['version']:
            raise ValueError('Episode version is ahead of its curriculum')
        update_id = _integer(update_id, 'update_id')
        if update_id < self._last_update_id:
            raise ValueError('Curriculum update_id must not move backwards')
        self._last_update_id = update_id
        self._expire(state, update_id)
        statistics = state['statistics']
        cohort = 'teacher' if teacher else 'student'
        statistics[cohort + '_episodes'] += 1
        statistics[cohort + '_successes'] += int(success)
        if teacher:
            return None
        if version != state['version']:
            statistics['ignored_version'] += 1
            return None
        if role != CURRENT:
            statistics['ignored_role'] += 1
            return None
        window = state['window']
        if window['episodes'] == 0:
            window['start_update'] = update_id
        window['end_update'] = update_id
        window['episodes'] += 1
        window['successes'] += int(success)
        window['boundary'] += int(boundary)
        window['delayed'] += int(delayed)
        if window['episodes'] < self.cfg['window_episodes']:
            return None
        return self._complete_window(course_id, state)

    def _complete_window(self, course_id, state):
        window = state['window']
        state['window'] = _empty_window()
        rate = window['successes'] / window['episodes']
        covered = (window['boundary'] >= self.cfg['min_boundary_episodes']
                   and window['delayed'] >= self.cfg['min_delayed_episodes'])
        if rate >= self.cfg['promote_success'] and covered:
            outcome, reason = 'promote', 'insufficient_consecutive_windows'
        elif rate < self.cfg['demote_success']:
            outcome, reason = 'demote', 'insufficient_consecutive_windows'
        else:
            outcome = 'hold'
            reason = 'insufficient_coverage' if rate >= self.cfg['promote_success'] else 'between_thresholds'
        completed = dict(window, outcome=outcome)
        state['last_window'] = deepcopy(completed)
        state['statistics']['completed_windows'] += 1
        recent = state['recent_windows']
        if outcome == 'hold':
            recent.clear()
        else:
            if recent and recent[-1]['outcome'] != outcome:
                recent.clear()
            recent.append(completed)
        event = 'hold' if outcome == 'hold' else 'window'
        if len(recent) >= self.cfg['consecutive_windows']:
            recent.clear()
            if outcome == 'promote':
                grade = self._advance(state)
                if grade is None:
                    reason = 'upper_bound'
                    event = 'hold'
                    state['statistics']['bound_holds'] += 1
                else:
                    state['history'].append({k: state[k] for k in
                                             ('geometry_level', 'dr_level', 'next_axis')})
                    state.update(grade)
                    event, reason = 'promote', 'student_success_windows'
            elif state['history']:
                state.update(state['history'].pop())
                event, reason = 'demote', 'student_failure_windows'
            else:
                event, reason = 'hold', 'lower_bound'
                state['statistics']['bound_holds'] += 1
            if event in ('promote', 'demote'):
                state['version'] += 1
                state['statistics']['promotions' if event == 'promote' else 'demotions'] += 1
        return dict(course_id=int(course_id), event=event, reason=reason, window=deepcopy(completed),
                    **self.current(course_id))

    def diagnostics(self):
        """Per-course raw counters; reporting owns naming and aggregation."""
        return [dict(self.current(i), **deepcopy(s['statistics']),
                     window=deepcopy(s['window']), last_window=deepcopy(s['last_window']),
                     consecutive_windows=len(s['recent_windows']))
                for i, s in enumerate(self._states)]

    def state_dict(self):
        return dict(schema_version=self.SCHEMA_VERSION, config=deepcopy(self.cfg),
                    num_courses=self.num_courses, states=deepcopy(self._states),
                    rng_state=self._rng.getstate(), last_update_id=self._last_update_id)

    snapshot = state_dict

    def _validate_window(self, window, completed, last_update):
        expected = set(_empty_window()) | ({'outcome'} if completed else set())
        if not isinstance(window, dict) or set(window) != expected:
            raise ValueError('Invalid curriculum window fields')
        n = _integer(window['episodes'], 'window episodes', 0, self.cfg['window_episodes'])
        if (completed and n != self.cfg['window_episodes']) or (not completed and n >= self.cfg['window_episodes']):
            raise ValueError('Invalid completed/partial curriculum window length')
        for key in ('successes', 'boundary', 'delayed'):
            _integer(window[key], 'window ' + key, 0, n)
        if n == 0:
            if window['start_update'] is not None or window['end_update'] is not None:
                raise ValueError('An empty curriculum window cannot have update timestamps')
        else:
            start = _integer(window['start_update'], 'window start update', 0, last_update)
            _integer(window['end_update'], 'window end update', start, last_update)
            if window['end_update'] - start > self.cfg['max_window_updates']:
                raise ValueError('Curriculum window exceeds its maximum update age')
        if completed:
            rate = window['successes'] / n
            covered = (window['boundary'] >= self.cfg['min_boundary_episodes']
                       and window['delayed'] >= self.cfg['min_delayed_episodes'])
            expected_outcome = ('promote' if rate >= self.cfg['promote_success'] and covered else
                                'demote' if rate < self.cfg['demote_success'] else 'hold')
            if window['outcome'] != expected_outcome:
                raise ValueError('Curriculum window outcome does not match its evidence')

    def _validate_state(self, state, last_update):
        if not isinstance(state, dict) or set(state) != set(_initial_state()):
            raise ValueError('Invalid curriculum course state fields')
        for axis in ('geometry', 'dr'):
            _integer(state[axis + '_level'], axis + ' level', 0, self.cfg[axis + '_levels'] - 1)
        _integer(state['version'], 'curriculum version')
        if state['next_axis'] not in ('geometry', 'dr') or not isinstance(state['history'], list):
            raise ValueError('Invalid curriculum axis/history')
        expected = dict(geometry_level=0, dr_level=0, next_axis='geometry')
        for previous in state['history']:
            if not isinstance(previous, dict) or set(previous) != set(expected):
                raise ValueError('Invalid curriculum history fields')
            for axis in ('geometry', 'dr'):
                _integer(previous[axis + '_level'], 'history ' + axis + ' level',
                         0, self.cfg[axis + '_levels'] - 1)
            if previous != expected:
                raise ValueError('Curriculum history is not a sequence of single-axis changes')
            expected = self._advance(previous)
            if expected is None:
                raise ValueError('Curriculum history extends beyond its upper bound')
        if expected != {key: state[key] for key in expected}:
            raise ValueError('Curriculum grade does not match its reversible history')
        stats = state['statistics']
        if not isinstance(stats, dict) or set(stats) != set(_empty_statistics()):
            raise ValueError('Invalid curriculum statistics fields')
        for key, value in stats.items():
            _integer(value, 'statistics ' + key)
        for cohort in ('teacher', 'student'):
            if stats[cohort + '_successes'] > stats[cohort + '_episodes']:
                raise ValueError('Curriculum successes exceed episode count')
        if (stats['ignored_role'] + stats['ignored_version'] > stats['student_episodes']
                or stats['promotions'] + stats['demotions'] != state['version']
                or stats['promotions'] - stats['demotions'] != len(state['history'])
                or (stats['promotions'] + stats['demotions'] + stats['bound_holds'])
                * self.cfg['consecutive_windows'] > stats['completed_windows']):
            raise ValueError('Curriculum diagnostics and progression are inconsistent')
        self._validate_window(state['window'], False, last_update)
        counted = stats['completed_windows'] * self.cfg['window_episodes'] + state['window']['episodes']
        if counted > stats['student_episodes'] - stats['ignored_role'] - stats['ignored_version']:
            raise ValueError('Curriculum windows exceed eligible student episodes')
        recent = state['recent_windows']
        if not isinstance(recent, list) or len(recent) >= self.cfg['consecutive_windows']:
            raise ValueError('Invalid curriculum consecutive-window history')
        for index, window in enumerate(recent):
            self._validate_window(window, True, last_update)
            if (window['outcome'] == 'hold' or
                    (index and (window['outcome'] != recent[0]['outcome'] or
                                window['start_update'] < recent[index - 1]['end_update']))):
                raise ValueError('Invalid curriculum consecutive-window evidence')
        if recent and state['window']['episodes'] and state['window']['start_update'] < recent[-1]['end_update']:
            raise ValueError('Curriculum windows are not chronologically ordered')
        if state['last_window'] is not None:
            self._validate_window(state['last_window'], True, last_update)
            if not stats['completed_windows']:
                raise ValueError('Curriculum last window lacks a completed-window count')
        elif stats['completed_windows']:
            raise ValueError('Curriculum completed-window count lacks its last window')
        if recent and recent[-1] != state['last_window']:
            raise ValueError('Curriculum streak does not end with its last window')

    def load_state_dict(self, saved):
        """Validate before mutation, including contract, counters, history and RNG."""
        expected = {'schema_version', 'config', 'num_courses', 'states', 'rng_state', 'last_update_id'}
        if not isinstance(saved, dict) or set(saved) != expected:
            raise ValueError('Invalid CTS all-terrain curriculum checkpoint fields')
        if (type(saved['schema_version']) is not int or saved['schema_version'] != self.SCHEMA_VERSION
                or type(saved['num_courses']) is not int or saved['num_courses'] != self.num_courses
                or saved['config'] != self.cfg):
            raise ValueError('CTS all-terrain curriculum checkpoint contract differs')
        validate_curriculum_config(saved['config'])
        last_update = _integer(saved['last_update_id'], 'last_update_id', -1)
        states = deepcopy(saved['states'])
        if not isinstance(states, list) or len(states) != self.num_courses:
            raise ValueError('CTS all-terrain curriculum checkpoint course count differs')
        for state in states:
            self._validate_state(state, last_update)
        rng = random.Random()
        try:
            rng.setstate(saved['rng_state'])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError('Invalid curriculum sampling RNG state') from exc
        self._states, self._rng, self._last_update_id = states, rng, last_update

    restore = load_state_dict
