"""Continuous request timing, exact delivery boundaries and archived RNG parity."""
import hashlib
import math

import pytest
import torch

from task_support.student_v1.configuration import StudentV1PerceptionCfg
from task_support.student_v1.perception import (
    StudentObservationState, continuous_delivery_steps, grid_points,
)


def scene(n):
    grid = grid_points('cpu')
    proprio = torch.zeros(n, 57)
    proprio[:, 5] = -1.
    hits = torch.cat((grid, .2 * grid[:, :1]), -1)[None].repeat(n, 1, 1)
    base = torch.tensor([[0., 0., .5]]).repeat(n, 1)
    return proprio, hits, base, torch.zeros(n), torch.eye(3)[None].repeat(n, 1, 1)


def continuous_cfg(**overrides):
    values = dict(continuous_delay=True, min_delay_s=.28, max_delay_s=.42,
                  zero_delay_probability=.2, terrain_visibility=False,
                  level=0., clean_probability=1., stress_probability=0.,
                  map_update_period_s=.10, update_jitter_s=0.)
    values.update(overrides)
    return StudentV1PerceptionCfg(**values)


def first_capture(state):
    state._initialize()
    state.next_capture.zero_()
    return state.observe(*scene(state.num_envs), 0)


def test_continuous_tick_rounding_handles_exact_and_adjacent_representable_boundaries():
    ticks = torch.arange(0, 24, dtype=torch.float64)
    exact = ticks * .02
    lower = torch.nextafter(exact[1:], torch.full_like(exact[1:], -math.inf))
    upper = torch.nextafter(exact, torch.full_like(exact, math.inf))
    assert torch.equal(continuous_delivery_steps(exact, .02), ticks.long())
    assert torch.equal(continuous_delivery_steps(lower, .02), ticks[1:].long())
    assert torch.equal(continuous_delivery_steps(upper, .02), ticks.long() + 1)
    decimal = torch.tensor([.28, .30, .32, .42], dtype=torch.float64)
    assert continuous_delivery_steps(decimal, .02).tolist() == [14, 15, 16, 21]


def test_actual_sensor_draws_continuous_seconds_and_delivers_at_or_after_request():
    torch.manual_seed(29)
    state = StudentObservationState(256, 'cpu', .02, continuous_cfg(), 4)
    first_capture(state)
    requested, actual = state.requested_delay_s, state.actual_delay_s
    zero, delayed = state.zero_delivery, ~state.zero_delivery
    assert 30 <= int(zero.sum()) <= 75
    assert (requested[zero] == 0).all() and (actual[zero] == 0).all()
    assert (requested[delayed] >= .28).all() and (requested[delayed] <= .42).all()
    assert requested[delayed].unique().numel() > 100
    assert (actual >= requested).all() and (actual - requested < .02 + 1e-15).all()
    assert torch.equal(state.packet_arrival[0], (actual / .02).round().long())
    assert (state.capture[zero] == 0).all() and (state.capture[delayed] == -1).all()
    args = scene(state.num_envs)
    state.next_capture.fill_(1000)
    for step in range(1, 23):
        state.observe(*args, step)
        assert torch.equal(state.capture >= 0, state.packet_arrival[0] <= step)


def test_continuous_delay_endpoint_does_not_add_an_extra_control_tick():
    state = StudentObservationState(2, 'cpu', .02,
                                    continuous_cfg(min_delay_s=.28, max_delay_s=.28,
                                                   zero_delay_probability=0.))
    first_capture(state)
    assert state.packet_arrival[0].tolist() == [14, 14]
    assert state.requested_delay_s.tolist() == [.28, .28]


def test_zero_mixture_is_independent_of_clean_noise_and_reset_clears_delay_diagnostics():
    state = StudentObservationState(64, 'cpu', .02, continuous_cfg())
    first_capture(state)
    assert (state.severity == 0).all()
    assert state.zero_delivery.any() and (~state.zero_delivery).any()
    original = state.requested_delay_s.clone()
    ids = torch.tensor([0, 1, 2])
    state.reset(ids)
    assert (state.requested_delay_s[ids] == 0).all() and (state.actual_delay_s[ids] == 0).all()
    assert torch.equal(state.requested_delay_s[3:], original[3:])
    assert (state.packet_arrival[:, ids] == -1).all()


def test_disabled_continuous_delay_matches_frozen_v12_outputs_timing_and_rng():
    # Oracle captured from the pre-change v12 implementation, source SHA256:
    # 1e7c8dfa09c84061eb2aa4d43dc4dcbbd9a3929866ca568aa25ab2fc5fd20066.
    # Includes noisy observations, packet timing and the final CPU RNG state.
    cfg = StudentV1PerceptionCfg(min_delay_s=.28, max_delay_s=.42,
                                zero_delay_probability=.2, continuous_delay=False)
    torch.manual_seed(139)
    state = StudentObservationState(4, 'cpu', .02, cfg, 4)
    args = scene(4)
    for step in range(30):
        output = state.observe(*args, step)
    digest = hashlib.sha256()
    for tensor in (output, state.packet_capture, state.packet_arrival,
                   state.next_capture, torch.get_rng_state()):
        digest.update(tensor.numpy().tobytes())
    assert digest.hexdigest() == '0d604a4a319b62f46abea55fbc9bc2beb992e91100f6efd817aa96504a064879'
    assert not hasattr(state, 'requested_delay_s')


@pytest.mark.parametrize('field,value', [
    ('continuous_delay', 1), ('continuous_delay', 'true'),
    ('min_delay_s', True), ('max_delay_s', False),
    ('min_delay_s', float('nan')), ('max_delay_s', float('inf')),
    ('min_delay_s', -.01), ('max_delay_s', .1),
])
def test_delay_configuration_rejects_invalid_duration_and_mode(field, value):
    cfg = continuous_cfg()
    setattr(cfg, field, value)
    with pytest.raises(ValueError):
        cfg.validate()
