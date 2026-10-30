"""Shared manual command behavior, without a viewer, checkpoint or optimizer."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'deploy/deploy_mujoco'))
import deploy_mujoco as runtime


@pytest.fixture
def controls(monkeypatch):
    held = set()
    window = SimpleNamespace(pressed=lambda: held.copy(), focused=lambda: True,
                             note_viewer_event=lambda: None, note_terminal_event=lambda: True,
                             close=lambda: None)
    monkeypatch.setattr(runtime.X11WindowKeys, 'open', lambda: window)
    keys = runtime.TeleopKeyboard()
    yield keys, held, window
    keys.close()


def ticks(keys, seconds):
    return np.asarray([keys.velocity_command() for _ in range(round(seconds / runtime.POLICY_DT) + 1)])


def test_combination_keeps_first_key_without_repeat_and_ramps_each_axis(controls):
    keys, held, _ = controls
    held.add('w')
    first = keys.velocity_command()
    assert 0 < first[0] < runtime.TELEOP_LIMITS[0]
    held.update({'a', 'q'})
    commands = ticks(keys, runtime.TELEOP_RISE_TIME + runtime.TELEOP_KEY_TIMEOUT)
    assert np.all(np.diff(commands, axis=0) >= -1e-7)
    np.testing.assert_allclose(commands[-1], runtime.TELEOP_LIMITS)
    held.remove('a')
    stopping = ticks(keys, runtime.TELEOP_STOP_TIME)
    assert 0 < stopping[0, 1] < runtime.TELEOP_LIMITS[1]
    np.testing.assert_allclose(stopping[:, 0], runtime.TELEOP_LIMITS[0])
    np.testing.assert_allclose(stopping[:, 2], runtime.TELEOP_LIMITS[2])
    assert stopping[-1, 1] == 0


def test_opposites_cancel_and_reverse_brakes_before_accelerating(controls):
    keys, held, _ = controls
    held.update({'w', 's', 'a', 'd', 'q', 'e'})
    np.testing.assert_array_equal(ticks(keys, 1.), 0.)
    held.clear()
    held.add('w')
    ticks(keys, runtime.TELEOP_RISE_TIME)
    held.clear()
    held.add('s')
    reverse = ticks(keys, runtime.TELEOP_STOP_TIME + runtime.TELEOP_RISE_TIME)
    assert reverse[0, 0] > 0
    assert np.any(reverse[:, 0] == 0)
    assert np.all(np.diff(reverse[:, 0]) <= 1e-7)
    assert reverse[-1, 0] == pytest.approx(-runtime.TELEOP_LIMITS[0])


def test_clear_and_focus_loss_stop_without_ramp_and_restart_from_zero(controls):
    keys, held, window = controls
    held.update({'w', 'a'})
    ticks(keys, runtime.TELEOP_RISE_TIME)
    keys.clear_motion()
    first = keys.velocity_command()
    assert np.all(first[:2] > 0)
    assert np.all(first[:2] < runtime.TELEOP_LIMITS[:2])
    window.focused = lambda: False
    np.testing.assert_array_equal(keys.velocity_command(), 0.)
    window.focused = lambda: True
    np.testing.assert_array_equal(keys.velocity_command(), first)


def test_terminal_uses_physical_keys_without_repeat_timeout(monkeypatch, controls):
    keys, held, _ = controls
    keys.note_terminal_event()
    keys.on_char('w')
    held.update({'w', 'a'})
    monkeypatch.setattr(runtime.time, 'monotonic', lambda: 1e9)
    np.testing.assert_allclose(ticks(keys, runtime.TELEOP_RISE_TIME)[-1], [.7, .5, 0])
    held.clear()
    np.testing.assert_array_equal(ticks(keys, runtime.TELEOP_STOP_TIME)[-1], 0.)


def test_reset_and_damping_remain_events(controls):
    keys, _, _ = controls
    keys.on_keycode(259)
    keys.on_char('\x7f')
    keys.on_char('r')
    keys.on_keycode(256)
    assert keys.take_events() == ['reset', 'reset', 'r', 'escape']


def test_delayed_terminal_repeat_after_focus_loss_cannot_restart_motion(controls):
    keys, held, window = controls
    keys.note_terminal_event()
    held.add('w')
    ticks(keys, runtime.TELEOP_RISE_TIME)
    window.focused = lambda: False
    window.note_terminal_event = lambda: False
    keys.note_terminal_event()
    keys.on_char('w')
    np.testing.assert_array_equal(keys.velocity_command(), 0.)


def test_fallback_expires_instead_of_latching(monkeypatch):
    monkeypatch.setattr(runtime.X11WindowKeys, 'open', lambda: None)
    now = [0.]
    monkeypatch.setattr(runtime.time, 'monotonic', lambda: now[0])
    keys = runtime.TeleopKeyboard()
    keys.on_char('w')
    assert keys.velocity_command()[0] > 0
    now[0] = runtime.TELEOP_KEY_TIMEOUT + .01
    np.testing.assert_array_equal(keys.velocity_command(), 0.)
    keys.close()
