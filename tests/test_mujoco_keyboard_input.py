"""Physical keyboard state and terminal provenance, without opening a display."""
import ctypes
import importlib.util
from pathlib import Path
import subprocess
import threading
from types import SimpleNamespace

import pytest


SPEC = importlib.util.spec_from_file_location(
    "mujoco_keyboard_input_under_test",
    Path(__file__).resolve().parents[1] / "deploy/deploy_mujoco/keyboard_input.py",
)
keyboard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(keyboard)


class FakeX11:
    def __init__(self):
        self.focus = 20
        self.held = set()
        self.codes = dict(zip("wsadqe", range(24, 30)))
        self.queries = self.closed = 0
        self.focus_after_query = None

    def XGetInputFocus(self, display, window, revert):
        ctypes.cast(window, ctypes.POINTER(ctypes.c_ulong))[0] = self.focus

    def XQueryKeymap(self, display, buffer):
        self.queries += 1
        bits = bytearray(32)
        for key in self.held:
            code = self.codes[key]
            bits[code // 8] |= 1 << (code % 8)
        buffer.raw = bytes(bits)
        if self.focus_after_query is not None:
            self.focus = self.focus_after_query

    def XCloseDisplay(self, display):
        self.closed += 1


@pytest.fixture
def reader():
    x11 = FakeX11()
    keys = keyboard.X11WindowKeys.__new__(keyboard.X11WindowKeys)
    keys._lock = threading.Lock()
    keys._x11, keys._display, keys._codes = x11, 1, x11.codes
    keys._viewer_window = keys._terminal_window = keys._terminal_checked_window = None
    return keys, x11


def test_viewer_tracks_all_held_keys_release_and_focus_without_repeats(reader):
    keys, x11 = reader
    x11.held = {"w", "a", "q"}
    assert not keys.focused()
    assert keys.pressed() == set()
    keys.note_viewer_event()
    assert keys.focused()
    for _ in range(100):
        assert keys.pressed() == {"w", "a", "q"}
    x11.held.remove("a")
    assert keys.pressed() == {"w", "q"}
    x11.focus = 99
    assert not keys.focused()
    assert keys.pressed() == set()


def test_terminal_child_window_requires_matching_ancestor_pid_and_caches(reader, monkeypatch):
    keys, x11 = reader
    monkeypatch.setattr(keyboard, "_terminal_ancestors", lambda: {100, 101})
    queries = []

    def query(program, window, *args):
        queries.append((program, window))
        return {
            ("xprop", 20): "_NET_WM_PID: not found.\n",
            ("xwininfo", 20): "  Parent window id: 0x28 (terminal)\n",
            ("xprop", 40): "_NET_WM_PID = 100\n",
        }.get((program, window), "")

    monkeypatch.setattr(keyboard, "_x11_query", query)
    assert keys.note_terminal_event()
    assert queries == [("xprop", 20), ("xwininfo", 20), ("xprop", 40)]
    for _ in range(10):
        assert keys.note_terminal_event()
    assert len(queries) == 3
    x11.held = {"w", "a"}
    assert keys.focused() and keys.pressed() == {"w", "a"}
    x11.focus = 50
    keys.note_viewer_event()
    assert keys.pressed() == {"w", "a"}
    x11.focus = 20
    assert keys.focused() and keys.pressed() == {"w", "a"}
    x11.focus = 99
    assert not keys.focused() and keys.pressed() == set()


def test_terminal_cannot_authorize_unrelated_foreground_window(reader, monkeypatch):
    keys, x11 = reader
    monkeypatch.setattr(keyboard, "_terminal_ancestors", lambda: {100})
    calls = []
    monkeypatch.setattr(keyboard, "_x11_query", lambda *args: calls.append(args) or "_NET_WM_PID = 999\n")
    assert not keys.note_terminal_event()
    assert not keys.note_terminal_event()  # Failed ownership is cached, too.
    assert len(calls) == 1
    x11.held = {"w", "a"}
    assert not keys.focused() and keys.pressed() == set()


def test_focus_change_during_key_query_discards_snapshot(reader):
    keys, x11 = reader
    keys.note_viewer_event()
    x11.held = {"w", "a"}
    x11.focus_after_query = 99
    assert keys.pressed() == set()


def test_focus_change_during_terminal_validation_does_not_authorize_either_window(reader, monkeypatch):
    keys, x11 = reader
    monkeypatch.setattr(keyboard, "_terminal_ancestors", lambda: {100})

    def query(*args):
        x11.focus = 99
        return "_NET_WM_PID = 100\n"

    monkeypatch.setattr(keyboard, "_x11_query", query)
    assert not keys.note_terminal_event()
    assert not keys.focused()
    x11.focus = 20
    assert not keys.focused()


@pytest.fixture
def local_terminal(monkeypatch):
    for name in ("SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setattr(keyboard.os, "isatty", lambda fd: True)
    monkeypatch.setattr(keyboard.os, "tcgetpgrp", lambda fd: 50)
    monkeypatch.setattr(keyboard.os, "getpgrp", lambda: 50)
    monkeypatch.setattr(keyboard.os, "getppid", lambda: 100)
    processes = {
        "100": "Name:\tbash\nPPid:\t101\n",
        "101": "Name:\tgnome-terminal-\nPPid:\t1\n",
    }
    monkeypatch.setattr(keyboard.Path, "read_text", lambda path: processes[path.parent.name])
    return processes


def test_local_foreground_terminal_ancestors(local_terminal):
    assert keyboard._terminal_ancestors() == {100, 101}


@pytest.mark.parametrize("name", ["SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY"])
def test_ssh_stdin_never_binds_desktop_even_with_local_display(reader, local_terminal, monkeypatch, name):
    keys, _ = reader
    monkeypatch.setenv(name, "present")
    monkeypatch.setattr(keyboard, "_terminal_window_matches", lambda *args: pytest.fail("SSH must not query desktop ownership"))
    assert not keys.note_terminal_event()
    assert not keys.focused()


def test_ssh_ancestor_still_rejected_when_environment_is_missing(local_terminal):
    local_terminal["101"] = "Name:\tsshd-session\nPPid:\t1\n"
    assert keyboard._terminal_ancestors() == set()


@pytest.mark.parametrize("kind", ["not_tty", "background", "remote_display", "no_display", "proc_denied"])
def test_terminal_unproven_provenance_rejected(local_terminal, monkeypatch, kind):
    if kind == "not_tty":
        monkeypatch.setattr(keyboard.os, "isatty", lambda fd: False)
    elif kind == "background":
        monkeypatch.setattr(keyboard.os, "tcgetpgrp", lambda fd: 99)
    elif kind == "remote_display":
        monkeypatch.setenv("DISPLAY", "remote-host:0")
    elif kind == "no_display":
        monkeypatch.delenv("DISPLAY")
    else:
        def denied(path):
            raise PermissionError("/proc unavailable")
        monkeypatch.setattr(keyboard.Path, "read_text", denied)
    assert keyboard._terminal_ancestors() == set()


@pytest.mark.parametrize("failure", [FileNotFoundError(), subprocess.TimeoutExpired("xprop", .25)])
def test_missing_or_stalled_x_utilities_fail_closed(monkeypatch, failure):
    def fail(*args, **kwargs):
        raise failure
    monkeypatch.setattr(keyboard.subprocess, "run", fail)
    assert not keyboard._terminal_window_matches(20, {100})


def test_x_window_destroyed_between_queries_fails_closed(monkeypatch):
    monkeypatch.setattr(keyboard.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout=""))
    assert not keyboard._terminal_window_matches(20, {100})


def test_close_is_idempotent_and_clears_window_authority(reader):
    keys, x11 = reader
    keys.note_viewer_event()
    keys.close()
    keys.close()
    assert x11.closed == 1
    assert not keys.focused() and keys.pressed() == set()
    assert not keys.note_terminal_event()


def test_wayland_does_not_claim_physical_key_support(monkeypatch):
    monkeypatch.setattr(keyboard.sys, "platform", "linux")
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setenv("XDG_SESSION_TYPE", "wayland")
    monkeypatch.setattr(keyboard.ctypes, "CDLL", lambda *args: pytest.fail("must not connect to display"))
    assert keyboard.X11WindowKeys.open() is None
