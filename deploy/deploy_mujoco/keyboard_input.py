"""Focused X11 window key state, independent of MuJoCo's press-only callback."""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
import re
import subprocess
import sys
import threading


def _terminal_ancestors() -> set[int]:
    """Prove stdin belongs to a foreground local process before using X focus.

    A remote SSH PTY can coexist with an unrelated desktop DISPLAY. Neither a
    TTY nor DISPLAY alone is evidence that its input came from that desktop.
    """
    if any(os.environ.get(name) for name in ("SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY")):
        return set()
    if not os.environ.get("DISPLAY", "").startswith((":", "unix:")):
        return set()
    try:
        if not os.isatty(0) or os.tcgetpgrp(0) != os.getpgrp():
            return set()
        ancestors = set()
        pid = os.getppid()
        while pid > 1 and pid not in ancestors:
            fields = dict(line.split(":", 1) for line in Path(f"/proc/{pid}/status").read_text().splitlines())
            if fields.get("Name", "").strip().startswith("sshd"):
                return set()
            ancestors.add(pid)
            pid = int(fields["PPid"].strip())
        return ancestors
    except (OSError, ValueError, KeyError):
        return set()


def _x11_query(program: str, window: int, *args: str) -> str:
    # Window-property requests can race with window destruction. A short-lived
    # X client contains Xlib's fatal BadWindow handler without replacing the
    # process-wide handler used by MuJoCo's rendering thread.
    try:
        result = subprocess.run(
            [program, "-id", hex(window), *args], capture_output=True, text=True,
            timeout=.25, check=False, env={**os.environ, "LC_ALL": "C"},
        )
        return result.stdout if result.returncode == 0 else ""
    except (OSError, subprocess.TimeoutExpired):
        return ""


def _terminal_window_matches(window: int, ancestors: set[int]) -> bool:
    """Follow child focus windows to the terminal's EWMH process identity."""
    visited = set()
    for _ in range(8):
        if window <= 1 or window in visited:
            return False
        visited.add(window)
        value = _x11_query("xprop", window, "-notype", "_NET_WM_PID")
        match = re.search(r"^_NET_WM_PID\s*=\s*(\d+)\s*$", value, re.MULTILINE)
        if match:
            return int(match[1]) in ancestors
        tree = _x11_query("xwininfo", window, "-tree")
        parent = re.search(r"^\s*Parent window id:\s*(0x[0-9a-fA-F]+)\b", tree, re.MULTILINE)
        if parent is None:
            return False
        window = int(parent[1], 16)
    return False


class X11WindowKeys:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._x11 = ctypes.CDLL("libX11.so.6")
        x = self._x11
        x.XInitThreads()
        x.XOpenDisplay.argtypes = [ctypes.c_char_p]
        x.XOpenDisplay.restype = ctypes.c_void_p
        x.XGetInputFocus.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong), ctypes.POINTER(ctypes.c_int)]
        x.XQueryKeymap.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        x.XStringToKeysym.argtypes = [ctypes.c_char_p]
        x.XStringToKeysym.restype = ctypes.c_ulong
        x.XKeysymToKeycode.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        x.XKeysymToKeycode.restype = ctypes.c_uint
        x.XCloseDisplay.argtypes = [ctypes.c_void_p]
        self._display = x.XOpenDisplay(None)
        if not self._display:
            raise OSError("Cannot connect to X11 display")
        self._viewer_window = None
        self._terminal_window = None
        self._terminal_checked_window = None
        self._codes = {
            key: x.XKeysymToKeycode(self._display, x.XStringToKeysym(key.encode()))
            for key in "wsadqe"
        }

    @classmethod
    def open(cls) -> X11WindowKeys | None:
        if not sys.platform.startswith("linux") or not os.environ.get("DISPLAY"):
            return None
        if os.environ.get("XDG_SESSION_TYPE") == "wayland":
            return None  # XWayland cannot reliably query keys outside its surface.
        try:
            return cls()
        except OSError:
            return None

    def _focus(self) -> int:
        window, revert = ctypes.c_ulong(), ctypes.c_int()
        self._x11.XGetInputFocus(self._display, ctypes.byref(window), ctypes.byref(revert))
        return window.value

    def note_viewer_event(self) -> None:
        # Called exclusively by the MuJoCo window callback, never terminal input.
        with self._lock:
            if self._display:
                focus = self._focus()
                self._viewer_window = focus if focus > 1 else None

    def note_terminal_event(self) -> bool:
        """Bind a proven local terminal after its actual stdin input arrives.

        Unrelated windows, SSH sessions, tmux sessions without an ancestor
        relationship, and unavailable X utilities keep the character fallback.
        Ownership checks are cached per focus window, never run per frame.
        """
        with self._lock:
            if not self._display:
                return False
            focus = self._focus()
            if focus <= 1:
                return False
            if focus == self._terminal_window:
                return True
            if focus == self._terminal_checked_window:
                return False
            self._terminal_checked_window = focus
            ancestors = _terminal_ancestors()
            if ancestors and _terminal_window_matches(focus, ancestors):
                if self._focus() == focus:
                    self._terminal_window = focus
                    return True
                self._terminal_checked_window = None
            return False

    def focused(self) -> bool:
        with self._lock:
            return bool(self._display and self._focus() in {
                self._viewer_window, self._terminal_window,
            })

    def pressed(self) -> set[str]:
        with self._lock:
            if not self._display:
                return set()
            focus = self._focus()
            if focus not in {self._viewer_window, self._terminal_window}:
                return set()
            bits = ctypes.create_string_buffer(32)
            self._x11.XQueryKeymap(self._display, bits)
            if self._focus() != focus:
                return set()
            return {
                key for key, code in self._codes.items()
                if code and bits.raw[code // 8] & (1 << (code % 8))
            }

    def close(self) -> None:
        with self._lock:
            if self._display:
                self._x11.XCloseDisplay(self._display)
                self._display = None
                self._viewer_window = None
                self._terminal_window = None
                self._terminal_checked_window = None
