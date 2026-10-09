"""Focused X11 window key state, independent of MuJoCo's press-only callback."""
from __future__ import annotations

import ctypes
import os
import sys
import threading


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
        self._window = None
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
                self._window = focus if focus > 1 else None

    def pressed(self) -> set[str]:
        with self._lock:
            if not self._display or self._window is None or self._focus() != self._window:
                return set()
            bits = ctypes.create_string_buffer(32)
            self._x11.XQueryKeymap(self._display, bits)
            if self._focus() != self._window:
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
                self._window = None
