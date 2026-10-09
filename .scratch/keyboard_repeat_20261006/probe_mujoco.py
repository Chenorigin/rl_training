"""Verify actual deployment callbacks using an isolated, empty MuJoCo viewer."""
import ctypes as c
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "deploy/deploy_mujoco"))
import deploy_mujoco as deploy
import mujoco

x = c.CDLL("libX11.so.6")
t = c.CDLL("libXtst.so.6")
x.XOpenDisplay.argtypes = [c.c_char_p]
x.XOpenDisplay.restype = c.c_void_p
x.XDefaultRootWindow.argtypes = [c.c_void_p]
x.XDefaultRootWindow.restype = c.c_ulong
x.XQueryTree.argtypes = [c.c_void_p, c.c_ulong, c.POINTER(c.c_ulong), c.POINTER(c.c_ulong), c.POINTER(c.POINTER(c.c_ulong)), c.POINTER(c.c_uint)]
x.XInternAtom.argtypes = [c.c_void_p, c.c_char_p, c.c_int]
x.XInternAtom.restype = c.c_ulong
x.XGetWindowProperty.argtypes = [c.c_void_p, c.c_ulong, c.c_ulong, c.c_long, c.c_long, c.c_int, c.c_ulong, c.POINTER(c.c_ulong), c.POINTER(c.c_int), c.POINTER(c.c_ulong), c.POINTER(c.c_ulong), c.POINTER(c.c_void_p)]
x.XGetInputFocus.argtypes = [c.c_void_p, c.POINTER(c.c_ulong), c.POINTER(c.c_int)]
x.XSetInputFocus.argtypes = [c.c_void_p, c.c_ulong, c.c_int, c.c_ulong]
x.XSync.argtypes = [c.c_void_p, c.c_int]
x.XFree.argtypes = [c.c_void_p]
x.XCloseDisplay.argtypes = [c.c_void_p]
t.XTestFakeKeyEvent.argtypes = [c.c_void_p, c.c_uint, c.c_int, c.c_ulong]
d = x.XOpenDisplay(None)
assert d
pid_atom = x.XInternAtom(d, b"_NET_WM_PID", 0)
old_focus, old_revert = c.c_ulong(), c.c_int()
x.XGetInputFocus(d, c.byref(old_focus), c.byref(old_revert))

def own_window(window):
    kind, fmt, count, remain, value = c.c_ulong(), c.c_int(), c.c_ulong(), c.c_ulong(), c.c_void_p()
    x.XGetWindowProperty(d, window, pid_atom, 0, 1, 0, 0, c.byref(kind), c.byref(fmt), c.byref(count), c.byref(remain), c.byref(value))
    try:
        if value and count.value and fmt.value == 32:
            if c.cast(value, c.POINTER(c.c_ulong))[0] == os.getpid():
                return window
    finally:
        if value:
            x.XFree(value)
    root, parent, children, size = c.c_ulong(), c.c_ulong(), c.POINTER(c.c_ulong)(), c.c_uint()
    x.XQueryTree(d, window, c.byref(root), c.byref(parent), c.byref(children), c.byref(size))
    try:
        for i in range(size.value):
            found = own_window(children[i])
            if found:
                return found
    finally:
        if children:
            x.XFree(children)
    return None

keys = deploy.TeleopKeyboard()
callbacks = []
def callback(code):
    callbacks.append((code, time.monotonic()))
    keys.on_keycode(code)

model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="plane" size="2 2 .1"/></worldbody></mujoco>')
data = mujoco.MjData(model)
viewer, threads = deploy.launch_viewer(model, data, callback)
held = False
window = None
try:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and window is None:
        viewer.sync()
        window = own_window(x.XDefaultRootWindow(d))
        time.sleep(.05)
    assert window, "No viewer window belonging to this diagnostic PID"
    x.XSetInputFocus(d, window, 1, 0)
    x.XSync(d, 0)
    time.sleep(.1)
    focus, revert = c.c_ulong(), c.c_int()
    x.XGetInputFocus(d, c.byref(focus), c.byref(revert))
    assert focus.value == window, "Viewer did not retain focus; refusing synthetic keys"
    t.XTestFakeKeyEvent(d, 25, 1, 0)  # W keycode measured on this desktop
    held = True
    x.XSync(d, 0)
    start = time.monotonic()
    samples = []
    while time.monotonic() - start < 2:
        viewer.sync()
        samples.append((time.monotonic()-start, float(keys.velocity_command()[0])))
        time.sleep(.02)
    x.XSetInputFocus(d, x.XDefaultRootWindow(d), 1, 0)
    x.XSync(d, 0)
    unfocused = keys.velocity_command().tolist()
    x.XSetInputFocus(d, window, 1, 0)
    x.XSync(d, 0)
    focused_again = keys.velocity_command().tolist()
    t.XTestFakeKeyEvent(d, 25, 0, 0)
    held = False
    x.XSync(d, 0)
    immediate_release = keys.velocity_command().tolist()
    deadline = time.monotonic() + deploy.TELEOP_KEY_TIMEOUT + .15
    while time.monotonic() < deadline:
        viewer.sync()
        time.sleep(.02)
    stopped = keys.velocity_command().tolist()
    settled = [v for seconds,v in samples if seconds >= .65]
    result = {"w_callbacks": sum(code == 87 for code,_ in callbacks), "held_seconds": 2,
              "moving_samples_after_initial_repeat_delay": sum(v > .6 for v in settled),
              "samples_after_initial_repeat_delay": len(settled), "released_command": stopped,
              "unfocused_command": unfocused, "focused_again_command": focused_again,
              "immediate_release_command": immediate_release}
    print(json.dumps(result), flush=True)
    assert result["w_callbacks"] >= 1
    assert settled and all(v > .6 for v in settled), "W motion did not remain active while held"
    assert stopped == [0.,0.,0.], "Motion did not stop after release timeout"
    assert immediate_release == [0.,0.,0.], "Release did not stop motion immediately"
    assert unfocused == [0.,0.,0.], "Losing focus did not stop motion"
    assert focused_again[0] > .6, "Held key did not resume in focused viewer"
finally:
    if held:
        t.XTestFakeKeyEvent(d, 25, 0, 0)
        x.XSync(d, 0)
    focus, revert = c.c_ulong(), c.c_int()
    x.XGetInputFocus(d, c.byref(focus), c.byref(revert))
    if window and focus.value == window:
        x.XSetInputFocus(d, old_focus, old_revert, 0)
        x.XSync(d, 0)
    deploy.close_viewer(viewer, threads)
    keys.close()
    x.XCloseDisplay(d)
