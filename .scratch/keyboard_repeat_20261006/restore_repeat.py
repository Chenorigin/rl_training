"""Restore per-key repeat for mapped, non-modifier X11 keys only."""
import ctypes as c
import re
import subprocess

modifiers = subprocess.check_output(["xmodmap", "-pm"], text=True)
exclude = {int(value, 16) for value in re.findall(r"\(0x([0-9a-f]+)\)", modifiers)}
mapping = subprocess.check_output(["xmodmap", "-pke"], text=True)
keycodes = []
for line in mapping.splitlines():
    match = re.match(r"keycode\s+(\d+)\s+=\s*(.*)", line)
    if match and match[2].strip() and match[2].strip() != "NoSymbol":
        code = int(match[1])
        if code not in exclude:
            keycodes.append(code)

class Controls(c.Structure):
    _fields_ = [(name, c.c_int) for name in (
        "key_click_percent", "bell_percent", "bell_pitch", "bell_duration",
        "led", "led_mode", "key", "auto_repeat_mode")]

x = c.CDLL("libX11.so.6")
x.XOpenDisplay.argtypes = [c.c_char_p]
x.XOpenDisplay.restype = c.c_void_p
x.XChangeKeyboardControl.argtypes = [c.c_void_p, c.c_ulong, c.POINTER(Controls)]
x.XAutoRepeatOn.argtypes = [c.c_void_p]
x.XSync.argtypes = [c.c_void_p, c.c_int]
x.XCloseDisplay.argtypes = [c.c_void_p]
display = x.XOpenDisplay(None)
if not display:
    raise RuntimeError("No X11 display")
try:
    for code in keycodes:
        controls = Controls(key=code, auto_repeat_mode=1)
        x.XChangeKeyboardControl(display, (1 << 6) | (1 << 7), c.byref(controls))
    x.XAutoRepeatOn(display)
    x.XSync(display, 0)
finally:
    x.XCloseDisplay(display)
print(f"Enabled repeat for {len(keycodes)} mapped ordinary keys; modifier keys unchanged.")
