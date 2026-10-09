"""Hold one synthetic X11 key in a private window; never synthesize repeats."""
import ctypes as c
import json
import time
import tkinter as tk

x11 = c.CDLL("libX11.so.6")
xtst = c.CDLL("libXtst.so.6")
x11.XOpenDisplay.argtypes = [c.c_char_p]
x11.XOpenDisplay.restype = c.c_void_p
x11.XGetInputFocus.argtypes = [c.c_void_p, c.POINTER(c.c_ulong), c.POINTER(c.c_int)]
x11.XSetInputFocus.argtypes = [c.c_void_p, c.c_ulong, c.c_int, c.c_ulong]
x11.XStringToKeysym.argtypes = [c.c_char_p]
x11.XStringToKeysym.restype = c.c_ulong
x11.XKeysymToKeycode.argtypes = [c.c_void_p, c.c_ulong]
x11.XKeysymToKeycode.restype = c.c_uint
x11.XSync.argtypes = [c.c_void_p, c.c_int]
x11.XCloseDisplay.argtypes = [c.c_void_p]
x11.XSelectInput.argtypes = [c.c_void_p, c.c_ulong, c.c_long]
x11.XPending.argtypes = [c.c_void_p]
x11.XNextEvent.argtypes = [c.c_void_p, c.c_void_p]
xtst.XTestFakeKeyEvent.argtypes = [c.c_void_p, c.c_uint, c.c_int, c.c_ulong]
display = x11.XOpenDisplay(None)
if not display:
    raise RuntimeError("No X11 display")
old_focus, old_revert = c.c_ulong(), c.c_int()
x11.XGetInputFocus(display, c.byref(old_focus), c.byref(old_revert))
root = tk.Tk()
root.overrideredirect(True)
root.title("Keyboard repeat diagnostic (auto closes)")
root.geometry("420x90+50+50")
tk.Label(root, text="Testing W / Left auto-repeat; closes automatically.").pack()
events = []
root.update()
deadline = time.monotonic() + 3.0
while not root.winfo_viewable() and time.monotonic() < deadline:
    root.update()
    time.sleep(0.005)
if not root.winfo_viewable():
    raise RuntimeError("Diagnostic window not mapped")
x11.XSetInputFocus(display, root.winfo_id(), 1, 0)
x11.XSync(display, 0)
root.update()
focus, revert = c.c_ulong(), c.c_int()
x11.XGetInputFocus(display, c.byref(focus), c.byref(revert))
if focus.value != root.winfo_id():
    root.destroy()
    x11.XCloseDisplay(display)
    raise RuntimeError("Diagnostic does not own input focus; refusing synthetic keys")
x11.XSelectInput(display, root.winfo_id(), 3)  # KeyPressMask | KeyReleaseMask
x11.XSync(display, 0)
held = None
try:
    results = {}
    for name in ("w", "Left"):
        events.clear()
        x11.XGetInputFocus(display, c.byref(focus), c.byref(revert))
        if focus.value != root.winfo_id():
            raise RuntimeError("Input focus changed; refusing synthetic keys")
        held = x11.XKeysymToKeycode(display, x11.XStringToKeysym(name.encode()))
        xtst.XTestFakeKeyEvent(display, held, 1, 0)
        x11.XSync(display, 0)
        deadline = time.monotonic() + 1.1
        while time.monotonic() < deadline:
            root.update()
            while x11.XPending(display):
                event = (c.c_long * 24)()
                x11.XNextEvent(display, event)
                if c.cast(event, c.POINTER(c.c_int))[0] == 2:
                    events.append((name, time.monotonic()))
            time.sleep(0.005)
        xtst.XTestFakeKeyEvent(display, held, 0, 0)
        x11.XSync(display, 0)
        held = None
        root.update()
        results[name] = {"press_events": len(events), "held_seconds": 1.1}
    print(json.dumps(results), flush=True)
finally:
    if held is not None:
        xtst.XTestFakeKeyEvent(display, held, 0, 0)
        x11.XSync(display, 0)
    current, revert = c.c_ulong(), c.c_int()
    x11.XGetInputFocus(display, c.byref(current), c.byref(revert))
    # Restore focus only if it still belongs to this diagnostic window.
    if current.value == root.winfo_id():
        x11.XSetInputFocus(display, old_focus, old_revert, 0)
    root.destroy()
    x11.XCloseDisplay(display)
