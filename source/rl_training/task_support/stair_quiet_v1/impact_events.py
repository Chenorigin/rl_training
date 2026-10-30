"""Extract explicitly model-dependent impact excitation from rigid-body contacts.

These are *inputs to a hypothetical sound model*, not audio measurements.
The first fixed window after contact onset or a sharp reload is integrated.
Continuous support outside that window is excluded; motors, rolling noise and
unresolved contact microstructure are not represented. Raw forces must be kept.
"""
from dataclasses import asdict, dataclass
import math

import numpy as np


@dataclass(frozen=True)
class ExtractionSettings:
    contact_on_N: float = 5.0
    contact_off_N: float = 2.0
    reload_jump_N: float = 50.0
    window_s: float = 0.04
    reset_grace_s: float = 0.1

    def validate(self):
        if any(not math.isfinite(x) or x < 0 for x in asdict(self).values()):
            raise ValueError("Invalid event extraction setting")
        if not 0 <= self.contact_off_N < self.contact_on_N:
            raise ValueError("Contact hysteresis requires off < on")
        if self.window_s <= 0 or self.reload_jump_N <= 0:
            raise ValueError("Impact window and reload threshold must be positive")


class ImpactEvents:
    """One independent history for each (wheel, static geometry, normal axis).

    Contacts on a tread and riser can coexist. Multiple solver points on one
    surface are aggregated before calling update. Each sample describes the
    force acting over [time_s, time_s + dt), with pre-integration position.
    """

    def __init__(self, dt, settings=None):
        self.settings = settings or ExtractionSettings()
        self.settings.validate()
        if not math.isfinite(dt) or dt <= 0 or dt > self.settings.window_s:
            raise ValueError("Invalid contact sampling timestep")
        self.dt = dt
        self.histories = {}
        self.events = []
        self.next_id = 0
        self.last_time = None

    def _flush(self, state):
        event = state.pop("event", None)
        if event is not None and event["normal_impulse_N_s"] > 0:
            event["position_m"] = (
                np.asarray(event.pop("position_impulse_sum")) / event["normal_impulse_N_s"]
            ).tolist()
            self.events.append(event)

    def update(self, time_s, contacts):
        if not math.isfinite(time_s) or time_s < 0:
            raise ValueError("Invalid sample time")
        if self.last_time is not None and not math.isclose(
                time_s - self.last_time, self.dt, abs_tol=1e-8, rel_tol=1e-6):
            raise ValueError("Contact samples must be contiguous with fixed dt")
        self.last_time = time_s
        s = self.settings
        for key in sorted(self.histories.keys() | contacts.keys()):
            row = contacts.get(key)
            force = float(row["normal_force_N"]) if row is not None else 0.0
            if not math.isfinite(force) or force < 0:
                raise ValueError("Contact normal force must be finite and nonnegative")
            if row is not None:
                point = np.asarray(row["position_m"], dtype=float)
                if point.shape != (3,) or not np.isfinite(point).all():
                    raise ValueError("Contact point must be a finite 3-vector")
            state = self.histories.setdefault(key, {"contact": False, "previous": 0.0})
            active = force > s.contact_off_N if state["contact"] else force >= s.contact_on_N
            if "event" in state and time_s >= state["event"]["time_s"] + s.window_s - 1e-9:
                self._flush(state)
            onset = active and not state["contact"]
            reload = active and force - state["previous"] >= s.reload_jump_N
            if time_s >= s.reset_grace_s and (onset or reload) and "event" not in state:
                self.next_id += 1
                state["event"] = {
                    "contact_id": f"impact_{self.next_id}",
                    "time_s": float(time_s), "wheel": row["wheel"], "surface": row["surface"],
                    "normal_impulse_N_s": 0.0, "position_impulse_sum": np.zeros(3),
                    "peak_normal_force_N": 0.0, "integrated_duration_s": 0.0,
                    "trigger": "contact_onset" if onset else "rapid_reload",
                }
            if "event" in state:
                event = state["event"]
                used_dt = min(self.dt, max(0., event["time_s"] + s.window_s - time_s))
                impulse = force * used_dt
                event["normal_impulse_N_s"] += impulse
                if row is not None:
                    event["position_impulse_sum"] += point * impulse
                event["peak_normal_force_N"] = max(event["peak_normal_force_N"], force)
                event["integrated_duration_s"] += used_dt
                if not active:
                    self._flush(state)
            state["contact"], state["previous"] = active, force

    def finish(self):
        for state in self.histories.values():
            self._flush(state)
        return sorted(self.events, key=lambda e: (e["time_s"], e["contact_id"]))


def select_window(events, start_s, end_s):
    """Select complete excitations in the *same predeclared traversal window*.

    An event that straddles a boundary cannot be exactly clipped from its
    integrated impulse; report that loss instead of silently pretending it can.
    """
    if not 0 <= start_s < end_s:
        raise ValueError("Invalid measurement window")
    selected, straddled = [], []
    for event in events:
        first = event["time_s"]
        last = first + event["integrated_duration_s"]
        if first >= start_s - 1e-9 and last <= end_s + 1e-9:
            selected.append({**event, "time_s": max(0., first - start_s)})
        elif first < end_s and last > start_s:
            straddled.append(event["contact_id"])
    return selected, straddled
