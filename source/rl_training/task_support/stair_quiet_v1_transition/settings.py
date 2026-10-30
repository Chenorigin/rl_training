"""Candidate training scales; not calibrated robot/wood safety thresholds."""
from dataclasses import dataclass
import math
from task_support.stair_quiet_v1_safe.settings import SafeSettings
from task_support.stair_quiet_v1_width.settings import GEOMETRIES as WIDTH_GEOMETRIES, PROFILES, Geometry

GEOMETRIES = (*WIDTH_GEOMETRIES, Geometry("deployment_17_30_closed", .17, .30, closed_riser=True))
PROTOCOLS = ("direct", "pause", "ramp", "direct_alt")


def cases():
    return tuple((g, p) for g in GEOMETRIES for p in PROFILES)


@dataclass
class TransitionSettings(SafeSettings):
    approach_x_range: tuple = (-.40, .25)
    reset_y_range: float = .06
    reset_yaw_range: float = .08
    entry_trigger_range_m: tuple = (.35, .55)
    entry_pause_range_s: tuple = (.4, 1.2)
    entry_ramp_range_s: tuple = (.3, .8)
    riser_force_reference_N: float = 200.
    riser_energy_reference_J: float = 5.
    riser_clearance_m: float = .02
    riser_work_weight: float = -.3
    riser_descent_weight: float = -.3

    def validate(self):
        super().validate()
        for name in ("approach_x_range", "entry_trigger_range_m", "entry_pause_range_s", "entry_ramp_range_s"):
            v = getattr(self, name)
            if len(v) != 2 or not all(math.isfinite(x) for x in v) or v[0] > v[1]:
                raise ValueError(f"Invalid Transition {name}")
        if (self.approach_x_range[0] < -.5 or self.approach_x_range[1] > .3
                or min(self.entry_trigger_range_m) <= .2 or min(self.entry_pause_range_s) <= 0
                or min(self.entry_ramp_range_s) <= 0):
            raise ValueError("Transition entry must start safely on the existing approach")
        for name in ("riser_force_reference_N", "riser_energy_reference_J", "riser_clearance_m"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"Invalid Transition {name}")
        if any(not math.isfinite(getattr(self, name)) or getattr(self, name) >= 0
               for name in ("riser_work_weight", "riser_descent_weight")):
            raise ValueError("Transition costs require negative finite weights")
