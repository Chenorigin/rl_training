"""Physical top-platform completion, including final landing capture."""
from dataclasses import dataclass, asdict
import math

import numpy as np


@dataclass(frozen=True)
class TraversalSettings:
    final_impact_capture_s: float = .25
    top_contact_height_tolerance_m: float = .02
    top_contact_normal_z_min: float = .9


class TopLanding:
    """Each wheel must touch the top; simultaneous four-foot stance is unnecessary."""

    def __init__(self, settings=None):
        self.settings = settings or TraversalSettings()
        if any(not math.isfinite(v) or v < 0 for v in asdict(self.settings).values()):
            raise ValueError("Invalid top landing settings")
        if self.settings.final_impact_capture_s <= 0 or not 0 < self.settings.top_contact_normal_z_min <= 1:
            raise ValueError("Invalid capture time or surface normal threshold")
        self.seen = np.zeros(4, dtype=bool)
        self.capture_started_s = None
        self.last_time = -math.inf

    def update(self, time_s, beyond_top, touched_top, *, invalid=False):
        if not math.isfinite(time_s) or time_s < self.last_time:
            raise ValueError("Top landing clock must be monotonic")
        self.last_time = time_s
        beyond, touched = np.asarray(beyond_top, dtype=bool), np.asarray(touched_top, dtype=bool)
        if beyond.shape != (4,) or touched.shape != (4,):
            raise ValueError("Four wheel conditions are required")
        self.seen = (self.seen | touched) & beyond
        if invalid or not beyond.all() or not self.seen.all():
            self.capture_started_s = None
            return False
        if self.capture_started_s is None:
            self.capture_started_s = time_s
        return time_s - self.capture_started_s >= self.settings.final_impact_capture_s - 1e-9
