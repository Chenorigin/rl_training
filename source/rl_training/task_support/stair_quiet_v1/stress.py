"""Deterministic paired stressors, not fitted M20 hardware parameters.

Every role receives the same profile, seed and actuator transfer function.
Contact dropout is indexed by physics time, independent of controller decisions.
These profiles do not modify collision compliance or acoustic acceptance limits.
"""
from collections import deque
from dataclasses import asdict, dataclass
import hashlib
import json
import math

import numpy as np


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class StressProfile:
    name: str = "nominal"
    terrain_height_bias_m: float = 0.
    contact_delay_s: float = 0.
    contact_dropout_probability: float = 0.
    actuator_delay_s: float = 0.
    actuator_time_constant_s: float = 0.
    actuator_strength: float = 1.
    schema: str = "stair_quiet_v1.paired_stress.v1"

    def document(self):
        return asdict(self)


PROFILES = {
    "nominal": StressProfile(),
    "terrain_high": StressProfile(name="terrain_high", terrain_height_bias_m=.025),
    "terrain_low": StressProfile(name="terrain_low", terrain_height_bias_m=-.025),
    "contact_delay_loss": StressProfile(name="contact_delay_loss", contact_delay_s=.015,
                                        contact_dropout_probability=.15),
    "actuator_chain": StressProfile(name="actuator_chain", actuator_delay_s=.010,
                                     actuator_time_constant_s=.015, actuator_strength=.85),
    "combined": StressProfile(name="combined", terrain_height_bias_m=.025,
                               contact_delay_s=.015, contact_dropout_probability=.15,
                               actuator_delay_s=.010, actuator_time_constant_s=.015,
                               actuator_strength=.85),
}


def load_stress(name):
    if name not in PROFILES:
        raise ValueError(f"Unknown paired stress profile: {name}")
    return PROFILES[name]


class StressChain:
    def __init__(self, profile, dt, seed):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("Stress timestep must be finite and positive")
        self.profile, self.dt = profile, dt
        self.rng = np.random.default_rng(seed)
        self.contact_steps = math.ceil(profile.contact_delay_s / dt - 1e-10)
        self.actuator_steps = math.ceil(profile.actuator_delay_s / dt - 1e-10)
        self.contacts = deque([(np.zeros(4), np.zeros(4, dtype=bool))
                               for _ in range(self.contact_steps)])
        self.efforts = deque([np.zeros(16) for _ in range(self.actuator_steps)])
        self.filtered_effort = np.zeros(16)
        self.samples = self.missing_contact_samples = 0

    def contact(self, force, valid):
        force, valid = np.asarray(force), np.asarray(valid, dtype=bool)
        if force.shape != (4,) or valid.shape != (4,) or not np.isfinite(force).all():
            raise ValueError("Invalid causal contact sample")
        self.contacts.append((force.copy(), valid.copy()))
        force, valid = self.contacts.popleft()
        valid &= self.rng.random(4) >= self.profile.contact_dropout_probability
        self.samples += 4
        self.missing_contact_samples += int(np.sum(~valid))
        return np.where(valid, force, 0.), valid

    def effort(self, nominal):
        nominal = np.asarray(nominal, dtype=np.float64)
        if nominal.shape != (16,) or not np.isfinite(nominal).all():
            raise ValueError("Invalid actuator effort")
        if (self.actuator_steps == 0 and self.profile.actuator_time_constant_s == 0
                and self.profile.actuator_strength == 1):
            return nominal.copy()
        self.efforts.append(nominal.copy())
        delayed = self.efforts.popleft()
        tau = self.profile.actuator_time_constant_s
        alpha = -math.expm1(-self.dt / tau) if tau else 1.
        self.filtered_effort += alpha * (delayed - self.filtered_effort)
        return self.profile.actuator_strength * self.filtered_effort.copy()

    def summary(self):
        return {
            "profile": self.profile.document(),
            "profile_sha256": canonical_hash(self.profile.document()),
            "contact_extra_delay_steps": self.contact_steps,
            "contact_inherent_age_s": self.dt,
            "contact_total_age_s": (1 + self.contact_steps) * self.dt,
            "actuator_extra_delay_steps": self.actuator_steps,
            "contact_samples": self.samples,
            "missing_contact_samples": self.missing_contact_samples,
            "parameters_calibrated_on_hardware": False,
        }
