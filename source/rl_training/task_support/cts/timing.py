"""Durations are counts of completed control frames, never float accumulation."""
import math
import torch


def required_steps(seconds, dt):
    if not math.isfinite(seconds) or not math.isfinite(dt) or seconds <= 0 or dt <= 0:
        raise ValueError('Control duration and dt must be finite and positive')
    # Remove only division roundoff at an integer, not a fraction of a frame.
    ratio = seconds / dt
    integer = round(ratio)
    if abs(ratio-integer) <= 4*math.ulp(ratio):
        ratio = integer
    return math.ceil(ratio)


def advance_steps(count, active):
    return torch.where(active, count+1, 0)
