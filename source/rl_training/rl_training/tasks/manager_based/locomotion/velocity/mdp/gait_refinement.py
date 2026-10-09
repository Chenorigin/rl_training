# Copyright (c) 2026 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause

"""Compatibility imports for saved stair-teacher configuration paths.

New code imports stair_teacher directly. All implementations live there; this
module only preserves historical YAML/pickle callable paths.
"""

from .stair_teacher import (
    TUNING,
    pure_turn,
    turn_tracking_quality,
    bounded_air_score,
    fold_excess,
    turn_size_excess,
    _context,
    turn_swing_size_cost,
    descent_rear_fold_cost,
    ascent_front_fold_cost,
    compact_turn_air_time,
    compact_turn_status,
    gait_quality_metrics,
)

__all__ = [
    "TUNING",
    "pure_turn",
    "turn_tracking_quality",
    "bounded_air_score",
    "fold_excess",
    "turn_size_excess",
    "_context",
    "turn_swing_size_cost",
    "descent_rear_fold_cost",
    "ascent_front_fold_cost",
    "compact_turn_air_time",
    "compact_turn_status",
    "gait_quality_metrics",
]
