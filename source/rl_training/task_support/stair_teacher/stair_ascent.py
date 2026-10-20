# Copyright (c) 2026 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause

"""Compatibility imports for saved stair-teacher configuration paths.

New code imports stair_teacher_env directly. All implementations live there; this
module only preserves historical YAML/pickle callable paths.
"""

from rl_training.tasks.manager_based.locomotion.velocity.mdp.stair_teacher_env import (
    TUNING,
    _new_state,
    _reset,
    _contact_on_tread,
    _visible_higher_riser,
    ascent_state,
)

__all__ = [
    "TUNING",
    "_new_state",
    "_reset",
    "_contact_on_tread",
    "_visible_higher_riser",
    "ascent_state",
]
