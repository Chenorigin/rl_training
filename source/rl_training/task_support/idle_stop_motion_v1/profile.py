"""Frozen active idle-stop-motion-v1 parameters from checkpoint 46400.

The task owns these values. Importing this module never updates an older task's
TUNING, reward table, or Isaac configuration.
"""
from copy import deepcopy
import json
from pathlib import Path

PROFILE_PATH = Path(__file__).with_suffix(".json")


def profile_settings():
    return json.loads(PROFILE_PATH.read_text())


_SETTINGS = profile_settings()
PROFILE = _SETTINGS["profile"]
VERSION = _SETTINGS["version"]
CONTROL_PROFILE = _SETTINGS["control_profile"]
BASELINE_ACTOR_SHA256 = _SETTINGS["baseline_actor_sha256"]
BASELINE_CONFIG_SHA256 = _SETTINGS["baseline_config_sha256"]
SOURCE_SHA256 = deepcopy(_SETTINGS["source_sha256"])
TUNING = deepcopy(_SETTINGS["stair_tuning"])
REWARD_WEIGHTS = deepcopy(_SETTINGS["reward_weights"])
POLICY_NOISE = deepcopy(_SETTINGS["policy_noise"])
GUIDE_SCHEDULE = deepcopy(_SETTINGS["descent_entry_guide_schedule"])
