"""Versioned actor ABI. History, when enabled, follows the original 618 fields."""
from copy import deepcopy

from m20_student_v0.contract import (PROPRIO_DIM, HEIGHT_DIM, ACTION_DIM,
                                    GRID_ROWS, GRID_COLS, GRID_RESOLUTION,
                                    HEIGHT_OFFSET, HEIGHT_CLIP)
from m20_student_v0.contract import observation_contract as v0_contract


def student_dim(history_steps=1):
    if not isinstance(history_steps, int) or not 1 <= history_steps <= 16:
        raise ValueError("history_steps must be an integer in [1,16]")
    return 618 + PROPRIO_DIM * (history_steps - 1)


def observation_contract(history_steps=1):
    contract = deepcopy(v0_contract())
    contract.update(version="m20_student_v1", student_observation_dim=student_dim(history_steps),
                    history_steps=history_steps, policy_dt_s=0.02,
                    previous_action="previous issued action, before actuator delay",
                    confidence="m * exp(-cell_age/tau) / (1 + effective_variance/ref_std**2)",
                    effective_variance="height + correlated height + relative pose + roll/pitch + cell-age growth",
                    map_state="rolling gravity/yaw grid; observed cells retain individual capture timestamps")
    if history_steps > 1:
        contract["channels"].append("past_proprio")
        contract["slices"]["past_proprio"] = [618, student_dim(history_steps)]
    contract["history"] = "past noisy proprio only; oldest to newest; zero padded on reset; current proprio first"
    return contract


def validate_contract(actual):
    expected = observation_contract(actual.get("history_steps", 1))
    for key, value in expected.items():
        if actual.get(key) != value:
            raise ValueError(f"V1 observation contract mismatch: {key}")
