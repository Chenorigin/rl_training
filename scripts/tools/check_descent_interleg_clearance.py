#!/usr/bin/env python3
"""CPU checks for the production descent-entry interleg clearance reward."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import ast
import json

import torch

from check_gait_safety import load
from check_stair_rewards import Entity, MDP, Scene


CONFIG = MDP.parent / "config/wheeled/deeprobotics_m20/stair_teacher_env_cfg.py"


def synthetic_positions():
    # Body order: FL/FR hip-y, FL/FR knee, FL/FR wheel, HL/HR wheel.
    return torch.tensor([[
        [-0.30, 0.20, 0.50], [-0.30, -0.20, 0.50],
        [-0.30, 0.20, 0.25], [-0.30, -0.20, 0.25],
        [-0.30, 0.20, 0.00], [-0.30, -0.20, 0.00],
        [-0.30, 0.20, 0.40], [-0.75, -0.20, 0.20],
    ]], dtype=torch.float32)


def configured_clearance_names():
    tree = ast.parse(CONFIG.read_text())
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
              and node.name == "_interleg_entities")
    names = []
    for node in ast.walk(fn):
        if isinstance(node, ast.keyword) and node.arg == "body_names":
            names = ast.literal_eval(node.value)
    return names


def check(ns):
    out = {}
    positions = synthetic_positions()
    clearance = ns["descent_interleg_clearance"](positions)
    out["synthetic_clearance_m"] = clearance.tolist()
    assert clearance[0, 0] < 0.0, clearance
    assert clearance[0, 1] > 0.0, clearance

    safe = positions.clone()
    safe[:, 6, :] = torch.tensor([-0.75, 0.20, 0.20])
    safe_clearance = ns["descent_interleg_clearance"](safe)
    out["safe_clearance_m"] = safe_clearance.tolist()
    assert (safe_clearance > 0.0).all(), safe_clearance
    warning = safe.clone()
    warning[:, 6, :] = torch.tensor([-0.49, 0.20, 0.00])
    warning_clearance = ns["descent_interleg_clearance"](warning)
    warning_cost = ns["descent_forward_excess"](
        (ns["TUNING"]["interleg_clearance_margin"] - warning_clearance).clamp_min(0.0)
        / ns["TUNING"]["interleg_clearance_scale"]
    )
    out["advance_warning_clearance_m"] = warning_clearance.tolist()
    out["advance_warning_cost"] = warning_cost.tolist()
    assert 0.0 < warning_clearance[0, 0] < ns["TUNING"]["interleg_clearance_margin"]
    assert warning_cost[0, 0] > 0.0
    try:
        ns["descent_interleg_clearance"](torch.zeros(1, 7, 3))
    except ValueError:
        out["invalid_shape_fails_closed"] = True
    else:
        raise AssertionError("invalid body order/shape was accepted")

    env = SimpleNamespace(
        num_envs=1, device="cpu", episode_length_buf=torch.tensor([2]),
        scene=Scene(robot=SimpleNamespace(data=SimpleNamespace(body_pos_w=positions))),
        _m20_descent_entry_state={"active": torch.tensor([True])},
    )
    ns["_context"] = lambda *args, **kwargs: {"down": torch.tensor([True])}
    gait = dict(
        asset_cfg=Entity("robot"), hip_cfg=Entity("robot"), knee_cfg=Entity("robot"),
        sensor_cfg=Entity("height_scanner"), contact_sensor_cfg=Entity("contact_forces"),
        clearance_cfg=Entity("robot", body_ids=list(range(8))),
    )
    cost = ns["stair_descent_interleg_clearance_cost"](env, **gait)
    out["active_collision_cost"] = cost.tolist()
    assert cost.item() > 0.0
    assert env._m20_interleg_clearance_stats["samples"].tolist() == [[1.0, 1.0]]
    assert env._m20_interleg_clearance_stats["collision"].tolist() == [[1.0, 0.0]]

    ns["_context"] = lambda *args, **kwargs: {"down": torch.tensor([False])}
    inactive = ns["stair_descent_interleg_clearance_cost"](env, **gait)
    out["outside_descent_cost"] = inactive.tolist()
    assert inactive.item() == 0.0
    ns["_context"] = lambda *args, **kwargs: {"down": torch.tensor([True])}
    env._m20_descent_entry_state["active"].fill_(False)
    after_entry = ns["stair_descent_interleg_clearance_cost"](env, **gait)
    out["descent_after_entry_cost"] = after_entry.tolist()
    assert after_entry.item() == 0.0
    tag_params = {key: value for key, value in gait.items() if key != "clearance_cfg"}
    assert ns["stair_descent_entry_mask"](env, **tag_params).item() == 0.0
    env._m20_descent_entry_state["active"].fill_(True)
    assert ns["stair_descent_entry_mask"](env, **tag_params).item() == 1.0
    ns["_context"] = lambda *args, **kwargs: {"down": torch.tensor([False])}
    assert ns["stair_descent_entry_mask"](env, **tag_params).item() == 0.0
    out["entry_training_tag_matches_reward_gate"] = True

    expected = [
        "fl_hipy", "fr_hipy", "fl_knee", "fr_knee",
        "fl_wheel", "fr_wheel", "hl_wheel", "hr_wheel",
    ]
    actual = configured_clearance_names()
    out["configured_body_order"] = actual
    assert actual == expected, actual
    config_text = CONFIG.read_text()
    assert "stair_descent_interleg_clearance_cost = RewTerm(" in config_text
    assert "func=stair_teacher.stair_descent_interleg_clearance_cost" in config_text
    out["reward_wired"] = True
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = check(load(MDP / "stair_teacher.py"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
