#!/usr/bin/env python3
"""Read-only physical rollout probe for the first descent-entry transfer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import torch
from tensordict import TensorDict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "deploy/deploy_mujoco"))
import deploy_mujoco as deploy
from gait_metrics import GaitMetrics


def body_name(model, body_id):
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(body_id)) or ""


def collision_geoms(model, prefixes):
    result = []
    for geom_id in range(model.ngeom):
        name = body_name(model, model.geom_bodyid[geom_id])
        if name.startswith(prefixes) and model.geom_contype[geom_id] and model.geom_conaffinity[geom_id]:
            result.append(geom_id)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.set_num_threads(1)
    actor = deploy.load_actor(args.checkpoint)
    cfg = argparse.Namespace(terrain="descent", terrain_xml=None, stair_height=0.15,
                             tread_depth=0.30, stair_count=5)
    model = deploy.load_model(args.model, cfg)
    contract = deploy.M20Contract(model, cfg.terrain_config)
    data = mujoco.MjData(model)
    contract.reset(data)
    metrics = GaitMetrics(contract, min_extension=0.30)
    command = np.array([0.5, 0.0, 0.0], dtype=np.float32)
    previous = np.zeros(16, dtype=np.float32)

    left_front = collision_geoms(model, ("fl_",))
    left_rear = collision_geoms(model, ("hl_",))
    right_front = collision_geoms(model, ("fr_",))
    right_rear = collision_geoms(model, ("hr_",))
    pairs = {
        "left": [(a, b) for a in left_front for b in left_rear],
        "right": [(a, b) for a in right_front for b in right_rear],
    }
    records = []
    self_contacts = []
    for _ in range(750):
        obs = contract.observe(data, command, previous)
        with torch.inference_mode():
            action = actor(TensorDict({"policy": torch.from_numpy(obs)[None]}, batch_size=[1])).squeeze(0).numpy()
        action = np.clip(action, -100, 100)
        for _ in range(4):
            contract.apply_pd(data, action)
            mujoco.mj_step(model, data)
        metrics.update(data, command)
        sample = metrics.samples[-1]
        force = np.asarray(sample["forces"])
        loaded = (force[:, 2] > 5) & (force[:, 2] > 0.5 * np.linalg.norm(force, axis=1))
        wheels = data.xpos[contract.wheel_body_ids].copy()
        hips = data.xpos[metrics.hips].copy()
        rotation = data.xmat[contract.base_id].reshape(3, 3)
        local = (rotation.T @ (wheels - hips).T).T
        # Root-yaw forward projected in world; gravity-aligned forward angle.
        forward_world = rotation[:, 0].copy()
        forward_world[2] = 0
        forward_world /= max(np.linalg.norm(forward_world), 1e-12)
        gravity_x = (wheels - hips) @ forward_world
        b = np.arctan2(local[:, 0], np.maximum(-local[:, 2], 1e-9))
        g = np.arctan2(gravity_x, np.maximum(-(wheels - hips)[:, 2], 1e-9))
        clearance = {}
        fromto = np.zeros(6)
        for side, side_pairs in pairs.items():
            values = [(mujoco.mj_geomDistance(model, data, a, c, 1.0, fromto), a, c)
                      for a, c in side_pairs]
            clearance[side] = min(values, key=lambda item: item[0])
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            first = body_name(model, model.geom_bodyid[contact.geom1])
            second = body_name(model, model.geom_bodyid[contact.geom2])
            names = (first, second)
            front_rear = ((first.startswith(("fl_", "fr_")) and second.startswith(("hl_", "hr_"))) or
                          (second.startswith(("fl_", "fr_")) and first.startswith(("hl_", "hr_"))))
            if front_rear:
                self_contacts.append({"time": float(data.time), "bodies": names,
                                      "distance": float(contact.dist)})
        records.append({
            "time": float(data.time), "ground": np.asarray(sample["wheel_ground"]).tolist(),
            "loaded": loaded.tolist(), "rear_x": local[2:, 0].tolist(),
            "rear_b_deg": np.rad2deg(b[2:]).tolist(), "rear_g_deg": np.rad2deg(g[2:]).tolist(),
            "left_clearance_m": float(clearance["left"][0]),
            "left_pair": [body_name(model, model.geom_bodyid[clearance["left"][1]]),
                          body_name(model, model.geom_bodyid[clearance["left"][2]])],
            "right_clearance_m": float(clearance["right"][0]),
            "right_pair": [body_name(model, model.geom_bodyid[clearance["right"][1]]),
                           body_name(model, model.geom_bodyid[clearance["right"][2]])],
        })
        previous = action.astype(np.float32)

    ground = np.asarray([r["ground"] for r in records])
    loaded = np.asarray([r["loaded"] for r in records])
    top = float(np.max(ground[0]))
    starts = np.flatnonzero((ground[:, :2] < top - 0.04).any(axis=1))
    if not len(starts):
        raise RuntimeError("Front wheels never established a lower-tread ground height")
    start = int(starts[0])
    settle = None
    for index in range(start, len(records) - 1):
        lower = (ground[index:index + 2, 2:] < top - 0.04).all()
        supported = loaded[index:index + 2, 2:].all()
        if lower and supported:
            settle = index + 1
            break
    if settle is None:
        settle = len(records) - 1
    chosen = records[start:settle + 1]

    summary = {"checkpoint": str(args.checkpoint.resolve()), "top_height_m": top,
               "entry_start_s": records[start]["time"], "rear_settled_below_s": records[settle]["time"],
               "entry_sample_count": len(chosen), "self_contact_count": len(self_contacts),
               "self_contacts": self_contacts}
    for leg_index, leg_name in enumerate(("hl", "hr")):
        values = np.asarray([[r["rear_x"][leg_index], r["rear_b_deg"][leg_index],
                              r["rear_g_deg"][leg_index], float(r["loaded"][leg_index + 2])]
                             for r in chosen])
        for phase, mask in (("all", np.ones(len(values), dtype=bool)),
                            ("loaded", values[:, 3].astype(bool)),
                            ("swing", ~values[:, 3].astype(bool))):
            if not mask.any():
                continue
            selected = values[mask]
            summary[f"{leg_name}_{phase}"] = {
                "samples": int(mask.sum()), "x_mean_m": float(selected[:, 0].mean()),
                "x_p95_m": float(np.percentile(selected[:, 0], 95)), "x_max_m": float(selected[:, 0].max()),
                "b_p95_deg": float(np.percentile(selected[:, 1], 95)), "b_max_deg": float(selected[:, 1].max()),
                "g_p95_deg": float(np.percentile(selected[:, 2], 95)), "g_max_deg": float(selected[:, 2].max()),
            }
    for side in ("left", "right"):
        distances = np.asarray([r[f"{side}_clearance_m"] for r in chosen])
        index = int(distances.argmin())
        summary[f"{side}_front_rear_clearance"] = {
            "min_m": float(distances[index]), "time_s": chosen[index]["time"],
            "body_pair": chosen[index][f"{side}_pair"],
        }
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
