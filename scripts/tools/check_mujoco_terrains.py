#!/usr/bin/env python3
"""Check terrain rays, XML edits, actor ABI and bounded physical rollouts on CPU."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import tempfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "deploy/deploy_mujoco"))
import deploy_mujoco as deploy
from terrain_loader import PRESETS, TERRAIN_DIR
import mujoco
import numpy as np
import torch
from tensordict import TensorDict

# Same model as the documented deployment command, not an arbitrary discovered XML.
REFERENCE_MODEL = Path("/home/ubuntu/桌面/m20_1/M20_perception_Lidar_rl_2/M20_perception_Lidar_rl/deep_robotics_model/M20/mjcf/M20.xml")


def load(path, robot=REFERENCE_MODEL, **overrides):
    args = argparse.Namespace(terrain=None, terrain_xml=path, **overrides)
    model = deploy.load_model(robot, args)
    contract = deploy.M20Contract(model, args.terrain_config)
    data = mujoco.MjData(model)
    contract.reset(data)
    return model, contract, data


def expected_height(kind, x, y, p):
    if abs(y) > p["width"][0] / 2:
        return 0.0
    start = p.get("start_x", [0.5])[0]
    if kind == "flat":
        return 0.0
    if kind in {"ascent", "descent"}:
        h, count, depth = p["step_height"][0], int(p["step_count"][0]), p["tread_depth"][0]
        stage = math.floor((x - start) / depth)
        if kind == "ascent":
            return max(0, min(count, stage + 1)) * h
        return max(0, count - max(0, stage + 1)) * h
    if kind in {"slope_up", "slope_down"}:
        length = p["ramp_length"][0]
        slope = math.tan(math.radians(p["slope_deg"][0]))
        return np.clip(x - start if kind == "slope_up" else start + length - x, 0, length) * slope
    raise ValueError(kind)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=REFERENCE_MODEL)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--render-dir", type=Path, help="Optional offscreen images (MUJOCO_GL=osmesa)")
    args = parser.parse_args()
    torch.set_num_threads(1)
    actor = deploy.load_actor(args.checkpoint)
    results = {"model": str(args.model.resolve()), "checkpoint": str(args.checkpoint.resolve()),
               "physics_steps_per_terrain": 200, "terrains": {}, "parameter_edits": {},
               "performance_evaluation": "not_run: this is a short contract smoke test"}
    original = mujoco.MjModel.from_xml_path(str(args.model.resolve()))
    for kind, filename in {**PRESETS, "custom": "custom.xml"}.items():
        model, contract, data = load(TERRAIN_DIR / filename, args.model)
        assert (model.nq, model.nv, model.nu) == (original.nq, original.nv, original.nu)
        assert [model.joint(i).name for i in range(model.njnt)] == [original.joint(i).name for i in range(original.njnt)]
        np.testing.assert_allclose(model.body_mass, original.body_mass)
        planes = (model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE) & (model.geom_bodyid == 0)
        assert int(planes.sum()) == 1, "robot floor must be replaced, not duplicated"
        assert np.all(model.geom_group[model.geom_bodyid == 0] == 0)
        p = contract.terrain.parameters
        spawn_z = float(data.xpos[contract.base_id, 2])
        contract.reset(data, prone=True)
        assert math.isclose(data.xpos[contract.base_id, 2], spawn_z - 0.38, abs_tol=1e-8)
        contract.reset(data)
        samples = []
        if kind in {"flat", "ascent", "descent", "slope_up", "slope_down"}:
            start = p.get("start_x", [0.5])[0]
            for x in [start - 0.2, start + 0.13, start + 0.46, start + 1.07, contract.terrain.goal_x + 0.2]:
                actual = contract.ground_height(data, x, 0)
                expected = expected_height(kind, x, 0, p)
                np.testing.assert_allclose(actual, expected, atol=1e-7)
                samples.append([x, actual])
            # Check all grid points against an independent height function, including rotated XY.
            root = int(contract.root_joint.qposadr[0])
            for yaw in [0.0, 0.43]:
                data.qpos[root:root + 7] = [start + 0.127, 0.031, 1.1,
                                          math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
                mujoco.mj_forward(model, data)
                scan = contract.height_scan(data)
                xy = np.column_stack((start + 0.127 + math.cos(yaw) * contract.grid[:, 0] - math.sin(yaw) * contract.grid[:, 1],
                                      0.031 + math.sin(yaw) * contract.grid[:, 0] + math.cos(yaw) * contract.grid[:, 1]))
                heights = np.array([expected_height(kind, x, y, p) for x, y in xy])
                np.testing.assert_allclose(contract.last_scan_hits[:, :2], xy, atol=1e-7)
                np.testing.assert_allclose(contract.last_scan_hits[:, 2], heights, atol=1e-7)
                np.testing.assert_allclose(scan, np.clip(1.1 - heights - 0.5, -1, 1), atol=1e-7)
        elif kind in {"obstacles", "custom"}:
            np.testing.assert_allclose(contract.ground_height(data, 1, 0), 0.12 if kind == "obstacles" else 0.2, atol=1e-7)
        elif kind == "hurdles":
            for i in range(int(p["obstacle_count"][0])):
                np.testing.assert_allclose(contract.ground_height(data, p["start_x"][0] + i * p["spacing"][0] + 0.05, 0),
                                           p["height"][0] + i * p["height_increment"][0], atol=1e-7)
        elif kind == "rough":
            second, _, _ = load(TERRAIN_DIR / filename, args.model)
            np.testing.assert_array_equal(model.geom_pos, second.geom_pos)
        elif kind == "mixed":
            np.testing.assert_allclose(contract.ground_height(data, 0.65, 0), p["step_height"][0], atol=1e-7)
            np.testing.assert_allclose(contract.ground_height(data, 1.6, 0), p["step_height"][0] * p["step_count"][0], atol=1e-7)
        contract.reset(data)
        previous = np.zeros(16, dtype=np.float32)
        for _ in range(50):
            obs = contract.observe(data, np.array([0.5, 0, 0], dtype=np.float32), previous)
            with torch.inference_mode():
                action = actor(TensorDict({"policy": torch.from_numpy(obs)[None]}, batch_size=[1])).squeeze(0).numpy()
            assert action.shape == (16,) and np.isfinite(action).all()
            action = np.clip(action, -100, 100)
            for _ in range(4):
                contract.apply_pd(data, action)
                mujoco.mj_step(model, data)
            previous = action.astype(np.float32)
        assert np.isfinite(data.qpos).all() and math.isclose(data.time, 1.0, abs_tol=1e-9)
        summary = {"geoms": model.ngeom, "spawn_z": spawn_z, "actor_input": len(obs),
                   "ray_samples": samples, "physics_time": data.time,
                   "contacts_at_end": data.ncon, "final_xyz": data.xpos[contract.base_id].tolist()}
        if args.render_dir:
            args.render_dir.mkdir(parents=True, exist_ok=True)
            contract.reset(data)
            camera = mujoco.MjvCamera()
            camera.lookat[:] = [(contract.terrain.spawn_xy[0] + contract.terrain.goal_x) / 2, 0, 0.3]
            camera.distance = max(6, (contract.terrain.goal_x - contract.terrain.spawn_xy[0]) * 1.2)
            camera.azimuth, camera.elevation = 135, -30
            with mujoco.Renderer(model, height=480, width=640) as renderer:
                renderer.update_scene(data, camera=camera)
                # OSMesa shadow maps produce artifacts on this host; geometry/contact unchanged.
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
                from PIL import Image
                Image.fromarray(renderer.render()).save(args.render_dir / f"{kind}.png")
        results["terrains"][kind] = summary

    with tempfile.TemporaryDirectory(prefix="m20_terrain_") as temp:
        for preset, edits, x, before, after in [
            ("ascent", {"step_height": "0.25", "friction": "0.6 0.02 0.003"}, 0.65, 0.15, 0.25),
            ("slope_up", {"slope_deg": "20"}, 2.0, 1.5 * math.tan(math.radians(12)), 1.5 * math.tan(math.radians(20)))]:
            path = Path(temp) / f"{preset}.xml"
            tree = ET.parse(TERRAIN_DIR / PRESETS[preset])
            for name, value in edits.items():
                tree.find(f"./custom/numeric[@name='{name}']").set("data", value)
            tree.write(path, encoding="utf-8")
            model, contract, data = load(path, args.model)
            actual = contract.ground_height(data, x, 0)
            np.testing.assert_allclose(actual, after, atol=1e-7)
            if preset == "ascent":
                np.testing.assert_allclose(model.geom_friction[model.geom_bodyid == 0],
                    np.tile([0.6, 0.02, 0.003], (sum(model.geom_bodyid == 0), 1)))
            results["parameter_edits"][preset] = {"before": before, "after": actual}
        # Negative control: a malformed terrain must fail instead of silently loading a flat floor.
        bad = Path(temp) / "bad.xml"
        tree = ET.parse(TERRAIN_DIR / "stairs_ascent.xml")
        tree.find("./custom/numeric[@name='step_height']").set("data", "0")
        tree.write(bad, encoding="utf-8")
        try:
            load(bad, args.model)
        except ValueError:
            results["invalid_config_rejected"] = True
        else:
            raise AssertionError("Zero step height was accepted")
        _, contract, data = load(TERRAIN_DIR / "stairs_ascent.xml", args.model, stair_height=0.22, stair_count=3)
        np.testing.assert_allclose(contract.ground_height(data, 0.65, 0), 0.22, atol=1e-7)
        assert math.isclose(contract.terrain.goal_x, 1.4)
        results["legacy_stair_overrides"] = "pass"
        negative_controls = []
        for value in [float("inf"), float("nan")]:
            try:
                load(TERRAIN_DIR / "stairs_ascent.xml", args.model, stair_height=value)
            except ValueError:
                negative_controls.append(f"nonfinite_override_{value}_rejected")
            else:
                raise AssertionError("Nonfinite CLI override was accepted")
        for filename in ["stairs_descent.xml", "slope_down.xml"]:
            tree = ET.parse(TERRAIN_DIR / filename)
            tree.find("./custom/numeric[@name='platform_length']").set("data", "0.5")
            tree.write(bad, encoding="utf-8")
            try:
                load(bad, args.model)
            except ValueError:
                negative_controls.append(f"{filename}_spawn_outside_platform_rejected")
            else:
                raise AssertionError("Descending terrain spawned at the bottom silently")
        results["negative_controls"] = negative_controls

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
