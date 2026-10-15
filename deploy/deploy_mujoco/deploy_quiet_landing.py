"""Independent quiet V1 fixed-command MuJoCo viewer and 200Hz signal export.

Reuses only the baseline actor/control contract and terrain loader. This entry
does not change the legacy teacher teleoperation or deployment behavior.
"""
import hashlib
import json
import time
import argparse
from pathlib import Path

import mujoco
import numpy as np
import torch
from tensordict import TensorDict

import deploy_mujoco as baseline
from quiet_landing_metrics import QuietMujocoMetrics, LiveSignals


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=baseline.REPO_ROOT / "logs/rsl_rl/deeprobotics_m20_stair_teacher/2026-10-09_01-55-35_gait_fix_descent_entry/model_214700.pt")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--terrain", choices=tuple(baseline.PRESETS), default=None)
    group.add_argument("--terrain-xml", type=Path)
    parser.add_argument("--command-x", type=float, default=.5)
    parser.add_argument("--command-y", type=float, default=0.)
    parser.add_argument("--command-yaw", type=float, default=0.)
    for name in ("stair-height", "tread-depth", "stair-start"):
        parser.add_argument("--" + name, type=float)
    parser.add_argument("--stair-count", type=int)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--stop-on-fall", action="store_true")
    parser.add_argument("--viewer", action="store_true", help="Fixed command policy with quiet overlays")
    parser.add_argument("--live-plot", action="store_true", help="Four-wheel live signal window (requires a desktop)")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--snapshot", type=Path, help="Save an offscreen image of the quiet markers at final frame")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    return args


def main(args):
    torch.set_num_threads(1)
    actor = baseline.load_actor(args.checkpoint)
    model = baseline.load_model(args.model, args)
    contract = baseline.M20Contract(model, args.terrain_config)
    data = mujoco.MjData(model)
    contract.reset(data)
    contract.height_scan(data)
    metrics = QuietMujocoMetrics(model, contract)
    command = np.array([args.command_x, args.command_y, args.command_yaw], dtype=np.float32)
    metrics.command_xy = command[:2]
    previous_action = np.zeros(baseline.ACTION_DIM, dtype=np.float32)
    viewer, threads = (None, [])
    if args.viewer:
        viewer, threads = baseline.launch_viewer(model, data)
        with viewer.lock():
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = contract.base_id
            viewer.cam.distance = 3.2
            viewer.cam.elevation = -18
    fell = False
    live = LiveSignals() if args.live_plot else None
    try:
        for step in range(args.steps):
            started = time.perf_counter()
            observation = contract.observe(data, command, previous_action)
            with torch.inference_mode():
                action = actor(TensorDict({"policy": torch.from_numpy(observation)[None]}, batch_size=[1]))[0].numpy()
            if action.shape != (16,) or not np.isfinite(action).all():
                raise RuntimeError("Invalid quiet actor action")
            action = np.clip(action, -100., 100.)
            metrics.tracker.begin_step()
            xy = data.xpos[contract.base_id, :2].copy()
            for _ in range(round(baseline.POLICY_DT / model.opt.timestep)):
                contract.apply_pd(data, action)
                # step1 refreshes kinematics/contacts with the current qpos/qvel;
                # step2 solves forces and integrates. Snapshot is before solve.
                mujoco.mj_step1(model, data)
                snapshot = metrics.pre_step(data)
                mujoco.mj_step2(model, data)
                metrics.post_step(data, snapshot, record=args.output is not None or live is not None)
                if live is not None and args.output is None:
                    for values in metrics.records.values():
                        if len(values) > 1000:
                            del values[:-1000]
                    for values in metrics.extended_records.values():
                        if len(values) > 1000:
                            del values[:-1000]
            metrics.tracker.distance[0] += float(np.linalg.norm(data.xpos[contract.base_id, :2] - xy))
            previous_action = action.astype(np.float32)
            position = data.xpos[contract.base_id]
            local_height = contract.ground_height(data, float(position[0]), float(position[1]))
            gravity_z = float(data.xmat[contract.base_id].reshape(3, 3)[:, 2] @ np.array([0., 0., -1.]))
            fell |= bool(position[2] - local_height < .22 or gravity_z > -.35)
            if viewer:
                baseline.draw_height_scan(viewer, contract, True)
                metrics.draw(viewer, data)
                viewer.sync()
                if not viewer.is_running():
                    break
                time.sleep(max(0., baseline.POLICY_DT - (time.perf_counter() - started)))
            if live is not None and step % 5 == 0:
                live.update(metrics)
            if args.stop_on_fall and fell:
                break
    finally:
        baseline.close_viewer(viewer, threads)
        if live is not None:
            live.close()
    result = {"checkpoint": str(args.checkpoint.resolve()),
              "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              "model": str(args.model.resolve()), "command": command.tolist(),
              "terrain": args.terrain_config.summary(), "fell": fell,
              "duration_s": float(data.time), "quiet": metrics.summary(),
              "proxy": "horizontal-tread pre-impact wheel-centre speed and specific normal energy; not sound"}
    result.update(contract.crossing_summary(data, fell))
    if args.snapshot:
        from contextlib import nullcontext
        from types import SimpleNamespace
        import matplotlib.pyplot as plt
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        camera.trackbodyid = contract.base_id
        camera.distance, camera.elevation = 2.5, -20
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, 960)
        model.vis.global_.offheight = max(model.vis.global_.offheight, 640)
        with mujoco.Renderer(model, height=640, width=960) as renderer:
            renderer.update_scene(data, camera=camera)
            metrics.draw(SimpleNamespace(lock=nullcontext, user_scn=renderer.scene), data)
            args.snapshot.parent.mkdir(parents=True, exist_ok=True)
            fig, ax = plt.subplots(figsize=(9.6, 6.4), dpi=100)
            ax.imshow(renderer.render())
            ax.text(.02, .98, metrics.table_text(), va="top", color="white", fontsize=10,
                    fontfamily="monospace", transform=ax.transAxes,
                    bbox={"facecolor": "black", "alpha": .65, "pad": 6})
            ax.axis("off")
            fig.savefig(args.snapshot, bbox_inches="tight", pad_inches=0)
            plt.close(fig)
    if args.output:
        metrics.save(args.output, result)
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    main(parse_args())
