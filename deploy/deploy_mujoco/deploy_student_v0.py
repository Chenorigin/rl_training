#!/usr/bin/env python3
"""Student-only M20 MuJoCo rollout with the 618-dimensional v0 contract.

CPU inference and actual mj_step physics. No teacher, optimizer or Isaac startup.
--viewer starts prone with keyboard control: Z stand, C student, W/S/A/D/Q/E
move, X lie down, R damping, H map, Esc quit. --autoplay selects fixed commands.
"""
from __future__ import annotations

import argparse
from collections import deque
from copy import copy
import json
import math
from pathlib import Path
import shutil
import time

import mujoco
import numpy as np
import torch

import deploy_mujoco as baseline
from student_v0_adapter import (StudentMapAdapter, load_student, profile_config,
                                sha256, observation_contract)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--policy-jit", type=Path)
    parser.add_argument("--model", type=Path, help="Explicit M20 MJCF path, including mesh assets")
    terrain = parser.add_mutually_exclusive_group()
    terrain.add_argument("--terrain", choices=tuple(baseline.PRESETS), default=None)
    terrain.add_argument("--terrain-xml", type=Path)
    parser.add_argument("--noise-profile", choices=("clean", "train", "lidar_fov", "realistic", "stress"), default="train")
    parser.add_argument("--profile-file", type=Path, help="Override student_v0_profiles.json")
    parser.add_argument("--noise-level", type=float)
    parser.add_argument("--command-x", type=float, default=.5)
    parser.add_argument("--command-y", type=float, default=0.)
    parser.add_argument("--command-yaw", type=float, default=0.)
    parser.add_argument("--warmup-s", type=float, default=0., help="Student controls with a zero velocity command initially")
    parser.add_argument("--track-course", action="store_true", help="Use the baseline's external heading/center-line command feedback; logged separately")
    parser.add_argument("--steps", type=int, help="Control-step limit; keyboard viewer defaults to unlimited, batch/autoplay to 1000")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--viewer", action="store_true", help="Keyboard control: Z stand, C student, W/S/A/D/Q/E move")
    parser.add_argument("--autoplay", action="store_true", help="Use fixed commands in the viewer instead of keyboard control")
    parser.add_argument("--snapshot", type=Path, help="Render the last recorded policy-input frame; headless use MUJOCO_GL=osmesa")
    parser.add_argument("--continue-after-fall", action="store_true")
    parser.add_argument("--output-dir", type=Path, help="Save summary.json and trajectory.npz; directory must be new")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--export-dir", type=Path)
    parser.add_argument("--suite", action="store_true", help="Serial clean/train/realistic evaluations on flat/ascent/descent/rough")
    parser.add_argument("--suite-profiles", nargs="+", choices=("clean", "train", "lidar_fov", "realistic", "stress"), default=["clean", "train", "realistic"])
    parser.add_argument("--suite-terrains", nargs="+", choices=tuple(baseline.PRESETS), default=["flat", "ascent", "descent", "rough"])
    parser.add_argument("--suite-seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--friction-scale", type=float, default=1., help="Scale static terrain sliding friction; a controlled physics test")
    for name in ("stair-height", "tread-depth", "stair-start"):
        parser.add_argument("--" + name, type=float)
    parser.add_argument("--stair-count", type=int)
    args = parser.parse_args(argv)
    if (args.steps is not None and args.steps < 1) or not math.isfinite(args.warmup_s) or args.warmup_s < 0:
        parser.error("steps must be positive and warmup-s finite/nonnegative")
    if not np.isfinite([args.command_x, args.command_y, args.command_yaw]).all():
        parser.error("commands must be finite")
    if not math.isfinite(args.friction_scale) or args.friction_scale <= 0:
        parser.error("friction-scale must be finite and positive")
    if args.noise_level is not None and (not math.isfinite(args.noise_level) or not 0 <= args.noise_level <= 1):
        parser.error("noise-level must be in [0,1]")
    if args.export_only:
        if args.export_dir is None or args.policy_jit is not None:
            parser.error("export-only requires checkpoint and export-dir")
    elif args.model is None:
        parser.error("model is required for MuJoCo physics")
    if args.suite and (args.viewer or args.output_dir is None or args.terrain_xml or args.terrain):
        parser.error("suite requires output-dir and does not use viewer/terrain/terrain-xml")
    if args.suite and args.snapshot:
        parser.error("snapshot applies to one rollout, not suite")
    if args.suite and any(getattr(args, name) is not None for name in baseline.STAIR_OVERRIDES):
        parser.error("suite uses each terrain XML's parameters; stair overrides apply to individual rollouts")
    if args.autoplay and not args.viewer:
        parser.error("autoplay requires viewer; headless rollouts already use fixed commands")
    if args.viewer and not args.autoplay and args.track_course:
        parser.error("keyboard mode supplies velocity commands; remove track-course, or add autoplay for automatic feedback")
    return args


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def prepare_output(path):
    if path is None:
        return
    path = path.resolve()
    existing_parent = path.parent
    while not existing_parent.exists():
        existing_parent = existing_parent.parent
    if shutil.disk_usage(existing_parent).free < 1024**3:
        raise RuntimeError("Insufficient disk space for simulation output (need 1 GiB free)")
    path.mkdir(parents=True, exist_ok=False)


def export_student(args, policy, provenance, identity):
    prepare_output(args.export_dir)
    policy_path = args.export_dir / "student_v0.pt"
    model = torch.jit.trace(policy, torch.zeros(1, 618))
    model.save(str(policy_path))
    reloaded = torch.jit.load(str(policy_path), map_location="cpu")
    inputs = torch.randn(32, 618)
    with torch.inference_mode():
        error = float((policy(inputs) - reloaded(inputs)).abs().max())
    if error > 1e-6:
        raise RuntimeError(f"Export changed student outputs: {error}")
    write_json(args.export_dir / "observation_contract.json", observation_contract())
    result = {**identity, "training_provenance": provenance, "export_sha256": sha256(policy_path),
              "export_max_action_error": error, "weights_updated": False,
              "dimensions": {"input": [1, 618], "output": [1, 16]}}
    write_json(args.export_dir / "deployment.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def draw_student_map(viewer, contract, channels, base_z):
    height, validity, confidence = channels
    hits = contract.last_scan_hits.copy()
    hits[:, 2] = base_z - height - baseline.SCAN_OFFSET
    with viewer.lock():
        scene = viewer.user_scn
        scene.ngeom = 0
        if scene.maxgeom < len(hits):
            raise RuntimeError("Viewer user scene cannot hold the student grid")
        for i, hit in enumerate(hits):
            # Red unknown / yellow low confidence / green high confidence.
            rgb = [1., .15, .15] if validity[i] == 0 else [1. - confidence[i], confidence[i], .15]
            if validity[i] == 0:
                hit[2] = base_z - baseline.SCAN_OFFSET
            mujoco.mjv_initGeom(scene.geoms[i], mujoco.mjtGeom.mjGEOM_SPHERE,
                                np.array([.015, 0., 0.]), hit + [0, 0, .025],
                                np.eye(3).ravel(), np.array([*rgb, .8], dtype=np.float32))
        scene.ngeom = len(hits)


def render_snapshot(path, model, contract, records, channels):
    from contextlib import nullcontext
    from types import SimpleNamespace
    import matplotlib.pyplot as plt
    # Snapshot of the saved physical rollout, at the exact last observation pose.
    data = mujoco.MjData(model)
    data.qpos[:] = records["qpos"][-1]
    data.qvel[:] = records["qvel"][-1]
    mujoco.mj_forward(model, data)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = contract.base_id
    camera.distance, camera.elevation, camera.azimuth = 3., -22., 135.
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, 960)
    model.vis.global_.offheight = max(model.vis.global_.offheight, 640)
    with mujoco.Renderer(model, height=640, width=960) as renderer:
        renderer.update_scene(data, camera=camera)
        # Use a separate user scene so terrain/robot render geoms are preserved.
        scene = mujoco.MjvScene(model, maxgeom=187)
        draw_student_map(SimpleNamespace(lock=nullcontext, user_scn=scene), contract, channels,
                         float(records["base_position"][-1, 2]))
        count = renderer.scene.ngeom
        for i in range(scene.ngeom):
            if count + i >= renderer.scene.maxgeom:
                raise RuntimeError("Insufficient renderer geometry capacity")
            geom = scene.geoms[i]
            mujoco.mjv_initGeom(renderer.scene.geoms[count + i], geom.type,
                                geom.size.astype(np.float64), geom.pos.astype(np.float64),
                                geom.mat.astype(np.float64).ravel(), geom.rgba)
        renderer.scene.ngeom += scene.ngeom
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.imsave(path, renderer.render())


def create_simulation(args, provenance, prone=False):
    torch.manual_seed(args.seed)
    cfg, extras = profile_config(args.noise_profile, provenance, args.profile_file, args.noise_level)
    model = baseline.load_model(args.model, args)
    terrain_ids = (model.geom_bodyid == 0) & (model.geom_group == 0) & ((model.geom_contype != 0) | (model.geom_conaffinity != 0))
    model.geom_friction[terrain_ids, 0] *= args.friction_scale
    # Override contact mixing as well: lower terrain friction must actually act.
    if args.friction_scale != 1.:
        model.geom_priority[terrain_ids] = np.maximum(model.geom_priority[terrain_ids], 1)
    contract = baseline.M20Contract(model, args.terrain_config)
    data = mujoco.MjData(model)
    contract.reset(data, prone=prone)
    adapter = StudentMapAdapter(contract, cfg, extras, args.seed)
    return model, contract, data, adapter, extras


def rollout(args, policy, provenance, identity):
    model, contract, data, adapter, extras = create_simulation(args, provenance)
    command = np.array([args.command_x, args.command_y, args.command_yaw], dtype=np.float32)
    previous_action = np.zeros(16, dtype=np.float32)
    delay = deque([previous_action.copy() for _ in range(extras.action_delay_steps)])
    substeps = round(baseline.POLICY_DT / model.opt.timestep)
    if not math.isclose(substeps * model.opt.timestep, baseline.POLICY_DT):
        raise ValueError("Physics timestep must divide the policy timestep")
    prepare_output(args.output_dir)
    keys = ("time_s", "qpos", "qvel", "observation", "action_issued", "action_applied", "base_position", "command",
            "velocity_body", "tilt_rad", "map_age_s", "lidar_visible_fraction", "blackout", "post_time_s", "fell_post_step")
    records = {key: [] for key in keys}
    viewer, threads = None, []
    fell, outside_course = False, False
    first_fall_time = None
    start_xy = data.xpos[contract.base_id, :2].copy()
    started = time.perf_counter()
    if args.viewer:
        viewer, threads = baseline.launch_viewer(model, data)
        with viewer.lock():
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = contract.base_id
            viewer.cam.distance, viewer.cam.elevation = 3., -20.
    try:
        for step in range(args.steps if args.steps is not None else 1000):
            frame_start = time.perf_counter()
            current_command = command.copy()
            if args.track_course:
                rotation = data.xmat[contract.base_id].reshape(3, 3)
                yaw = math.atan2(rotation[1, 0], rotation[0, 0])
                target = contract.terrain.spawn_yaw
                lateral = float(np.clip(-.5 * (data.xpos[contract.base_id, 1] - contract.terrain.spawn_xy[1]), -.25, .25))
                world_x = args.command_x * math.cos(target) - lateral * math.sin(target)
                world_y = args.command_x * math.sin(target) + lateral * math.cos(target)
                current_command[:2] = (math.cos(yaw) * world_x + math.sin(yaw) * world_y,
                                       -math.sin(yaw) * world_x + math.cos(yaw) * world_y)
                heading_error = math.atan2(math.sin(target - yaw), math.cos(target - yaw))
                current_command[2] = np.clip(.5 * heading_error, -.5, .5)
            if step * baseline.POLICY_DT < args.warmup_s:
                current_command[:] = 0.
            obs = adapter.observe(data, current_command, previous_action, step)
            with torch.inference_mode():
                raw_action = policy(torch.from_numpy(obs)[None])[0].numpy()
            if raw_action.shape != (16,) or not np.isfinite(raw_action).all():
                raise RuntimeError("Invalid student action")
            issued = np.clip(raw_action, -100., 100.).astype(np.float32)
            delay.append(issued.copy())
            applied = delay.popleft()
            pre_time = float(data.time)
            pre_pos = data.xpos[contract.base_id].copy()
            velocity = np.zeros(6)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, contract.base_id, velocity, 1)
            # Observation/action records share the pre-step frame. Outcomes below
            # have explicit post-step names and timestamps.
            records["time_s"].append(pre_time)
            records["qpos"].append(data.qpos.copy())
            records["qvel"].append(data.qvel.copy())
            records["observation"].append(obs)
            records["action_issued"].append(issued)
            records["action_applied"].append(applied.copy())
            records["base_position"].append(pre_pos)
            records["command"].append(current_command.copy())
            records["velocity_body"].append(np.array([*velocity[3:5], velocity[2]]))
            rotation = data.xmat[contract.base_id].reshape(3, 3)
            records["tilt_rad"].append(math.acos(float(np.clip(rotation[2, 2], -1, 1))))
            capture = int(adapter.map.capture[0])
            records["map_age_s"].append((step - capture) * baseline.POLICY_DT if capture >= 0 else -1.)
            records["lidar_visible_fraction"].append(float(adapter.last_visibility.mean()))
            records["blackout"].append(adapter.last_blackout)
            for _ in range(substeps):
                contract.apply_pd(data, applied)
                mujoco.mj_step(model, data)
            # mj_step integrates qpos; refresh kinematics before measuring post-step pose.
            mujoco.mj_forward(model, data)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise RuntimeError("Nonfinite MuJoCo state")
            previous_action = issued
            position = data.xpos[contract.base_id]
            ground = contract.ground_height(data, float(position[0]), float(position[1]))
            gravity_z = -float(data.xmat[contract.base_id].reshape(3, 3)[2, 2])
            fall_now = bool(position[2] - ground < .22 or gravity_z > -.35)
            if fall_now and first_fall_time is None:
                first_fall_time = float(data.time)
            fell |= fall_now
            outside_course |= bool(abs(position[1]) > args.terrain_config.course_width / 2)
            records["post_time_s"].append(float(data.time))
            records["fell_post_step"].append(fall_now)
            if viewer:
                draw_student_map(viewer, contract, adapter.last_channels, float(pre_pos[2]))
                viewer.sync()
                if not viewer.is_running():
                    break
                time.sleep(max(0., baseline.POLICY_DT - (time.perf_counter() - frame_start)))
            if fall_now and not args.continue_after_fall:
                break
    finally:
        baseline.close_viewer(viewer, threads)
    values = {key: np.asarray(value) for key, value in records.items()}
    observations = values["observation"]
    valid = observations[:, 244:431] > .5
    confidence = observations[:, 431:618]
    moving = values["time_s"] >= args.warmup_s
    selected_error = (values["velocity_body"] - values["command"])[moving]
    if len(selected_error) == 0:
        selected_error = values["velocity_body"] - values["command"]
    saturated = np.abs(values["action_issued"]) >= 100.
    result = {**identity, "model_path": str(args.model.resolve()), "model_xml_sha256": sha256(args.model),
              "mujoco_version": mujoco.__version__, "terrain": args.terrain_config.summary(),
              "model_physics_summary": {"total_mass_kg": float(model.body_mass.sum()), "actuators": model.nu,
                                        "robot_collision_geoms": int(((model.geom_bodyid != 0) & (model.geom_contype != 0)).sum()),
                                        "joint_order": list(baseline.JOINT_NAMES)},
              "noise_profile": args.noise_profile, **adapter.summary(), "seed": args.seed,
              "controller": "student_only", "teacher_executed_fraction": 0., "weights_updated": False,
              "physics": "closed loop mj_step", "physics_dt_s": float(model.opt.timestep),
              "policy_dt_s": baseline.POLICY_DT, "friction_scale": args.friction_scale,
              "command": command.tolist(), "warmup_s": args.warmup_s,
              "track_course": args.track_course,
              "duration_s": float(data.time), "control_steps": len(observations),
              "wall_time_s": time.perf_counter() - started, "fell": fell, "first_fall_time_s": first_fall_time,
              "ever_outside_course": outside_course, "end_base_position": data.xpos[contract.base_id].tolist(),
              "end_tilt_deg": float(np.rad2deg(math.acos(float(np.clip(-gravity_z, -1, 1))))),
              "end_base_ground_clearance_m": float(position[2] - ground),
              "net_xy_distance_m": float(np.linalg.norm(data.xpos[contract.base_id, :2] - start_xy)),
              "velocity_xy_rmse_m_s": float(np.sqrt(np.mean(np.sum(selected_error[:, :2]**2, axis=1)))),
              "velocity_yaw_rmse_rad_s": float(np.sqrt(np.mean(selected_error[:, 2]**2))),
              "tilt_p95_deg": float(np.rad2deg(np.quantile(values["tilt_rad"], .95))),
              "valid_fraction_mean": float(valid.mean()),
              "all_unknown_frames": int((~valid.any(axis=1)).sum()),
              "confidence_valid_mean": float(confidence[valid].mean()) if valid.any() else 0.,
              "lidar_visible_fraction_mean": float(values["lidar_visible_fraction"].mean()),
              "blackout_frames": int(values["blackout"].sum()),
              "action_abs_max": float(np.abs(values["action_issued"]).max()),
              "action_clipped_fraction": float(saturated.mean()),
              "evaluation_limit": "single seed fixed-command episode; crossing does not prove obstacle traversal or real-world capability"}
    result.update(contract.crossing_summary(data, fell))
    if args.output_dir:
        np.savez_compressed(args.output_dir / "trajectory.npz", **values)
        write_json(args.output_dir / "summary.json", result)
    if args.snapshot:
        render_snapshot(args.snapshot, model, contract, values, adapter.last_channels)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def interactive_rollout(args, policy, provenance, identity):
    """Manual velocity commands with explicit stand and student-policy handover."""
    model, contract, data, adapter, extras = create_simulation(args, provenance, prone=True)
    prepare_output(args.output_dir)
    keys = baseline.TeleopKeyboard()
    terminal = baseline.TerminalInput(keys)
    viewer, threads = None, []
    previous_action = np.zeros(16, dtype=np.float32)
    delay = deque()
    mode, pending_policy, show_map = "prone", False, True
    stand_start = lie_start = policy_start = 0.
    stand_initial = lie_initial = baseline.PRONE_JOINT_POS.copy()
    transitions = [{"time_s": 0., "mode": mode}]
    record_keys = ("time_s", "qpos", "qvel", "observation", "action_issued", "action_applied",
                   "base_position", "command", "mode", "post_time_s", "fell_post_step")
    records = {key: [] for key in record_keys} if args.output_dir or args.snapshot else None
    step = policy_steps = 0
    fell = False
    first_fall_time = None
    last_channels = None
    start_xy = data.xpos[contract.base_id, :2].copy()
    substeps = round(baseline.POLICY_DT / model.opt.timestep)
    if not math.isclose(substeps * model.opt.timestep, baseline.POLICY_DT):
        raise ValueError("Physics timestep must divide the policy timestep")

    def clear_policy():
        previous_action.fill(0.)
        delay.clear()
        delay.extend(previous_action.copy() for _ in range(extras.action_delay_steps))

    def enter(new_mode):
        nonlocal mode
        mode = new_mode
        transitions.append({"time_s": float(data.time), "mode": mode})
        print(f"[MODE] {mode} t={data.time:.2f}s", flush=True)

    clear_policy()
    started = time.perf_counter()
    try:
        viewer, threads = baseline.launch_viewer(model, data, key_callback=keys.on_keycode)
        with viewer.lock():
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = contract.base_id
            viewer.cam.distance, viewer.cam.elevation = 3.2, -18.
        terminal_active = terminal.start()
        print("[KEYS] Z stand | C student policy | W/S forward/back | A/D left/right | Q/E turn", flush=True)
        print("[KEYS] X lie down | R damping | H map | Esc quit", flush=True)
        print("[HEIGHTMAP] red=unknown, yellow=low confidence, green=high confidence", flush=True)
        print(f"[KEYS] Input: focused MuJoCo window{' or terminal' if terminal_active else ''}", flush=True)
        if keys._window_keys is None:
            print("[KEYS] Window hold/release queries unavailable; use the focused terminal with key repeat", flush=True)
        print("[MODE] prone; press Z to stand, then C to enter the student policy", flush=True)
        while viewer.is_running() and (args.steps is None or step < args.steps):
            frame_start = time.perf_counter()
            terminal.poll()
            quit_requested = False
            for key in keys.take_events():
                if key == "escape":
                    quit_requested = True
                elif key == "h":
                    show_map = not show_map
                elif key == "r":
                    pending_policy = False
                    keys.clear_motion()
                    clear_policy()
                    enter("damping")
                elif key == "z" and mode in {"prone", "damping", "lying"}:
                    stand_start = float(data.time)
                    stand_initial = data.qpos[contract.qpos_ids].copy()
                    pending_policy = False
                    keys.clear_motion()
                    clear_policy()
                    enter("standing")
                elif key == "c":
                    if mode == "standing":
                        pending_policy = True
                        print("[MODE] student handover queued until standing finishes", flush=True)
                    elif mode == "ready":
                        policy_start = float(data.time)
                        clear_policy()
                        enter("policy")
                    elif mode != "policy":
                        print("[MODE] press Z and wait for ready before C", flush=True)
                elif key == "x" and mode not in {"prone", "lying"}:
                    lie_start = float(data.time)
                    lie_initial = data.qpos[contract.qpos_ids].copy()
                    pending_policy = False
                    keys.clear_motion()
                    clear_policy()
                    enter("lying")
            if quit_requested:
                break
            if mode == "standing" and data.time - stand_start >= 2 * baseline.STAND_PHASE_DURATION:
                position = data.xpos[contract.base_id]
                ground = contract.ground_height(data, float(position[0]), float(position[1]))
                upright = data.xmat[contract.base_id].reshape(3, 3)[2, 2] > .8
                if position[2] >= ground + .42 and upright:
                    enter("ready")
                    if pending_policy:
                        policy_start = float(data.time)
                        pending_policy = False
                        clear_policy()
                        enter("policy")
            if mode == "lying" and data.time - lie_start >= baseline.STAND_PHASE_DURATION:
                enter("prone")

            command = keys.velocity_command() if mode == "policy" else np.zeros(3, dtype=np.float32)
            # Advance the same delayed map state while standing/ready as well.
            obs = adapter.observe(data, command, previous_action, step)
            last_channels = adapter.last_channels
            issued = np.zeros(16, dtype=np.float32)
            applied = issued.copy()
            target = baseline.DEFAULT_JOINT_POS
            if mode == "policy":
                with torch.inference_mode():
                    raw = policy(torch.from_numpy(obs)[None])[0].numpy()
                if raw.shape != (16,) or not np.isfinite(raw).all():
                    raise RuntimeError("Invalid student action")
                blend = min(1., (data.time - policy_start) / .3)
                issued = np.clip(raw * blend, -100., 100.).astype(np.float32)
                delay.append(issued.copy())
                applied = delay.popleft()
                policy_steps += 1
            elif mode == "standing":
                elapsed = data.time - stand_start
                if elapsed < baseline.STAND_PHASE_DURATION:
                    blend = baseline.smoothstep(elapsed / baseline.STAND_PHASE_DURATION)
                    target = stand_initial + blend * (baseline.PRE_STAND_JOINT_POS - stand_initial)
                else:
                    blend = baseline.smoothstep((elapsed - baseline.STAND_PHASE_DURATION) / baseline.STAND_PHASE_DURATION)
                    target = baseline.PRE_STAND_JOINT_POS + blend * (baseline.DEFAULT_JOINT_POS - baseline.PRE_STAND_JOINT_POS)
            elif mode == "lying":
                blend = baseline.smoothstep((data.time - lie_start) / baseline.STAND_PHASE_DURATION)
                target = lie_initial + blend * (baseline.PRONE_JOINT_POS - lie_initial)
            pre_pos = data.xpos[contract.base_id].copy()
            if records is not None:
                frame = {"time_s": float(data.time), "qpos": data.qpos.copy(), "qvel": data.qvel.copy(),
                         "observation": obs, "action_issued": issued, "action_applied": applied.copy(),
                         "base_position": pre_pos, "command": command.copy(), "mode": mode}
            was_policy = mode == "policy"
            for _ in range(substeps):
                if was_policy:
                    contract.apply_pd(data, applied)
                elif mode in {"standing", "ready", "lying"}:
                    contract.apply_pose_pd(data, target)
                else:
                    contract.apply_damping(data)
                mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise RuntimeError("Nonfinite MuJoCo state")
            previous_action[:] = issued
            position = data.xpos[contract.base_id]
            fall_now = False
            if was_policy:
                ground = contract.ground_height(data, float(position[0]), float(position[1]))
                upright = data.xmat[contract.base_id].reshape(3, 3)[2, 2] > .35
                fall_now = bool(position[2] < ground + .22 or not upright)
                fell |= fall_now
                if fall_now and first_fall_time is None:
                    first_fall_time = float(data.time)
                if fall_now and not args.continue_after_fall:
                    pending_policy = False
                    keys.clear_motion()
                    clear_policy()
                    enter("damping")
                    print("[MODE] fall detected; R damping, or Z then C to recover", flush=True)
            if records is not None:
                frame.update(post_time_s=float(data.time), fell_post_step=fall_now)
                for key, value in frame.items():
                    if args.output_dir:
                        records[key].append(value)
                    else:
                        records[key][:] = [value]  # Snapshot only: retain one frame.
            if show_map:
                draw_student_map(viewer, contract, last_channels, float(pre_pos[2]))
            else:
                with viewer.lock():
                    viewer.user_scn.ngeom = 0
            viewer.sync()
            step += 1
            time.sleep(max(0., baseline.POLICY_DT - (time.perf_counter() - frame_start)))
    finally:
        try:
            terminal.close()
        finally:
            try:
                baseline.close_viewer(viewer, threads)
            finally:
                keys.close()
    result = {**identity, "model_path": str(args.model.resolve()), "model_xml_sha256": sha256(args.model),
              "terrain": args.terrain_config.summary(), "noise_profile": args.noise_profile, **adapter.summary(),
              "seed": args.seed, "control_mode": "keyboard", "controller": "student_policy_with_manual_pose_modes",
              "teacher_executed_fraction": 0., "weights_updated": False, "track_course": False,
              "physics": "closed loop mj_step", "physics_dt_s": float(model.opt.timestep),
              "policy_dt_s": baseline.POLICY_DT, "friction_scale": args.friction_scale,
              "duration_s": float(data.time), "control_steps": step, "student_policy_steps": policy_steps,
              "wall_time_s": time.perf_counter() - started, "final_mode": mode, "transitions": transitions,
              "fell": fell, "first_fall_time_s": first_fall_time,
              "end_base_position": data.xpos[contract.base_id].tolist(),
              "net_xy_distance_m": float(np.linalg.norm(data.xpos[contract.base_id, :2] - start_xy)),
              "evaluation_limit": "operator-driven session including pose modes; not a fixed-command benchmark"}
    result.update(contract.crossing_summary(data, fell))
    if args.output_dir:
        if step:
            np.savez_compressed(args.output_dir / "trajectory.npz", **{key: np.asarray(value) for key, value in records.items()})
        write_json(args.output_dir / "summary.json", result)
    if args.snapshot and step:
        render_snapshot(args.snapshot, model, contract, {key: np.asarray(value) for key, value in records.items()}, last_channels)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def main(args):
    torch.set_num_threads(1)
    source = args.policy_jit or args.checkpoint
    policy, provenance, identity = load_student(source, jit=args.policy_jit is not None)
    if args.export_only:
        return export_student(args, policy, provenance, identity)
    if args.viewer and not args.autoplay:
        return interactive_rollout(args, policy, provenance, identity)
    if not args.suite:
        return rollout(args, policy, provenance, identity)
    prepare_output(args.output_dir)
    results = []
    for terrain in args.suite_terrains:
        for profile in args.suite_profiles:
            for seed in args.suite_seeds:
                case = copy(args)
                case.terrain, case.noise_profile, case.seed = terrain, profile, seed
                case.output_dir = args.output_dir / f"{terrain}_{profile}_seed{seed}"
                results.append(rollout(case, policy, provenance, identity))
                # Incremental receipt survives an interrupted suite.
                write_json(args.output_dir / "suite_summary.json", results)
    return results


if __name__ == "__main__":
    main(parse_args())
