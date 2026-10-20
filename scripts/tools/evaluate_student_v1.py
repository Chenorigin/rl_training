#!/usr/bin/env python3
"""Freeze a v0/v1/teacher MuJoCo comparison plan; --run-plan executes a saved plan.

By default this writes only plan.json. No simulation, viewer or optimizer.
All students use the SAME v1 sensor pipeline for controlled comparison. Teacher
is an oracle reference and is only evaluated under the no-noise teacher profile.
"""
import argparse
from collections import deque
from dataclasses import replace
import json
import math
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"deploy/deploy_mujoco"))
sys.path.insert(0, str(ROOT/"source/rl_training"))
from m20_student_v1.profiles import PROFILE_NAMES
from m20_student_v0.noise_profiles import canonical_profile_name
from m20_student_v1.model_io import file_hash

COMMANDS = ("forward", "backward", "lateral", "yaw", "start_stop", "turn")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v0", type=Path)
    parser.add_argument("--v1", type=Path)
    parser.add_argument("--teacher", type=Path)
    parser.add_argument("--model", type=Path)
    terrain = parser.add_mutually_exclusive_group()
    terrain.add_argument("--terrain", choices=("flat", "ascent", "descent", "rough", "mixed", "obstacles", "hurdles", "slope_up", "slope_down"))
    terrain.add_argument("--terrain-xml", type=Path)
    parser.add_argument("--profiles", nargs="+", type=canonical_profile_name, choices=PROFILE_NAMES, default=["teacher", "lidar_fov", "height", "pose", "latency", "student-v0"])
    parser.add_argument("--commands", nargs="+", choices=COMMANDS, default=["forward"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--duration-s", type=float, default=120.)
    parser.add_argument("--track-course", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run", action="store_true", help="Explicitly execute the frozen plan with CPU MuJoCo physics")
    parser.add_argument("--run-plan", type=Path, help="Execute an existing frozen plan; pass this option alone")
    args = parser.parse_args(argv)
    if args.run_plan:
        raw = list(sys.argv[1:] if argv is None else argv)
        if any(token.startswith("--") and token.split("=")[0] != "--run-plan" for token in raw):
            parser.error("--run-plan must be passed alone; all settings come from the saved plan")
        return args
    if not all((args.v0, args.v1, args.model, args.output_dir)) or not (args.terrain or args.terrain_xml):
        parser.error("Planning requires --v0, --v1, --model, --output-dir, and one terrain option")
    if not math.isfinite(args.duration_s) or args.duration_s <= 0:
        parser.error("duration-s must be finite and positive")
    if args.track_course and args.commands != ["forward"]:
        parser.error("Route feedback applies only to the forward course test")
    return args


def freeze_plan(args):
    from m20_student_v1.model_io import resolve_checkpoint
    from m20_student_v1.configuration import StudentV1PerceptionCfg
    from dataclasses import asdict
    paths = {name: str(resolve_checkpoint(getattr(args, name))) for name in ("v0", "v1", "teacher") if getattr(args, name)}
    if not args.model.is_file() or (args.terrain_xml and not args.terrain_xml.is_file()):
        raise ValueError("Missing MJCF/terrain XML")
    if args.terrain_xml:
        terrain = args.terrain_xml.resolve()
    else:
        from terrain_loader import select_terrain
        terrain = select_terrain(args).resolve()
    cases = []
    for command in args.commands:
        for seed in args.seeds:
            for profile in args.profiles:
                for name in paths:
                    if name == "teacher" and canonical_profile_name(profile) != "teacher":
                        continue
                    cases.append({"model": name, "profile": profile, "command": command, "seed": seed,
                                  "y_offset_m": ((seed % 3)-1)*.03, "yaw_offset_rad": ((seed % 3)-1)*.03})
    source_paths = [Path(__file__), ROOT/"deploy/deploy_mujoco/deploy_mujoco.py",
                    ROOT/"deploy/deploy_mujoco/student_v1_adapter.py", ROOT/"deploy/deploy_mujoco/quiet_landing_metrics.py",
                    ROOT/"source/rl_training/m20_student_v0/__init__.py",
                    ROOT/"source/rl_training/m20_student_v1/__init__.py",
                    *sorted((ROOT/"source/rl_training/task_support").rglob("*.py"))]
    plan = {"version": "v1", "checkpoints": paths, "checkpoint_sha256": {name: file_hash(path) for name, path in paths.items()},
            "model": str(args.model.resolve()), "model_sha256": file_hash(args.model),
            "terrain_xml": str(terrain), "terrain_sha256": file_hash(terrain),
            "source_sha256": {str(path): file_hash(path) for path in source_paths},
            "cases": cases, "duration_s": args.duration_s, "policy_dt_s": .02, "physics_dt_s": .005,
            "track_course": args.track_course, "fall": {"min_body_ground_height_m": .22, "min_Rzz": .35},
            "sensor_reference": "v1 parameter ledger; same single-factor pipeline for both students; measured hardware calibration pending",
            "sensor_defaults": asdict(StudentV1PerceptionCfg()),
            "rule": "report every case; compare identical profile/command/initial pose; inspect task progress together with per-meter energy, impact tail and slip; no benefit declaration from action loss alone",
            "teacher_reference": "oracle clean only; never compare teacher clean and student noisy as same-condition robustness",
            "initial_pose_samples": "seed-indexed sensitivity offsets; not independent randomized significance trials"}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    plan_path = args.output_dir/"plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False)+"\n")
    plan_path.chmod(0o444)
    return plan


def verify_plan(plan):
    pairs = [(path, plan["checkpoint_sha256"][name]) for name, path in plan["checkpoints"].items()]
    pairs += [(plan["model"], plan["model_sha256"]), (plan["terrain_xml"], plan["terrain_sha256"])]
    pairs += list(plan["source_sha256"].items())
    for path, digest in pairs:
        if file_hash(path) != digest:
            raise ValueError(f"Frozen evaluation input changed: {path}")


def command_at(name, t):
    import numpy as np
    command = {"forward": [.5, 0, 0], "backward": [-.5, 0, 0], "lateral": [0, .3, 0], "yaw": [0, 0, .5],
               "start_stop": [.5 if t % 10 < 5 else 0., 0, 0], "turn": [.5, 0, .3 if t % 10 < 5 else -.3]}[name]
    return np.asarray(command, dtype=np.float32)


def planar_distance(before, after):
    """Both positions must describe consecutive policy boundaries after integration."""
    import numpy as np
    return float(np.linalg.norm(np.asarray(after)[:2]-np.asarray(before)[:2]))


def touchdown_windows(records, dt, window_s=.04):
    """Per-wheel force peaks and impulse over a fixed post-touchdown time window."""
    import numpy as np
    events = np.asarray(records["event"], dtype=bool)
    force = np.asarray(records["force_norm_N"])
    peaks, impulses = [], []
    samples = math.ceil(window_s/dt-1e-9)
    for index, wheel in np.argwhere(events):
        end = index+samples
        # Exclude truncated end-of-episode windows rather than report artificially small impulses.
        if end > len(force):
            continue
        values = force[index:end, wheel]
        peaks.append(float(values.max()))
        impulses.append(float(values.sum()*dt))
    return {"window_s": window_s, "full_window_events": len(peaks),
            "force_peak_p95_N": float(np.quantile(peaks, .95)) if peaks else None,
            "force_impulse_p95_N_s": float(np.quantile(impulses, .95)) if impulses else None,
            "truncated_events": int(events.sum())-len(peaks)}


def run_case(plan, case, output):
    import mujoco
    import numpy as np
    import torch
    from tensordict import TensorDict
    import deploy_mujoco as baseline
    from quiet_landing_metrics import QuietMujocoMetrics
    from student_v0_adapter import load_student as load_v0
    from student_v1_adapter import load_student as load_v1, profile_config, StudentMapAdapter
    started = time.monotonic()
    name = case["model"]
    if name == "teacher" and canonical_profile_name(case["profile"]) != "teacher":
        raise ValueError("Teacher reference only accepts the no-noise teacher profile")
    _, reference_provenance, _ = load_v1(Path(plan["checkpoints"]["v1"]))
    path = Path(plan["checkpoints"][name])
    teacher = name == "teacher"
    if teacher:
        actor = baseline.load_actor(path).eval().requires_grad_(False)
        provenance = {}
    else:
        actor, provenance, _ = (load_v0 if name == "v0" else load_v1)(path)
    torch.manual_seed(case["seed"])
    args = SimpleNamespace(terrain=None, terrain_xml=Path(plan["terrain_xml"]))
    model = baseline.load_model(Path(plan["model"]), args)
    if not math.isclose(model.opt.timestep, plan["physics_dt_s"]):
        raise ValueError("Physics timestep differs from frozen criteria")
    contract = baseline.M20Contract(model, args.terrain_config)
    data = mujoco.MjData(model)
    contract.reset(data)
    adr = int(contract.root_joint.qposadr[0])
    data.qpos[adr+1] += case["y_offset_m"]
    yaw = contract.terrain.spawn_yaw+case["yaw_offset_rad"]
    data.qpos[adr+3:adr+7] = [math.cos(yaw/2), 0, 0, math.sin(yaw/2)]
    mujoco.mj_forward(model, data)
    cfg, extras = profile_config(case["profile"], reference_provenance)
    if name == "v0":
        extras = replace(extras, history_steps=1)
    adapter = StudentMapAdapter(contract, cfg, extras, case["seed"]) if not teacher else None
    delay_steps = extras.action_delay_steps if adapter else 0
    previous = np.zeros(16, dtype=np.float32)
    queue = deque([previous.copy() for _ in range(delay_steps)])
    metrics = QuietMujocoMetrics(model, contract)
    telemetry = {key: [] for key in ("time_s", "base_position", "command", "velocity_body", "action", "valid_fraction")}
    fell = boundary = complete = False
    substeps = round(plan["policy_dt_s"]/model.opt.timestep)
    for step in range(math.ceil(plan["duration_s"]/plan["policy_dt_s"])):
        # mj_step2 leaves original xpos/xmat/cvel one substep behind qpos/qvel.
        # Scratch kinematics synchronizes observations without overwriting solved forces.
        observation_data = metrics.refresh_post_kinematics(data)
        command = command_at(case["command"], step*plan["policy_dt_s"])
        if plan["track_course"]:
            r = observation_data.xmat[contract.base_id].reshape(3, 3)
            yaw = math.atan2(r[1, 0], r[0, 0])
            target = contract.terrain.spawn_yaw
            lateral = float(np.clip(-.5*(observation_data.xpos[contract.base_id, 1]-contract.terrain.spawn_xy[1]), -.25, .25))
            vx = .5*math.cos(target)-lateral*math.sin(target)
            vy = .5*math.sin(target)+lateral*math.cos(target)
            command[:2] = [math.cos(yaw)*vx+math.sin(yaw)*vy, -math.sin(yaw)*vx+math.cos(yaw)*vy]
            command[2] = np.clip(.5*math.atan2(math.sin(target-yaw), math.cos(target-yaw)), -.5, .5)
        obs = adapter.observe(observation_data, command, previous, step) if adapter else contract.observe(observation_data, command, previous)
        with torch.inference_mode():
            tensor = torch.from_numpy(obs)[None]
            action = (actor(TensorDict({"policy": tensor}, batch_size=[1])) if teacher else actor(tensor))[0].numpy()
        if action.shape != (16,) or not np.isfinite(action).all():
            raise ValueError("Nonfinite evaluation action")
        issued = np.clip(action, -100, 100).astype(np.float32)
        queue.append(issued.copy())
        applied = queue.popleft()
        metrics.command_xy = command[:2].copy()
        metrics.tracker.begin_step()
        before_position = observation_data.xpos[contract.base_id].copy()
        for _ in range(substeps):
            contract.apply_pd(data, applied)
            mujoco.mj_step1(model, data)
            snapshot = metrics.pre_step(data)
            mujoco.mj_step2(model, data)
            metrics.post_step(data, snapshot, record=True)
        post = metrics.kinematics
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise ValueError("Nonfinite evaluation physics")
        position = post.xpos[contract.base_id].copy()
        metrics.tracker.distance[0] += planar_distance(before_position, position)
        previous = issued.copy()
        spatial = np.zeros(6)
        mujoco.mj_objectVelocity(model, post, mujoco.mjtObj.mjOBJ_BODY, contract.base_id, spatial, 1)
        velocity = np.asarray([spatial[3], spatial[4], spatial[2]])
        for key, value in (("time_s", float(data.time)), ("base_position", position), ("command", command.copy()),
                           ("velocity_body", velocity), ("action", issued),
                           ("valid_fraction", float(adapter.last_channels[1].mean()) if adapter else 1.)):
            telemetry[key].append(value)
        ground = contract.ground_height(post, float(position[0]), float(position[1]))
        fell = bool(position[2]-ground < plan["fall"]["min_body_ground_height_m"] or post.xmat[contract.base_id].reshape(3, 3)[2, 2] < plan["fall"]["min_Rzz"])
        boundary = bool(np.any(np.abs(post.xpos[contract.wheel_body_ids, 1]) > contract.terrain.course_width/2))
        complete = bool(contract.crossing_summary(post, fell)["cleared_terrain"])
        if fell or boundary or (complete and case["command"] == "forward"):
            break
    folder = output/f'{name}_{case["profile"]}_{case["command"]}_seed{case["seed"]}'
    folder.mkdir(exist_ok=False)
    arrays = {key: np.asarray(value) for key, value in telemetry.items()}
    velocity_error = arrays["velocity_body"]-arrays["command"]
    event_speed = np.asarray(metrics.records["speed"])[np.asarray(metrics.records["event"], dtype=bool)]
    force = np.asarray(metrics.records["force_norm_N"])
    specific_energy = .5*event_speed**2
    distance = float(metrics.tracker.distance[0])
    result = {"case": case, "weights_updated": False, "duration_s": float(data.time), "fell": fell,
              "boundary_exit": boundary, "completed_course": complete and not fell and not boundary,
              "survived": not fell and not boundary, "distance_m": distance,
              "velocity_rmse": np.sqrt((velocity_error**2).mean(0)).tolist(),
              "touchdown_speed_p95_m_s": float(np.quantile(event_speed, .95)) if len(event_speed) else None,
              "touchdown_events_per_m": len(event_speed)/distance if distance > 1e-6 else None,
              "touchdown_specific_energy_per_m": float(specific_energy.sum())/distance if distance > 1e-6 else None,
              "wheel_force_p95_N": float(np.quantile(force, .95)),
              "touchdown_force": touchdown_windows(metrics.records, model.opt.timestep),
              "quiet_aggregate": metrics.summary(), "sensor": adapter.summary() if adapter else "oracle clean",
              "wall_time_s": time.monotonic()-started,
              "force_metric_definition": "wheel_force: all-substep force norm P95; touchdown_force: per-wheel 40 ms post-touchdown peak/impulse, complete windows only"}
    np.savez_compressed(folder/"telemetry.npz", **arrays)
    np.savez_compressed(folder/"quiet.npz", **{key: np.asarray(value) for key, value in metrics.records.items()},
                        **{key: np.asarray(value) for key, value in metrics.extended_records.items()})
    (folder/"summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n")
    return result


def main(argv=None):
    args = parse_args(argv)
    if args.run_plan:
        if not args.run_plan.is_file():
            raise ValueError("Saved evaluation plan does not exist")
        plan = json.loads(args.run_plan.read_text())
        if plan.get("version") != "v1":
            raise ValueError("Unsupported evaluation plan version")
        args.output_dir = args.run_plan.resolve().parent
        if (args.output_dir/"suite_summary.json").exists() or (args.output_dir/"execution.json").exists():
            raise ValueError("This plan has already executed; choose a new output directory for a new comparison")
        args.run = True
    else:
        plan = freeze_plan(args)
    print(f'V1_EVALUATION_PLAN={args.output_dir/"plan.json"}; cases={len(plan["cases"])}; run={args.run}', flush=True)
    if not args.run:
        return
    import psutil
    import torch
    from m20_student_v0.output_storage import require_disk_space
    torch.set_num_threads(1)
    host = psutil.virtual_memory()
    estimate = len(plan["cases"])*plan["duration_s"]*500000+1024**3
    require_disk_space(args.output_dir, estimate)
    if host.total-host.available+2*1024**3 > .85*host.total:
        raise RuntimeError("Insufficient whole-host headroom for single-thread CPU evaluation")
    (args.output_dir/"resource_preflight.json").write_text(json.dumps({"host_available_bytes": host.available,
         "planned_ram_bytes": 2*1024**3, "planned_output_bytes": estimate, "cpu_threads": 1,
         "basis": "previous three-policy CPU benchmark peak RSS ~1.15 GiB; 2 GiB reserve; 0.5 MB/s raw telemetry allowance"}, indent=2)+"\n")
    verify_plan(plan)
    results = []
    for case in plan["cases"]:
        verify_plan(plan)
        results.append(run_case(plan, case, args.output_dir))
        (args.output_dir/"suite_summary.json").write_text(json.dumps(results, indent=2, ensure_ascii=False)+"\n")
        print("V1_CASE_DONE="+json.dumps(results[-1], ensure_ascii=False), flush=True)
    verify_plan(plan)
    (args.output_dir/"execution.json").write_text(json.dumps({"cases": len(results), "weights_updated": False,
        "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})+"\n")


if __name__ == "__main__":
    main()
