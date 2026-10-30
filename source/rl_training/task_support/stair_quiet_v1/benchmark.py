"""Bounded, CPU-only MuJoCo rollout with raw contact and impact-source export.

Loads existing deployment adapters. Baseline roles retain their original PD;
explicit control roles add a controller to an unchanged pinned actor. Every
paired role receives the same declared stress profile. No optimization,
contact-softening or acoustic acceptance is performed here.
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

from .baselines import PROJECT_ROOT, TEACHER, TEACHER_SHA256, STUDENT, STUDENT_SHA256, checked_path
from .impact_events import ImpactEvents, select_window
from .traversal import TopLanding
from .mujoco_landing import (MODES, ROLES, MujocoLanding,
                             upward_support_force, validate_checkpoint_control, validate_role)
from .stress import PROFILES, StressChain, load_stress


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def run(args):
    mode = getattr(args, "landing_control", "off")
    validate_role(args.role, mode)
    stress_profile = load_stress(getattr(args, "stress_profile", "nominal"))
    # Import from the very deployment tree that the report fingerprints.
    deploy = PROJECT_ROOT / "deploy/deploy_mujoco"
    sys.path.insert(0, str(deploy))
    import mujoco
    import torch
    import deploy_mujoco as teacher_deploy
    import student_v1_adapter as student_deploy
    from terrain_loader import terrain_contact_bodies

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    baseline_role = args.role.startswith("baseline-")
    control_role = args.role.startswith("control-")
    is_student = args.role.endswith("student")
    if baseline_role or control_role:
        checkpoint = checked_path(
            args.checkpoint or (STUDENT if is_student else TEACHER),
            STUDENT_SHA256 if is_student else TEACHER_SHA256,
        )
    else:
        if args.checkpoint is None:
            raise ValueError("Candidate role requires a checkpoint")
        checkpoint = args.checkpoint.resolve()
        if digest(checkpoint) in {TEACHER_SHA256, STUDENT_SHA256}:
            raise ValueError("An unchanged baseline is not a candidate")
    from .landing import LandingSettings, control_contract, contract_hash
    settings = LandingSettings(mode=mode)
    control_document = control_contract(settings)
    if is_student:
        actor, provenance, identity = student_deploy.load_student(
            checkpoint, execution_control=control_document)
        if identity["actor_input_dim"] != 789:
            raise ValueError("This benchmark requires the v3.5 789-dimensional student ABI")
    else:
        actor = teacher_deploy.load_actor(checkpoint, allow_execution_control=True)
        provenance = {}
        identity = {"policy_sha256": digest(checkpoint), "actor_input_dim": 244}
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    checkpoint_control_checked = validate_checkpoint_control(saved, settings, control_document)
    del saved
    if not baseline_role and not control_role:
        reference = torch.load(
            checked_path(STUDENT if is_student else TEACHER, STUDENT_SHA256 if is_student else TEACHER_SHA256),
            map_location="cpu", weights_only=True,
        )
        key = "student_state_dict" if is_student else "actor_state_dict"
        reference_state = reference[key]
        state = actor.state_dict()
        if set(reference_state) == set(state) and all(torch.equal(state[k], v) for k, v in reference_state.items()):
            raise ValueError("Candidate policy weights equal the baseline; metadata/critic-only changes are not a new policy")
    terrain_args = SimpleNamespace(
        terrain="ascent", terrain_xml=None, stair_height=args.stair_height,
        tread_depth=args.tread_depth, stair_count=args.stair_count, stair_start=None,
    )
    model = teacher_deploy.load_model(args.model, terrain_args)
    if model.opt.integrator == mujoco.mjtIntegrator.mjINT_RK4:
        raise ValueError("Split-step contact sampling does not support RK4")
    contract = teacher_deploy.M20Contract(model, terrain_args.terrain_config)
    data = mujoco.MjData(model)
    contract.reset(data)
    # Apply the identical map error before either teacher observations or the
    # student's acquisition pipeline, including controller input. Collision
    # geometry and outcome measurements continue to use the unchanged terrain.
    if stress_profile.terrain_height_bias_m:
        original_scan = contract.height_scan

        def biased_scan(current_data):
            scan = original_scan(current_data)
            contract.last_scan_hits[:, 2] += stress_profile.terrain_height_bias_m
            return np.clip(scan - stress_profile.terrain_height_bias_m, -1., 1.)

        contract.height_scan = biased_scan
    if is_student:
        cfg, extras = student_deploy.profile_config("student-v1", provenance)
        adapter = student_deploy.StudentMapAdapter(contract, cfg, extras, args.seed)
        delay_steps = extras.action_delay_steps
        heading = None
    else:
        adapter = None
        delay_steps = 0
        heading = teacher_deploy.heading_controller(contract, actor)
    delay = deque([np.zeros(16, dtype=np.float32) for _ in range(delay_steps)])
    command = np.array([args.command_x, 0., 0.], dtype=np.float32)
    previous = np.zeros(16, dtype=np.float32)
    dt = float(model.opt.timestep)
    substeps = round(teacher_deploy.POLICY_DT / dt)
    if not math.isclose(substeps * dt, teacher_deploy.POLICY_DT):
        raise ValueError("Physics timestep does not divide policy timestep")
    stress = StressChain(stress_profile, dt, args.seed)
    controller = MujocoLanding(model, contract, settings, teacher_deploy) if mode != "off" else None
    if controller is not None:
        controller.contact_rng = np.random.default_rng(args.seed)
    feedback_force = np.zeros(4)
    feedback_valid = np.zeros(4, dtype=bool)
    modified_samples = 0
    control_samples = 0
    extractor = ImpactEvents(dt)
    landing = TopLanding()
    wheel_index = {int(body): i for i, body in enumerate(contract.wheel_body_ids)}
    legs = ("fl", "fr", "hl", "hr")
    terrain = terrain_args.terrain_config
    scalar = lambda key: float(np.asarray(terrain.parameters[key]).reshape(()))
    start_x = scalar("start_x")
    top_z = scalar("step_height") * int(scalar("step_count"))
    records = {key: [] for key in ("time_s", "normal_force_N", "wheel_com_velocity_m_s",
                                   "wheel_position_m", "root_position_m", "action_issued")}
    raw_contacts = []
    top_touches = np.zeros(4, dtype=bool)
    first_stair = None
    completed = fell = outside = False
    stationary = max_stationary = 0.
    before_weights = {k: v.detach().clone() for k, v in actor.state_dict().items()}
    started = time.monotonic()
    for step in range(args.steps):
        if adapter is not None:
            obs = adapter.observe(data, command, previous, step)
            inputs = torch.as_tensor(obs[None], dtype=torch.float32)
        else:
            effective = heading.update(data, command) if heading else command
            obs = contract.observe(data, effective, previous)
            inputs = teacher_deploy.policy_inputs(obs, command, actor)
        if controller is not None:
            controller.observe(data, adapter)
        with torch.inference_mode():
            issued = actor(inputs)[0].numpy()
        if issued.shape != (16,) or not np.isfinite(issued).all():
            raise RuntimeError("Invalid actor action")
        issued = np.clip(issued, -100., 100.).astype(np.float32)
        delay.append(issued.copy())
        applied = delay.popleft()
        before_xy = data.xpos[contract.base_id, :2].copy()
        top_touches.fill(False)
        for _ in range(substeps):
            mujoco.mj_step1(model, data)
            sensed_force, sensed_valid = stress.contact(feedback_force, feedback_valid)
            diagnostics = {}
            if controller is None:
                contract.apply_pd(data, applied)
            else:
                effort, diagnostics = controller.apply(
                    data, applied, sensed_force, sensed_valid,
                    bool(np.max(np.abs(command)) > settings.command_active_threshold), dt,
                    (1 + stress.contact_steps) * dt)
                data.ctrl[contract.actuator_ids] = effort
                modified_samples += int(np.count_nonzero(diagnostics["landing_modified"]))
                control_samples += 4
            requested_effort = data.ctrl[contract.actuator_ids].copy()
            actual_effort = stress.effort(requested_effort)
            data.ctrl[contract.actuator_ids] = actual_effort
            sample_time = float(data.time)
            wheel_pos = data.xpos[contract.wheel_body_ids].copy()
            root_pos = data.xpos[contract.base_id].copy()
            wheel_vel = np.zeros((4, 3))
            for i, body in enumerate(contract.wheel_body_ids):
                spatial = np.zeros(6)
                mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, int(body), spatial, 0)
                wheel_vel[i] = spatial[3:]
            mujoco.mj_step2(model, data)
            groups = {}
            totals = np.zeros(4)
            feedback_force = np.zeros(4)
            for ci in range(data.ncon):
                contact = data.contact[ci]
                if contact.efc_address < 0:
                    continue
                b1, b2 = terrain_contact_bodies(model, contact, contract.terrain_flex_ids)
                if b1 in wheel_index and b2 == 0:
                    wi, geom = wheel_index[b1], int(contact.geom2)
                    wheel_is_first = True
                elif b2 in wheel_index and b1 == 0:
                    wi, geom = wheel_index[b2], int(contact.geom1)
                    wheel_is_first = False
                else:
                    continue
                local = np.zeros(6)
                mujoco.mj_contactForce(model, data, ci, local)
                force = max(0., float(local[0]))
                normal = contact.frame[:3]
                # Contact normal points from geom1 to geom2; force on the first
                # body has the opposite sign. Vertical risers are not support.
                feedback_force[wi] += upward_support_force(normal, force, wheel_is_first)
                axis = int(np.argmax(np.abs(normal)))
                key = f"{legs[wi]}/{geom}/{axis}"
                point = contact.pos.copy()
                raw_contacts.append([len(records["time_s"]), wi, geom, axis, force, *point, *normal])
                row = groups.setdefault(key, {
                    "normal_force_N": 0., "position_sum": np.zeros(3),
                    "wheel": legs[wi], "surface": f"geom_{geom}_normal_axis_{axis}",
                })
                row["normal_force_N"] += force
                row["position_sum"] += point * force
                totals[wi] += force
                if (force >= extractor.settings.contact_on_N
                        and point[0] >= terrain.goal_x
                        and abs(point[2] - top_z) <= landing.settings.top_contact_height_tolerance_m
                        and abs(normal[2]) >= landing.settings.top_contact_normal_z_min):
                    top_touches[wi] = True
                if first_stair is None and force >= extractor.settings.contact_on_N and point[0] >= start_x:
                    first_stair = sample_time
            for row in groups.values():
                row["position_m"] = row.pop("position_sum") / max(row["normal_force_N"], 1e-12)
            extractor.update(sample_time, groups)
            feedback_valid.fill(True)
            for key, value in {
                "time_s": sample_time, "normal_force_N": totals,
                "wheel_com_velocity_m_s": wheel_vel, "wheel_position_m": wheel_pos,
                "root_position_m": root_pos, "action_issued": issued,
                "action_applied": applied, "effort_requested_Nm": requested_effort,
                "effort_applied_Nm": actual_effort,
                **diagnostics,
            }.items():
                records.setdefault(key, []).append(value)
        # Refresh post-integration positions for every outcome check.
        mujoco.mj_forward(model, data)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise RuntimeError("Non-finite physics")
        pos = data.xpos[contract.base_id]
        ground = contract.ground_height(data, float(pos[0]), float(pos[1]))
        fell |= bool(pos[2] - ground < .22 or data.xmat[contract.base_id].reshape(3, 3)[2, 2] < .35)
        wheels = data.xpos[contract.wheel_body_ids]
        outside |= bool(np.any(np.abs(wheels[:, 1]) + teacher_deploy.WHEEL_RADIUS > terrain.course_width / 2))
        progress_speed = float((pos[0] - before_xy[0]) / teacher_deploy.POLICY_DT)
        if first_stair is not None:
            stationary = stationary + teacher_deploy.POLICY_DT if progress_speed < .05 else 0.
            max_stationary = max(stationary, max_stationary)
        beyond_top = ((wheels[:, 0] >= terrain.goal_x + teacher_deploy.WHEEL_RADIUS)
                      & (wheels[:, 2] >= top_z + teacher_deploy.WHEEL_RADIUS - landing.settings.top_contact_height_tolerance_m))
        completed = landing.update(float(data.time), beyond_top, top_touches,
                                   invalid=fell or outside or first_stair is None)
        previous = issued
        if fell or outside or completed:
            break
    all_events = extractor.finish()
    window_start = first_stair if first_stair is not None else 0.
    window_end = float(data.time)
    selected, straddled = select_window(all_events, window_start, window_end)
    if any(not torch.equal(v, actor.state_dict()[k]) for k, v in before_weights.items()):
        raise RuntimeError("Inference unexpectedly modified actor weights")
    # Serialized MjModel includes compiled geometry/mesh and physics arrays.
    binary = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, None, binary)
    model_hash = hashlib.sha256(binary.tobytes()).hexdigest()
    source_paths = [Path(__file__), Path(__file__).with_name("impact_events.py"),
                    Path(__file__).with_name("traversal.py"),
                    Path(__file__).with_name("mujoco_landing.py"), Path(__file__).with_name("landing.py"),
                    Path(__file__).with_name("terrain_memory.py"),
                    Path(__file__).with_name("stress.py"),
                    deploy / "deploy_mujoco.py", deploy / "motion_residual_adapter.py",
                    deploy / "student_v1_adapter.py", deploy / "terrain_loader.py"]
    for relative in ("m20_student_v0", "m20_student_v1", "task_support/student_v0",
                     "task_support/student_v1", "task_support/student_v3_5",
                     "task_support/idle_stop_motion_v1"):
        source_paths.extend((PROJECT_ROOT / "source/rl_training" / relative).glob("*.py"))
    summary = {
        "schema": "stair_quiet_v1.mujoco_impact_rollout.v1",
        "status": "completed" if completed else "incomplete",
        "role": args.role, "checkpoint": str(checkpoint), "checkpoint_sha256": digest(checkpoint),
        "baseline_identity_checked": baseline_role, "actor_input_dim": identity["actor_input_dim"],
        "pinned_actor_identity_checked": baseline_role or control_role,
        "candidate_kind": ("baseline_actor_plus_controller" if control_role else
                           "trained_actor" if not baseline_role else "pinned_baseline"),
        "control_contract": control_document,
        "control_contract_sha256": contract_hash(control_document),
        "checkpoint_control_contract_checked": checkpoint_control_checked,
        "landing_control": mode,
        "controller_execution": {
            "enabled": controller is not None,
            "wheel_samples": control_samples, "modified_wheel_samples": modified_samples,
            "state": controller.control.summary() if controller is not None else {},
            "terrain_source": "student_delivered_map" if is_student else "teacher_observed_height_rays",
            "terrain_observation_dt_s": teacher_deploy.POLICY_DT,
            "contact_source": "previous completed physics step; upward support component only",
            "contact_inherent_age_s": dt,
            "hardware_feedback_available": False,
            "terrain_memory": controller.memory.summary() if controller is not None and controller.memory is not None else None,
            "terrain_memory_pose_source": "simulation_ground_truth_position_and_yaw; hardware_LIO_unvalidated",
        },
        "stress": stress.summary(),
        "model_path": str(args.model.resolve()), "model_xml_sha256": digest(args.model),
        "compiled_model_sha256": model_hash, "mujoco_version": mujoco.__version__,
        "seed": args.seed, "command": command.tolist(), "terrain": terrain.summary(),
        "physics_dt_s": dt, "policy_dt_s": teacher_deploy.POLICY_DT,
        "action_delay_steps": delay_steps,
        "student_sensor": adapter.summary() if adapter is not None else None,
        "duration_s": float(data.time), "wall_seconds": time.monotonic() - started,
        "completed": completed, "fell": fell, "ever_outside_course": outside,
        "max_stationary_interval_s": max_stationary,
        "measurement_start_s": window_start, "measurement_end_s": window_end,
        "measurement_duration_s": window_end - window_start,
        "window_valid": completed and first_stair is not None,
        "criterion": "first stair contact to every wheel having landed on top, plus fixed final-impact capture; no fall or bypass",
        "top_landing_seen": landing.seen.tolist(),
        "final_capture_started_s": landing.capture_started_s,
        "traversal_settings": asdict(landing.settings),
        "event_extraction": asdict(extractor.settings),
        "total_events": len(all_events), "window_events": len(selected),
        "boundary_straddling_events": straddled,
        "weights_updated": False, "optimizer_updates": 0,
        "sound_pressure_measured": False, "actual_acoustic_pass": False,
        "source_sha256": {str(p.relative_to(PROJECT_ROOT)): digest(p) for p in source_paths},
    }
    document = {
        "schema": "stair_quiet_v1.impact_events.v1",
        # Keep pre-window excitations so their modal/filter history is not lost.
        "duration_s": window_end, "events": all_events,
        "measurement_window_s": [window_start, window_end],
        "provenance": {
            "origin": "MuJoCo rigid-body contact impulses; unvalidated acoustic excitation",
            "checkpoint_sha256": summary["checkpoint_sha256"],
            "control_contract_sha256": summary["control_contract_sha256"],
            "stress_profile_sha256": summary["stress"]["profile_sha256"],
            "rollout_completed": completed, "window_valid": summary["window_valid"],
            "extraction": asdict(extractor.settings), "physics_dt_s": dt,
            "boundary_straddling_events": straddled,
            "boundary_handling": "full source history retained; pressure is windowed only after synthesis/filtering",
            "omissions": ["motors", "continuous rolling/sliding sound", "unresolved contact duration",
                          "loading below event thresholds", "structure/room calibration"],
            "impulse_definition": "total normal impulse in triggered window, including support preload; not baseline-subtracted dynamic impulse",
        },
    }
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output / "contacts.npz", **{k: np.asarray(v) for k, v in records.items()})
    np.savez_compressed(args.output / "contact_points.npz",
                        values=np.asarray(raw_contacts, dtype=np.float64).reshape(-1, 11),
                        columns=np.asarray(["sample_index", "wheel_index", "terrain_geom_id", "normal_axis",
                                            "normal_force_N", "x_m", "y_m", "z_m", "normal_x", "normal_y", "normal_z"]))
    write_json(args.output / "events.json", document)
    write_json(args.output / "all_events.json", {"events": all_events})
    write_json(args.output / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in (
        "status", "role", "duration_s", "wall_seconds", "window_events", "completed", "fell",
        "max_stationary_interval_s", "actual_acoustic_pass",
    )}, indent=2))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=ROLES, required=True)
    parser.add_argument("--landing-control", choices=MODES, default="off")
    parser.add_argument("--stress-profile", choices=tuple(PROFILES), default="nominal")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1800)
    parser.add_argument("--stair-height", type=float, default=.17)
    parser.add_argument("--tread-depth", type=float, default=.30)
    parser.add_argument("--stair-count", type=int, default=8)
    parser.add_argument("--command-x", type=float, default=.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    try:
        validate_role(args.role, args.landing_control)
    except ValueError as exc:
        parser.error(str(exc))
    if not 1 <= args.steps <= 10000 or not 1 <= args.stair_count <= 100:
        parser.error("Invalid bounded rollout size")
    if any(not math.isfinite(x) or x <= 0 for x in (args.command_x, args.stair_height, args.tread_depth)):
        parser.error("Command and stair dimensions must be finite and positive")
    if args.output.exists():
        parser.error("Output directory must be new")
    run(args)


if __name__ == "__main__":
    main()
