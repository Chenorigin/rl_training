#!/usr/bin/env python3
"""Bounded CPU closed-loop CTS evaluation through the actual Z -> C viewer path.

The headless viewer replaces drawing and operator input only. Observation,
history, startup blend, actuator delay, landing control and physics remain in
the deployment entrypoints. This is simulation evidence, never hardware approval.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

if __name__ == "__main__":
    for _thread_key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[_thread_key] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCENARIOS = ("handover", "left", "right", "forward_yaw")


def protocol(scenario, elapsed, speed):
    """Absolute policy time, with a common two-second zero-command handover."""
    vx, vy, yaw = speed
    if elapsed < 2 or scenario == "handover":
        return "handover", (), np.zeros(3)
    if scenario in ("left", "right"):
        if elapsed < 6:
            sign = 1 if scenario == "left" else -1
            return scenario, (("a" if sign > 0 else "d"),), np.array([0, sign * vy, 0.])
        return "stop", (), np.zeros(3)
    if scenario != "forward_yaw":
        raise ValueError(f"unknown scenario: {scenario}")
    if elapsed < 4:
        return "forward", ("w",), np.array([vx, 0., 0.])
    if elapsed < 6:
        return "add_yaw", ("w", "q"), np.array([vx, 0., yaw])
    if elapsed < 8:
        return "withdraw_yaw", ("w",), np.array([vx, 0., 0.])
    if elapsed < 10:
        return "reverse_yaw", ("w", "e"), np.array([vx, 0., -yaw])
    return "stop", (), np.zeros(3)


class ScriptedInput:
    """Queue the same Z/C events, then drive the original keyboard ramp or steps."""
    def __init__(self, keyboard, window, scenario, dt, speed, mode):
        self.keyboard, self.window = keyboard, window
        self.scenario, self.dt, self.speed, self.mode = scenario, dt, speed, mode
        self.calls, self.started = 0, False
        self.phase, self.requested = "standing", np.zeros(3)
        self._window_keys = window

    def take_events(self):
        if self.started:
            return []
        self.started = True
        return ["z", "c"]

    def velocity_command(self):
        self.phase, held, target = protocol(self.scenario, self.calls * self.dt, self.speed)
        self.window.held = set(held)
        self.requested = (self.keyboard.velocity_command() if self.mode == "keyboard"
                          else target.astype(np.float32))
        self.calls += 1
        return self.requested.copy()

    def clear_motion(self):
        self.keyboard.clear_motion()

    def on_keycode(self, key):
        return self.keyboard.on_keycode(key)

    def close(self):
        self.keyboard.close()


def summarize(rows, *, policy_dt, completed, fault=None):
    """Metrics have no invented physical acceptance thresholds."""
    if not rows:
        return {"status": "FAIL" if fault else "NOT_RUN", "reason": fault or "policy not entered",
                "telemetry_samples": 0, "telemetry_dt_s": None, "hardware_validation": "NOT_RUN"}
    numeric = ("time_s", "rpy", "velocity_body", "omega_body", "position", "target_q",
               "wheel_target", "joint_margin", "target_margin", "effort", "effort_limit",
               "requested", "effective_command", "support_z")
    finite = all(np.isfinite(np.asarray([r[k] for r in rows], dtype=float)).all() for k in numeric)
    if not finite:
        return {"status": "FAIL", "reason": "nonfinite physical telemetry", "telemetry_samples": len(rows),
                "telemetry_dt_s": None,
                "hardware_validation": "NOT_RUN"}
    a = {k: np.asarray([r[k] for r in rows], dtype=float) for k in numeric}
    t = a["time_s"] - a["time_s"][0]
    early = t <= 2. + 1.e-9
    # Include the actual standing target immediately preceding the first action.
    targets = np.vstack([rows[0]["handoff_target_q"], a["target_q"]])
    increments = np.abs(np.diff(targets, axis=0))
    measured_violation = bool(np.min(a["joint_margin"]) < 0)
    target_violation = bool(np.min(a["target_margin"]) < 0)
    failed = bool(fault or measured_violation or target_violation or any(r["fell"] for r in rows))
    phases = {}
    for phase in dict.fromkeys(r["phase"] for r in rows):
        mask = np.asarray([r["phase"] == phase for r in rows])
        indices = np.flatnonzero(mask)
        first, last = indices[0], indices[-1]
        yaw0 = a["rpy"][first, 2]
        delta = a["position"][last, :2] - a["position"][first, :2]
        phases[phase] = {
            "samples": int(mask.sum()), "duration_s": float(t[last] - t[first]),
            "mean_vx_vy_m_s": a["velocity_body"][mask, :2].mean(axis=0).tolist(),
            "mean_requested_command": a["requested"][mask].mean(axis=0).tolist(),
            "displacement_in_phase_start_heading_m": [float(math.cos(yaw0)*delta[0]+math.sin(yaw0)*delta[1]),
                                                      float(-math.sin(yaw0)*delta[0]+math.cos(yaw0)*delta[1])],
            "yaw_change_rad": float(np.unwrap(a["rpy"][mask, 2])[-1] - yaw0),
            "peak_abs_roll_pitch_rad": np.abs(a["rpy"][mask, :2]).max(axis=0).tolist(),
        }
    return {
        "status": "FAIL" if failed else ("COMPLETED" if completed else "INCOMPLETE"),
        "reason": fault, "telemetry_samples": len(rows), "policy_dt_s": policy_dt,
        "telemetry_dt_s": float(np.median(np.diff(a["time_s"]))) if len(rows) > 1 else None,
        "observed_policy_duration_s": float(t[-1]), "requested_sequence_completed": bool(completed),
        "weights_updated": False, "hardware_validation": "NOT_RUN", "phases": phases,
        "handover_first_2s": {
            "observed_duration_s": float(t[early][-1]),
            "peak_abs_roll_pitch_rad": np.abs(a["rpy"][early, :2]).max(axis=0).tolist(),
            "peak_abs_body_angular_rate_rad_s": np.abs(a["omega_body"][early]).max(axis=0).tolist(),
            "peak_nominal_leg_target_step_rad": float(increments[early].max()),
            "peak_abs_wheel_target_rad_s": float(np.abs(a["wheel_target"][early]).max()),
            "min_measured_mjcf_joint_margin_rad": float(a["joint_margin"][early].min()),
            "initial_wheel_ground_height_m": a["support_z"][0].tolist(),
            "initial_ground_height_relief_m": float(np.ptp(a["support_z"][0])),
            "support_scope": "collision terrain below wheel centres; not measured contact load",
        },
        "min_measured_mjcf_joint_margin_rad": float(a["joint_margin"].min()),
        "min_nominal_target_mjcf_joint_margin_rad": float(a["target_margin"].min()),
        "measured_joint_range_violation": measured_violation,
        "nominal_target_outside_mjcf_range": target_violation,
        "nominal_target_scope": "failure: CTS rejects nominal targets outside the verified software position envelope",
        "actuator_effort_at_total_clip_fraction": float(np.mean(np.abs(a["effort"]) >= a["effort_limit"] - 1.e-7)),
        "effort_scope": "MuJoCo total PD effort after shared controller clipping; not hardware/software target bounds",
        "fell": bool(any(r["fell"] for r in rows)),
    }


class Recorder:
    def __init__(self, runtime, model, contract, keys):
        self.runtime, self.model, self.contract, self.keys = runtime, model, contract, keys
        self.rows, self.fault = [], None
        self.action = None
        self.effective = np.zeros(3)
        self.handoff = runtime.DEFAULT_JOINT_POS[:12].copy()
        # mj_step2 integrates qpos but leaves xpos/xmat at the previous stage.
        # Recompute telemetry on separate data, without touching controller state.
        self.kinematics = runtime.mujoco.MjData(model)
        joints = [model.joint(name) for name in runtime.JOINT_NAMES[:12]]
        if not all(bool(j.limited[0]) for j in joints):
            raise ValueError("expected finite MJCF position limits on all twelve leg joints")
        self.bounds = np.asarray([j.range for j in joints])

    def sample(self, data, effort):
        if self.action is None:
            return
        rt, c = self.runtime, self.contract
        source = data
        data = self.kinematics
        data.qpos[:] = source.qpos
        data.qvel[:] = source.qvel
        data.mocap_pos[:] = source.mocap_pos
        data.mocap_quat[:] = source.mocap_quat
        data.time = source.time
        rt.mujoco.mj_fwdPosition(self.model, data)
        rt.mujoco.mj_fwdVelocity(self.model, data)
        rotation = data.xmat[c.base_id].reshape(3, 3)
        rpy = [math.atan2(rotation[2, 1], rotation[2, 2]),
               math.asin(float(np.clip(-rotation[2, 0], -1, 1))),
               math.atan2(rotation[1, 0], rotation[0, 0])]
        velocity = np.zeros(6)
        rt.mujoco.mj_objectVelocity(self.model, data, rt.mujoco.mjtObj.mjOBJ_BODY, c.base_id, velocity, 1)
        q = data.qpos[c.qpos_ids[:12]]
        target = rt.DEFAULT_JOINT_POS[:12] + self.action[:12] * rt.ACTION_SCALE[:12]
        margin = np.minimum(q-self.bounds[:, 0], self.bounds[:, 1]-q)
        target_margin = np.minimum(target-self.bounds[:, 0], self.bounds[:, 1]-target)
        position = data.xpos[c.base_id].copy()
        ground = c.ground_height(data, float(position[0]), float(position[1]))
        support_z = [c.ground_height(data, *map(float, data.xpos[b, :2])) for b in c.wheel_body_ids]
        fell = bool(position[2] < ground + .22 or rotation[2, 2] <= .35)  # existing viewer predicate
        row = dict(time_s=float(data.time), rpy=rpy, velocity_body=velocity[3:].copy(),
                   omega_body=velocity[:3].copy(), position=position, target_q=target,
                   wheel_target=self.action[12:]*rt.ACTION_SCALE[12:], joint_margin=margin,
                   target_margin=target_margin, effort=np.asarray(effort).copy(),
                   effort_limit=np.array([rt.LEG_EFFORT_LIMIT]*12+[rt.WHEEL_EFFORT_LIMIT]*4),
                   requested=self.keys.requested.copy(), effective_command=self.effective.copy(),
                   phase=self.keys.phase, support_z=support_z, fell=fell,
                   handoff_target_q=self.handoff.copy())
        self.rows.append(row)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            self.fault = "nonfinite generalized state"
        elif (margin < 0).any():
            self.fault = "measured leg position crossed MJCF joint range"
        elif fell:
            self.fault = "existing interactive rollout fall condition"


class RecordedExecution:
    def __init__(self, delegate, recorder):
        self.delegate, self.recorder, self.effort = delegate, recorder, None

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def reset(self):
        self.recorder.action = None
        return self.delegate.reset()

    def apply(self, data, action, command):
        self.recorder.action = np.clip(action, -100., 100.).copy()
        self.effort = self.delegate.apply(data, action, command)
        return self.effort

    def after_step(self, data):
        self.delegate.after_step(data)
        self.recorder.sample(data, self.effort)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n")


def run_case(args, rt, entry, actor, control, scenario, support, folder):
    folder.mkdir(parents=True, exist_ok=False)
    if support == "mat" and (args.spawn_mat_xy is None or args.spawn_mat_size is None):
        result = {"status": "NOT_RUN", "reason": "mat requires explicit --spawn-mat-xy and --spawn-mat-size"}
        write_json(folder/"result.json", result)
        return result
    case = SimpleNamespace(**vars(args))
    case.steps = args.startup_steps + args.policy_steps
    case.no_spawn_mat = support == "flat"
    case.output = folder/"playback.npz"
    case.quiet_landing, case.live_plot, case.snapshot = False, False, None
    case.stop_on_fall, case.print_every = True, case.steps + 1
    case.noise_reference = args.checkpoint
    window = SimpleNamespace(held=set(), focused=lambda: True, close=lambda: None,
                             note_viewer_event=lambda: None, note_terminal_event=lambda: True)
    window.pressed = lambda: window.held.copy()
    recorder = None
    with ExitStack() as stack:
        stack.enter_context(patch.object(rt.X11WindowKeys, "open", return_value=window))
        keyboard = rt.TeleopKeyboard()
        keys = ScriptedInput(keyboard, window, scenario, rt.POLICY_DT,
                             (args.speed_x, args.speed_y, args.yaw_rate), args.command_mode)
        def factory(model, contract):
            nonlocal recorder
            recorder = Recorder(rt, model, contract, keys)
            return RecordedExecution(entry.quiet.execution_factory(control, actor)(model, contract), recorder)
        def perception_factory(a, contract):
            perception = entry.StudentPerception(contract, a, actor.student_provenance)
            original = perception.observe
            def observe(*values):
                obs = original(*values)
                recorder.effective = obs[6:9].copy()
                return obs
            perception.observe, perception.draw = observe, lambda *_: None
            return perception
        class Viewer:
            cam = SimpleNamespace()
            ticks = 0
            def lock(self): return nullcontext()
            def sync(self): self.ticks += 1
            def is_running(self):
                return (self.ticks < case.steps and keys.calls < args.policy_steps
                        and (keys.calls > 0 or self.ticks < args.startup_steps)
                        and not (recorder and recorder.fault))
        viewer = Viewer()
        replacements = dict(TeleopKeyboard=lambda: keys,
                            TerminalInput=lambda _: SimpleNamespace(start=lambda: False, poll=lambda: None, close=lambda: None),
                            launch_viewer=lambda *_a, **_k: (viewer, ()), close_viewer=lambda *_: None,
                            make_noise_adapter=perception_factory, draw_height_scan=lambda *_: None,
                            heading_controller=entry.sensor_heading_controller,
                            TELEOP_LIMITS=np.array([args.speed_x, args.speed_y, args.yaw_rate]),
                            TELEOP_VELOCITIES={"w": (args.speed_x, 0, 0), "s": (-args.speed_x, 0, 0),
                                               "a": (0, args.speed_y, 0), "d": (0, -args.speed_y, 0),
                                               "q": (0, 0, args.yaw_rate), "e": (0, 0, -args.yaw_rate)})
        for name, value in replacements.items():
            stack.enter_context(patch.object(rt, name, value))
        stack.enter_context(patch.object(rt.time, "sleep", lambda _: None))
        playback, error = None, None
        try:
            playback = rt.interactive_rollout(case, actor=actor, execution_factory=factory)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        rows = recorder.rows if recorder else []
        result = summarize(rows, policy_dt=rt.POLICY_DT, completed=keys.calls >= args.policy_steps,
                           fault=error or (recorder.fault if recorder else None))
        result.update(scenario=scenario, support=support, command_mode=args.command_mode,
                      policy_steps_requested=args.policy_steps, policy_steps_executed=keys.calls,
                      total_step_cap=case.steps, playback=playback)
        expected = (["handover"] if scenario == "handover" else
                    ["handover", scenario, "stop"] if scenario in ("left", "right") else
                    ["handover", "forward", "add_yaw", "withdraw_yaw", "reverse_yaw", "stop"])
        result["phase_coverage"] = {phase: ("OBSERVED" if phase in result.get("phases", {}) else "NOT_RUN")
                                    for phase in expected}
        minimum = 2. if scenario == "handover" else (8. if scenario in ("left", "right") else 12.)
        result["full_protocol_observed"] = bool(keys.calls * rt.POLICY_DT >= minimum
                                                and all(v == "OBSERVED" for v in result["phase_coverage"].values()))
        if result["status"] == "COMPLETED" and not result["full_protocol_observed"]:
            result["status"] = "INCOMPLETE"
            result["reason"] = "step cap shorter than full scenario; see phase_coverage"
        if rows:
            np.savez_compressed(folder/"telemetry.npz", **{key: np.asarray([r[key] for r in rows]) for key in rows[0]})
        write_json(folder/"result.json", result)
        return result


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="New result directory")
    terrain = p.add_mutually_exclusive_group()
    terrain.add_argument("--terrain", default=None)
    terrain.add_argument("--terrain-xml", type=Path)
    p.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    p.add_argument("--supports", nargs="+", choices=("flat", "mat"), default=["flat", "mat"])
    p.add_argument("--spawn-xy", nargs=2, type=float)
    p.add_argument("--spawn-mat-xy", nargs=2, type=float)
    p.add_argument("--spawn-mat-size", nargs=3, type=float)
    p.add_argument("--spawn-mat-yaw-deg", type=float)
    p.add_argument("--startup-steps", type=int, default=400)
    p.add_argument("--policy-steps", type=int, default=600)
    p.add_argument("--command-mode", choices=("keyboard", "step"), default="keyboard")
    p.add_argument("--speed-x", type=float, default=.4)
    p.add_argument("--speed-y", type=float, default=.25)
    p.add_argument("--yaw-rate", type=float, default=.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--noise-profile", default="student-v1")
    p.add_argument("--profile-file", type=Path)
    p.add_argument("--noise-level", type=float)
    p.add_argument("--map-capture-padding", type=float)
    a = p.parse_args(argv)
    if not 1 <= a.startup_steps <= 5000 or not 1 <= a.policy_steps <= 5000:
        p.error("startup/policy steps must each be in [1,5000]")
    for name, maximum in (("speed_x", .7), ("speed_y", .5), ("yaw_rate", .6)):
        if not math.isfinite(getattr(a, name)) or not 0 < getattr(a, name) <= maximum:
            p.error(f"{name} must be positive and within the existing viewer command envelope")
    for name in ("spawn_xy", "spawn_mat_xy", "spawn_mat_size"):
        value = getattr(a, name)
        if value is not None and (not np.isfinite(value).all() or (name == "spawn_mat_size" and min(value) <= 0)):
            p.error(f"invalid {name}")
    if len(set(a.scenarios)) != len(a.scenarios) or len(set(a.supports)) != len(a.supports):
        p.error("duplicate scenarios/supports")
    if a.terrain is None and a.terrain_xml is None:
        a.terrain = "flat"
    return a


def main(argv=None):
    args = parse_args(argv)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path[:0] = [str(ROOT/"deploy/deploy_mujoco"), str(ROOT/"source/rl_training")]
    import torch
    import deploy_cts_student as entry
    rt = entry.quiet.baseline
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    entry.quiet.validate_noise_arguments(args, argparse.ArgumentParser())
    actor, control = entry.load_policy(args.checkpoint)
    if args.map_capture_padding is None:
        args.map_capture_padding = actor.student_provenance["perception"].get("map_capture_padding_m", 0.)
    args.output.mkdir(parents=True, exist_ok=False)
    results = {}
    for support in args.supports:
        for scenario in args.scenarios:
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            name = f"{support}_{scenario}"
            results[name] = run_case(args, rt, entry, actor, control, scenario, support, args.output/name)
    manifest = {"checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": digest(args.checkpoint),
                "model": str(args.model.resolve()), "model_sha256": digest(args.model),
                "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in
                                  [Path(__file__), ROOT/"deploy/deploy_mujoco/deploy_mujoco.py",
                                   ROOT/"deploy/deploy_mujoco/deploy_cts_student.py",
                                   ROOT/"deploy/deploy_mujoco/deploy_quiet_landing.py"]},
                "device": "cpu", "weights_updated": False, "results": results,
                "policy_input_contract": actor.student_provenance["student_inputs"],
                "acceptance": "COMPLETED means requested simulated samples collected; no hardware acceptance"}
    write_json(args.output/"result.json", manifest)
    return 0 if all(r["status"] == "COMPLETED" for r in results.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
