"""Quiet teacher keyboard-controlled MuJoCo viewer and 200Hz signal export.

Marked StairQuiet/Width/Safe checkpoints execute their saved shared landing control
at every physics substep. Unmarked teachers retain nominal PD control.
"""
import hashlib
import json
import math
import argparse
from pathlib import Path

import mujoco
import numpy as np
import torch

import deploy_mujoco as baseline
from quiet_landing_metrics import QuietMujocoMetrics, LiveSignals
from terrain_loader import terrain_contact_bodies
from teacher_noise_adapter import add_noise_arguments, validate_noise_arguments, make_noise_adapter, noise_metadata
from task_support.stair_quiet_v1.deployment import execution_contract, require_runtime
from task_support.stair_quiet_v1.landing import LandingSettings, control_contract
from task_support.stair_quiet_v1.mujoco_landing import (
    MujocoLanding, upward_support_force, validate_checkpoint_control,
)


def load_policy(checkpoint):
    """Read the execution contract before allowing nominal actor loading."""
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    noise = saved.get("m20_training_manifest", {}).get("noise_refinement", {})
    declared_settings = noise.get("contract", {}).get("environment", {}).get("landing_control")
    document = execution_contract(saved, required=declared_settings is not None)
    if document is not None:
        settings = LandingSettings(**document["settings"])
        require_runtime(saved, control_contract(settings))
        validate_checkpoint_control(saved, settings, document)
    actor = baseline.load_actor(checkpoint, allow_execution_control=document is not None)
    return actor, document


def support_feedback(model, data, contract):
    """Read only the completed solve; riser normals cannot count as support."""
    forces = np.zeros(4)
    wheels = {int(body): i for i, body in enumerate(contract.wheel_body_ids)}
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        if contact.efc_address < 0:
            continue
        first, second = terrain_contact_bodies(model, contact, contract.terrain_flex_ids)
        if first in wheels and second == 0:
            wheel, wheel_is_first = wheels[first], True
        elif second in wheels and first == 0:
            wheel, wheel_is_first = wheels[second], False
        else:
            continue
        local = np.zeros(6)
        mujoco.mj_contactForce(model, data, contact_id, local)
        forces[wheel] += upward_support_force(contact.frame[:3], local[0], wheel_is_first)
    return forces, np.ones(4, dtype=bool)


class QuietExecution:
    """Shared landing control hooks for fixed and manual policy execution."""

    def __init__(self, model, contract, document, actor=None):
        self.model, self.contract, self.document = model, contract, document
        self.adapter = MujocoLanding(model, contract, LandingSettings(**document["settings"]), baseline)
        require_runtime({"quiet_execution_control": document}, self.adapter.contract_document)
        self.applied_substeps = 0
        self.joint_limits = None
        if getattr(actor, 'joint_limit_contract', None) is not None:
            from task_support.cts.joint_limits_mujoco import MujocoJointLimits
            self.joint_limits = MujocoJointLimits(actor.joint_limit_contract, model, contract, baseline)
        self.safe = None
        if getattr(actor, "uses_safe_control", False):
            if getattr(actor, "uses_transition_control", False):
                from transition_actor_adapter import TransitionMujocoAdapter
                self.safe = TransitionMujocoAdapter(model, contract, actor)
            else:
                from safe_actor_adapter import SafeMujocoAdapter
                self.safe = SafeMujocoAdapter(model, contract, actor)
        self.reset()
        print(f"[CONTROL] StairQuiet landing={self.adapter.settings.mode}; checkpoint settings verified", flush=True)

    def reset(self):
        # Handover starts fresh contact/terrain state. Lifetime counters remain.
        # apply() retains inference tensors in the shared controller state.
        with torch.inference_mode():
            self.adapter.control.reset()
        self.adapter.memory = None
        self.adapter.map_time_s = None
        self.adapter.contact_history.clear()
        self.adapter.contact_history.extend(
            (np.zeros(4), np.zeros(4, dtype=bool))
            for _ in range(self.adapter.settings.contact_delay_steps))
        self.feedback_force, self.feedback_valid = np.zeros(4), np.zeros(4, dtype=bool)
        self.diagnostics = None
        if self.safe is not None:
            self.safe.reset()

    def policy_inputs(self, observation, command, actor, data, perception=None, trot_control=None):
        if self.safe is not None:
            return self.safe.inputs(observation, command, data, perception)
        if trot_control is not None:
            return baseline.policy_inputs(observation, command, actor, trot_control)
        return baseline.policy_inputs(observation, command, actor)

    def observe(self, data, observation_adapter=None):
        self.adapter.observe(data, observation_adapter)

    def check_issued(self, data, action):
        if self.joint_limits is not None:
            self.joint_limits.check_action(data, action)

    def apply(self, data, action, command):
        self.check_issued(data, action)
        dt = float(self.model.opt.timestep)
        active = bool(np.max(np.abs(command)) > self.adapter.settings.command_active_threshold)
        effort, self.diagnostics = self.adapter.apply(
            data, action, self.feedback_force, self.feedback_valid, active, dt, dt)
        if self.joint_limits is not None:
            self.joint_limits.check_corrected(data, self.diagnostics['landing_leg_target'])
        self.applied_substeps += 1
        return effort

    def after_step(self, data):
        if self.joint_limits is not None:
            self.joint_limits.check_feedback(data)
        self.after_idle_step(data)

    def after_idle_step(self, data):
        # Manual prone/stand/damping modes do not execute the CTS controller.
        # Keep contact feedback fresh without applying its takeover-only gate.
        # check_issued/apply still validate feedback before policy actuation.
        if self.safe is not None:
            self.safe.sample_contacts(data)
        self.feedback_force, self.feedback_valid = support_feedback(self.model, data, self.contract)
        self.sample_time = float(data.time)

    def record(self, records):
        if self.safe is not None:
            self.safe.record(records)
        if self.diagnostics is not None:
            # Manual traces contain policy substeps only, with their own time axis.
            records.setdefault("landing_time_s", []).append(self.sample_time)
            for key, value in self.diagnostics.items():
                records.setdefault(key, []).append(np.array(value, copy=True))

    def metadata(self):
        return {**(self.joint_limits.metadata() if self.joint_limits is not None else {}),
                **({"safe_runtime": self.safe.metadata()} if self.safe is not None else {}),
                "quiet_execution_control": self.document, "controller_execution": {
            "enabled": True, "applied_physics_substeps": self.applied_substeps,
            **self.adapter.control.summary(), "control_contract_sha256": self.adapter.contract_sha256}}


def execution_factory(document, actor=None):
    return (lambda model, contract: QuietExecution(model, contract, document, actor)) if document is not None else None


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_noise_arguments(parser)
    baseline.add_spawn_arguments(parser)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=baseline.REPO_ROOT / "logs/rsl_rl/deeprobotics_m20_stair_teacher/2026-10-09_01-55-35_gait_fix_descent_entry/model_214700.pt")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--terrain", choices=tuple(baseline.PRESETS), default=None)
    group.add_argument("--terrain-xml", type=Path)
    parser.add_argument("--command-x", type=float, default=.5, help="Headless fixed-command rollout only")
    parser.add_argument("--command-y", type=float, default=0., help="Headless fixed-command rollout only")
    parser.add_argument("--command-yaw", type=float, default=0., help="Headless fixed-command rollout only")
    for name in ("stair-height", "tread-depth", "stair-start"):
        parser.add_argument("--" + name, type=float)
    parser.add_argument("--stair-count", type=int)
    parser.add_argument("--steps", type=int, help="Optional step limit; viewer runs until you quit, headless defaults to 1000")
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--stop-on-fall", action="store_true")
    parser.add_argument("--viewer", action="store_true", help="Keyboard control: Z stand, C policy, W/S/A/D/Q/E move")
    parser.add_argument("--live-plot", action="store_true", help="Four-wheel live signal window (requires a desktop)")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--snapshot", type=Path, help="Save an offscreen image of the quiet markers at final frame")
    args = parser.parse_args()
    validate_noise_arguments(args, parser)
    if args.steps is not None and args.steps < 1:
        parser.error("steps must be positive")
    if args.print_every < 1:
        parser.error("print-every must be positive")
    return args


def main(args):
    torch.set_num_threads(1)
    actor, execution_document = load_policy(args.checkpoint)
    if args.viewer:
        args.quiet_landing = True
        return baseline.interactive_rollout(args, actor=actor, execution_factory=execution_factory(execution_document, actor))
    model = baseline.load_model(args.model, args)
    contract = baseline.M20Contract(model, args.terrain_config)
    data = mujoco.MjData(model)
    contract.reset(data)
    perception = make_noise_adapter(args, contract)
    contract.height_scan(data)
    metrics = QuietMujocoMetrics(model, contract)
    factory = execution_factory(execution_document, actor)
    controller = factory(model, contract) if factory is not None else None
    dt = float(model.opt.timestep)
    substeps = round(baseline.POLICY_DT / dt)
    if not math.isclose(substeps * dt, baseline.POLICY_DT):
        raise ValueError("Physics timestep does not divide policy timestep")
    command = np.array([args.command_x, args.command_y, args.command_yaw], dtype=np.float32)
    heading_hold = baseline.heading_controller(contract, actor)
    metrics.command_xy = command[:2]
    previous_action = np.zeros(baseline.ACTION_DIM, dtype=np.float32)
    fell = False
    live = LiveSignals() if args.live_plot else None
    try:
        for step in range(args.steps if args.steps is not None else 1000):
            effective_command = heading_hold.update(data, command) if heading_hold else command
            observation = (perception.observe(data, effective_command, previous_action, step) if perception is not None
                           else contract.observe(data, effective_command, previous_action))
            if controller is not None:
                controller.observe(data, perception.map_adapter if perception is not None else None)
            with torch.inference_mode():
                tensor_obs = (controller.policy_inputs(observation, command, actor, data, perception)
                              if controller is not None else baseline.policy_inputs(observation, command, actor))
                action = actor(tensor_obs)[0].numpy()
            if action.shape != (16,) or not np.isfinite(action).all():
                raise RuntimeError("Invalid quiet actor action")
            action = np.clip(action, -100., 100.)
            if controller is not None:
                controller.check_issued(data, action)
            applied_action = perception.action(action) if perception is not None else action
            metrics.tracker.begin_step()
            xy = data.xpos[contract.base_id, :2].copy()
            for _ in range(substeps):
                # step1 refreshes kinematics/contacts with the current qpos/qvel;
                # step2 solves forces and integrates. Snapshot is before solve.
                mujoco.mj_step1(model, data)
                if controller is None:
                    contract.apply_pd(data, applied_action)
                else:
                    data.ctrl[contract.actuator_ids] = controller.apply(data, applied_action, effective_command)
                snapshot = metrics.pre_step(data)
                mujoco.mj_step2(model, data)
                metrics.post_step(data, snapshot, record=args.output is not None or live is not None)
                if controller is not None:
                    controller.after_step(data)
                    if args.output is not None or live is not None:
                        controller.record(metrics.extended_records)
                if live is not None and args.output is None:
                    for values in metrics.records.values():
                        if len(values) > 1000:
                            del values[:-1000]
                    for values in metrics.extended_records.values():
                        if len(values) > 1000:
                            del values[:-1000]
            # Refresh the integrated pose before policy-time rays and outcomes;
            # contact feedback above remains from the preceding completed solve.
            mujoco.mj_fwdPosition(model, data)
            mujoco.mj_fwdVelocity(model, data)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise RuntimeError("Non-finite MuJoCo state")
            metrics.tracker.distance[0] += float(np.linalg.norm(data.xpos[contract.base_id, :2] - xy))
            previous_action = action.astype(np.float32)
            position = data.xpos[contract.base_id]
            local_height = contract.ground_height(data, float(position[0]), float(position[1]))
            gravity_z = float(data.xmat[contract.base_id].reshape(3, 3)[:, 2] @ np.array([0., 0., -1.]))
            fell |= bool(position[2] - local_height < .22 or gravity_z > -.35)
            if live is not None and step % 5 == 0:
                live.update(metrics)
            if args.stop_on_fall and fell:
                break
    finally:
        if live is not None:
            live.close()
    result = {"checkpoint": str(args.checkpoint.resolve()),
              "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              "model": str(args.model.resolve()), "command": command.tolist(),
              "terrain": args.terrain_config.summary(), "fell": fell,
              "duration_s": float(data.time), "quiet": metrics.summary(),
              "proxy": "horizontal-tread pre-impact wheel-centre speed and specific normal energy; not sound"}
    if controller is not None:
        result.update(controller.metadata())
    else:
        result["controller_execution"] = {"enabled": False, "applied_physics_substeps": 0}
    result.update(contract.crossing_summary(data, fell))
    result.update(noise_metadata(args, perception))
    if args.snapshot:
        metrics.snapshot(data, args.snapshot)
    if args.output:
        metrics.save(args.output, result)
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    main(parse_args())
