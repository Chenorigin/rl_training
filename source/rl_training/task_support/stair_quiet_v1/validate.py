"""Bounded Isaac inference check. This entry never calls learn/update/optimizer.step.

Run from the repository with PYTHONPATH=source/rl_training. Omit --checkpoint
to check the exact source teacher under the new environment and noise telemetry.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--student", action="store_true",
                        help="Use the pinned 21000 nominal actor in the 789D quiet student task")
    parser.add_argument("--landing-mode", choices=("combined", "off", "precontact", "phase"))
    parser.add_argument("--contact-delay-steps", type=int)
    parser.add_argument("--contact-dropout-probability", type=float)
    parser.add_argument("--output", type=Path, required=True)
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if not 1 <= args.num_envs <= 16 or not 1 <= args.steps <= 256:
        parser.error("Inference check is bounded to 1..16 envs and 1..256 steps")
    if args.student and args.checkpoint:
        parser.error("--student verifies the pinned 21000 source; --checkpoint is teacher-only")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"status": "running", "optimizer_updates": 0}) + "\n")
    app = AppLauncher(args).app
    env = None
    try:
        import gymnasium as gym
        import torch
        from isaaclab_tasks.utils import load_cfg_from_registry
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        import rl_training  # noqa: F401 -- task registration
        from .baselines import (PROJECT_ROOT, TASK, TEACHER, TEACHER_SHA256,
                                STUDENT, STUDENT_SHA256, checked_path)
        from .runner import StairQuietRunner
        from .landing import control_contract
        from copy import deepcopy
        sys.path.insert(0, str(PROJECT_ROOT / "scripts/reinforcement_learning/rsl_rl"))
        from cli_args import convert_rsl_rl_cfg_dict

        task = TASK + "-Student" if args.student else TASK
        cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
        agent = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
        cfg.scene.num_envs = args.num_envs
        cfg.sim.device = args.device
        cfg.seed = 42
        for argument, field in (("landing_mode", "mode"), ("contact_delay_steps", "contact_delay_steps"),
                                ("contact_dropout_probability", "contact_dropout_probability")):
            if getattr(args, argument) is not None:
                setattr(cfg.landing_control, field, getattr(args, argument))
        cfg.landing_control.validate()
        source_student = None
        if args.student:
            # Load only the pinned actor. No distillation algorithm, replay
            # allocation, teacher optimizer or optimizer state is constructed.
            from task_support.student_v1.model_io import actor_from_state
            teacher_saved = torch.load(checked_path(TEACHER, TEACHER_SHA256),
                                       map_location="cpu", weights_only=False)
            cfg.motion_teacher_tuning = deepcopy(teacher_saved["m20_training_manifest"]["stair_tuning"])
            del teacher_saved
            saved = torch.load(checked_path(STUDENT, STUDENT_SHA256),
                               map_location="cpu", weights_only=False)
            source_student = deepcopy(saved["student_v1"]["provenance"])
            if source_student["contract"]["student_observation_dim"] != 789:
                raise ValueError("Pinned student must have the 789D observation ABI")
            model = actor_from_state(saved["student_state_dict"], 789,
                                     source_student["training_config"]["student"]).to(args.device)
            del saved
        # Keep the actual task geometry/rewards. Reducing num_envs never changes them.
        env = RslRlVecEnvWrapper(gym.make(task, cfg=cfg), clip_actions=agent.clip_actions)
        if args.student:
            actor = lambda observation: model(observation["policy"])
            reference = None
            manifest = {
                "source_student_sha256": STUDENT_SHA256,
                "source_teacher_sha256": TEACHER_SHA256,
                "control_contract": control_contract(cfg.landing_control),
                "source_student_observation_contract": source_student["contract"],
            }
        else:
            runner = StairQuietRunner(env, convert_rsl_rl_cfg_dict(agent.to_dict()), device=args.device)
            if args.checkpoint:
                runner.load(args.checkpoint)
                runner.initialize_task({"resume": True})
            else:
                runner.initialize_task({})
            manifest = runner._record_manifest()
            model, reference = runner.alg.actor, runner.alg.reference_actor
            actor = model
        before = {k: v.clone() for k, v in model.state_dict().items()}
        obs = env.get_observations()
        expected_dims = {"policy": 789, "critic": 789, "teacher": 244, "motion_control": 3} if args.student else {
            "policy": 244, "critic": 285, "motion_control": 3}
        for name, width in expected_dims.items():
            assert obs[name].shape == (args.num_envs, width), (name, obs[name].shape)
        parity = 0.
        resets = 0
        nonzero_noise_steps = 0
        reward_sums = 0.
        maximum_landing_correction = 0.
        maximum_landing_gain_change = 0.
        kinematics_maxima = {}
        student_history_checks = None
        with torch.inference_mode():
            for _ in range(args.steps):
                actions = actor(obs)
                assert torch.isfinite(actions).all()
                if not args.student and not args.checkpoint:
                    parity = max(parity, (actions-reference(obs)).abs().max().item())
                obs, reward, done, extras = env.step(actions)
                assert torch.isfinite(reward).all()
                resets += int(done.sum().item())
                reward_sums += float(reward.sum().item())
                tracker = getattr(env.unwrapped, "stair_quiet_tracker", None)
                if tracker is not None:
                    costs = getattr(tracker, "costs", {})
                    nonzero_noise_steps += int(any(bool((v > 0).any()) for v in costs.values()))
                landing = env.unwrapped.stair_quiet_landing
                maximum_landing_correction = max(
                    maximum_landing_correction, landing.last["correction"].abs().max().item())
                maximum_landing_gain_change = max(maximum_landing_gain_change,
                    (landing.last["kp_scale"] - 1).abs().max().item(),
                    (landing.last["kd_scale"] - 1).abs().max().item())
                for key, value in landing.kinematics_diagnostic().items():
                    kinematics_maxima[key] = max(kinematics_maxima.get(key, 0.), value)
            if args.student:
                from task_support.student_v1 import environment as glue
                state, delay = glue.sensor_state(env.unwrapped), glue.action_delay(env.unwrapped)
                retained_history = state.history[1:].clone()
                retained_issued = delay.previous_issued[1:].clone()
            # Force partial reset through the real environment reset path.
            env.unwrapped._reset_idx(torch.tensor([0], device=env.device))
            assert env.unwrapped.stair_quiet_landing.controller.correction[0].count_nonzero() == 0
            if args.student:
                assert state.history[0].count_nonzero() == 0
                assert delay.previous_issued[0].count_nonzero() == 0
                assert torch.equal(state.history[1:], retained_history)
                assert torch.equal(delay.previous_issued[1:], retained_issued)
            obs = env.get_observations()
            assert torch.isfinite(actor(obs)).all()
            if args.student:
                # The first post-reset observation exposes zero past history;
                # repeated reads at the same policy tick cannot advance it.
                assert obs["policy"][0, 618:].count_nonzero() == 0
                assert torch.equal(state.history[1:], retained_history)
                current_history = state.history.clone()
                repeated = env.get_observations()
                assert torch.equal(repeated["policy"], obs["policy"])
                assert torch.equal(state.history, current_history)
                student_history_checks = {
                    "reset_history_zero": True, "reset_issued_action_zero": True,
                    "other_environment_history_unchanged": True,
                    "same_tick_observation_does_not_advance_history": True,
                }
        assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
        if not args.student and not args.checkpoint:
            assert parity == 0.
        assert env.unwrapped.stair_quiet_landing.applied_substeps == args.steps * cfg.decimation
        landing_summary = env.unwrapped.stair_quiet_landing.summary()
        activated = landing_summary["modified_samples"] > 0
        report = {
            "status": ("passed" if activated or cfg.landing_control.mode == "off" else
                       "incomplete_control_coverage"),
            "task": task, "source_teacher_sha256": TEACHER_SHA256,
            "source_student_sha256": STUDENT_SHA256 if args.student else None,
            "rl_training_import": rl_training.__file__,
            "num_envs": args.num_envs, "steps": args.steps, "resets": resets,
            "forced_partial_reset": True, "optimizer_updates": 0,
            "actor_weights_unchanged": True,
            "source_teacher_action_max_abs_diff": parity if not args.student and not args.checkpoint else None,
            "student_history_checks": student_history_checks,
            "observation_shapes": {k: list(v.shape) for k, v in obs.items()},
            "finite_reward_sum": reward_sums,
            "nonzero_noise_steps": nonzero_noise_steps,
            "quiet_cost_names": list(env.unwrapped.stair_quiet_tracker.costs),
            "quiet_curriculum_scale": env.unwrapped.stair_quiet_reward_scale,
            "quiet_task_iteration": env.unwrapped.stair_quiet_iteration,
            "wheel_collision_geometry": env.unwrapped.stair_quiet_geometry,
            "landing_control": landing_summary,
            "landing_adapter": env.unwrapped.stair_quiet_landing.contract(),
            "maximum_landing_joint_correction_rad": maximum_landing_correction,
            "maximum_landing_gain_scale_change": maximum_landing_gain_change,
            "read_only_physx_kinematics_maxima": kinematics_maxima,
            "physx_generalized_velocity_order": "root_COM_world_linear3_angular3_then_joint_velocity",
            "nonzero_landing_control_observed": activated,
            "nominal_action_parity_is_not_actuator_target_parity": True,
            "manifest": manifest,
            "performance_status": "not_evaluated",
            "real_acoustic_status": "not_measured",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({k: v for k, v in report.items() if k != "manifest"}, indent=2))
    except BaseException:
        # Kit shutdown can terminate Python before an uncaught exception prints.
        # Persist the original failure before closing; exit status alone is not a receipt.
        failure = {"status": "failed", "traceback": traceback.format_exc(), "optimizer_updates": 0}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(failure, indent=2) + "\n")
        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
