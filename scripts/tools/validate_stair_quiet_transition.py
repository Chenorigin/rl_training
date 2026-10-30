"""Bounded original-Isaac validation of Safe->Transition wiring; never learns.

Run with the Isaac environment Python, --headless --device cuda:0. The saved
snapshot is INITIALIZED ONLY, not a trained policy. No optimizer step is run.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"source/rl_training"))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--steps", type=int, default=350)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not 1 <= args.steps <= 500:
    parser.error("steps must be in [1,500]")
args.output.mkdir(parents=True, exist_ok=True)
started = time.monotonic()


def gpu_guard():
    free = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).splitlines()[0])
    rows = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], text=True)
    own = sum(int(row.split(",")[1]) for row in rows.splitlines() if int(row.split(",")[0]) == os.getpid())
    if free < 24576 or own > 8192:
        raise RuntimeError(f"Validation GPU budget exceeded: own={own}MiB free={free}MiB")
    return dict(own_MiB=own, free_MiB=free)


gpu_guard()
launcher = AppLauncher(args)
app, env, failed = launcher.app, None, False
try:
    import numpy as np
    import torch
    import gymnasium as gym
    from tensordict import TensorDict
    import rl_training.tasks
    from isaaclab_tasks.utils import load_cfg_from_registry
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from task_support.stair_quiet_v1_transition.runner import StairQuietTransitionRunner, sources, canonical
    from task_support.stair_quiet_v1_transition.course import course_state, validate_transition_counts
    from task_support.stair_quiet_v1_transition.observations import physical_sample
    from task_support.stair_quiet_v1_transition.settings import cases
    torch.set_num_threads(1)
    task = "Rough-Deeprobotics-M20-StairQuiet-v1-Transition"
    cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    cfg.scene.num_envs, cfg.seed, cfg.sim.device = 35, 42, args.device
    cfg.commands.base_velocity.debug_vis = False
    cfg.scene.height_scanner.debug_vis = False
    env = gym.make(task, cfg=cfg).unwrapped
    agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    train_cfg = agent_cfg.to_dict()
    # Match the training entrypoint if installed RSL uses its newer cfg schema.
    from importlib.metadata import version
    if int(version("rsl-rl-lib").split(".")[0]) >= 5:
        sys.path.insert(0, str(ROOT/"scripts/reinforcement_learning/rsl_rl"))
        from cli_args import convert_rsl_rl_cfg_dict
        train_cfg = convert_rsl_rl_cfg_dict(train_cfg)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = StairQuietTransitionRunner(wrapped, train_cfg, log_dir=None, device=args.device)
    runner.initialize_task(dict(init_actor_from=str(args.checkpoint.resolve())))
    source = torch.load(args.checkpoint, weights_only=True, map_location="cpu")
    source_manifest = source["m20_training_manifest"]
    assert source_manifest["noise_refinement"]["task"] == "stair_quiet_v1_safe"
    for kind in ("actor", "critic"):
        assert all(torch.equal(v, getattr(runner.alg, kind).state_dict()[k].cpu())
                   for k, v in source[kind+"_state_dict"].items())
    assert env.tail_stage == source_manifest["tail_curriculum"]["stage"]
    # learn() normally creates the writer. This inference-only run never calls
    # learn(), so explicitly disable it before exercising checkpoint saving.
    runner.logger.writer = None
    assert not runner.alg.optimizer.state and runner.alg.guard_updates == 0
    before = {k: v.detach().cpu().clone() for k, v in runner.alg.actor.state_dict().items()}
    snapshot = args.output/"safe_initialized_only.pt"
    runner.save(str(snapshot))
    source_before = sources()
    state, terrain = course_state(env), env.scene.terrain
    assert len(cases()) == 35 and terrain.cfg.terrain_generator.num_cols == 35
    ids = torch.arange(35, device=env.device)
    terrain.terrain_types[:] = ids
    terrain.terrain_levels[:] = 8  # one fixed physical width for all 35 cases
    origins = terrain.terrain_origins[terrain.terrain_levels, ids]
    terrain.env_origins[:] = origins
    env.scene.env_origins[:] = origins
    root = env.scene["robot"].data.default_root_state.clone()
    root[:, :3] += origins
    robot = env.scene["robot"]
    robot.write_root_state_to_sim(root)
    robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), torch.zeros_like(robot.data.default_joint_vel))
    robot.reset()
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.scene.update(dt=0.)
    env.episode_length_buf.zero_()
    env.tail_stage = 2
    state = course_state(env)
    state["protocol"].reset(ids, torch.zeros_like(ids))
    state["speed"][:] = .6
    env.tail_tracker.reset(ids, root[:, 0]-origins[:, 0])
    env.stair_quiet_tracker.reset(ids)
    env.stair_quiet_landing.reset(ids)
    if hasattr(env.tail_tracker, "last"):
        env.tail_tracker.last["crossed"][:] = False
    command = env.command_manager.get_term("base_velocity")
    command.time_left[:] = 100.
    command._apply_course(ids)
    obs = TensorDict(env.observation_manager.compute(update_history=True), batch_size=[35])
    for key, size in (("policy", 244), ("motion_control", 3), ("locomotion_scan", 187), ("safety", 48)):
        assert obs[key].shape == (35, size) and torch.isfinite(obs[key]).all()
    assert torch.allclose(state["height"][30:], torch.full_like(state["height"][30:], .17))
    assert torch.allclose(state["depth"][30:], torch.full_like(state["depth"][30:], .30))
    assert state["closed"][30:].all()
    active = torch.ones(35, dtype=torch.bool, device=env.device)
    outcomes = []
    original_reset = env._reset_idx
    def capture_reset(reset_ids):
        for i in reset_ids[active[reset_ids]].tolist():
            outcomes.append(dict(env=i, case=i, protocol=int(state["protocol"].mode[i]),
                success=bool(env.tail_tracker.success[i]), edge=bool(env.tail_tracker.edge_failed[i]),
                recovery=bool(env.tail_tracker.recovery_failed[i])))
        active[reset_ids] = False
        original_reset(reset_ids)
    env._reset_idx = capture_reset
    print("[VALIDATION_READY]", gpu_guard(), flush=True)
    traces = []
    with torch.inference_mode():
        for step in range(args.steps):
            if time.monotonic()-started > 265:
                raise TimeoutError("Validation wall budget exceeded")
            action = runner.alg.actor(obs, stochastic_output=True)
            assert torch.isfinite(action).all()
            obs, rewards, *_ = wrapped.step(action)
            assert torch.isfinite(rewards).all()
            assert all(torch.isfinite(obs[k]).all() for k in ("policy", "critic", "motion_control", "locomotion_scan", "safety"))
            d = physical_sample(env)
            traces.append(torch.cat((active[:, None].float(), d["root"], d["yaw"][:, None], command.vel_command_b), -1).cpu().numpy())
            if step%50 == 0:
                print(f"[PROGRESS] {step} active={int(active.sum())}", gpu_guard(), flush=True)
            if not bool(active.any()):
                break
    assert not runner.alg.optimizer.state and runner.alg.guard_updates == 0
    assert all(torch.equal(v, runner.alg.actor.state_dict()[k].cpu()) for k, v in before.items())
    validate_transition_counts(course_state(env))
    assert source_before == sources()
    runner.load(str(snapshot))
    assert not runner.alg.optimizer.state and runner.alg.guard_updates == 0
    np.savez_compressed(args.output/"closed_loop.npz", trace=np.stack(traces))
    receipt = dict(task=task, status="passed_wiring_and_bounded_inference", num_envs=35,
        physical_cases=canonical(cases()), physics_dt=env.physics_dt, policy_steps=step+1,
        actor_input_shapes={k: list(obs[k].shape) for k in ("policy", "motion_control", "locomotion_scan", "safety")},
        control_adapter=env.stair_quiet_landing.contract(), robot_usd=cfg.scene.robot.spawn.usd_path,
        material_bindings=terrain.contact_receipt, widths_m=[4.-8/3]*35,
        forced_case_geometry_tables_verified=True, optimizer_updates=0, actor_weights_unchanged=True,
        source_task="stair_quiet_v1_safe", source_iteration=source["iter"],
        complete_source_actor_and_critic_exact=True, inherited_material_stage=source_manifest["tail_curriculum"]["stage"],
        rewards_finite=True,
        strict_resume_verified=True, sources_unchanged=True, checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        first_episode_outcomes=outcomes, unfinished=active.nonzero().flatten().cpu().tolist(),
        snapshot=str(snapshot.resolve()), snapshot_status="safe_initialized_only_not_trained",
        GPU=gpu_guard(), wall_s=time.monotonic()-started,
        limitation="Finite wiring/inference smoke, not a trained-policy success benchmark; no optimizer update")
    (args.output/"isaac_validation.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+"\n")
    print("[VALIDATION_PASSED]", len(outcomes), "first episodes ended; no optimization", flush=True)
except BaseException as error:
    failed = True
    traceback.print_exc()
    (args.output/"isaac_validation.json").write_text(json.dumps(dict(status="failed", error=str(error), traceback=traceback.format_exc()), indent=2)+"\n")
finally:
    if env is not None:
        env.close()
    try:
        app.close()
    except SystemExit:
        pass
if failed:
    raise SystemExit(1)
