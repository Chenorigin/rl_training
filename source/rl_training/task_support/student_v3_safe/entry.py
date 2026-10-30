"""Task-specific coordination for the shared distillation CLI and backend."""
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
import json
import shutil
import subprocess

from .configuration import ROOT, TASK, CONFIG_PATH, TEACHER_GROUPS, ALGORITHM_CLASS, DeeproboticsM20StudentV3SafeDistillationCfg


def settings(args, saved):
    from m20_student_v1.configuration import StudentV1PerceptionCfg
    from m20_student_v1.contract import validate_contract
    from m20_student_v1.model_io import resolve_checkpoint
    from task_support.stair_quiet_v1.baselines import STUDENT, STUDENT_SHA256, checked_path
    from .teacher import teacher_manifest, teacher_config
    from .deployment import validate_resume
    if args.xr:
        raise ValueError("Safe student sensor timing requires XR disabled")
    args.task = TASK
    meta = validate_resume(saved) if args.resume else {}
    if args.resume and any(getattr(args, key) is not None for key in (
            "init_student", "config", "history_steps", "teacher_ratio", "teacher_ratio_end", "perception_level", "seed")):
        raise ValueError("Safe resume preserves observations/schedule; use --init-student for changed settings")
    training = deepcopy(meta.get("training_config", DeeproboticsM20StudentV3SafeDistillationCfg().to_dict()))
    perception = deepcopy(meta.get("perception", json.loads(CONFIG_PATH.read_text())["perception"]))
    if not args.resume:
        args.config = args.config or CONFIG_PATH
        supplied = json.loads(args.config.read_text())
        if set(supplied) - {"training", "perception", "history_steps"}:
            raise ValueError("Unknown Safe student configuration section")
        perception.update(supplied.get("perception", {}))
        for key, value in supplied.get("training", {}).items():
            if key not in training:
                raise ValueError(f"Unknown Safe training setting: {key}")
            if isinstance(training[key], dict):
                if not isinstance(value, dict) or set(value) - set(training[key]):
                    raise ValueError(f"Unknown Safe nested training setting: {key}")
                training[key].update(value)
            else:
                training[key] = value
        training["history_steps"] = supplied.get("history_steps", training["history_steps"])
        for arg, key in (("teacher_ratio", "teacher_ratio"), ("teacher_ratio_end", "teacher_ratio_end")):
            if getattr(args, arg) is not None:
                training["algorithm"][key] = getattr(args, arg)
        if args.seed is not None:
            training["seed"] = args.seed
        if args.perception_level is not None:
            training["algorithm"].update(noise_level_start=args.perception_level, noise_level_end=args.perception_level)
        perception["level"] = training["algorithm"]["noise_level_start"]
        args.init_student = resolve_checkpoint(args.init_student) if args.init_student else checked_path(STUDENT, STUDENT_SHA256)
    if args.history_steps is not None:
        training["history_steps"] = args.history_steps
    if training["history_steps"] != 4:
        raise ValueError("student_v3_safe preserves the 789D ABI: history_steps must be 4")
    if (training["obs_groups"] != {"student": ["policy"], "teacher": TEACHER_GROUPS}
            or training["algorithm"]["class_name"] != ALGORITHM_CLASS):
        raise ValueError("Safe student must consume policy only and use its reward-aware distillation algorithm")
    manifest = teacher_manifest(saved, bool(args.resume))
    expected = teacher_config(manifest)
    if args.resume and training["teacher"] != expected:
        raise ValueError("Safe resume teacher differs from its manifest")
    training["teacher"] = expected
    n = args.num_envs or meta.get("num_envs", 4 if args.validate_only else 64)
    if args.resume:
        if n != meta["num_envs"]:
            raise ValueError("Safe resume preserves num_envs; use --init-student for a new run")
        validate_contract(meta["contract"])
    perception = StudentV1PerceptionCfg(**perception)
    perception.validate()
    from .gait_reward import validate_settings
    from .gait_policy_gradient import validate_pg
    validate_settings(training["diagonal_reward"])
    validate_pg(training["algorithm"]["gait_policy_gradient"])
    from .terrain_geometry import validate_course
    validate_course(training['terrain_course'])
    if training["algorithm"]["gait_policy_gradient"]["action_delay_steps"] != perception.action_delay_steps:
        raise ValueError("Gait reward credit delay differs from the actual action queue")
    from m20_student_v1.stability import sensor_noise_std
    from .start_stop import validate_practice
    from task_support.stair_quiet_v1_safe.settings import SafeSettings
    tail_settings = SafeSettings(**manifest["noise_refinement"]["contract"]["environment"]["stair_tail"])
    validate_practice(training["idle_practice"], tail_settings.no_progress_timeout_s)
    if training["algorithm"].get("spatial_loss_weight", 0.):
        expected_noise = sensor_noise_std(perception)
        configured = training["algorithm"].get("stability_noise_std")
        if configured is not None and configured != expected_noise:
            raise ValueError("Stability perturbations differ from configured sensor units")
        training["algorithm"]["stability_noise_std"] = expected_noise
    args.log_root = (args.log_root or ROOT / "logs/rsl_rl/deeprobotics_m20_student_v3_safe").expanduser().resolve()
    args.run_name = args.run_name or "student_v3_safe"
    args.livestream = args.livestream if args.livestream is not None else 0
    return training, perception, n, meta, manifest


def sample_observations(n=2):
    import torch
    from tensordict import TensorDict
    return TensorDict({"policy": torch.zeros(n, 789), "teacher": torch.zeros(n, 244),
                       "motion_control": torch.zeros(n, 3), "locomotion_scan": torch.zeros(n, 187),
                       "safety": torch.zeros(n, 48)}, batch_size=[n])


def initialize(algorithm, saved, args):
    import torch
    from m20_student_v1.model_io import initialize_student
    if args.resume:
        algorithm.load(saved)
        return {"mode": "student_v3_safe_resume", "optimizer_updates": 0}
    algorithm.load(saved, {"teacher": True, "iteration": False}, True)
    source = torch.load(args.init_student, map_location="cpu", weights_only=True)
    state = source.get("student_state_dict", source.get("actor_state_dict", {}))
    if any(key.startswith(("residual_mlp.", "safety_mlp.")) for key in state):
        raise ValueError("--init-student requires plain student weights, not a Safe teacher")
    return initialize_student(algorithm.student, source)


def budget_for(training, n, iterations, perception, start, reserve):
    from m20_student_v1.output_storage import output_budget
    result = output_budget(training, n, iterations, perception, start, reserve)
    # Existing budget already includes raw command3 and the inherited residual.
    extra_rollout = n * training["num_steps_per_env"] * (187 + 48) * 4
    extra_weights = 4 * ((295 + 1) * 128 + (128 + 1) * 64 + (64 + 1) * 16)
    result["rollout_bytes"] += extra_rollout
    result["sample_buffer_bytes"] += extra_rollout
    from m20_student_v0.output_storage import OutputStorageCfg
    import math
    extra_checkpoint = math.ceil(extra_weights * OutputStorageCfg().checkpoint_margin)
    result["checkpoint_estimate_bytes"] += extra_checkpoint
    extra_output = (result["checkpoint_count"] + 2) * extra_checkpoint
    result["run_output_estimate_bytes"] += extra_output
    result["required_free_bytes"] += extra_output
    result["safe_teacher_extra_rollout_bytes"] = extra_rollout
    result["safe_teacher_extra_weight_bytes"] = extra_weights
    # Two 16D float arrays, masks, return/advantage and selected indices.
    pg_bytes = n*training["num_steps_per_env"]*(32*4+4+3*4+2*8)
    result["gait_policy_gradient_bytes"] = pg_bytes
    result["rollout_bytes"] += pg_bytes
    result["sample_buffer_bytes"] += pg_bytes
    return result


def preflight(args, saved, training, perception, n, source_path):
    import psutil
    import torch
    from m20_student_v1.distillation import StudentV1Distillation, validate_resume_checkpoint
    from m20_student_v1.model_io import file_hash, actor_from_state
    from m20_student_v0.output_storage import require_disk_space
    from .teacher import SafeTeacherModel, SafeResidualActor
    torch.set_num_threads(1)
    torch.manual_seed(training["seed"])
    cfg = deepcopy(training)
    cfg["num_steps_per_env"] = 1
    cfg["multi_gpu"] = None
    from types import SimpleNamespace
    algorithm = StudentV1Distillation.construct_algorithm(sample_observations(), SimpleNamespace(num_envs=2, num_actions=16), cfg, "cpu")
    if args.resume:
        rollout = n * training["num_steps_per_env"]
        validate_resume_checkpoint(saved, 789, rollout * training["algorithm"]["replay_rollouts"],
                                   rollout * training["algorithm"]["hard_replay_rollouts"], training["algorithm"]["teacher_decay_updates"])
        # The tiny preflight storage differs deliberately from the real run.
        algorithm.student.load_state_dict(saved["student_state_dict"], strict=True)
        algorithm.teacher.load_state_dict(saved["teacher_state_dict"], strict=True)
        initialization = {"mode": "student_v3_safe_resume", "optimizer_updates": 0}
    else:
        initialization = initialize(algorithm, saved, args)
        source = torch.load(args.init_student, map_location="cpu", weights_only=True)
        source_state = source.get("student_state_dict", source.get("actor_state_dict"))
        reference = actor_from_state(source_state, source_state["mlp.0.weight"].shape[1], training["student"])
        samples = torch.randn(8, 789)
        with torch.inference_mode():
            torch.testing.assert_close(algorithm.student.mlp(samples), reference(samples[:, :reference[0].in_features]), atol=1e-4, rtol=1e-5)
    obs = sample_observations(4).apply(lambda value: torch.randn_like(value))
    obs["motion_control"][0] = 0.
    packed = SafeTeacherModel.pack(obs)
    model_cfg = deepcopy(training["teacher"])
    model_cfg.pop("class_name")
    reference = SafeResidualActor(packed, {"actor": ["policy"]}, "actor", 16, **model_cfg)
    reference.load_state_dict(algorithm.teacher.state_dict(), strict=True)
    with torch.inference_mode():
        teacher_error = float((algorithm.teacher(obs) - reference(packed)).abs().max())
        exported = torch.jit.script(algorithm.student.as_jit())
        student_error = float((exported(obs["policy"]) - algorithm.student(obs)).abs().max())
    import math
    if teacher_error != 0. or not math.isfinite(student_error) or student_error > 1e-6 or any(p.requires_grad for p in algorithm.teacher.parameters()):
        raise RuntimeError("Safe teacher/student parity or freeze check failed")
    start = saved.get("student_v1", {}).get("num_updates", 0) if args.resume else 0
    budget = budget_for(training, n, args.max_iterations or training["max_iterations"], perception, start, args.min_free_disk_gb)
    required = budget["min_free_disk_bytes"] + 128 * 1024**2 if args.validate_only else budget["required_free_bytes"]
    disk_free = require_disk_space(args.log_root, required)
    from dataclasses import replace
    from .terrain_geometry import geometries, course_mesh
    from task_support.stair_quiet_v1_safe.settings import SafeSettings
    terrain_settings = replace(SafeSettings(), tile_length=training['terrain_course']['tile_length'])
    geometry_bytes = 0
    for geometry in geometries(training['terrain_course']):
        mesh = course_mesh(geometry, terrain_settings)
        geometry_bytes += mesh.vertices.nbytes + mesh.faces.nbytes
    # Five materials, all width rows; reserve copies/BVH/cooked collision data.
    budget['terrain_mesh_reserve_bytes'] = geometry_bytes*5*len(terrain_settings.widths_m)*8
    allowance = 8 * 1024**3 + n * 4 * 1024**2 + budget["sample_buffer_bytes"] + budget['terrain_mesh_reserve_bytes']
    host = psutil.virtual_memory()
    if host.total - host.available + allowance > .85 * host.total:
        raise RuntimeError("Insufficient whole-host memory headroom")
    report = {"task": TASK, "actor_dim": 789, "teacher_input_groups": {"teacher": 244, "motion_control": 3, "locomotion_scan": 187, "safety": 48},
              "teacher_reference_max_action_error": teacher_error, "export_max_action_error": student_error,
              "teacher_frozen": True, "initialization": initialization, "budget": budget,
              "planned_increment_bytes": allowance, "host_available_bytes": host.available, "host_total_bytes": host.total,
              "disk_free_bytes": disk_free, "source_path": str(source_path), "source_sha256": file_hash(source_path),
              "num_envs": n, "device": args.device, "occupancy_cap_fraction": .85,
              "basis": "8 GiB simulator + 4 MiB/env + actual replay/map/rollout tensors + 8x full terrain mesh bytes; engineering reserve",
              "simulator_started": False, "optimizer_updates": 0, "trained": False}
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        free, total = torch.cuda.mem_get_info(torch.device(args.device))
        if total - free + allowance > .85 * total:
            raise RuntimeError("Insufficient whole-GPU headroom")
        report.update(gpu_free_bytes=free, gpu_total_bytes=total)
        report["nvidia_smi"] = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu", "--format=csv"], text=True).strip()
    if args.preflight_report:
        args.preflight_report.parent.mkdir(parents=True, exist_ok=True)
        args.preflight_report.write_text(json.dumps(report, indent=2) + "\n")
    print("STUDENT_V3_SAFE_PREFLIGHT=" + json.dumps(report), flush=True)
    return report


def train_or_validate(args, saved, training, perception, n, meta, manifest, source_path, report, launch_app):
    import torch
    app = launch_app(args)
    env = None
    failed = False
    try:
        import gymnasium as gym
        import importlib
        importlib.import_module("rl_training.tasks")
        from isaaclab_tasks.utils import parse_env_cfg
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaaclab.utils.io import dump_yaml
        from m20_student_v1.model_io import file_hash
        from m20_student_v1.contract import observation_contract
        from task_support.stair_quiet_v1.deployment import execution_contract
        from task_support.stair_quiet_v1_safe.settings import SafeSettings
        from task_support.stair_quiet_v1.landing import LandingSettings
        from .environment import initialize_course
        from .runner import StudentV3SafeRunner
        from .deployment import provenance_for, export_bundle
        cfg = parse_env_cfg(TASK, device=args.device, num_envs=n)
        cfg.student_perception, cfg.student_history_steps = perception, 4
        cfg.student_idle_practice = deepcopy(training["idle_practice"])
        cfg.student_diagonal_reward = deepcopy(training["diagonal_reward"])
        cfg.rewards.student_diagonal_alternation.weight = cfg.student_diagonal_reward["weight"]
        cfg.motion_teacher_tuning = deepcopy(manifest["stair_tuning"])
        cfg.seed = training["seed"]
        control = execution_contract({"teacher_manifest": manifest}, required=True)
        cfg.landing_control = LandingSettings(**control["settings"])
        cfg.stair_tail = SafeSettings(**manifest["noise_refinement"]["contract"]["environment"]["stair_tail"])
        # Terrain subconfigs also hold the settings object; keep one ledger.
        for case in cfg.scene.terrain.terrain_generator.sub_terrains.values():
            case.settings = cfg.stair_tail
        from .terrain_importer import configure_course
        configure_course(cfg,training['terrain_course'])
        if args.validate_only:
            cfg.scene.terrain.visual_material = None
            cfg.scene.sky_light = None
            cfg.commands.base_velocity.debug_vis = False
            cfg.sim.enable_scene_query_support = True
            # Never shrink the real width/case matrix for a smoke check.
        run_dir = args.log_root / (datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f") + "_" + args.run_name)
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "resource_preflight.json").write_text(json.dumps(report, indent=2) + "\n")
        env = RslRlVecEnvWrapper(gym.make(TASK, cfg=cfg), clip_actions=training["clip_actions"])
        budget = report["budget"]
        runner = StudentV3SafeRunner(env, deepcopy(training), log_dir=None if args.validate_only else str(run_dir), device=args.device,
                                   min_free_disk_bytes=budget["min_free_disk_bytes"], checkpoint_estimate_bytes=budget["checkpoint_estimate_bytes"])
        if args.resume:
            runner.load(str(source_path), map_location=args.device)
            runner.current_learning_iteration = runner.alg.num_updates
        else:
            initialize(runner.alg, saved, args)
            initialize_course(env.unwrapped, manifest)
        provenance = provenance_for({
            "version": "v1", "task": TASK, "task_version": "student_v3_safe", "num_envs": n, "seed": cfg.seed,
            "teacher_manifest": manifest, "teacher_checkpoint": meta.get("teacher_checkpoint", str(source_path)),
            "teacher_sha256": meta.get("teacher_sha256", file_hash(source_path)),
            "init_student_checkpoint": str(args.init_student) if args.init_student else meta.get("init_student_checkpoint"),
            "init_student_sha256": file_hash(args.init_student) if args.init_student else meta.get("init_student_sha256"),
            "resume_checkpoint": str(source_path) if args.resume else None,
            "training_config": deepcopy(training), "perception": asdict(perception), "contract": observation_contract(4),
            "command_feedback": {"version": "legacy_v1_v3_5", "tuning": manifest["stair_tuning"]},
            "teacher_action_target": "complete frozen Safe actor including residual_mlp and safety_mlp",
            "student_input": "sensor policy only; safety48 and longitudinal187 are teacher-only",
            "course_resume": "completed course counters and next-reset targets; physics and unsaved replay restart",
            "sensor_parameters_calibrated_on_hardware": False,
        }, control)
        runner.alg.provenance = provenance
        for name, value in (("provenance", provenance), ("observation_contract", provenance["contract"])):
            (run_dir / (name + ".json")).write_text(json.dumps(value, indent=2) + "\n")
        dump_yaml(str(run_dir / "env.yaml"), cfg)
        dump_yaml(str(run_dir / "distillation.yaml"), training)
        print("STUDENT_V3_SAFE_RUN_DIR=" + str(run_dir), flush=True)
        obs = env.get_observations()
        if args.validate_only:
            import numpy as np
            from .start_stop import environment_masks
            from .terrain_validation import collision_receipt
            from .terrain_course import course_state, FAMILIES
            terrain_receipt = collision_receipt(env.unwrapped)
            terrain_samples = {name: 0 for name in FAMILIES}
            terrain_contacts = {name: {'loaded_foot_samples': 0, 'raised_tread_foot_samples': 0} for name in FAMILIES}
            before = {k: v.clone() for k, v in runner.alg.student.state_dict().items()}
            teacher_before = {k: v.clone() for k, v in runner.alg.teacher.state_dict().items()}
            updates = runner.alg.num_updates
            # Keep the diagnostic trace bounded independently of a caller's
            # requested environment/step count. Invariants still check all envs.
            trace_envs, trace_steps = min(n, 4), min(args.validation_steps, 256)
            trace = {k: [] for k in ("policy", "labels", "actions", "command", "episode_step",
                                    "selected", "scheduled_pause", "done", "phase", "gait_reward", "gait_eligible",
                                    "student_execution", "gait_sampled")}
            pg_samples, pg_kl, gait_positive, gait_eligible = 0, 0., 0, 0
            gait_gates = {key: 0 for key in ("safe", "diagonal", "sustained", "cycling", "switch")}
            gait_trace = {key: [] for key in ("x", "supported", "airborne", "eligible", "safe", "reward", "done", "reset_x")}
            def inspect_rollout():
                runner.alg.prepare_policy_gradient()
                loss, diagnostic = runner.alg.auxiliary_loss()
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite gait policy-gradient objective")
                result = (len(runner.alg.pg_ids), float(diagnostic["gait_pg_kl"]))
                runner.alg.storage.clear()
                runner.alg.rollout_samples = runner.alg.executed_teacher = runner.alg.executed_total = 0
                return result
            with torch.inference_mode():
                for _ in range(args.validation_steps):
                    family = course_state(env.unwrapped)['family']
                    for code, name in enumerate(FAMILIES):
                        terrain_samples[name] += int((family == code).sum())
                    action = runner.alg.act(obs)
                    labels = runner.alg.transition.privileged_actions
                    selected, paused = environment_masks(env.unwrapped)
                    command = env.unwrapped.command_manager.get_command("base_velocity")
                    if paused.any() and command[paused].abs().max() != 0:
                        raise RuntimeError("Student idle-practice command was not applied")
                    if len(trace["policy"]) < trace_steps:
                        for key, value in (("policy", obs["policy"]), ("labels", labels), ("actions", action),
                                           ("command", command), ("episode_step", env.unwrapped.episode_length_buf),
                                           ("selected", selected), ("scheduled_pause", paused),
                                           ("student_execution", runner.alg.pg_student[runner.alg.storage.step]),
                                           ("gait_sampled", runner.alg.pg_sampled[runner.alg.storage.step])):
                            trace[key].append(value[:trace_envs].detach().cpu().numpy().copy())
                    if labels.shape != (n, 16) or not torch.isfinite(labels).all():
                        raise RuntimeError("Invalid Safe teacher labels")
                    if not torch.equal(obs["teacher"][:, 6:9], obs["policy"][:, 6:9]):
                        raise RuntimeError("Teacher/student effective commands differ")
                    obs, reward, done, extras = env.step(action)
                    runner.alg.process_env_step(obs, reward, done, extras)
                    gait = extras["student_diagonal"]
                    from .terrain_observations import physical_sample, safety_geometry
                    physical = physical_sample(env.unwrapped)
                    support_geometry = safety_geometry(physical, env.unwrapped.cfg.stair_tail)
                    for code, name in enumerate(FAMILIES):
                        active = (family == code) & ~done.bool()
                        loaded = support_geometry['supported'] & active[:, None]
                        terrain_contacts[name]['loaded_foot_samples'] += int(loaded.sum())
                        terrain_contacts[name]['raised_tread_foot_samples'] += int((loaded & (support_geometry['top'] > .05)).sum())
                    gait_positive += int((gait["reward"] > 0).sum())
                    gait_eligible += int(gait["eligible"].sum())
                    for key in gait_gates:
                        gait_gates[key] += int((gait[key] & gait["eligible"]).sum())
                    if len(gait_trace["done"]) < 4096:
                        for key in gait_trace:
                            if key == "reset_x":
                                value = (env.unwrapped.scene["robot"].data.root_pos_w[:, 0]
                                         - env.unwrapped.scene.env_origins[:, 0])
                            else:
                                value = done if key == "done" else gait[key]
                            gait_trace[key].append(value[:trace_envs].detach().cpu().numpy().copy())
                    if len(trace["done"]) < trace_steps:
                        trace["done"].append(done[:trace_envs].detach().cpu().numpy().copy())
                        trace["phase"].append(extras["student_v1"]["phase"][:trace_envs].detach().cpu().numpy().copy())
                        trace["gait_reward"].append(gait["reward"][:trace_envs].detach().cpu().numpy().copy())
                        trace["gait_eligible"].append(gait["eligible"][:trace_envs].detach().cpu().numpy().copy())
                    if not all(torch.isfinite(value).all() for value in obs.values()) or not torch.isfinite(reward).all():
                        raise RuntimeError("Nonfinite Safe inference state")
                    if not torch.equal(obs["policy"], obs["critic"]):
                        raise RuntimeError("Safe student critic placeholder differs")
                    if runner.alg.storage.step == runner.alg.storage.num_transitions_per_env:
                        count, kl = inspect_rollout()
                        pg_samples += count
                        pg_kl = max(pg_kl, kl)
                if runner.alg.storage.step:
                    count, kl = inspect_rollout()
                    pg_samples += count
                    pg_kl = max(pg_kl, kl)
            result = {"steps": args.validation_steps, "optimizer_updates": runner.alg.num_updates - updates,
                      "student_weights_unchanged": all(torch.equal(v, runner.alg.student.state_dict()[k]) for k, v in before.items()),
                      "teacher_weights_unchanged": all(torch.equal(v, runner.alg.teacher.state_dict()[k]) for k, v in teacher_before.items()),
                      "observation_shapes": {k: list(v.shape) for k, v in obs.items()}, "trained": False,
                      "physics_substeps": env.unwrapped._sim_step_counter,
                      "stability_config": {k: training["algorithm"][k] for k in (
                          "temporal_loss_weight", "spatial_loss_weight", "stability_batch_size",
                          "spatial_noise_scale", "stability_noise_std")},
                      "practice_config": deepcopy(training["idle_practice"]),
                      "motion_retention": None if runner.alg.retention is None else runner.alg.retention.receipt,
                      "diagonal_reward": deepcopy(training["diagonal_reward"]),
                      "gait_policy_gradient": deepcopy(training["algorithm"]["gait_policy_gradient"]),
                      "gait_eligible_samples": gait_eligible, "gait_positive_samples": gait_positive,
                      "gait_gate_eligible_samples": gait_gates,
                      "gait_trace_steps": len(gait_trace["done"]), "step_dt": env.unwrapped.step_dt,
                      "gait_pg_samples": pg_samples, "gait_pg_max_kl_without_updates": pg_kl,
                      "collection_mode": "actual teacher/student mixture and gated exploration, no optimizer updates",
                      "trace_envs": trace_envs, "trace_steps": trace_steps,
                      "scheduled_pause_samples": int(np.stack(trace["scheduled_pause"]).sum()),
                      "practice_command_class": type(env.unwrapped.command_manager.get_term("base_velocity")).__name__,
                      "course": {"stage": int(env.unwrapped.tail_stage), "real_width_rows": cfg.scene.terrain.terrain_generator.num_rows,
                                 "family_samples": terrain_samples, "physical_contacts": terrain_contacts, "collision": terrain_receipt,
                                 "settings": deepcopy(training['terrain_course'])}}
            if result["optimizer_updates"] or not result["student_weights_unchanged"] or not result["teacher_weights_unchanged"]:
                raise RuntimeError("Safe inference updated weights")
            np.savez_compressed(run_dir / "validation_trace.npz", **{k: np.stack(v) for k, v in trace.items()})
            np.savez_compressed(run_dir / "gait_trace.npz", **{k: np.stack(v) for k, v in gait_trace.items()})
            # Also exercise real runner persistence without optimizer updates.
            runner.save(str(run_dir / "validation_checkpoint.pt"))
            runner.load(str(run_dir / "validation_checkpoint.pt"), map_location=args.device)
            export_bundle(runner, run_dir / "exported", runner.alg.provenance)
            (run_dir / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
            print("STUDENT_V3_SAFE_VALIDATION=" + json.dumps(result), flush=True)
            return
        for relative in ("source/rl_training/task_support", "source/rl_training/m20_student_v1", "source/rl_training/m20_student_v0",
                         "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity"):
            shutil.copytree(ROOT / relative, run_dir / "source_snapshot" / relative,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "datasets"))
        shutil.copy2(ROOT / "scripts/reinforcement_learning/rsl_rl/distill_student.py", run_dir / "source_snapshot/distill_student.py")
        shutil.copy2(args.config or CONFIG_PATH, run_dir / "source_snapshot/student_v3_safe.json")
        runner.add_git_repo_to_log(__file__)
        runner.learn(args.max_iterations or training["max_iterations"], init_at_random_ep_len=False)
        if not args.no_export:
            export_bundle(runner, run_dir / "exported", runner.alg.provenance)
    except BaseException:
        # Kit shutdown can terminate the interpreter before Python's default
        # exception hook runs. Preserve the real failure before closing Kit.
        import traceback
        import sys
        failed = True
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    finally:
        try:
            if env is not None:
                env.close()
            app.close()
        finally:
            if failed:
                # Kit installs an exception hook which may turn uncaught
                # Python errors into status 0 after shutdown. Fail explicitly.
                import os
                os._exit(1)


def run(args, saved, source_path, launch_app):
    training, perception, n, meta, manifest = settings(args, saved)
    report = preflight(args, saved, training, perception, n, source_path)
    if not args.preflight_only:
        train_or_validate(args, saved, training, perception, n, meta, manifest, source_path, report, launch_app)
