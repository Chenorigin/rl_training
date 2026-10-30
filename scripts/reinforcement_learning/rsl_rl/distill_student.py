#!/usr/bin/env python3
"""M20 v0/v1/v3/v3.5 distillation. --preflight-only never launches Isaac or optimization.

--task selects the environment and v0/v1 algorithm. --resume restores the
matching version; --init-student and --history-steps are available for v1.
The user starts training explicitly by invoking this entrypoint without either
--preflight-only or --validate-only. See docs/m20_student_v1.md.
"""
import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/"source/rl_training"))

from m20_student_v1.configuration import DeeproboticsM20StudentV1DistillationCfg, StudentV1PerceptionCfg
from m20_student_v1.v3_configuration import V3_CONFIG_PATH
from m20_student_v1.v3_5_configuration import V3_5_CONFIG_PATH
from m20_student_v1.contract import observation_contract, student_dim, validate_contract
from m20_student_v1.model_io import file_hash, resolve_checkpoint, initialize_student, actor_from_state
from m20_student_v1.output_storage import output_budget
from m20_student_v0.output_storage import require_disk_space, OutputStorageCfg
from m20_student_v0.output_storage import output_budget as output_budget_v0
from m20_student_v0.configuration import DeeproboticsM20StudentV0DistillationCfg
from m20_student_v0.perception import StudentV0PerceptionCfg
from task_support.stair_quiet_v1.deployment import STUDENT_TASK as QUIET_STUDENT_TASK
from task_support.student_v3_safe.configuration import TASK as SAFE_STUDENT_TASK


def motion_task(task):
    return task in ("Rough-Deeprobotics-M20-Student-v3.5", QUIET_STUDENT_TASK)

TASK_VERSIONS = {
    SAFE_STUDENT_TASK: "v1",
    "Rough-Deeprobotics-M20-Student-v0": "v0",
    "Rough-Deeprobotics-M20-Student-v1": "v1",
    # V3 retains the v1 checkpoint/observation format and distillation backend.
    "Rough-Deeprobotics-M20-Student-v3": "v1",
    "Rough-Deeprobotics-M20-Student-v3.5": "v1",
    QUIET_STUDENT_TASK: "v1",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--teacher-checkpoint", type=Path)
    source.add_argument("--resume", type=Path)
    parser.add_argument("--init-student", type=Path, help="v0/v1 checkpoint, or run directory selecting latest complete model_N.pt; weights only")
    parser.add_argument("--task", choices=tuple(TASK_VERSIONS))
    parser.add_argument("--config", type=Path, help="JSON overrides: perception, training, history_steps")
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--history-steps", type=int)
    parser.add_argument("--teacher-ratio", type=float)
    parser.add_argument("--teacher-ratio-end", type=float)
    parser.add_argument("--perception-level", type=float, help="Sets fixed start/end severity; hardware timing and FOV stay configured")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--log-root", type=Path, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--min-free-disk-gb", type=float, default=2.)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight-only", action="store_true", help="CPU weights/contract/parity + resource budget only; no simulator/optimizer update")
    mode.add_argument("--validate-only", action="store_true", help="Isaac inference only; no replay collection or optimizer updates")
    parser.add_argument("--preflight-report", type=Path)
    parser.add_argument("--validation-steps", type=int, default=8)
    parser.add_argument("--no-export", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--enable-cameras", "--enable_cameras", dest="enable_cameras", action="store_true")
    parser.add_argument("--livestream", type=int, choices=(0, 1, 2), default=None)
    parser.add_argument("--experience", default="")
    parser.add_argument("--rendering_mode", choices=("performance", "balanced", "quality"))
    parser.add_argument("--kit_args", default="")
    parser.add_argument("--xr", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--info", action="store_true")
    parser.add_argument("--anim_recording_enabled", action="store_true")
    parser.add_argument("--anim_recording_start_time", type=float, default=0.)
    parser.add_argument("--anim_recording_stop_time", type=float, default=10.)
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--cpu", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.cpu:
        parser.error("Use --device cpu instead of --cpu")
    for name in ("num_envs", "max_iterations", "validation_steps"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"{name} must be positive")
    for name in ("teacher_ratio", "teacher_ratio_end", "perception_level"):
        value = getattr(args, name)
        if value is not None and not 0 <= value <= 1:
            parser.error(f"{name} must be in [0,1]")
    if not 0 < args.min_free_disk_gb < float("inf"):
        parser.error("min-free-disk-gb must be finite and positive")
    if args.distributed or int(os.environ.get("WORLD_SIZE", "1")) > 1:
        parser.error("Student distillation supports one process/GPU")
    if args.run_name is not None and (Path(args.run_name).name != args.run_name or not args.run_name):
        parser.error("run-name must be a single directory label")
    return args


def settings_v1(args, saved):
    if args.resume and any(value is not None for value in (args.init_student, args.config, args.history_steps,
                                                          args.teacher_ratio, args.teacher_ratio_end, args.perception_level, args.seed)):
        raise ValueError("Resume preserves v1 observations/dataset/schedule; use --init-student for a changed configuration")

    meta = saved.get("student_v1", {}).get("provenance", {}) if args.resume else {}
    if args.resume and not meta:
        raise ValueError("--resume requires a v1 checkpoint with provenance")
    training = deepcopy(meta.get("training_config", DeeproboticsM20StudentV1DistillationCfg().to_dict()))
    perception = deepcopy(meta.get("perception", asdict(StudentV1PerceptionCfg())))
    if args.config:
        supplied = json.loads(args.config.read_text())
        if set(supplied)-{"training", "perception", "history_steps"}:
            raise ValueError("Unknown v1 configuration section")
        perception.update(supplied.get("perception", {}))
        for key, value in supplied.get("training", {}).items():
            if key not in training:
                raise ValueError(f"Unknown training setting: {key}")
            if isinstance(training[key], dict):
                if set(value)-set(training[key]):
                    raise ValueError(f"Unknown nested training setting in {key}")
                training[key].update(value)
            else:
                training[key] = value
        if "history_steps" in supplied:
            training["history_steps"] = supplied["history_steps"]
    if args.history_steps is not None:
        training["history_steps"] = args.history_steps
    if args.seed is not None:
        training["seed"] = args.seed
    for argument, setting in (("teacher_ratio", "teacher_ratio"), ("teacher_ratio_end", "teacher_ratio_end")):
        if getattr(args, argument) is not None:
            training["algorithm"][setting] = getattr(args, argument)
    if args.perception_level is not None:
        perception["level"] = args.perception_level
        training["algorithm"].update(noise_level_start=args.perception_level, noise_level_end=args.perception_level)
    elif not args.resume:
        perception["level"] = training["algorithm"]["noise_level_start"]
    cfg = StudentV1PerceptionCfg(**perception)
    cfg.validate()
    student_dim(training["history_steps"])
    if args.task == QUIET_STUDENT_TASK and not args.resume:
        training["experiment_name"] = "deeprobotics_m20_stair_quiet_v1_student"
    teacher_groups = ['teacher', 'motion_control'] if args.resume and motion_task(args.task) else ['teacher']
    if training["obs_groups"] != {"student": ["policy"], "teacher": teacher_groups}:
        raise ValueError("V1 actor may only consume the sensor policy observation")
    if training["algorithm"]["class_name"] != "m20_student_v1.distillation.StudentV1Distillation":
        raise ValueError("Unexpected v1 algorithm")
    task = args.task or meta.get("task", "Rough-Deeprobotics-M20-Student-v1")
    if motion_task(task):
        from m20_student_v1.v3_5_teacher import teacher_manifest, teacher_config
        if training['history_steps'] != 4:
            raise ValueError('V3.5 preserves the 789D student input: history_steps must be 4')
        manifest = teacher_manifest(saved, bool(args.resume), allow_quiet=task == QUIET_STUDENT_TASK)
        expected = teacher_config(manifest)
        if args.resume and training['teacher'] != expected:
            raise ValueError('V3.5 resume teacher configuration differs from its manifest')
        training['teacher'] = expected
        training['obs_groups']['teacher'] = ['teacher', 'motion_control']
    n = args.num_envs or meta.get("num_envs", 4 if args.validate_only else 64)
    # Environment count is governed by the measured resource preflight, not an
    # arbitrary fixed ceiling. Full resume keeps replay capacities unchanged.
    if args.resume and n != meta["num_envs"]:
        raise ValueError("Resume preserves num_envs; use --init-student for a new environment count")
    if args.resume:
        if task != meta["task"]:
            raise ValueError("Resume task differs from checkpoint")
        validate_contract(meta["contract"])
    return training, cfg, task, n, meta


def preflight_v1(args, saved, training, perception, n, source_path):
    import psutil
    import torch
    from tensordict import TensorDict
    from rsl_rl.models import MLPModel
    from m20_student_v1.distillation import StudentV1Distillation, validate_resume_checkpoint
    from rsl_rl.storage import RolloutStorage
    torch.set_num_threads(1)
    torch.manual_seed(training["seed"])
    dim = student_dim(training["history_steps"])
    obs = TensorDict({"policy": torch.zeros(2, dim), "teacher": torch.zeros(2, 244)}, batch_size=[2])
    motion = motion_task(args.task)
    model_cfg = deepcopy(training)
    if model_cfg['student'].pop('class_name') != 'rsl_rl.models.MLPModel':
        raise ValueError('Student must remain MLPModel with explicit optional history')
    student = MLPModel(obs, training["obs_groups"], "student", 16, **model_cfg["student"])
    teacher_class = MLPModel
    teacher_name = model_cfg['teacher'].pop('class_name')
    if motion:
        from m20_student_v1.v3_5_teacher import MotionTeacherModel, TEACHER_CLASS
        if teacher_name != TEACHER_CLASS:
            raise ValueError('V3.5 requires the complete motion residual teacher')
        teacher_class = MotionTeacherModel
        obs['motion_control'] = torch.zeros(2, 3)
    elif teacher_name != 'rsl_rl.models.MLPModel':
        raise ValueError('Legacy distillation requires the ordinary MLP teacher')
    teacher = teacher_class(obs, training["obs_groups"], "teacher", 16, **deepcopy(model_cfg["teacher"]))
    # Tiny storage validates actual installed algorithm arguments, without act/update/learn.
    algorithm_cfg = deepcopy(training["algorithm"])
    algorithm_cfg.pop("class_name")
    algorithm = StudentV1Distillation(student, teacher, RolloutStorage("distillation", 2, 1, obs, [16], "cpu"), **algorithm_cfg)
    if args.resume:
        rollout = n*training["num_steps_per_env"]
        validate_resume_checkpoint(saved, dim, rollout*training["algorithm"]["replay_rollouts"],
                                   rollout*training["algorithm"]["hard_replay_rollouts"],
                                   training["algorithm"]["teacher_decay_updates"])
        student.load_state_dict(saved["student_state_dict"], strict=True)
        teacher.load_state_dict(saved["teacher_state_dict"], strict=True)
        initialization = {"mode": "v1_resume", "optimizer_updates_this_preflight": 0}
    else:
        algorithm.load(saved, {"teacher": True, "iteration": False}, True)
        initialization_source = saved
        if args.init_student:
            initialization_source = torch.load(resolve_checkpoint(args.init_student), map_location="cpu", weights_only=True)
        if motion:
            from m20_student_v1.v3_5_teacher import initialization_source as motion_initialization
            initialization_source = motion_initialization(saved, initialization_source if args.init_student else None)
        initialization = initialize_student(student, initialization_source)
        if motion and not args.init_student:
            initialization['mode'] = 'frozen_teacher_base_only; labels use complete residual teacher'
        state = initialization_source.get("student_state_dict", initialization_source.get("actor_state_dict"))
        source_dim = state["mlp.0.weight"].shape[1]
        reference = actor_from_state(state, source_dim, training["student"])
        samples = torch.randn(16, dim)
        with torch.inference_mode():
            expected = reference(samples[:, :source_dim])
            actual = student(TensorDict({"policy": samples}, batch_size=[16]))
        error = float((actual-expected).abs().max())
        # Padding the first GEMM changes its float32 accumulation order. Check
        # mathematical equivalence in float64 and report the actual FP32 error.
        with torch.inference_mode():
            reference64 = deepcopy(reference).double()
            student64 = deepcopy(student).double()
            expected64 = reference64(samples[:, :source_dim].double())
            actual64 = student64(TensorDict({"policy": samples.double()}, batch_size=[16]))
        error64 = float((actual64-expected64).abs().max())
        if not torch.allclose(actual64, expected64, atol=1e-10, rtol=1e-10):
            raise RuntimeError(f"Weights-only initialization changed float64 source actions: {error64}")
        if not torch.allclose(actual, expected, atol=1e-4, rtol=1e-5):
            raise RuntimeError(f"Weights-only initialization changed source actions: {error}")
        initialization["max_source_action_error"] = error
        initialization["float64_max_source_action_error"] = error64
        initialization["float32_parity_tolerance"] = {"atol": 1e-4, "rtol": 1e-5}
        if args.init_student:
            initialization.update(checkpoint=str(args.init_student), sha256=file_hash(args.init_student))
    with torch.inference_mode():
        exported = torch.jit.script(student.as_jit())
        samples = torch.randn(16, dim)
        error = float((exported(samples)-student(TensorDict({"policy": samples}, batch_size=[16]))).abs().max())
    if error > 1e-6:
        raise RuntimeError("TorchScript parity failed")
    teacher_check = {}
    if motion:
        from m20_student_v1.v3_5_teacher import _module
        packed = TensorDict({'policy': torch.randn(4, 244),
                             'motion_control': torch.tensor([[0., 0., 0.], [.5, 0., 0.], [0., 0., .6], [-.5, 0., 0.]])}, [4])
        reference = _module.MotionResidualMLPModel(packed, {'actor': ['policy']}, 'actor', 16, **model_cfg['teacher'])
        reference.load_state_dict(teacher.state_dict(), strict=True)
        with torch.inference_mode():
            actual = teacher(TensorDict({'teacher': packed['policy'], 'motion_control': packed['motion_control']}, [4]))
            reference_error = float((actual-reference(packed)).abs().max())
        if reference_error != 0.:
            raise RuntimeError('V3.5 teacher adapter differs from the original residual actor')
        teacher_check = {'teacher_kind': 'idle_stop_motion_v1', 'motion_control_dim': 3,
                         'teacher_reference_max_action_error': reference_error,
                         'teacher_frozen': not any(p.requires_grad for p in teacher.parameters())}
    start = saved.get("student_v1", {}).get("num_updates", 0) if args.resume else 0
    iterations = args.max_iterations or training["max_iterations"]
    budget = output_budget(training, n, iterations, perception, start, args.min_free_disk_gb)
    required = budget["min_free_disk_bytes"]+128*1024**2 if args.validate_only else budget["required_free_bytes"]
    free_disk = require_disk_space(args.log_root, required)
    allowance = 8*1024**3+n*4*1024**2+budget["sample_buffer_bytes"]
    host = psutil.virtual_memory()
    report = {"version": "v1", "task_version": ('stair_quiet_v1' if args.task == QUIET_STUDENT_TASK else ('v3.5' if motion else 'legacy')), **teacher_check,
              "source_path": str(source_path), "source_sha256": file_hash(source_path),
              "initialization": initialization, "actor_dim": dim, "teacher_dim": 244,
              "export_max_action_error": error, "num_envs": n, "device": args.device,
              "host_available_bytes": host.available, "host_total_bytes": host.total,
              "disk_free_bytes": free_disk, "planned_increment_bytes": allowance,
              "budget": budget, "occupancy_cap_fraction": .85,
              "basis": "8 GiB simulator allowance + 4 MiB/env + actual tensor budget; engineering reserve, validate peak before scaling",
              "trained": False, "simulator_started": False, "optimizer_updates": 0}
    if args.task == QUIET_STUDENT_TASK:
        from task_support.stair_quiet_v1.distill import check_source
        report["quiet_execution_control"] = check_source(args.task, saved, resume=bool(args.resume))
        from task_support.stair_quiet_v1.deployment import RUNTIME_SCOPE
        report["execution_control_scope"] = RUNTIME_SCOPE
    if host.total-host.available+allowance > .85*host.total:
        raise RuntimeError("Insufficient whole-host memory headroom")
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        free, total = torch.cuda.mem_get_info(torch.device(args.device))
        report.update(gpu_free_bytes=free, gpu_total_bytes=total)
        report["nvidia_smi"] = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu", "--format=csv"], text=True).strip()
        if total-free+allowance > .85*total:
            raise RuntimeError("Insufficient whole-GPU headroom")
    print("STUDENT_V1_PREFLIGHT="+json.dumps(report), flush=True)
    if args.preflight_report:
        args.preflight_report.parent.mkdir(parents=True, exist_ok=True)
        args.preflight_report.write_text(json.dumps(report, indent=2)+"\n")
    return report


def train_or_validate_v1(args, saved, training, perception, task, n, old_meta, source_path, report):
    import torch
    app = launch_app(args)
    try:
        import gymnasium as gym
        from isaaclab.utils.io import dump_yaml
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaaclab_tasks.utils import parse_env_cfg
        import rl_training.tasks  # noqa: F401; local Gym registration
        from m20_student_v1.runner import StudentV1Runner
        cfg = parse_env_cfg(task, device=args.device, num_envs=n)
        cfg.student_perception = perception
        cfg.student_history_steps = training["history_steps"]
        cfg.seed = training["seed"]
        motion = motion_task(task)
        quiet = task == QUIET_STUDENT_TASK
        quiet_contract = None
        if quiet:
            from task_support.stair_quiet_v1.distill import check_source, configure_environment
            quiet_contract = check_source(task, saved, resume=bool(args.resume))
            configure_environment(cfg, quiet_contract)
        manifest = None
        if motion:
            from m20_student_v1.v3_5_teacher import teacher_manifest
            manifest = teacher_manifest(saved, bool(args.resume), allow_quiet=quiet)
            cfg.motion_teacher_tuning = deepcopy(manifest['stair_tuning'])
        if args.validate_only:
            cfg.scene.terrain.terrain_generator.num_rows = 2
            cfg.scene.terrain.terrain_generator.num_cols = 2
            cfg.scene.terrain.max_init_terrain_level = 1
            cfg.scene.terrain.visual_material = None
            cfg.scene.sky_light = None
            cfg.commands.base_velocity.debug_vis = False
        run_dir = args.log_root.expanduser().resolve()/(datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")+"_"+args.run_name)
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir/"resource_preflight.json").write_text(json.dumps(report, indent=2)+"\n")
        env = RslRlVecEnvWrapper(gym.make(task, cfg=cfg), clip_actions=training["clip_actions"])
        try:
            obs = env.get_observations()
            dim = student_dim(training["history_steps"])
            if obs["policy"].shape != (n, dim) or obs["teacher"].shape != (n, 244):
                raise RuntimeError("V1 observation shape mismatch")
            if motion and (obs['motion_control'].shape != (n, 3) or not torch.equal(
                    obs['motion_control'], env.unwrapped.command_manager.get_command('base_velocity'))):
                raise RuntimeError('V3.5 raw command group differs from the requested command')
            if not torch.equal(obs["policy"][:, 6:9], obs["teacher"][:, 6:9]) or not torch.equal(obs["policy"][:, 41:57], obs["teacher"][:, 41:57]):
                raise RuntimeError("Student/teacher command or issued-action semantics differ")
            budget = report["budget"]
            runner = StudentV1Runner(env, deepcopy(training), log_dir=None if args.validate_only else str(run_dir), device=args.device,
                                     min_free_disk_bytes=budget["min_free_disk_bytes"], checkpoint_estimate_bytes=budget["checkpoint_estimate_bytes"])
            if args.resume:
                runner.load(str(source_path), map_location=args.device)
                runner.current_learning_iteration = runner.alg.num_updates
            else:
                runner.alg.load(saved, {"teacher": True, "iteration": False}, True)
                initialization_source = saved
                if args.init_student:
                    initialization_source = torch.load(resolve_checkpoint(args.init_student), map_location="cpu", weights_only=True)
                if motion:
                    from m20_student_v1.v3_5_teacher import initialization_source as motion_initialization
                    initialization_source = motion_initialization(saved, initialization_source if args.init_student else None)
                initialize_student(runner.alg.student, initialization_source)
            provenance = {"version": "v1", "task": task, "num_envs": n, "seed": cfg.seed,
                          "teacher_checkpoint": old_meta.get("teacher_checkpoint", str(source_path)),
                          "teacher_sha256": old_meta.get("teacher_sha256", file_hash(source_path)),
                          "init_student_checkpoint": str(resolve_checkpoint(args.init_student)) if args.init_student else old_meta.get("init_student_checkpoint"),
                          "init_student_sha256": file_hash(resolve_checkpoint(args.init_student)) if args.init_student else old_meta.get("init_student_sha256"),
                          "resume_checkpoint": str(source_path) if args.resume else None,
                          "perception": asdict(perception), "training_config": deepcopy(training),
                          "contract": observation_contract(training["history_steps"]),
                          "rsl_rl_version": importlib.metadata.version("rsl-rl-lib"),
                          "sensor_parameters_calibrated_on_hardware": False,
                          "critic": "sensor placeholder only; no value network or reward optimization in distillation"}
            if motion:
                provenance.update(task_version='v3.5', teacher_manifest=manifest,
                                  command_feedback={'version': 'legacy_v1_v3_5', 'tuning': manifest['stair_tuning']},
                                  teacher_action_target='complete frozen base + raw-command-gated residual',
                                  student_initialization='explicit student' if args.init_student else old_meta.get('student_initialization', 'frozen teacher base'))
            if quiet:
                from task_support.stair_quiet_v1.distill import provenance_for
                provenance = provenance_for(provenance, quiet_contract)
            runner.alg.provenance = provenance
            (run_dir/"provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
            (run_dir/"observation_contract.json").write_text(json.dumps(provenance["contract"], indent=2)+"\n")
            dump_yaml(str(run_dir/"env.yaml"), cfg)
            dump_yaml(str(run_dir/"distillation.yaml"), training)
            print(f"STUDENT_V1_RUN_DIR={run_dir}", flush=True)
            if args.validate_only:
                before = {key: value.clone() for key, value in runner.alg.student.state_dict().items()}
                teacher_before = {key: value.clone() for key, value in runner.alg.teacher.state_dict().items()}
                updates_before = runner.alg.num_updates
                for _ in range(args.validation_steps):
                    with torch.inference_mode():
                        teacher_action = runner.alg.teacher(obs)
                        if teacher_action.shape != (n, 16) or not torch.isfinite(teacher_action).all():
                            raise RuntimeError('Invalid frozen teacher action during inference validation')
                        obs, reward, done, extras = env.step(runner.alg.student(obs))
                    if not all(torch.isfinite(value).all() for value in obs.values()) or not torch.isfinite(reward).all():
                        raise RuntimeError("Nonfinite inference rollout")
                    if not torch.equal(obs["policy"], obs["critic"]):
                        raise RuntimeError("Student/placeholder critic snapshots differ")
                unchanged = all(torch.equal(value, runner.alg.student.state_dict()[key]) for key, value in before.items())
                validation = {"version": "v1", "steps": args.validation_steps, "weights_unchanged": unchanged,
                              "teacher_weights_unchanged": all(torch.equal(v, runner.alg.teacher.state_dict()[k]) for k, v in teacher_before.items()),
                              "observation_shapes": {key: list(value.shape) for key, value in obs.items()},
                              "optimizer_updates_this_validation": runner.alg.num_updates-updates_before,
                              "teacher_frozen": not any(p.requires_grad for p in runner.alg.teacher.parameters()), "trained": False}
                if not unchanged or not validation['teacher_weights_unchanged'] or validation["optimizer_updates_this_validation"]:
                    raise RuntimeError("Inference validation unexpectedly updated weights")
                (run_dir/"validation.json").write_text(json.dumps(validation, indent=2)+"\n")
                print("STUDENT_V1_VALIDATION="+json.dumps(validation), flush=True)
                return
            # Small, scoped source receipt; no datasets or previous run logs.
            for relative in ("source/rl_training/task_support", "source/rl_training/m20_student_v1", "source/rl_training/m20_student_v0",
                             "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20",
                             "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp"):
                shutil.copytree(ROOT/relative, run_dir/"source_snapshot"/relative, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            shutil.copy2(__file__, run_dir/"source_snapshot"/Path(__file__).name)
            if motion:
                shutil.copy2(V3_5_CONFIG_PATH, run_dir/'source_snapshot/student_v3_5.json')
            runner.add_git_repo_to_log(__file__)
            runner.learn(args.max_iterations or training["max_iterations"], init_at_random_ep_len=False)
            if not args.no_export and quiet:
                from task_support.stair_quiet_v1.distill import export_bundle
                export_bundle(runner, run_dir / "exported", runner.alg.provenance)
            elif not args.no_export:
                export_dir = run_dir/"exported"
                runner.export_policy_to_jit(str(export_dir), "student_v1.pt")
                shutil.copy2(run_dir/"observation_contract.json", export_dir/"observation_contract.json")
                (export_dir/"deployment.json").write_text(json.dumps({"training_provenance": runner.alg.provenance}, indent=2)+"\n")
                model = torch.jit.load(str(export_dir/"student_v1.pt"), map_location="cpu")
                with torch.inference_mode():
                    last = env.get_observations()
                    expected = runner.alg.student(last.to(args.device)).cpu()
                    error = float((model(last["policy"].cpu())-expected).abs().max())
                if error > 1e-4:
                    raise RuntimeError("Deployment export parity failed")
                (export_dir/"verification.json").write_text(json.dumps({"max_action_error": error})+"\n")
        finally:
            env.close()
    except BaseException:
        import traceback
        traceback.print_exc()
        app.close()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    else:
        app.close()



def settings_v0(args, saved):
    meta = saved.get("student_v0", {}).get("provenance", {}) if args.resume else {}
    if args.resume and not meta:
        raise ValueError("--resume requires a v0 checkpoint with provenance")
    training = deepcopy(meta.get("training_config", DeeproboticsM20StudentV0DistillationCfg().to_dict()))
    perception = StudentV0PerceptionCfg(**meta.get("perception", {}))
    if args.seed is not None:
        training["seed"] = args.seed
    if args.teacher_ratio is not None:
        training["algorithm"]["teacher_ratio"] = args.teacher_ratio
    elif args.resume:
        training["algorithm"]["teacher_ratio"] = saved["student_v0"].get("teacher_ratio", training["algorithm"]["teacher_ratio"])
    if args.perception_level is not None:
        perception.level = args.perception_level
    perception.validate()
    task = args.task or meta.get("task", "Rough-Deeprobotics-M20-Student-v0")
    n = args.num_envs or meta.get("num_envs", 4 if args.validate_only else 64)
    if args.resume and (task != meta["task"] or n != meta["num_envs"]):
        raise ValueError("V0 resume preserves task and num_envs")
    if n > 512:
        raise ValueError("V0 is limited to 512 environments; v1 supports larger counts subject to resource preflight")
    return training, perception, task, n, meta


def configure_run(args, saved, default_task="Rough-Deeprobotics-M20-Student-v0"):
    """Task is the selector; a resume checkpoint must agree with that selector."""
    checkpoint_version = None
    if args.resume:
        versions = [version for version in ("v0", "v1") if saved.get("student_"+version, {}).get("provenance")]
        if len(versions) != 1:
            raise ValueError("--resume requires an unambiguous v0/v1 student checkpoint")
        checkpoint_version = versions[0]
        checkpoint_task = saved["student_"+checkpoint_version]["provenance"].get("task")
        args.task = args.task or checkpoint_task
        if args.task != checkpoint_task:
            raise ValueError("Resume task differs from checkpoint")
    else:
        args.task = args.task or default_task
    version = TASK_VERSIONS.get(args.task)
    if version is None or (checkpoint_version is not None and version != checkpoint_version):
        raise ValueError("Task does not match the checkpoint version")
    from task_support.stair_quiet_v1.distill import check_source, prepare_initialization
    check_source(args.task, saved, resume=bool(args.resume))
    prepare_initialization(args)
    if version == "v0":
        for name in ("init_student", "config", "history_steps", "teacher_ratio_end"):
            if getattr(args, name) is not None:
                raise ValueError(f"--{name.replace('_', '-')} is only supported for v1 tasks")
    elif args.xr:
        raise ValueError("V1 sensor timing requires XR disabled")
    task_version = "v3" if args.task == "Rough-Deeprobotics-M20-Student-v3" else version
    if motion_task(args.task):
        task_version = 'stair_quiet_v1_student' if args.task == QUIET_STUDENT_TASK else 'v3_5'
        if not args.resume and args.config is None:
            args.config = V3_5_CONFIG_PATH
    if task_version == "v3" and not args.resume and args.config is None:
        args.config = V3_CONFIG_PATH
    args.log_root = args.log_root or ROOT/"logs/rsl_rl"/("deeprobotics_m20_student_"+task_version)
    args.run_name = args.run_name or "student_"+task_version
    args.livestream = args.livestream if args.livestream is not None else (0 if version == "v1" else -1)
    return version


def launch_app(args):
    from isaaclab.app import AppLauncher
    # Freeze the already checked device; optional launcher settings stay explicit.
    if not args.xr:
        os.environ["XR"] = "0"
    launcher = {key: getattr(args, key) for key in ("device", "headless", "enable_cameras", "livestream", "xr", "experience", "kit_args", "anim_recording_enabled", "anim_recording_start_time", "anim_recording_stop_time")}
    launcher["device_explicit"] = True
    if args.rendering_mode is not None:
        launcher["rendering_mode"] = args.rendering_mode
    return AppLauncher(launcher).app


def main(argv=None, default_task="Rough-Deeprobotics-M20-Student-v0"):
    args = parse_args(argv)
    import torch
    source_path = resolve_checkpoint(args.resume or args.teacher_checkpoint)
    if args.init_student:
        args.init_student = resolve_checkpoint(args.init_student)
    saved = torch.load(source_path, map_location="cpu", weights_only=True)
    checkpoint_task = saved.get("student_v1", {}).get("provenance", {}).get("task")
    if args.task == SAFE_STUDENT_TASK or (args.task is None and args.resume and checkpoint_task == SAFE_STUDENT_TASK):
        if args.resume and checkpoint_task != SAFE_STUDENT_TASK:
            raise ValueError("Safe student resume task differs from checkpoint")
        from task_support.student_v3_safe.entry import run
        return run(args, saved, source_path, launch_app)
    version = configure_run(args, saved, default_task)
    settings = settings_v0 if version == "v0" else settings_v1
    training, perception, task, n, meta = settings(args, saved)
    preflight = preflight_v0 if version == "v0" else preflight_v1
    report = preflight(args, saved, training, perception, n, source_path)
    if not args.preflight_only:
        backend = train_or_validate_v0 if version == "v0" else train_or_validate_v1
        backend(args, saved, training, perception, task, n, meta, source_path, report)


def preflight_v0(args, saved, training, perception, n, source_path):
    import psutil
    import torch
    from tensordict import TensorDict
    from rsl_rl.models import MLPModel
    torch.set_num_threads(1)
    torch.manual_seed(training["seed"])
    dim = 618
    observations = TensorDict({"policy": torch.zeros(2, dim), "teacher": torch.zeros(2, 244)}, batch_size=[2])
    model_cfg = deepcopy(training["student"])
    model_cfg.pop("class_name")
    student = MLPModel(observations, training["obs_groups"], "student", 16, **model_cfg)
    teacher_cfg = deepcopy(training["teacher"])
    teacher_cfg.pop("class_name")
    teacher = MLPModel(observations, training["obs_groups"], "teacher", 16, **teacher_cfg)
    teacher.load_state_dict(saved["teacher_state_dict"] if args.resume else saved["actor_state_dict"], strict=True)
    if args.resume:
        checkpoint = saved["student_v0"]
        capacity = n*training["num_steps_per_env"]*training["algorithm"]["replay_rollouts"]
        size = checkpoint["replay_size"]
        if checkpoint["version"] != 1 or checkpoint["replay_capacity"] != capacity or not 0 <= size <= capacity:
            raise ValueError("Incompatible v0 replay checkpoint")
        for key, width in (("replay_obs", 618), ("replay_labels", 16)):
            if saved[key].shape != (size, width) or not torch.isfinite(saved[key]).all():
                raise ValueError("Invalid v0 replay checkpoint")
        student.load_state_dict(saved["student_state_dict"], strict=True)
    else:
        initialize_student(student, saved)
    # Check actual CPU actor serialization, without collection or optimization.
    with torch.inference_mode():
        exported = torch.jit.script(student.as_jit())
        samples = torch.randn(2, dim)
        parity = float((exported(samples)-student(TensorDict({"policy": samples}, batch_size=[2]))).abs().max())
    if parity > 1e-6:
        raise RuntimeError("V0 TorchScript parity failed")
    num_envs, device = n, args.device
    log_root = args.log_root.expanduser().resolve()
    disk_probe = log_root
    while not disk_probe.exists():
        disk_probe = disk_probe.parent
    training_cfg = training
    disk_budget = output_budget_v0(training_cfg,num_envs,args.max_iterations or training_cfg["max_iterations"],
                                saved.get("student_v0",{}).get("num_updates",0) if args.resume else 0,
                                OutputStorageCfg(min_free_disk_gb=args.min_free_disk_gb))
    required_disk = (disk_budget["min_free_disk_bytes"]+128*1024**2 if args.validate_only
                     else disk_budget["required_free_bytes"])
    # This runs before CUDA or Isaac startup and measures the chosen output volume.
    require_disk_space(log_root,required_disk)
    rollout_steps = training_cfg["num_steps_per_env"]
    replay_rollouts = training_cfg["algorithm"]["replay_rollouts"]
    resource = {
        "version": "v0", "task": args.task, "trained": False,
        "simulator_started": False, "optimizer_updates": 0,
        "actor_dim": dim, "teacher_dim": 244, "export_max_action_error": parity,
        "num_envs": num_envs, "device": device,
        "mem_available_bytes": psutil.virtual_memory().available,
        "swap_used_bytes": psutil.swap_memory().used,
        "disk_free_bytes": shutil.disk_usage(disk_probe).free,
        "log_root": str(log_root),
        "disk_output_budget":disk_budget,
        "load_average": list(os.getloadavg()),
        "sample_buffer_formula": "N*T*(618+16)*4*replay_rollouts + N*T*((618+618+244+16+16+1)*4+1)",
        "basis": "budget = 8 GiB startup allowance + 4 MiB per env + rollout/replay tensors; engineering reserve, validate measured peak before scaling",
    }
    replay_bytes = num_envs * rollout_steps * (618+16) * 4 * replay_rollouts
    rollout_bytes = num_envs * rollout_steps * ((618+618+244+16+16+1)*4+1)
    buffer_bytes = replay_bytes + rollout_bytes
    planned_bytes = 8*1024**3 + num_envs*4*1024**2 + buffer_bytes
    resource.update(planned_increment_bytes=planned_bytes, sample_buffer_bytes=buffer_bytes,
                    replay_bytes=replay_bytes, rollout_bytes=rollout_bytes, fleet_cap_fraction=0.85)
    # Whole-device availability includes other workloads; refuse missing readings.
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; no simulation started")
        free, total = torch.cuda.mem_get_info(torch.device(device))
        resource.update(gpu_free_bytes=free, gpu_total_bytes=total)
        resource["nvidia_smi"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu", "--format=csv"], text=True).strip()
        if total-free+planned_bytes > total*resource["fleet_cap_fraction"]:
            raise RuntimeError("Planned whole-GPU occupancy exceeds the fleet cap; no simulation started")
    memory_total = psutil.virtual_memory().total
    if memory_total-resource["mem_available_bytes"]+planned_bytes > memory_total*resource["fleet_cap_fraction"]:
        raise RuntimeError("Insufficient host memory headroom; no simulation started")
    print("STUDENT_RESOURCE_PREFLIGHT=" + json.dumps(resource), flush=True)
    if args.preflight_report:
        args.preflight_report.parent.mkdir(parents=True, exist_ok=True)
        args.preflight_report.write_text(json.dumps(resource, indent=2)+"\n")
    return resource

def train_or_validate_v0(args, saved, training, perception, task, n, meta, source_path, report):
    app = launch_app(args)
    try:
        import torch
        import gymnasium as gym
        from m20_student_v0.disk_runner import StorageDistillationRunner
        from isaaclab.utils.io import dump_yaml
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
        import rl_training.tasks
        from task_support.stair_student.perception import initialize_student_actor
        from m20_student_v0.contract import observation_contract, STUDENT_OBS_DIM, TEACHER_OBS_DIM
        num_envs, device, path = n, args.device, source_path
        log_root = args.log_root.expanduser().resolve()
        disk_budget, resource = report["disk_output_budget"], report
        cfg = parse_env_cfg(task, device=device, num_envs=num_envs)
        agent = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
        if args.resume:
            for name, value in meta["training_config"].items():
                setattr(agent, name, deepcopy(value))
            from m20_student_v0.perception import StudentV0PerceptionCfg
            cfg.student_perception = StudentV0PerceptionCfg(**meta["perception"])
        cfg.seed = args.seed if args.seed is not None else meta.get("seed", agent.seed)
        agent.seed = cfg.seed
        agent.algorithm["teacher_ratio"] = (args.teacher_ratio if args.teacher_ratio is not None
                                            else saved.get("student_v0", {}).get("teacher_ratio", agent.algorithm["teacher_ratio"]))
        cfg.student_perception.level = (args.perception_level if args.perception_level is not None
                                       else meta.get("perception", {}).get("level", cfg.student_perception.level))
        cfg.student_perception.validate()
        if args.validate_only:
            cfg.scene.terrain.terrain_generator.num_rows = 2
            cfg.scene.terrain.terrain_generator.num_cols = 2
            cfg.scene.terrain.visual_material = None
            cfg.scene.sky_light = None
            cfg.commands.base_velocity.debug_vis = False
        run_dir = log_root / (datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f") + "_" + args.run_name)
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir/"resource_preflight.json").write_text(json.dumps(resource, indent=2)+"\n")
        wrapped = RslRlVecEnvWrapper(gym.make(task, cfg=cfg), clip_actions=agent.clip_actions)
        try:
            obs = wrapped.get_observations()
            if tuple(obs["policy"].shape) != (num_envs, STUDENT_OBS_DIM) or tuple(obs["teacher"].shape) != (num_envs, TEACHER_OBS_DIM):
                raise RuntimeError("Student/teacher observation contract mismatch")
            if not torch.equal(obs["policy"][:, :57], obs["teacher"][:, :57]):
                raise RuntimeError("Teacher and student do not share the executed proprioceptive state")
            train_cfg = agent.to_dict()
            runner = StorageDistillationRunner(wrapped,train_cfg,log_dir=None if args.validate_only else str(run_dir),device=device,
                                               min_free_disk_bytes=disk_budget["min_free_disk_bytes"],
                                               checkpoint_estimate_bytes=disk_budget["checkpoint_estimate_bytes"])
            if args.resume:
                runner.load(str(path), map_location=device)
                runner.current_learning_iteration = runner.alg.num_updates
            else:
                runner.alg.load(saved, {"teacher": True, "iteration": False}, strict=True)
                initialize_student_actor(runner.alg.student, saved["actor_state_dict"])
            provenance = {
                "task": task, "num_envs": num_envs, "seed": cfg.seed,
                "teacher_sha256": meta.get("teacher_sha256") if args.resume else file_hash(path),
                "teacher_checkpoint": meta.get("teacher_checkpoint") if args.resume else str(path),
                "resume_checkpoint": str(path) if args.resume else None,
                "perception": dict(cfg.student_perception.__dict__),
                "training_config": agent.to_dict(),
                "contract": observation_contract(), "rsl_rl_version": importlib.metadata.version("rsl-rl-lib"),
            }
            runner.alg.provenance = provenance
            contract = observation_contract()
            contract["perception_config"] = dict(cfg.student_perception.__dict__)
            (run_dir/"observation_contract.json").write_text(json.dumps(contract, indent=2)+"\n")
            (run_dir/"provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
            dump_yaml(str(run_dir/"env.yaml"), cfg)
            dump_yaml(str(run_dir/"distillation.yaml"), agent)
            print(f"STUDENT_RUN_DIR={run_dir}", flush=True)
            if args.validate_only:
                initial_error = None
                if not args.resume and cfg.student_perception.level == 0:
                    with torch.inference_mode():
                        initial_error = float((runner.alg.student(obs) - runner.alg.teacher(obs)).abs().max())
                    if initial_error > 1e-5:
                        raise RuntimeError(f"Clean warm start differs from teacher: {initial_error}")
                # No collection/update calls: this path performs inference only.
                student_before = {k: v.clone() for k, v in runner.alg.student.state_dict().items()}
                updates_before = runner.alg.num_updates
                ages, valid_fraction = [], []
                for _ in range(args.validation_steps):
                    with torch.inference_mode():
                        action = runner.alg.student(obs)
                        if not torch.isfinite(action).all():
                            raise RuntimeError("Nonfinite student action")
                        obs, reward, dones, extras = wrapped.step(action)
                    if not all(torch.isfinite(value).all() for value in obs.values()) or not torch.isfinite(reward).all():
                        raise RuntimeError("Nonfinite environment observation/reward")
                    if not torch.equal(obs["policy"], obs["critic"]):
                        raise RuntimeError("Policy and critic map snapshots differ")
                    state = wrapped.unwrapped._m20_student_v0_map
                    ages.append(float(((wrapped.unwrapped.common_step_counter-state.capture).float()*wrapped.unwrapped.step_dt).max()))
                    valid_fraction.append(float(obs["policy"][:, 244:431].mean()))
                unchanged = all(torch.equal(value, runner.alg.student.state_dict()[key]) for key, value in student_before.items())
                if not unchanged:
                    raise RuntimeError("Validation unexpectedly changed student weights")
                report = {"task": task, "num_envs": num_envs, "steps": args.validation_steps,
                          "observation_shapes": {k: list(v.shape) for k, v in obs.items()},
                          "clean_warm_start_max_error": initial_error, "weights_unchanged": unchanged,
                          "optimizer_updates": runner.alg.num_updates, "teacher_requires_grad": any(p.requires_grad for p in runner.alg.teacher.parameters()),
                          "optimizer_updates_this_validation":runner.alg.num_updates-updates_before,
                          "max_map_age_s": max(ages), "valid_fraction_range": [min(valid_fraction), max(valid_fraction)],
                          "trained": False, "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.startswith("cuda") else None}
                (run_dir/"validation.json").write_text(json.dumps(report, indent=2)+"\n")
                print("STUDENT_VALIDATION="+json.dumps(report), flush=True)
                return
            # Keep a small code snapshot; no datasets/checkpoints are copied.
            for directory in ("source", "scripts"):
                shutil.copytree(ROOT/directory, run_dir/"source_snapshot"/directory,
                                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            runner.add_git_repo_to_log(__file__)
            runner.learn(args.max_iterations or agent.max_iterations, init_at_random_ep_len=False)
            if not args.no_export:
                export_dir = run_dir/"exported"
                runner.export_policy_to_jit(str(export_dir), "student_v0.pt")
                shutil.copy2(run_dir/"observation_contract.json", export_dir/"observation_contract.json")
                # Export parity with the actual last observation, using deterministic actions.
                model = torch.jit.load(str(export_dir/"student_v0.pt"), map_location="cpu")
                with torch.inference_mode():
                    last_obs = wrapped.get_observations()
                    expected = runner.alg.student(last_obs.to(device)).cpu()
                    actual = model(last_obs["policy"].cpu())
                error = float((actual-expected).abs().max())
                if error > 1e-4:
                    raise RuntimeError(f"Export parity failed: {error}")
                (export_dir/"verification.json").write_text(json.dumps({"max_action_error": error})+"\n")
        finally:
            wrapped.close()

    except BaseException:
        import traceback
        traceback.print_exc()
        app.close()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    else:
        app.close()


if __name__ == "__main__":
    main()
