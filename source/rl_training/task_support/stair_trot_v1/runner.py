"""Task-specific initialization and full resume; never starts training on import."""
import copy
from dataclasses import asdict
import json
from pathlib import Path

import torch
from rsl_rl.runners import OnPolicyRunner

from task_support.idle_stop_motion_v1.runner import (
    PROJECT_ROOT, file_sha256, use_compact_console_logging, validate_resume_manifest,
)

VERSION = "m20_stair_trot_v1"
ACTOR_CLASS = "task_support.stair_trot_v1.actor:StairTrotActor"


def task_contract(env_cfg):
    """Persist terrain and route semantics as well as the policy input ABI."""
    contract = {"objective": "stable_ascent", "trot_control_size": 6}
    if hasattr(env_cfg, "stair_course"):
        contract["course"] = asdict(env_cfg.stair_course)
        generator = env_cfg.scene.terrain.terrain_generator
        keys = ("proportion", "size", "step_height_range", "step_depth_range",
                "num_steps", "start_platform_length", "end_platform_length", "riser", "safety_floor_depth")
        contract["terrain"] = {
            "num_rows": generator.num_rows, "num_cols": generator.num_cols,
            "size": generator.size, "difficulty_range": generator.difficulty_range,
            "sub_terrains": {name: {key: getattr(cfg, key) for key in keys if hasattr(cfg, key)}
                             for name, cfg in generator.sub_terrains.items()},
        }
        params = env_cfg.events.randomize_rigid_body_material.params
        contract["material"] = {key: copy.deepcopy(params[key]) for key in (
            "static_friction_range", "dynamic_friction_range", "restitution_range", "make_consistent")
                               if key in params}
    return contract


class StairTrotRunner(OnPolicyRunner):
    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        cfg = copy.deepcopy(train_cfg)
        settings = env.unwrapped.cfg.trot
        self._settings = asdict(settings)
        self._task_contract = task_contract(env.unwrapped.cfg)
        limits = [settings.hipx_residual_limit, settings.hipy_residual_limit,
                  settings.knee_residual_limit] * 4 + [settings.wheel_residual_limit] * 4
        cfg["actor"].update(class_name=ACTOR_CLASS, residual_hidden_dims=[128, 64],
                            trot_hidden_dims=[256, 128], correction_limits=limits,
                            command_limits=[settings.forward_command_min, settings.lateral_command_max,
                                            settings.yaw_command_max])
        cfg["obs_groups"] = {"actor": ["policy", "motion_control", "trot_control"],
                             "critic": ["critic", "motion_control", "trot_control"]}
        if cfg["actor"].get("obs_normalization", False):
            raise ValueError("StairTrot must preserve the teacher's unnormalized policy ABI")
        self._actor_config = copy.deepcopy(cfg["actor"])
        self._algorithm_config = copy.deepcopy(cfg["algorithm"])
        self._reward_weights = {name: term.weight for name, term in vars(env.unwrapped.cfg.rewards).items()
                                if term is not None and hasattr(term, "weight")}
        super().__init__(env, cfg, log_dir, device)
        self._initialized = False
        self._restored = False
        # The source teacher's descent guide was already fully decayed.
        env.unwrapped._m20_descent_entry_guide_schedule = {
            "guided_iterations": 0, "decay_iterations": 0,
            "base_weights": {"entry_front_lead_weight": 0.5,
                             "entry_front_progress_weight": 0.5, "entry_sequence_weight": 0.5},
            "steps_per_iteration": cfg["num_steps_per_env"], "iteration_offset": 0,
        }

    def initialize_task(self, arguments):
        if arguments.get("preserve_actor_from") or arguments.get("init_critic_from"):
            raise ValueError("StairTrot freezes the complete source teacher and starts a fresh critic")
        if self._restored:
            if arguments.get("init_actor_from"):
                raise ValueError("Full resume cannot also initialize a teacher")
            return
        if self._initialized:
            return
        if arguments.get("resume") or self.cfg.get("resume"):
            raise RuntimeError("StairTrot resume requires load() before initialization")
        path = Path(arguments.get("init_actor_from") or self.cfg["baseline_checkpoint"]).expanduser()
        path = (path if path.is_absolute() else PROJECT_ROOT / path).resolve()
        saved = torch.load(path, map_location="cpu", weights_only=True)
        validate_resume_manifest(saved.get("m20_training_manifest", {}))
        self.alg.actor.load_teacher_state(saved["actor_state_dict"])
        manifest = {
            "refinement_version": VERSION,
            "actor_model_config": self._actor_config,
            "actor_input_groups": {"policy": 244, "motion_control": 3, "trot_control": 6},
            "critic_input_groups": {"critic": 285, "motion_control": 3, "trot_control": 6},
            "source_teacher": str(path), "source_teacher_sha256": file_sha256(path),
            "source_teacher_manifest": saved["m20_training_manifest"],
            "trot_settings": self._settings,
            "task_contract": self._task_contract,
            "reward_weights": self._reward_weights,
            "algorithm_config": self._algorithm_config,
        }
        self.alg.m20_training_manifest = copy.deepcopy(manifest)
        self._initialized = True

    def load(self, path, load_cfg=None, strict=True, map_location=None):
        if not strict or load_cfg is not None:
            raise ValueError("StairTrot load supports strict full resume only")
        saved = torch.load(path, map_location=map_location or "cpu", weights_only=True)
        required = {"actor_state_dict", "critic_state_dict", "optimizer_state_dict",
                    "iter", "preservation_updates", "m20_training_manifest"}
        if not required.issubset(saved):
            raise ValueError(f"Incomplete StairTrot checkpoint: {sorted(required - saved.keys())}")
        manifest = saved["m20_training_manifest"]
        if manifest.get("refinement_version") != VERSION:
            raise ValueError("Use --init_actor_from for a source teacher; --resume requires StairTrot-v1")
        if manifest.get("task_contract") != self._task_contract:
            raise ValueError("StairTrot resume terrain/observation/route configuration differs")
        if manifest.get("trot_settings") != self._settings or manifest.get("actor_model_config") != self._actor_config:
            raise ValueError("StairTrot resume configuration differs from the saved contract")
        if (manifest.get("reward_weights") != self._reward_weights
                or manifest.get("algorithm_config") != self._algorithm_config):
            raise ValueError("StairTrot resume reward/algorithm configuration differs from the saved contract")
        if any(type(saved[key]) is not int or saved[key] < 0 for key in ("iter", "preservation_updates")):
            raise ValueError("StairTrot counters must be nonnegative integers")
        self.alg.load(saved, strict=True)
        self.current_learning_iteration = saved["iter"]
        self.alg.m20_training_manifest = copy.deepcopy(manifest)
        self._initialized = self._restored = True
        return saved.get("infos")

    def record_initialization(self, arguments):
        self.initialize_task(arguments)
        self.alg.m20_training_manifest["cli_args"] = copy.deepcopy(arguments)

    def learn(self, *args, **kwargs):
        if not self._initialized:
            self.initialize_task({})
        use_compact_console_logging(self.logger, fast=True)
        if self.logger.log_dir is not None:
            path = Path(self.logger.log_dir) / "params/launch_manifest.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self.alg.m20_training_manifest, indent=2) + "\n")
        return super().learn(*args, **kwargs)
