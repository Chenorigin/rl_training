"""Weights-only warm start from the complete 46400 actor; strict task resume."""
from __future__ import annotations

import copy
from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path

import torch

from task_support.idle_stop_motion_v1.runner import (
    IdleStopMotionRunner, file_sha256, validate_resume_manifest,
)
from .baselines import (
    PROJECT_ROOT, TASK, TEACHER, TEACHER_SHA256, STUDENT, STUDENT_SHA256, checked_path,
)
from .landing import control_contract


def canonical(value):
    """Stable config encoding: never serialize callable memory addresses."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        return [canonical(item) for item in value]
    if isinstance(value, dict):
        return {str(key): canonical(item) for key, item in value.items()}
    if callable(value):
        return f"{value.__module__}:{value.__qualname__}"
    if is_dataclass(value):
        return canonical(asdict(value))
    if hasattr(value, "to_dict"):
        return canonical(value.to_dict())
    if hasattr(value, "__dict__"):
        return canonical(vars(value))
    return str(value)


def refinement_contract(env_cfg, train_cfg):
    """Freeze the actual optimization objective and physical/observation contract."""
    fields = (
        "rewards", "observations", "actions", "commands", "terminations",
        "curriculum", "events", "decimation", "episode_length_s",
        "stair_quiet", "landing_control",
    )
    contract = {
        "task": TASK,
        "teacher_sha256": TEACHER_SHA256,
        "student_sha256": STUDENT_SHA256,
        "environment": {key: canonical(getattr(env_cfg, key, None)) for key in fields},
        "algorithm": canonical(train_cfg["algorithm"]),
        "actor": canonical(train_cfg["actor"]),
        "critic": canonical(train_cfg["critic"]),
        "num_steps_per_env": train_cfg["num_steps_per_env"],
        "obs_groups": canonical(train_cfg["obs_groups"]),
    }
    if hasattr(env_cfg, "sim"):
        contract["physics"] = canonical(env_cfg.sim)
    if hasattr(env_cfg, "scene"):
        contract["terrain"] = canonical(env_cfg.scene.terrain)
        contract["robot"] = canonical(env_cfg.scene.robot)
        for name in ("height_scanner", "contact_forces"):
            contract[name] = canonical(getattr(env_cfg.scene, name, None))
    # Paths are provenance, not permission to silently load a different model.
    return json.loads(json.dumps(contract, allow_nan=False))


def training_sources():
    folder = Path(__file__).resolve().parent
    cfg = PROJECT_ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity"
    sources = [folder / name for name in (
        "runner.py", "signals.py", "baselines.py", "landing.py", "isaac_landing.py", "terrain_memory.py",
    )]
    sources += list((folder.parent / "idle_stop_motion_v1").glob("*.py"))
    sources += [
        cfg / "mdp/stair_quiet_v1_env.py",
        cfg / "velocity_env_cfg.py",
        cfg / "config/wheeled/deeprobotics_m20/idle_stop_motion_v1_env_cfg.py",
        cfg / "config/wheeled/deeprobotics_m20/rough_env_cfg.py",
        cfg / "config/wheeled/deeprobotics_m20/agents/idle_stop_motion_v1_ppo_cfg.py",
        cfg / "config/wheeled/deeprobotics_m20/agents/rough_ppo_cfg.py",
        cfg / "config/wheeled/deeprobotics_m20/stair_quiet_v1_env_cfg.py",
        cfg / "config/wheeled/deeprobotics_m20/agents/stair_quiet_v1_ppo_cfg.py",
        PROJECT_ROOT / "source/rl_training/rl_training/assets/deeprobotics.py",
        PROJECT_ROOT / "scripts/reinforcement_learning/rsl_rl/train.py",
        PROJECT_ROOT / "scripts/reinforcement_learning/rsl_rl/cli_args.py",
    ]
    sources += [cfg / "mdp" / name for name in (
        "commands.py", "rewards.py", "events.py", "curriculums.py",
    ) if (cfg / "mdp" / name).is_file()]
    return {str(p.relative_to(PROJECT_ROOT)): file_sha256(p) for p in sorted(sources)}


class StairQuietRunner(IdleStopMotionRunner):
    """Same deployable actor ABI; new objective, optimizer and provenance."""

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        self._quiet_train_cfg = copy.deepcopy(train_cfg)
        super().__init__(env, train_cfg, log_dir, device)
        self._guide_schedule["guided_iterations"] = 0
        self._guide_schedule["decay_iterations"] = 0
        self._install_guide_schedule()
        self._install_quiet_schedule()

    def _install_quiet_schedule(self):
        """Anchor curriculum to *this* run, including resumed/repeated learn calls."""
        env = self.env.unwrapped
        env._m20_stair_quiet_schedule = {
            # RSL saves the last zero-based iteration, while guard_updates is
            # the number of completed updates. The NEXT rollout starts there.
            "iteration_offset": int(self.alg.guard_updates),
            "step_offset": int(getattr(env, "common_step_counter", 0)),
            "steps_per_iteration": int(self.cfg["num_steps_per_env"]),
        }

    def initialize_task(self, arguments):
        self._cli_args = copy.deepcopy(arguments)
        if arguments.get("preserve_actor_from") or arguments.get("init_critic_from"):
            raise ValueError("StairQuiet initializes actor, critic and reference from its pinned full teacher")
        if self._restored:
            if arguments.get("init_actor_from"):
                raise ValueError("Cannot combine StairQuiet resume with actor initialization")
            self._initialized = True
            self._install_quiet_schedule()
            return
        if arguments.get("resume") or self.cfg.get("resume"):
            raise RuntimeError("StairQuiet resume must load its own complete checkpoint first")
        path = checked_path(arguments.get("init_actor_from") or TEACHER, TEACHER_SHA256)
        checked_path(STUDENT, STUDENT_SHA256)
        saved = torch.load(path, map_location="cpu", weights_only=True)
        prior = validate_resume_manifest(saved["m20_training_manifest"])
        # Do not call alg.load(): that would also restore source optimizer/counters.
        self.alg.actor.load_state_dict(saved["actor_state_dict"], strict=True)
        self.alg.critic.load_state_dict(saved["critic_state_dict"], strict=True)
        self.alg.set_reference(saved["actor_state_dict"])
        self.alg.guard_updates = 0
        self.current_learning_iteration = 0
        if self.alg.optimizer.state:
            raise RuntimeError("StairQuiet warm start requires a fresh optimizer")
        self._prior_manifest = prior
        self._initialization = {
            "initial_actor_path": str(path), "initial_actor_sha256": TEACHER_SHA256,
            "initial_critic_path": str(path), "initial_critic_sha256": TEACHER_SHA256,
            "critic_initialization": "complete 46400 critic; fresh optimizer",
            "preservation_actor_sha256": TEACHER_SHA256,
        }
        self._initialized = True
        self._install_guide_schedule()
        self._install_quiet_schedule()

    def learn(self, *args, **kwargs):
        if not self._initialized:
            self.initialize_task({})
        self.current_learning_iteration = int(self.alg.guard_updates)
        self._install_quiet_schedule()
        return super().learn(*args, **kwargs)

    def _record_manifest(self):
        manifest = super()._record_manifest()
        contract = refinement_contract(self.env.unwrapped.cfg, self._quiet_train_cfg)
        manifest["noise_refinement"] = {
            "task": TASK, "version": 1,
            "teacher_baseline": {"path": TEACHER, "sha256": TEACHER_SHA256},
            "student_baseline": {"path": STUDENT, "sha256": STUDENT_SHA256},
            "contract": contract,
            "contract_sha256": hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest(),
            "source_sha256": training_sources(),
            "wheel_collision_geometry": canonical(getattr(self.env.unwrapped, "stair_quiet_geometry", None)),
            "control_contract": self._control_contract(),
            "control_adapter": (self.env.unwrapped.stair_quiet_landing.contract()
                                if hasattr(self.env.unwrapped, "stair_quiet_landing") else None),
            "acoustic_status": "not_measured",
            "acoustic_requirement": "candidate LAeq and LAFmax <= 0.5 * each baseline dB reading",
        }
        self.alg.m20_training_manifest = copy.deepcopy(manifest)
        return manifest

    def _control_contract(self):
        settings = getattr(self.env.unwrapped.cfg, "landing_control", None)
        return control_contract(settings) if settings is not None else None

    def load(self, path, load_cfg=None, strict=True, map_location=None):
        if load_cfg is not None or not strict:
            raise ValueError("StairQuiet supports strict full resume only")
        saved = torch.load(path, map_location="cpu", weights_only=True)
        noise = saved.get("m20_training_manifest", {}).get("noise_refinement", {})
        if noise.get("task") != TASK or noise.get("version") != 1:
            raise ValueError("Use the pinned teacher for initialization, not as a StairQuiet resume")
        contract = refinement_contract(self.env.unwrapped.cfg, self._quiet_train_cfg)
        if noise.get("contract") != contract:
            raise ValueError("StairQuiet resume environment/optimization contract changed")
        digest = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
        if noise.get("contract_sha256") != digest:
            raise ValueError("StairQuiet resume contract hash mismatch")
        if noise.get("source_sha256") != training_sources():
            raise ValueError("StairQuiet resume training source changed")
        if noise.get("control_contract") != self._control_contract():
            raise ValueError("StairQuiet resume landing control contract changed")
        if "preservation_reference_actor" not in saved:
            raise ValueError("StairQuiet resume lacks its complete frozen reference actor")
        source = torch.load(checked_path(TEACHER, TEACHER_SHA256), map_location="cpu", weights_only=True)
        reference = saved["preservation_reference_actor"]
        expected = source["actor_state_dict"]
        if set(reference) != set(expected) or any(
            not torch.equal(reference[key], tensor) for key, tensor in expected.items()
        ):
            raise ValueError("StairQuiet resume reference differs from pinned complete teacher")
        actor_state = saved.get("actor_state_dict", {})
        if any(key not in actor_state or not torch.equal(actor_state[key], tensor)
               for key, tensor in expected.items() if not key.startswith("residual_mlp.")):
            raise ValueError("StairQuiet resume frozen baseline/distribution changed")
        guide = saved["m20_training_manifest"].get("descent_entry_guide_schedule", {})
        if guide.get("guided_iterations") != 0 or guide.get("decay_iterations") != 0:
            raise ValueError("StairQuiet must not restart the old descent guide")
        iteration, completed = saved.get("iter"), saved.get("preservation_updates")
        if (type(iteration) is not int or type(completed) is not int
                or iteration < 0 or completed < 0
                or not ((iteration == completed == 0) or completed == iteration + 1)):
            raise ValueError("StairQuiet resume iteration/completed-update counters disagree")
        result = super().load(path, load_cfg, strict, map_location)
        self.current_learning_iteration = int(self.alg.guard_updates)
        self._install_quiet_schedule()
        return result
