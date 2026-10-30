"""Width-to-safe weight continuation and strict, complete safe-task resume."""
import copy
import hashlib
import json
import math
from pathlib import Path
import torch
from rsl_rl.runners import OnPolicyRunner
from task_support.idle_stop_motion_v1.runner import IdleStopMotionRunner, file_sha256, validate_resume_manifest
from task_support.stair_quiet_v1.runner import StairQuietRunner, canonical
from task_support.stair_quiet_v1.baselines import PROJECT_ROOT, TEACHER, TEACHER_SHA256, checked_path
from task_support.stair_quiet_v1_width.runner import contract as width_contract, sources as width_sources, checked_counts
from task_support.stair_quiet_v1_width.curriculum import WidthCurriculum
from .course import course_state, validate_counts
from .settings import SAFETY_DIM

TASK = "stair_quiet_v1_safe"
DEFAULT_START = "logs/rsl_rl/deeprobotics_m20_stair_quiet_v1_width/2026-10-22_00-43-59_teacher_width_speed_recovery_fix/model_9500.pt"
DEFAULT_START_SHA256 = "8319905c59a49e3381099ea63d6c936045b0cf3e8328ac2038d6dabb69a7a916"
ACTOR = "task_support.stair_quiet_v1_safe.actor:SafeResidualActor"


def contract(env_cfg, train_cfg):
    result = width_contract(env_cfg, train_cfg)
    result["task"] = TASK
    result["perception"] = "raw244 + longitudinal187 + privileged_safety48; critic unchanged285+command3"
    return result


def sources():
    result = width_sources()
    base = PROJECT_ROOT/"source/rl_training/rl_training/tasks/manager_based/locomotion/velocity"
    cfg = base/"config/wheeled/deeprobotics_m20"
    paths = list(Path(__file__).parent.glob("*.py"))+[
        base/"mdp/stair_quiet_v1_safe_env.py", cfg/"stair_quiet_v1_safe_env_cfg.py",
        cfg/"agents/stair_quiet_v1_safe_ppo_cfg.py", cfg/"stair_quiet_v1_safe/__init__.py"]
    result.update({str(p.relative_to(PROJECT_ROOT)): file_sha256(p) for p in paths})
    return result


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def validate_model(actual, expected):
    if not isinstance(actual, dict) or set(actual) != set(expected) or any(
            not isinstance(actual[k], torch.Tensor) or actual[k].shape != v.shape
            or not torch.isfinite(actual[k]).all() for k, v in expected.items()):
        raise ValueError("Invalid/nonfinite safe model state")


def validate_optimizer(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Nonfinite safe optimizer state")
    if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
        raise ValueError("Nonfinite safe optimizer state")
    if isinstance(value, dict):
        for item in value.values():
            validate_optimizer(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            validate_optimizer(item)


def mutable_cache(value):
    """Keep rollout-created heading tensors usable by RSL's initial obs read."""
    if isinstance(value, torch.Tensor) and value.is_inference():
        with torch.inference_mode(False):
            return value.clone()
    if isinstance(value, dict):
        return {key: mutable_cache(item) for key, item in value.items()}
    return value


def validate_width_source(saved, teacher):
    manifest = validate_resume_manifest(saved.get("m20_training_manifest", {}))
    noise = manifest.get("noise_refinement", {})
    if (noise.get("task") != "stair_quiet_v1_width" or noise.get("version") != 1
            or noise.get("contract_sha256") != digest(noise.get("contract", {}))
            or noise.get("source_sha256") != width_sources()):
        raise ValueError("Safe initialization requires an intact current width checkpoint; safe checkpoints use --resume")
    expected = teacher["actor_state_dict"]
    for kind in ("actor", "critic"):
        reference, actual = teacher[kind+"_state_dict"], saved.get(kind+"_state_dict", {})
        if set(reference) != set(actual) or any(
                v.shape != actual[k].shape or not torch.isfinite(actual[k]).all() for k, v in reference.items()):
            raise ValueError("Nonfinite or incompatible width model")
    if any(not torch.equal(saved["actor_state_dict"][k], v)
           for k, v in expected.items() if not k.startswith("residual_mlp.")):
        raise ValueError("Width frozen base/distribution differs from the pinned teacher")
    return manifest


class StairQuietSafeRunner(StairQuietRunner):
    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        # Parent constructor hard-codes the legacy actor. Build the same RSL
        # algorithm directly, with all new branch parameters in its optimizer.
        configured = copy.deepcopy(train_cfg)
        configured["actor"].update(class_name=ACTOR, residual_hidden_dims=[128, 64])
        configured["obs_groups"].update(actor=["policy", "motion_control", "locomotion_scan", "safety"],
                                         critic=["critic", "motion_control"])
        self._quiet_train_cfg = copy.deepcopy(configured)
        self._actor_model_config = copy.deepcopy(configured["actor"])
        OnPolicyRunner.__init__(self, env, configured, log_dir, device)
        self._restored = self._initialized = False
        self._cli_args, self._initialization, self._prior_manifest = {}, {}, {}
        self._guide_schedule = dict(guided_iterations=0, decay_iterations=0,
            base_weights={"entry_front_lead_weight": .5, "entry_front_progress_weight": .5, "entry_sequence_weight": .5})
        self._install_guide_schedule()
        self._install_quiet_schedule()

    def initialize_task(self, arguments):
        self._cli_args = copy.deepcopy(arguments)
        if arguments.get("preserve_actor_from") or arguments.get("init_critic_from"):
            raise ValueError("Safe loads actor and critic together from width")
        if self._restored:
            if arguments.get("init_actor_from"):
                raise ValueError("Cannot combine safe resume with width initialization")
            return
        if arguments.get("resume") or self.cfg.get("resume"):
            raise ValueError("Safe resume must load a safe checkpoint first")
        path = self._project_path(arguments.get("init_actor_from") or DEFAULT_START)
        sha = file_sha256(path)
        if not arguments.get("init_actor_from") and sha != DEFAULT_START_SHA256:
            raise ValueError("Default width checkpoint hash changed")
        saved = torch.load(path, map_location="cpu", weights_only=True)
        teacher = torch.load(checked_path(TEACHER, TEACHER_SHA256), map_location="cpu", weights_only=True)
        prior = validate_width_source(saved, teacher)
        levels = prior["width_curriculum"]["level"]
        s = self.env.unwrapped.cfg.stair_tail
        if (not levels or any(type(x) is not int or not 0 <= x < len(s.widths_m) for x in levels)
                or tuple(prior["noise_refinement"]["contract"]["environment"]["stair_tail"]["widths_m"]) != tuple(s.widths_m)):
            raise ValueError("Width source curriculum is incompatible")
        self.alg.actor.load_width_state(saved["actor_state_dict"])
        self.alg.critic.load_state_dict(saved["critic_state_dict"], strict=True)
        reference = copy.deepcopy(self.alg.actor)
        reference.load_width_state(teacher["actor_state_dict"])
        self.alg.set_reference(reference.state_dict())
        self.alg.guard_updates = self.current_learning_iteration = 0
        if self.alg.optimizer.state:
            raise RuntimeError("New safe branch requires a fresh optimizer")
        env, state = self.env.unwrapped, course_state(self.env.unwrapped)
        # Preserve width targets, but newly defined returns/material evidence
        # have fresh statistics. Different num_envs resample targets evenly.
        source = torch.tensor(levels, device=env.device)
        indices = torch.arange(env.num_envs, device=env.device)*len(levels)//env.num_envs
        state["width_curriculum"].level.copy_(source[indices])
        state["width_curriculum"].level[state["material_practice"]] = s.material_practice_level
        state["width_curriculum"].success_streak.zero_()
        self._initialization = dict(initial_actor_path=str(path), initial_actor_sha256=sha,
            initial_critic_path=str(path), initial_critic_sha256=sha,
            critic_initialization="complete width critic; new-target critic warmup; fresh optimizer",
            source_iteration=int(saved["iter"]), source_completed_updates=int(saved["preservation_updates"]),
            source_width_statistics=prior["width_curriculum"]["statistics"],
            continuation="actor+critic and width targets preserved; optimizer, material evidence and safe statistics restarted",
            preservation_actor_sha256=TEACHER_SHA256)
        self._initialized = True
        self._reset_physics()
        self._install_quiet_schedule()

    def _reset_physics(self):
        env = self.env.unwrapped
        env._tail_skip_episode_record = True
        try:
            # Heading and landing caches can be inference tensors after a
            # rollout. Reset must use the same mode as physical stepping.
            with torch.inference_mode():
                env.reset()
            cache = getattr(env, "_m20_zero_hold_heading_state", None)
            if cache is not None:
                env._m20_zero_hold_heading_state = mutable_cache(cache)
        finally:
            env._tail_skip_episode_record = False

    def _record_manifest(self):
        manifest = IdleStopMotionRunner._record_manifest(self)
        env, state = self.env.unwrapped, course_state(self.env.unwrapped)
        frozen = contract(env.cfg, self._quiet_train_cfg)
        manifest.update(refinement_version="m20_stair_quiet_v1_safe.v1",
            actor_model_config=copy.deepcopy(self._actor_model_config),
            actor_input_groups={"policy": 244, "motion_control": 3, "locomotion_scan": 187, "safety": SAFETY_DIM})
        manifest["noise_refinement"] = dict(task=TASK, version=1, contract=frozen, contract_sha256=digest(frozen),
            source_sha256=sources(), teacher_baseline={"path": TEACHER, "sha256": TEACHER_SHA256},
            wheel_collision_geometry=canonical(env.stair_quiet_geometry),
            control_contract=self._control_contract(), control_adapter=env.stair_quiet_landing.contract(),
            terrain_contacts=env.scene.terrain.contact_receipt, wheel_contacts=env.tail_wheel_material_receipt,
            acoustic_status="not_measured", deployment_status="privileged_teacher_requires_safe_observation_adapter")
        manifest["tail_curriculum"] = dict(stage=env.tail_stage,
            case_metrics={k: v.cpu().tolist() for k, v in state["case_metrics"].items()},
            **{k: state[k].cpu().tolist() for k in ("episodes", "successes", "stage_episodes", "stage_successes", "assign_count")})
        manifest["width_curriculum"] = dict(**state["width_curriculum"].state_dict(),
            statistics={k: v.cpu().tolist() for k, v in state["width_statistics"].items()})
        manifest["safe_mechanics"] = {k: v.cpu().tolist() for k, v in state["mechanics"].items()}
        self.alg.m20_training_manifest = copy.deepcopy(manifest)
        return manifest

    def save(self, path, infos=None):
        self._record_manifest()
        return OnPolicyRunner.save(self, path, infos)

    def load(self, path, load_cfg=None, strict=True, map_location=None):
        if load_cfg is not None or not strict:
            raise ValueError("Safe resume requires complete strict state")
        saved = torch.load(path, map_location="cpu", weights_only=True)
        m = saved.get("m20_training_manifest", {})
        noise = m.get("noise_refinement", {})
        frozen = contract(self.env.unwrapped.cfg, self._quiet_train_cfg)
        if (noise.get("task") != TASK or noise.get("version") != 1
                or noise.get("contract") != frozen or noise.get("contract_sha256") != digest(frozen)
                or noise.get("source_sha256") != sources()):
            raise ValueError("Safe resume task/physics/observations/optimization/source contract differs")
        if any(k not in saved for k in ("actor_state_dict", "critic_state_dict", "optimizer_state_dict", "preservation_reference_actor")):
            raise ValueError("Safe resume is missing models/optimizer/reference")
        validate_model(saved["actor_state_dict"], self.alg.actor.state_dict())
        validate_model(saved["critic_state_dict"], self.alg.critic.state_dict())
        validate_model(saved["preservation_reference_actor"], self.alg.actor.state_dict())
        validate_optimizer(saved["optimizer_state_dict"])
        if noise.get("control_contract") != self._control_contract():
            raise ValueError("Safe landing control contract changed")
        i, completed = saved.get("iter"), saved.get("preservation_updates")
        if (type(i) is not int or type(completed) is not int or i < 0
                or not (i == completed == 0 or completed == i+1)):
            raise ValueError("Safe resume iteration counters disagree")
        teacher = torch.load(checked_path(TEACHER, TEACHER_SHA256), map_location="cpu", weights_only=True)
        actor, reference = saved["actor_state_dict"], saved["preservation_reference_actor"]
        for k, v in teacher["actor_state_dict"].items():
            if (k not in reference or not torch.equal(reference[k], v)
                    or (not k.startswith("residual_mlp.") and not torch.equal(actor[k], v))):
                raise ValueError("Safe resume frozen base/reference changed")
        for name in ("safety_mlp.4.weight", "safety_mlp.4.bias"):
            if name not in reference or reference[name].count_nonzero():
                raise ValueError("Safe preservation reference has a nonzero safety branch")
        env, current = self.env.unwrapped, course_state(self.env.unwrapped)
        tail, width = m["tail_curriculum"], m["width_curriculum"]
        if type(tail["stage"]) is not int or tail["stage"] not in (0, 1, 2):
            raise ValueError("Invalid safe contact stage")
        restored = WidthCurriculum(env.num_envs, env.device, env.cfg.stair_tail)
        restored.load_state_dict(width)
        if (restored.level[current["material_practice"]] != env.cfg.stair_tail.material_practice_level).any():
            raise ValueError("Material practice targets changed")
        staged = {k: checked_counts(tail[k], current[k]) for k in ("episodes", "successes", "stage_episodes", "stage_successes", "assign_count")}
        staged["width_statistics"] = {k: checked_counts(width["statistics"][k], v) for k, v in current["width_statistics"].items()}
        validate_counts(staged, env.cfg.stair_tail)
        for group, raw in (("case_metrics", tail["case_metrics"]), ("mechanics", m["safe_mechanics"])):
            staged[group] = {}
            if set(raw) != set(current[group]):
                raise ValueError("Incomplete safe mechanical evidence")
            for k, target in current[group].items():
                value = torch.as_tensor(raw[k], dtype=target.dtype, device=env.device)
                if value.shape != target.shape or not torch.isfinite(value).all() or (value < 0).any():
                    raise ValueError("Invalid safe mechanical evidence")
                staged[group][k] = value
        self.alg.load(saved, strict=True)
        self._prior_manifest = copy.deepcopy(m)
        self._initialization = {"resumed_checkpoint": str(Path(path).resolve()), "resumed_checkpoint_sha256": file_sha256(path)}
        self._restored = self._initialized = True
        self.current_learning_iteration = int(self.alg.guard_updates)
        env.tail_stage = tail["stage"]
        for k, value in staged.items():
            if isinstance(value, dict):
                for name, tensor in value.items():
                    current[k][name].copy_(tensor)
            else:
                current[k].copy_(value)
        current["width_curriculum"].load_state_dict(restored.state_dict())
        self._reset_physics()
        self._install_guide_schedule()
        self._install_quiet_schedule()
        return saved.get("infos")
