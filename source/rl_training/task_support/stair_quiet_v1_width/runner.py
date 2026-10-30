"""Independent width task: original warm start, strict full curriculum resume."""
import copy
import hashlib
import json
from pathlib import Path
import torch
from task_support.idle_stop_motion_v1.runner import IdleStopMotionRunner, file_sha256, validate_resume_manifest
from task_support.stair_quiet_v1.runner import (
    StairQuietRunner, canonical, refinement_contract as parent_contract, training_sources as parent_sources)
from task_support.stair_quiet_v1.baselines import PROJECT_ROOT, TEACHER, TEACHER_SHA256, checked_path
from task_support.stair_quiet_v1.baselines import TASK as V1_TASK
from .settings import GEOMETRIES, PROFILES
from .curriculum import WidthCurriculum, validate_outcome_counts

TASK = "stair_quiet_v1_width"
DEFAULT_START = "logs/rsl_rl/deeprobotics_m20_stair_quiet_v1/2026-10-21_01-00-52_quiet_v1_fusion/model_12300.pt"
DEFAULT_START_SHA256 = "d0cc825e4f2a24fb7819df1a08f2ef4845f6d4f85210c08054bedbf5d81bbf17"



def contract(env_cfg, train_cfg):
    result = parent_contract(env_cfg, train_cfg)
    result["task"] = TASK
    result["environment"]["stair_tail"] = canonical(env_cfg.stair_tail)
    result["geometries"] = canonical(GEOMETRIES)
    result["contact_profiles"] = canonical(PROFILES)
    return result


def sources():
    result = parent_sources()
    base = PROJECT_ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity"
    # Hash only runtime dependencies. Disposable validation tools must not
    # become mandatory inputs to every subsequent training checkpoint.
    paths = [Path(__file__).parent / name for name in (
        "__init__.py", "settings.py", "curriculum.py", "geometry.py", "signals.py",
        "observations.py", "terrain.py", "course.py", "runner.py")] + [
        base / "mdp/stair_quiet_v1_width_env.py",
        base / "config/wheeled/deeprobotics_m20/stair_quiet_v1_width_env_cfg.py",
        base / "config/wheeled/deeprobotics_m20/agents/stair_quiet_v1_width_ppo_cfg.py",
        base / "config/wheeled/deeprobotics_m20/__init__.py"]
    result.update({str(p.relative_to(PROJECT_ROOT)): file_sha256(p) for p in paths})
    return result


def checked_counts(raw, reference):
    value = torch.as_tensor(raw, device=reference.device)
    if (value.shape != reference.shape or value.dtype == torch.bool
            or value.is_floating_point() or (value < 0).any()):
        raise ValueError("Curriculum requires nonnegative integer counts with unchanged shape")
    return value.to(torch.long)


def validate_warm_start(saved, teacher):
    prior = validate_resume_manifest(saved.get("m20_training_manifest", {}))
    noise = prior.get("noise_refinement", {})
    if noise and (noise.get("task") != V1_TASK or noise.get("version") != 1
                  or noise.get("teacher_baseline", {}).get("sha256") != TEACHER_SHA256):
        raise ValueError("Warm start must be complete original teacher or teacher-only quiet v1")
    expected = teacher["actor_state_dict"]
    actor = saved.get("actor_state_dict", {})
    if set(actor) != set(expected) or any(
            actor[k].shape != v.shape or not torch.isfinite(actor[k]).all() for k, v in expected.items()):
        raise ValueError("Warm-start actor ABI/nonfinite weights")
    if any(not torch.equal(actor[k], v) for k, v in expected.items() if not k.startswith("residual_mlp.")):
        raise ValueError("Warm-start frozen teacher/distribution differs")
    critic = saved.get("critic_state_dict", {})
    if set(critic) != set(teacher["critic_state_dict"]) or any(
            critic[k].shape != v.shape or not torch.isfinite(critic[k]).all()
            for k, v in teacher["critic_state_dict"].items()):
        raise ValueError("Warm-start critic ABI/nonfinite weights")
    return prior


class StairQuietWidthRunner(StairQuietRunner):
    def initialize_task(self, arguments):
        self._cli_args = copy.deepcopy(arguments)
        if self._restored:
            if any(arguments.get(k) for k in ("init_actor_from", "init_critic_from", "preserve_actor_from")):
                raise ValueError("Full task resume cannot also initialize weights")
            self._initialized = True
            self._install_quiet_schedule()
            return
        if arguments.get("resume") or self.cfg.get("resume"):
            raise ValueError("Resume requires load of this task's full checkpoint")
        if arguments.get("preserve_actor_from") or arguments.get("init_critic_from"):
            raise ValueError("Task loads complete actor+critic and pins original preservation teacher")
        path = self._project_path(arguments.get("init_actor_from") or DEFAULT_START)
        sha = file_sha256(path)
        if not arguments.get("init_actor_from") and sha != DEFAULT_START_SHA256:
            raise ValueError("Default v1 warm-start checkpoint hash changed")
        teacher = torch.load(checked_path(TEACHER, TEACHER_SHA256), map_location="cpu", weights_only=True)
        saved = torch.load(path, map_location="cpu", weights_only=True)
        self._prior_manifest = validate_warm_start(saved, teacher)
        self.alg.actor.load_state_dict(saved["actor_state_dict"], strict=True)
        self.alg.critic.load_state_dict(saved["critic_state_dict"], strict=True)
        self.alg.set_reference(teacher["actor_state_dict"])
        if self.alg.optimizer.state:
            raise RuntimeError("Warm start requires fresh optimizer")
        self.alg.guard_updates = self.current_learning_iteration = 0
        self._initialization = {
            "initial_actor_path": str(path), "initial_actor_sha256": sha,
            "initial_critic_path": str(path), "initial_critic_sha256": sha,
            "critic_initialization": "complete warm-start critic; fresh optimizer; new-objective warmup",
            "preservation_actor_sha256": TEACHER_SHA256}
        self._initialized = True
        self._install_guide_schedule()
        self._install_quiet_schedule()


    def _record_manifest(self):
        from .course import course_state
        manifest = IdleStopMotionRunner._record_manifest(self)
        env = self.env.unwrapped
        frozen = contract(env.cfg, self._quiet_train_cfg)
        state = course_state(env)
        # Replace inherited v1 acoustic claims. This objective measures physical
        # proxies only and is not evidence of real sound reduction or safety.
        manifest["noise_refinement"] = {
            "task": TASK, "version": 1, "contract": frozen,
            "contract_sha256": hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest(),
            "source_sha256": sources(),
            "teacher_baseline": {"path": TEACHER, "sha256": TEACHER_SHA256},
            "wheel_collision_geometry": canonical(env.stair_quiet_geometry),
            "control_contract": self._control_contract(),
            "control_adapter": env.stair_quiet_landing.contract(),
            "terrain_contacts": env.scene.terrain.contact_receipt,
            "wheel_contacts": env.tail_wheel_material_receipt,
            "material_calibration": "uncalibrated effective contact candidates; static boards; no structural modes",
            "acoustic_status": "not_measured", "deployment_status": "teacher_training_only"}
        manifest["tail_curriculum"] = {
            "stage": env.tail_stage,
            "case_metrics": {k: v.cpu().tolist() for k, v in state["case_metrics"].items()},
            **{k: state[k].cpu().tolist() for k in (
                "episodes", "successes", "stage_episodes", "stage_successes", "assign_count")}}
        manifest["width_curriculum"] = {
            **state["width_curriculum"].state_dict(),
            "statistics": {k: v.cpu().tolist() for k, v in state["width_statistics"].items()},
            "resume_semantics": "restart physics at saved next-reset width targets; completed-episode counts preserved"}
        self.alg.m20_training_manifest = copy.deepcopy(manifest)
        return manifest

    def save(self, path, infos=None):
        self._record_manifest()
        return super().save(path, infos)

    def load(self, path, load_cfg=None, strict=True, map_location=None):
        from .course import course_state
        if load_cfg is not None or not strict:
            raise ValueError("Width task supports strict full resume only")
        saved = torch.load(path, map_location="cpu", weights_only=True)
        manifest = saved.get("m20_training_manifest", {})
        noise = manifest.get("noise_refinement", {})
        env = self.env.unwrapped
        frozen = contract(env.cfg, self._quiet_train_cfg)
        digest = hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()
        if (noise.get("task") != TASK or noise.get("version") != 1 or noise.get("contract") != frozen
                or noise.get("contract_sha256") != digest or noise.get("source_sha256") != sources()):
            raise ValueError("Width resume objective/physics/source contract changed; "
                             "start a new run with --init_actor_from and without --resume")
        expected = torch.load(checked_path(TEACHER, TEACHER_SHA256), map_location="cpu", weights_only=True)["actor_state_dict"]
        reference, actor = saved.get("preservation_reference_actor", {}), saved.get("actor_state_dict", {})
        if set(reference) != set(expected) or any(not torch.equal(reference[k], v) for k, v in expected.items()):
            raise ValueError("Resume frozen preservation reference differs")
        if any(k not in actor or not torch.equal(actor[k], v) for k, v in expected.items()
               if not k.startswith("residual_mlp.")):
            raise ValueError("Resume frozen actor differs")
        iteration, completed = saved.get("iter"), saved.get("preservation_updates")
        if (type(iteration) is not int or type(completed) is not int or iteration < 0 or completed < 0
                or not ((iteration == completed == 0) or completed == iteration+1)):
            raise ValueError("Resume iteration/update counter disagreement")
        guide = manifest.get("descent_entry_guide_schedule", {})
        if guide.get("guided_iterations") != 0 or guide.get("decay_iterations") != 0:
            raise ValueError("Width task must not restart old descent guide")
        state = course_state(env)
        tail, width = manifest["tail_curriculum"], manifest["width_curriculum"]
        if type(tail["stage"]) is not int or tail["stage"] not in (0, 1, 2):
            raise ValueError("Invalid saved contact stage")
        restored_width = WidthCurriculum(env.num_envs, env.device, env.cfg.stair_tail)
        restored_width.load_state_dict(width)
        counts = {k: checked_counts(tail[k], state[k]) for k in (
            "episodes", "successes", "stage_episodes", "stage_successes", "assign_count")}
        statistics = {k: checked_counts(width["statistics"][k], v) for k, v in state["width_statistics"].items()}
        validate_outcome_counts(counts, statistics)
        metrics = {}
        for k, target in state["case_metrics"].items():
            value = torch.as_tensor(tail["case_metrics"][k], device=env.device)
            if value.shape != target.shape or not torch.isfinite(value).all() or (value < 0).any():
                raise ValueError("Invalid saved per-case metrics")
            metrics[k] = value
        # Validate all curriculum data before mutating actor/optimizer or scene.
        result = IdleStopMotionRunner.load(self, path, load_cfg, strict, map_location)
        env.tail_stage = tail["stage"]
        for k, value in counts.items():
            state[k].copy_(value)
        for k, value in metrics.items():
            state["case_metrics"][k].copy_(value)
        for k, value in statistics.items():
            state["width_statistics"][k].copy_(value)
        state["width_curriculum"].load_state_dict(restored_width.state_dict())
        env._tail_skip_episode_record = True
        try:
            env.reset()
        finally:
            env._tail_skip_episode_record = False
        self.current_learning_iteration = int(self.alg.guard_updates)
        self._install_quiet_schedule()
        return result
