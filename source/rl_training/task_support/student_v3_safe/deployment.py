"""Student provenance, source receipts and mandatory execution controller."""
from copy import deepcopy
import hashlib

from task_support.stair_quiet_v1.deployment import (
    execution_contract, contract_hash, validate_control_contract, RUNTIME_SCOPE,
)
from .configuration import ROOT, TASK


def source_hashes():
    support = ROOT / "source/rl_training/task_support"
    paths = []
    for folder in ("student_v0", "student_v1", "student_v3_5", "student_v3_safe"):
        paths.extend((support / folder).glob("*.py"))
    task = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity"
    paths.extend([
        task / "mdp/student_v3_safe_env.py",
        task / "config/wheeled/deeprobotics_m20/student_v3_safe_env_cfg.py",
        task / "config/wheeled/deeprobotics_m20/agents/student_v3_safe_ppo_cfg.py",
        task / "config/wheeled/deeprobotics_m20/student_v3_safe/__init__.py",
        ROOT / "scripts/reinforcement_learning/rsl_rl/distill_student.py",
        ROOT / "configs/distillation/student_v3_safe.json",
        ROOT / "deploy/deploy_mujoco/deploy_student_v3_safe.py",
        ROOT / "deploy/deploy_mujoco/student_v1_adapter.py",
        support / "stair_quiet_v1/deployment.py", support / "stair_quiet_v1/distill.py",
    ])
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def provenance_for(provenance, control):
    result = deepcopy(provenance)
    result.update(quiet_execution_control=validate_control_contract(control),
                  quiet_execution_control_sha256=contract_hash(control),
                  quiet_runtime_scope=RUNTIME_SCOPE,
                  nominal_action_label="complete Safe teacher nominal 16D before shared execution controller",
                  student_v3_safe_sources=source_hashes())
    return result


def validate_provenance(provenance):
    control = execution_contract(provenance, required=True)
    if (provenance.get("task") != TASK or provenance.get("task_version") != "student_v3_safe"
            or provenance.get("quiet_execution_control_sha256") != contract_hash(control)
            or provenance.get("contract", {}).get("student_observation_dim") != 789
            or provenance.get("command_feedback", {}).get("version") != "legacy_v1_v3_5"):
        raise ValueError("Invalid student_v3_safe deployment provenance")
    if provenance["command_feedback"].get("tuning") != provenance.get("teacher_manifest", {}).get("stair_tuning"):
        raise ValueError("Safe student command feedback differs from the teacher manifest")
    return control


def validate_resume(saved):
    provenance = saved.get("student_v1", {}).get("provenance", {})
    validate_provenance(provenance)
    if provenance.get("student_v3_safe_sources") != source_hashes():
        raise ValueError("student_v3_safe source/environment changed; use --init-student for a new run")
    return provenance


def export_bundle(runner, directory, provenance):
    import json
    from task_support.stair_quiet_v1.distill import export_bundle as export_quiet
    result = export_quiet(runner, directory, provenance)
    (directory / "quiet_nominal_actor.pt").rename(directory / "student_v3_safe.pt")
    result["deployment_family"] = "m20_student_v3_safe"
    (directory / "deployment.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
