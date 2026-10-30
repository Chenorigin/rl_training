"""Dedicated quiet student CLI; training starts only when invoked by the user.

  python -m task_support.stair_quiet_v1.distill --teacher-checkpoint QUIET.pt --preflight-only --device cpu

The teacher's nominal 16D labels are distilled. Both rollout policies execute
the identical environment landing controller. This is not gain distillation.
The old SDK cannot deploy the resulting actor without its execution controller.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path
import sys

from .baselines import PROJECT_ROOT, STUDENT, STUDENT_SHA256, checked_path
from .deployment import (
    STUDENT_TASK, execution_contract, add_execution_provenance, validate_provenance,
    reject_unsupported_quiet_checkpoint,
)
from .landing import LandingSettings, control_contract


def check_source(task, saved, *, resume=False):
    if task != STUDENT_TASK:
        reject_unsupported_quiet_checkpoint(saved, "Legacy student distillation")
        return None
    contract = execution_contract(saved, required=True)
    if resume:
        provenance = saved.get("student_v1", {}).get("provenance", {})
        validate_provenance(provenance)
        if provenance.get("quiet_distillation_sources") != source_hashes():
            raise ValueError("Quiet student resume source/environment changed; start a new experiment")
    else:
        noise = saved.get("m20_training_manifest", {}).get("noise_refinement", {})
        if noise.get("control_contract") != contract:
            raise ValueError("Quiet distillation requires a refined teacher with its control contract")
    return contract


def source_hashes():
    task = PROJECT_ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity"
    files = [
        Path(__file__), Path(__file__).with_name("deployment.py"),
        PROJECT_ROOT / "scripts/reinforcement_learning/rsl_rl/distill_student.py",
        task / "mdp/stair_quiet_v1_env.py",
        task / "config/wheeled/deeprobotics_m20/stair_quiet_v1_env_cfg.py",
        task / "config/wheeled/deeprobotics_m20/student_v1_env_cfg.py",
        task / "config/wheeled/deeprobotics_m20/agents/stair_quiet_v1_ppo_cfg.py",
        task / "config/wheeled/deeprobotics_m20/student_v3_env_cfg.py",
        task / "config/wheeled/deeprobotics_m20/student_v3_5_env_cfg.py",
        task / "config/wheeled/deeprobotics_m20/quiet_landing_env_cfg.py",
        task / "config/wheeled/deeprobotics_m20/idle_stop_motion_v1_env_cfg.py",
        task / "config/wheeled/deeprobotics_m20/rough_env_cfg.py",
        task / "velocity_env_cfg.py",
        PROJECT_ROOT / "source/rl_training/rl_training/assets/deeprobotics.py",
    ]
    files.extend(Path(__file__).with_name(name) for name in (
        "landing.py", "isaac_landing.py", "signals.py", "terrain_memory.py",
    ))
    support = PROJECT_ROOT / "source/rl_training/task_support"
    for folder in ("student_v0", "student_v1", "student_v3", "student_v3_5"):
        files.extend(sorted((support / folder).glob("*.py")))
    return {str(path.relative_to(PROJECT_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(files)}


def prepare_initialization(args):
    if args.task == STUDENT_TASK and not args.resume:
        args.init_student = checked_path(args.init_student or STUDENT, STUDENT_SHA256)


def configure_environment(cfg, contract):
    cfg.landing_control = LandingSettings(**contract["settings"])
    if control_contract(cfg.landing_control) != contract:
        raise ValueError("Quiet student environment would alter the teacher controller")


def provenance_for(provenance, contract):
    result = add_execution_provenance(provenance, contract)
    result["quiet_distillation_sources"] = source_hashes()
    return result


def export_bundle(runner, export_dir, provenance):
    """Export an explicitly incomplete nominal actor with mandatory sidecars."""
    import json
    import torch
    validate_provenance(provenance)
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=False)
    actor_path = export_dir / "quiet_nominal_actor.pt"
    runner.export_policy_to_jit(str(export_dir), actor_path.name)
    # Embed the execution contract into the TorchScript archive as well as
    # sidecars so metadata travels with the nominal actor.
    actor = torch.jit.load(str(actor_path), map_location="cpu")
    embedded = json.dumps(provenance["quiet_execution_control"], sort_keys=True, allow_nan=False)
    torch.jit.save(actor, str(actor_path), _extra_files={"quiet_execution_control.json": embedded})
    contract = deepcopy(provenance["contract"])
    contract["quiet_execution_control"] = provenance["quiet_execution_control"]
    (export_dir / "observation_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    deployment = {
        "deployment_family": "m20_stair_quiet_v1_nominal_actor",
        "training_provenance": provenance,
        "quiet_execution_control": provenance["quiet_execution_control"],
        "actor_sha256": hashlib.sha256(actor_path.read_bytes()).hexdigest(),
        "standalone_controller": False, "hardware_approved": False,
    }
    (export_dir / "deployment.json").write_text(json.dumps(deployment, indent=2, allow_nan=False) + "\n")
    samples = torch.randn(16, provenance["contract"]["student_observation_dim"])
    from tensordict import TensorDict
    with torch.inference_mode():
        expected = runner.alg.student(TensorDict({"policy": samples.to(runner.device)}, [16])).cpu()
        actual = actor(samples)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-5)
    return deployment


def main(argv=None):
    path = PROJECT_ROOT / "scripts/reinforcement_learning/rsl_rl/distill_student.py"
    spec = importlib.util.spec_from_file_location("stair_quiet_distill_entry", path)
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    entry.main(sys.argv[1:] if argv is None else argv, default_task=STUDENT_TASK)


if __name__ == "__main__":
    main()
