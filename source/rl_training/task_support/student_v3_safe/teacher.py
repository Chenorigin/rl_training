"""Strict adapter of the original Safe actor; no teacher weights are dropped."""
from copy import deepcopy
import hashlib

import torch
from tensordict import TensorDict

from task_support.stair_quiet_v1_safe.actor import SafeResidualActor
from task_support.stair_quiet_v1.deployment import execution_contract
from .configuration import ROOT, TEACHER_CLASS

INPUTS = {"policy": 244, "motion_control": 3, "locomotion_scan": 187, "safety": 48}
ACTOR_CLASS = "task_support.stair_quiet_v1_safe.actor:SafeResidualActor"


def teacher_manifest(saved, resume=False):
    manifest = (saved.get("student_v1", {}).get("provenance", {}).get("teacher_manifest")
                if resume else saved.get("m20_training_manifest"))
    manifest = deepcopy(manifest or {})
    if (manifest.get("actor_input_groups") != INPUTS
            or manifest.get("actor_model_config", {}).get("class_name") != ACTOR_CLASS
            or manifest.get("noise_refinement", {}).get("task") != "stair_quiet_v1_safe"):
        raise ValueError("student_v3_safe requires a complete v1_safe teacher, including safety48")
    cfg = manifest["actor_model_config"]
    if (cfg.get("hidden_dims") != [512, 256, 128] or cfg.get("residual_hidden_dims") != [128, 64]
            or cfg.get("activation") != "elu" or cfg.get("obs_normalization") is not False
            or manifest.get("motion_control_version") != "m20_raw_command_gate_20261017"):
        raise ValueError("Unsupported Safe teacher architecture/command contract")
    noise = manifest["noise_refinement"]
    frozen = noise.get("contract", {})
    import json
    digest = hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()
    if digest != noise.get("contract_sha256"):
        raise ValueError("Safe teacher environment contract hash mismatch")
    # Preserve current teacher physics and routing. Never silently train on a
    # changed implementation after accepting an older teacher checkpoint.
    sources = noise.get("source_sha256", {})
    required = "source/rl_training/task_support/stair_quiet_v1_safe/actor.py"
    if required not in sources:
        raise ValueError("Safe teacher source hash receipt is missing")
    for relative, expected in sources.items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Safe teacher source hash changed: {relative}")
    execution_contract({"teacher_manifest": manifest}, required=True)
    from m20_student_v1.v3_5_feedback import LegacyHeadingFeedback
    LegacyHeadingFeedback(1, "cpu", manifest.get("stair_tuning", {}))
    return manifest


def teacher_config(manifest):
    result = deepcopy(manifest["actor_model_config"])
    result["class_name"] = TEACHER_CLASS
    return result


class SafeTeacherModel(SafeResidualActor):
    def __init__(self, obs, obs_groups, obs_set, output_dim, **kwargs):
        if obs_set != "teacher":
            raise ValueError("SafeTeacherModel is exclusively a frozen teacher")
        super().__init__(self.pack(obs), {"teacher": ["policy"]}, "teacher", output_dim, **kwargs)
        self.requires_grad_(False).eval()

    @staticmethod
    def pack(obs):
        result = {name: obs["teacher" if name == "policy" else name] for name in INPUTS}
        for name, dim in INPUTS.items():
            value = result[name]
            if value.shape != (*obs.batch_size, dim) or not torch.isfinite(value).all():
                raise ValueError(f"Invalid Safe teacher input: {name}")
        return TensorDict(result, batch_size=obs.batch_size)

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        return super().forward(self.pack(obs), masks, hidden_state, stochastic_output)
