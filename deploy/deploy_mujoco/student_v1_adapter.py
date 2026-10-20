"""V1 CPU deployment consumes exactly the shared training sensor observation."""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"source/rl_training"))
from m20_student_v1.contract import observation_contract, validate_contract, student_dim
from m20_student_v1.model_io import file_hash as sha256, actor_from_state
from m20_student_v1.perception import StudentObservationState
from m20_student_v1.profiles import profile_config as shared_profile_config, PROFILE_NAMES
from m20_student_v0.noise_profiles import canonical_profile_name


def load_student(checkpoint, jit=False):
    checkpoint = Path(checkpoint).resolve()
    if jit:
        contract = json.loads((checkpoint.parent/"observation_contract.json").read_text())
        validate_contract(contract)
        saved_meta = json.loads((checkpoint.parent/"deployment.json").read_text())
        provenance = saved_meta.get("training_provenance", saved_meta)
        policy = torch.jit.load(str(checkpoint), map_location="cpu").eval()
        source = "torchscript"
    else:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved.get("student_v1", {}).get("version") != "v1":
            raise ValueError("Expected a v1 student checkpoint; teacher/v0 checkpoints are rejected")
        provenance = saved["student_v1"]["provenance"]
        contract = provenance["contract"]
        validate_contract(contract)
        policy = actor_from_state(saved["student_state_dict"], contract["student_observation_dim"], provenance["training_config"]["student"])
        source = "student_state_dict"
    dim = student_dim(contract["history_steps"])
    with torch.inference_mode():
        output = policy(torch.zeros(1, dim))
    if output.shape != (1, 16) or not torch.isfinite(output).all():
        raise ValueError("Invalid v1 policy output")
    return policy, provenance, {"version": "v1", "policy_path": str(checkpoint), "policy_sha256": sha256(checkpoint),
                                "policy_source_state_key": source, "actor_input_dim": dim}


@dataclass
class SensorExtras:
    action_delay_steps: int
    history_steps: int
    command_feedback: dict | None = None


def profile_config(name, provenance, profile_file=None, level=None):
    cfg = shared_profile_config(name, provenance, profile_file, level)
    feedback = provenance.get('command_feedback')
    if provenance.get('task_version') == 'v3.5' and (not feedback or feedback.get('version') != 'legacy_v1_v3_5'):
        raise ValueError('V3.5 deployment requires its saved command feedback contract')
    return cfg, SensorExtras(cfg.action_delay_steps, provenance.get("contract", {}).get("history_steps", 1), feedback)


class StudentMapAdapter:
    def __init__(self, contract, cfg, extras, seed=42):
        self.contract, self.cfg, self.extras = contract, cfg, extras
        self.map = StudentObservationState(1, "cpu", .02, cfg, extras.history_steps)
        self.last_channels = None
        self.last_visibility = np.zeros(187, dtype=bool)
        self.last_blackout = False
        self.heading_feedback = None
        if extras.command_feedback:
            from m20_student_v1.v3_5_feedback import SensorHeadingFeedback
            if extras.command_feedback.get('version') != 'legacy_v1_v3_5':
                raise ValueError('Unsupported student command feedback version')
            self.heading_feedback = SensorHeadingFeedback(1, 'cpu', extras.command_feedback['tuning'])
            self.map.command_transform = self.heading_feedback
        # The entrypoint seeds torch once for each rollout, including model loading.

    def observe(self, data, command, previous_action, step):
        clean = self.contract.observe(data, command, previous_action)
        base = data.xpos[self.contract.base_id]
        rotation = data.xmat[self.contract.base_id].reshape(3, 3)
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
        obs = self.map.observe(torch.as_tensor(clean[None, :57], dtype=torch.float32),
                                torch.as_tensor(self.contract.last_scan_hits[None], dtype=torch.float32),
                                torch.as_tensor(base[None], dtype=torch.float32),
                                torch.tensor([yaw], dtype=torch.float32),
                                torch.as_tensor(rotation[None], dtype=torch.float32), step)
        self.last_channels = tuple(value[0].numpy().copy() for value in self.map.last_channels)
        self.last_visibility = self.map.last_visibility[0].numpy().copy()
        self.last_blackout = step < int(self.map.blackout_until[0])
        return obs[0].numpy().copy()

    def reset(self):
        self.map.reset()
        self.last_channels = None
        self.last_visibility.fill(False)
        self.last_blackout = False
        if self.heading_feedback is not None:
            self.heading_feedback.reset()

    def reset_for_policy_handover(self):
        """Start a fresh sensor episode before the first manual policy action."""
        self.reset()

    def summary(self):
        return {"perception": asdict(self.cfg), "history_steps": self.extras.history_steps,
                "command_feedback": self.extras.command_feedback,
                "confidence_pose_covariance_included": True,
                "parameters_calibrated_on_hardware": False,
                "map_model": "shared training 2.5D FOV/terrain visibility + fixed acquisition packets + per-cell history/age; no beam scan, reflectivity, overhangs or robot self-shadow"}
