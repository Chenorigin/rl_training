"""CPU-only v0 inference and the SAME synthetic map state used in distillation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
from pathlib import Path
import sys

import mujoco
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "source/rl_training"))
from m20_student_v0.contract import observation_contract, STUDENT_OBS_DIM
from m20_student_v0.perception import StudentV0PerceptionCfg, SyntheticMapState


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_contract(contract):
    expected = observation_contract()
    for key in expected:
        if contract.get(key) != expected[key]:
            raise ValueError(f"Student observation contract mismatch: {key}")


def load_student(checkpoint: Path, jit: bool = False):
    """Load only deterministic student weights; no runner, optimizer or teacher."""
    checkpoint = checkpoint.resolve()
    if jit:
        sidecar = checkpoint.parent / "observation_contract.json"
        validate_contract(json.loads(sidecar.read_text()))
        policy = torch.jit.load(str(checkpoint), map_location="cpu").eval()
        provenance_path = checkpoint.parent / "deployment.json"
        provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
        provenance = provenance.get("training_provenance", provenance)
        source_key = "torchscript"
    else:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if "student_v0" not in saved or "student_state_dict" not in saved:
            raise ValueError("Expected v0 distillation checkpoint with student_state_dict; teacher checkpoints are rejected")
        provenance = saved["student_v0"].get("provenance", {})
        validate_contract(provenance.get("contract", {}))
        cfg = provenance.get("training_config", {}).get("student", {})
        if cfg.get("obs_normalization") is not False or cfg.get("activation") != "elu" or cfg.get("hidden_dims") != [512, 256, 128]:
            raise ValueError("Expected unnormalized 618 -> 512 -> 256 -> 128 -> 16 ELU student")
        policy = torch.nn.Sequential(
            torch.nn.Linear(618, 512), torch.nn.ELU(), torch.nn.Linear(512, 256), torch.nn.ELU(),
            torch.nn.Linear(256, 128), torch.nn.ELU(), torch.nn.Linear(128, 16))
        state = saved["student_state_dict"]
        unexpected = [key for key in state if not key.startswith("mlp.") and key != "distribution.log_std_param"]
        if unexpected:
            raise ValueError(f"Unsupported student state keys: {unexpected}")
        policy.load_state_dict({key.removeprefix("mlp."): value for key, value in state.items() if key.startswith("mlp.")}, strict=True)
        policy.eval().requires_grad_(False)
        source_key = "student_state_dict"
    with torch.inference_mode():
        output = policy(torch.zeros(1, STUDENT_OBS_DIM))
    if output.shape != (1, 16) or not torch.isfinite(output).all():
        raise ValueError("Student policy must accept [1,618] and return finite [1,16]")
    return policy, provenance, {"policy_path": str(checkpoint), "policy_sha256": sha256(checkpoint), "policy_source_state_key": source_key}


@dataclass
class SensorExtras:
    lidar_visibility: bool = False
    sensor_offset_body_m: tuple = (0.10, 0.0, 0.15)
    lidar_horizontal_fov_deg: float = 360.0
    rear_sensor_offset_body_m: tuple | None = None
    gyro_std_rad_s: float = 0.0
    gravity_tilt_std_rad: float = 0.0
    joint_position_std_rad: float = 0.0
    joint_velocity_std_rad_s: float = 0.0
    action_delay_steps: int = 0
    blackout_start_s: float = 3.0
    blackout_period_s: float = 8.0
    blackout_duration_s: float = 0.0

    def validate(self):
        if not isinstance(self.lidar_visibility, bool):
            raise ValueError("lidar_visibility must be boolean")
        if len(self.sensor_offset_body_m) != 3 or not np.isfinite(self.sensor_offset_body_m).all():
            raise ValueError("sensor_offset_body_m must have three finite values")
        if self.rear_sensor_offset_body_m is not None:
            if len(self.rear_sensor_offset_body_m) != 3 or not np.isfinite(self.rear_sensor_offset_body_m).all():
                raise ValueError("rear_sensor_offset_body_m must have three finite values")
        if not isinstance(self.lidar_horizontal_fov_deg, (float, int)) or not math.isfinite(self.lidar_horizontal_fov_deg) or not 0 < self.lidar_horizontal_fov_deg <= 360:
            raise ValueError("lidar_horizontal_fov_deg must be in (0,360]")
        for field in fields(self):
            if field.name in {"lidar_visibility", "sensor_offset_body_m", "rear_sensor_offset_body_m", "lidar_horizontal_fov_deg"}:
                continue
            value = getattr(self, field.name)
            if not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid sensor option: {field.name}")
        if not isinstance(self.action_delay_steps, int) or self.action_delay_steps > 100:
            raise ValueError("action_delay_steps must be an integer in [0,100]")
        if self.blackout_period_s <= 0 or self.blackout_duration_s >= self.blackout_period_s:
            raise ValueError("blackout_duration_s must be shorter than a positive period")


def profile_config(name, provenance, profile_file=None, level=None):
    registry = json.loads(Path(profile_file or Path(__file__).with_name("student_v0_profiles.json")).read_text())
    profile = dict(registry[name])
    base = dict(provenance.get("perception", {}))
    if name == "train" and not base:
        raise ValueError("train profile requires saved perception configuration; use an exported deployment.json or an explicit profile")
    base.update(profile.pop("perception", {}))
    if level is not None:
        base["level"] = level
    cfg = StudentV0PerceptionCfg(**base)
    extras = SensorExtras(**profile)
    cfg.validate()
    extras.validate()
    return cfg, extras


def horizontal_fov_mask(hits, origin, rotation, fov_deg, rear=False):
    """Horizontal sector in the mounted sensor frame (+X front, -X rear).

    Rotation maps body coordinates to world coordinates. The total opening
    angle is fov_deg; vertical FOV and range are not modeled. A restricted
    horizontal sector cannot observe the point directly below its origin.
    """
    if not math.isfinite(fov_deg) or not 0 < fov_deg <= 360:
        raise ValueError("Horizontal FOV must be in (0,360]")
    finite = np.isfinite(hits).all(axis=1)
    if fov_deg == 360:
        return finite
    # Row vectors: world -> body is multiplication by body-to-world R.
    delta_body = (np.where(finite[:, None], hits, origin) - origin) @ rotation
    forward = -delta_body[:, 0] if rear else delta_body[:, 0]
    azimuth = np.arctan2(delta_body[:, 1], forward)
    has_horizontal_direction = np.linalg.norm(delta_body[:, :2], axis=1) > 1e-8
    return finite & has_horizontal_direction & (np.abs(azimuth) <= math.radians(fov_deg / 2) + 1e-10)


def visible_terrain(model, data, base_id, hits, sensor_offset,
                    horizontal_fov_deg=360.0, rear_sensor_offset=None):
    """Geometric terrain line of sight proxy, not a beam LiDAR or fused map.

    Include only group-0 static terrain, excluding robot meshes. Height cell is
    usable if it is in a sensor's horizontal FOV AND its top is the first
    surface seen from that same sensor origin. Front and rear sensors face
    opposite body X directions; their visibility masks are united. The old
    single-origin full-view behavior remains available with default arguments.
    Robot self-shadow, vertical FOV, reflectivity and angular sampling are absent.
    """
    rotation = data.xmat[base_id].reshape(3, 3)
    groups = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
    geom_id = np.zeros(1, dtype=np.int32)
    visible = np.zeros(len(hits), dtype=bool)
    sensors = [(sensor_offset, False)]
    if rear_sensor_offset is not None:
        sensors.append((rear_sensor_offset, True))
    for offset, rear in sensors:
        origin = data.xpos[base_id] + rotation @ np.asarray(offset)
        candidates = horizontal_fov_mask(hits, origin, rotation, horizontal_fov_deg, rear)
        for i in np.flatnonzero(candidates & ~visible):
            delta = hits[i] - origin
            distance = np.linalg.norm(delta)
            if distance < 1e-8:
                visible[i] = True
                continue
            first = mujoco.mj_ray(model, data, origin, delta / distance, groups, True, -1, geom_id)
            visible[i] = first >= 0 and first >= distance - 0.005
    return visible


class StudentMapAdapter:
    def __init__(self, contract, cfg, extras, seed=42):
        self.contract, self.cfg, self.extras = contract, cfg, extras
        self.map = SyntheticMapState(1, "cpu", .02, cfg)
        self.rng = np.random.default_rng(seed)
        self.last_channels = None
        self.last_visibility = np.ones(187, dtype=bool)
        self.last_blackout = False

    def observe(self, data, command, previous_action, step):
        clean = self.contract.observe(data, command, previous_action)
        hits = self.contract.last_scan_hits.copy()
        base = data.xpos[self.contract.base_id].copy()
        rotation = data.xmat[self.contract.base_id].reshape(3, 3)
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
        if self.extras.lidar_visibility:
            self.last_visibility = visible_terrain(
                self.contract.model, data, self.contract.base_id, hits,
                self.extras.sensor_offset_body_m, self.extras.lidar_horizontal_fov_deg,
                self.extras.rear_sensor_offset_body_m)
            hits[~self.last_visibility] = np.nan
        time_s = step * .02
        elapsed = time_s - self.extras.blackout_start_s
        self.last_blackout = elapsed >= 0 and elapsed % self.extras.blackout_period_s < self.extras.blackout_duration_s
        if self.last_blackout:
            # Block new captures; already captured packets retain their actual age.
            self.map.next_capture.fill_(step + 1)
        h, m, c = self.map.snapshot(torch.as_tensor(hits[None], dtype=torch.float32),
                                    torch.as_tensor(base[None], dtype=torch.float32),
                                    torch.tensor([yaw], dtype=torch.float32), step)
        self.last_channels = tuple(channel[0].numpy() for channel in (h, m, c))
        proprio = clean[:57].copy()
        e = self.extras
        proprio[:3] += self.rng.normal(0, e.gyro_std_rad_s * .25, 3)
        # Rotation perturbation keeps projected gravity at unit norm.
        gravity = proprio[3:6].astype(np.float64)
        tilt = self.rng.normal(0, e.gravity_tilt_std_rad, 3)
        gravity += np.cross(tilt, gravity)
        proprio[3:6] = gravity / max(np.linalg.norm(gravity), 1e-8)
        proprio[9:21] += self.rng.normal(0, e.joint_position_std_rad, 12)
        proprio[25:41] += self.rng.normal(0, e.joint_velocity_std_rad_s * .05, 16)
        # Wheel positions 21:25 remain zero; commands/last action are exact.
        obs = np.concatenate((proprio, *self.last_channels)).astype(np.float32)
        if obs.shape != (618,) or not np.isfinite(obs).all():
            raise RuntimeError("Nonfinite or incompatible student observation")
        return obs

    def summary(self):
        return {"perception": asdict(self.cfg), "sensor_extras": asdict(self.extras),
                "confidence_pose_covariance_included": False,
                "map_model": "training synthetic delayed patches + optional front/rear horizontal FOV and geometric terrain visibility; no beam scan, vertical FOV, self-shadow or multi-frame fusion"}
