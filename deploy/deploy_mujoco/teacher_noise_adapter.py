"""Existing student perception presets projected onto the teacher's 244D ABI."""
from collections import deque
from dataclasses import asdict
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from student_v1_adapter import StudentMapAdapter, SensorExtras, sha256
from m20_student_v1.profiles import PROFILE_NAMES as V1_PROFILE_NAMES, profile_config
from m20_student_v1.configuration import StudentV1PerceptionCfg
from m20_student_v1.perception import map_output_mask
from m20_student_v0.noise_profiles import canonical_profile_name, profile_entry
from task_support.stair_quiet_v1.baselines import PROJECT_ROOT, STUDENT

DEFAULT_NOISE_REFERENCE = PROJECT_ROOT / STUDENT
CLEAN_FRONT_PROFILE = "front_only_clean"
PROFILE_NAMES = (*V1_PROFILE_NAMES, CLEAN_FRONT_PROFILE)


def add_noise_arguments(parser):
    parser.add_argument("--noise-profile", type=canonical_profile_name, choices=PROFILE_NAMES,
                        default="teacher", help="Existing student sensor preset; teacher keeps clean observations")
    parser.add_argument("--profile-file", type=Path, help="Existing v1 JSON preset overrides")
    parser.add_argument("--noise-level", type=float)
    parser.add_argument('--map-capture-padding',type=float,default=None,
                        help='Extra measured map support in metres; CTS defaults to its capture ledger, 0 replays old geometry')
    parser.add_argument("--noise-reference", type=Path, default=DEFAULT_NOISE_REFERENCE,
                        help="Student checkpoint or deployment.json supplying saved perception settings")
    parser.add_argument("--seed", type=int, default=42)


def validate_noise_arguments(args, parser):
    pad = getattr(args,'map_capture_padding',None)
    if pad is not None and (not math.isfinite(pad) or not 0 <= pad <= 2.):
        parser.error('map-capture-padding must be finite in [0,2]')
    if args.noise_level is not None and (not math.isfinite(args.noise_level) or not 0 <= args.noise_level <= 1):
        parser.error("noise-level must be in [0,1]")
    if args.noise_profile == CLEAN_FRONT_PROFILE and args.noise_level not in (None, 0):
        parser.error("front_only_clean is noiseless; noise-level must be 0 or omitted")


def read_reference(path):
    path = Path(path).expanduser().resolve()
    saved = (json.loads(path.read_text()) if path.suffix == ".json" else
             torch.load(path, map_location="cpu", weights_only=True))
    if not isinstance(saved, dict):
        raise ValueError("noise-reference must contain saved v1 student perception settings")
    provenance = saved.get("student_v1", {}).get("provenance", saved.get("training_provenance", saved))
    if not isinstance(provenance, dict) or not isinstance(provenance.get("perception"), dict) or not provenance["perception"]:
        raise ValueError("noise-reference must contain saved v1 student perception settings")
    return provenance, {"path": str(path), "sha256": sha256(path)}


class TeacherNoiseAdapter:
    def __init__(self, contract, args):
        self.contract = contract
        self.profile = canonical_profile_name(args.noise_profile)
        provenance, self.reference = read_reference(args.noise_reference)
        self.cfg = profile_config(self.profile, provenance, args.profile_file, args.noise_level)
        padding = getattr(args,'map_capture_padding',None)
        if padding is not None:
            self.cfg.map_capture_padding_m = padding
            self.cfg.validate()
        self.seed = args.seed
        self.profile_file = ({"path": str(args.profile_file.resolve()), "sha256": sha256(args.profile_file)}
                             if args.profile_file is not None else None)
        torch.manual_seed(self.seed)
        # The teacher already applies its own checkpoint-specific heading loop.
        # Reuse sensor errors, without a second student command-feedback loop.
        self.map_adapter = StudentMapAdapter(contract, self.cfg, SensorExtras(self.cfg.action_delay_steps, 1), self.seed)
        self.steps = self.valid_samples = 0
        self.reset()
        print(f"[NOISE] profile={self.profile}; reference={self.reference['path']}; "
              f"action_delay_steps={self.cfg.action_delay_steps}; seed={self.seed}", flush=True)

    def reset(self):
        self.map_adapter.reset_for_policy_handover()
        self.delay = deque(np.zeros(16, dtype=np.float32) for _ in range(self.cfg.action_delay_steps))

    def observe(self, data, command, previous_action, step):
        complete = self.map_adapter.observe(data, command, previous_action, step)
        # Keep exactly the existing teacher ordering: noisy proprio + height.
        # Missing cells stay zero, never replaced by clean rays. Mask/confidence
        # remain available to the landing controller, outside the teacher ABI.
        observation = complete[:244].copy()
        self.steps += 1
        self.valid_samples += int(np.count_nonzero(self.map_adapter.last_channels[1]))
        if observation.shape != (244,) or not np.isfinite(observation).all():
            raise RuntimeError("Invalid noisy teacher observation")
        return observation

    def action(self, nominal):
        self.delay.append(nominal.copy())
        return self.delay.popleft()

    def terrain_points(self, data):
        from task_support.stair_quiet_v1.mujoco_landing import map_points
        points, valid = map_points(self.contract, data, self.map_adapter)
        points[~valid] = np.nan
        return points

    def draw(self, viewer, data, visible):
        if not visible or self.map_adapter.last_channels is None:
            with viewer.lock():
                viewer.user_scn.ngeom = 0
            return
        from deploy_student_v0 import draw_student_map
        draw_student_map(viewer, self.contract, self.map_adapter.last_channels,
                         float(data.xpos[self.contract.base_id, 2]))

    def summary(self):
        return {"enabled": True, "reference": self.reference, "profile_file": self.profile_file,
                "seed": self.seed, "teacher_input_dim": 244, "policy_steps": self.steps,
                "valid_fraction": self.valid_samples / (187 * self.steps) if self.steps else None,
                **self.map_adapter.summary()}


class CleanFrontTeacherAdapter(TeacherNoiseAdapter):
    """Instantaneous teacher observations with exactly the front_only XY mask."""

    def __init__(self, contract, args):
        self.contract, self.profile, self.seed = contract, CLEAN_FRONT_PROFILE, args.seed
        provenance, self.reference = read_reference(args.noise_reference)
        front = profile_config("front_only", provenance)
        values = asdict(profile_config("teacher", provenance))
        values.update(front_only=True, front_blind_distance_m=front.front_blind_distance_m,
                      front_sensor_offset_m=front.front_sensor_offset_m,
                      lidar_horizontal_fov_deg=front.lidar_horizontal_fov_deg,
                      level=0., clean_probability=1., stress_probability=0.)
        self.profile_file = None
        if args.profile_file is not None:
            override = profile_entry(json.loads(args.profile_file.read_text()), self.profile)
            allowed = {"front_sensor_offset_m", "lidar_horizontal_fov_deg"}
            if not isinstance(override, dict) or set(override) - allowed:
                raise ValueError("front_only_clean overrides only front_sensor_offset_m and lidar_horizontal_fov_deg")
            values.update(override)
            self.profile_file = {"path": str(args.profile_file.resolve()), "sha256": sha256(args.profile_file)}
        if args.noise_level not in (None, 0):
            raise ValueError("front_only_clean is noiseless; noise-level must be 0 or omitted")
        self.cfg = StudentV1PerceptionCfg(**values)
        self.cfg.validate()
        self.keep = map_output_mask(torch.as_tensor(contract.grid, dtype=torch.float32), self.cfg).numpy()
        # Match the existing landing adapter's delivered-map interface, without
        # entering a stochastic capture/reprojection/confidence pipeline.
        self.map_adapter = SimpleNamespace(last_channels=None, map=SimpleNamespace(
            cell_age=torch.full((1, 187), torch.inf)))
        self.steps = self.valid_samples = 0
        self.reset()
        print(f"[NOISE] profile={self.profile}; spatial mask=front_only; noise=off; delay=0", flush=True)

    def reset(self):
        self.map_adapter.last_channels = None
        self.map_adapter.map.cell_age.fill_(torch.inf)

    def observe(self, data, command, previous_action, step):
        clean = self.contract.observe(data, command, previous_action)
        valid = self.keep & np.isfinite(self.contract.last_scan_hits).all(-1)
        observation = clean.copy()
        observation[57:] = np.where(valid, clean[57:], 0.)
        if observation.shape != (244,) or not np.isfinite(observation).all():
            raise RuntimeError("Invalid front-only clean teacher observation")
        known = valid.astype(np.float32)
        self.map_adapter.last_channels = (observation[57:].copy(), known, known.copy())
        self.map_adapter.map.cell_age[0] = torch.from_numpy(np.where(valid, 0., np.inf).astype(np.float32))
        self.steps += 1
        self.valid_samples += int(valid.sum())
        return observation

    def action(self, nominal):
        return nominal.copy()

    def summary(self):
        return {"enabled": True, "noise_added": False, "reference": self.reference,
                "reference_usage": "front_only_spatial_settings_only", "profile_file": self.profile_file,
                "seed": self.seed, "teacher_input_dim": 244, "policy_steps": self.steps,
                "valid_fraction": self.valid_samples / (187 * self.steps) if self.steps else None,
                "perception": asdict(self.cfg), "history_steps": 1,
                "map_model": "instantaneous_clean_teacher_height_with_front_only_mask",
                "confidence_semantics": "one_for_current_valid_cells_zero_for_missing"}


def make_noise_adapter(args, contract):
    profile = canonical_profile_name(getattr(args, "noise_profile", "teacher"))
    if profile == "teacher" and getattr(args, "profile_file", None) is None and getattr(args, "noise_level", None) is None:
        return None
    if profile == CLEAN_FRONT_PROFILE:
        return CleanFrontTeacherAdapter(contract, args)
    return TeacherNoiseAdapter(contract, args)


def noise_metadata(args, adapter):
    return {"noise_profile": canonical_profile_name(getattr(args, "noise_profile", "teacher")),
            "noise": adapter.summary() if adapter is not None else {"enabled": False}}
