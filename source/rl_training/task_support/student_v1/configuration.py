"""V1 numerical ledger. Values are engineering defaults pending real sensor calibration."""
from copy import deepcopy
from dataclasses import dataclass, field, fields
import math

from m20_student_v0.configuration import model_cfg


@dataclass
class StudentV1PerceptionCfg:
    level: float = 0.5
    # Mixture changes error severity, never hardware FOV or update period.
    clean_probability: float = 0.15
    stress_probability: float = 0.15
    stress_multiplier: float = 2.0
    lidar_horizontal_fov_deg: float = 135.0
    # Optional yaw-aligned map output restriction; disabled for existing profiles.
    front_only: bool = False
    front_blind_distance_m: float = 0.0
    front_sensor_offset_m: tuple = (0.10, 0.0, 0.15)
    rear_sensor_offset_m: tuple = (-0.10, 0.0, 0.15)
    lidar_min_range_m: float = 0.05
    lidar_max_range_m: float = 5.0
    vertical_fov_min_deg: float = -90.0
    vertical_fov_max_deg: float = 90.0
    terrain_visibility: bool = True
    visibility_samples: int = 16
    visibility_tolerance_m: float = 0.015
    map_update_period_s: float = 0.10
    update_jitter_s: float = 0.02
    min_delay_s: float = 0.0
    max_delay_s: float = 0.06
    max_map_age_s: float = 1.00
    packet_dropout: float = 0.05
    burst_probability: float = 0.02
    burst_duration_s: float = 0.30
    point_dropout: float = 0.03
    occlusion_probability: float = 0.30
    occlusion_width_m: float = 0.40
    occlusion_height_m: float = 0.30
    occlusion_hold_s: float = 0.50
    height_noise_std: float = 0.02
    height_bias_std: float = 0.015
    correlated_height_std: float = 0.01
    map_artifact_probability: float = 0.03
    map_artifact_height_std: float = 0.05
    xy_bias_std: float = 0.015
    yaw_bias_std: float = 0.01
    pose_drift_xy_std: float = 0.015
    pose_drift_yaw_std: float = 0.01
    drift_time_constant_s: float = 2.0
    roll_pitch_std: float = 0.01
    gyro_std_rad_s: float = 0.015
    gyro_bias_std_rad_s: float = 0.005
    joint_position_std_rad: float = 0.002
    joint_velocity_std_rad_s: float = 0.05
    action_delay_steps: int = 1
    variance_growth_per_s: float = 0.001
    confidence_reference_std: float = 0.05
    confidence_age_tau_s: float = 0.50
    # Matches the mapper's confidence-only step-location ambiguity weight.
    confidence_step_ambiguity_weight: float = 1.0
    confidence_log_std: float = 0.20
    confidence_overestimate_probability: float = 0.02
    confidence_overestimate_factor: float = 0.25

    def validate(self):
        if not isinstance(self.front_only, bool):
            raise ValueError("front_only must be a boolean")
        if isinstance(self.front_blind_distance_m, bool):
            raise ValueError("front_blind_distance_m must be a distance, not a boolean")
        for item in fields(self):
            value = getattr(self, item.name)
            if item.name.endswith("offset_m"):
                if len(value) != 3 or not all(math.isfinite(x) for x in value):
                    raise ValueError(f"Invalid {item.name}")
            elif isinstance(value, bool):
                continue
            elif not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError(f"{item.name} must be finite")
            elif item.name not in ("vertical_fov_min_deg", "vertical_fov_max_deg") and value < 0:
                raise ValueError(f"{item.name} must be nonnegative")
        for item in fields(self):
            if "probability" in item.name or item.name in ("level", "packet_dropout", "point_dropout"):
                if not 0 <= getattr(self, item.name) <= 1:
                    raise ValueError(f"{item.name} must be in [0,1]")
        if self.clean_probability + self.stress_probability > 1:
            raise ValueError("Mixture probabilities exceed one")
        if not 0 < self.lidar_horizontal_fov_deg <= 360:
            raise ValueError("Invalid horizontal FOV")
        if not -90 <= self.vertical_fov_min_deg < self.vertical_fov_max_deg <= 90:
            raise ValueError("Invalid vertical FOV")
        if not self.lidar_min_range_m < self.lidar_max_range_m:
            raise ValueError("Invalid LiDAR range")
        for key in ("map_update_period_s", "max_map_age_s", "drift_time_constant_s",
                    "confidence_reference_std", "confidence_age_tau_s", "occlusion_hold_s"):
            if getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive")
        if self.update_jitter_s >= self.map_update_period_s:
            raise ValueError("Update jitter must be smaller than update period")
        if self.min_delay_s > self.max_delay_s:
            raise ValueError("Minimum delay must not exceed maximum delay")
        if not 0 <= self.confidence_step_ambiguity_weight <= 1:
            raise ValueError("Confidence step ambiguity weight must be in [0,1]")
        if not isinstance(self.visibility_samples, int) or not 2 <= self.visibility_samples <= 64:
            raise ValueError("visibility_samples must be in [2,64]")
        if not isinstance(self.action_delay_steps, int) or not 0 <= self.action_delay_steps <= 10:
            raise ValueError("action_delay_steps must be in [0,10]")
        if self.stress_multiplier < 1 or not 0 < self.confidence_overestimate_factor <= 1:
            raise ValueError("Invalid stress/confidence multiplier")


@dataclass
class DeeproboticsM20StudentV1DistillationCfg:
    seed: int = 42
    num_steps_per_env: int = 32
    max_iterations: int = 10000
    save_interval: int = 200
    experiment_name: str = "deeprobotics_m20_student_v1"
    logger: str = "tensorboard"
    check_for_nan: bool = True
    clip_actions: float = 100.0
    history_steps: int = 1
    student: dict = field(default_factory=model_cfg)
    teacher: dict = field(default_factory=model_cfg)
    obs_groups: dict = field(default_factory=lambda: {"student": ["policy"], "teacher": ["teacher"]})
    algorithm: dict = field(default_factory=lambda: {
        "class_name": "m20_student_v1.distillation.StudentV1Distillation",
        "learning_rate": 5e-5, "num_learning_epochs": 2, "max_grad_norm": 1.0,
        "batch_size": 1024, "replay_rollouts": 8, "hard_replay_rollouts": 2,
        # Keep replay in memory; ordinary checkpoints resume with fresh replay.
        "save_replay": False,
        "teacher_ratio": 0.5, "teacher_ratio_end": 0.0,
        "teacher_warmup_updates": 1000, "teacher_decay_updates": 7000,
        "student_failure_gate": 0.20, "gate_min_episodes": 32,
        "leg_loss_weight": 1.0, "wheel_loss_weight": 1.0,
        "landing_loss_weight": 2.0, "recovery_loss_weight": 2.0,
        "stratified_fraction": 0.5, "hard_fraction": 0.25,
        "noise_level_start": 0.5, "noise_level_end": 1.0, "noise_ramp_updates": 4000,
        "rnd_cfg": None, "symmetry_cfg": None, "symmetry_loss_weight": 0.0,
    })

    def to_dict(self):
        return deepcopy(self.__dict__)
