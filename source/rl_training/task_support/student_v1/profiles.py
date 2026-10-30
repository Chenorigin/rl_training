"""Controlled single-factor profiles derived from the v1 parameter ledger."""
from copy import deepcopy
from dataclasses import asdict

from .configuration import StudentV1PerceptionCfg
from m20_student_v0.noise_profiles import PRIMARY_PROFILE_NAMES, canonical_profile_name, profile_entry

HEIGHT = ("height_noise_std", "height_bias_std", "correlated_height_std", "map_artifact_probability", "map_artifact_height_std")
POSE = ("xy_bias_std", "yaw_bias_std", "pose_drift_xy_std", "pose_drift_yaw_std", "roll_pitch_std")
PROPRIO = ("gyro_std_rad_s", "gyro_bias_std_rad_s", "joint_position_std_rad", "joint_velocity_std_rad_s")
DROPOUT = ("point_dropout", "packet_dropout", "burst_probability", "occlusion_probability")
TIMING = ("map_update_period_s", "update_jitter_s", "min_delay_s", "max_delay_s")
QUALITY_ERROR = ("confidence_log_std", "confidence_overestimate_probability")
FOV = ("lidar_horizontal_fov_deg", "front_sensor_offset_m", "rear_sensor_offset_m", "terrain_visibility",
       "lidar_min_range_m", "lidar_max_range_m", "vertical_fov_min_deg", "vertical_fov_max_deg")
FRONT_ONLY_PROFILE = "front_only"
LEGACY_FRONT_ONLY_PROFILE = "front_only_30cm"
FRONT_BLIND_DISTANCE_M = 0.0
# Current factory installation candidate in m20_heightmap_ws/deploy/site.json.
# This is deployment configuration, not an independently measured calibration.
FRONT_DEPLOYMENT_OFFSET_M = (0.32028, 0.0, -0.013)
PROFILE_NAMES = (*PRIMARY_PROFILE_NAMES, FRONT_ONLY_PROFILE, LEGACY_FRONT_ONLY_PROFILE, "lidar_fov", "height", "pose", "dropout", "latency", "proprio", "action_delay", "quality_reference", "confidence", "stress")


def profile_config(name, provenance, profile_file=None, level=None):
    name = canonical_profile_name(name)
    front_only = name in (FRONT_ONLY_PROFILE, LEGACY_FRONT_ONLY_PROFILE)
    if name not in PROFILE_NAMES:
        raise ValueError(f"Unknown v1 profile: {name}")
    saved = provenance.get("perception", {})
    if (name == "student-v1" or front_only) and not saved:
        raise ValueError(f"{name} profile requires v1 training provenance")
    nominal = asdict(StudentV1PerceptionCfg())
    nominal.update(saved)
    nominal.update(clean_probability=0., stress_probability=0., level=1.)
    if name == "student-v1" or front_only:
        values = asdict(StudentV1PerceptionCfg())
        values.update(saved)
    elif name in ("student-v0", "stress"):
        values = deepcopy(nominal)
        if name == "stress":
            values.update(clean_probability=0., stress_probability=1.)
    else:
        values = deepcopy(nominal)
        for key in (*HEIGHT, *POSE, *PROPRIO, *DROPOUT, *QUALITY_ERROR):
            values[key] = 0.
        values.update(map_update_period_s=.02, update_jitter_s=0., min_delay_s=0., max_delay_s=0.,
                      zero_delay_probability=0.,
                      action_delay_steps=0, lidar_horizontal_fov_deg=360., terrain_visibility=False,
                      variance_growth_per_s=0.)
        group = {"lidar_fov": FOV, "height": HEIGHT, "pose": POSE, "dropout": DROPOUT,
                 "latency": TIMING, "proprio": PROPRIO, "action_delay": ("action_delay_steps",),
                 "quality_reference": (*HEIGHT, *POSE),
                 "confidence": (*HEIGHT, *POSE, *QUALITY_ERROR)}.get(name, ())
        for key in group:
            values[key] = nominal[key]
    if front_only:
        values["front_sensor_offset_m"] = FRONT_DEPLOYMENT_OFFSET_M
    if profile_file:
        import json
        from pathlib import Path
        # Explicit overrides, same schema and validation as training --config.
        values.update(profile_entry(json.loads(Path(profile_file).read_text()), name))
    if front_only:
        values.update(front_only=True, front_blind_distance_m=FRONT_BLIND_DISTANCE_M)
    if level is not None:
        values["level"] = level
    cfg = StudentV1PerceptionCfg(**values)
    cfg.validate()
    return cfg
