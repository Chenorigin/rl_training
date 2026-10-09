"""M20 stair-teacher gait envelopes. No changes to the actor observation ABI.

TUNING is the numeric source for these additions. Thresholds are conservative
soft envelopes, not mandatory joint trajectories or physical safety limits.
"""
from __future__ import annotations

import torch

TUNING = {
    "translation_threshold": 0.10, "yaw_threshold": 0.10,
    "turn_posture_scale": 0.20,
    "air_min": 0.10, "air_target": 0.20, "air_max": 0.35,
    "turn_height_flat": 0.08, "turn_height_terrain": 0.16,
    "turn_reach_flat": 0.22, "turn_reach_terrain": 0.30,
    "height_scale": 0.08, "reach_scale": 0.10,
    "rear_min_extension": 0.35, "extension_scale": 0.10,
    "rear_max_knee": 1.90, "knee_scale": 0.50,
    "support_grace": 0.12, "same_tread_grace": 0.12,
    "wheel_radius": 0.09, "nearest_valid_distance": 0.12,
    "turn_size_weight": -1.0, "rear_fold_weight": -1.0,
    "same_tread_event_weight": -1.0, "same_tread_dwell_weight": -1.0,
    "turn_status_weight": 1.0, "turn_symmetry_weight": 2.0,
}


def pure_turn(command):
    return ((torch.linalg.vector_norm(command[:, :2], dim=1) < TUNING["translation_threshold"])
            & (command[:, 2].abs() > TUNING["yaw_threshold"]))


def turn_tracking_quality(yaw, command):
    """Stationary leg dancing or turning opposite to the command earns zero."""
    progress = (yaw * command.sign() / command.abs().clamp_min(TUNING["yaw_threshold"])).clamp(0, 1)
    return progress * torch.exp(-((yaw - command) / 0.5).square())


def bounded_air_score(air, low, target, high):
    """Time-valued triangle; zero for nonflight or prolonged flight."""
    peak = target - low
    return torch.minimum((air - low).clamp_min(0),
                         ((high - air) / (high - target) * peak).clamp_min(0))


def fold_excess(extension, knee):
    length = ((TUNING["rear_min_extension"] - extension).clamp_min(0)
              / TUNING["extension_scale"]).square()
    angle = ((knee.abs() - TUNING["rear_max_knee"]).clamp_min(0)
             / TUNING["knee_scale"]).square()
    return torch.maximum(length, angle).clamp_max(1)


def turn_size_excess(clearance, reach, terrain):
    height_limit = TUNING["turn_height_flat"] + terrain[:, None] * (
        TUNING["turn_height_terrain"] - TUNING["turn_height_flat"])
    reach_limit = TUNING["turn_reach_flat"] + terrain[:, None] * (
        TUNING["turn_reach_terrain"] - TUNING["turn_reach_flat"])
    return (((clearance - height_limit).clamp_min(0) / TUNING["height_scale"]).square()
            + ((reach - reach_limit).clamp_min(0) / TUNING["reach_scale"]).square()).clamp_max(1)


def _context(env, asset_cfg, hip_cfg, knee_cfg, sensor_cfg, contact_sensor_cfg,
             command_name="base_velocity"):
    step = int(env.common_step_counter)
    cached = getattr(env, "_m20_gait_context", None)
    if cached is not None and cached["step"] == step:
        return cached
    from .stair_teacher import _stair_scan_context
    scan = _stair_scan_context(env, sensor_cfg, command_name)
    robot = env.scene[asset_cfg.name]
    contact = env.scene.sensors[contact_sensor_cfg.name]
    wheels = robot.data.body_pos_w[:, asset_cfg.body_ids]
    hips = robot.data.body_pos_w[:, hip_cfg.body_ids]
    hits = env.scene[sensor_cfg.name].data.ray_hits_w
    valid = torch.isfinite(hits).all(dim=-1)
    distance = torch.linalg.vector_norm(wheels[:, :, None, :2] - hits[:, None, :, :2], dim=-1)
    distance = torch.where(valid[:, None], distance, torch.inf)
    closest, index = distance.min(dim=-1)
    ground = hits[..., 2].gather(1, index)
    ground_valid = (closest < TUNING["nearest_valid_distance"]) & torch.isfinite(ground)
    # Missing terrain cannot turn into a fabricated large clearance reward/cost.
    ground = torch.where(ground_valid, ground, wheels[..., 2] - TUNING["wheel_radius"])
    force = contact.data.net_forces_w[:, contact_sensor_cfg.body_ids]
    airborne = contact.data.current_contact_time[:, contact_sensor_cfg.body_ids] <= 0
    support = ((force[..., 2] > 5.0)
               & (force[..., 2] > 0.5 * torch.linalg.vector_norm(force, dim=-1))
               & (contact.data.current_contact_time[:, contact_sensor_cfg.body_ids] >= TUNING["support_grace"]))
    clearance = wheels[..., 2] - ground - TUNING["wheel_radius"]
    support &= ground_valid & (clearance.abs() <= 0.05)
    command = env.command_manager.get_command(command_name)
    down = scan["down_gate"] > 0
    descent = getattr(env, "_m20_stair_step_state", None)
    if descent is not None:
        down |= descent["active"] & ~descent["up"] & (env.episode_length_buf > 1)
    down &= command[:, 0] > TUNING["translation_threshold"]
    extension = torch.linalg.vector_norm(wheels - hips, dim=-1)
    reach = torch.linalg.vector_norm((wheels - hips)[..., :2], dim=-1)
    knee = robot.data.joint_pos[:, knee_cfg.joint_ids]
    turn = pure_turn(command)
    size = turn_size_excess(clearance, reach, scan["terrain_gate"])
    size = torch.where(ground_valid & airborne, size, 0.0)
    fold = torch.where(support[:, 2:] & down[:, None], fold_excess(extension[:, 2:], knee[:, 2:]), 0.0)
    tracking = turn_tracking_quality(robot.data.root_ang_vel_b[:, 2], command[:, 2])
    cached = dict(step=step, turn=turn, tracking=tracking, support=support, airborne=airborne, clearance=clearance,
                  ground_valid=ground_valid, extension=extension, knee=knee, reach=reach,
                  size=size, fold=fold, down=down)
    env._m20_gait_context = cached
    # Once per control step, independently of how many reward terms use it.
    stats = getattr(env, "_m20_gait_stats", None)
    if stats is None:
        stats = {name: torch.zeros(env.num_envs, device=env.device) for name in
                 ("turn_steps", "turn_clearance_sum", "turn_reach_sum", "turn_size_sum",
                  "rear_support_samples", "rear_extension_sum", "rear_fold_samples",
                  "down_steps", "turn_yaw_error_sum")}
        env._m20_gait_stats = stats
    reset = env.episode_length_buf <= 1
    for value in stats.values():
        value[reset] = 0
    valid_turn = turn & ground_valid.all(dim=1)
    stats["turn_steps"] += valid_turn.float()
    stats["turn_clearance_sum"] += torch.where(valid_turn, clearance.clamp_min(0).mean(dim=1), 0)
    stats["turn_reach_sum"] += torch.where(valid_turn, reach.mean(dim=1), 0)
    stats["turn_size_sum"] += size.mean(dim=1) * turn.float()
    stats["turn_yaw_error_sum"] += (robot.data.root_ang_vel_b[:, 2] - command[:, 2]).abs() * valid_turn.float()
    rear_samples = support[:, 2:] & down[:, None]
    stats["rear_support_samples"] += rear_samples.sum(dim=1)
    stats["rear_extension_sum"] += (extension[:, 2:] * rear_samples.float()).sum(dim=1)
    stats["rear_fold_samples"] += ((fold > 0) & rear_samples).sum(dim=1)
    stats["down_steps"] += down.float()
    return cached


def turn_swing_size_cost(env, asset_cfg, hip_cfg, knee_cfg, sensor_cfg, contact_sensor_cfg):
    c = _context(env, asset_cfg, hip_cfg, knee_cfg, sensor_cfg, contact_sensor_cfg)
    return c["size"].mean(dim=1) * c["turn"].float()


def descent_rear_fold_cost(env, asset_cfg, hip_cfg, knee_cfg, sensor_cfg, contact_sensor_cfg):
    return _context(env, asset_cfg, hip_cfg, knee_cfg, sensor_cfg, contact_sensor_cfg)["fold"].mean(dim=1)


def compact_turn_air_time(env, command_name, sensor_cfg, asset_cfg, threshold,
                          stair_sensor_cfg, cmd_threshold=0.1, foot_height_threshold=0.05):
    """Bound airtime at touchdown; moving forward cannot activate this term.

    Keep the inherited call signature so saved config parameters still resolve.
    Height shaping is handled by turn_swing_size_cost using local ground.
    """
    from .rewards import get_gait_level_tensor
    sensor = env.scene.sensors[sensor_cfg.name]
    first = sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    air = sensor.data.last_air_time[:, sensor_cfg.body_ids]
    command = env.command_manager.get_command(command_name)
    score = bounded_air_score(air, TUNING["air_min"], threshold, TUNING["air_max"])
    yaw = env.scene[asset_cfg.name].data.root_ang_vel_b[:, 2]
    tracking = turn_tracking_quality(yaw, command[:, 2])
    return (score * first).sum(dim=1) * pure_turn(command).float() * tracking * get_gait_level_tensor(env)


def compact_turn_status(env, asset_cfg, hip_cfg, knee_cfg, sensor_cfg, contact_sensor_cfg):
    """Only actual airborne diagonal pairs count, measured above local terrain."""
    from .rewards import get_gait_level_tensor
    c = _context(env, asset_cfg, hip_cfg, knee_cfg, sensor_cfg, contact_sensor_cfg)
    pattern = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    for grounded, lifted in (((0, 3), (1, 2)), ((1, 2), (0, 3))):
        lo = c["clearance"][:, lifted]
        airborne = c["airborne"][:, lifted]
        pattern |= (c["support"][:, grounded].all(dim=1) & airborne.all(dim=1)
                    & c["ground_valid"][:, lifted].all(dim=1)
                    & (lo > 0.02).all(dim=1) & (lo < TUNING["turn_height_terrain"]).all(dim=1))
    return pattern.float() * c["turn"].float() * c["tracking"] * get_gait_level_tensor(env)


def gait_quality_metrics(env, env_ids):
    stats = getattr(env, "_m20_gait_stats", None)
    if stats is None:
        return {"samples": torch.zeros((), device=env.device)}
    s = {key: value[env_ids].sum() for key, value in stats.items()}
    turns = s["turn_steps"].clamp_min(1)
    rear = s["rear_support_samples"].clamp_min(1)
    return {"turn_samples": s["turn_steps"], "descent_samples": s["down_steps"],
            "rear_support_samples": s["rear_support_samples"],
            "turn_clearance_mean_m": s["turn_clearance_sum"] / turns,
            "turn_reach_mean_m": s["turn_reach_sum"] / turns,
            "turn_yaw_error_mean": s["turn_yaw_error_sum"] / turns,
            "rear_extension_mean_m": s["rear_extension_sum"] / rear,
            "rear_fold_support_fraction": s["rear_fold_samples"] / rear}
