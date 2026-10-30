"""StairTrot-v1 environment, ascent gate, phase observations and measured rewards."""
import math

import torch
from isaaclab.envs import ManagerBasedRLEnv

from task_support.idle_stop_motion_v1.profile import TUNING
from task_support.stair_trot_v1.signals import (
    ascent_geometry, command_gate, gait_scores, paired_events,
    yaw_from_quaternion, quat_rotate, new_route_progress,
)
from task_support.stair_trot_v1.stability import (
    rolling_slip, posture_cost, action_jerk_cost, touchdown_impact_proxy,
)


class M20StairTrotV1Env(ManagerBasedRLEnv):
    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        state = getattr(self, "_trot_state", None)
        if state is not None:
            for key in ("phase", "gate", "distance", "frontier"):
                state[key][env_ids] = 0
            for key in ("lift_age", "touch_age"):
                state[key][env_ids] = float("inf")
            state["contact"][env_ids] = False
            state["air"][env_ids] = False
            state["ready"][env_ids] = False
            state["route_ready"][env_ids] = False
            state["previous_velocity"][env_ids] = 0
            state["previous_action"][env_ids] = 0
            state["older_action"][env_ids] = 0
            state["armed"][env_ids] = False
            state["last_tick"][env_ids] = -1
            for value in state["stats"].values():
                value[env_ids] = 0
            for value in state["counts"].values():
                value[env_ids] = 0
        self._trot_cache = None
        # Parent diagnostics cache by global tick; a partial reset changes data.
        for name in ("_m20_stair_scan_context", "_m20_fast_flat_cache"):
            if hasattr(self, name):
                setattr(self, name, None)


def _project_control(env, data):
    """Commands resample after rewards in Isaac; never reuse their old gate."""
    current = env.command_manager.get_command("base_velocity")
    gate = data["physical_gate"] * command_gate(current, env.cfg.trot)
    phase = env._trot_state["phase"] * (2 * math.pi)
    control = torch.stack((gate, phase.sin(), phase.cos(), data["cross_track"].clamp(-2, 2),
                           data["heading_error"].sin(), data["heading_error"].cos()), -1)
    data["control"] = torch.where((gate > 0)[:, None], control, 0.0)
    data["gate"] = gate
    return data


def _context(env, asset_cfg, sensor_cfg, contact_sensor_cfg):
    tick = env.common_step_counter
    cache = getattr(env, "_trot_cache", None)
    if cache is not None and cache[0] == tick:
        return _project_control(env, cache[1])
    cfg = env.cfg.trot
    robot = env.scene[asset_cfg.name]
    scanner = env.scene[sensor_cfg.name]
    sensor = env.scene[contact_sensor_cfg.name]
    hits = scanner.data.ray_hits_w
    ray_xy = scanner.ray_starts
    if ray_xy.ndim == 3:
        ray_xy = ray_xy[0]
    command = env.command_manager.get_command("base_velocity")
    raw_gate = ascent_geometry(hits[..., 2], ray_xy[:, :2], command, cfg)
    wheels = robot.data.body_pos_w[:, asset_cfg.body_ids]
    valid = torch.isfinite(hits).all(-1)
    safe_hits = torch.nan_to_num(hits)
    distances = torch.linalg.vector_norm(wheels[:, :, None, :2] - safe_hits[:, None, :, :2], dim=-1)
    distances = torch.where(valid[:, None], distances, float("inf"))
    nearest, indices = distances.min(-1)
    ground = safe_hits[..., 2].gather(1, indices)
    ground_valid = nearest < cfg.ground_distance_max
    clearance = wheels[..., 2] - ground - TUNING["wheel_radius"]
    clearance = torch.where(ground_valid, clearance, 0.0)
    forces = sensor.data.net_forces_w[:, contact_sensor_cfg.body_ids]
    contact = ((forces[..., 2] > cfg.contact_force)
               & (forces[..., 2] > .5 * torch.linalg.vector_norm(forces, dim=-1)))
    air = (~contact & ground_valid & (clearance > cfg.clearance_min)
           & (sensor.data.current_air_time[:, contact_sensor_cfg.body_ids] >= cfg.air_time_min))
    state = getattr(env, "_trot_state", None)
    if state is None:
        zero = torch.zeros(env.num_envs, device=env.device)
        state = {name: zero.clone() for name in ("phase", "gate", "distance", "frontier")}
        state.update(
            lift_age=torch.full_like(clearance, float("inf")),
            touch_age=torch.full_like(clearance, float("inf")),
            contact=torch.zeros_like(contact), air=torch.zeros_like(air),
            ready=torch.zeros_like(raw_gate, dtype=torch.bool),
            route_ready=torch.zeros_like(raw_gate, dtype=torch.bool),
            route_origin=robot.data.root_pos_w[:, :2].clone(),
            route_yaw=zero.clone(),
            previous_velocity=torch.zeros_like(wheels),
            previous_action=torch.zeros_like(env.action_manager.action),
            older_action=torch.zeros_like(env.action_manager.action),
            armed=torch.zeros((env.num_envs, 2), device=env.device, dtype=torch.bool),
            last_tick=torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long),
            stats={}, counts={},
        )
        env._trot_state = state
    active = (raw_gate > 0) & ground_valid.all(-1)
    update = state["last_tick"] != tick
    entering = active & (state["gate"] == 0)
    # Start with the pair that is already airborne; otherwise use FL+HR.
    start_phase = torch.where(air[:, 1] & air[:, 2], 0.5, 0.0)
    frequency = (command[:, 0].abs() / cfg.nominal_stride).clamp(cfg.frequency_min, cfg.frequency_max)
    phase_next = torch.where(entering, start_phase,
                            (state["phase"] + env.step_dt * frequency * active) % 1.0)
    state["phase"] = torch.where(update, phase_next, state["phase"])
    gate_next = torch.where(active, torch.minimum(
        raw_gate, state["gate"] + env.step_dt / cfg.gate_rise_time), 0.0)
    state["gate"] = torch.where(update, gate_next, state["gate"])
    # Latch a world route once per episode, never reset the frontier on gate flicker.
    heading = yaw_from_quaternion(robot.data.root_quat_w)
    first_route = active & ~state["route_ready"] & update
    route_origin, route_yaw = robot.data.root_pos_w[:, :2], heading
    if hasattr(env.cfg, "stair_course"):
        from task_support.stair_trot_v1.course import course_state
        course = course_state(env)
        route_origin = torch.where(course["mask"][:, None], env.scene.env_origins[:, :2], route_origin)
        route_yaw = torch.where(course["mask"], 0.0, route_yaw)
    state["route_origin"] = torch.where(first_route[:, None], route_origin, state["route_origin"])
    state["route_yaw"] = torch.where(first_route, route_yaw, state["route_yaw"])
    state["route_ready"] |= first_route
    advance, frontier, cross_track = new_route_progress(
        robot.data.root_pos_w[:, :2], state["route_origin"], state["route_yaw"], state["frontier"])
    # First observation establishes the frontier; spawning/approach earns no gait credit.
    advance = torch.where(first_route | ~state["route_ready"], 0.0, advance)
    state["frontier"] = torch.where(update & state["route_ready"], frontier, state["frontier"])
    heading_error = torch.atan2((heading - state["route_yaw"]).sin(),
                               (heading - state["route_yaw"]).cos())
    progress = (advance / (env.step_dt * command[:, 0].clamp_min(cfg.forward_command_min))).clamp(0, 1)
    scores = gait_scores(contact, air, clearance, ground_valid, state["phase"], progress, cfg)
    new_lift = air & ~state["air"] & (state["ready"] & update)[:, None]
    new_touch = contact & ~state["contact"] & (state["ready"] & update)[:, None]
    for name, event in (("lift_age", new_lift), ("touch_age", new_touch)):
        ages = torch.where(active[:, None], state[name] + env.step_dt, float("inf"))
        ages = torch.where(event & active[:, None], 0.0, ages)
        state[name] = torch.where(update[:, None], ages, state[name])
    state["armed"] &= active[:, None]
    # Credit once on the later event in a pair, never every frame of support.
    sync, state["lift_age"], state["touch_age"], state["armed"] = paired_events(
        state["lift_age"], state["touch_age"], new_lift, new_touch,
        contact, air, ground_valid, state["phase"], state["armed"], cfg)
    scores["events"] = sync * progress / env.step_dt
    velocity = robot.data.body_link_lin_vel_w[:, asset_cfg.body_ids]
    angular = robot.data.body_ang_vel_w[:, asset_cfg.body_ids]
    axle_local = torch.zeros_like(wheels)
    axle_local[..., 1] = -1
    axle = quat_rotate(robot.data.body_quat_w[:, asset_cfg.body_ids], axle_local)
    slip_sq, loaded = rolling_slip(velocity, angular, axle, forces,
                                  radius=TUNING["wheel_radius"],
                                  capsule_half_length=cfg.capsule_half_length,
                                  load_min=cfg.contact_force)
    scores["rear_slip"] = (slip_sq[:, 2:] * loaded[:, 2:]).sum(-1) / loaded[:, 2:].sum(-1).clamp_min(1)
    scores["front_slip"] = (slip_sq[:, :2] * loaded[:, :2]).sum(-1) / loaded[:, :2].sum(-1).clamp_min(1)
    scores["rear_slip"] /= cfg.slip_speed_scale ** 2
    scores["front_slip"] /= cfg.slip_speed_scale ** 2
    axle_distance = torch.linalg.vector_norm(wheels[:, :2, :2].mean(1) - wheels[:, 2:, :2].mean(1), dim=-1)
    scores["posture"] = posture_cost(
        robot.data.projected_gravity_b, robot.data.root_ang_vel_b[:, 0],
        ground[:, :2].mean(-1), ground[:, 2:].mean(-1), axle_distance) * ground_valid.all(-1)
    scores["jerk"] = action_jerk_cost(env.action_manager.action, state["previous_action"],
                                     state["older_action"], env.episode_length_buf)
    history = sensor.data.net_forces_w_history[:, :, contact_sensor_cfg.body_ids]
    scores["impact"] = touchdown_impact_proxy(
        state["previous_velocity"], new_touch, history, env.episode_length_buf)
    # Edge proximity is a scan-resolution proxy; apply only to first touchdown.
    other_level = (safe_hits[:, None, :, 2] - ground[:, :, None]).abs() > cfg.edge_height_threshold
    edge_distance = torch.where(other_level & valid[:, None], distances, float("inf")).amin(-1)
    edge_cost = ((cfg.edge_margin - edge_distance) / cfg.edge_margin).clamp(0, 1).square()
    scores["edge_touch"] = (edge_cost * new_touch * ground_valid).mean(-1) / env.step_dt
    scores["route"] = ((cross_track / cfg.route_lateral_scale).square()
                       + (heading_error / cfg.route_heading_scale).square()).clamp_max(10)
    scores["heading_abs_rad"] = heading_error.abs()
    scores["cross_track_abs_m"] = cross_track.abs()
    scores["power"] = (robot.data.applied_torque * robot.data.joint_vel).abs().sum(-1) / cfg.power_scale_w
    state["previous_velocity"] = torch.where(update[:, None, None], velocity, state["previous_velocity"])
    state["older_action"] = torch.where(update[:, None], state["previous_action"], state["older_action"])
    state["previous_action"] = torch.where(update[:, None], env.action_manager.action, state["previous_action"])
    stats = state["stats"]
    samples = {"steps": torch.ones_like(raw_gate), "active_steps": active.float(),
               **{key: value * active for key, value in scores.items()}}
    for key, value in samples.items():
        if key not in stats:
            stats[key] = torch.zeros_like(raw_gate)
            state["counts"][key] = torch.zeros_like(raw_gate)
        sample_valid = active
        if key in ("rear_slip", "front_slip"):
            sample_valid = active & loaded[:, 2 if key == "rear_slip" else 0:
                                           4 if key == "rear_slip" else 2].any(-1)
        elif key == "impact":
            sample_valid = active & (new_touch & loaded).any(-1) & (env.episode_length_buf >= 1)
        elif key == "edge_touch":
            sample_valid = active & (new_touch & ground_valid).any(-1)
        stats[key] += value * update
        state["counts"][key] += sample_valid * update
    state["contact"] = torch.where(update[:, None], contact, state["contact"])
    state["air"] = torch.where(update[:, None], air, state["air"])
    state["ready"].fill_(True)
    state["last_tick"].fill_(tick)
    result = dict(physical_gate=state["gate"], scores=scores,
                  contact=contact, air=air, clearance=clearance,
                  progress=progress, command=command, cross_track=cross_track, heading_error=heading_error)
    env._trot_cache = (tick, result)
    return _project_control(env, result)


def trot_control(env, asset_cfg, sensor_cfg, contact_sensor_cfg):
    return _context(env, asset_cfg, sensor_cfg, contact_sensor_cfg)["control"]


def trot_reward(env, asset_cfg, sensor_cfg, contact_sensor_cfg, metric):
    data = _context(env, asset_cfg, sensor_cfg, contact_sensor_cfg)
    return data["gate"] * data["scores"][metric]


def trot_metrics(env, env_ids):
    """Log measured pair contact alongside task success; no gait-success claim."""
    state = getattr(env, "_trot_state", None)
    if state is None or not state["stats"]:
        return {}
    ids = env_ids if env_ids is not None else slice(None)
    stats = state["stats"]
    count = stats["active_steps"][ids].sum()
    steps = stats["steps"][ids].sum()
    result = {
        "active_samples": count,
        "active_fraction": count / steps.clamp_min(1),
        "__weighted_total__/active_fraction": count,
        "__weighted_count__/active_fraction": steps,
    }
    for key, value in stats.items():
        if key not in ("steps", "active_steps"):
            total = value[ids].sum()
            metric_count = state["counts"][key][ids].sum()
            result[key] = total / metric_count.clamp_min(1)
            result[f"{key}_samples"] = metric_count
            result[f"__weighted_total__/{key}"] = total
            result[f"__weighted_count__/{key}"] = metric_count
    return result
