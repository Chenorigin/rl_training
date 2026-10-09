from pathlib import Path
p=Path('source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/stair_teacher.py')
s=p.read_text()
s=s.replace('"front_min_extension": 0.35, "front_max_knee": 2.0,\n    "front_support_grace": 0.20, "front_fold_weight": -0.5,\n    "front_swing_min_extension": 0.22, "front_swing_max_knee": 2.50,\n    "front_swing_scale": 0.50,', '''# a = pi - abs(q_knee), measured in the knee hinge plane.
    # These are soft preferences; higher risers relax swing to preserve clearance.
    "front_min_extension": 0.36, "front_max_knee": np.deg2rad(90.0),
    "front_support_grace": 0.0, "front_fold_weight": -1.5,
    "front_swing_min_extension": 0.30, "front_swing_max_knee": np.deg2rad(100.0),
    "front_high_rise_relax": np.deg2rad(10.0), "front_knee_scale": np.deg2rad(20.0),
    "front_swing_scale": 1.0, "front_load_force": 80.0,
    "front_body_x": (0.10, 0.24, 0.36), "front_body_y": (-0.10, 0.0, 0.10),
    "body_underside_z": -0.07, "body_gap_target": 0.10,
    "body_gap_scale": 0.08, "body_gap_weight": -1.0,
    "rear_body_angle": np.deg2rad(40.0), "rear_gravity_angle": np.deg2rad(30.0),
    "rear_high_drop_relax": np.deg2rad(5.0), "rear_angle_scale": np.deg2rad(15.0),
    "rear_forward_weight": -1.0,''')
s=s.replace('"same_tread_event_weight": -1.0,', '"same_tread_event_weight": -4.5,')
s=s.replace('        "same_tread_seconds_total": zeros(2),','''        "same_tread_seconds_total": zeros(2),
        # Each physical tread remembers both legs, even after the lead leaves.
        "tread_contact_time": zeros(2, 64, 2),
        "tread_seen": flags(2, 64, 2),
        "tread_duplicate_paid": flags(2, 64),
        "tread_intermediate": flags(64),''')
s=s.replace('    state["front_same_tread_event"] = front_pair & (state["front_same_tread_paid"] != front_last)\n    state["front_same_tread_paid"] = torch.where(\n        state["front_same_tread_event"], front_last, state["front_same_tread_paid"],\n    )','')
s=s.replace('    state["rear_same_tread_event"] = rear_pair & (state["rear_same_tread_paid"] != rear_last)\n    state["rear_same_tread_paid"] = torch.where(\n        state["rear_same_tread_event"], rear_last, state["rear_same_tread_paid"],\n    )','''    _temporal_tread_contacts(env, state, wheel, force_z, force_norm,
                             contact, front_intermediate, commanded_forward, heading)''')
# Add one canonical ledger helper, called after both front/rear target retirements.
point=s.index('\ndef ascent_state(')
s=s[:point]+'''
def _temporal_tread_contacts(env, state, wheel, force_z, force_norm,
                             contact, front_intermediate, forward, heading):
    """Remember stable arrivals to all recorded treads, independent of overlap.

    A confirmed higher riser makes its predecessor an intermediate tread.
    The highest tread is eligible only when a higher edge is actually visible;
    broad terminal platforms may legitimately hold both legs. No temporal
    grace can be renewed by lifting, turning, pausing, or changing targets.
    """
    slots = torch.arange(64, device=env.device)[None]
    count = state["front_count"]
    valid = slots < count[:, None]
    state["tread_intermediate"] |= slots < (count - 1)[:, None]
    row = torch.arange(env.num_envs, device=env.device)
    last = (count - 1).clamp_min(0)
    state["tread_intermediate"][row, last] |= front_intermediate & (count > 0)
    xy = state["front_history_xy"]
    directions = state["front_history_heading"]
    heights = state["front_history_target_z"]
    # A recorded next edge bounds the tread horizontally, preventing a tall
    # wheel above a later riser from being labelled on an earlier tread.
    next_slot = (slots + 1).clamp_max(63).expand(env.num_envs, -1)
    next_xy = xy.gather(1, next_slot[..., None].expand(-1, -1, 2))
    tread_depth = ((next_xy - xy) * directions).sum(-1)
    has_next = slots + 1 < count[:, None]
    aligned = (directions * heading[:, None]).sum(-1) > 0.8
    for axle in range(2):
        a = slice(2 * axle, 2 * axle + 2)
        delta = wheel[:, a, None, :2] - xy[:, None]
        along = (delta * directions[:, None]).sum(-1)
        lateral = (delta[..., 0] * directions[:, None, :, 1]
                   - delta[..., 1] * directions[:, None, :, 0]).abs()
        physical = (valid[:, None] & contact[:, a, None]
            & torch.isfinite(state["radius"][:, a, None])
            & (force_z[:, a, None] > 5.0)
            & (force_z[:, a, None] > 0.5 * force_norm[:, a, None])
            & ((wheel[:, a, None, 2] - heights[:, None]
                - state["radius"][:, a, None]).abs() <= 0.035)
            & (along > 0.06) & (lateral <= 0.60)
            & (~has_next[:, None] | (along < tread_depth[:, None] - 0.025)))
        physical = physical.transpose(1, 2)
        timers = state["tread_contact_time"][:, axle]
        timers[:] = torch.where(physical, timers + env.step_dt, torch.zeros_like(timers))
        state["tread_seen"][:, axle] |= timers >= 0.04
        both = state["tread_seen"][:, axle].all(-1)
        # Turning closes style judgement, retaining all recorded arrivals.
        violations = (both & valid & state["tread_intermediate"] & aligned
                      & forward[:, None] & ~state["tread_duplicate_paid"][:, axle])
        state["tread_duplicate_paid"][:, axle] |= violations
        state[("front_same_tread_event", "rear_same_tread_event")[axle]] = violations.any(-1)

''' +s[point:]
s=s.replace('return (state["front_same_tread_event"] | state["rear_same_tread_event"]).float() / env.step_dt', 'return (state["front_same_tread_event"].float()\n            + state["rear_same_tread_event"].float()) / env.step_dt')
s=s.replace('"""Cost once per riser if both front or both rear wheels share its tread."""','"""Cost for sequential or simultaneous two-leg arrivals to an intermediate tread."""')
# keep legacy fold_excess untouched (rear behavior), use smooth front-specific penalty.
point=s.index('\ndef _context(')
s=s[:point]+'''
def soft_excess(value):
    """Bounded monotone cost without a hard plateau for severe violations."""
    excess = value.clamp_min(0)
    return excess.square() / (1.0 + excess.square())


def front_fold_excess(extension, knee, max_knee, min_extension):
    angle = (knee.abs() - max_knee) / TUNING["front_knee_scale"]
    length = (min_extension - extension) / TUNING["extension_scale"]
    return soft_excess(torch.maximum(angle, length))


def front_body_clearance(env, robot, hits, up):
    """Vertical gap proxy under the front chassis collision envelope.

    Box/cylinders in the pinned M20 model reach z=-0.07 in base coordinates.
    Nine physical underside samples are transformed by full base attitude;
    this is a height-map clearance proxy, not an exact mesh distance.
    """
    offsets = torch.tensor([(x, y, TUNING["body_underside_z"])
                            for x in TUNING["front_body_x"] for y in TUNING["front_body_y"]],
                           dtype=hits.dtype, device=env.device)
    points = offsets[None].expand(env.num_envs, -1, -1)
    quat = robot.data.root_quat_w[:, None].expand(-1, offsets.shape[0], -1)
    points = quat_apply(quat.reshape(-1, 4), points.reshape(-1, 3)).reshape_as(points)
    points = points + robot.data.root_pos_w[:, None]
    valid = torch.isfinite(hits).all(-1)
    distance = torch.linalg.vector_norm(points[:, :, None, :2] - hits[:, None, :, :2], dim=-1)
    distance = torch.where(valid[:, None], distance, torch.inf)
    closest, index = distance.min(-1)
    ground = hits[..., 2].gather(1, index)
    usable = (closest <= 0.075) & torch.isfinite(ground)
    gap = torch.where(usable, points[..., 2] - ground, torch.inf).amin(-1)
    cost = soft_excess((TUNING["body_gap_target"] - gap) / TUNING["body_gap_scale"])
    return torch.where(up & torch.isfinite(gap), cost, 0.0), gap

''' +s[point:]
a=s.index('    front_support = (support[:, :2]',s.index('def _context'))
b=s.index('    tracking = turn_tracking_quality',a)
s=s[:a]+'''    contact_time = contact.data.current_contact_time[:, contact_sensor_cfg.body_ids]
    load = ((force[..., 2] > 5.0)
            & (force[..., 2] > 0.5 * torch.linalg.vector_norm(force, dim=-1))
            & (contact_time > 0) & ground_valid & (clearance.abs() <= 0.05))
    # No .20s blind interval: finite force-based load gain starts at touchdown.
    front_support = load[:, :2] & up[:, None]
    load_gain = 0.25 + 0.75 * (force[:, :2, 2] / TUNING["front_load_force"]).clamp(0, 1)
    rise = scan["edge_delta"].clamp_min(0)
    if ascent is not None and "target_z" in ascent:
        rise = torch.maximum(rise, torch.where(ascent["active"],
                             ascent["target_z"] - ascent["source_z"], 0.0))
    relax = ((rise - 0.15) / 0.10).clamp(0, 1)[:, None]
    support_limit = TUNING["front_max_knee"] + 0.5 * relax * TUNING["front_high_rise_relax"]
    swing_limit = TUNING["front_swing_max_knee"] + relax * TUNING["front_high_rise_relax"]
    front_fold = torch.where(front_support, load_gain * front_fold_excess(
        extension[:, :2], knee[:, :2], support_limit, TUNING["front_min_extension"]), 0.0)
    # Face contacts cannot turn off folding: a wheel without useful vertical
    # load remains in the swing/obstruction envelope, including grazing force.
    front_swing = ~load[:, :2] & ground_valid[:, :2] & up[:, None]
    front_swing_fold = torch.where(front_swing, front_fold_excess(
        extension[:, :2], knee[:, :2], swing_limit, TUNING["front_swing_min_extension"]), 0.0)
    body_cost, body_gap = front_body_clearance(env, robot, hits, up)
    leg_w = wheels - hips
    quat = robot.data.root_quat_w[:, None].expand(-1, 4, -1)
    leg_b = quat_apply_inverse(quat.reshape(-1, 4), leg_w.reshape(-1, 3)).reshape_as(leg_w)
    body_angle = torch.atan2(leg_b[..., 0], (-leg_b[..., 2]).clamp_min(1e-4))
    yaw = yaw_quat(robot.data.root_quat_w)
    leg_g = quat_apply_inverse(yaw[:, None].expand(-1, 4, -1).reshape(-1, 4),
                               leg_w.reshape(-1, 3)).reshape_as(leg_w)
    gravity_angle = torch.atan2(leg_g[..., 0], (-leg_g[..., 2]).clamp_min(1e-4))
    drop = (-scan["edge_delta"]).clamp_min(0)
    if descent is not None and "source_z" in descent:
        drop = torch.maximum(drop, torch.where(descent["active"] & ~descent["up"],
                             descent["source_z"] - descent["target_z"], 0.0))
    rear_relax = ((drop - 0.15) / 0.10).clamp(0, 1)[:, None] * TUNING["rear_high_drop_relax"]
    rear_angle_excess = torch.maximum(
        body_angle[:, 2:] - TUNING["rear_body_angle"] - rear_relax,
        gravity_angle[:, 2:] - TUNING["rear_gravity_angle"] - rear_relax)
    rear_forward = torch.where(load[:, 2:] & down[:, None],
        soft_excess(rear_angle_excess / TUNING["rear_angle_scale"]), 0.0)
''' +s[b:]
s=s.replace('front_swing=front_swing, front_swing_fold=front_swing_fold)', '''front_swing=front_swing, front_swing_fold=front_swing_fold,
                  body_cost=body_cost, body_gap=body_gap, rear_forward=rear_forward,
                  body_angle=body_angle, gravity_angle=gravity_angle)''')
s=s.replace('"front_swing_samples", "front_swing_fold_samples")', '''"front_swing_samples", "front_swing_fold_samples", "front_swing_a_sum",
                  "front_loaded_a_sum", "body_gap_sum", "body_gap_samples", "body_gap_low_samples",
                  "rear_forward_samples", "rear_forward_bad_samples", "rear_body_angle_sum",
                  "rear_gravity_angle_sum")''')
s=s.replace('    return cached\n\n\ndef turn_swing_size_cost', '''    front_a = np.pi - knee[:, :2].abs()
    stats["front_loaded_a_sum"] += (front_a * front_support).sum(-1)
    stats["front_swing_a_sum"] += (front_a * front_swing).sum(-1)
    body_valid = up & torch.isfinite(body_gap)
    stats["body_gap_samples"] += body_valid.float()
    stats["body_gap_sum"] += torch.where(body_valid, body_gap, 0.0)
    stats["body_gap_low_samples"] += (body_valid & (body_gap < TUNING["body_gap_target"])).float()
    rear_load = load[:, 2:] & down[:, None]
    stats["rear_forward_samples"] += rear_load.sum(-1)
    stats["rear_forward_bad_samples"] += ((rear_forward > 0) & rear_load).sum(-1)
    stats["rear_body_angle_sum"] += (body_angle[:, 2:] * rear_load).sum(-1)
    stats["rear_gravity_angle_sum"] += (gravity_angle[:, 2:] * rear_load).sum(-1)
    return cached


def ascent_front_body_clearance_cost(env, asset_cfg, hip_cfg, knee_cfg, sensor_cfg, contact_sensor_cfg):
    return _context(env, asset_cfg, hip_cfg, knee_cfg, sensor_cfg, contact_sensor_cfg)["body_cost"]


def descent_rear_forward_cost(env, asset_cfg, hip_cfg, knee_cfg, sensor_cfg, contact_sensor_cfg):
    return _context(env, asset_cfg, hip_cfg, knee_cfg, sensor_cfg, contact_sensor_cfg)["rear_forward"].mean(-1)


def turn_swing_size_cost''')
s=s.replace('''    Normal swing and initial touchdown compliance are free; only severe swing
    flexion is charged. Commands and missing local ground close the gates.''', '''    Loaded contact starts immediately with a soft force gain. Swing has a
    wider, rise-dependent envelope; no hard pose or exact IK target is imposed.''')
s=s.replace('"front_extreme_swing_fold_fraction": s["front_swing_fold_samples"] / swing}', '''"front_extreme_swing_fold_fraction": s["front_swing_fold_samples"] / swing,
            "front_loaded_a_mean_deg": torch.rad2deg(s["front_loaded_a_sum"] / front),
            "front_swing_a_mean_deg": torch.rad2deg(s["front_swing_a_sum"] / swing),
            "front_body_gap_mean_m": s["body_gap_sum"] / s["body_gap_samples"].clamp_min(1),
            "front_body_gap_low_fraction": s["body_gap_low_samples"] / s["body_gap_samples"].clamp_min(1),
            "rear_forward_excess_fraction": s["rear_forward_bad_samples"] / s["rear_forward_samples"].clamp_min(1),
            "rear_body_b_mean_deg": torch.rad2deg(s["rear_body_angle_sum"] / s["rear_forward_samples"].clamp_min(1)),
            "rear_gravity_g_mean_deg": torch.rad2deg(s["rear_gravity_angle_sum"] / s["rear_forward_samples"].clamp_min(1))}''')
s=s.replace('        forward_steps = up_state["forward_steps"]', '''        slots = torch.arange(64, device=env.device)[None]
        for axle, name in enumerate(("front", "rear")):
            reached = (up_state["tread_seen"][env_ids, axle].any(-1)
                       & up_state["tread_intermediate"][env_ids]
                       & (slots < up_state["front_count"][env_ids, None]))
            duplicates = up_state["tread_seen"][env_ids, axle].all(-1) & reached
            result[f"up_{name}_unique_tread_fraction"] = (
                (reached & ~duplicates).sum().float() / reached.sum().clamp_min(1))
            result[f"up_{name}_recorded_intermediate_treads"] = reached.sum(-1).float().mean()
        forward_steps = up_state["forward_steps"]''')
p.write_text(s)
p=Path('source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/stair_teacher_env_cfg.py')
s=p.read_text().replace('    turn_swing_size_cost = RewTerm(', '''    stair_ascent_front_body_clearance_cost = RewTerm(
        func=stair_teacher.ascent_front_body_clearance_cost,
        weight=stair_teacher.TUNING["body_gap_weight"], params=_gait_entities(),
    )
    stair_descent_rear_forward_cost = RewTerm(
        func=stair_teacher.descent_rear_forward_cost,
        weight=stair_teacher.TUNING["rear_forward_weight"], params=_gait_entities(),
    )
    turn_swing_size_cost = RewTerm(''')
p.write_text(s)
