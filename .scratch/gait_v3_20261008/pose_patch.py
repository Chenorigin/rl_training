from pathlib import Path
p=Path('source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/stair_teacher.py');s=p.read_text()
s=s.replace('state["axis_run"][duplicate.any(-1) & enabled, axle] = 0','state["axis_run"][new_duplicate.any(-1), axle] = 0')
a=s.index('    fold = torch.where(',s.index('def _context('));b=s.index('    tracking = turn_tracking_quality',a)
s=s[:a]+'''    # Force-supported posture is measurable without a terrain-nearest point.
    # Ground validity remains required only by ground-relative turn/clearance.
    load = (force[...,2] > 5) & (force[...,2] > 0.5*torch.linalg.vector_norm(force,dim=-1)) & ~airborne
    load_gain = (force[...,2] / TUNING["pose_load_force_scale"]).clamp(0,1)
    fold = torch.where(load[:,2:] & down[:,None], fold_excess(extension[:,2:],knee[:,2:]),0)
    rise = scan["edge_delta"].abs()
    if ascent is not None:
        rise = torch.maximum(rise,(ascent["target_z"]-ascent["source_z"]).abs())
    relax = ((rise-TUNING["pose_low_rise"]) / (TUNING["pose_high_rise"]-TUNING["pose_low_rise"])).clamp(0,1)
    a_support = TUNING["front_a_support_low_deg"] + relax*(TUNING["front_a_support_high_deg"]-TUNING["front_a_support_low_deg"])
    a_swing = TUNING["front_a_swing_low_deg"] + relax*(TUNING["front_a_swing_high_deg"]-TUNING["front_a_swing_low_deg"])
    front_a = (torch.pi-knee[:,:2].abs()).clamp_min(0)
    front_support = load[:,:2] & up[:,None]
    # Face brushing cannot pretend to be valid support and escape the wider
    # swing envelope; no initial 0.2s free deep fold and no nearest-map veto.
    front_swing = ~load[:,:2] & up[:,None]
    def bounded(value):
        square = value.clamp_min(0).square()
        return square/(1+square)
    front_loaded_angle = bounded((torch.deg2rad(a_support)[:,None]-front_a)/TUNING["pose_angle_scale"])
    front_air_angle = bounded((torch.deg2rad(a_swing)[:,None]-front_a)/TUNING["pose_angle_scale"])
    front_loaded_length = bounded((TUNING["front_min_extension"]-extension[:,:2])/TUNING["pose_length_scale"])
    front_air_length = bounded((0.30-extension[:,:2])/TUNING["pose_length_scale"])
    # Use the worse of angle/length, since both describe the same flexion.
    front_fold = torch.maximum(front_loaded_angle,front_loaded_length)*front_support*load_gain[:,:2]
    front_swing_fold = torch.maximum(front_air_angle,front_air_length)*front_swing
    delta = wheels-hips
    qb = robot.data.root_quat_w[:,None].expand(-1,4,-1).reshape(-1,4)
    body_delta = quat_apply_inverse(qb,delta.reshape(-1,3)).reshape(-1,4,3)
    qy = yaw_quat(robot.data.root_quat_w)[:,None].expand(-1,4,-1).reshape(-1,4)
    gravity_delta = quat_apply_inverse(qy,delta.reshape(-1,3)).reshape(-1,4,3)
    body_angle = torch.atan2(body_delta[...,0],(-body_delta[...,2]).clamp_min(1e-6))
    gravity_angle = torch.atan2(gravity_delta[...,0],(-gravity_delta[...,2]).clamp_min(1e-6))
    rear_limit = relax[:,None]*TUNING["rear_forward_relax"]
    excess_b = (body_angle[:,2:]-TUNING["rear_b_limit"]-rear_limit)/TUNING["pose_angle_scale"]
    excess_g = (gravity_angle[:,2:]-TUNING["rear_g_limit"]-rear_limit)/TUNING["pose_angle_scale"]
    rear_load = load[:,2:] & down[:,None]
    rear_forward = bounded(torch.maximum(excess_b,excess_g))*rear_load*load_gain[:,2:]
    # A gravity-aligned safety lower envelope is a height-map proxy, not an
    # exact collision mesh distance. It alone requires local map validity.
    points = wheels.new_tensor([[x,y,TUNING["body_bottom_z"]] for x in (0.10,0.22,0.34) for y in (-0.16,0,0.16)])
    count = points.shape[0]
    q = robot.data.root_quat_w[:,None].expand(-1,count,-1).reshape(-1,4)
    world = quat_apply(q,points[None].expand(env.num_envs,-1,-1).reshape(-1,3)).reshape(env.num_envs,count,3)+robot.data.root_pos_w[:,None]
    distances = torch.linalg.vector_norm(world[:,:,None,:2]-hits[:,None,:,:2],dim=-1)
    local = valid[:,None] & (distances < TUNING["nearest_valid_distance"])
    floor = torch.where(local,hits[:,None,:,2],-torch.inf).amax(-1)
    gap_valid = local.any(-1).any(-1) & up
    gap = torch.where(local.any(-1),world[...,2]-floor,torch.inf).amin(-1)
    body_gap = torch.where(gap_valid,bounded((TUNING["body_gap_target"]-gap)/TUNING["body_gap_scale"]),0)
    _update_pose_stats(env,up,down,front_a,front_support,front_swing,front_fold,front_swing_fold,
                       body_angle[:,2:],gravity_angle[:,2:],rear_load,rear_forward,gap,gap_valid,load,ground_valid)
''' +s[b:]
s=s.replace('front_swing=front_swing, front_swing_fold=front_swing_fold)','front_swing=front_swing, front_swing_fold=front_swing_fold,\n                  rear_forward=rear_forward, body_angle=body_angle, gravity_angle=gravity_angle,\n                  body_gap=body_gap, body_gap_m=gap, body_gap_valid=gap_valid, load=load, rear_load=rear_load)')
s=s.replace('return (c["front_fold"] + TUNING["front_swing_scale"] * c["front_swing_fold"]).mean(dim=1)','return (c["front_fold"] + c["front_swing_fold"]).mean(dim=1)')
a=s.index('def stair_route_track_lin_vel_xy_exp(');b=s.index('\n\ndef stair_step_completion_metrics',a)
s=s[:a]+'''def stair_route_track_lin_vel_xy_exp(
    env, std: float, command_name: str,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    hip_cfg=None, knee_cfg=None, contact_sensor_cfg=None,
) -> torch.Tensor:
    """Retain most native tracking; unsafe posture forfeits a bounded fraction.

    Do not suppress driving throughout an entire swing/target or require a
    periodic gait clock. Legacy saved calls without gait entities stay native.
    """
    from .rewards import track_lin_vel_xy_exp
    native = track_lin_vel_xy_exp(env,std,command_name,asset_cfg)
    if hip_cfg is None or knee_cfg is None or contact_sensor_cfg is None:
        return native
    c = _context(env,asset_cfg,hip_cfg,knee_cfg,sensor_cfg,contact_sensor_cfg,command_name)
    severity = torch.maximum((c["front_fold"]+c["front_swing_fold"]).mean(-1),c["rear_forward"].mean(-1))
    severity = torch.maximum(severity,c["body_gap"])
    return native*(1-TUNING["pose_tracking_discount"]*severity)
''' +s[b:]
# Preserve old output tags, add metrics to their production function paths.
s=s.replace('def stair_step_completion_metrics(env, env_ids)', 'def _legacy_step_completion_metrics(env, env_ids)')
s=s.replace('def gait_quality_metrics(env, env_ids)', 'def _legacy_gait_quality_metrics(env, env_ids)')
s=s.replace('"body_gap_target": 0.22,', '"front_swing_min_extension_v3": 0.30, "descent_entry_window": 1.0,\n    "body_gap_target": 0.22,')
s=s.replace('bounded((0.30-extension[:,:2])','bounded((TUNING["front_swing_min_extension_v3"]-extension[:,:2])')
p.write_text(s)
