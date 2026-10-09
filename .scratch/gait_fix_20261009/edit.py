from pathlib import Path
p=Path('source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/stair_teacher.py');s=p.read_text()
s=s.replace('"direction_velocity_mix": 0.5, "direction_heading_mix": 0.5,','"direction_velocity_mix": 1.0, "direction_heading_mix": 1.0,\n    "direction_soft_cap": 4.0,')
s=s.replace('"rear_b_limit": np.deg2rad(35.0), "rear_g_limit": np.deg2rad(25.0),','"rear_b_limit": np.deg2rad(25.0), "rear_g_limit": np.deg2rad(20.0),\n    "rear_swing_b_limit": np.deg2rad(45.0), "rear_swing_g_limit": np.deg2rad(35.0),\n    "rear_swing_mix": 0.4, "rear_load_floor": 0.5,\n    "rear_target_x_scale": 0.30, "rear_target_z_scale": 0.20,')
s=s.replace('"outside_samples", "cost_sum", "max_lateral")','"outside_samples", "cost_sum", "max_lateral", "eligible_samples", "tail_samples")')
s=s.replace('''        x = ((value.abs() - tolerance).clamp_min(0.0) / scale).square()
        return x / (1.0 + x)''','''        u = (value.abs() - tolerance).clamp_min(0.0) / scale
        huber = torch.sqrt(1.0 + u.square()) - 1.0
        return huber / (1.0 + huber / TUNING["direction_soft_cap"])''')
s=s.replace('stats["samples"] += active','''stats["samples"] += active
    stats["eligible_samples"] += moving_up
    stats["tail_samples"] += active & (state["cross_track"].abs() >
        TUNING["direction_lateral_tolerance"] + TUNING["direction_lateral_scale"]*TUNING["direction_soft_cap"])''')
s=s.replace('cost_mean=s["cost_sum"] / count)','''cost_mean=s["cost_sum"] / count,
                activation_fraction=s["samples"] / s["eligible_samples"].clamp_min(1),
                large_error_fraction=s["tail_samples"] / count)''')
# Keep the final descending edge latched through rear swing and mixed support.
s=s.replace('''    down &= command[:, 0] > TUNING["translation_threshold"]''','''    phase = getattr(env, "_m20_descent_pose_phase", None)
    if phase is None:
        phase = dict(active=torch.zeros(env.num_envs,dtype=torch.bool,device=env.device),
                     xy=torch.zeros(env.num_envs,2,device=env.device),
                     heading=torch.zeros(env.num_envs,2,device=env.device),
                     lower=torch.zeros(env.num_envs,device=env.device))
        env._m20_descent_pose_phase = phase
    phase["active"][env.episode_length_buf <= 1] = False
    seen_down = scan["down_gate"] > 0
    forward = wheels.new_zeros(env.num_envs,3); forward[:,0] = 1
    heading = quat_apply(yaw_quat(env.scene[sensor_cfg.name].data.quat_w),forward)[:,:2]
    edge_xy = env.scene[sensor_cfg.name].data.pos_w[:,:2] + heading*scan["edge_x"][:,None]
    phase["xy"][seen_down] = edge_xy[seen_down]
    phase["heading"][seen_down] = heading[seen_down]
    phase["lower"][seen_down] = scan["lower_z"][seen_down]
    phase["active"] |= seen_down
    dist = torch.linalg.vector_norm(wheels[:,:,:2]-phase["xy"][:,None],dim=-1).amin(-1)
    past = ((wheels[:,:,:2]-phase["xy"][:,None])*phase["heading"][:,None]).sum(-1)
    rear_loaded = (force[:,2:,2]>5) & (force[:,2:,2]>.5*torch.linalg.vector_norm(force[:,2:],dim=-1)) & ~airborne[:,2:]
    rear_low = wheels[:,2:,2]-TUNING["wheel_radius"] <= phase["lower"][:,None]+.05
    finished = (past[:,2:]>.06).all(-1) & rear_low.all(-1) & rear_loaded.all(-1)
    phase["active"] &= (dist < TUNING["pose_phase_radius"]) & ~(finished & ~seen_down)
    down |= phase["active"]
    down &= command[:, 0] > TUNING["translation_threshold"]''')
s=s.replace('''    rear_forward = bounded(torch.maximum(excess_b,excess_g))*rear_load*load_gain[:,2:]''','''    rear_swing = ~load[:,2:] & down[:,None]
    support_gain = TUNING["rear_load_floor"] + (1-TUNING["rear_load_floor"])*load_gain[:,2:]
    rear_support_cost = bounded(torch.maximum(excess_b,excess_g))*rear_load*support_gain
    swing_b = (body_angle[:,2:]-TUNING["rear_swing_b_limit"]-rear_limit)/TUNING["pose_angle_scale"]
    swing_g = (gravity_angle[:,2:]-TUNING["rear_swing_g_limit"]-rear_limit)/TUNING["pose_angle_scale"]
    rear_swing_cost = bounded(torch.maximum(swing_b,swing_g))*rear_swing*TUNING["rear_swing_mix"]
    rear_forward = rear_support_cost + rear_swing_cost
    _update_rear_swing_stats(env,down,body_angle[:,2:],gravity_angle[:,2:],rear_swing,rear_swing_cost)''')
s=s.replace('"""Penalize excess rear forward reach while actually bearing vertical load."""','"""Separate loaded/swing forward envelopes throughout the descent transfer."""')
# Replace only the rear lift score; front behavior remains identical.
s=s.replace('''        progress = torch.where(eligible, progress, 0)
        lead = torch.where(lead >= 0, lead, progress.argmax(-1))''','''        if axle == 1:
            # Aim at the next tread, rather than paying ever higher pre-edge lifts.
            target_x = torch.maximum(state["map_half_width"][row,target],
                wheels.new_full((n,),TUNING["tread_inset"])) + TUNING["tread_inset"]
            target_x = torch.minimum(target_x,state["map_next_x"][row,target]-TUNING["tread_inset"])
            score = rear_target_score(tx,wheels[:,sl,2]-state["radius"][:,sl],
                target_x[:,None],state["map_z"][row,target,None])
            # Two other supports are needed, but lifting and touchdown can both
            # make progress; stop shaping after this target is physically paid.
            eligible = has_target[:,None] & enabled[:,None] & (other_support>=2) & (
                tx > -TUNING["prep_distance"]) & (tx < state["map_next_x"][row,target,None])
            progress = torch.where(eligible,score,0)
        else:
            progress = torch.where(eligible, progress, 0)
        lead = torch.where(lead >= 0, lead, progress.argmax(-1))''')
# variable n exists? statefunction uses n? use env.num_envs
s=s.replace('wheels.new_full((n,),TUNING["tread_inset"])','wheels.new_full((env.num_envs,),TUNING["tread_inset"])')
s=s.replace('''        gain = (current-state["map_prep_max"][row, target, axle]).clamp_min(0)
        budget =''','''        # A target's initial posture earns nothing. Only subsequent new maxima
        # earn credit; returning to an old pose never pays again.
        if axle == 1:
            state["map_prep_max"][row[changed],target[changed],axle] = current[changed]
        gain = (current-state["map_prep_max"][row, target, axle]).clamp_min(0)
        budget =''')
# per-target budget existing total restrict keep replace rear with eachmax only noinitial isfinite bounded
s=s.replace('''        gain = torch.minimum(gain,budget)*has_target''','''        gain = (gain if axle == 1 else torch.minimum(gain,budget))*has_target''')
insert='''
def rear_target_score(x, bottom_z, target_x, target_z):
    """Bounded next-tread proximity; over-lift and overshoot reduce the score."""
    error = ((x-target_x)/TUNING["rear_target_x_scale"]).square()
    error += ((bottom_z-target_z)/TUNING["rear_target_z_scale"]).square()
    return torch.exp(-0.5*error)


def _update_rear_swing_stats(env,down,b,g,mask,cost):
    state = getattr(env,"_m20_rear_swing_stats",None)
    if state is None:
        state = {key:torch.zeros(env.num_envs,2,device=env.device) for key in
                 ("samples","b_sum","g_sum","bad","b_max","g_max")}
        env._m20_rear_swing_stats = state
    reset = env.episode_length_buf <= 1
    for key,value in state.items(): value[reset] = 0
    state["samples"] += mask
    state["b_sum"] += b.clamp_min(0)*mask; state["g_sum"] += g.clamp_min(0)*mask
    state["bad"] += (cost>0)&mask
    state["b_max"] = torch.maximum(state["b_max"],torch.where(mask,b,0))
    state["g_max"] = torch.maximum(state["g_max"],torch.where(mask,g,0))

'''
s=s.replace('def stair_rear_wheel_lift(',insert+'def stair_rear_wheel_lift(')
s=s.replace('''    result["loaded_without_local_map_samples"] = sums["loaded_without_local_map"]
    return result''','''    result["loaded_without_local_map_samples"] = sums["loaded_without_local_map"]
    swing = getattr(env,"_m20_rear_swing_stats",None)
    if swing is not None:
        ss = {k:v[env_ids].sum(0) for k,v in swing.items()}
        count = ss["samples"].sum().clamp_min(1)
        result["rear_swing_samples"] = ss["samples"].sum()
        result["rear_swing_excess_fraction"] = ss["bad"].sum()/count
        for coord in ("b","g"):
            result[f"rear_swing_{coord}_positive_mean_deg"] = torch.rad2deg(ss[f"{coord}_sum"].sum()/count)
            result[f"rear_swing_{coord}_max_deg"] = torch.rad2deg(swing[f"{coord}_max"][env_ids].amax())
        for i,leg in enumerate(("hl","hr")):
            result[f"{leg}_swing_excess_fraction"] = ss["bad"][i]/ss["samples"][i].clamp_min(1)
    return result''')
s=s.replace('''result[f"up_{name}_transition_successes"] = state[name+"_transition_successes"][ids].mean()''','''result[f"up_{name}_transition_successes"] = state[name+"_transition_successes"][ids].mean()
        result[f"up_{name}_transition_per_landing"] = state[name+"_transition_successes"][ids].sum()/state["physical_total"][ids,axle].sum().clamp_min(1)''')
p.write_text(s)
