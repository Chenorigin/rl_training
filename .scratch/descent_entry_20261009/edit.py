from pathlib import Path
p=Path('source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/stair_teacher.py');s=p.read_text()
s=s.replace('"pose_phase_radius": 1.50,','''"pose_phase_radius": 1.50,
    "entry_x_lower": -0.05, "entry_x_upper": 0.15,
    "entry_x_high_rise_relax": 0.02, "entry_angle_upper": np.deg2rad(20.0),
    "entry_x_scale": 0.10, "entry_start_distance": 0.60,
    "entry_ground_tolerance": 0.04, "entry_contact_time": 0.06,
    "entry_forward_speed": 0.05, "entry_wheel_reverse_tolerance": 0.05,
    "entry_rearm_time": 0.50, "entry_retract_weight": 0.75,
    "entry_range_cost_weight": -0.25,''')
s=s.replace('phase["lower"][seen_down] = scan["lower_z"][seen_down]','''# Scan lower_z/upper_z name the pre/post-edge columns, not min/max height.
    phase["lower"][seen_down] = torch.minimum(scan["lower_z"],scan["upper_z"])[seen_down]''')
s=s.replace('''    rear_forward = rear_support_cost + rear_swing_cost''','''    rear_forward = rear_support_cost + rear_swing_cost
    entry = _descent_entry_retract_context(env,scan,down,wheels,body_delta,ground,ground_valid,
        load,contact.data.current_contact_time[:,contact_sensor_cfg.body_ids],
        robot.data.body_lin_vel_w[:,asset_cfg.body_ids],robot.data.root_lin_vel_b,
        robot.data.root_quat_w,command,relax)''')
s=s.replace('''rear_forward=rear_forward, body_angle=body_angle, gravity_angle=gravity_angle,''','''rear_forward=rear_forward, body_angle=body_angle, gravity_angle=gravity_angle,
                  entry_retract_gain=entry["gain"], entry_range_cost=entry["cost"],''')
insert='''
def descent_entry_position_band(depth,relax):
    """Soft hip-relative X band, limited by both measured reach and leg tilt."""
    lower = torch.full_like(depth,TUNING["entry_x_lower"])
    upper = TUNING["entry_x_upper"] + relax[:,None]*TUNING["entry_x_high_rise_relax"]
    upper = torch.minimum(upper,depth.clamp_min(0)*torch.tan(
        depth.new_tensor(TUNING["entry_angle_upper"])+relax[:,None]*TUNING["rear_forward_relax"]))
    return lower,upper


def _descent_entry_retract_context(env,scan,down,wheels,body_delta,ground,valid,loaded,
                                    contact_time,wheel_velocity,root_velocity,root_quat,command,relax):
    """One bounded hip-relative adjustment per physical descending flight entry.

    World-ground contacts may stay planted while the trunk advances. No bonus
    is paid for reverse rolling, stopped/backward trunks or repeated poses.
    The first edge is latched, never switched to each successive stair riser.
    """
    n=env.num_envs;device=env.device
    state=getattr(env,"_m20_descent_entry_state",None)
    if state is None:
        zero=lambda *shape:torch.zeros(n,*shape,device=device)
        state=dict(active=zero().bool(),ready=torch.ones(n,dtype=torch.bool,device=device),
            top=zero(),xy=zero(2),heading=zero(2),best=zero(2),quiet=zero(),
            count=torch.zeros(n,dtype=torch.long,device=device),history_xy=zero(64,2),history_top=zero(64),
            stats={k:zero(2) for k in ("samples","x_sum","in_band","too_forward","too_rearward","credit","blocked")},
            started=zero())
        env._m20_descent_entry_state=state
    reset=env.episode_length_buf<=1
    for key,value in state.items():
        if key=="stats":
            for item in value.values():item[reset]=0
        else:value[reset]=True if key=="ready" else 0
    lower,upper=descent_entry_position_band(-body_delta[:,2:,2],relax)
    rear_x=body_delta[:,2:,0]
    error=(lower-rear_x).clamp_min(0)+(rear_x-upper).clamp_min(0)
    normalized=error/TUNING["entry_x_scale"]
    score=torch.exp(-0.5*normalized.square())
    direction=wheels.new_zeros(n,3);direction[:,0]=1
    heading=quat_apply(yaw_quat(root_quat),direction)[:,:2]
    # The scan and body share yaw, the actual raycaster position is supplied
    # through the canonical scan cache by _context before this call.
    scanner_pos=env.scene["height_scanner"].data.pos_w[:,:2]
    edge_xy=scanner_pos+heading*scan["edge_x"][:,None]
    rear_top=ground[:,2:].amax(-1)
    rear_same=valid[:,2:].all(-1)&((ground[:,2:]-rear_top[:,None]).abs()<TUNING["entry_ground_tolerance"]).all(-1)
    stable=loaded&(contact_time>=TUNING["entry_contact_time"])
    pre_edge=(scan["down_gate"]>0)&(scan["edge_x"]>0)&(scan["edge_x"]<TUNING["entry_start_distance"])
    front_lower=(valid[:,:2]&stable[:,:2]&(ground[:,:2]<rear_top[:,None]-TUNING["entry_ground_tolerance"])).any(-1)
    known=torch.arange(64,device=device)[None]<state["count"][:,None]
    same=(torch.linalg.vector_norm(state["history_xy"]-edge_xy[:,None],dim=-1)<.20)&(
        (state["history_top"]-rear_top[:,None]).abs()<TUNING["entry_ground_tolerance"])&known
    start=state["ready"]&~state["active"]&down&rear_same&stable[:,2:].all(-1)&(
        stable.sum(-1)>=3)&(pre_edge|front_lower)&~same.any(-1)&(state["count"]<64)
    ids=torch.nonzero(start,as_tuple=False).squeeze(-1);slots=state["count"][ids]
    state["history_xy"][ids,slots]=edge_xy[ids];state["history_top"][ids,slots]=rear_top[ids]
    state["count"][ids]+=1;state["started"]+=start
    state["top"][start]=rear_top[start];state["xy"][start]=edge_xy[start]
    state["heading"][start]=heading[start];state["best"][start]=score[start]
    state["active"]|=start;state["ready"][start]=False
    # Each rear foot loses its entrance bonus permanently on leaving this top.
    on_top=valid[:,2:]&((ground[:,2:]-state["top"][:,None]).abs()<TUNING["entry_ground_tolerance"])
    gone=valid[:,2:]&(ground[:,2:]<state["top"][:,None]-TUNING["entry_ground_tolerance"])
    state["best"]=torch.where(gone,torch.ones_like(state["best"]),state["best"])
    near=torch.linalg.vector_norm(wheels[:,:,:2]-state["xy"][:,None],dim=-1).amin(-1)<TUNING["pose_phase_radius"]
    aligned=(heading*state["heading"]).sum(-1)>.5
    state["active"]&=near&aligned&~gone.all(-1)&down
    # Only a new flat landing with no active descending transfer rearms the
    # detector. The physical-entry history also blocks retreat/reentry farming.
    flat=valid.all(-1)&stable.all(-1)&((ground.amax(-1)-ground.amin(-1))<TUNING["entry_ground_tolerance"])&~down
    state["quiet"]=torch.where(flat,state["quiet"]+env.step_dt,0)
    state["ready"]|=~state["active"]&(state["quiet"]>=TUNING["entry_rearm_time"])
    other=stable.sum(-1,keepdim=True)-stable[:,2:].long()
    support=state["active"][:,None]&on_top&stable[:,2:]&(other>=2)
    forward=(command[:,0]>TUNING["translation_threshold"])&(root_velocity[:,0]>TUNING["entry_forward_speed"])
    forward_wheel=(wheel_velocity[:,2:,:2]*state["heading"][:,None]).sum(-1)>=-TUNING["entry_wheel_reverse_tolerance"]
    upright=1-2*(root_quat[:,1].square()+root_quat[:,2].square())>.5
    allowed=support&forward[:,None]&forward_wheel&upright[:,None]
    gain=(score-state["best"]).clamp_min(0)*allowed
    # Record even unsupported/blocked progress while still on the original top;
    # it cannot be banked and collected later simply by switching the gate on.
    observed=state["active"][:,None]&on_top&valid[:,2:]
    state["best"]=torch.where(observed,torch.maximum(state["best"],score),state["best"])
    cost=(normalized.square()/(1+normalized.square()))*support
    stats=state["stats"];stats["samples"]+=support;stats["x_sum"]+=rear_x*support
    stats["in_band"]+=support&(error<=1e-6);stats["too_forward"]+=support&(rear_x>upper)
    stats["too_rearward"]+=support&(rear_x<lower);stats["credit"]+=gain;stats["blocked"]+=support&~allowed
    return dict(gain=gain.mean(-1),cost=cost.mean(-1))


def stair_descent_entry_retract(env,asset_cfg,hip_cfg,knee_cfg,sensor_cfg,contact_sensor_cfg):
    """Once-only progress toward a supported hip-relative rear-wheel band."""
    return _context(env,asset_cfg,hip_cfg,knee_cfg,sensor_cfg,contact_sensor_cfg)["entry_retract_gain"]/env.step_dt


def stair_descent_entry_position_cost(env,asset_cfg,hip_cfg,knee_cfg,sensor_cfg,contact_sensor_cfg):
    """Soft guard against forward/rearward overshoot, only at descent entry."""
    return _context(env,asset_cfg,hip_cfg,knee_cfg,sensor_cfg,contact_sensor_cfg)["entry_range_cost"]

'''
s=s.replace('\ndef descent_rear_forward_cost(', '\n'+insert+'\ndef descent_rear_forward_cost(')
# Expose entry metrics through the existing actual TensorBoard curriculum hook.
a=s.index('def gait_quality_metrics');b=s.index('\ndef stair_step_completion_metrics',a);pos=s.rfind('    return result',a,b)
block='''    entry=getattr(env,"_m20_descent_entry_state",None)
    if entry is not None:
        result["descent_entry_started"]=entry["started"][env_ids].sum()
        for i,leg in enumerate(("hl","hr")):
            ss={k:v[env_ids,i].sum() for k,v in entry["stats"].items()}
            count=ss["samples"].clamp_min(1)
            result[f"{leg}_entry_samples"]=ss["samples"]
            result[f"{leg}_entry_x_mean_m"]=ss["x_sum"]/count
            result[f"{leg}_entry_in_band_fraction"]=ss["in_band"]/count
            result[f"{leg}_entry_too_forward_fraction"]=ss["too_forward"]/count
            result[f"{leg}_entry_too_rearward_fraction"]=ss["too_rearward"]/count
            result[f"{leg}_entry_retract_credit"]=ss["credit"]
            result[f"{leg}_entry_blocked_fraction"]=ss["blocked"]/count
'''
s=s[:pos]+block+s[pos:];p.write_text(s)
p=Path('source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/stair_teacher_env_cfg.py');s=p.read_text();s=s.replace('    stair_ascent_front_body_clearance_cost = RewTerm(','''    stair_descent_entry_retract = RewTerm(
        func=stair_teacher.stair_descent_entry_retract,
        weight=stair_teacher.TUNING["entry_retract_weight"], params=_gait_entities(),
    )
    stair_descent_entry_position_cost = RewTerm(
        func=stair_teacher.stair_descent_entry_position_cost,
        weight=stair_teacher.TUNING["entry_range_cost_weight"], params=_gait_entities(),
    )
    stair_ascent_front_body_clearance_cost = RewTerm(''');p.write_text(s)
