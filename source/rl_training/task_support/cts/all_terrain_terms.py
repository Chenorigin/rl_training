"""Actual two-dimensional support, task rewards and clean teacher inputs."""
import torch
from .all_terrain_course import course


def failed(env):
    state = course(env)
    state.advance()
    return state.failed


def succeeded(env):
    state = course(env)
    state.advance()
    return state.success & ~state.failed


def timed_out(env):
    state = course(env)
    state.advance()
    timeout = env.episode_length_buf.double()*env.step_dt >= state.timeout
    guard = getattr(env, "cts_limit_monitor", None)
    return timeout if guard is None else timeout & ~guard.faulted


def reward(env):
    state, data = course(env), env.scene['robot'].data
    g, cfg = state.observe(), state.cfg['rewards']
    q = data.root_quat_w
    body_up = torch.stack((2*(q[:,1]*q[:,3]+q[:,0]*q[:,2]),
        2*(q[:,2]*q[:,3]-q[:,0]*q[:,1]),1-2*(q[:,1].square()+q[:,2].square())),-1)
    normal = g['root_normal']
    hold = getattr(state,'stair_hold',None)
    if hold is not None:
        from task_support.student_v1.perception import quaternion_rotation
        reference = quaternion_rotation(hold.quaternion)[:,:,2]
        normal = torch.where(hold.active[:,None],reference,normal)
    tilt = (1-(body_up*normal).sum(-1).clamp(-1.,1.)).clamp_min(0)
    delta = env.action_manager.action-env.action_manager.prev_action
    if not hasattr(state,'leg_ids'):
        state.leg_ids = env.scene['robot'].find_joints(env.cfg.leg_joint_names,preserve_order=True)[0]
    pose = data.joint_pos[:,state.leg_ids]-data.default_joint_pos[:,state.leg_ids]
    air = ((g['clearance']-cfg['flat_clearance_free_m']).clamp_min(0)/cfg['flat_clearance_scale_m']).square().mean(-1)
    flat_cost = (cfg['flat_air_cost']*air+cfg['flat_support_cost']*(~g['supported']).float().mean(-1)
        +cfg['flat_pose_cost']*pose.square().mean(-1)+cfg['flat_leg_rate_cost']*delta[:,:12].square().mean(-1))
    moving = g['command'].abs().amax(-1) > 1.e-4
    flat = state.flat_tiles[state.tile] & (state.mode <= 1) & moving
    if state.walls is not None:
        flat |= state.walls.active  # Walls have level walkable floor, including stop/turn phases.
    impact = env.stair_quiet_tracker.costs['touchdown'] if hasattr(env,'stair_quiet_tracker') else torch.zeros_like(g['error'])
    # Tracking has zero incentive to reduce wheel speed on traversable terrain.
    from .stability import linear_tracking_reward
    tracking = linear_tracking_reward(state, g)
    yaw_tracking = torch.exp(-g['yaw_error']/.25)
    if getattr(state, 'compound', False):
        from .compound_commands import tracking_rewards
        tracking, yaw_tracking = tracking_rewards(state, g, tracking, yaw_tracking)
    if hasattr(state,'platform_risk') and getattr(state, 'objectives', None) is None:
        stopping = torch.exp(-data.root_lin_vel_b[:,:2].square().sum(-1)/.25)
        tracking = torch.maximum(tracking,state.platform_risk*stopping)
    stopped, stop_cost = g['stopped'], torch.zeros_like(tracking)
    if state.recovery is not None and state.recovery.hold is not None:
        tracking, yaw_tracking, stopped, stop_cost = state.recovery.hold.reward_terms(tracking, yaw_tracking, stopped)
    if state.quality is not None:
        tracking, yaw_tracking, stopped, quality_cost = state.quality.reward_terms(tracking, yaw_tracking, stopped)
        stop_cost = stop_cost + quality_cost
    success_weight = torch.full_like(tracking,cfg['success'])
    if hold is not None:
        tracking = torch.where(hold.active,hold.linear,tracking)
        yaw_tracking = torch.where(hold.active,hold.angular,yaw_tracking)
        stopped = torch.where(hold.active,hold.ready,stopped)
        stop_cost += torch.where(hold.active,hold.cost,0.)
        success_weight = torch.where(hold.active,hold.cfg['success_reward'],success_weight)
    wall_hold = getattr(state, 'wall_hold', None)
    if wall_hold is not None:
        tracking = torch.where(wall_hold.active,wall_hold.linear,tracking)
        yaw_tracking = torch.where(wall_hold.active,wall_hold.angular,yaw_tracking)
        stopped = torch.where(wall_hold.active,wall_hold.ready,stopped)
        stop_cost += torch.where(wall_hold.active,wall_hold.cost,0.)
        success_weight = torch.where(wall_hold.active,wall_hold.cfg['success_reward'],success_weight)
    result = (cfg['linear']*tracking+cfg['yaw']*yaw_tracking
        +cfg['progress']*state.progress/env.step_dt+success_weight*state.success/env.step_dt
        +cfg['stop']*(~moving)*stopped-cfg['tilt']*tilt-stop_cost
        -cfg['action_rate']*delta.square().mean(-1)-cfg['effort']*data.applied_torque.square().mean(-1)
        -cfg['slip']*g['slip'].square().mean(-1)-cfg['impact']*impact/env.step_dt
        -cfg['failure']*state.failed/env.step_dt-flat_cost*flat)
    if getattr(state,'stair_walls',None) is not None:
        result = result-state.stair_walls.cost()
    if getattr(state,'recovery',None) is not None:
        result = result-state.recovery.cost()
    if state.walls is not None:
        result = result-state.walls.cost()
    if 'startup_stability' in state.cfg:
        from .stability import stability_costs
        result = result-sum(stability_costs(env, state, g).values())
    guard = getattr(env, 'cts_limit_monitor', None)
    if guard is not None:
        result = result-guard.reward_cost()
    return torch.nan_to_num(result,nan=-cfg['failure']/env.step_dt,posinf=0.,neginf=-cfg['failure']/env.step_dt)


def locomotion_scan(env):
    state = course(env)
    sensor = env.scene.sensors['height_scanner'].data
    hits = sensor.ray_hits_w
    raw = torch.nan_to_num(sensor.pos_w[:,2,None]-hits[...,2]-.5,posinf=1.,neginf=-1.).clamp(-1.,1.)
    data = env.scene['robot'].data
    q = data.root_quat_w
    yaw = torch.atan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2].square()+q[:,3].square()))
    xy = hits[...,:2]-data.root_pos_w[:,None,:2]
    x = xy[...,0]*yaw.cos()[:,None]+xy[...,1]*yaw.sin()[:,None]
    from task_support.stair_quiet_v1_safe.observations import longitudinal_scan
    filtered = longitudinal_scan(raw,x,hits[...,1]-state.origins[state.tile,1,None],state.widths[state.tile])
    return torch.where(state.stairs[:,None],filtered,raw)


def _margins(state, xy, extent, support_height=None, reach=None):
    """Signed support-patch margins on the actual top-surface rectangles.

    The frozen teacher expects distance to the nearer fore/aft edge, minus its
    contact patch and tread margin. A wheel's full disk is used only when
    recognizing a wheel perched on a neighboring lip, never as that patch.
    """
    local = xy-state.origins[state.tile,None,:2]
    rect = state.surface.rectangles[state.tile]
    x,y = local[...,0,None],local[...,1,None]
    valid = torch.isfinite(rect[:,None,:,4]) & (rect[:,None,:,1] > rect[:,None,:,0])
    lateral = valid & (y >= rect[:,None,:,2]) & (y <= rect[:,None,:,3])
    inside = lateral & (rect[:,None,:,0] <= x) & (rect[:,None,:,1] >= x)
    height = torch.where(inside,rect[:,None,:,4],-torch.inf).amax(-1,keepdim=True)
    if support_height is not None:
        # Match the source teacher's height-aware lip assignment, including a
        # wheel center just outside the next tread's XY rectangle.
        difference = (support_height[...,None]-rect[:,None,:,4]).abs()
        plausible = lateral & (difference < .05)
        plausible &= (x >= rect[:,None,:,0]-reach[...,None]) & (x <= rect[:,None,:,1]+reach[...,None])
        nearest = torch.where(plausible,difference,torch.inf).argmin(-1,keepdim=True)
        candidate = rect[:,None,:,4].expand_as(difference).gather(-1,nearest)
        height = torch.where(plausible.any(-1,keepdim=True),candidate,height)
    same = lateral & (rect[:,None,:,4] == height)
    # Merge connected equal-height rectangles. This removes the fictitious
    # seam between the last stair tread and its landing, without joining two
    # separated stones merely because they happen to have the same height.
    lower,order = torch.where(same,rect[:,None,:,0],torch.inf).sort(-1)
    upper = torch.where(same,rect[:,None,:,1],-torch.inf).gather(-1,order)
    present = same.gather(-1,order)
    prefix = upper.cummax(-1).values
    previous = torch.cat((torch.full_like(prefix[...,:1],-torch.inf),prefix[...,:-1]),-1)
    component = (lower > previous+1.e-6).long().cumsum(-1)
    distance = (lower-x).clamp_min(0)+(x-upper).clamp_min(0)
    nearest = distance.argmin(-1,keepdim=True)
    connected = present & (component == component.gather(-1,nearest))
    lo = torch.where(connected,lower,torch.inf).amin(-1)
    hi = torch.where(connected,upper,-torch.inf).amax(-1)
    bound = state.surface.local_bounds[state.tile]
    ahead = lateral & (rect[:,None,:,0] > x)
    behind = lateral & (rect[:,None,:,1] < x)
    floor_lo = torch.maximum(torch.where(behind,rect[:,None,:,1],-torch.inf).amax(-1),bound[:,0,None])
    floor_hi = torch.minimum(torch.where(ahead,rect[:,None,:,0],torch.inf).amin(-1),bound[:,1,None])
    on_rect = same.any(-1)
    lo,hi = torch.where(on_rect,lo,floor_lo),torch.where(on_rect,hi,floor_hi)
    forward = torch.minimum(local[...,0]-lo,hi-local[...,0])-extent[...,0]
    # Lateral support follows the actual selected rectangle on isolated blocks;
    # the stair corridor retains its original four-wheel width convention.
    across = present & (lower <= x) & (upper >= x)
    chosen = torch.zeros_like(present).scatter_(-1,nearest,True) & present
    across = torch.where(across.any(-1,keepdim=True),across,chosen)
    side_lo = torch.where(across,rect[:,None,:,2].expand_as(same).gather(-1,order),torch.inf).amin(-1)
    side_hi = torch.where(across,rect[:,None,:,3].expand_as(same).gather(-1,order),-torch.inf).amax(-1)
    side_lo = torch.where(on_rect,side_lo,bound[:,2,None])
    side_hi = torch.where(on_rect,side_hi,bound[:,3,None])
    corridor = state.stairs[:,None]
    half = state.widths[state.tile,None]/2
    side_lo = torch.where(corridor,-half,side_lo)
    side_hi = torch.where(corridor,half,side_hi)
    return forward,side_hi-local[...,1]-extent[...,1],local[...,1]-side_lo-extent[...,1]


def teacher_safety(env):
    state, data = course(env),env.scene['robot'].data
    g = state.observe()
    cfg = env.cfg.stair_tail
    from task_support.stair_quiet_v1_width.geometry import wheel_margins
    tread_margin,lateral_margin = wheel_margins(cfg,g['position'])
    extent = torch.stack((cfg.contact_patch_radius+cfg.wheel_half_length*g['axis'][...,0].abs()+tread_margin,
                          cfg.wheel_radius+cfg.wheel_half_length*g['axis'][...,1].abs()+lateral_margin),-1)
    # Frozen Safe safety columns are link-origin translation, whereas slip is
    # evaluated at the rotating contact point. Mixing them erases rolling speed.
    ids = state.body_ids
    link_velocity = data.body_lin_vel_w[:,ids]+torch.linalg.cross(data.body_ang_vel_w[:,ids],
        data.body_link_pos_w[:,ids]-data.body_com_pos_w[:,ids],dim=-1)
    bottom = g['position'][...,2]-cfg.wheel_radius-cfg.wheel_half_length*g['axis'][...,2].abs()
    reach = cfg.wheel_radius+cfg.wheel_half_length*g['axis'][...,0].abs()
    now = _margins(state,g['position'][...,:2],extent,bottom,reach)
    future = _margins(state,g['position'][...,:2]+link_velocity[...,:2]*cfg.prediction_s,
                      extent,bottom+link_velocity[...,2]*cfg.prediction_s,reach)
    feet = torch.stack((*[x/.1 for x in now],*[x/.1 for x in future],g['supported'].float(),
        g['slip']/.3,g['force'][...,2]/500.,link_velocity[...,0],link_velocity[...,1]),-1).flatten(1)
    q = data.root_quat_w
    yaw = torch.atan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2].square()+q[:,3].square()))
    local = data.root_pos_w-state.origins[state.tile]
    body = torch.stack((local[:,1]/(state.widths[state.tile]/2),data.root_lin_vel_w[:,1],yaw.sin(),yaw.cos()),-1)
    return torch.nan_to_num(torch.cat((feet,body),-1)).clamp(-10.,10.)


def material_friction(env, asset_cfg=None):
    state = course(env)
    if state.physics is not None:return state.physics.friction()
    return env.scene['robot'].root_physx_view.get_material_properties()[:,0,:2].to(env.device)


def clean_commands(env, command_name='base_velocity'):
    """Teacher feedback uses clean instantaneous pose/rays, never student channels."""
    from task_support.student_v3_5.v3_5_feedback import LegacyHeadingFeedback, stair_scan_mask
    from task_support.student_v1.environment import sensor_state
    if not hasattr(env,'_cts_clean_heading'):
        env._cts_clean_heading = LegacyHeadingFeedback(env.num_envs,env.device,env.cfg.motion_teacher_tuning)
    data = env.scene['robot'].data
    q = data.root_quat_w
    yaw = torch.atan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2].square()+q[:,3].square()))
    stairs = stair_scan_mask(env.scene.sensors['height_scanner'].data.ray_hits_w,sensor_state(env).grid)
    return env._cts_clean_heading.update(env.command_manager.get_command(command_name),yaw,
        data.root_ang_vel_b[:,2],stairs,env.common_step_counter)


def reset_clean_commands(env,env_ids):
    if hasattr(env,'_cts_clean_heading'):env._cts_clean_heading.reset(env_ids)
