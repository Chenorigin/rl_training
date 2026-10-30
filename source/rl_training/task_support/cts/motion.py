"""CTS continuous-ground motion course; geometry is shared by rays and costs.

Only training uses surface geometry. Student observations keep the saved noisy,
delayed front-map ABI. Rolling penalties are trajectory rewards, not supervised
action labels, and apply only to straight motion on genuinely flat patches.
"""
import math

import torch

from .practice import PracticeBase, practice_mask

TERRAINS = ('flat', 'slope', 'wave')
COMMANDS = ('forward', 'backward', 'yaw_left', 'yaw_right',
            'forward_left', 'forward_right', 'backward_left', 'backward_right')


def validate_motion(cfg):
    from .configuration import default_settings
    if not isinstance(cfg, dict) or set(cfg) != set(default_settings()['motion_practice']):
        raise ValueError('Unknown or missing CTS motion practice settings')
    for key, value in cfg.items():
        values = value if isinstance(value, list) else [value]
        if not values or any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
            raise ValueError('Invalid CTS motion setting: '+key)
    if not 0 < cfg['fraction'] < .5:
        raise ValueError('CTS motion practice requires a nonempty minority')
    for key in ('speed_range','yaw_rate_range','arc_yaw_range','push_interval_range_s'):
        v = cfg[key]
        if not isinstance(v, list) or len(v) != 2 or not 0 < v[0] <= v[1]:
            raise ValueError('Invalid positive CTS motion range: '+key)
    for key in ('reset_xy_range','push_velocity_range'):
        if not isinstance(cfg[key], list) or len(cfg[key]) != 2:
            raise ValueError('CTS motion requires two endpoints: '+key)
    slopes = cfg['slope_degrees']
    if (not isinstance(slopes, list) or not min(slopes) < 0 < max(slopes)
            or any(not 0 < abs(v) < 45 for v in slopes)):
        raise ValueError('CTS slopes must include uphill and downhill grades')
    for key in ('wave_amplitudes_m', 'wave_lengths_m'):
        if not isinstance(cfg[key], list) or min(cfg[key]) <= 0:
            raise ValueError('CTS waves require positive amplitudes and wavelengths')
    for key in ('patch_length','patch_width','patch_gap','mesh_resolution','initial_hold_s',
                'move_s','stop_s','boundary_margin','tracking_std','min_root_height',
                'support_distance_m','ray_height_tolerance_m','flat_clearance_free_m','flat_clearance_scale_m'):
        if cfg[key] <= 0:
            raise ValueError('CTS motion requires positive '+key)
    if (cfg['mesh_resolution'] > min(cfg['wave_lengths_m'])/8
            or cfg['push_interval_range_s'][1] >= cfg['move_s']
            or not cfg['push_velocity_range'][0] <= 0 <= cfg['push_velocity_range'][1]
            or min(cfg['reset_xy_range']) < 0 or cfg['reset_yaw_range'] < 0
            or cfg['reset_velocity_range'] < 0 or cfg['reset_yaw_range'] >= math.pi/2):
        raise ValueError('Invalid CTS motion mesh or randomization')
    if any(v < 0 for k,v in cfg.items() if k.endswith(('_cost','_reward'))):
        raise ValueError('CTS motion reward/cost weights must be nonnegative')
    from task_support.stair_quiet_v1_safe.settings import SafeSettings
    s = SafeSettings()
    footprint = math.hypot(s.body_half_length, s.body_half_width)+s.wheel_radius+s.wheel_half_length
    speed, duration = cfg['speed_range'][1], cfg['move_s']
    reach_x = speed*duration+cfg['reset_xy_range'][0]+footprint+cfg['boundary_margin']
    # Integral bound on lateral velocity, including reset yaw and curved motion.
    reach_y = speed*duration*abs(math.sin(cfg['reset_yaw_range']))
    reach_y += speed*cfg['arc_yaw_range'][1]*duration**2/2
    reach_y += cfg['reset_xy_range'][1]+footprint+cfg['boundary_margin']
    if reach_x >= cfg['patch_length']/2 or reach_y >= cfg['patch_width']/2:
        raise ValueError('CTS motion patch cannot contain commanded motion and footprint')
    return cfg


def variants(cfg):
    return ((0,0.,0.), *[(1,math.tan(math.radians(v)),0.) for v in cfg['slope_degrees']],
            *[(2,a,w) for a in cfg['wave_amplitudes_m'] for w in cfg['wave_lengths_m']])


def surface(x, variant, cfg):
    """Exact piecewise-linear height/normal of the actual extruded mesh."""
    table = x.new_tensor(variants(cfg))
    v = table[variant]
    while v.ndim < x.ndim+1:
        v = v.unsqueeze(-2)
    segments = math.ceil(cfg['patch_length']/cfg['mesh_resolution'])
    dx = cfg['patch_length']/segments
    index = ((x+cfg['patch_length']/2)/dx).floor().clamp(0, segments-1)
    left = -cfg['patch_length']/2+index*dx
    def z_at(t):
        wave = v[...,1]*torch.sin(2*math.pi*t/v[...,2].clamp_min(dx))
        return torch.where(v[...,0] == 1, v[...,1]*t,
                           torch.where(v[...,0] == 2, wave, 0.))
    z0,z1 = z_at(left),z_at(left+dx)
    slope = (z1-z0)/dx
    height = z0+(x-left)*slope
    normal = torch.stack((-slope, torch.zeros_like(x), torch.ones_like(x)), -1)
    normal = normal/normal.norm(dim=-1, keepdim=True)
    return height, normal


def motion_patch_mesh(generator, cfg):
    """One physical patch per variant/contact profile, outside all old courses.

    The original Y material columns determine contact domains. Duplicating every
    old staircase/width case is unnecessary for a continuous-ground course.
    """
    import numpy as np
    import trimesh
    from task_support.stair_quiet_v1_width.settings import PROFILES
    if cfg['patch_width'] > generator.size[1]:
        raise ValueError('CTS motion patch would cross contact material columns')
    count = len(variants(cfg)); profiles = len(PROFILES)
    if generator.num_cols < profiles:
        raise ValueError('CTS motion requires all contact material columns')
    origins = np.zeros((count,profiles,3), dtype=np.float32)
    segments = math.ceil(cfg['patch_length']/cfg['mesh_resolution'])
    x = torch.linspace(-cfg['patch_length']/2,cfg['patch_length']/2,segments+1,dtype=torch.float64)
    parts=[]
    start = generator.num_rows*generator.size[0]/2+cfg['patch_gap']
    for vi in range(count):
        z,_ = surface(x,torch.full_like(x,vi,dtype=torch.long),cfg)
        for profile in range(profiles):
            origin = (start+(vi+.5)*(cfg['patch_length']+cfg['patch_gap']),
                      -generator.num_cols*generator.size[1]/2+(profile+.5)*generator.size[1],0.)
            origins[vi,profile] = origin
            vertices = [[float(xx)+origin[0], yy+origin[1],float(zz)]
                        for xx,zz in zip(x,z) for yy in (-cfg['patch_width']/2,cfg['patch_width']/2)]
            faces = [[2*i,2*i+2,2*i+3] for i in range(segments)]
            faces += [[2*i,2*i+3,2*i+1] for i in range(segments)]
            parts.append(trimesh.Trimesh(vertices=vertices,faces=faces,process=False))
    return trimesh.util.concatenate(parts),origins


def start_quaternion(yaw, normal):
    """Small reset yaw, body up aligned to local ground; wxyz convention."""
    heading = torch.stack((yaw.cos(),yaw.sin(),-normal[:,0]/normal[:,2]*yaw.cos()),-1)
    heading = heading/heading.norm(dim=-1,keepdim=True)
    left = torch.linalg.cross(normal,heading)
    rotation = torch.stack((heading,left,normal),-1)
    w = .5*torch.sqrt(1+rotation.diagonal(dim1=-2,dim2=-1).sum(-1))
    xyz = torch.stack((rotation[:,2,1]-rotation[:,1,2],rotation[:,0,2]-rotation[:,2,0],
                       rotation[:,1,0]-rotation[:,0,1]),-1)/(4*w[:,None])
    return torch.cat((w[:,None],xyz),-1)


def wheel_geometry(d, variant, cfg, settings):
    pos,axis = d['position'],d['axis']
    top,normal = surface(pos[...,0],variant,cfg)
    # Training USD uses the verified capsule support envelope.
    extent = settings.wheel_radius+settings.wheel_half_length*(axis*normal).sum(-1).abs()
    clearance = (pos[...,2]-top)*normal[...,2]-extent
    force_n = (d['forces']*normal).sum(-1)
    tangent_force = (d['forces']-force_n[...,None]*normal).norm(dim=-1)
    supported = ((force_n > settings.support_force_N) & (force_n > tangent_force)
                 & (clearance.abs() < cfg['support_distance_m']))
    offset = -settings.wheel_radius*normal
    offset -= settings.wheel_half_length*torch.sign((axis*normal).sum(-1))[...,None]*axis
    velocity = d['velocity']+torch.linalg.cross(d['omega'],offset)
    tangent = velocity-(velocity*normal).sum(-1,keepdim=True)*normal
    return dict(clearance=clearance, supported=supported, normal=normal,
                force_n=force_n, slip=tangent.norm(dim=-1))


def rolling_costs(clearance, supported, leg_pose_error, leg_action_delta, cfg):
    """Unbounded lift excess prevents a cheap high-lift saturation exploit."""
    air = ((clearance-cfg['flat_clearance_free_m']).clamp_min(0)/cfg['flat_clearance_scale_m']).square().mean(-1)
    return (cfg['flat_air_cost']*air+cfg['flat_support_cost']*(~supported).float().mean(-1)
            +cfg['flat_leg_pose_cost']*leg_pose_error.square().mean(-1)
            +cfg['flat_leg_rate_cost']*leg_action_delta.square().mean(-1))


class MotionPractice(PracticeBase):
    def __init__(self,env,cfg,algorithm,excluded,straight_cfg=None):
        from .practice import cohort_mask
        teacher = cohort_mask(env.cfg.scene.num_envs,algorithm['teacher_fraction'],algorithm['cohort_seed'],env.cfg.sim.device)
        mask = practice_mask(teacher,cfg['fraction'],algorithm['cohort_seed']+113,excluded)
        super().__init__(env,cfg,algorithm,validator=validate_motion,mask=mask)
        from .straight import validate_straight
        from copy import deepcopy
        self.straight_cfg = deepcopy(validate_straight(straight_cfg,cfg)) if straight_cfg is not None else None
        self.straight_mask = (practice_mask(teacher,straight_cfg['fraction'],algorithm['cohort_seed']+211,~mask)
                              if straight_cfg is not None else torch.zeros_like(mask))
        if straight_cfg is not None and any(not (mask & ~self.straight_mask & c).any() for c in (teacher,~teacher)):
            raise ValueError('CTS needs ordinary and straight motion in both cohorts; increase num_envs')
        n,device = len(mask),mask.device
        self.rank = torch.zeros(n,dtype=torch.long,device=device)
        for cohort in (teacher,~teacher):
            ids = (mask & ~self.straight_mask & cohort).nonzero().flatten()
            self.rank[ids] = torch.arange(len(ids),device=device)
        self.kind = torch.zeros(n,dtype=torch.long,device=device)
        self.mode = self.kind.clone();self.variant = self.kind.clone()
        self.command_value = torch.zeros(n,3,device=device)
        self.cell_resets = torch.zeros(2,len(TERRAINS),len(COMMANDS),dtype=torch.long,device=device)
        self.cell_frames = self.cell_resets.clone()
        self.cell_failures = self.cell_resets.clone()
        self.cell_profile_resets = torch.zeros(2,len(TERRAINS),len(COMMANDS),5,dtype=torch.long,device=device)
        self.idle_frames = torch.zeros(2,len(TERRAINS),dtype=torch.long,device=device)
        self.wait_frames = torch.zeros_like(self.idle_frames)
        self.stop_frames = torch.zeros_like(self.idle_frames)
        self.tracking_error_sum = torch.zeros_like(self.cell_resets,dtype=torch.float64)
        self.clearance_excess_sum = torch.zeros_like(self.tracking_error_sum)
        self.reward_tick = -1
        self.ray_error_max = torch.zeros((),device=device)
        self.rays_checked = torch.zeros((),dtype=torch.long,device=device)
        self.missing_map_cells = self.rays_checked.clone()
        self.reduced_confidence_cells = self.rays_checked.clone()
        self.straight_frames = torch.zeros(2,dtype=torch.long,device=device)
        self.straight_clearance_sum = torch.zeros(2,4,dtype=torch.float64,device=device)
        self.straight_clearance_peak = torch.zeros_like(self.straight_clearance_sum)
        self.straight_airborne_frames = torch.zeros(2,4,dtype=torch.long,device=device)

    @property
    def movement_duration(self):
        duration = torch.full_like(self.speed,self.cfg['move_s'])
        if self.straight_cfg is not None:
            duration[self.straight_mask] = self.straight_cfg['move_s']
        return duration

    @property
    def patch_lengths(self):
        length = torch.full_like(self.speed,self.cfg['patch_length'])
        if self.straight_cfg is not None:
            length[self.straight_mask] = self.straight_cfg['patch_length']
        return length

    def straight_active(self):
        command = self.env.command_manager.get_command('base_velocity')
        return (self.straight_mask & (command[:,0] > 1.e-4)
                & (command[:,1:].abs().amax(-1) <= 1.e-4)).clone()

    def override_field(self,key,value):
        # Legacy trackers are masked, but keep their intermediate geometry finite.
        super().override_field(key,value)
        if key == 'width': value[self.mask] = self.cfg['patch_width']
        if self.straight_cfg is not None:
            if key == 'spans':
                value[self.straight_mask,0,0] = -self.straight_cfg['patch_length']/2
                value[self.straight_mask,0,1] = self.straight_cfg['patch_length']/2
            elif key in ('goal','ascent_end'):
                value[self.straight_mask] = self.straight_cfg['patch_length']/2

    def reset(self,env_ids):
        env,cfg = self.env,self.cfg
        ids = env_ids[self.mask[env_ids]]
        if not len(ids): return
        cycle = self.resets[ids]+self.rank[ids]
        self.kind[ids] = cycle%len(TERRAINS)
        self.mode[ids] = (cycle//len(TERRAINS))%len(COMMANDS)
        self.kind[self.straight_mask] = 0
        self.mode[self.straight_mask] = 0
        table = torch.tensor(variants(cfg),device=ids.device)
        for kind in range(len(TERRAINS)):
            selected = ids[self.kind[ids] == kind]
            pool = (table[:,0] == kind).nonzero().flatten()
            self.variant[selected] = pool[(self.resets[selected]+self.rank[selected]//24)%len(pool)]
        profile = cycle%5
        terrain,robot = env.scene.terrain,env.scene['robot']
        types = terrain.terrain_types[ids]//5*5+profile
        terrain.terrain_types[ids] = types
        origin = terrain.cts_motion_origins[self.variant[ids],profile]
        straight = self.straight_mask[ids]
        if straight.any():
            origin[straight] = terrain.cts_straight_origins[profile[straight]]
        terrain.env_origins[ids] = origin;env.scene.env_origins[ids] = origin
        xy = torch.empty(len(ids),2,device=ids.device)
        for axis in (0,1): xy[:,axis].uniform_(-cfg['reset_xy_range'][axis],cfg['reset_xy_range'][axis])
        height,normal = surface(xy[:,0],self.variant[ids],cfg)
        position = robot.data.default_root_state[ids,:3].clone()+origin
        position[:,:2] += xy
        position[:,2] = origin[:,2]+height+robot.data.default_root_state[ids,2]/normal[:,2]
        yaw = torch.empty(len(ids),device=ids.device).uniform_(-cfg['reset_yaw_range'],cfg['reset_yaw_range'])
        quat = start_quaternion(yaw,normal)
        robot.write_root_pose_to_sim(torch.cat((position,quat),-1),env_ids=ids)
        velocity = torch.zeros(len(ids),6,device=ids.device)
        velocity[:,:2].uniform_(-cfg['reset_velocity_range'],cfg['reset_velocity_range'])
        velocity[:,2] = -normal[:,0]/normal[:,2]*velocity[:,0]
        robot.write_root_velocity_to_sim(velocity,env_ids=ids)
        speed = torch.empty(len(ids),device=ids.device).uniform_(*cfg['speed_range'])
        yaw_rate = torch.empty(len(ids),device=ids.device).uniform_(*cfg['yaw_rate_range'])
        arc = torch.empty(len(ids),device=ids.device).uniform_(*cfg['arc_yaw_range'])
        mode = self.mode[ids]
        vx = torch.where((mode == 0) | (mode == 4) | (mode == 5),speed,-speed)
        vx = torch.where((mode == 2) | (mode == 3),0.,vx)
        wz = torch.where(mode < 2,0.,torch.where(mode < 4,yaw_rate,arc))
        wz *= torch.where(mode%2 == 0,1.,-1.)
        self.command_value[ids] = torch.stack((vx,torch.zeros_like(vx),wz),-1)
        self.push_at[ids] = cfg['initial_hold_s']
        self.push_at[ids[straight]] = float('inf')  # No added practice pushes on the straight course.
        cohort = (~self.teacher[ids]).long()
        flat = (cohort*len(TERRAINS)+self.kind[ids])*len(COMMANDS)+mode
        self.cell_resets.view(-1).add_(torch.bincount(flat,minlength=self.cell_resets.numel()))
        self.cell_profile_resets.view(-1).add_(torch.bincount(flat*5+profile,minlength=self.cell_profile_resets.numel()))
        self.profile_counts += torch.bincount(profile,minlength=5)
        self.resets[ids] += 1
        self.flat_state(env._tail_course)

    def command(self,term,ids):
        ids = torch.arange(len(self.mask),device=self.mask.device)[ids]
        ids = ids[self.mask[ids]]
        age = self.env.episode_length_buf[ids].double()*self.env.step_dt
        active = (age >= self.cfg['initial_hold_s']) & (age < self.cfg['initial_hold_s']+self.movement_duration[ids])
        term.vel_command_b[ids] = self.command_value[ids]*active[:,None]
        term.is_standing_env[ids] = ~active;term.is_heading_env[ids] = False

    def completed_episodes(self,ids):
        done = self.env.termination_manager.terminated | self.env.termination_manager.time_outs
        selected = ids[self.mask[ids] & (self.env.episode_length_buf[ids] > 0) & done[ids]]
        if len(selected):
            flat = ((~self.teacher[selected]).long()*len(TERRAINS)+self.kind[selected])*len(COMMANDS)+self.mode[selected]
            failed = self.env.termination_manager.terminated[selected].long()
            self.cell_failures.view(-1).index_add_(0,flat,failed)
        super().completed_episodes(ids)

    def diagnostics(self):
        return dict(selected=int(self.mask.sum()),teacher_selected=int((self.mask & self.teacher).sum()),
                    student_selected=int((self.mask & ~self.teacher).sum()),resets=int(self.resets.sum()),
                    pushes=int(self.pushes.sum()),completed=int(self.completed.sum()),failures=int(self.failures.sum()),
                    terrain_names=list(TERRAINS),command_names=list(COMMANDS),
                    cell_resets=self.cell_resets.tolist(),cell_frames=self.cell_frames.tolist(),
                    cell_failures=self.cell_failures.tolist(),cell_profile_resets=self.cell_profile_resets.tolist(),
                    idle_frames=self.idle_frames.tolist(),wait_frames=self.wait_frames.tolist(),
                    stop_frames=self.stop_frames.tolist(),
                    cell_tracking_mse=(self.tracking_error_sum/self.cell_frames.clamp_min(1)).tolist(),
                    cell_clearance_excess_mean_m=(self.clearance_excess_sum/self.cell_frames.clamp_min(1)).tolist(),
                    frame_counts_flat=self.cell_frames.flatten().tolist(),
                    ray_height_error_max_m=float(self.ray_error_max),rays_checked=int(self.rays_checked),
                    missing_map_cells=int(self.missing_map_cells),reduced_confidence_cells=int(self.reduced_confidence_cells))

    def straight_diagnostics(self):
        result = dict(selected=int(self.straight_mask.sum()),
                      teacher_selected=int((self.straight_mask & self.teacher).sum()),
                      student_selected=int((self.straight_mask & ~self.teacher).sum()),
                      actual_fraction=float(self.straight_mask.float().mean()))
        for index,name in enumerate(('teacher','student')):
            frames = self.straight_frames[index].clamp_min(1)
            result[name+'_frames'] = int(self.straight_frames[index])
            for leg,wheel in enumerate(('FL','FR','HL','HR')):
                result[name+'_'+wheel+'_clearance_mean_m'] = float(self.straight_clearance_sum[index,leg]/frames)
                result[name+'_'+wheel+'_clearance_peak_m'] = float(self.straight_clearance_peak[index,leg])
                result[name+'_'+wheel+'_airborne_fraction'] = float(self.straight_airborne_frames[index,leg]/frames)
        return result

    def observe_sensors(self,policy):
        hits = self.env.scene.sensors['height_scanner'].data.ray_hits_w-self.env.scene.env_origins[:,None]
        valid = self.mask[:,None] & torch.isfinite(hits).all(-1)
        valid &= (hits[...,0].abs() < self.patch_lengths[:,None]/2) & (hits[...,1].abs() < self.cfg['patch_width']/2)
        top,_ = surface(hits[...,0],self.variant,self.cfg)
        error = torch.where(valid,(hits[...,2]-top).abs(),0.)
        self.ray_error_max.copy_(torch.maximum(self.ray_error_max,error.max()))
        self.rays_checked += valid.sum()
        if self.ray_error_max > self.cfg['ray_height_tolerance_m']:
            raise RuntimeError('CTS motion height rays disagree with physical mesh geometry')
        known = policy[:,244:431] > .5
        self.missing_map_cells += ((~known) & self.mask[:,None]).sum()
        self.reduced_confidence_cells += ((policy[:,431:618] < 1) & known & self.mask[:,None]).sum()

    def snapshot(self):
        keys = ('mask','resets','pushes','completed','failures','profile_counts','cell_resets',
                'cell_frames','cell_failures','cell_profile_resets','idle_frames','wait_frames','stop_frames','tracking_error_sum','clearance_excess_sum',
                'ray_error_max','rays_checked','missing_map_cells','reduced_confidence_cells',
                'straight_mask','straight_frames','straight_clearance_sum','straight_clearance_peak','straight_airborne_frames')
        return {k:getattr(self,k).detach().cpu().clone() for k in keys}

    def restore(self,saved):
        from task_support.student_v3_safe.environment import checked_tree
        checked = checked_tree(saved,self.snapshot())
        if not torch.equal(checked['mask'].to(self.mask.device),self.mask):
            raise ValueError('CTS motion cohort allocation changed')
        if not torch.equal(checked['straight_mask'].to(self.mask.device),self.straight_mask):
            raise ValueError('CTS straight cohort allocation changed')
        if any((v < 0).any() for k,v in checked.items() if k not in ('mask','straight_mask')):
            raise ValueError('Negative CTS motion accounting')
        for key,value in checked.items(): getattr(self,key).copy_(value.to(self.mask.device))


def motion_reward(env):
    from task_support.student_v3_safe.terrain_observations import physical_sample
    p,robot = env.cts_motion,env.scene['robot'].data
    cfg = p.cfg;d = physical_sample(env)
    g = wheel_geometry(d,p.variant,cfg,env.cfg.stair_tail)
    command = env.command_manager.get_command('base_velocity')
    error = (robot.root_lin_vel_b[:,:2]-command[:,:2]).square().sum(-1)
    yaw_error = (robot.root_ang_vel_b[:,2]-command[:,2]).square()
    _,normal = surface(d['root'][:,0],p.variant,cfg)
    q = robot.root_quat_w
    body_up = torch.stack((2*(q[:,1]*q[:,3]+q[:,0]*q[:,2]),
                           2*(q[:,2]*q[:,3]-q[:,0]*q[:,1]),1-2*(q[:,1].square()+q[:,2].square())),-1)
    tilt = (1-(body_up*normal).sum(-1).clamp(-1.,1.)).clamp_min(0)
    action_delta = env.action_manager.action-env.action_manager.prev_action
    leg_ids = env._cts_motion_leg_ids
    pose_error = robot.joint_pos[:,leg_ids]-robot.default_joint_pos[:,leg_ids]
    moving = command.abs().amax(-1) > 1.e-4
    rolling = p.mask & (p.kind == 0) & (p.mode < 2) & moving
    cost = rolling_costs(g['clearance'],g['supported'],pose_error,action_delta[:,:12],cfg)
    failed = env.termination_manager.terminated
    value = (cfg['linear_reward']*torch.exp(-error/cfg['tracking_std']**2)
             +cfg['yaw_reward']*torch.exp(-yaw_error/cfg['tracking_std']**2)
             -cfg['normal_tilt_cost']*tilt-cfg['action_rate_cost']*action_delta.square().mean(-1)
             -cfg['effort_cost']*robot.applied_torque.square().mean(-1)-cfg['failure_cost']*failed
             -cost*rolling)
    if p.reward_tick != env.common_step_counter:
        ids = p.mask.nonzero().flatten()
        cohort = (~p.teacher[ids]).long()
        active_ids = ids[moving[ids]]
        flat = ((~p.teacher[active_ids]).long()*len(TERRAINS)+p.kind[active_ids])*len(COMMANDS)+p.mode[active_ids]
        p.cell_frames.view(-1).add_(torch.bincount(flat,minlength=p.cell_frames.numel()))
        p.tracking_error_sum.view(-1).index_add_(0,flat,(error+yaw_error)[active_ids].double())
        excess = (g['clearance']-cfg['flat_clearance_free_m']).clamp_min(0).mean(-1)
        p.clearance_excess_sum.view(-1).index_add_(0,flat,excess[active_ids].double())
        idle = ~moving[ids]
        p.idle_frames.view(-1).add_(torch.bincount(cohort[idle]*len(TERRAINS)+p.kind[ids[idle]],minlength=p.idle_frames.numel()))
        age = env.episode_length_buf[ids]*env.step_dt
        stopping = age >= cfg['initial_hold_s']+p.movement_duration[ids]
        for selected,counts in ((idle & ~stopping,p.wait_frames),(idle & stopping,p.stop_frames)):
            counts.view(-1).add_(torch.bincount(cohort[selected]*len(TERRAINS)+p.kind[ids[selected]],minlength=counts.numel()))
        for index,cohort_mask in enumerate((p.teacher,~p.teacher)):
            selected = p.straight_mask & cohort_mask & moving
            p.straight_frames[index] += selected.sum()
            if selected.any():
                clearance = g['clearance'][selected].clamp_min(0)
                p.straight_clearance_sum[index] += clearance.double().sum(0)
                p.straight_clearance_peak[index] = torch.maximum(p.straight_clearance_peak[index],clearance.double().amax(0))
                p.straight_airborne_frames[index] += ((clearance > cfg['flat_clearance_free_m']) & ~g['supported'][selected]).sum(0)
        p.reward_tick = env.common_step_counter
    return torch.where(p.mask,value,0.)


def motion_boundary(env):
    p = env.cts_motion
    root = env.scene['robot'].data.root_pos_w-env.scene.env_origins
    top,normal = surface(root[:,0],p.variant,p.cfg)
    outside = ((root[:,0].abs() > p.patch_lengths/2-p.cfg['boundary_margin'])
               | (root[:,1].abs() > p.cfg['patch_width']/2-p.cfg['boundary_margin']))
    low = (root[:,2]-top)*normal[:,2] < p.cfg['min_root_height']
    return p.mask & (outside | low | ~torch.isfinite(root).all(-1))


def motion_timeout(env):
    p = env.cts_motion
    return p.mask & (env.episode_length_buf*env.step_dt >= p.cfg['initial_hold_s']+p.movement_duration+p.cfg['stop_s'])


def motion_safety_features(env):
    from task_support.student_v3_safe.terrain_observations import physical_sample
    p = env.cts_motion;cfg = p.cfg;s = env.cfg.stair_tail
    d = physical_sample(env);g = wheel_geometry(d,p.variant,cfg,s)
    pos,axis = d['position'],d['axis']
    future = pos+d['velocity']*s.prediction_s
    ex = s.wheel_radius+s.wheel_half_length*axis[...,0].abs()
    ey = s.wheel_radius+s.wheel_half_length*axis[...,1].abs()
    forward = p.patch_lengths[:,None]/2-pos[...,0].abs()-ex
    left,right = cfg['patch_width']/2-pos[...,1]-ey,cfg['patch_width']/2+pos[...,1]-ey
    predicted = p.patch_lengths[:,None]/2-future[...,0].abs()-ex
    pl,pr = cfg['patch_width']/2-future[...,1]-ey,cfg['patch_width']/2+future[...,1]-ey
    feet = torch.stack((forward/.1,left/.1,right/.1,predicted/.1,pl/.1,pr/.1,g['supported'].float(),
                        g['slip']/.3,g['force_n']/500.,d['velocity'][...,0],d['velocity'][...,1]),-1).flatten(1)
    body = torch.stack((d['root'][:,1]/(cfg['patch_width']/2),d['root_velocity'][:,1],d['yaw'].sin(),d['yaw'].cos()),-1)
    return torch.cat((feet,body),-1).clamp(-10.,10.)
