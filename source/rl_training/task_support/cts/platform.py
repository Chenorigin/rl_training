"""Narrow raised platforms and small step-down counterexamples, CTS only.

Geometry/labels here are privileged training inputs. The deployed student
retains its existing measured height, validity, confidence and history ABI.
"""
import math

import torch

from .practice import PracticeBase, cohort_mask, practice_mask


def _numeric_leaves(value):
    if isinstance(value,(list,tuple)):
        for child in value:
            yield from _numeric_leaves(child)
    else:
        yield value


def validate_platform(cfg, *, legacy=False):
    from .configuration import default_settings
    expected = default_settings()['platform_practice']
    new_keys = {'height_bands','samples_per_band','terrain_seed','risk_height_range','approach_speed_range',
                'fall_speed_m_s','fall_grace_s','impact_cost','departure_cost','stop_speed_m_s'}
    old_keys = {'high_heights','low_heights','low_every','low_speed_range','fall_clearance'}
    old = legacy and isinstance(cfg,dict) and 'high_heights' in cfg
    keys = (set(expected)-new_keys)|old_keys if old else set(expected)
    if old and 'low_speed_range' not in cfg:
        # Early real v5 checkpoints predate the separate descent-speed field.
        # Read their embedded schema as-is; never reinterpret it as v6 physics.
        keys.discard('low_speed_range')
    if not isinstance(cfg, dict) or set(cfg) != keys:
        raise ValueError('Unknown or missing CTS platform practice settings')
    for key, value in cfg.items():
        values = list(_numeric_leaves(value))
        if not values or any(type(x) not in (int, float) or not math.isfinite(x) for x in values):
            raise ValueError('Nonfinite/non-numeric CTS platform setting: '+key)
        if key in ('speed_range','low_speed_range','approach_speed_range','risk_height_range',
                   'reset_xy_range','push_velocity_range','push_interval_range_s'):
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError('CTS platform range requires two endpoints: '+key)
    if old:
        # v5 deployment/weight migration only. Its former physics/course state
        # can never resume under the new contract, and is not reinterpreted.
        if (not 0 <= cfg['fraction'] < .5 or type(cfg['low_every']) is not int or cfg['low_every'] < 2
                or not 0 < min(cfg['low_heights']) <= max(cfg['low_heights']) < min(cfg['high_heights'])):
            raise ValueError('Invalid legacy CTS platform settings')
        return cfg
    bands = cfg['height_bands']
    if (not isinstance(bands,list) or len(bands) != 4
            or any(not isinstance(pair,list) or len(pair)!=2 or not 0 < pair[0] < pair[1] for pair in bands)
            or any(not math.isclose(left[1],right[0],abs_tol=1e-9) for left,right in zip(bands,bands[1:]))
            or type(cfg['samples_per_band']) is not int or cfg['samples_per_band'] < 2
            or type(cfg['terrain_seed']) is not int or cfg['terrain_seed'] < 0
            or not bands[0][0] <= cfg['risk_height_range'][0] < cfg['risk_height_range'][1] <= bands[-1][1]
            or not 0 <= cfg['fraction'] < .5):
        raise ValueError('Platform requires contiguous height bands and valid stratified sampling')
    for key in ('width','patch_length','floor_padding','patch_gap','initial_hold_s','move_s','stop_s',
                'boundary_margin','edge_margin','prediction_s','tracking_std','min_root_height','fall_speed_m_s','fall_grace_s',
                'low_start_from_end','stop_speed_m_s'):
        if cfg[key] <= 0:
            raise ValueError('CTS platform requires positive '+key)
    if (not 0 < cfg['speed_range'][0] <= cfg['speed_range'][1]
            or not 0 < cfg['approach_speed_range'][0] <= cfg['approach_speed_range'][1]
            or not 0 < cfg['push_interval_range_s'][0] <= cfg['push_interval_range_s'][1] < cfg['move_s']
            or not cfg['push_velocity_range'][0] <= 0 <= cfg['push_velocity_range'][1]
            or min(cfg['reset_xy_range']) < 0 or cfg['reset_xy_range'][1]+cfg['edge_margin'] >= cfg['width']/2
            or cfg['reset_yaw_range'] < 0 or cfg['reset_velocity_range'] < 0
            or any(cfg[key] < 0 for key in ('linear_reward','yaw_reward','tilt_cost','edge_cost','center_cost','impact_cost','departure_cost',
                                          'action_rate_cost','effort_cost','failure_cost','descent_bonus'))):
        raise ValueError('Invalid CTS platform randomization/reward ranges')
    if (cfg['speed_range'][1]*cfg['move_s']+cfg['reset_xy_range'][0]+cfg['boundary_margin'] >= cfg['patch_length']/2
            or cfg['floor_padding'] <= cfg['approach_speed_range'][1]*cfg['move_s']+cfg['boundary_margin']
            or cfg['low_start_from_end'] >= cfg['patch_length']/2):
        raise ValueError('CTS platform cannot contain nominal movement/descent')
    # A completed descent includes rear wheels, not only the base crossing.
    # Conservative body envelope contains the rear wheel centres; add capsule
    # radius/half-length, initial yaw and reset error before allowing a run.
    from task_support.stair_quiet_v1_safe.settings import SafeSettings
    s = SafeSettings()
    required = (cfg['low_start_from_end']+s.body_half_length+s.body_half_width*abs(math.sin(cfg['reset_yaw_range']))
                +s.wheel_radius+s.wheel_half_length+cfg['reset_xy_range'][0])
    if cfg['approach_speed_range'][0]*cfg['move_s']*math.cos(cfg['reset_yaw_range']) <= required:
        raise ValueError('CTS low-step command cannot carry the rear wheels past the edge')
    return cfg


def height_risk(height, cfg):
    lo,hi = cfg['risk_height_range']
    x = ((height-lo)/(hi-lo)).clamp(0.,1.)
    return x.square()*(3.-2.*x)


def physical_heights(generator,cfg):
    """Seeded stratified UNIFORM samples, one true height per mesh patch.

    The finite terrain library is built once, then cycled on resets. It is not
    a claim that static collision geometry is resampled/moved every episode.
    """
    import numpy as np
    rng = np.random.default_rng(cfg['terrain_seed'])
    count = cfg['samples_per_band']
    result = []
    for lo,hi in cfg['height_bands']:
        u = rng.random((count,generator.num_rows,generator.num_cols))
        fraction = (np.arange(count)[:,None,None]+u)/count
        result.append(lo+(hi-lo)*fraction)
    return np.concatenate(result)


def platform_patch_mesh(generator, cfg, backward_length=0.):
    """Closed raised slabs over a real lower catch floor; original Y profiles.

Keep each original width-row identity, plus a height axis. The optional
backward length is only for historical geometry readers. Both sensors and contacts use this mesh.
"""
    import numpy as np
    import trimesh
    heights = physical_heights(generator,cfg)
    rows, cols, tile_width = generator.num_rows, generator.num_cols, generator.size[1]
    if cfg['width'] >= tile_width or cfg['boundary_margin'] >= tile_width/2:
        raise ValueError('Platform must leave a lower floor on both sides')
    floor_length = cfg['patch_length']+2*cfg['floor_padding']
    end = -rows*generator.size[0]/2-rows*backward_length-cfg['patch_gap']
    origins = np.zeros((len(heights), rows, cols, 3), dtype=np.float32)
    parts = []
    for hi in range(len(heights)):
        for row in range(rows):
            x = end-floor_length/2-(hi*rows+row)*(floor_length+cfg['patch_gap'])
            for col in range(cols):
                height = heights[hi,row,col]
                y = -cols*tile_width/2+(col+.5)*tile_width
                origins[hi,row,col] = (x,y,height)
                ground = trimesh.creation.box((floor_length,tile_width,.2))
                ground.apply_translation((x,y,-.1))
                platform = trimesh.creation.box((cfg['patch_length'],cfg['width'],height))
                platform.apply_translation((x,y,height/2))
                parts.extend((ground,platform))
    return trimesh.util.concatenate(parts), origins


def platform_raw_scan(raw, filtered, mask):
    """Keep actual exterior drops for this course; preserve old scan elsewhere."""
    return torch.where(mask[:,None], raw, filtered)


def locomotion_scan(env):
    from task_support.stair_quiet_v1_safe.observations import locomotion_scan as original
    sensor = env.scene.sensors['height_scanner'].data
    raw = (sensor.pos_w[:,2:3]-sensor.ray_hits_w[...,2]-.5).clamp(-1.,1.)
    # The sample's 90 cm deck also exposes true lateral drops to the teacher.
    mask = env.cts_platform.mask | env.cts_sample.mask
    if hasattr(env,'cts_motion'):
        mask = mask | env.cts_motion.mask
    return platform_raw_scan(raw, original(env), mask)


def platform_safety_features(d, cfg, s, height, floor_width):
    """Use real 2D support on the slab AND lower floor, including low drops."""
    from task_support.stair_quiet_v1_width.geometry import capsule_contact_velocity
    pos,axis = d['position'],d['axis']
    future = pos+d['velocity']*cfg['prediction_s']
    on_top = (pos[...,0].abs() <= cfg['patch_length']/2) & (pos[...,1].abs() <= cfg['width']/2)
    top = torch.where(on_top,0.,-height[:,None])
    width = torch.where(on_top,cfg['width'],floor_width)
    extent = s.wheel_radius+s.wheel_half_length*axis[...,1].abs()+cfg['edge_margin']
    left,right = width/2-pos[...,1]-extent,width/2+pos[...,1]-extent
    pred_left,pred_right = width/2-future[...,1]-extent,width/2+future[...,1]-extent
    length = torch.where(on_top,cfg['patch_length']/2,cfg['patch_length']/2+cfg['floor_padding'])
    forward = length-pos[...,0].abs()-s.wheel_radius-s.wheel_half_length*axis[...,0].abs()
    pred_forward = length-future[...,0].abs()-s.wheel_radius-s.wheel_half_length*axis[...,0].abs()
    bottom = pos[...,2]-s.wheel_radius-s.wheel_half_length*axis[...,2].abs()
    f = d['forces']
    supported = (f[...,2]>s.support_force_N) & (f[...,2]>torch.linalg.vector_norm(f[...,:2],dim=-1)) & ((bottom-top).abs()<.05)
    slip = torch.linalg.vector_norm(capsule_contact_velocity(d['velocity'],d['omega'],axis,s.wheel_radius,s.wheel_half_length)[...,:2],dim=-1)
    feet = torch.stack((forward/.1,left/.1,right/.1,pred_forward/.1,pred_left/.1,pred_right/.1,
                        supported.float(),slip/.3,f[...,2]/500.,d['velocity'][...,0],d['velocity'][...,1]),-1).flatten(1)
    body = torch.stack((d['root'][:,1]/(cfg['width']/2),d['root_velocity'][:,1],d['yaw'].sin(),d['yaw'].cos()),-1)
    return torch.cat((feet,body),-1).clamp(-10,10)


def teacher_safety(env):
    from task_support.student_v3_safe.terrain_observations import teacher_safety as original, physical_sample
    p = env.cts_platform
    replacement = platform_safety_features(physical_sample(env),p.cfg,env.cfg.stair_tail,p.height,
                                            env.cfg.stair_tail.tile_width)
    result = torch.where(p.mask[:,None],replacement,original(env))
    if hasattr(env,'cts_motion'):
        from .motion import motion_safety_features
        result = torch.where(env.cts_motion.mask[:,None],motion_safety_features(env),result)
    return result


def platform_geometry(d, cfg, settings, height):
    """Capsule/body footprint and predicted margin to the real raised slab."""
    pos, axis, root, yaw = d['position'], d['axis'], d['root'], d['yaw']
    extent_y = settings.wheel_radius+settings.wheel_half_length*axis[...,1].abs()
    extent_x = settings.wheel_radius+settings.wheel_half_length*axis[...,0].abs()
    side = cfg['width']/2-pos[...,1].abs()-extent_y
    end = cfg['patch_length']/2-pos[...,0].abs()-extent_x
    future = pos+d['velocity']*cfg['prediction_s']
    predicted = torch.minimum(cfg['width']/2-future[...,1].abs()-extent_y,
                              cfg['patch_length']/2-future[...,0].abs()-extent_x)
    body_y = settings.body_half_length*yaw.sin().abs()+settings.body_half_width*yaw.cos().abs()
    body_x = settings.body_half_length*yaw.cos().abs()+settings.body_half_width*yaw.sin().abs()
    body_margin = torch.minimum(cfg['width']/2-root[:,1].abs()-body_y,
                                cfg['patch_length']/2-root[:,0].abs()-body_x)
    margin = torch.minimum(torch.minimum(side,end).amin(-1),body_margin)
    near = (1-torch.minimum(margin,predicted.amin(-1))/cfg['edge_margin']).clamp(0.,1.)
    risk = height_risk(height,cfg)*near.square()*(3.-2.*near)
    bottom = pos[...,2]-settings.wheel_radius-settings.wheel_half_length*axis[...,2].abs()
    on_top = (pos[...,0].abs() <= cfg['patch_length']/2) & (pos[...,1].abs() <= cfg['width']/2)
    top = torch.where(on_top,0.,-height[:,None])
    f = d['forces']
    supported = (f[...,2] > settings.support_force_N) & (f[...,2] > torch.linalg.vector_norm(f[...,:2],dim=-1))
    supported &= (bottom-top).abs() < .05
    return dict(risk=risk,margin=margin,bottom=bottom,supported=supported)


def fall_evidence(age, peak, supported_count, root_vz, dt, cfg):
    """Latch at physics rate, before a landing can erase downward velocity."""
    unsupported = supported_count < 2
    duration = age+unsupported*dt
    maximum = torch.maximum(peak,(-root_vz).clamp_min(0))
    fell = (duration >= cfg['fall_grace_s']) & (maximum >= cfg['fall_speed_m_s'])
    return torch.where(unsupported,duration,0.),torch.where(unsupported,maximum,0.),fell


class PlatformPractice(PracticeBase):
    def __init__(self, env, cfg, algorithm, excluded):
        teacher = cohort_mask(env.cfg.scene.num_envs,algorithm['teacher_fraction'],algorithm['cohort_seed'],env.cfg.sim.device)
        mask = practice_mask(teacher,cfg['fraction'],algorithm['cohort_seed']+71,excluded)
        super().__init__(env,cfg,algorithm,validator=validate_platform,mask=mask)
        n, device = len(mask), mask.device
        self.band = torch.zeros(n,dtype=torch.long,device=device)
        self.approach = torch.zeros(n,dtype=torch.bool,device=device)
        self.height = torch.zeros(n,device=device)
        self.rank = torch.zeros(n,dtype=torch.long,device=device)
        for cohort in (teacher,~teacher):
            ids = (mask & cohort).nonzero().flatten()
            self.rank[ids] = torch.arange(len(ids),device=device)
        self.risk = torch.zeros(n,device=device)
        self.unsupported_age = torch.zeros(n,device=device)
        self.down_speed_peak = torch.zeros(n,device=device)
        self.drop_loss_max = torch.zeros(n,device=device)
        self.start_root_height = torch.zeros(n,device=device)
        self.impact_peak = torch.zeros(n,device=device)
        self.departed = torch.zeros(n,dtype=torch.bool,device=device)
        self.departure_event = torch.zeros(n,dtype=torch.bool,device=device)
        self.fell = torch.zeros(n,dtype=torch.bool,device=device)
        self.descended = torch.zeros_like(self.fell)
        self.descent_event = torch.zeros_like(self.fell)
        self.band_completed = torch.zeros(len(cfg['height_bands']),dtype=torch.long,device=device)
        self.band_failures = torch.zeros_like(self.band_completed)
        self.band_descents = torch.zeros_like(self.band_completed)
        self.band_resets = torch.zeros_like(self.band_completed)
        self.band_tracking_samples = torch.zeros_like(self.band_completed)
        self.hazard_stop_samples = self.small_refusal_samples = 0
        self.fall_speed_max = self.drop_loss_seen_max = self.impact_seen_max = 0.
        self.command_samples = torch.zeros(n,3,dtype=torch.long,device=device)  # idle,forward,reverse
        self.variant_resets = torch.zeros(len(cfg['height_bands'])*cfg['samples_per_band'],dtype=torch.long,device=device)
        self.exterior_rays = self.exterior_lower_rays = self.unknown_cells = self.known_cells = 0
        self.invalid_unknown_encoding = 0
        self.known_clipped_cells = self.teacher_saturated_rays = 0
        self.ray_error_max = self.action_rate_sum = 0.
        self.measurements = 0

    @property
    def movement_duration(self):
        return self.cfg['move_s']

    def reset_origins(self, ids, types):
        cfg = self.cfg
        cycle = self.resets[ids]+self.rank[ids]
        bands = len(cfg['height_bands'])
        self.band[ids] = cycle % bands
        # Nested counting avoids reset%4 aliasing band to one stratum forever.
        stratum = (cycle//bands) % cfg['samples_per_band']
        hi = self.band[ids]*cfg['samples_per_band']+stratum
        high = self.band[ids] == bands-1
        route_tick = cycle//(bands*cfg['samples_per_band'])+self.rank[ids]
        self.approach[ids] = ~high | (route_tick%2 == 0)
        terrain = self.env.scene.terrain
        origins = terrain.cts_platform_origins
        origin = origins[hi,ids % origins.shape[1],types].clone()
        self.height[ids] = origin[:,2]
        self.variant_resets += torch.bincount(hi,minlength=len(self.variant_resets))
        self.band_resets += torch.bincount(self.band[ids],minlength=bands)
        return origin

    def reset(self, env_ids):
        super().reset(env_ids)
        ids = env_ids[self.mask[env_ids]]
        if not len(ids):
            return
        robot, cfg = self.env.scene['robot'],self.cfg
        pose = torch.cat((robot.data.root_pos_w[ids],robot.data.root_quat_w[ids]),-1).clone()
        pose[:,0] += self.approach[ids]*(cfg['patch_length']/2-cfg['low_start_from_end'])
        robot.write_root_pose_to_sim(pose,env_ids=ids)
        # High platforms alternate forward and reverse. Low steps always move
        # toward the platform's end and are allowed onto the lower floor.
        cycle = self.resets[ids]-1+self.rank[ids]
        route_tick = cycle//(len(cfg['height_bands'])*cfg['samples_per_band'])+self.rank[ids]
        direction = torch.where(self.approach[ids] | ((route_tick//2)%2 == 0),1.,-1.)
        low_ids = ids[self.approach[ids]]
        self.speed[low_ids] = torch.empty(len(low_ids),device=self.env.device).uniform_(*cfg['approach_speed_range'])
        self.speed[ids] *= direction
        self.fell[ids] = self.descended[ids] = self.descent_event[ids] = False
        self.departed[ids] = self.departure_event[ids] = False
        self.risk[ids] = 0.
        self.unsupported_age[ids] = self.down_speed_peak[ids] = self.drop_loss_max[ids] = self.impact_peak[ids] = 0.
        self.start_root_height[ids] = pose[:,2]-self.height[ids]

    def override_field(self, key, value):
        cfg, mask = self.cfg,self.mask
        if key in ('height','steps','goal_height'):
            value[mask] = 0
        elif key == 'width':
            value[mask] = cfg['width']
        elif key == 'spans':
            value[mask] = 0.
            value[mask,0,0],value[mask,0,1] = -cfg['patch_length']/2,cfg['patch_length']/2
            if value.shape[1] < 3:
                raise ValueError('Platform step-down requires three physical support spans')
            for index, (start,end) in enumerate(((-cfg['patch_length']/2-cfg['floor_padding'],-cfg['patch_length']/2),
                                                (cfg['patch_length']/2,cfg['patch_length']/2+cfg['floor_padding'])),1):
                value[mask,index,0],value[mask,index,1] = start,end
                value[mask,index,2] = -self.height[mask]
        elif key == 'round_trip':
            value[mask] = True
        elif key in ('goal','ascent_end'):
            value[mask] = cfg['patch_length']/2+cfg['floor_padding']

    def command(self, term, ids):
        ids = torch.arange(len(self.mask),device=self.mask.device)[ids]
        ids = ids[self.mask[ids]]
        age = self.env.episode_length_buf[ids]*self.env.step_dt
        cfg = self.cfg
        moving = (age >= cfg['initial_hold_s']) & (age < cfg['initial_hold_s']+cfg['move_s'])
        term.vel_command_b[ids] = 0.
        term.vel_command_b[ids,0] = torch.where(moving,self.speed[ids],0.)
        term.is_standing_env[ids],term.is_heading_env[ids] = ~moving,False

    def begin_step(self):
        self.descent_event.zero_()
        self.departure_event.zero_()
        self.impact_peak.zero_()

    def sample_physics(self):
        if not self.mask.any():
            return
        from task_support.student_v3_safe.terrain_observations import physical_sample
        d = physical_sample(self.env)
        cfg, s = self.cfg,self.env.cfg.stair_tail
        g = platform_geometry(d,cfg,s,self.height)
        support_count = g['supported'].sum(-1)
        age,peak,fell = fall_evidence(self.unsupported_age,self.down_speed_peak,support_count,
                                     d['root_velocity'][:,2],self.env.physics_dt,cfg)
        self.unsupported_age.copy_(age);self.down_speed_peak.copy_(peak)
        self.fell |= self.mask & fell
        loss = (self.start_root_height-d['root'][:,2]).clamp_min(0)
        self.drop_loss_max = torch.maximum(self.drop_loss_max,loss)
        force = d['forces']
        supported = (force[...,2] > s.support_force_N) & (force[...,2] > torch.linalg.vector_norm(force[...,:2],dim=-1))
        supported &= (g['bottom']+self.height[:,None]).abs() < .05
        crossed = (d['position'][...,0]-s.wheel_radius-s.wheel_half_length*d['axis'][...,0].abs() > cfg['patch_length']/2).all(-1)
        descended = self.mask & crossed & (supported.sum(-1) >= 2) & (d['upright'] > .9)
        descended &= d['root_velocity'][:,2].abs() < cfg['fall_speed_m_s']
        self.descent_event |= descended & ~self.descended
        self.descended |= descended
        safe_floor = self.descended & crossed & (supported.sum(-1) >= 2) & (d['upright'] > .9)
        self.risk.copy_(torch.where(safe_floor,0.,g['risk']))
        left_slab = (d['root'][:,0].abs()>cfg['patch_length']/2) | (d['root'][:,1].abs()>cfg['width']/2)
        self.departure_event |= self.mask & left_slab & ~self.departed
        self.departed |= self.mask & left_slab
        impact = ((torch.linalg.vector_norm(force,dim=-1).amax(-1)-s.tail_force_N)/s.tail_force_scale_N).clamp_min(0).square()
        self.impact_peak = torch.maximum(self.impact_peak,impact)
        self.fall_speed_max = max(self.fall_speed_max,float((-d['root_velocity'][self.mask,2]).clamp_min(0).max()))
        self.drop_loss_seen_max = max(self.drop_loss_seen_max,float(self.drop_loss_max[self.mask].max()))
        self.impact_seen_max = max(self.impact_seen_max,float(self.impact_peak[self.mask].max()))

    def observe_sensors(self, policy):
        ids = self.mask.nonzero().flatten()
        if not len(ids):
            return
        env,cfg = self.env,self.cfg
        hits = env.scene.sensors['height_scanner'].data.ray_hits_w[ids]
        local = hits-env.scene.env_origins[ids,None]
        finite = torch.isfinite(local).all(-1)
        exterior = (local[...,0].abs() > cfg['patch_length']/2) | (local[...,1].abs() > cfg['width']/2)
        lower = exterior & finite & (local[...,2] < -self.height[ids,None]/2)
        self.exterior_rays += int((exterior & finite).sum())
        self.exterior_lower_rays += int(lower.sum())
        expected = torch.where(exterior,-self.height[ids,None],0.)
        if finite.any():
            self.ray_error_max = max(self.ray_error_max,float((local[...,2]-expected).abs()[finite].max()))
        height,valid,confidence = policy[ids,57:244],policy[ids,244:431]>.5,policy[ids,431:618]
        self.unknown_cells += int((~valid).sum());self.known_cells += int(valid.sum())
        self.invalid_unknown_encoding += int(((~valid) & ((height != 0) | (confidence != 0))).sum())
        self.known_clipped_cells += int((valid & (height.abs() >= 1.-1e-6)).sum())
        data = env.scene.sensors['height_scanner'].data
        raw = data.pos_w[ids,2,None]-hits[...,2]-.5
        self.teacher_saturated_rays += int((finite & (raw.abs() >= 1.)).sum())

    def completed_episodes(self, ids):
        from task_support.stair_quiet_v1_width.signals import failure_mask
        selected = ids[self.mask[ids] & (self.env.episode_length_buf[ids] > 0)]
        done = self.env.termination_manager.terminated | self.env.termination_manager.time_outs
        selected = selected[done[selected]]
        failed = failure_mask(self.env)
        for name,weights in (('band_completed',None),('band_failures',failed[selected].long()),
                             ('band_descents',(self.descended[selected] & ~failed[selected]).long())):
            getattr(self,name).index_add_(0,self.band[selected],torch.ones_like(selected) if weights is None else weights)
        super().completed_episodes(ids)

    def snapshot(self):
        state = super().snapshot()
        state.update({key:getattr(self,key).detach().cpu().clone() for key in
                      ('band_completed','band_failures','band_descents','band_resets','band_tracking_samples','command_samples','variant_resets')})
        return state

    def diagnostics(self):
        result = super().diagnostics()
        result['tracking_samples'] = result.pop('reverse_samples')
        result['velocity_mse'] = result.pop('vx_mse')
        result.pop('flat_ray_height_error_max');result.pop('invalid_flat_rays')
        result.update(band_selected=torch.bincount(self.band[self.mask],minlength=len(self.band_resets)).tolist(),
            band_resets=self.band_resets.tolist(),band_completed=self.band_completed.tolist(),
            band_failures=self.band_failures.tolist(),band_descents=self.band_descents.tolist(),
            band_tracking_samples=self.band_tracking_samples.tolist(),height_bands=self.cfg['height_bands'],
            hazard_stop_samples=self.hazard_stop_samples,small_refusal_samples=self.small_refusal_samples,
            fall_speed_max=self.fall_speed_max,drop_loss_max=self.drop_loss_seen_max,impact_max=self.impact_seen_max,
            variant_resets=self.variant_resets.tolist(),command_samples=self.command_samples.sum(0).tolist(),
            exterior_rays=self.exterior_rays,exterior_lower_rays=self.exterior_lower_rays,
            ray_geometry_error_max=self.ray_error_max,unknown_cells=self.unknown_cells,known_cells=self.known_cells,
            invalid_unknown_encoding=self.invalid_unknown_encoding,
            known_clipped_cells=self.known_clipped_cells,teacher_saturated_rays=self.teacher_saturated_rays,
            action_rate_mean=self.action_rate_sum/max(1,self.measurements))
        return result


def platform_failure(env):
    p = env.cts_platform
    root = env.scene['robot'].data.root_pos_w-env.scene.env_origins
    cfg = p.cfg
    outside_floor = ((root[:,0].abs() > cfg['patch_length']/2+cfg['floor_padding']-cfg['boundary_margin'])
                     | (root[:,1].abs() > env.cfg.stair_tail.tile_width/2-cfg['boundary_margin']))
    below_floor = root[:,2]+p.height < cfg['min_root_height']
    return p.mask & (p.fell | outside_floor | below_floor | ~torch.isfinite(root).all(-1))


def platform_timeout(env):
    p = env.cts_platform
    return p.mask & (env.episode_length_buf*env.step_dt >= p.cfg['initial_hold_s']+p.cfg['move_s']+p.cfg['stop_s'])


def platform_reward(env):
    p, robot = env.cts_platform,env.scene['robot'].data
    cfg = p.cfg
    command = env.command_manager.get_command('base_velocity')
    error = (robot.root_lin_vel_b[:,:2]-command[:,:2]).square().sum(-1)
    yaw_error = (robot.root_ang_vel_b[:,2]-command[:,2]).square()
    rate = (env.action_manager.action-env.action_manager.prev_action).square().mean(-1)
    root = robot.root_pos_w-env.scene.env_origins
    from task_support.stair_quiet_v1_width.signals import failure_mask
    failed = failure_mask(env)
    tracking = torch.exp(-error/cfg['tracking_std']**2)
    stopping = torch.exp(-robot.root_lin_vel_b[:,:2].square().sum(-1)/cfg['tracking_std']**2)
    # Keep the requested command nonzero: raw-command rewriting here would
    # disable the existing teacher/decoder PPO gradients precisely at an edge.
    tracking = torch.maximum(tracking,p.risk*stopping)
    value = (cfg['linear_reward']*tracking
             +cfg['yaw_reward']*torch.exp(-yaw_error/cfg['tracking_std']**2)
             -cfg['tilt_cost']*robot.projected_gravity_b[:,:2].square().sum(-1)
             -cfg['edge_cost']*p.risk-cfg['center_cost']*root[:,1].square()*height_risk(p.height,cfg)
             -cfg['action_rate_cost']*rate-cfg['effort_cost']*robot.applied_torque.square().mean(-1)
             # Isaac scales reward rates by dt; terminal/event magnitudes
             # must be divided by dt once, like the original Safe objectives.
             -cfg['failure_cost']*failed/env.step_dt
             -cfg['impact_cost']*p.impact_peak/env.step_dt
             -cfg['departure_cost']*height_risk(p.height,cfg)*p.departure_event/env.step_dt
             +cfg['descent_bonus']*(1-height_risk(p.height,cfg))*(p.descent_event & ~failed)/env.step_dt)
    p.vx_error_sum += error.detach()*p.mask;p.reverse_samples += p.mask.long()
    category = torch.where(command[:,0]>0,1,torch.where(command[:,0]<0,2,0))
    p.command_samples[torch.arange(env.num_envs,device=env.device),category] += p.mask.long()
    p.action_rate_sum += float(rate[p.mask].sum());p.measurements += int(p.mask.sum())
    p.band_tracking_samples += torch.bincount(p.band[p.mask],minlength=len(p.band_tracking_samples))
    stopped = torch.linalg.vector_norm(robot.root_lin_vel_b[:,:2],dim=-1) < cfg['stop_speed_m_s']
    moving = command[:,:2].norm(dim=-1) > 0
    p.hazard_stop_samples += int((p.mask & moving & stopped & (p.risk > .5)).sum())
    p.small_refusal_samples += int((p.mask & moving & stopped & (p.band == 0)).sum())
    return torch.where(p.mask,value,0.)
