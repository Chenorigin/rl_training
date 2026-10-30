"""CTS course ownership, command phases and measured episode outcomes."""
from copy import deepcopy
import numpy as np
import torch

from .all_terrain_config import COURSES
from .all_terrain_curriculum import CourseCurriculum, allocate_courses
from .practice import cohort_mask
from .command_sequences import (MODE_NAMES, GROUP_NAMES, MODE_GROUPS, command_sequence,
                                lateral_spawn, lateral_duration, transition_spawn)


def course(env):
    if getattr(env, 'cts_all_terrain', None) is None:
        env.cts_all_terrain = AllTerrainCourse(env)
    return env.cts_all_terrain


class AllTerrainCourse:
    def __init__(self, env):
        from .all_terrain_geometry import TorchSurface
        self.env, self.cfg = env, env.cfg.cts_all_terrain
        n, device = env.num_envs, env.device
        alg = env.cfg.cts_cohort_config
        self.teacher = cohort_mask(n, alg['teacher_fraction'], alg['cohort_seed'], device)
        self.course_ids = allocate_courses(self.teacher, self.cfg['fractions'], alg['cohort_seed'])
        self.motion_stability = self.cfg.get('motion_stability')
        self.compound = self.cfg.get('compound_commands', {}).get('enabled', False)
        self.has_walls = bool(self.cfg.get('wall_practice',{}).get('enabled'))
        self.base_num_modes = 19 if self.compound else 11 if self.motion_stability is not None else 9
        self.num_modes = 24 if self.has_walls else 19 if self.compound else 11 if self.motion_stability is not None else 9
        self.num_groups = 14 if self.has_walls else 9 if self.compound else 7 if self.motion_stability is not None else 5
        self.curriculum = CourseCurriculum(self.cfg['curriculum'], num_courses=len(COURSES)*self.num_groups)
        self.rng = np.random.default_rng(self.cfg['geometry']['seed']+927)
        self.atlas = env.scene.terrain.atlas
        self.surface = TorchSurface(self.atlas, device)
        self.origins = torch.as_tensor(self.atlas.origins, device=device, dtype=torch.float32)
        self.goals = torch.as_tensor(self.atlas.goals, device=device, dtype=torch.float32)
        self.bounds = torch.as_tensor(self.atlas.bounds, device=device, dtype=torch.float32)
        self.widths = torch.tensor([r['width'] for r in self.atlas.records], device=device)
        self.heights = torch.tensor([r.get('height',0.) for r in self.atlas.records],device=device)
        self.ends = torch.tensor([r.get('route_end',0.) for r in self.atlas.records],device=device)
        self.flat_tiles = torch.tensor([r['variant'] == 'flat' or r['course'] == 'straight' for r in self.atlas.records], device=device)
        self.boundary_tiles = torch.tensor([r['boundary'] for r in self.atlas.records], device=device)
        self.pools = {(c,g):np.flatnonzero((self.atlas.course_ids == c) & (self.atlas.geometry_levels == g))
                      for c in range(len(COURSES)) for g in range(self.cfg['geometry']['geometry_levels'])}
        self.tile = env.scene.terrain.initial_tiles.clone()
        self.geometry_level = torch.zeros(n, dtype=torch.long, device=device)
        self.dr_level = self.geometry_level.clone()
        self.version = self.geometry_level.clone()
        self.role = self.geometry_level.clone()
        self.mode = self.geometry_level.clone()
        self.group = self.geometry_level.clone()
        self.resets = self.geometry_level.clone()
        self.command_value = torch.zeros(n,3,device=device)
        self.initial_yaw = torch.zeros(n,device=device)
        self.timeout = torch.full((n,),30.,device=device)
        self.move_duration = torch.full((n,),self.cfg['commands']['move_s'],device=device)
        self.success = torch.zeros(n,dtype=torch.bool,device=device)
        self.failed = self.success.clone()
        self.reached = self.success.clone()
        self.finished = self.success.clone()
        self.foot_passed = torch.zeros(n,4,dtype=torch.bool,device=device)
        self.stable_age = torch.zeros(n,device=device)
        self.best_x = self.stable_age.clone()
        self.progress = self.stable_age.clone()
        self.tracking_error = torch.zeros(n,2,device=device)
        self.tracking_frames = torch.zeros(n,device=device)
        self.frames = torch.zeros(2,len(COURSES),self.num_modes,dtype=torch.long,device=device)
        self.contact_frames = torch.zeros(2,len(COURSES),dtype=torch.long,device=device)
        self.episodes = torch.zeros_like(self.frames)
        self.successes = torch.zeros_like(self.frames)
        self.failures = torch.zeros_like(self.frames)
        if self.motion_stability is not None:
            self.command_ramp = torch.zeros(n, device=device)
            self.arc_transition = torch.zeros(n, dtype=torch.bool, device=device)
            self.last_root_xy = torch.zeros(n, 2, device=device)
            self.last_support_xy = torch.zeros(n, 2, device=device)
            self.support_position_valid = torch.zeros(n, dtype=torch.bool, device=device)
            self.lateral_actual = torch.zeros(n, device=device)
            self.lateral_support_actual = torch.zeros(n, device=device)
            self.lateral_expected = torch.zeros(n, device=device)
            self.command_energy = torch.zeros(n, 2, device=device)
            self.tracking_ok = torch.zeros(n, dtype=torch.bool, device=device)
            self.displacement_ok = torch.ones(n, dtype=torch.bool, device=device)
            self.stop_ok = torch.zeros(n, dtype=torch.bool, device=device)
            # Independent failure indicators: a timed-out episode can also
            # fail tracking/stop. They are not an exclusive partition.
            self.failure_counts = {key:torch.zeros_like(self.episodes)
                for key in ('physical', 'tracking', 'displacement', 'stop', 'timeout')}
            self.motion_totals = {key:torch.zeros_like(self.episodes, dtype=torch.float64)
                for key in ('linear_tracking_rms_sum', 'yaw_tracking_rms_sum',
                            'lateral_actual_m_sum', 'lateral_expected_m_sum',
                            'lateral_support_actual_m_sum')}
            self.sequence_counts = {key:torch.zeros_like(self.episodes)
                for key in ('transition_episodes', 'transition_successes',
                            'abrupt_episodes', 'abrupt_successes')}
        if self.compound:
            self.compound_secondary_last = torch.zeros(n, dtype=torch.bool, device=device)
            self.axis_tracking_error = torch.zeros(n, 3, device=device)
            self.axis_command_energy = torch.zeros(n, 3, device=device)
            self.motion_totals.update({key:torch.zeros_like(self.episodes, dtype=torch.float64)
                for key in ('vx_tracking_rms_sum', 'vy_tracking_rms_sum', 'wz_tracking_rms_sum')})
        self.walls = None
        if self.has_walls:
            from .wall_practice import WallPractice
            self.walls = WallPractice(self)
        self.physics = None
        self.update_id = 0
        self.last_tick = -1
        self.geometry = None
        self.ray_error_max = 0.
        self.ray_checks = 0
        self.ray_checks_by_course = torch.zeros(len(COURSES),dtype=torch.long,device=device)
        self.platform_risk = torch.zeros(n,device=device)
        self.measure_tick = -1
        self.straight_frames = torch.zeros(2, device=device, dtype=torch.long)
        self.straight_clearance = torch.zeros(2,4, device=device, dtype=torch.float64)
        self.straight_airborne = torch.zeros_like(self.straight_clearance)
        self.rank = torch.zeros(n,dtype=torch.long,device=device)
        for c in range(len(COURSES)):
            for t in (self.teacher, ~self.teacher):
                ids = ((self.course_ids == c) & t).nonzero().flatten()
                self.rank[ids] = torch.arange(len(ids),device=device)

    @property
    def stairs(self):
        return self.course_ids < 3

    def raw_command(self):
        age = self.env.episode_length_buf.double()*self.env.step_dt
        cfg = self.cfg['commands']
        active = (age >= cfg['initial_hold_s']) & torch.where(self.stairs, ~self.reached,
                   age < cfg['initial_hold_s']+self.move_duration)
        if self.motion_stability is None:
            return self.command_value*active[:,None]
        sequence = command_sequence(self.command_value, age-cfg['initial_hold_s'],
            self.move_duration, self.command_ramp, self.arc_transition)
        if self.compound:
            from .compound_commands import compound_sequence
            combined = (self.command_value.abs() > 0).sum(-1) == 2
            replacement = compound_sequence(self.command_value, age-cfg['initial_hold_s'],
                self.move_duration, self.command_ramp, self.arc_transition, self.compound_secondary_last)
            sequence = torch.where(combined[:, None], replacement, sequence)
        practice = (self.course_ids >= 3) & (self.course_ids <= 7)
        result = torch.where(practice[:,None],sequence,self.command_value)*active[:,None]
        if self.walls is not None:
            result = torch.where(self.walls.active[:,None],self.walls.command(),result)
        return result

    def tags(self):
        cmd = self.raw_command()
        straight = (self.course_ids == 8) & (cmd[:,0] > 0)
        anchor = (self.course_ids == 3) | (self.course_ids == 8) | (self.course_ids == 9)
        anchor |= self.stairs & (self.geometry_level == 0)
        if self.walls is not None:anchor &= ~self.walls.active
        return {'cts_straight':straight.clone(), 'cts_anchor':anchor.clone(), 'cts_course':self.course_ids.clone()}

    def reset(self, ids):
        env, cfg = self.env, self.cfg['commands']
        if not len(ids):return
        mode = (self.resets[ids]+self.rank[ids]) % self.base_num_modes
        mode[self.stairs[ids] | (self.course_ids[ids] == 8)] = 0
        platform = self.course_ids[ids] == 9
        platform_mode = (self.resets[ids[platform]]+self.rank[ids[platform]]) % 9
        mode[platform] = torch.where(platform_mode < 6, 0, 8)
        if self.has_walls:
            from .wall_practice import KINDS, MODE_START
            motion = self.course_ids[ids] == 3
            wall = motion & torch.as_tensor(self.rng.random(len(ids)) <
                self.cfg['wall_practice']['episode_fraction'],device=ids.device)
            kind = (self.resets[ids]+self.rank[ids]) % len(KINDS)
            mode[wall] = MODE_START+kind[wall]
        self.mode[ids] = mode
        groups = torch.tensor(MODE_GROUPS[:self.num_modes],device=ids.device)[mode]
        self.group[ids] = self.course_ids[ids]*self.num_groups+groups
        tiles, levels, domains, versions, roles = [], [], [], [], []
        for c, group, current_mode in zip(self.course_ids[ids].tolist(),self.group[ids].tolist(),mode.tolist()):
            draw = self.curriculum.sample(group)
            g = draw['geometry_level'];pool = self.pools[c,g]
            if self.has_walls and c == 3:
                from .wall_practice import KINDS, MODE_START
                kind = KINDS[current_mode-MODE_START] if current_mode >= MODE_START else None
                pool = np.array([i for i in pool if self.atlas.records[i].get('wall_kind') == kind],dtype=np.int64)
                if not len(pool):raise ValueError('Wall course has no matching physical geometry pool')
            # Explicit boundary exposure, still sampling actual continuously generated geometry.
            edge = np.array([i for i in pool if self.atlas.records[i]['boundary']],dtype=np.int64)
            choices = edge if len(edge) and self.rng.random() < .25 else pool
            tiles.append(int(self.rng.choice(choices)))
            levels.append(g);domains.append(draw['dr_level']);versions.append(draw['version']);roles.append(draw['role'])
        for target, value in ((self.tile,tiles),(self.geometry_level,levels),(self.dr_level,domains),(self.version,versions),(self.role,roles)):
            target[ids] = torch.tensor(value,device=ids.device,dtype=target.dtype)
        origin = self.origins[self.tile[ids]]
        env.scene.terrain.env_origins[ids] = origin
        env.scene.env_origins[ids] = origin
        env.scene.terrain.terrain_levels[ids] = self.geometry_level[ids]
        env.scene.terrain.terrain_types[ids] = self.tile[ids]
        robot = env.scene['robot']
        position = origin+robot.data.default_root_state[ids,:3]
        # Small-radius arcs can circle forever inside the flat launch pad.
        # All turning/stationary modes therefore start on actual terrain.
        # Place one front support near a real obstacle/pit, with spawn height
        # derived from the highest neighbouring support to avoid penetration.
        on_terrain = (mode >= 2) & (mode < 19)
        for row in (on_terrain & (self.course_ids[ids] >= 3) & (self.course_ids[ids] <= 7)).nonzero().flatten().tolist():
            record = self.atlas.records[int(self.tile[ids[row]])]
            features = record.get('obstacles',record.get('pits',[]))
            if features:
                feature = features[int(self.rng.integers(len(features)))]
                position[row,0] += feature['x']-env.cfg.stair_tail.body_half_length
                position[row,1] += feature['y']-env.cfg.stair_tail.body_half_width
            else:
                position[row,0] += self.rng.choice((-1.,1.))*cfg['terrain_exposure_m']
        approach = platform & (mode == 0)
        position[approach,0] += self.ends[self.tile[ids[approach]]]-env.cfg.cts_platform_practice['low_start_from_end']
        position[:,:2] += torch.empty(len(ids),2,device=ids.device).uniform_(-.05,.05)
        if self.motion_stability is not None:
            lateral_ids = (mode == 9) | (mode == 10)
            # Reserve the commanded displacement plus a conservative rotating
            # support footprint. Recompute the true terrain support below.
            position[lateral_ids,:2] = lateral_spawn(position[lateral_ids,:2],
                self.bounds[self.tile[ids[lateral_ids]]], mode[lateral_ids],
                self.motion_stability['lateral_travel_m'],
                env.cfg.stair_tail.body_half_length, env.cfg.stair_tail.body_half_width,
                env.cfg.stair_tail.wheel_radius, env.cfg.stair_tail.wheel_half_length,
                self.cfg['rewards']['boundary_margin'])
        z,normal = self.surface.sample(self.tile[ids],position[:,:2])
        offsets = torch.tensor([[x,y] for x in (-env.cfg.stair_tail.body_half_length,env.cfg.stair_tail.body_half_length)
                               for y in (-env.cfg.stair_tail.body_half_width,env.cfg.stair_tail.body_half_width)],device=ids.device)
        support_top,_ = self.surface.sample(self.tile[ids],position[:,None,:2]+offsets)
        spawn_z = torch.maximum(z,support_top.amax(-1))
        position[:,2] = spawn_z+robot.data.default_root_state[ids,2]
        yaw = torch.empty(len(ids),device=ids.device).uniform_(-.1,.1)
        self.initial_yaw[ids] = yaw
        from .motion import start_quaternion
        quat = start_quaternion(yaw,normal)
        forward = torch.empty(len(ids),device=ids.device).uniform_(*cfg['forward_speed'])
        forward[platform] = torch.empty(int(platform.sum()),device=ids.device).uniform_(*env.cfg.cts_platform_practice['approach_speed_range'])
        backward = torch.empty(len(ids),device=ids.device).uniform_(*cfg['backward_speed'])
        turn = torch.empty(len(ids),device=ids.device).uniform_(*cfg['yaw_rate'])
        vx = torch.where((mode == 1) | (mode == 6) | (mode == 7), -backward, forward)
        vx[(mode == 2) | (mode == 3) | (mode == 8)] = 0
        wz = torch.zeros_like(vx)
        turning = (mode >= 2) & (mode < 8)
        wz[turning] = turn[turning]*torch.where(mode[turning]%2 == 0,1.,-1.)
        wz[(mode >= 4) & (mode < 8)] *= .5
        vy = torch.zeros_like(vx)
        if self.motion_stability is not None:
            st = self.motion_stability
            lateral_ids = (mode == 9) | (mode == 10)
            vx[lateral_ids] = 0
            lateral_speed = torch.as_tensor(self.rng.uniform(*st['lateral_speed'],len(ids)),
                                            device=ids.device,dtype=vx.dtype)
            vy[lateral_ids] = lateral_speed[lateral_ids]*torch.where(mode[lateral_ids] == 9,1.,-1.)
            ramp = self.rng.uniform(*st['command_ramp_s'],len(ids))
            abrupt = self.rng.random(len(ids)) < st['abrupt_command_probability']
            # Only the general motion courses change their command schedule;
            # staircase goal/stop, straight and platform semantics stay intact.
            practice = (self.course_ids[ids] >= 3) & (self.course_ids[ids] <= 7)
            self.command_ramp[ids] = torch.as_tensor(np.where(abrupt,0.,ramp),
                device=ids.device,dtype=vx.dtype)*practice
            transition = torch.as_tensor(self.rng.random(len(ids)) < st['arc_transition_probability'],
                                         device=ids.device)
            self.arc_transition[ids] = transition & (mode >= 4) & (mode < 8)
        self.command_value[ids] = torch.stack((vx,vy,wz),-1)
        if self.compound:
            from .compound_commands import sample_commands
            replacement = sample_commands(torch.where(mode >= 19,0,mode), self.cfg['compound_commands'])
            self.command_value[ids] = torch.where(practice[:, None], replacement, self.command_value[ids])
            vx, vy, wz = self.command_value[ids].unbind(-1)
            combined = practice & ((self.command_value[ids].abs() > 0).sum(-1) == 2)
            selected = torch.as_tensor(self.rng.random(len(ids)) <
                self.cfg['compound_commands']['transition_probability'], device=ids.device)
            self.arc_transition[ids] = selected & combined
            self.compound_secondary_last[ids] = torch.as_tensor(self.rng.random(len(ids)) < .5, device=ids.device)
            self.axis_tracking_error[ids] = 0.
            self.axis_command_energy[ids] = 0.
        self.move_duration[ids] = torch.where(self.course_ids[ids] == 8,cfg['straight_move_s'],cfg['move_s'])
        # Slow backward commands must leave the spawn pad and reach the terrain.
        translating = vx.abs() > 0
        exposure_time = cfg['terrain_exposure_m'] / vx.abs().clamp_min(.1)
        self.move_duration[ids] = torch.where(translating & ~self.stairs[ids] & ~platform,
            torch.maximum(self.move_duration[ids],exposure_time),self.move_duration[ids])
        if self.motion_stability is not None:
            lateral_ids = (mode == 9) | (mode == 10)
            self.move_duration[ids[lateral_ids]] = lateral_duration(vy[lateral_ids].abs(),
                self.motion_stability['lateral_travel_m'], self.command_ramp[ids[lateral_ids]])
            if self.compound:
                from .compound_commands import reserve_compound_path
                chosen = ids[combined]
                speed = self.command_value[chosen, :2].norm(dim=-1)
                self.move_duration[chosen] = self.cfg['compound_commands']['travel_m']/speed
                self.command_ramp[chosen] = torch.minimum(self.command_ramp[chosen], self.move_duration[chosen]/8)
                position[combined, :2] = reserve_compound_path(position[combined, :2],
                    self.bounds[self.tile[chosen]], self.cfg['compound_commands']['travel_m'],
                    env.cfg.stair_tail, self.cfg['rewards']['boundary_margin'])
            sequence_ids = combined if self.compound else self.arc_transition[ids]
            if sequence_ids.any():
                chosen = ids[sequence_ids]
                if not self.compound:
                    position[sequence_ids,:2] = transition_spawn(position[sequence_ids,:2],
                        self.bounds[self.tile[chosen]], self.command_value[chosen],
                        self.move_duration[chosen], self.command_ramp[chosen], yaw[sequence_ids],
                        env.cfg.stair_tail.body_half_length, env.cfg.stair_tail.body_half_width,
                        env.cfg.stair_tail.wheel_radius, env.cfg.stair_tail.wheel_half_length,
                        self.cfg['rewards']['boundary_margin'])
                z,normal = self.surface.sample(self.tile[chosen],position[sequence_ids,:2])
                support_top,_ = self.surface.sample(self.tile[chosen],position[sequence_ids,None,:2]+offsets)
                position[sequence_ids,2] = torch.maximum(z,support_top.amax(-1))+robot.data.default_root_state[chosen,2]
                quat[sequence_ids] = start_quaternion(yaw[sequence_ids],normal)
        if self.walls is not None:
            self.walls.reset(ids,position,yaw,quat)
            self.initial_yaw[ids] = yaw
            chosen = ids[self.walls.active[ids]]
            self.move_duration[chosen] = self.cfg['wall_practice']['timeout_s']
            self.command_value[chosen] = torch.stack((self.walls.speed[chosen],
                torch.zeros_like(self.walls.speed[chosen]),torch.zeros_like(self.walls.speed[chosen])),-1)
            self.arc_transition[chosen] = False
        robot.write_root_pose_to_sim(torch.cat((position,quat),-1),env_ids=ids)
        robot.write_root_velocity_to_sim(torch.zeros(len(ids),6,device=ids.device),env_ids=ids)
        route = (self.goals[self.tile[ids],0]-origin[:,0]).clamp_min(0)
        self.timeout[ids] = torch.where(self.stairs[ids], route/cfg['forward_speed'][0]+cfg['route_time_margin_s'],
            self.move_duration[ids]+cfg['initial_hold_s']+cfg['stop_s'])
        if self.physics is None:
            from .all_terrain_physics import ContinuousPhysics
            self.physics = ContinuousPhysics(env,self.cfg['domain'])
        self.physics.apply(ids,self.dr_level[ids])
        for tensor in (self.success,self.failed,self.reached,self.finished,self.foot_passed,self.stable_age,
                       self.progress,self.tracking_error,self.tracking_frames):
            tensor[ids] = 0
        if self.motion_stability is not None:
            self.last_root_xy[ids] = position[:,:2]
            self.lateral_actual[ids] = self.lateral_expected[ids] = 0.
            self.lateral_support_actual[ids] = 0.
            self.support_position_valid[ids] = False
            self.command_energy[ids] = 0.
            self.tracking_ok[ids] = self.stop_ok[ids] = False
            self.displacement_ok[ids] = (mode != 9) & (mode != 10)
        self.best_x[ids] = position[:,0]
        self.resets[ids] += 1
        self.measure_tick = -1
        self.geometry = None

    def observe(self):
        """Pure current geometry, safe during observation-manager construction/reset."""
        env = self.env
        if self.measure_tick == env.common_step_counter and self.geometry is not None:
            return self.geometry
        robot, sensor = env.scene['robot'],env.scene.sensors['contact_forces']
        names = [f'{leg}_wheel' for leg in ('fl','fr','hl','hr')]
        if not hasattr(self,'body_ids'):
            self.body_ids = robot.find_bodies(names,preserve_order=True)[0]
            self.contact_ids = sensor.find_bodies(names,preserve_order=True)[0]
        d = robot.data
        pos = d.body_link_pos_w[:,self.body_ids]
        from task_support.stair_quiet_v1.signals import wheel_axis_world
        axis = wheel_axis_world(d.body_link_quat_w[:,self.body_ids],env.cfg.stair_quiet,validate=False)
        height, normal = self.surface.sample(self.tile,pos[...,:2])
        radius,half = env.cfg.stair_tail.wheel_radius,env.cfg.stair_tail.wheel_half_length
        clearance = (pos[...,2]-height)*normal[...,2]-radius-half*(axis*normal).sum(-1).abs()
        force = sensor.data.net_forces_w[:,self.contact_ids]
        force_n = (force*normal).sum(-1)
        supported = (force_n > env.cfg.stair_tail.support_force_N) & (clearance.abs() < .05) & torch.isfinite(height)
        offset = -radius*normal-half*torch.sign((axis*normal).sum(-1))[...,None]*axis
        offset += pos-d.body_com_pos_w[:,self.body_ids]
        velocity = d.body_lin_vel_w[:,self.body_ids]+torch.linalg.cross(d.body_ang_vel_w[:,self.body_ids],offset)
        tangent = velocity-(velocity*normal).sum(-1,keepdim=True)*normal
        slip = tangent.norm(dim=-1)*supported
        root_height, root_normal = self.surface.sample(self.tile,d.root_pos_w[:,:2])
        cmd = env.command_manager.get_command('base_velocity')
        error = (d.root_lin_vel_b[:,:2]-cmd[:,:2]).square().sum(-1)
        yaw_error = (d.root_ang_vel_b[:,2]-cmd[:,2]).square()
        cc = self.cfg['commands']
        stopped = (d.root_lin_vel_b[:,:2].norm(dim=-1) <= cc['stop_speed_m_s']) & (d.root_ang_vel_b[:,2].abs() <= cc['stop_yaw_rad_s'])
        self.geometry = {'position':pos,'normal':normal,'height':height,'clearance':torch.nan_to_num(clearance),
            'supported':supported,'force':force,'force_n':force_n,'velocity':velocity,'axis':axis,
            'slip':torch.nan_to_num(slip),'root_normal':root_normal,'root_height':root_height,
            'error':error,'yaw_error':yaw_error,'stopped':stopped,'command':cmd.clone(),
            'measured_velocity':torch.cat((d.root_lin_vel_b[:, :2], d.root_ang_vel_b[:, 2:3]), -1)}
        self.measure_tick = env.common_step_counter
        return self.geometry

    def advance(self):
        """Account exactly once, after physics and before termination/reset."""
        env = self.env
        if self.last_tick == env.common_step_counter:return
        g = self.observe()
        d = env.scene['robot'].data
        pos, supported = g['position'],g['supported']
        root_height,root_normal = g['root_height'],g['root_normal']
        half = env.cfg.stair_tail.wheel_half_length
        b = self.bounds[self.tile];rw = self.cfg['rewards']
        root = d.root_pos_w
        outside = (root[:,0] < b[:,0]+rw['boundary_margin']) | (root[:,0] > b[:,1]-rw['boundary_margin'])
        outside |= (root[:,1] < b[:,2]+rw['boundary_margin']) | (root[:,1] > b[:,3]-rw['boundary_margin'])
        lateral = pos[...,1]-self.origins[self.tile,1,None]
        outside |= self.stairs & ((lateral.abs()+half) > self.widths[self.tile,None]/2).any(-1)
        upright = -d.projected_gravity_b[:,2]
        low = (root[:,2]-root_height)*root_normal[:,2] < rw['min_root_clearance']
        guard = getattr(env, 'cts_limit_monitor', None)
        if guard is not None:
            self.failed |= guard.faulted
        if self.walls is not None:
            self.walls.measure()
            self.failed |= self.walls.active & (self.walls.clearance < -self.walls.cfg['penetration_limit'])
        self.failed |= outside | low | (upright < rw['min_upright']) | ~torch.isfinite(root).all(-1) | ~torch.isfinite(root_height)
        platform = self.course_ids == 9
        from .platform import height_risk
        height_risk_value = height_risk(self.heights[self.tile],env.cfg.cts_platform_practice)*platform
        front_margin = self.ends[self.tile]-(pos[...,0]-self.origins[self.tile,0,None]).amax(-1)-env.cfg.stair_tail.wheel_radius
        side_margin = self.widths[self.tile]/2-lateral.abs().amax(-1)-env.cfg.stair_tail.wheel_radius
        near = (1-torch.minimum(front_margin,side_margin)/env.cfg.cts_platform_practice['edge_margin']).clamp(0.,1.)
        self.platform_risk = height_risk_value*near.square()*(3-2*near)
        platform_drop = root[:,2] < self.origins[self.tile,2]+d.default_root_state[:,2]-self.heights[self.tile]*.5
        self.failed |= platform & (height_risk_value > .5) & platform_drop
        self.progress = (root[:,0]-self.best_x).clamp_min(0)*self.stairs
        self.best_x = torch.maximum(root[:,0],self.best_x)
        passed = pos[...,0] > self.goals[self.tile,0,None]+.03
        self.foot_passed = (self.foot_passed | (passed & supported)) & passed
        self.reached |= self.stairs & self.foot_passed.all(-1)
        cmd,error,yaw_error = g['command'],g['error'],g['yaw_error']
        age = env.episode_length_buf.double()*env.step_dt
        cc = self.cfg['commands']
        moving_phase = (age >= cc['initial_hold_s']) & torch.where(self.stairs,~self.reached,age < cc['initial_hold_s']+self.move_duration)
        if self.walls is not None:
            moving_phase = torch.where(self.walls.active,cmd.abs().amax(-1) > 1.e-4,moving_phase)
        self.tracking_error += torch.stack((error,yaw_error),-1)*moving_phase[:,None]
        self.tracking_frames += moving_phase
        stopped = g['stopped']
        stable = stopped & (upright > .85) & (supported.sum(-1) >= 2) & ~self.failed
        stop_phase = torch.where(self.stairs,self.reached,age >= cc['initial_hold_s']+self.move_duration)
        if self.walls is not None:
            stop_phase = torch.where(self.walls.active,self.walls.done,stop_phase)
        self.stable_age = torch.where(stable & stop_phase,self.stable_age+env.step_dt,0.)
        tolerance = torch.stack(((self.command_value[:,:2].norm(dim=-1)*cc['relative_tracking_tolerance']+cc['tracking_floor_m_s']).clamp(max=cc['tracking_rms_m_s']),
                                 (self.command_value[:,2].abs()*cc['relative_tracking_tolerance']+cc['yaw_tracking_floor_rad_s']).clamp(max=cc['yaw_tracking_rms_rad_s'])),-1)
        if self.motion_stability is not None:
            self.command_energy += torch.stack((cmd[:,:2].square().sum(-1),cmd[:,2].square()),-1)*moving_phase[:,None]
            # Straight segments must not dilute yaw error enough that never
            # turning passes a turning sequence. Tighten only this new case;
            # fixed-command and all legacy course tolerances are unchanged.
            yaw_rms = (self.command_energy[:,1]/self.tracking_frames.clamp_min(1)).sqrt()
            sequence_tolerance = (yaw_rms*cc['relative_tracking_tolerance']).clamp_min(cc['yaw_tracking_floor_rad_s'])
            tolerance[:,1] = torch.where(self.arc_transition,
                torch.minimum(tolerance[:,1],sequence_tolerance),tolerance[:,1])
        if self.walls is not None:
            rms_command = (self.command_energy/self.tracking_frames.clamp_min(1)[:,None]).sqrt()
            floor = rms_command.new_tensor([cc['tracking_floor_m_s'],cc['yaw_tracking_floor_rad_s']])
            cap = rms_command.new_tensor([cc['tracking_rms_m_s'],cc['yaw_tracking_rms_rad_s']])
            wall_tolerance = torch.minimum(rms_command*cc['relative_tracking_tolerance']+floor,cap)
            tolerance = torch.where(self.walls.active[:,None],wall_tolerance,tolerance)
        if self.compound:
            measured = g['measured_velocity']
            self.axis_tracking_error += (measured-cmd).square()*moving_phase[:, None]
            self.axis_command_energy += cmd.square()*moving_phase[:, None]
        good_tracking = ((self.tracking_error/self.tracking_frames.clamp_min(1)[:,None]).sqrt() <= tolerance).all(-1)
        if self.compound:
            from .compound_commands import axis_tracking_ok
            use = (self.course_ids >= 3) & (self.course_ids <= 7) & (self.mode != 8)
            if self.walls is not None:use &= ~self.walls.active
            good_tracking &= ~use | axis_tracking_ok(self.axis_tracking_error,
                self.axis_command_energy, self.tracking_frames, cc)
        if self.motion_stability is not None:
            lateral_ids = (self.mode == 9) | (self.mode == 10)
            left_axis = torch.stack((-self.initial_yaw.sin(),self.initial_yaw.cos()),-1)
            # World position differences, projected onto the initial horizontal
            # left axis, cannot mistake body roll/tilt for lateral translation.
            delta_xy = root[:,:2]-self.last_root_xy
            signed_step = (delta_xy*left_axis).sum(-1)*self.command_value[:,1].sign()
            self.lateral_actual += signed_step*moving_phase*lateral_ids
            # A body lean can translate the root while all feet stay planted.
            # Require the wheel footprint itself to translate as well. Average
            # all wheel centers so gait contact switches cannot jump the datum.
            support_xy = pos[...,:2].mean(1)
            support_delta = support_xy-self.last_support_xy
            support_step = (support_delta*left_axis).sum(-1)*self.command_value[:,1].sign()
            self.lateral_support_actual += support_step*moving_phase*lateral_ids*self.support_position_valid
            self.last_support_xy.copy_(support_xy)
            self.support_position_valid.fill_(True)
            self.lateral_expected += cmd[:,1].abs()*env.step_dt*moving_phase*lateral_ids
            self.last_root_xy.copy_(root[:,:2])
            self.tracking_ok.copy_(good_tracking)
            self.displacement_ok = (~lateral_ids) | ((self.lateral_expected > 0) &
                (torch.minimum(self.lateral_actual,self.lateral_support_actual) >=
                 self.lateral_expected*(1-cc['relative_tracking_tolerance'])))
            good_tracking &= self.displacement_ok
            self.stop_ok.copy_(self.stable_age >= cc['success_hold_s'])
        high_stop = (height_risk_value > .5) & (self.platform_risk > .5) & stopped
        descended = (height_risk_value <= .5) & (
            pos[...,0] > self.origins[self.tile,0,None]+self.ends[self.tile,None]).all(-1)
        platform_complete = torch.where(self.mode == 8,good_tracking,high_stop | (descended & good_tracking))
        good_tracking = torch.where(platform,platform_complete,good_tracking)
        complete = torch.where(self.stairs,self.reached,good_tracking & (self.tracking_frames > 0))
        if self.walls is not None:
            complete &= ~self.walls.active | self.walls.done
        self.success |= complete & (self.stable_age >= cc['success_hold_s']) & ~self.failed
        flat = ((~self.teacher).long()*len(COURSES)+self.course_ids)*self.num_modes+self.mode
        self.frames.view(-1).add_(torch.bincount(flat,minlength=self.frames.numel()))
        contact_cell = (~self.teacher).long()*len(COURSES)+self.course_ids
        self.contact_frames.view(-1).add_(torch.bincount(contact_cell[supported.any(-1)],
            minlength=self.contact_frames.numel()))
        for index,cohort in enumerate((self.teacher,~self.teacher)):
            selected = cohort & (self.course_ids == 8) & (cmd[:,0] > 0)
            self.straight_frames[index] += selected.sum()
            self.straight_clearance[index] += g['clearance'][selected].clamp_min(0).double().sum(0)
            self.straight_airborne[index] += ((g['clearance'][selected] > rw['flat_clearance_free_m']) & ~supported[selected]).double().sum(0)
        self.last_tick = env.common_step_counter

    def record(self, ids):
        if not len(ids):return
        done = self.env.termination_manager.terminated | self.env.termination_manager.time_outs
        ids = ids[done[ids] & ~self.finished[ids] & (self.env.episode_length_buf[ids] > 0)]
        from task_support.student_v1.environment import sensor_state
        delayed = ~sensor_state(self.env).zero_delivery
        for i in ids.tolist():
            success = bool(self.success[i] and not self.failed[i])
            cohort,c,mode = int(not self.teacher[i]),int(self.course_ids[i]),int(self.mode[i])
            self.episodes[cohort,c,mode] += 1
            self.successes[cohort,c,mode] += success
            self.failures[cohort,c,mode] += not success
            if self.motion_stability is not None:
                cell = cohort,c,mode
                if not success:
                    conditions = {'physical':self.failed[i],
                        'tracking':(self.tracking_frames[i] > 0) & ~self.tracking_ok[i],
                        'displacement':~self.displacement_ok[i],
                        'stop':~self.failed[i] & ~self.stop_ok[i],
                        'timeout':self.env.termination_manager.time_outs[i]}
                    for key,condition in conditions.items():
                        self.failure_counts[key][cell] += bool(condition)
                rms = (self.tracking_error[i]/self.tracking_frames[i].clamp_min(1)).sqrt()
                for key,value in zip(self.motion_totals,
                        (rms[0],rms[1],self.lateral_actual[i],self.lateral_expected[i],self.lateral_support_actual[i])):
                    self.motion_totals[key][cell] += value
                if self.compound:
                    axis_rms = (self.axis_tracking_error[i]/self.tracking_frames[i].clamp_min(1)).sqrt()
                    for axis, key in enumerate(('vx_tracking_rms_sum', 'vy_tracking_rms_sum', 'wz_tracking_rms_sum')):
                        self.motion_totals[key][cell] += axis_rms[axis]
                transition = bool(self.arc_transition[i])
                abrupt = bool(self.command_ramp[i] == 0)
                for prefix,selected in (('transition',transition),('abrupt',abrupt)):
                    self.sequence_counts[prefix+'_episodes'][cell] += selected
                    self.sequence_counts[prefix+'_successes'][cell] += selected and success
            self.curriculum.record(int(self.group[i]),success,bool(self.teacher[i]),int(self.role[i]),
                int(self.version[i]),self.update_id,boundary=bool(self.boundary_tiles[self.tile[i]]),delayed=bool(delayed[i]))
        self.finished[ids] = True

    def check_rays(self):
        hits = self.env.scene.sensors['height_scanner'].data.ray_hits_w
        expected,_ = self.surface.sample(self.tile,hits[...,:2])
        valid = torch.isfinite(hits).all(-1) & torch.isfinite(expected)
        if valid.any():
            per_env = valid.sum(-1)
            self.ray_checks_by_course.index_add_(0,self.course_ids,per_env)
            error = float((expected[valid]-hits[...,2][valid]).abs().max())
            self.ray_error_max = max(self.ray_error_max,error)
            self.ray_checks += int(valid.sum())
            if error > .005:
                delta = torch.where(valid,(expected-hits[...,2]).abs(),-1.)
                index = int(delta.argmax());row,col = divmod(index,hits.shape[1])
                tile = int(self.tile[row])
                raise RuntimeError(f'All-terrain collision/surface/ray geometry differs: {error}; '
                    f'tile={tile}, course={COURSES[int(self.course_ids[row])]}, hit={hits[row,col].tolist()}, '
                    f'expected={float(expected[row,col])}, origin={self.origins[tile].tolist()}, record={self.atlas.records[tile]}')

    def diagnostics(self):
        out = {'ray_checks':self.ray_checks,'ray_height_error_max_m':self.ray_error_max}
        episodes_by_mode, successes_by_mode, frames_by_mode = self.episodes, self.successes, self.frames
        if self.motion_stability is not None:
            # Copy the complete mode ledger in two batches. Converting each
            # CUDA scalar inside the reporting loops would synchronize the
            # device thousands of times per rollout. Keep zero-coverage cells
            # visible: absent left/right exposure must remain observable.
            mode_counts = {'episodes':self.episodes, 'successes':self.successes, 'frames':self.frames,
                **{'failure_'+key:value for key,value in self.failure_counts.items()},
                **self.sequence_counts}
            count_values = torch.stack(tuple(mode_counts.values())).detach().cpu()
            mode_counts = dict(zip(mode_counts, count_values))
            total_values = torch.stack(tuple(self.motion_totals.values())).detach().cpu()
            mode_totals = dict(zip(self.motion_totals, total_values))
            episodes_by_mode = mode_counts['episodes']
            successes_by_mode = mode_counts['successes']
            frames_by_mode = mode_counts['frames']
        for cohort,label in enumerate(('teacher','student')):
            for c,name in enumerate(COURSES):
                count = int(episodes_by_mode[cohort,c].sum())
                out[f'{label}/{name}/episodes'] = count
                out[f'{label}/{name}/success_rate'] = float(successes_by_mode[cohort,c].sum())/max(count,1)
                out[f'{label}/{name}/frames'] = int(frames_by_mode[cohort,c].sum())
                out[f'{label}/{name}/supported_frames'] = int(self.contact_frames[cohort,c])
                if self.motion_stability is not None:
                    for mode,command_name in enumerate(MODE_NAMES[:self.num_modes]):
                        cell = cohort,c,mode
                        episodes = int(episodes_by_mode[cell])
                        prefix = f'{label}/{name}/command/{command_name}/'
                        out[prefix+'episodes'] = episodes
                        out[prefix+'success_rate'] = float(successes_by_mode[cell])/max(episodes,1)
                        out[prefix+'frames'] = int(frames_by_mode[cell])
                        for reason in self.failure_counts:
                            out[prefix+'failure_'+reason] = int(mode_counts['failure_'+reason][cell])
                        for metric,values in mode_totals.items():
                            out[prefix+metric.removesuffix('_sum')] = float(values[cell])/max(episodes,1)
                        for metric in self.sequence_counts:
                            out[prefix+metric] = int(mode_counts[metric][cell])
        grades = self.curriculum.diagnostics()
        for c,name in enumerate(COURSES):
            out[f'rays/{name}/checked'] = int(self.ray_checks_by_course[c])
            for group,label in enumerate(GROUP_NAMES[:self.num_groups]):
                state = grades[c*self.num_groups+group]
                out[f'level/{name}/{label}/geometry'] = state['geometry_level']
                out[f'level/{name}/{label}/physics'] = state['dr_level']
                prefix = f'curriculum/{name}/{label}/'
                for key in ('version','promotions','demotions','ignored_role','ignored_version','expired_windows','completed_windows'):
                    out[prefix+key] = state[key]
                for key in ('episodes','successes','boundary','delayed'):
                    out[prefix+'window_'+key] = state['window'][key]
                last = state['last_window']
                out[prefix+'last_success_rate'] = last['successes']/last['episodes'] if last else -1.
                out[prefix+'consecutive_windows'] = state['consecutive_windows']
        if self.physics is not None:
            out.update({'physics/'+key:value for key,value in self.physics.diagnostics().items()})
        if self.walls is not None:
            out['walls/active_environments'] = int(self.walls.active.sum())
            out['walls/envelope_min_clearance_m'] = float(self.walls.clearance[self.walls.active].min()) if self.walls.active.any() else 0.
        for index,label in enumerate(('teacher','student')):
            for leg,name in enumerate(('FL','FR','HL','HR')):
                denominator = self.straight_frames[index].clamp_min(1)
                out[f'{label}/straight/{name}_clearance_m'] = float(self.straight_clearance[index,leg]/denominator)
                out[f'{label}/straight/{name}_airborne_fraction'] = float(self.straight_airborne[index,leg]/denominator)
        return out

    def validate_actual_contacts(self):
        # A static scene-query hit proves geometry/material cooking, but does
        # not prove that the GPU robot receives contact response from it.
        missing = [(label,name) for cohort,label in enumerate(('teacher','student'))
                   for c,name in enumerate(COURSES) if self.contact_frames[cohort,c] < 1]
        if missing:
            raise RuntimeError('No measured wheel/terrain support for: '+str(missing))

    def snapshot(self):
        result = {'curriculum':self.curriculum.state_dict(),'rng':deepcopy(self.rng.bit_generator.state),
            'course_ids':self.course_ids.cpu().clone(),'resets':self.resets.cpu().clone(),
            'frames':self.frames.cpu().clone(),'episodes':self.episodes.cpu().clone(),
            'successes':self.successes.cpu().clone(),'failures':self.failures.cpu().clone(),'update_id':self.update_id,
            **{k:getattr(self,k).cpu().clone() for k in ('contact_frames','straight_frames','straight_clearance','straight_airborne')}}
        if self.motion_stability is not None:
            result['motion_stability'] = {name:{key:value.cpu().clone() for key,value in getattr(self,name).items()}
                for name in ('failure_counts','motion_totals','sequence_counts')}
        return result

    def restore(self, saved):
        expected = self.snapshot()
        if set(saved) != set(expected):raise ValueError('All-terrain snapshot keys changed')
        if not torch.equal(saved['course_ids'],expected['course_ids']):raise ValueError('All-terrain allocation changed')
        checked = {}
        for name in ('resets','frames','episodes','successes','failures','contact_frames','straight_frames','straight_clearance','straight_airborne'):
            v = saved[name]
            if v.shape != expected[name].shape or v.dtype != expected[name].dtype or (v < 0).any():
                raise ValueError('Invalid all-terrain snapshot '+name)
            checked[name] = v.to(self.env.device)
        motion_checked = {}
        if self.motion_stability is not None:
            motion = saved['motion_stability']
            expected_motion = expected['motion_stability']
            if not isinstance(motion,dict) or set(motion) != set(expected_motion):
                raise ValueError('All-terrain motion snapshot keys changed')
            for name,values in motion.items():
                if not isinstance(values,dict) or set(values) != set(expected_motion[name]):
                    raise ValueError('All-terrain motion snapshot counters changed')
                motion_checked[name] = {}
                for key,value in values.items():
                    target = expected_motion[name][key]
                    if (not isinstance(value,torch.Tensor) or value.shape != target.shape
                            or value.dtype != target.dtype or not torch.isfinite(value).all()
                            or (name != 'motion_totals' and (value < 0).any())
                            or (name == 'motion_totals' and key not in ('lateral_actual_m_sum','lateral_support_actual_m_sum')
                                and (value < 0).any())):
                        raise ValueError('Invalid all-terrain motion snapshot '+name+'/'+key)
                    motion_checked[name][key] = value.to(self.env.device)
        self.curriculum.load_state_dict(saved['curriculum'])
        self.rng.bit_generator.state = deepcopy(saved['rng'])
        for name,value in checked.items():getattr(self,name).copy_(value)
        for name,values in motion_checked.items():
            for key,value in values.items():getattr(self,name)[key].copy_(value)
        self.update_id = saved['update_id']
        # Physics histories are intentionally restarted. In-flight episodes are
        # not counted as successes or failures at a checkpoint boundary.
        self.finished.fill_(True)
