"""Independent on-stair zero-command episodes; original traversal stays scored separately."""
from copy import deepcopy
import math
import numpy as np
import torch
from .all_terrain_curriculum import CourseCurriculum
from .stair_stop import yaw
from .timing import advance_steps, required_steps
from .stair_hold_spawn import candidates, sample_pose


def validate(c, terrain):
    keys={'enabled','episode_fraction','settle_grace_s','timeout_s','hold_s_range',
          'spawn_clearance_m','max_pitch_rad','rear_step_max','back_edge_margin_m',
          'riser_margin_m','minimum_spawn_interval_m','lateral_spawn_m','wheel_centers_body_m',
          'motion_deadband_m_s','position_deadband_m','motion_weight','success_reward','attitude_tolerance_rad','hold_speed_m_s','hold_yaw_rate_rad_s'}
    if not isinstance(c,dict) or set(c)!=keys or type(c['enabled']) is not bool:
        raise ValueError('Invalid stair_hold_practice settings')
    for k,v in c.items():
        if k in ('enabled','hold_s_range','wheel_centers_body_m'):continue
        if type(v) not in (int,float) or not math.isfinite(v) or v<0:
            raise ValueError('Invalid stair hold parameter '+k)
    if not 0<c['episode_fraction']<.5 or type(c['rear_step_max']) is not int:
        raise ValueError('Stair hold must preserve most original ascent episodes')
    times=c['hold_s_range']
    if not isinstance(times,list) or len(times)!=2 or not all(type(v) in (int,float) and math.isfinite(v) for v in times) or not 0<times[0]<=times[1]<c['timeout_s']-c['settle_grace_s']:
        raise ValueError('Invalid stair hold timing')
    raw=c['wheel_centers_body_m']
    if not isinstance(raw,list) or len(raw)!=4 or any(not isinstance(row,list) or len(row)!=3 or any(type(v) not in (int,float) for v in row) for row in raw):
        raise ValueError('Invalid stair hold wheel FK numbers')
    wheels=np.asarray(raw,dtype=float)
    if wheels.shape!=(4,3) or not np.isfinite(wheels).all() or not (np.allclose(wheels[:,2],wheels[0,2]) and np.allclose(wheels[:2,0],-wheels[2:,0]) and wheels[0,0]>0):
        raise ValueError('Stair hold requires verified symmetric default wheel FK')
    if not 0<c['spawn_clearance_m']<=.01 or not 0<c['max_pitch_rad']<math.acos(terrain['rewards']['min_upright']):
        raise ValueError('Stair hold pose violates physical upright/clearance limits')
    if c['enabled'] and not (terrain.get('stair_stop',{}).get('enabled') and terrain.get('stair_recovery',{}).get('enabled')):
        raise ValueError('Stair hold requires the original stop/recovery contract')
    if not 0<c['hold_speed_m_s']<=terrain['commands']['stop_speed_m_s'] or not 0<c['hold_yaw_rate_rad_s']<=terrain['commands']['stop_yaw_rad_s']:
        raise ValueError('Stair hold requires strict positive stationary tolerances')
    if not 0<c['attitude_tolerance_rad']<math.pi/2 or c['minimum_spawn_interval_m']<=0:
        raise ValueError('Invalid stair hold support tolerance')
    if c['success_reward']>terrain['rewards']['success']:
        raise ValueError('Stair hold must not increase route completion bonus')


class StairHoldPractice:
    def __init__(self,s):
        self.state=s;self.cfg=s.cfg['stair_hold_practice'];self.stop=s.cfg['stair_stop']
        self.rng=np.random.default_rng(s.cfg['geometry']['seed']+2491)
        self.curriculum=CourseCurriculum(s.cfg['curriculum'],num_courses=2)
        n,dev=s.env.num_envs,s.env.device
        self.active=torch.zeros(n,dtype=torch.bool,device=dev)
        self.anchor=torch.zeros(n,3,device=dev);self.quaternion=torch.zeros(n,4,device=dev);self.quaternion[:,0]=1.
        self.foot_anchor=torch.zeros(n,2,device=dev);self.previous_feet=self.foot_anchor.clone()
        self.max_drift=torch.zeros(n,device=dev);self.peak_speed=self.max_drift.clone()
        self.steps=torch.zeros(n,dtype=torch.long,device=dev);self.ready=self.active.clone()
        self.required=self.steps.clone();self.cost=self.max_drift.clone();self.linear=self.max_drift.clone();self.angular=self.max_drift.clone()
        self.tick=-1
        self.totals={k:torch.zeros(2,2,dtype=torch.float64,device=dev) for k in
            ('episodes','successes','timeouts','physical_failures','frames','stable_frames','front_rear_frames',
             'max_drift_m_sum','max_speed_m_s_sum','position_failures','support_failures','velocity_failures')}
        self.options={i:candidates(r,self.cfg,s.env.cfg.stair_tail.wheel_radius) for i,r in enumerate(s.atlas.records) if r['course'] in ('up','round_trip')}
        self.pools={}
        self.boundary_tiles=torch.zeros_like(s.boundary_tiles)
        for course in (0,2):
            for level in range(s.cfg['geometry']['geometry_levels']):
                pool=s.pools[course,level];good=np.array([i for i in pool if self.options[int(i)]],dtype=np.int64)
                if not len(good):raise ValueError(f'No feasible on-stair default support pose for course {course}, level {level}')
                self.pools[course,level]=good
                # Boundary is the hardest feasible support geometry in this pool.
                ranked=sorted(good,key=lambda i:max(x['pitch'] for x in self.options[int(i)]))
                self.boundary_tiles[ranked[-max(1,math.ceil(len(ranked)*.25)):]]=True
        self.initial_receipt=None

    def select(self,ids):
        s=self.state
        self.active[ids]=((s.course_ids[ids]==0)|(s.course_ids[ids]==2)) & torch.as_tensor(self.rng.random(len(ids))<self.cfg['episode_fraction'],device=ids.device)

    def reset(self,ids,position,quat):
        s=self.state;mask=self.active[ids];chosen=ids[mask]
        for value in (self.steps,self.ready,self.max_drift,self.peak_speed,self.cost,self.linear,self.angular):value[ids]=0
        if not len(chosen):return
        robot=s.env.scene['robot']
        for row,i in zip(mask.nonzero().flatten().tolist(),chosen.tolist()):
            tile=int(s.tile[i]);p,q=sample_pose(s.atlas.records[tile],self.options[tile],self.cfg,self.rng)
            position[row]=s.origins[tile]+position.new_tensor(p);quat[row]=quat.new_tensor(q)
        robot.write_joint_state_to_sim(robot.data.default_joint_pos[chosen],torch.zeros_like(robot.data.default_joint_pos[chosen]),env_ids=chosen)
        self.anchor[chosen]=position[mask];self.quaternion[chosen]=quat[mask]
        from task_support.student_v1.perception import quaternion_rotation
        rot=quaternion_rotation(quat[mask]);local=position.new_tensor(self.cfg['wheel_centers_body_m'])
        feet=(rot@local.T).transpose(1,2)+position[mask,None]
        self.foot_anchor[chosen]=feet[...,:2].mean(1);self.previous_feet[chosen]=self.foot_anchor[chosen]
        s.command_value[chosen]=0;s.initial_yaw[chosen]=0
        if s.recovery is not None:
            s.recovery.active[chosen]=False;s.recovery.phase[chosen]=0;s.recovery.perturb_pending[chosen]=False
        lo,hi=self.cfg['hold_s_range'];duration=lo+(hi-lo)*s.geometry_level[chosen].float()/max(1,s.cfg['geometry']['geometry_levels']-1)
        self.required[chosen]=torch.ceil(duration/s.env.step_dt-1.e-6).long()

    def validate_initial(self):
        """Actual Isaac default-joint FK after reset/forward, before any action."""
        s=self.state;ids=self.active.nonzero().flatten()
        if not len(ids):raise RuntimeError('Stair hold validation has no selected initial environments')
        g=s.observe();d=s.env.scene['robot'].data
        from task_support.student_v1.perception import quaternion_rotation
        rot=quaternion_rotation(d.root_quat_w[ids]);local=(g['position'][ids]-d.root_pos_w[ids,None])@rot
        for value in (local,d.root_pos_w[ids],d.root_quat_w[ids],g['position'][ids],g['height'][ids],g['normal'][ids]):
            if not torch.isfinite(value).all():raise RuntimeError('Nonfinite stair hold initial support geometry')
        target=local.new_tensor(self.cfg['wheel_centers_body_m']);err=float((local-target).abs().max())
        clearance=g['clearance'][ids]
        if err>.002 or float(clearance.min())<-.003 or float(clearance.max())>.015:
            raise RuntimeError(f'Stair hold default FK/support invalid: FK={err}, clearance={clearance.min().item(),clearance.max().item()}')
        if s.raw_command()[ids].abs().max()>0:raise RuntimeError('On-stair reset did not produce zero command')
        self.initial_receipt=dict(environments=len(ids),fk_max_error_m=err,min_wheel_clearance_m=float(clearance.min()),max_wheel_clearance_m=float(clearance.max()),raw_command_max=0.)
        return self.initial_receipt

    def advance(self,g):
        s,c,st=self.state,self.cfg,self.stop;env=s.env;d=env.scene['robot'].data
        if self.tick==env.common_step_counter:return
        self.tick=env.common_step_counter
        active=self.active;age=env.episode_length_buf.double()*env.step_dt
        measured=active & (age>=c['settle_grace_s'])
        feet=g['position'][...,:2].mean(1);v=d.root_lin_vel_w[:,:2];omega=d.root_ang_vel_b
        foot_v=(feet-self.previous_feet)/env.step_dt;self.previous_feet.copy_(feet)
        finite=torch.isfinite(d.root_pos_w).all(-1)&torch.isfinite(v).all(-1)&torch.isfinite(omega).all(-1)&torch.isfinite(d.root_quat_w).all(-1)
        for key in ('position','velocity','normal','force'):finite &= torch.isfinite(g[key]).all(dim=(-1,-2))
        delta=torch.stack((d.root_pos_w[:,:2]-self.anchor[:,:2],feet-self.foot_anchor),1)
        drift=torch.nan_to_num(delta.norm(dim=-1).amax(-1),nan=10.,posinf=10.)
        self.max_drift=torch.maximum(self.max_drift,torch.where(active,drift,0.))
        speed=torch.maximum(v.norm(dim=-1),foot_v.norm(dim=-1));self.peak_speed=torch.maximum(self.peak_speed,torch.where(measured,torch.nan_to_num(speed,nan=10.),0.))
        heading=yaw(d.root_quat_w);region=(delta[...,0].abs().amax(-1)<=st['backward_m'])&(delta[...,1].abs().amax(-1)<=st['lateral_m'])&(heading.abs()<=st['yaw_rad'])
        supported=g['supported']&finite[:,None];front_rear=supported[:,:2].any(-1)&supported[:,2:].any(-1)&(supported.sum(-1)>=3)
        # Contact must remain on raised steps; ground below the flight cannot pass.
        on_steps=(g['position'][...,2]-env.cfg.stair_tail.wheel_radius>s.origins[s.tile,2,None]+s.heights[s.tile,None]*.5).all(-1)
        tangent=g['velocity']-(g['velocity']*g['normal']).sum(-1,keepdim=True)*g['normal']
        slip=torch.nan_to_num(tangent.norm(dim=-1),nan=10.,posinf=10.)*supported
        velocity_ok=(speed<=c['hold_speed_m_s'])&(omega[:,2].abs()<=c['hold_yaw_rate_rad_s'])&(omega[:,:2].norm(dim=-1)<=st['roll_pitch_rate_rad_s'])&(slip.amax(-1)<=st['slip_tolerance_m_s'])
        # Limit attitude change relative to the feasible pitched reset, not global level.
        attitude=(d.root_quat_w*self.quaternion).sum(-1).abs().clamp(0,1)
        posture_ok=attitude>=math.cos(c['attitude_tolerance_rad']/2)
        # External zero command permits the existing internal heading feedback.
        tracking=g.get('tracking_command',g['command'])
        executed_zero=(g['command'].abs().amax(-1)<=1.e-4)&torch.isfinite(tracking).all(-1)&(tracking[:,:2].abs().amax(-1)<=1.e-4)
        self.ready.copy_(measured&executed_zero&finite&region&front_rear&on_steps&velocity_ok&posture_ok&~s.failed)
        self.steps=advance_steps(self.steps,self.ready)
        s.failed |= active & ~finite
        s.success |= active & (self.steps>=self.required) & ~s.failed
        pc=((delta.abs()-c['position_deadband_m']).clamp_min(0)/st['position_scale_m']).square().sum((-1,-2)).clamp(max=16)
        pc+=(heading/st['yaw_rad']).square().clamp(max=16)
        vc=((speed-c['motion_deadband_m_s']).clamp_min(0)/st['speed_scale_m_s']).square().clamp(max=16)
        sc=((slip-st['slip_tolerance_m_s']).clamp_min(0)/st['slip_scale_m_s']).square().clamp(max=16).mean(-1)
        support_cost=1-supported.float().mean(-1)+(~front_rear).float()
        posture=(omega[:,:2].norm(dim=-1)/st['roll_pitch_rate_rad_s']).square().clamp(max=16)
        self.cost.copy_(torch.where(measured,torch.nan_to_num(st['position_weight']*pc+c['motion_weight']*vc+st['slip_weight']*sc+st['support_weight']*support_cost+st['posture_weight']*posture,nan=32.,posinf=32.),0.))
        credit=torch.exp(-torch.nan_to_num(pc,nan=16.,posinf=16.))
        self.linear.copy_(torch.where(measured,torch.exp(-torch.nan_to_num(v.square().sum(-1),nan=10.)/st['speed_scale_m_s']**2)*credit,0.))
        self.angular.copy_(torch.where(measured,torch.exp(-torch.nan_to_num(omega[:,2].square(),nan=10.)/st['yaw_rate_scale_rad_s']**2)*credit,0.))
        cell=(~s.teacher[active]).long()*2+(s.course_ids[active]==2).long()
        values={'frames':measured,'stable_frames':self.ready,'front_rear_frames':measured&front_rear,'position_failures':measured&~region,'support_failures':measured&~front_rear,'velocity_failures':measured&~velocity_ok}
        for k,vv in values.items():self.totals[k].view(-1).index_add_(0,cell,vv[active].double())

    def record(self,ids,delayed):
        s=self.state;chosen=ids[self.active[ids]]
        for i in chosen.tolist():
            cell=int(not s.teacher[i]),int(s.course_ids[i]==2);success=bool(s.success[i] and not s.failed[i])
            for k,v in [('episodes',1),('successes',success),('timeouts',s.env.termination_manager.time_outs[i] and not success),('physical_failures',s.failed[i]),('max_drift_m_sum',self.max_drift[i]),('max_speed_m_s_sum',self.peak_speed[i])]:self.totals[k][cell]+=v
            self.curriculum.record(cell[1],success,bool(s.teacher[i]),int(s.role[i]),int(s.version[i]),s.update_id,boundary=bool(self.boundary_tiles[s.tile[i]]),delayed=bool(delayed[i]))
        s.finished[chosen]=True
        return ids[~self.active[ids]]

    def diagnostics(self):
        out={f'{co}/{course}/{key}':float(v[t,i]) for key,v in self.totals.items() for t,co in enumerate(('teacher','student')) for i,course in enumerate(('up','round_trip'))}
        for i,st in enumerate(self.curriculum.diagnostics()):
            for k in ('geometry_level','dr_level','promotions','demotions'):out[f'level/{i}/{k}']=st[k]
        out['active_environments']=int(self.active.sum())
        out['eligible_tiles']=sum(len(v) for v in self.pools.values());out['total_stair_tiles']=len(self.options)
        return out

    def snapshot(self):
        return dict(curriculum=self.curriculum.state_dict(),rng=deepcopy(self.rng.bit_generator.state),totals={k:v.cpu().clone() for k,v in self.totals.items()})

    def checked_snapshot(self,saved):
        if not isinstance(saved,dict) or set(saved)!= {'curriculum','rng','totals'} or saved['totals'].keys()!=self.totals.keys():raise ValueError('Invalid stair hold snapshot')
        cur=CourseCurriculum(self.state.cfg['curriculum'],num_courses=2);cur.load_state_dict(saved['curriculum'])
        rng=np.random.default_rng();rng.bit_generator.state=deepcopy(saved['rng'])
        for k,v in saved['totals'].items():
            target=self.totals[k]
            if not isinstance(v,torch.Tensor) or v.shape!=target.shape or v.dtype!=target.dtype or not torch.isfinite(v).all() or (v<0).any():raise ValueError('Invalid stair hold totals '+k)
        if (saved['totals']['successes']>saved['totals']['episodes']).any():raise ValueError('Invalid stair hold success ledger')
        return saved

    def restore(self,saved):
        self.curriculum.load_state_dict(saved['curriculum']);self.rng.bit_generator.state=deepcopy(saved['rng'])
        for k,v in saved['totals'].items():self.totals[k].copy_(v)
