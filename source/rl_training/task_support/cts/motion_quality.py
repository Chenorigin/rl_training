"""World heading and anchored platform stops, optional v21 training objectives.

All anchors are captured before the executed physics step. No actor observation
or deployed feedback changes. Old actor-yaw RMS remains a diagnostic.
"""
import math
import torch
from .stair_stop import yaw


def validate(c):
    keys = {'enabled','heading_scale_rad','heading_peak_rad','rate_scale_rad_s','rate_filter_s',
            'stop_speed_m_s','stop_yaw_rad_s','position_m','position_scale_m','braking_m_s2',
            'braking_max_m','position_weight','transition_weight','lateral_scale_m_s',
            'slip_scale_m_s','posture_scale_rad_s','transition_margin_m','landing_s'}
    if not isinstance(c,dict) or set(c)-{'idle_terminal_hold'}!=keys or type(c['enabled']) is not bool:
        raise ValueError('Invalid motion_quality settings')
    if 'idle_terminal_hold' in c and type(c['idle_terminal_hold']) is not bool:
        raise ValueError('idle_terminal_hold must be boolean')
    for k,v in c.items():
        if k not in ('enabled', 'idle_terminal_hold') and (type(v) not in (int,float) or not math.isfinite(v) or v<=0):
            raise ValueError('Invalid motion_quality value: '+k)
    if c['heading_peak_rad'] < c['heading_scale_rad']:
        raise ValueError('motion_quality peak below RMS tolerance')


def wrap(x):
    return torch.atan2(torch.sin(x),torch.cos(x))


class MotionQuality:
    def __init__(self,s):
        self.state=s;self.cfg=s.cfg['motion_quality'];n=s.env.num_envs;device=s.env.device
        self.selected=(s.course_ids==8)|(s.course_ids==9)
        z=torch.zeros(n,device=device)
        for k in ('heading','previous_yaw','filtered_rate','heading_sum','rate_sum','heading_peak',
                  'frames','stop_heading','braking','landing_age','linear','angular','cost'):
            setattr(self,k,z.clone())
        for k in ('moving','holding','seen_move','seen_hold','valid','heading_ok','hold_ok','transition','landed'):
            setattr(self,k,torch.zeros(n,dtype=torch.bool,device=device))
        self.landing_steps=torch.zeros(n,dtype=torch.long,device=device)
        self.anchor=torch.zeros(n,2,device=device);self.foot_anchor=self.anchor.clone()
        self.pre_command=torch.zeros(n,3,device=device)
        self.tick=-1;self.begin_tick=-1
        names=('moving_frames','heading_sq_rad','heading_peak_rad','filtered_rate_sq',
               'transition_frames','transition_cost','hold_frames','hold_ready_frames',
               'hold_outside_frames','hold_speed_sq','hold_yaw_rate_sq','hold_heading_sq','hold_drift_max_m','invalid_frames')
        self.totals={k:torch.zeros(2,2,dtype=torch.float64,device=device) for k in names}

    def reset(self,ids):
        for k,v in vars(self).items():
            if isinstance(v,torch.Tensor) and k!='selected':v[ids]=0

    def begin_step(self):
        s=self.state
        if self.begin_tick==s.env.common_step_counter:return
        self.begin_tick=s.env.common_step_counter
        d=s.env.scene['robot'].data;c=s.raw_command();angle=yaw(d.root_quat_w)
        moving=self.selected & (s.mode==0) & (c[:,0]>1.e-4)
        holding=(s.course_ids==9)&(c.abs().amax(-1)<=1.e-4)
        start=moving & ~self.moving;stop=holding & ~self.holding
        self.heading[start]=angle[start]
        for name in ('heading_sum','rate_sum','heading_peak','frames','filtered_rate'):getattr(self,name)[start]=0
        self.anchor[stop]=d.root_pos_w[stop,:2]
        self.foot_anchor[stop]=d.body_link_pos_w[stop][:,s.body_ids,:2].mean(1)
        self.stop_heading[stop]=angle[stop]
        # Brake only forward in the locked heading; no lateral/backward allowance.
        v=d.root_lin_vel_w[:,:2]
        along=v[:,0]*angle.cos()+v[:,1]*angle.sin()
        allowance=(along.clamp_min(0).square()/(2*self.cfg['braking_m_s2'])).clamp(max=self.cfg['braking_max_m'])
        self.braking[stop]=allowance[stop]
        self.seen_move |= start;self.seen_hold |= stop
        self.previous_yaw.copy_(angle);self.pre_command.copy_(c)
        self.moving.copy_(moving);self.holding.copy_(holding)
        self.valid.copy_(self.selected)

    def add(self,key,value,maximum=False):
        s=self.state
        for j,cohort in enumerate((s.teacher,~s.teacher)):
            for i,ci in enumerate((8,9)):
                x=value[cohort & (s.course_ids==ci)].double()
                if maximum:
                    if x.numel():self.totals[key][j,i]=torch.maximum(self.totals[key][j,i],x.max())
                else:self.totals[key][j,i]+=x.sum()

    def advance(self,g):
        s,c=self.state,self.cfg
        if self.tick==s.env.common_step_counter:return
        self.tick=s.env.common_step_counter;d=s.env.scene['robot'].data;dt=s.env.step_dt
        angle=yaw(d.root_quat_w);rate=wrap(angle-self.previous_yaw)/dt
        finite=torch.isfinite(d.root_pos_w).all(-1)&torch.isfinite(d.root_quat_w).all(-1)
        finite &= torch.isfinite(d.root_lin_vel_w).all(-1)&torch.isfinite(d.root_ang_vel_b).all(-1)
        for k in ('position','velocity','normal','force'):finite &= torch.isfinite(g[k]).all(dim=(-1,-2))
        # Almost vertical poses do not define a useful horizontal heading.
        finite &= (-d.projected_gravity_b[:,2]>.6)
        measured=self.valid & (self.begin_tick==s.env.common_step_counter-1)
        s.failed |= measured & ~finite
        active=measured & finite & ~s.failed
        moving=measured & self.moving;holding=measured & self.holding
        error=wrap(angle-self.heading)
        self.filtered_rate.lerp_(torch.nan_to_num(rate),1-math.exp(-dt/c['rate_filter_s']))
        self.heading_sum+=torch.nan_to_num(error.square())*moving
        self.rate_sum+=self.filtered_rate.square()*moving
        self.heading_peak=torch.maximum(self.heading_peak,torch.where(moving,torch.nan_to_num(error.abs(),nan=math.pi),0.))
        self.frames+=moving
        self.heading_ok.copy_(self.seen_move & (self.frames>0) & finite
            & (self.heading_sum/self.frames.clamp_min(1)<=c['heading_scale_rad']**2)
            & (self.rate_sum/self.frames.clamp_min(1)<=c['rate_scale_rad_s']**2)
            & (self.heading_peak<=c['heading_peak_rad']))
        delta=torch.stack((d.root_pos_w[:,:2]-self.anchor,g['position'][...,:2].mean(1)-self.foot_anchor),1)
        co,si=self.stop_heading.cos()[:,None],self.stop_heading.sin()[:,None]
        dx=delta[...,0]*co+delta[...,1]*si;dy=-delta[...,0]*si+delta[...,1]*co
        excess=torch.stack(((dx-self.braking[:,None]-c['position_m']).clamp_min(0).amax(-1),
                            (-dx-c['position_m']).clamp_min(0).amax(-1),
                            (dy.abs()-c['position_m']).clamp_min(0).amax(-1)),-1)
        drift=delta.norm(dim=-1).amax(-1);stop_error=wrap(angle-self.stop_heading)
        region=(excess.amax(-1)<=1.e-6)&(stop_error.abs()<=c['heading_scale_rad'])
        position=(excess/c['position_scale_m']).square().sum(-1)+(stop_error/c['heading_scale_rad']).square()
        speed=d.root_lin_vel_w[:,:2].norm(dim=-1)
        front=g['supported'][:,:2].any(-1);rear=g['supported'][:,2:].any(-1)
        self.hold_ok.copy_(active & holding & region & front & rear
            & (speed<=c['stop_speed_m_s']) & (rate.abs()<=c['stop_yaw_rad_s']))
        # Platform descent: all wheels must clear the edge and settle on the
        # lower floor before the transition cost releases. Initial full support
        # on the top slab cannot prematurely complete this latch.
        p=g['position']-s.origins[s.tile,None];edge=s.ends[s.tile,None]
        extent=s.env.cfg.stair_tail.wheel_radius+s.env.cfg.stair_tail.wheel_half_length*g['axis'][...,0].abs()
        clear=(p[...,0]-extent>edge).all(-1)
        floor=(g['height'].abs()<s.cfg['hind_clearance']['support_tolerance_m']).all(-1)
        landed=clear & floor & g['supported'].all(-1)
        from .timing import advance_steps, required_steps
        self.landing_steps=advance_steps(self.landing_steps,landed)
        self.landing_age=self.landing_steps.to(self.landing_age.dtype)*dt
        self.landed = (self.landed & clear & floor) | (self.landing_steps>=required_steps(c['landing_s'],dt))
        near=(p[...,0]+extent>=edge-c['transition_margin_m']).any(-1)
        low=(s.course_ids==9)&(s.mode==0)&~s.objectives.high_platform
        self.transition.copy_(active & low & near & ~self.landed)
        v=d.root_lin_vel_w
        lateral=-v[:,0]*self.heading.sin()+v[:,1]*self.heading.cos()
        tangent=g['velocity']-(g['velocity']*g['normal']).sum(-1,keepdim=True)*g['normal']
        slip=(tangent.norm(dim=-1)/c['slip_scale_m_s']).square()*g['supported']
        # Allow rotation that reduces the heading error, with a bounded budget.
        corrective=torch.where(error*rate<0,(error.abs()/c['rate_filter_s']).clamp(max=c['heading_peak_rad']/c['rate_filter_s']),0.)
        excess_rate=(rate.abs()-corrective-c['rate_scale_rad_s']).clamp_min(0)/c['rate_scale_rad_s']
        transition_cost=(excess_rate.square().clamp(max=4.)
            +(lateral/c['lateral_scale_m_s']).square().clamp(max=4.)
            +slip.clamp(max=4.).mean(-1)
            +(d.root_ang_vel_b[:,:2].norm(dim=-1)/c['posture_scale_rad_s']).square().clamp(max=4.))/4
        self.linear.copy_(torch.exp(-speed.square()/c['stop_speed_m_s']**2-position.clamp(max=16)))
        self.angular.copy_(torch.where(holding,torch.exp(-rate.square()/c['stop_yaw_rad_s']**2-position.clamp(max=16)),
            torch.exp(-error.square()/c['heading_scale_rad']**2-self.filtered_rate.square()/c['rate_scale_rad_s']**2)))
        self.cost.copy_(torch.nan_to_num(c['position_weight']*position.clamp(max=4)*holding
            +c['transition_weight']*transition_cost*self.transition,nan=16.,posinf=16.))
        self.valid.copy_(active)
        self.add('moving_frames',moving);self.add('heading_sq_rad',torch.nan_to_num(error.square())*moving)
        self.add('heading_peak_rad',torch.where(moving,torch.nan_to_num(error.abs(),nan=math.pi),0.),True)
        self.add('filtered_rate_sq',self.filtered_rate.square()*moving)
        self.add('transition_frames',self.transition);self.add('transition_cost',torch.nan_to_num(transition_cost)*self.transition)
        self.add('hold_frames',holding);self.add('hold_ready_frames',self.hold_ok)
        self.add('hold_outside_frames',holding & ~region);self.add('hold_speed_sq',torch.nan_to_num(speed.square())*holding)
        self.add('hold_yaw_rate_sq',torch.nan_to_num(rate.square())*holding)
        self.add('hold_heading_sq',torch.nan_to_num(stop_error.square())*holding)
        self.add('hold_drift_max_m',torch.where(holding,torch.nan_to_num(drift),0.),True)
        self.add('invalid_frames',measured & ~finite)

    def tracking(self,axes):
        selected=self.selected & (self.state.mode==0)
        out=axes.clone();out[:,1]=torch.where(selected,self.heading_ok,out[:,1])
        if self.cfg.get('idle_terminal_hold', False):
            idle=(self.state.course_ids==9)&(self.state.mode==8)
            out=torch.where(idle[:,None],self.hold_ok[:,None],out)
        return out

    def stable(self,stable):
        return stable & (~self.holding | self.hold_ok)

    def reward_terms(self,linear,angular,stopped):
        hold=self.valid & self.holding;move=self.valid & self.moving
        return (torch.where(hold,torch.nan_to_num(self.linear),linear),
            torch.where(hold|move,torch.nan_to_num(self.angular),angular),
            torch.where(hold,self.hold_ok,stopped),torch.where(self.valid,self.cost,0.))

    def snapshot(self):return {k:v.cpu().clone() for k,v in self.totals.items()}

    def checked_snapshot(self,saved):
        if not isinstance(saved,dict) or set(saved)!=set(self.totals):raise ValueError('Invalid motion_quality snapshot keys')
        for k,v in saved.items():
            t=self.totals[k]
            if not isinstance(v,torch.Tensor) or v.shape!=t.shape or v.dtype!=t.dtype or not torch.isfinite(v).all() or (v<0).any():
                raise ValueError('Invalid motion_quality snapshot '+k)
        return {k:v.to(self.state.env.device) for k,v in saved.items()}

    def diagnostics(self):
        return {f'{cohort}/{course}/{k}':float(v[j,i]) for k,v in self.totals.items()
                for j,cohort in enumerate(('teacher','student')) for i,course in enumerate(('straight','platform'))}
