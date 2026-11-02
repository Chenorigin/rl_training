"""Per-front-wheel participation during actual descent, without a gait clock.

Negative costs only. Normal transfer has a geometry/speed-aware grace period;
brief touches, command pauses and another wheel's support cannot clear absence.
"""
import math
import torch


def validate(c):
    keys = {'enabled','command_min_m_s','min_drop_m','approach_m','exit_margin_m',
            'contact_confirm_s','swing_grace_s','swing_grace_max_s','tread_time_fraction',
            'air_ramp_s','air_weight','stall_grace_s','stall_ramp_s','stall_weight','progress_epsilon_m'}
    if not isinstance(c, dict) or set(c) != keys or type(c['enabled']) is not bool:
        raise ValueError('Invalid front descent configuration')
    for k in keys - {'enabled'}:
        if type(c[k]) not in (float,int) or not math.isfinite(c[k]) or c[k] <= 0:
            raise ValueError('Invalid front descent value: '+k)
    if c['swing_grace_s'] > c['swing_grace_max_s'] or c['tread_time_fraction'] > 1:
        raise ValueError('Invalid front transfer grace')


def falling_edges(rectangles, min_drop):
    r = rectangles
    valid = torch.isfinite(r).all(-1) & (r[...,1] > r[...,0])
    adjacent = torch.isclose(r[:,:,None,0], r[:,None,:,1], atol=1.e-5, rtol=0)
    adjacent &= valid[:,None,:] & (r[:,:,None,2] < r[:,None,:,3]) & (r[:,:,None,3] > r[:,None,:,2])
    before = torch.where(adjacent, r[:,None,:,4], -torch.inf).amax(-1)
    return valid & (before-r[...,4] >= min_drop)


class FrontDescent:
    def __init__(self, state):
        self.state, self.cfg = state, state.cfg['front_descent']
        r = state.surface.rectangles
        # The atlas stores raised boxes; the last drop lands on its actual z=0
        # base mesh, which has no rectangle. Include that final landing edge.
        positive = torch.isfinite(r).all(-1) & (r[...,4] > 0)
        index = torch.where(positive,r[...,1],-torch.inf).argmax(-1)
        last = r[torch.arange(len(r),device=r.device),index].clone()
        last[:,0] = last[:,1]
        last[:,1] += state.cfg['geometry']['goal_landing']
        last[:,4] = torch.where(positive.any(-1),0.,-torch.inf)
        self.rectangles = torch.cat((r,last[:,None]),1)
        r = self.rectangles
        self.valid = falling_edges(r, self.cfg['min_drop_m'])
        self.valid &= r[...,0] >= state.cfg['geometry']['route_start']-1.e-5
        self.first = torch.where(self.valid,r[...,0],torch.inf).amin(-1)
        self.last = torch.where(self.valid,r[...,0],-torch.inf).amax(-1)
        n, device = state.env.num_envs, state.env.device
        self.air_age = torch.zeros(n,2,device=device)
        self.contact_age = torch.zeros_like(self.air_age)
        self.stall_age = torch.zeros(n,device=device)
        self.best_x = torch.full_like(self.stall_age,-torch.inf)
        self.tick = -1
        self.values = {k:torch.zeros(n,device=device) for k in ('air','stall')}
        self.totals = {k:torch.zeros(2,2,dtype=torch.float64,device=device) for k in
                       ('active_s','supported_s','overdue_s','air_cost','stall_cost','stall_s','max_air_s')}

    def reset(self, ids):
        self.air_age[ids] = self.contact_age[ids] = 0.
        self.stall_age[ids] = 0.; self.best_x[ids] = -torch.inf
        for value in self.values.values(): value[ids] = 0.

    def advance(self, g):
        s, c, dt = self.state, self.cfg, self.state.env.step_dt
        if self.tick == s.env.common_step_counter: return
        self.tick = s.env.common_step_counter
        p, axis = g['position'][:,:2], g['axis'][:,:2]
        finite = torch.isfinite(torch.cat((p,axis,g['force'][:,:2],g['normal'][:,:2],g['velocity'][:,:2]),-1)).all(-1)
        finite &= torch.isfinite(g['height'][:,:2]) & torch.isfinite(g['force_n'][:,:2])
        x = p[...,0]-s.origins[s.tile,0,None]
        first,last = self.first[s.tile],self.last[s.tile]
        descending = ((s.course_ids == 1) | (s.course_ids == 2)) & torch.isfinite(first)
        zone = descending & (x.amax(-1) >= first-c['approach_m']) & (x.amin(-1) <= last+c['exit_margin_m'])
        s.failed |= descending & ~finite.all(-1)
        valid = zone & ~s.failed & ~s.reached
        moving = valid & (g['command'][:,0] > c['command_min_m_s'])
        tail = s.env.cfg.stair_tail
        bottom = p[...,2]-tail.wheel_radius-tail.wheel_half_length*axis[...,2].abs()
        supported = (g['supported'][:,:2] & finite
                     & ((bottom-g['height'][:,:2]).abs() <= s.cfg['hind_clearance']['support_tolerance_m'])
                     & (g['force_n'][:,:2] > 0))
        # Count genuine continuous contact even during a requested pause.
        self.contact_age = torch.where(valid[:,None] & supported, self.contact_age+dt, 0.)
        confirmed = self.contact_age+1.e-7 >= max(c['contact_confirm_s'], 2*dt)
        self.air_age = torch.where(valid[:,None], self.air_age, 0.)
        self.air_age = torch.where(confirmed, 0., self.air_age+moving[:,None]*dt)
        r = self.rectangles[s.tile]
        # Closest real descending edge per wheel, not a course-wide maximum
        # (the long final landing must not hide persistent single-leg absence).
        distance = (x[...,None]-r[:,None,:,0]).abs()
        index = torch.where(self.valid[s.tile,None],distance,torch.inf).argmin(-1)
        depth = (r[...,1]-r[...,0]).gather(1,index)
        speed = g['command'][:,0,None].abs().clamp_min(c['command_min_m_s'])
        grace = (c['swing_grace_s']+c['tread_time_fraction']*depth/speed).clamp(max=c['swing_grace_max_s'])
        overdue = (self.air_age-grace).clamp_min(0.)
        cost = .5*c['air_weight']*torch.where(moving[:,None], (overdue/c['air_ramp_s']).clamp(max=1.).square(),0.)
        # Root progress cannot be refreshed by waving an unsupported front leg.
        root_x = s.env.scene['robot'].data.root_pos_w[:,0]
        fresh = ~torch.isfinite(self.best_x)
        progress = root_x >= self.best_x+c['progress_epsilon_m']
        self.stall_age = torch.where(~valid | fresh | progress, 0., self.stall_age+moving*dt)
        self.best_x = torch.where(fresh | progress, root_x, self.best_x)
        stall = c['stall_weight']*torch.where(moving, ((self.stall_age-c['stall_grace_s'])/c['stall_ramp_s']).clamp(0.,1.).square(),0.)
        self.values['air'] = -cost.sum(-1)
        self.values['stall'] = -stall
        cohort = (~s.teacher).long()
        updates = dict(active_s=moving[:,None].expand(-1,2)*dt,
                       supported_s=(moving[:,None] & supported)*dt,
                       overdue_s=(moving[:,None] & (overdue>0))*dt,air_cost=cost*dt,
                       stall_cost=stall[:,None].expand(-1,2)*dt/2,
                       stall_s=(moving & (self.stall_age>c['stall_grace_s']))[:,None].expand(-1,2)*dt)
        for name,value in updates.items(): self.totals[name].index_add_(0,cohort,value.double())
        for i in (0,1):
            selected = moving & (cohort == i)
            if selected.any(): self.totals['max_air_s'][i] = torch.maximum(self.totals['max_air_s'][i], self.air_age[selected].amax(0).double())

    def diagnostics(self):
        return {f'{cohort}/{leg}/{name}':float(value[i,j])
                for name,value in self.totals.items() for i,cohort in enumerate(('teacher','student'))
                for j,leg in enumerate(('left','right'))}

    def snapshot(self):
        return {k:v.cpu().clone() for k,v in self.totals.items()}

    def checked_snapshot(self, saved):
        if not isinstance(saved,dict) or saved.keys() != self.totals.keys():
            raise ValueError('Front descent ledger keys changed')
        for k,v in saved.items():
            if not isinstance(v,torch.Tensor) or v.shape != self.totals[k].shape or v.dtype != self.totals[k].dtype or not torch.isfinite(v).all() or (v<0).any():
                raise ValueError('Invalid front descent ledger: '+k)
        return {k:v.to(self.state.env.device) for k,v in saved.items()}


def front_descent_air(env):
    return env.cts_all_terrain.front_descent.values['air']


def front_descent_stall(env):
    return env.cts_all_terrain.front_descent.values['stall']
