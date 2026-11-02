"""Training-only cold starts and physically reached stop/restart states.

No teleport onto a stair, no deployment heading latch change. Reward headings
are route-relative; tracking uses the command actually consumed by the actor.
"""
import torch


def yaw_of(q):
    return torch.atan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]), 1-2*(q[:,2].square()+q[:,3].square()))


class StairRecovery:
    def __init__(self, state):
        self.state, self.cfg = state, state.cfg['stair_recovery']
        n, device = state.env.num_envs, state.env.device
        self.active = (state.course_ids == 0) | (state.course_ids == 2)
        self.near = torch.zeros(n,dtype=torch.bool,device=device)
        self.phase = torch.zeros(n,dtype=torch.long,device=device)  # 0 continuous, 1 waiting, 2 pause, 3 restarted
        self.trigger_x = torch.zeros(n,device=device)
        self.pause_age = torch.zeros(n,device=device)
        self.settled_age = torch.zeros(n,device=device)
        self.settled_steps = torch.zeros(n,dtype=torch.long,device=device)
        self.pause_steps = self.settled_steps.clone()
        self.stall_age = torch.zeros(n,device=device)
        self.progress_x = torch.zeros(n,device=device)
        self.support_x = torch.full((n,2),-torch.inf,device=device)
        self.perturb_pending = self.near.clone()
        self.effective_command = torch.zeros(n,3,device=device)
        self.command_valid = self.near.clone()
        self.events = {k:torch.zeros(n,dtype=torch.long,device=device) for k in
            ('cold_start','pause','restart','pause_failure','stall_failure','perturb')}
        from .stair_stop import StairStop
        self.hold = StairStop(self) if state.cfg.get('stair_stop', {}).get('enabled') else None

    def reset(self, ids, position):
        s, c = self.state, self.cfg
        self.active[ids] = (s.course_ids[ids] == 0) | (s.course_ids[ids] == 2)
        if getattr(s,'stair_hold',None) is not None:self.active[ids] &= ~s.stair_hold.active[ids]
        active = self.active[ids]
        draw = lambda: torch.as_tensor(s.rng.random(len(ids)),device=ids.device)
        self.near[ids] = active & (draw() < c['near_start_probability'])
        gap = torch.as_tensor(s.rng.uniform(*c['start_gap_m'],len(ids)),device=ids.device,dtype=position.dtype)
        near = self.near[ids]
        # The conservative complete front envelope remains BEFORE the first riser.
        position[near,0] = s.origins[s.tile[ids[near]],0]+s.cfg['geometry']['route_start']-c['front_extent_m']-gap[near]
        self.phase[ids] = (active & (draw() < c['pause_probability'])).long()
        steps = s.rng.integers(c['pause_step'][0],c['pause_step'][1]+1,len(ids))
        depths = position.new_tensor([s.atlas.records[int(t)].get('depth',0.) for t in s.tile[ids]])
        self.trigger_x[ids] = s.origins[s.tile[ids],0]+s.cfg['geometry']['route_start']+depths*position.new_tensor(steps)
        self.pause_age[ids] = self.settled_age[ids] = self.stall_age[ids] = 0.
        self.settled_steps[ids] = self.pause_steps[ids] = 0
        self.progress_x[ids] = -torch.inf
        self.support_x[ids] = -torch.inf
        self.command_valid[ids] = False
        self.perturb_pending[ids] = active & (draw() < c['perturb_probability'])
        speed = position.new_tensor(s.rng.uniform(*c['speed_m_s'],len(ids)))
        s.command_value[ids[active],0] = speed[active]
        self.events['cold_start'][ids] += near.long()
        if self.hold is not None:self.hold.reset(ids)

    def command(self, command):
        return torch.where((self.phase == 2)[:,None],torch.zeros_like(command),command)

    def begin_step(self, obs):
        s, env = self.state, self.state.env
        # Copy the already-computed observation: never update either stateful
        # heading controller a second time while measuring reward.
        cmd = torch.where(s.teacher[:,None],obs['teacher'][:,6:9],obs['policy'][:,6:9])
        self.effective_command.copy_(cmd)
        self.command_valid.fill_(True)
        g = s.observe()
        selected = self.perturb_pending & (g['supported'][:,:2].any(-1))
        selected &= (g['position'][:,:2,0].amin(-1) > self.trigger_x)
        selected &= s.raw_command()[:,0] > .1
        ids = selected.nonzero().flatten()
        if ids.numel():
            robot = env.scene['robot']
            velocity = robot.data.root_vel_w[ids].clone()
            kicks = velocity.new_tensor(s.rng.uniform(-1,1,(len(ids),2)))
            velocity[:,1] += kicks[:,0]*self.cfg['perturb_lateral_m_s']
            velocity[:,5] += kicks[:,1]*self.cfg['perturb_yaw_rate_rad_s']
            robot.write_root_velocity_to_sim(velocity,env_ids=ids)
            self.perturb_pending[ids] = False
            self.events['perturb'][ids] += 1

    def advance(self, g, stable):
        s, c, dt = self.state,self.cfg,self.state.env.step_dt
        age = s.env.episode_length_buf*s.env.step_dt
        from .wall_stability import initial_hold
        eligible = self.active & ~s.reached & ~s.failed & (age >= initial_hold(s))
        front_on = (g['position'][:,:2,0].amin(-1) > self.trigger_x) & g['supported'][:,:2].any(-1)
        entering = eligible & (self.phase == 1) & front_on
        self.phase[entering] = 2
        self.events['pause'] += entering.long()
        # Only account a completed zero-command physics step, not the step
        # that first entered the pause phase.
        paused = eligible & (self.phase == 2) & ~entering
        if self.hold is not None:
            stable = self.hold.advance(g, entering, paused, stable)
        from .timing import advance_steps, required_steps
        self.pause_steps += paused.long()
        self.settled_steps = advance_steps(self.settled_steps, paused & stable)
        self.pause_age = self.pause_steps.to(self.pause_age.dtype)*dt
        self.settled_age = self.settled_steps.to(self.settled_age.dtype)*dt
        restart = paused & (self.settled_steps >= required_steps(c['settle_s'], dt))
        timeout = paused & ~restart & (self.pause_steps >= required_steps(c['pause_timeout_s'], dt))
        self.phase[restart] = 3
        self.events['restart'] += restart.long()
        self.events['pause_failure'] += timeout.long()
        if self.hold is not None:self.hold.outcome(restart, timeout)
        s.failed |= timeout
        moving = eligible & (g['command'][:,0] > .1) & (self.phase != 2)
        # Measure rear-foot advance, not foreleg flailing/body pitch.
        rear_x = g['position'][:,2:,0]
        valid = g['supported'][:,2:] & torch.isfinite(rear_x)
        if 'velocity' in g:
            valid &= torch.isfinite(g['velocity'][:,2:]).all(-1)
        self.support_x = torch.where(valid,torch.maximum(self.support_x,rear_x),self.support_x)
        rear = self.support_x.amin(-1)
        progress = torch.isfinite(rear) & (rear > self.progress_x+c['progress_epsilon_m'])
        self.progress_x = torch.where(progress,rear,self.progress_x)
        self.stall_age = torch.where(moving & ~progress,self.stall_age+dt,0.)
        stalled = moving & (self.stall_age >= c['stall_s']) & ~s.failed
        self.events['stall_failure'] += stalled.long()
        s.failed |= stalled

    def cost(self):
        s, c = self.state,self.cfg
        d = s.env.scene['robot'].data
        heading = yaw_of(d.root_quat_w)  # stair route is world +X, including after a stop
        lateral = d.root_pos_w[:,1]-s.origins[s.tile,1]
        return self.active*(c['heading_weight']*(heading/c['heading_scale_rad']).square().clamp(max=4)
            +c['lateral_weight']*(lateral/c['lateral_scale_m']).square().clamp(max=4)
            +c['stall_weight']*(self.stall_age/c['stall_s']).square())

    def diagnostics(self):
        return {**{k+'_total':v.sum() for k,v in self.events.items()},
            'paused_fraction':(self.phase == 2).float().mean(),
            'stair_command_min_m_s':self.state.command_value[self.active,0].amin() if self.active.any() else 0.,
            'stair_command_max_m_s':self.state.command_value[self.active,0].amax() if self.active.any() else 0.}
