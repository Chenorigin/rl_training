"""Position-preserving stops during the existing ascending pause lesson.

The anchor is fixed after the last moving physics step. Measurements and rewards
describe the executed zero-command step, including the step that resumes motion.
No actor inputs, deployment controller, initial hold or descending task change.
"""
import math
import torch


def validate(cfg):
    keys = {'enabled', 'backward_m', 'lateral_m', 'forward_base_m',
            'braking_m_s2', 'forward_max_m', 'tread_fraction', 'yaw_rad',
            'position_scale_m', 'speed_scale_m_s', 'yaw_rate_scale_rad_s',
            'roll_pitch_rate_rad_s', 'slip_tolerance_m_s', 'slip_scale_m_s',
            'position_weight', 'slip_weight', 'rear_slip_multiplier', 'support_weight', 'posture_weight'}
    if not isinstance(cfg, dict) or set(cfg)-{'retreat_weight'} != keys or type(cfg['enabled']) is not bool:
        raise ValueError('Invalid stair_stop configuration')
    for key, value in cfg.items():
        if key == 'enabled':
            continue
        if (type(value) not in (int, float) or not math.isfinite(value)
                or value < 0 or (not key.endswith('_weight') and value == 0)):
            raise ValueError('Invalid stair_stop value: '+key)
    if not 0 < cfg['tread_fraction'] <= 1 or cfg['forward_base_m'] > cfg['forward_max_m']:
        raise ValueError('Invalid stair_stop braking bounds')


def yaw(q):
    return torch.atan2(2*(q[:, 0]*q[:, 3]+q[:, 1]*q[:, 2]),
                       1-2*(q[:, 2].square()+q[:, 3].square()))


class StairStop:
    def __init__(self, recovery):
        self.recovery, self.state = recovery, recovery.state
        s = self.state
        self.cfg = s.cfg['stair_stop']
        n, device = s.env.num_envs, s.env.device
        self.anchor = torch.zeros(n, 2, device=device)
        self.foot_anchor = self.anchor.clone()
        self.previous_feet = self.anchor.clone()
        self.anchor_yaw = torch.zeros(n, device=device)
        self.forward = self.anchor_yaw.clone()
        self.max_back = self.anchor_yaw.clone()
        self.depth = torch.tensor([r.get('depth', 0.) for r in s.atlas.records], device=device)
        self.active = torch.zeros(n, dtype=torch.bool, device=device)
        self.ready = self.active.clone()
        self.region = self.active.clone()
        self.seen = self.active.clone()
        self.resumed = self.active.clone()
        self.linear = self.anchor_yaw.clone()
        self.angular = self.anchor_yaw.clone()
        self.cost = self.anchor_yaw.clone()
        self.tick = -1
        names = ('entries', 'resumes', 'resume_time_s', 'pause_frames', 'stable_frames',
                 'outside_frames', 'invalid_frames', 'slip_distance_m', 'support_wheel_s', 'timeouts',
                 'completed_pauses', 'completed_resumed', 'success_after_resume',
                 'max_back_m_sum', 'max_back_m_max', 'continuous_completed',
                 'continuous_successes', 'before_pause_failures')
        self.totals = {k: torch.zeros(2, 3, dtype=torch.float64, device=device) for k in names}

    def reset(self, ids):
        for value in (self.anchor, self.foot_anchor, self.previous_feet, self.anchor_yaw, self.forward,
                      self.max_back, self.active, self.ready, self.region, self.seen,
                      self.resumed, self.linear, self.angular, self.cost):
            value[ids] = 0

    def add(self, name, values):
        s = self.state
        mask = self.recovery.active
        cell = (~s.teacher[mask]).long()*3+s.course_ids[mask]
        self.totals[name].view(-1).index_add_(0, cell, values[mask].double())

    def advance(self, g, entering, paused, stable):
        s, c = self.state, self.cfg
        if self.tick == s.env.common_step_counter:
            return self.ready
        self.tick = s.env.common_step_counter
        d = s.env.scene['robot'].data
        xy = d.root_pos_w[:, :2]
        feet = g['position'][..., :2].mean(1)
        heading = yaw(d.root_quat_w)
        # Fixed world +X stair route; body pitch must not hide world XY drift.
        velocity = d.root_lin_vel_w[:, :2]
        angular = d.root_ang_vel_b
        self.anchor[entering] = xy[entering]
        self.foot_anchor[entering] = feet[entering]
        self.previous_feet[entering] = feet[entering]
        footprint_vx = (feet[:, 0]-self.previous_feet[:, 0])/s.env.step_dt
        self.previous_feet.copy_(feet)
        self.anchor_yaw[entering] = heading[entering]
        braking = c['forward_base_m']+velocity[:, 0].clamp_min(0.).square()/(2*c['braking_m_s2'])
        allowance = torch.minimum(braking.clamp(max=c['forward_max_m']),
                                  self.depth[s.tile]*c['tread_fraction'])
        self.forward[entering] = allowance[entering]
        self.seen |= entering
        self.add('entries', entering)
        # Entering executed forward, even though phase has just become PAUSE.
        self.active.copy_(paused & (g['command'].abs().amax(-1) <= 1.e-4) & self.seen)
        # Failure is decided before recovery. Still measure the final physical
        # zero-command step, even if a low-body/boundary failure excludes reward.
        measured = (self.recovery.active & (self.recovery.phase == 2) & ~entering
                    & (g['command'].abs().amax(-1) <= 1.e-4) & self.seen)
        finite = (torch.isfinite(d.root_pos_w).all(-1) & torch.isfinite(d.root_quat_w).all(-1)
                  & torch.isfinite(velocity).all(-1) & torch.isfinite(angular).all(-1))
        for key in ('position', 'velocity', 'normal', 'force'):
            finite &= torch.isfinite(g[key]).all(dim=(-1, -2))
        s.failed |= (entering | measured) & ~finite
        delta = torch.stack((xy-self.anchor, feet-self.foot_anchor), dim=1)
        backward = torch.nan_to_num((-delta[..., 0]).clamp_min(0.).amax(-1))
        self.max_back = torch.maximum(self.max_back, torch.where(measured & finite, backward, 0.))
        excess = torch.stack(((delta[..., 0]-self.forward[:, None]).clamp_min(0.).amax(-1),
                              (-delta[..., 0]-c['backward_m']).clamp_min(0.).amax(-1),
                              (delta[..., 1].abs()-c['lateral_m']).clamp_min(0.).amax(-1)), -1)
        heading_error = torch.atan2(torch.sin(heading-self.anchor_yaw), torch.cos(heading-self.anchor_yaw)).abs()
        position_cost = (excess/c['position_scale_m']).square().sum(-1)
        position_cost += ((heading_error-c['yaw_rad']).clamp_min(0.)/c['yaw_rad']).square()
        position_cost = torch.nan_to_num(position_cost, nan=16., posinf=16.).clamp(max=16.)
        self.region.copy_((excess.amax(-1) <= 1.e-6) & (heading_error <= c['yaw_rad']) & finite)
        supported = g['supported'] & finite[:, None]
        front, rear = supported[:, :2].any(-1), supported[:, 2:].any(-1)
        # Contact material-point velocity, NOT wheel-center velocity: rolling
        # retreat is detected by the anchors, and sliding by this measurement.
        tangent = g['velocity']-(g['velocity']*g['normal']).sum(-1, keepdim=True)*g['normal']
        slip = torch.nan_to_num(tangent.norm(dim=-1), nan=1., posinf=1.)*supported
        loaded_slip = (slip-c['slip_tolerance_m_s']).clamp_min(0.)/c['slip_scale_m_s']
        weights = slip.new_tensor([1., 1., c['rear_slip_multiplier'], c['rear_slip_multiplier']])
        slip_cost = (loaded_slip.square().clamp(max=16.)*weights).sum(-1)/weights.sum()
        support_cost = 1-supported.float().mean(-1)
        support_cost += .5*((~front).float()+(~rear).float())
        posture = (angular[:, :2].norm(dim=-1)/c['roll_pitch_rate_rad_s']).square().clamp(max=16.)
        self.cost.copy_(torch.nan_to_num(c['position_weight']*position_cost+c['slip_weight']*slip_cost
                        +c['support_weight']*support_cost+c['posture_weight']*posture, nan=16., posinf=16.))
        # Unlike contact slip, this also sees rolling retreat. Penalize it from
        # the first backward step, before exhausting the anchor tolerance.
        backward_speed = torch.maximum((-velocity[:, 0]).clamp_min(0.),
                                       (-footprint_vx).clamp_min(0.))
        retreat = torch.nan_to_num((backward_speed/c['speed_scale_m_s']).square(), nan=4., posinf=4.).clamp(max=4.)
        self.cost += c.get('retreat_weight', 0.)*retreat
        spatial_credit = torch.exp(-position_cost)
        self.linear.copy_(torch.nan_to_num(torch.exp(-velocity.square().sum(-1)/c['speed_scale_m_s']**2)*spatial_credit))
        self.angular.copy_(torch.nan_to_num(torch.exp(-angular[:, 2].square()/c['yaw_rate_scale_rad_s']**2)*spatial_credit))
        cc = s.cfg['commands']
        self.ready.copy_(self.active & stable & finite & self.region & front & rear
                         & (velocity.norm(dim=-1) <= cc['stop_speed_m_s'])
                         & (slip.amax(-1) <= c['slip_tolerance_m_s'])
                         & (angular[:, :2].norm(dim=-1) <= c['roll_pitch_rate_rad_s']))
        self.add('pause_frames', measured)
        self.add('stable_frames', self.ready)
        self.add('outside_frames', measured & finite & ~self.region)
        self.add('invalid_frames', measured & ~finite)
        self.add('slip_distance_m', slip.sum(-1)*measured*s.env.step_dt)
        self.add('support_wheel_s', supported.sum(-1)*measured*s.env.step_dt)
        return self.ready

    def outcome(self, restart, timeout):
        self.resumed |= restart
        self.add('resumes', restart)
        self.add('resume_time_s', self.recovery.pause_age*restart)
        self.add('timeouts', timeout)

    def reward_terms(self, linear, angular, stopped):
        # Replace existing stop credit, including outside-region stationary
        # counterexamples; do not stack a second generic stopping bonus.
        return (torch.where(self.active, self.linear, linear),
                torch.where(self.active, self.angular, angular),
                torch.where(self.active, self.ready, stopped),
                torch.where(self.active, self.cost, 0.))

    def record(self, ids):
        s, r = self.state, self.recovery
        ids = ids[r.active[ids]]
        for i in ids.tolist():
            cell = int(not s.teacher[i]), int(s.course_ids[i])
            success = bool(s.success[i] and not s.failed[i])
            if self.seen[i]:
                for k, value in (('completed_pauses', 1), ('completed_resumed', self.resumed[i]),
                                 ('success_after_resume', self.resumed[i] & success),
                                 ('max_back_m_sum', self.max_back[i])):
                    self.totals[k][cell] += value
                self.totals['max_back_m_max'][cell] = torch.maximum(self.totals['max_back_m_max'][cell], self.max_back[i])
            elif r.phase[i] == 0:
                self.totals['continuous_completed'][cell] += 1
                self.totals['continuous_successes'][cell] += success
            else:
                self.totals['before_pause_failures'][cell] += not success

    def diagnostics(self):
        return {f'{cohort}/{course}/{key}': float(value[j, i])
                for key, value in self.totals.items()
                for j, cohort in enumerate(('teacher', 'student'))
                for i, course in ((0, 'up'), (2, 'roundtrip'))}

    def snapshot(self):
        return {key: value.cpu().clone() for key, value in self.totals.items()}

    def checked_snapshot(self, saved):
        if not isinstance(saved, dict) or set(saved) != set(self.totals):
            raise ValueError('Invalid stair_stop snapshot keys')
        for key, value in saved.items():
            target = self.totals[key]
            if (not isinstance(value, torch.Tensor) or value.shape != target.shape
                    or value.dtype != target.dtype or not torch.isfinite(value).all() or (value < 0).any()):
                raise ValueError('Invalid stair_stop snapshot: '+key)
        return {key: value.to(self.state.env.device) for key, value in saved.items()}
