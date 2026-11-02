"""Flat, two-wall zero-command practice with fixed reset anchors and separate ledgers.

Only measured motion earns credit. No brake, action clamp or position controller
is installed. A position excursion cannot be erased by stopping somewhere else.
"""
from copy import deepcopy
import math
import numpy as np
import torch
from .all_terrain_curriculum import CourseCurriculum
from .stair_stop import yaw
from .timing import advance_steps


def validate(c, terrain):
    keys = {'enabled', 'motion_environment_fraction', 'settle_grace_s', 'timeout_s',
            'hold_s_range', 'position_m', 'position_deadband_m', 'speed_m_s',
            'yaw_rad', 'yaw_rate_rad_s', 'roll_pitch_rate_rad_s', 'slip_m_s',
            'position_scale_m', 'speed_scale_m_s', 'slip_scale_m_s',
            'position_weight', 'motion_weight', 'slip_weight', 'support_weight',
            'posture_weight', 'success_reward'}
    if isinstance(c, dict) and 'handover' in c:
        from .c_handover import validate
        validate(c['handover'])
        keys.add('handover')
    if not isinstance(c, dict) or set(c) != keys or type(c['enabled']) is not bool:
        raise ValueError('Invalid wall_hold_practice settings')
    for key, value in c.items():
        if key in ('enabled', 'hold_s_range', 'handover'):
            continue
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError('Invalid wall hold parameter '+key)
    times = c['hold_s_range']
    if (not isinstance(times, list) or len(times) != 2
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in times)
            or not 0 < times[0] <= times[1] < c['timeout_s']-c['settle_grace_s']):
        raise ValueError('Invalid wall hold timing')
    if not 0 < c['motion_environment_fraction'] <= .5:
        raise ValueError('Wall hold must preserve at least half the motion cohort')
    if not 0 < c['position_deadband_m'] < c['position_m'] <= .06:
        raise ValueError('Wall hold requires a strict fixed position region')
    if c['speed_m_s'] > terrain['commands']['stop_speed_m_s'] or c['yaw_rate_rad_s'] > terrain['commands']['stop_yaw_rad_s']:
        raise ValueError('Wall hold cannot relax stationary tolerances')
    if c['success_reward'] > terrain['rewards']['success']:
        raise ValueError('Wall hold cannot increase route completion bonus')
    if c['enabled'] and (not terrain.get('wall_practice', {}).get('enabled')
                         or 'stair_hold_practice' not in terrain):
        raise ValueError('Wall hold requires physical walls and the verified default wheel FK ledger')


class WallHoldPractice:
    def __init__(self, state):
        self.state = state
        self.cfg = state.cfg['wall_hold_practice']
        self.num_groups = 2 if 'handover' in self.cfg else 1
        self.curriculum = CourseCurriculum(state.cfg['curriculum'], num_courses=self.num_groups)
        n, dev = state.env.num_envs, state.env.device
        self.active = torch.zeros(n, dtype=torch.bool, device=dev)
        # Reserve environments, not reset draws: shorter holds must not vanish
        # behind long wall-traversal episodes. Preserve teacher/student cohorts.
        for teacher in (True, False):
            ids = ((state.course_ids == 3) & (state.teacher == teacher)).nonzero().flatten()
            count = int(len(ids)*self.cfg['motion_environment_fraction'])
            self.active[ids[:count]] = True
        self.group = torch.zeros(n, dtype=torch.long, device=dev)
        self.handover = None
        if 'handover' in self.cfg:
            from .c_handover import CHandover
            self.handover = CHandover(self)
        self.anchor = torch.zeros(n, 3, device=dev)
        self.quaternion = torch.zeros(n, 4, device=dev)
        self.quaternion[:, 0] = 1.
        self.foot_anchor = torch.zeros(n, 4, 2, device=dev)
        self.previous_feet = self.foot_anchor.clone()
        self.steps = torch.zeros(n, dtype=torch.long, device=dev)
        self.required = self.steps.clone()
        self.ready = self.active.clone().fill_(False)
        self.escaped = self.ready.clone()
        self.max_drift = torch.zeros(n, device=dev)
        self.max_backward = self.max_drift.clone()
        self.peak_speed = self.max_drift.clone()
        self.cost = self.max_drift.clone()
        self.linear = self.max_drift.clone()
        self.angular = self.max_drift.clone()
        self.tick = -1
        self.totals = {key: torch.zeros(2*self.num_groups, dtype=torch.float64, device=dev) for key in (
            'episodes', 'successes', 'timeouts', 'physical_failures', 'escaped_episodes',
            'handover_entries', 'prehold_frames', 'prehold_timeouts',
            'frames', 'stable_frames', 'support_failures', 'position_failures',
            'body_speed_failures', 'wheel_speed_failures', 'yaw_rate_failures',
            'roll_pitch_rate_failures', 'slip_failures', 'command_failures',
            'max_drift_m_sum', 'max_backward_m_sum', 'peak_speed_m_s_sum',
            'body_speed_sq_sum', 'wheel_speed_sq_sum', 'contact_slip_sq_sum',
            'position_cost_sum', 'motion_cost_sum', 'slip_cost_sum',
            'support_cost_sum', 'posture_cost_sum', 'left_offset_episodes',
            'right_offset_episodes', 'boundary_episodes', 'delayed_episodes')}

    def reset(self, ids, position, quat):
        s = self.state
        selected = self.active[ids]
        chosen = ids[selected]
        for value in (self.steps, self.ready, self.escaped, self.max_drift,
                      self.max_backward, self.peak_speed, self.cost, self.linear, self.angular):
            value[ids] = 0
        if not len(chosen):
            return
        for tile in s.tile[chosen].tolist():
            if s.atlas.records[tile].get('wall_kind') != 'corridor':
                raise ValueError('Wall hold must spawn inside a real two-sided corridor')
        fk = s.cfg['stair_hold_practice']
        local = position.new_tensor(fk['wheel_centers_body_m'])
        from task_support.student_v1.perception import quaternion_rotation
        position[selected, 0] = s.origins[s.tile[chosen], 0]+torch.empty(len(chosen), device=ids.device).uniform_(-1.5, 1.5)
        position[selected, 2] = s.origins[s.tile[chosen], 2]-local[0, 2]+s.env.cfg.stair_tail.wheel_radius+fk['spawn_clearance_m']
        robot = s.env.scene['robot']
        robot.write_joint_state_to_sim(robot.data.default_joint_pos[chosen],
            torch.zeros_like(robot.data.default_joint_pos[chosen]), env_ids=chosen)
        self.anchor[chosen] = position[selected]
        self.quaternion[chosen] = quat[selected]
        rot = quaternion_rotation(quat[selected])
        feet = (rot@local.T).transpose(1, 2)+position[selected, None]
        self.foot_anchor[chosen] = feet[..., :2]
        self.previous_feet[chosen] = feet[..., :2]
        s.command_value[chosen] = 0
        s.arc_transition[chosen] = False
        lo, hi = self.cfg['hold_s_range']
        duration = lo+(hi-lo)*s.geometry_level[chosen].float()/max(s.cfg['geometry']['geometry_levels']-1, 1)
        self.required[chosen] = torch.ceil(duration/s.env.step_dt-1.e-6).long()
        if self.handover is not None:
            self.handover.reset(chosen)

    def validate_initial(self):
        s = self.state
        ids = self.active.nonzero().flatten()
        if not len(ids) or not s.teacher[ids].any() or not (~s.teacher[ids]).any():
            raise RuntimeError('Wall hold requires initial teacher and student exposure')
        g = s.observe()
        d = s.env.scene['robot'].data
        values = (g['position'][ids], g['clearance'][ids], d.root_pos_w[ids], d.root_quat_w[ids])
        if not all(torch.isfinite(v).all() for v in values):
            raise RuntimeError('Nonfinite wall hold initial geometry')
        error = float((g['position'][ids, :, :2]-self.foot_anchor[ids]).abs().max())
        clearance = g['clearance'][ids]
        s.walls.measure()
        wall_clearance = s.walls.clearance[ids]
        if error > .002 or clearance.min() < -.003 or clearance.max() > .015 or wall_clearance.min() < 0:
            raise RuntimeError('Wall hold reset has invalid FK, support or wall clearance')
        if s.raw_command()[ids].abs().max() > 0:
            raise RuntimeError('Wall hold delivered a nonzero reset command')
        widths = s.widths[s.tile[ids]]
        return dict(corridor_width_min_m=float(widths.min()), corridor_width_max_m=float(widths.max()),
                    course_groups=self.num_groups, environments=len(ids), teacher=int(s.teacher[ids].sum()),
                    student=int((~s.teacher[ids]).sum()), fk_xy_max_error_m=error,
                    min_wheel_clearance_m=float(clearance.min()), max_wheel_clearance_m=float(clearance.max()),
                    min_wall_clearance_m=float(wall_clearance.min()))

    def advance(self, g):
        s, c = self.state, self.cfg
        env, d = s.env, s.env.scene['robot'].data
        if self.tick == env.common_step_counter:
            return
        self.tick = env.common_step_counter
        active = self.active
        age = env.episode_length_buf.double()*env.step_dt
        if self.handover is not None:
            waiting = self.handover.active & ~self.handover.started
            cells = 2*self.group[waiting]+(~s.teacher[waiting]).long()
            self.totals['prehold_frames'].index_add_(0, cells, torch.ones_like(cells, dtype=torch.float64))
            active = active & self.handover.policy_active()
            age = self.handover.policy_age(age)
        measured = active & (age >= c['settle_grace_s'])
        feet = g['position'][..., :2]
        foot_speed = ((feet-self.previous_feet)/env.step_dt).norm(dim=-1).amax(-1)
        self.previous_feet.copy_(feet)
        velocity = d.root_lin_vel_w[:, :2]
        omega = d.root_ang_vel_b
        body_speed = velocity.norm(dim=-1)
        heading = yaw(d.root_quat_w)-yaw(self.quaternion)
        heading = torch.atan2(heading.sin(), heading.cos())
        delta = torch.cat(((d.root_pos_w[:, :2]-self.anchor[:, :2])[:, None], feet-self.foot_anchor), 1)
        drift = delta.norm(dim=-1).amax(-1)
        axis = torch.stack((yaw(self.quaternion).cos(), yaw(self.quaternion).sin()), -1)
        backward = (-(delta*axis[:, None]).sum(-1)).clamp_min(0).amax(-1)
        finite = torch.isfinite(delta).all(dim=(-1, -2)) & torch.isfinite(velocity).all(-1) & torch.isfinite(omega).all(-1) & torch.isfinite(d.root_quat_w).all(-1)
        for key in ('position', 'velocity', 'normal', 'force'):
            finite &= torch.isfinite(g[key]).all(dim=(-1, -2))
        # The anchor never moves, including during the settling grace. An
        # out-and-back manoeuvre or stopped retreat must not receive success.
        region = (drift <= c['position_m']) & (heading.abs() <= c['yaw_rad'])
        self.escaped |= active & ~region
        self.max_drift = torch.maximum(self.max_drift, torch.where(active, torch.nan_to_num(drift, nan=10.), 0.))
        self.max_backward = torch.maximum(self.max_backward, torch.where(active, torch.nan_to_num(backward, nan=10.), 0.))
        speed = torch.maximum(body_speed, foot_speed)
        self.peak_speed = torch.maximum(self.peak_speed, torch.where(measured, torch.nan_to_num(speed, nan=10.), 0.))
        supported = g['supported'] & finite[:, None]
        support_ok = (supported.sum(-1) >= 3) & supported[:, :2].any(-1) & supported[:, 2:].any(-1)
        tangent = g['velocity']-(g['velocity']*g['normal']).sum(-1, keepdim=True)*g['normal']
        slip = torch.nan_to_num(tangent.norm(dim=-1), nan=10.)*supported
        body_ok = body_speed <= c['speed_m_s']
        wheel_ok = foot_speed <= c['speed_m_s']
        yaw_ok = omega[:, 2].abs() <= c['yaw_rate_rad_s']
        posture_ok = omega[:, :2].norm(dim=-1) <= c['roll_pitch_rate_rad_s']
        slip_ok = slip.amax(-1) <= c['slip_m_s']
        raw = g['command']
        # Heading feedback may command angular correction; it cannot request translation.
        tracking = g.get('tracking_command', raw)
        command_ok = (raw.abs().amax(-1) <= 1.e-4) & torch.isfinite(tracking).all(-1) & (tracking[:, :2].abs().amax(-1) <= 1.e-4)
        self.ready.copy_(measured & finite & ~self.escaped & support_ok & body_ok & wheel_ok & yaw_ok & posture_ok & slip_ok & command_ok & ~s.failed)
        self.steps = advance_steps(self.steps, self.ready)
        s.failed |= active & ~finite
        s.success |= active & (self.steps >= self.required) & ~s.failed
        pc = ((drift-c['position_deadband_m']).clamp_min(0)/c['position_scale_m']).square().clamp(max=16)
        pc += (heading/c['yaw_rad']).square().clamp(max=16)
        vc = (speed/c['speed_scale_m_s']).square().clamp(max=16)
        sc = (slip/c['slip_scale_m_s']).square().clamp(max=16).mean(-1)
        uc = 1-supported.float().mean(-1)+(~support_ok).float()
        ac = (omega[:, :2].norm(dim=-1)/c['roll_pitch_rate_rad_s']).square().clamp(max=16)
        parts = dict(position=c['position_weight']*pc, motion=c['motion_weight']*vc,
                     slip=c['slip_weight']*sc, support=c['support_weight']*uc, posture=c['posture_weight']*ac)
        self.cost.copy_(torch.where(measured, torch.nan_to_num(sum(parts.values()), nan=32., posinf=32.), 0.))
        credit = torch.exp(-torch.nan_to_num(pc, nan=16.))
        self.linear.copy_(torch.where(measured, torch.exp(-torch.nan_to_num(vc, nan=16.))*credit, 0.))
        self.angular.copy_(torch.where(measured, torch.exp(-torch.nan_to_num((omega[:, 2]/c['yaw_rate_rad_s']).square(), nan=16.))*credit, 0.))
        values = dict(frames=measured, stable_frames=self.ready, support_failures=measured & ~support_ok,
            position_failures=measured & ~region, body_speed_failures=measured & ~body_ok,
            wheel_speed_failures=measured & ~wheel_ok, yaw_rate_failures=measured & ~yaw_ok,
            roll_pitch_rate_failures=measured & ~posture_ok, slip_failures=measured & ~slip_ok,
            command_failures=measured & ~command_ok,
            body_speed_sq_sum=torch.where(measured, torch.nan_to_num(body_speed.square(), nan=100., posinf=100.), 0.),
            wheel_speed_sq_sum=torch.where(measured, torch.nan_to_num(foot_speed.square(), nan=100., posinf=100.), 0.),
            contact_slip_sq_sum=torch.where(measured, slip.square().mean(-1), 0.))
        values.update({k+'_cost_sum': torch.where(measured, torch.nan_to_num(v, nan=32., posinf=32.), 0.) for k, v in parts.items()})
        cells = 2*self.group[active]+(~s.teacher[active]).long()
        for key, value in values.items():
            self.totals[key].index_add_(0, cells, value[active].double())

    def record(self, ids, delayed):
        s = self.state
        for i in ids[self.active[ids]].tolist():
            cohort = 2*int(self.group[i])+int(not s.teacher[i])
            success = bool(s.success[i] and not s.failed[i])
            offset = float(self.anchor[i, 1]-s.origins[s.tile[i], 1])
            boundary = bool(s.boundary_tiles[s.tile[i]])
            values = dict(episodes=1, successes=success, timeouts=bool(s.env.termination_manager.time_outs[i]) and not success,
                physical_failures=s.failed[i], escaped_episodes=self.escaped[i], max_drift_m_sum=self.max_drift[i],
                max_backward_m_sum=self.max_backward[i], peak_speed_m_s_sum=self.peak_speed[i],
                left_offset_episodes=offset >= 0, right_offset_episodes=offset < 0,
                boundary_episodes=boundary, delayed_episodes=delayed[i])
            if self.handover is not None and self.handover.active[i] and not self.handover.started[i]:
                self.totals['prehold_timeouts'][cohort] += int(bool(s.env.termination_manager.time_outs[i]))
            for key, value in values.items():
                self.totals[key][cohort] += value
            self.curriculum.record(int(self.group[i]), success, bool(s.teacher[i]), int(s.role[i]), int(s.version[i]),
                                   s.update_id, boundary=boundary, delayed=bool(delayed[i]))
        s.finished[ids[self.active[ids]]] = True
        return ids[~self.active[ids]]

    def diagnostics(self):
        result = {}
        for group in range(self.num_groups):
            prefix = 'c_handover/' if group else ''
            for key, value in self.totals.items():
                for t, co in enumerate(('teacher', 'student')):
                    result[prefix+co+'/'+key] = float(value[2*group+t])
            grade = self.curriculum.diagnostics()[group]
            for key in ('geometry_level', 'dr_level', 'promotions', 'demotions'):
                result[prefix+'level/'+key] = grade[key]
            result[prefix+'active_environments'] = int((self.active & (self.group == group)).sum())
        return result

    def snapshot(self):
        return dict(curriculum=self.curriculum.state_dict(), totals={k: v.cpu().clone() for k, v in self.totals.items()})

    def checked_snapshot(self, saved):
        if not isinstance(saved, dict) or set(saved) != {'curriculum', 'totals'} or not isinstance(saved['totals'], dict) or saved['totals'].keys() != self.totals.keys():
            raise ValueError('Invalid wall hold snapshot')
        cur = CourseCurriculum(self.state.cfg['curriculum'], num_courses=self.num_groups)
        cur.load_state_dict(saved['curriculum'])
        for key, value in saved['totals'].items():
            target = self.totals[key]
            if not isinstance(value, torch.Tensor) or value.shape != target.shape or value.dtype != target.dtype or not torch.isfinite(value).all() or (value < 0).any():
                raise ValueError('Invalid wall hold totals '+key)
        if (saved['totals']['successes'] > saved['totals']['episodes']).any():
            raise ValueError('Invalid wall hold success count')
        return saved

    def restore(self, saved):
        self.curriculum.load_state_dict(saved['curriculum'])
        for key, value in saved['totals'].items():
            self.totals[key].copy_(value)
