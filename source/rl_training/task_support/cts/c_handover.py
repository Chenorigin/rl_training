"""Independent ready-to-C course: physical stand hold, cold history, issued-action ramp.

This covers ready -> C, not the prone -> Z stand-up trajectory. All policy samples
remain raw for PPO; the simulator applies a non-mutating execution transform.
"""
import math
import torch


def validate(c):
    if not isinstance(c, dict) or set(c) != {'fraction', 'standing_s_range', 'blend_s'}:
        raise ValueError('Invalid C handover configuration')
    if type(c['fraction']) not in (int, float) or not 0 < c['fraction'] < 1:
        raise ValueError('C handover must retain a separate direct-hold cohort')
    times = c['standing_s_range']
    if (not isinstance(times, list) or len(times) != 2 or
            any(type(t) not in (int, float) or not math.isfinite(t) or t <= 0 for t in times)
            or times[0] > times[1]):
        raise ValueError('Invalid physical standing interval')
    if type(c['blend_s']) not in (int, float) or c['blend_s'] != .3:
        raise ValueError('C handover must match the deployed 0.3s issued-action ramp')


class CHandover:
    def __init__(self, hold):
        self.hold, self.state = hold, hold.state
        self.cfg = hold.cfg['handover']
        self.active = torch.zeros_like(hold.active)
        for teacher in (True, False):
            ids = (hold.active & (self.state.teacher == teacher)).nonzero().flatten()
            count = int(len(ids)*self.cfg['fraction'])
            if not 0 < count < len(ids):
                raise ValueError('C handover needs direct and handover samples in both cohorts')
            self.active[ids[:count]] = True
        hold.group[self.active] = 1
        self.started = torch.zeros_like(self.active)
        self.start_age = torch.zeros_like(self.active, dtype=torch.float64)
        self.wait_until = torch.zeros_like(self.start_age)

    def reset(self, ids):
        self.started[ids] = False
        self.start_age[ids] = 0.
        lo, hi = self.cfg['standing_s_range']
        self.wait_until[ids] = torch.empty(len(ids), device=ids.device, dtype=torch.float64).uniform_(lo, hi)

    def policy_active(self):
        return ~self.active | self.started

    def policy_age(self, age):
        return torch.where(self.active, age-self.start_age, age)

    def issued_action(self, action):
        age = self.state.env.episode_length_buf.double()*self.state.env.step_dt
        scale = (self.policy_age(age)/self.cfg['blend_s']).clamp(0., 1.)
        scale = torch.where(self.started, scale, 0.)
        scale = torch.where(self.active, scale, 1.).to(action.dtype)
        return action*scale[:, None]  # Never mutate pending PPO actions/log probabilities.

    def after_physics(self):
        """Called before next actor observation, after this step's physics."""
        s, h = self.state, self.hold
        age = (s.env.episode_length_buf.double()+1)*s.env.step_dt
        candidates = self.active & ~self.started & (age >= self.wait_until)
        if not candidates.any():
            return torch.empty(0, dtype=torch.long, device=age.device)
        s.measure_tick = -1
        g = s.observe()
        d = s.env.scene['robot'].data
        q = d.root_quat_w
        upright = 1-2*(q[:, 1].square()+q[:, 2].square())
        support = g['supported']
        ready = (support.sum(-1) >= 3) & support[:, :2].any(-1) & support[:, 2:].any(-1)
        ready &= (upright > .8) & torch.isfinite(d.root_pos_w).all(-1) & ~s.failed
        ids = (candidates & ready).nonzero().flatten()
        if not len(ids):
            return ids
        cells = 2*h.group[ids]+(~s.teacher[ids]).long()
        h.totals['handover_entries'].index_add_(0, cells, torch.ones_like(cells, dtype=torch.float64))
        self.started[ids] = True
        self.start_age[ids] = age[ids]
        # Only the C boundary can establish this reference; never relabel later drift.
        h.anchor[ids] = d.root_pos_w[ids]
        h.quaternion[ids] = q[ids]
        h.foot_anchor[ids] = g['position'][ids, :, :2]
        h.previous_feet[ids] = g['position'][ids, :, :2]
        from task_support.student_v1.environment import reset_student
        from task_support.student_v3_5.v3_5_environment import reset_feedback
        from .all_terrain_terms import reset_clean_commands
        reset_student(s.env, ids)
        from task_support.student_v1.environment import sensor_state, action_delay
        if sensor_state(s.env).history[ids].abs().max() > 0 or action_delay(s.env).previous_issued[ids].abs().max() > 0:
            raise RuntimeError('C handover retained pre-policy observation/action history')
        reset_feedback(s.env, ids)
        reset_clean_commands(s.env, ids)
        s.env.stair_quiet_landing.before_reset(ids)
        s.env.stair_quiet_landing.reset(ids)
        s.env.action_manager.reset(ids)
        # No root/joint writes: standing momentum and contact state cross C unchanged.
        return ids


def install_standing_execution(env):
    """Restore ready-mode PD targets/gains after landing processing, before physics.

    The shared landing implementation is frozen by the source checkpoint. Keep
    its behavior for every other environment and bypass its physical edits only
    for the waiting cohort, just as the MuJoCo ready branch bypasses execution.
    Controller state is reset at C, so pre-C internal calculations cannot leak.
    """
    landing = env.stair_quiet_landing
    original = landing.apply

    def apply(dt):
        original(dt)
        hold = env.cts_all_terrain.wall_hold
        hand = hold.handover if hold is not None else None
        if hand is None:
            return
        ids = (hand.active & ~hand.started).nonzero().flatten()
        if not len(ids):
            return
        robot = env.scene['robot']
        legs, wheels = landing.leg_ids.flatten(), landing.wheel_ids
        target = robot.data.default_joint_pos[ids][:, legs]
        robot.set_joint_position_target(target, joint_ids=legs, env_ids=ids)
        robot.set_joint_velocity_target(torch.zeros(len(ids), len(wheels), device=ids.device),
                                        joint_ids=wheels, env_ids=ids)
        landing.before_reset(ids)

    landing.apply = apply
