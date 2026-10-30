"""Substep safety, rolling slip and unsaturated impact-tail objectives."""
import torch
from .geometry import foothold_regions, capsule_contact_velocity, wheel_margins


class TailTracker:
    def __init__(self, n, device, settings):
        self.s = settings
        self.cost = {k: torch.zeros(n, device=device) for k in (
            "edge", "foothold", "slip", "recovery_impact", "tail_force", "tail_impulse", "progress")}
        self.support_age = torch.zeros(n, 4, device=device)
        self.slip_age = torch.zeros(n, device=device)
        self.recovering = torch.zeros(n, dtype=torch.bool, device=device)
        self.recovery_age = torch.zeros(n, device=device)
        self.recovery_no_progress_age = torch.zeros(n, device=device)
        self.recovery_progress_x = torch.zeros(n, device=device)
        self.recovery_support_loss_age = torch.zeros(n, device=device)
        self.recovery_stable = torch.zeros(n, device=device)
        self.recovery_start_x = torch.zeros(n, device=device)
        self.recovery_failed = torch.zeros(n, dtype=torch.bool, device=device)
        self.edge_failed = torch.zeros(n, dtype=torch.bool, device=device)
        self.success = torch.zeros(n, dtype=torch.bool, device=device)
        self.success_age = torch.zeros(n, device=device)
        self.top_seen = torch.zeros(n, 4, dtype=torch.bool, device=device)
        self.stall_age = torch.zeros(n, device=device)
        self.best_x = torch.zeros(n, device=device)
        self.force_peak = torch.zeros(n, device=device)
        self.impulse_peak = torch.zeros(n, device=device)
        self.slip_seconds = torch.zeros(n, device=device)
        self.unsafe_seconds = torch.zeros(n, device=device)
        self.unsafe_foot_seconds = torch.zeros(n, 4, device=device)
        self.slip_foot_seconds = torch.zeros(n, 4, device=device)
        self.support_foot_seconds = torch.zeros(n, 4, device=device)
        self.recovery_events = torch.zeros(n, device=device)
        self.prev_force_tail = torch.zeros(n, 4, device=device)
        self.prev_impulse_tail = torch.zeros(n, 4, device=device)
        self.window = None
        self.cursor = 0

    def begin_step(self):
        for v in self.cost.values():
            v.zero_()

    def reset(self, ids, root_x):
        for name, v in vars(self).items():
            if isinstance(v, torch.Tensor):
                if name == "window":
                    v[:, ids] = 0
                else:
                    v[ids] = 0
        self.best_x[ids] = root_x

    def update(self, *, position, axis, velocity, omega, forces, root, root_velocity,
               yaw, upright, height, depth, width, steps, dt):
        s = self.s
        tread_margin, lateral_margin = wheel_margins(s, position)
        if self.window is None:
            length = max(1, round(s.tail_window_s/dt))
            self.window = torch.zeros(length, *forces.shape[:2], device=forces.device)
        forward, side, top = foothold_regions(position, axis, height, depth, width, steps, s)
        force = torch.linalg.vector_norm(forces, dim=-1)
        # Vertical tread support only; horizontal riser forces cannot count.
        loaded = (forces[..., 2] > s.support_force_N) & (
            forces[..., 2] > torch.linalg.vector_norm(forces[..., :2], dim=-1))
        bottom = position[..., 2]-s.wheel_radius-s.wheel_half_length*axis[..., 2].abs()
        supported = loaded & ((bottom-top).abs() < .05)
        self.support_age = torch.where(supported, self.support_age+dt, 0.)
        stable_support = supported & (self.support_age >= s.landing_grace_s)
        point_v = capsule_contact_velocity(velocity, omega, axis, s.wheel_radius, s.wheel_half_length)
        tangential = torch.linalg.vector_norm(point_v[..., :2], dim=-1)
        # No penalty on airborne spin or normal rolling. Unclipped excess.
        slip = (tangential-s.slip_speed_m_s).clamp_min(0)*supported
        slipping = (slip > 0).any(-1)
        self.slip_age = torch.where(slipping, self.slip_age+dt, 0.)
        onset = (self.slip_age >= s.slip_on_s) & ~self.recovering
        self.recovery_start_x[onset] = root[onset, 0]
        self.recovery_progress_x[onset] = root[onset, 0]
        self.recovery_age[onset] = 0.
        self.recovery_no_progress_age[onset] = 0.
        self.recovery_support_loss_age[onset] = 0.
        self.recovering |= onset
        self.recovery_events += onset
        self.recovery_age = torch.where(self.recovering, self.recovery_age+dt, 0.)
        # Normal stair motion may keep one loaded contact above the slip
        # threshold. Charge its slip cost below, but do not end a progressing
        # ascent merely because the recovery state has lasted too long.
        # Milestones only move forward: rocking back and forth or noisy speed
        # cannot repeatedly renew the grace period without new net progress.
        progress = self.recovering & (root[:, 0] >= self.recovery_progress_x+s.forward_progress_m)
        self.recovery_progress_x = torch.where(progress, root[:, 0], self.recovery_progress_x)
        self.recovery_no_progress_age = torch.where(
            self.recovering & ~progress, self.recovery_no_progress_age+dt, 0.)
        self.recovery_support_loss_age = torch.where(
            self.recovering & (supported.sum(-1) < 2), self.recovery_support_loss_age+dt, 0.)
        stable = (~slipping & (stable_support.sum(-1) >= 2) & (upright > .9)
                  & (root[:, 0] >= self.recovery_start_x+s.forward_progress_m))
        self.recovery_stable = torch.where(self.recovering & stable, self.recovery_stable+dt, 0.)
        self.recovering &= self.recovery_stable < s.recovery_hold_s
        self.recovery_failed |= self.recovering & (
            (self.recovery_no_progress_age > s.recovery_timeout_s)
            | (self.recovery_support_loss_age > s.recovery_support_loss_s))

        projected_y = position[..., 1]+velocity[..., 1]*s.prediction_s
        extent_y = s.wheel_radius+s.wheel_half_length*axis[..., 1].abs()
        predicted_side = width[:, None]/2-projected_y.abs()-extent_y-lateral_margin
        body_extent = s.body_half_length*yaw.sin().abs()+s.body_half_width*yaw.cos().abs()
        body_margin = width/2-root[:, 1].abs()-body_extent-s.body_lateral_margin
        edge_cost = ((-side).clamp_min(0)/lateral_margin).square().sum(-1)
        edge_cost += ((-predicted_side).clamp_min(0)/lateral_margin).square().sum(-1)
        edge_cost += ((-body_margin).clamp_min(0)/s.body_lateral_margin).square()
        self.edge_failed |= ((position[..., 1].abs()+extent_y > width[:, None]/2).any(-1)
                             | (root[:, 1].abs()+body_extent > width/2))
        unsafe = ((forward < 0) | (side < 0)) & stable_support
        self.cost["edge"] += edge_cost*dt
        self.cost["foothold"] += (((-forward).clamp_min(0)/tread_margin).square()*stable_support).sum(-1)*dt
        self.cost["slip"] += (slip/.3).square().sum(-1)*dt
        self.slip_seconds += slipping*dt
        self.unsafe_seconds += unsafe.any(-1)*dt
        self.unsafe_foot_seconds += unsafe*dt
        self.slip_foot_seconds += (slip > 0)*dt
        self.support_foot_seconds += supported*dt

        self.window[self.cursor] = force
        self.cursor = (self.cursor+1) % len(self.window)
        window_peak = self.window.amax(0)
        impulse = (self.window-100.).clamp_min(0).sum(0)*dt
        phi_force = ((window_peak-s.tail_force_N).clamp_min(0)/s.tail_force_scale_N).square()
        phi_impulse = ((impulse-s.tail_impulse_N_s).clamp_min(0)/s.tail_impulse_scale_N_s).square()
        # Charge increases immediately, before possible failure/reset. No
        # saturation of large outliers, and no repeated charge of a plateau.
        self.cost["tail_force"] += (phi_force-self.prev_force_tail).clamp_min(0).sum(-1)
        self.cost["tail_impulse"] += (phi_impulse-self.prev_impulse_tail).clamp_min(0).sum(-1)
        self.prev_force_tail.copy_(phi_force)
        self.prev_impulse_tail.copy_(phi_impulse)
        self.cost["recovery_impact"] += self.recovering*(
            (force-200.).clamp_min(0)/s.tail_force_scale_N).square().sum(-1)*dt
        self.force_peak = torch.maximum(self.force_peak, force.amax(-1))
        self.impulse_peak = torch.maximum(self.impulse_peak, impulse.amax(-1))
        advance = (root[:, 0]-self.best_x).clamp_min(0)
        self.cost["progress"] += advance
        self.best_x = torch.maximum(self.best_x, root[:, 0])
        self.stall_age = torch.where(advance/dt >= s.min_progress_speed_m_s, 0., self.stall_age+dt)
        goal = s.start_length/2+steps*depth
        beyond = position[..., 0] > goal[:, None]+tread_margin
        crossed = beyond.all(-1)
        self.top_seen = (self.top_seen | (beyond & supported)) & beyond
        safe = ~self.edge_failed & ~self.recovery_failed
        landed = (crossed & self.top_seen.all(-1) & safe & (stable_support.sum(-1) >= 2) & (upright > .95)
                  & (yaw.abs() < .2) & (tangential.amax(-1) < .3)
                  & (root[:, 2] >= steps*height+.15)
                  & (torch.linalg.vector_norm(root_velocity, dim=-1) < .65))
        self.success_age = torch.where(landed, self.success_age+dt, 0.)
        self.success |= self.success_age >= s.success_hold_s
        self.last = {"forward_margin": forward, "side_margin": side,
                     "supported": supported, "slip_speed": tangential,
                     "crossed": crossed & self.top_seen.all(-1)}

    def failed(self):
        return self.edge_failed | self.recovery_failed | (self.stall_age > self.s.no_progress_timeout_s)


def tail_cost(env, name):
    tracker = getattr(env, "tail_tracker", None)
    if tracker is None:
        return torch.zeros(env.num_envs, device=env.device)
    if env.cfg.stair_tail.objective == "control" and name in (
            "slip", "recovery_impact", "tail_force", "tail_impulse"):
        return torch.zeros_like(tracker.cost[name])
    scale = 1.
    if name in ("tail_force", "tail_impulse", "recovery_impact"):
        scale = (.1, .35, 1.)[env.tail_stage]
    return tracker.cost[name]*scale/env.step_dt


def tail_failure(env):
    tracker = getattr(env, "tail_tracker", None)
    if tracker is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    root = env.scene["robot"].data.root_pos_w-env.scene.env_origins
    s = env.cfg.stair_tail
    # Under-board fall, backwards escape and end-of-tile escape are failures.
    return (tracker.failed() | ~torch.isfinite(root).all(-1) | (root[:, 2] < .10)
            | (root[:, 0] < -.7) | (root[:, 0] > s.tile_length-s.start_length/2-.3))


def tail_success(env):
    tracker = getattr(env, "tail_tracker", None)
    if tracker is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    # Existing fall/tilt failures still take precedence in episode accounting.
    return tracker.success & ~tail_failure(env)


def success_reward(env):
    return (tail_success(env) & ~failure_mask(env)).float()/env.step_dt


def failure_reward(env):
    return failure_mask(env).float()/env.step_dt


def failure_mask(env):
    """A real completed ascent is terminal but never a failure. Simultaneous
    illegal contact/tilt/escape still wins over the completion latch."""
    manager = env.termination_manager
    failed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for name in manager.active_terms:
        if name != "tail_success" and not manager.get_term_cfg(name).time_out:
            failed |= manager.get_term(name)
    return failed
