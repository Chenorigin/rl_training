"""Geometry-aware student tracker; old teacher kernels stay frozen for provenance.

Retains the width/Safe cost and recovery equations, replacing only support
geometry and the physical finish line/height. Regression checks legacy parity.
"""
import torch
from task_support.stair_quiet_v1_safe.signals import SafeTailTracker
from task_support.stair_quiet_v1_width.geometry import wheel_margins, capsule_contact_velocity
from .terrain_observations import safety_geometry

class StudentTerrainTracker(SafeTailTracker):
    def __init__(self,n,device,settings):
        super().__init__(n,device,settings)
        self.crest_seen=torch.zeros(n,4,dtype=torch.bool,device=device)

    def update(self, *, position, axis, velocity, omega, forces, root, root_velocity,
               yaw, upright, height, depth, width, steps, dt, spans, round_trip, goal, goal_height):
        d = dict(position=position,axis=axis,velocity=velocity,omega=omega,forces=forces,root=root,
                 root_velocity=root_velocity,yaw=yaw,upright=upright,height=height,depth=depth,width=width,
                 steps=steps,dt=dt,spans=spans,round_trip=round_trip)
        previous_unsafe_seconds = self.unsafe_seconds.clone()
        g = safety_geometry(d,self.s)
        s = self.s
        tread_margin, lateral_margin = wheel_margins(s, position)
        if self.window is None:
            length = max(1, round(s.tail_window_s/dt))
            self.window = torch.zeros(length, *forces.shape[:2], device=forces.device)
        forward, side, top = g["forward"], g["side"], g["top"]
        force = torch.linalg.vector_norm(forces, dim=-1)
        # Vertical tread support only; horizontal riser forces cannot count.
        loaded = (forces[..., 2] > s.support_force_N) & (
            forces[..., 2] > torch.linalg.vector_norm(forces[..., :2], dim=-1))
        bottom = position[..., 2]-s.wheel_radius-s.wheel_half_length*axis[..., 2].abs()
        supported = loaded & ((bottom-top).abs() < .05)
        self.crest_seen |= supported & round_trip[:,None] & ((top-steps[:,None]*height[:,None]).abs()<.01)
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
        beyond = position[..., 0] > goal[:, None]+tread_margin
        crossed = beyond.all(-1) & (~round_trip | self.crest_seen.all(-1))
        self.top_seen = (self.top_seen | (beyond & supported)) & beyond
        safe = ~self.edge_failed & ~self.recovery_failed
        landed = (crossed & self.top_seen.all(-1) & safe & (stable_support.sum(-1) >= 2) & (upright > .95)
                  & (yaw.abs() < .2) & (tangential.amax(-1) < .3)
                  & (root[:, 2] >= goal_height+.15)
                  & (torch.linalg.vector_norm(root_velocity, dim=-1) < .65))
        self.success_age = torch.where(landed, self.success_age+dt, 0.)
        self.success |= self.success_age >= s.success_hold_s
        self.last = {"forward_margin": forward, "side_margin": side,
                     "supported": supported, "slip_speed": tangential,
                     "crossed": crossed & self.top_seen.all(-1)}

        s, dt = self.s, d["dt"]
        supported = g["supported"]
        unsafe = (g["forward"] < 0) | (g["side"] < 0)
        early = supported & (self.support_age < s.landing_grace_s)
        approaching = (~supported & (d["velocity"][..., 2] < -s.prediction_descent_m_s)
            & (g["clearance"] > -.05) & (g["clearance"] < s.prediction_clearance_m))
        predicted_side = torch.minimum(g["predicted_left"], g["predicted_right"])
        risk_now = ((-g["forward"]).clamp_min(0)/.1).square()+((-g["side"]).clamp_min(0)/.1).square()
        risk_future = ((-g["predicted_forward"]).clamp_min(0)/.1).square()+((-predicted_side).clamp_min(0)/.1).square()
        self.cost["landing_risk"] += (risk_now*early+risk_future*approaching).sum(-1)*dt
        self.early_unsafe_foot_seconds += (unsafe & early)*dt
        self.predicted_unsafe_foot_seconds += (approaching & (
            (g["predicted_forward"] < 0) | (predicted_side < 0)))*dt
        self.touchdown_unsafe_count += (supported & ~self.previous_supported & unsafe)
        self.previous_supported.copy_(supported)
        # Account from first load; keep original stable-support metric separately.
        self.unsafe_foot_seconds += (unsafe & early)*dt
        self.unsafe_seconds.copy_(previous_unsafe_seconds+(unsafe & supported).any(-1)*dt)
        self.slip_foot_distance_m += g["slip_speed"]*(
            supported & (g["slip_speed"] > s.slip_speed_m_s))*dt
        self.elapsed_s += dt
        self.backward_distance_m += (self.previous_x-d["root"][:, 0]).clamp_min(0)
        self.previous_x.copy_(d["root"][:, 0])
