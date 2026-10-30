"""Substep lip-contact work and descending swing risk; original hard failures."""
import torch
from task_support.stair_quiet_v1_safe.signals import SafeTailTracker

METRICS = ("riser_work_proxy_J", "riser_impulse_proxy_Ns", "riser_descent_seconds", "entry_yaw_peak_rad",
           "entry_side_peak_m", "post_stop_yaw_peak_rad")


class TransitionTracker(SafeTailTracker):
    def __init__(self, n, device, settings):
        super().__init__(n, device, settings)
        for key in ("riser_work", "riser_descent"):
            self.cost[key] = torch.zeros(n, device=device)
        for key in METRICS:
            setattr(self, key, torch.zeros(n, device=device))
        self.previous_wheel_vx = torch.zeros(n, 4, device=device)
        self.previous_velocity_valid = torch.zeros(n, dtype=torch.bool, device=device)

    def update(self, *, stopped=None, closed_riser=None, command_paused=None, **d):
        prior_stall = self.stall_age.clone()
        prior_recovery_clock = self.recovery_no_progress_age.clone()
        prior_failure = self.recovery_failed.clone()
        super().update(**d)
        s, dt = self.s, d["dt"]
        if command_paused is not None:
            # Intentional supported entry waits are not commanded progress.
            # Never exempt edge escape, lost support or an earlier failure.
            waiting = command_paused & (self.last["supported"].sum(-1) >= 2) & (d["upright"] > .9)
            self.stall_age = torch.where(waiting, prior_stall, self.stall_age)
            self.recovery_no_progress_age = torch.where(waiting, prior_recovery_clock, self.recovery_no_progress_age)
            physical_loss = self.recovering & (self.recovery_support_loss_age > s.recovery_support_loss_s)
            self.recovery_failed = torch.where(waiting, prior_failure | physical_loss, self.recovery_failed)
        pos, axis, velocity = d["position"], d["axis"], d["velocity"]
        start = s.start_length/2
        # Nearest actual stair front (not merely any horizontal force/side wall).
        index = torch.round((pos[..., 0]-start)/d["depth"][:, None]).long().clamp_min(0)
        index = torch.minimum(index, d["steps"][:, None]-1)
        front = start+index*d["depth"][:, None]
        top = (index+1)*d["height"][:, None]
        extent_x = s.wheel_radius+s.wheel_half_length*axis[..., 0].abs()
        bottom = pos[..., 2]-s.wheel_radius-s.wheel_half_length*axis[..., 2].abs()
        near = ((pos[..., 0]-front).abs() < extent_x+s.riser_clearance_m) & (bottom < top)
        backward = (-d["forces"][..., 0]).clamp_min(0)
        closed = torch.zeros_like(d["width"], dtype=torch.bool) if closed_riser is None else closed_riser.bool()
        extent_z = s.wheel_radius+s.wheel_half_length*axis[..., 2].abs()
        face_exists = closed[:, None] | (pos[..., 2]+extent_z >= top-s.board_thickness)
        force_gate = (near & face_exists & (backward > s.riser_force_reference_N)
                      & (backward > d["forces"][..., 2].abs())
                      & (self.previous_wheel_vx > .05) & self.previous_velocity_valid[:, None])
        # Previous post-step velocity is this step's PRE-solve sample. Skip the
        # first sample after reset; never multiply post-impact speed into work.
        work = backward*self.previous_wheel_vx.clamp_min(0)*force_gate*self.previous_velocity_valid[:, None]*dt
        self.riser_work_proxy_J += work.sum(-1)
        self.riser_impulse_proxy_Ns += (backward*force_gate).sum(-1)*dt
        self.cost["riser_work"] += work.sum(-1)/s.riser_energy_reference_J
        supported = self.last["supported"]
        next_index = torch.ceil((pos[..., 0]-start)/d["depth"][:, None]).long().clamp_min(0)
        next_index = torch.minimum(next_index, d["steps"][:, None]-1)
        next_front = start+next_index*d["depth"][:, None]
        next_top = (next_index+1)*d["height"][:, None]
        distance = next_front-pos[..., 0]
        crossing_time = (distance-extent_x).clamp_min(0)/velocity[..., 0].clamp_min(.01)
        # Use the SOURCE tread behind the upcoming front. The loaded-support
        # selector may assign an airborne wheel to the next lip by height;
        # using that assignment here would incorrectly exempt a collision.
        source_top = next_index*d["height"][:, None]
        time_to_current_tread = (bottom-source_top).clamp_min(0)/(-velocity[..., 2]).clamp_min(.01)
        approaching = (~supported & (velocity[..., 0] > .05) & (velocity[..., 2] < -s.prediction_descent_m_s)
                       & (distance > 0) & (crossing_time < s.prediction_s)
                       & (crossing_time < time_to_current_tread))
        predicted_bottom = bottom+velocity[..., 2]*crossing_time.clamp(0, s.prediction_s)
        risk = ((next_top+s.riser_clearance_m-predicted_bottom).clamp_min(0)/d["height"][:, None]).square()
        # The envelope may already overlap the front plane while its centre
        # is still above the lip. Check closest approach over the WHOLE short
        # trajectory, so a later fast descent cannot disappear at t=0.
        relative = torch.stack((distance, next_top-pos[..., 2]), -1)
        velocity_xz = velocity[..., (0, 2)]
        closest_time = ((relative*velocity_xz).sum(-1)/velocity_xz.square().sum(-1).clamp_min(1.e-6)).clamp(0, s.prediction_s)
        separation = (relative-velocity_xz*closest_time[..., None]).norm(dim=-1)
        # Frozen-axis XZ bounding circle conservatively includes the capsule's
        # projected axial segment. This remains reward shaping, not a CCD solver.
        projected_radius = s.wheel_radius+s.wheel_half_length*axis[..., (0, 2)].norm(dim=-1)
        swept_risk = ((projected_radius+s.riser_clearance_m-separation).clamp_min(0)/projected_radius).square()
        risk = torch.maximum(risk, swept_risk)
        self.cost["riser_descent"] += (risk*approaching).sum(-1)*dt
        self.riser_descent_seconds += approaching.any(-1)*dt
        early = (d["root"][:, 0] >= start-.8) & (d["root"][:, 0] < start+3*d["depth"]+.4)
        self.entry_yaw_peak_rad = torch.maximum(self.entry_yaw_peak_rad, d["yaw"].abs()*early)
        self.entry_side_peak_m = torch.maximum(self.entry_side_peak_m, d["root"][:, 1].abs()*early)
        if stopped is not None:
            self.post_stop_yaw_peak_rad = torch.maximum(self.post_stop_yaw_peak_rad, d["yaw"].abs()*stopped)
        self.previous_wheel_vx.copy_(velocity[..., 0])
        self.previous_velocity_valid.fill_(True)


def transition_cost(env, name):
    return env.tail_tracker.cost[name]/env.step_dt
