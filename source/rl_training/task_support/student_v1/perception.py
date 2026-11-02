"""Shared Isaac/MuJoCo observation path; no simulator imports.

This is a bounded 2.5D height-map sensor surrogate, not a beam LiDAR/ROS mapper.
Acquisition frames remain fixed: repeated reprojection cannot round away motion.
Newest observed cells across delivered packets win; missing cells retain their
own original timestamps. Quality uses measured geometry, never truth residuals.
"""
import math

import torch
import torch.nn.functional as F

from .contract import (GRID_ROWS, GRID_COLS, GRID_RESOLUTION, HEIGHT_DIM,
                       HEIGHT_OFFSET, HEIGHT_CLIP, student_dim)


def capture_grid_shape(padding_m=0.):
    extra = math.ceil(padding_m/GRID_RESOLUTION-1.e-7)
    return GRID_ROWS+2*extra, GRID_COLS+2*extra


def grid_points(device, padding_m=0.):
    rows,cols = capture_grid_shape(padding_m)
    x = (torch.arange(cols, device=device) - cols//2) * GRID_RESOLUTION
    y = (torch.arange(rows, device=device) - rows//2) * GRID_RESOLUTION
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx.flatten(), yy.flatten()), -1)


def map_output_mask(points, cfg):
    """Allowed map cells in the current gravity/yaw frame, from the front LiDAR.

    The front sector retains the existing horizontal FOV and front XY offset.
    Optional near exclusion uses planar radius; front_only sets it to zero.
    The mask itself does not change acquisition, RNG or map quality.
    """
    if not cfg.front_only:
        return torch.ones(points.shape[:-1], dtype=torch.bool, device=points.device)
    delta = points - points.new_tensor(cfg.front_sensor_offset_m[:2])
    angle = torch.atan2(delta[..., 1], delta[..., 0]).abs()
    distance = delta.norm(dim=-1)
    # Inclusive metric boundary despite float32 representation of grid centers.
    return ((distance > cfg.front_blind_distance_m + 1e-6)
            & (delta[..., 0] > 0.)
            & (angle <= math.radians(cfg.lidar_horizontal_fov_deg / 2) + 1e-7))


def quaternion_rotation(q):
    """wxyz body-to-world rotation, independent of simulator conventions."""
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    w, x, y, z = q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), -1).reshape(-1, 3, 3)


def query_grid(current, anchor, grid, source_shape=(GRID_ROWS, GRID_COLS)):
    """current/anchor [N,x,y,yaw]; query acquisition coordinates of current grid."""
    co, si = current[:, 2:3].cos(), current[:, 2:3].sin()
    wx = current[:, 0:1] + co*grid[:, 0] - si*grid[:, 1]
    wy = current[:, 1:2] + si*grid[:, 0] + co*grid[:, 1]
    dx, dy = wx-anchor[:, 0:1], wy-anchor[:, 1:2]
    co, si = anchor[:, 2:3].cos(), anchor[:, 2:3].sin()
    gx, gy = co*dx+si*dy, -si*dx+co*dy
    rows,cols = source_shape
    half_x,half_y = (cols-1)*GRID_RESOLUTION/2,(rows-1)*GRID_RESOLUTION/2
    inside = (gx.abs() <= half_x+1e-5) & (gy.abs() <= half_y+1e-5)
    return torch.stack((gx/half_x, gy/half_y), -1).reshape(-1, GRID_ROWS, GRID_COLS, 2), inside


def terrain_visibility(hits, base, yaw, rotation, cfg, source_shape=(GRID_ROWS, GRID_COLS),
                       surface_normals=None):
    """Same front/rear FOV and height-field line of sight in both simulators.

    No reflectivity, beam angular sampling, overhangs or robot self-shadow.
    Sensor extrinsics are configurable engineering placeholders until calibrated.
    """
    rows,cols = source_shape
    half_x,half_y = (cols-1)*GRID_RESOLUTION/2,(rows-1)*GRID_RESOLUTION/2
    finite = torch.isfinite(hits).all(-1)
    if cfg.support_surface_visibility:
        if surface_normals is None or surface_normals.shape != hits.shape:
            raise ValueError('Support-surface sensing requires measured mesh normals matching hits')
        norm = surface_normals.norm(dim=-1)
        normal_valid = torch.isfinite(surface_normals).all(-1) & (norm > 1.e-6)
        normals = torch.nan_to_num(surface_normals) / norm.nan_to_num().clamp_min(1.e-6)[..., None]
        # Downward rays select the upper face. Normalize importer winding.
        normals = torch.where(normals[..., 2:3] < 0, -normals, normals)
    safe = torch.where(finite[..., None], hits, base[:, None])
    visible = torch.zeros_like(finite)
    z = torch.where(finite, hits[..., 2], 0.).reshape(-1, 1, rows, cols)
    mask = finite.float().reshape_as(z)
    for offset, rear in ((cfg.front_sensor_offset_m, False), (cfg.rear_sensor_offset_m, True)):
        off = torch.as_tensor(offset, device=hits.device, dtype=hits.dtype)
        origin = base + (rotation @ off)
        delta = safe-origin[:, None]
        body = delta @ rotation
        horizontal = body[..., :2].norm(dim=-1)
        angle = torch.atan2(body[..., 1], -body[..., 0] if rear else body[..., 0]).abs()
        elevation = torch.atan2(body[..., 2], horizontal)
        distance = delta.norm(dim=-1)
        candidate = finite & (distance >= cfg.lidar_min_range_m) & (distance <= cfg.lidar_max_range_m)
        if cfg.support_surface_visibility:
            # Front-face visibility, not a height cutoff: visible rising ramps
            # may be above the sensor, whereas horizontal wall tops cannot be.
            candidate &= normal_valid & (normals[..., 2] > 1.e-6)
            candidate &= ((origin[:, None]-safe)*normals).sum(-1) > 1.e-6
        if cfg.lidar_horizontal_fov_deg < 360:
            candidate &= (horizontal > 1e-8) & (angle <= math.radians(cfg.lidar_horizontal_fov_deg/2)+1e-7)
        candidate &= (elevation >= math.radians(cfg.vertical_fov_min_deg)) & (elevation <= math.radians(cfg.vertical_fov_max_deg))
        if cfg.terrain_visibility:
            # Mid-segment samples exclude the endpoint and the sensor itself.
            fractions = torch.linspace(.05, .90, cfg.visibility_samples, device=hits.device)
            path = origin[:, None, None] + delta[:, :, None]*fractions[None, None, :, None]
            rel = path[..., :2]-base[:, None, None, :2]
            co, si = yaw.cos()[:, None, None], yaw.sin()[:, None, None]
            gx = co*rel[..., 0]+si*rel[..., 1]
            gy = -si*rel[..., 0]+co*rel[..., 1]
            query = torch.stack((gx/half_x, gy/half_y), -1)
            ground = F.grid_sample(z, query, mode="nearest", align_corners=True)[:, 0]
            known = F.grid_sample(mask, query, mode="nearest", align_corners=True)[:, 0] > .5
            blocked = known & (ground > path[..., 2]+cfg.visibility_tolerance_m)
            candidate &= ~blocked.any(-1)
        visible |= candidate
    return visible


def tilt_height_jacobian(points, base, rotation):
    """d(world z)/d(body roll,pitch) for R_hat=R_true*Exp(tilt)."""
    body = (points-base[:, None]) @ rotation
    row = rotation[:, 2]
    roll = row[:, 2:3]*body[..., 1]-row[:, 1:2]*body[..., 2]
    pitch = row[:, 0:1]*body[..., 2]-row[:, 2:3]*body[..., 0]
    return torch.stack((roll, pitch), -1)


def pose_height_variance(z, valid, grid, xy_std, yaw_std, tilt_std, tilt_jacobian_norm_sq=None,
                         confidence_step_ambiguity_weight=1.0, source_shape=(GRID_ROWS, GRID_COLS)):
    """Confidence variance from finite differences and weighted step ambiguity.

    Relative map/query pose covariance is diagonal in this surrogate. Actual
    mapper covariance must be propagated once, without double counting.
    Uses only measured heights with valid neighbors, including on stair edges.
    """
    n = z.shape[0]
    rows,cols = source_shape
    z, valid = z.reshape(n, rows, cols), valid.reshape(n, rows, cols)
    valid = valid & torch.isfinite(z)
    z = torch.where(valid, z, 0.)
    dx = (z[:, :, 1:]-z[:, :, :-1]) * (valid[:, :, 1:] & valid[:, :, :-1])
    dy = (z[:, 1:]-z[:, :-1]) * (valid[:, 1:] & valid[:, :-1])
    left, right = F.pad(dx, (1, 0)), F.pad(dx, (0, 1))
    down, up = F.pad(dy, (0, 0, 1, 0)), F.pad(dy, (0, 0, 0, 1))
    gx = torch.where(left.abs() > right.abs(), left, right).flatten(1)/GRID_RESOLUTION
    gy = torch.where(down.abs() > up.abs(), down, up).flatten(1)/GRID_RESOLUTION
    jyaw = -grid[:, 1]*gx + grid[:, 0]*gy
    linear = (gx.square()+gy.square())*xy_std[:, None].square() + jyaw.square()*yaw_std[:, None].square()
    radius = grid.norm(dim=-1)
    edge = torch.maximum(torch.maximum(left.abs(), right.abs()), torch.maximum(down.abs(), up.abs())).flatten(1)
    displacement = xy_std[:, None]+radius*yaw_std[:, None]
    ambiguity = edge.square() * (2*displacement/GRID_RESOLUTION).clamp(0, 1)
    tilt_geometry = radius.square() if tilt_jacobian_norm_sq is None else tilt_jacobian_norm_sq
    return linear + confidence_step_ambiguity_weight*ambiguity + tilt_geometry*tilt_std[:, None].square()


def confidence_from_quality(valid, variance, age, cfg):
    usable = valid.bool() & torch.isfinite(variance) & (variance >= 0)
    usable &= torch.isfinite(age) & (age >= 0) & (age <= cfg.max_map_age_s)
    safe_variance = torch.where(usable, variance, 0.)
    safe_age = torch.where(usable, age, 0.)
    score = torch.exp(-safe_age/cfg.confidence_age_tau_s)/(1+safe_variance/cfg.confidence_reference_std**2)
    return torch.where(usable, score, 0.)


def continuous_delivery_steps(requested_s, step_dt):
    """First control tick at or after a continuous delay, without an epsilon.

    Correct the division's rounding at exact tick boundaries by comparing actual
    tick times. Subtracting an epsilon would deliver some just-above-boundary
    requests early. Callers supply nonnegative finite float64 requested seconds.
    """
    ticks = torch.ceil(requested_s/step_dt).long()
    ticks -= ((ticks-1).to(requested_s.dtype)*step_dt >= requested_s).long()
    ticks += (ticks.to(requested_s.dtype)*step_dt < requested_s).long()
    return ticks


class StudentObservationState:
    def __init__(self, num_envs, device, step_dt, cfg, history_steps=1):
        cfg.validate()
        if not math.isclose(step_dt, .02, abs_tol=1e-7):
            raise ValueError("V1 contract requires a 20ms policy step")
        self.cfg, self.device, self.step_dt, self.num_envs = cfg, device, step_dt, num_envs
        self.history_steps, self.obs_dim = history_steps, student_dim(history_steps)
        self.grid = grid_points(device)
        self.capture_grid = grid_points(device,cfg.map_capture_padding_m)
        self.capture_shape = capture_grid_shape(cfg.map_capture_padding_m)
        rows,cols = self.capture_shape
        iy = torch.arange(rows,device=device)[rows//2-5:rows//2+6]
        ix = torch.arange(cols,device=device)[cols//2-8:cols//2+9]
        self.output_indices = (iy[:,None]*cols+ix[None,:]).flatten()
        self.output_mask = map_output_mask(self.grid, cfg)
        minimum_period = max(step_dt, cfg.map_update_period_s-cfg.update_jitter_s)
        self.packet_slots = math.ceil((cfg.max_map_age_s+cfg.max_delay_s)/minimum_period)+3
        shape = (self.packet_slots, num_envs, self.capture_grid.shape[0])
        self.packet_z = torch.zeros(shape, device=device)
        self.packet_valid = torch.zeros(shape, device=device, dtype=torch.bool)
        self.packet_variance = torch.zeros(shape, device=device)
        self.packet_anchor = torch.zeros(self.packet_slots, num_envs, 3, device=device)
        self.packet_capture = torch.full((self.packet_slots, num_envs), -1, device=device, dtype=torch.long)
        self.packet_arrival = self.packet_capture.clone()
        self.next_slot = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.next_capture = self.next_slot.clone()
        self.next_occlusion = self.next_slot.clone()
        self.blackout_until = self.next_slot.clone()
        self.bias = torch.zeros(num_envs, 6, device=device)  # z,x,y,yaw,roll,pitch
        self.drift = torch.zeros_like(self.bias)
        self.gyro_bias = torch.zeros(num_envs, 3, device=device)
        self.severity = torch.zeros(num_envs, device=device)
        self.mixture = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.occluded = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self.occlusion_center = torch.zeros(num_envs, 2, device=device)
        self.needs_reset = torch.ones(num_envs, device=device, dtype=torch.bool)
        self.zero_delivery = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self.zero_delivery_resets = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.delayed_delivery_resets = torch.zeros_like(self.zero_delivery_resets)
        self.zero_delivery_packets = torch.zeros_like(self.zero_delivery_resets)
        self.delayed_delivery_packets = torch.zeros_like(self.zero_delivery_resets)
        if cfg.continuous_delay:
            self.requested_delay_s = torch.zeros(num_envs, device=device, dtype=torch.float64)
            self.actual_delay_s = torch.zeros_like(self.requested_delay_s)
        self.history = torch.zeros(num_envs, history_steps-1, 57, device=device)
        self.capture = torch.full((num_envs,), -1, device=device, dtype=torch.long)
        self.cell_age = torch.zeros(num_envs, HEIGHT_DIM, device=device)
        self.capture_visibility = torch.zeros_like(self.packet_valid[0])
        self.last_visibility = torch.zeros(num_envs,HEIGHT_DIM,dtype=torch.bool,device=device)
        self.last_channels = None
        self.command_transform = None  # Optional sensor-only command feedback, v3.5.
        self.last_step, self.last_obs = None, None
        self.observed_step = torch.full((num_envs,), -1, device=device, dtype=torch.long)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.packet_capture[:, ids] = -1
        self.packet_arrival[:, ids] = -1
        self.packet_valid[:, ids] = False
        self.next_capture[ids] = 0
        self.next_occlusion[ids] = 0
        self.blackout_until[ids] = 0
        self.next_slot[ids] = 0
        self.drift[ids] = 0
        self.history[ids] = 0
        self.needs_reset[ids] = True
        self.capture[ids] = -1
        self.observed_step[ids] = -1
        if self.cfg.continuous_delay:
            self.requested_delay_s[ids] = 0.
            self.actual_delay_s[ids] = 0.
        self.last_step = None

    def _initialize(self):
        ids = self.needs_reset.nonzero().flatten()
        if not ids.numel():
            return
        cfg = self.cfg
        sample = torch.rand(ids.numel(), device=self.device)
        mixture = torch.where(sample < cfg.clean_probability, 0,
                              torch.where(sample > 1-cfg.stress_probability, 2, 1))
        self.mixture[ids] = mixture
        self.severity[ids] = torch.where(mixture == 0, 0., torch.where(mixture == 2, cfg.stress_multiplier, 1.))
        # Separate draw from clean/stress selection. Do not consume randomness
        # at p=0: old tasks/checkpoints retain their exact seeded behavior.
        if cfg.zero_delay_probability:
            self.zero_delivery[ids] = torch.rand(ids.numel(),device=self.device) < cfg.zero_delay_probability
        else:
            self.zero_delivery[ids] = False
        self.zero_delivery_resets[ids] += self.zero_delivery[ids].long()
        self.delayed_delivery_resets[ids] += (~self.zero_delivery[ids]).long()
        scale = torch.tensor([cfg.height_bias_std, cfg.xy_bias_std, cfg.xy_bias_std,
                              cfg.yaw_bias_std, cfg.roll_pitch_std, cfg.roll_pitch_std], device=self.device)
        self.bias[ids] = torch.randn(ids.numel(), 6, device=self.device)*scale
        self.gyro_bias[ids] = torch.randn(ids.numel(), 3, device=self.device)*cfg.gyro_bias_std_rad_s
        # Independent asynchronous sensor phase, including clean-mixture episodes.
        self.next_capture[ids] = torch.randint(max(1, math.ceil(cfg.map_update_period_s/self.step_dt)),
                                               (ids.numel(),), device=self.device)
        self.needs_reset[ids] = False

    def observe(self, clean_proprio, hits, base, yaw, rotation, step, *, surface_normals=None):
        if self.last_step == step and self.last_obs is not None:
            return self.last_obs.clone()
        if clean_proprio.shape != (self.num_envs, 57) or hits.shape != (self.num_envs, self.capture_grid.shape[0], 3):
            raise ValueError(f'Expected clean proprio [N,57], ordered capture hits [N,{len(self.capture_grid)},3]')
        if not all(torch.isfinite(v).all() for v in (clean_proprio, base, yaw, rotation)):
            raise ValueError("Nonfinite proprio/pose")
        self._initialize()
        cfg = self.cfg
        active = self.observed_step != step
        scale = self.severity*cfg.level
        alpha = math.exp(-self.step_dt/cfg.drift_time_constant_s)
        drift_std = torch.tensor([0., cfg.pose_drift_xy_std, cfg.pose_drift_xy_std,
                                  cfg.pose_drift_yaw_std, cfg.roll_pitch_std*.5,
                                  cfg.roll_pitch_std*.5], device=self.device)
        updated_drift = alpha*self.drift + torch.randn_like(self.drift)*drift_std*math.sqrt(1-alpha*alpha)
        self.drift[active] = updated_drift[active]
        pose_error = (self.bias+self.drift)*scale[:, None]
        self.capture_visibility = terrain_visibility(hits, base, yaw, rotation, cfg,self.capture_shape,
                                                     surface_normals=surface_normals)
        self.last_visibility = self.capture_visibility[:,self.output_indices]
        redraw = self.next_occlusion <= step
        self.occluded[redraw] = torch.rand(int(redraw.sum()), device=self.device) < (cfg.occlusion_probability*scale[redraw]).clamp_max(1)
        self.occlusion_center[redraw] = (torch.rand(int(redraw.sum()), 2, device=self.device)-.5)*torch.tensor([1.6, 1.], device=self.device)
        self.next_occlusion[redraw] = step+max(1, math.ceil(cfg.occlusion_hold_s/self.step_dt))
        acquire = self.next_capture <= step
        if acquire.any():
            ids = acquire.nonzero().flatten()
            period = cfg.map_update_period_s + (2*torch.rand(ids.numel(), device=self.device)-1)*cfg.update_jitter_s
            self.next_capture[ids] = step+torch.ceil(period/self.step_dt).long().clamp_min(1)
            burst = torch.rand(self.num_envs, device=self.device) < (cfg.burst_probability*scale).clamp_max(1)
            self.blackout_until[acquire & burst] = step+math.ceil(cfg.burst_duration_s/self.step_dt)
            accepted = acquire & (step >= self.blackout_until)
            accepted &= torch.rand(self.num_envs, device=self.device) >= (cfg.packet_dropout*scale).clamp_max(1)
            ids = accepted.nonzero().flatten()
            if ids.numel():
                slots = self.next_slot[ids] % self.packet_slots
                self.next_slot[ids] += 1
                valid = self.capture_visibility[ids].clone()
                z = torch.where(torch.isfinite(hits[ids, :, 2]), hits[ids, :, 2], 0.)
                radius = self.capture_grid.norm(dim=-1)
                std = cfg.height_noise_std*scale[ids, None]*(1+radius)
                std *= .5+torch.rand(ids.numel(), 1, device=self.device)
                coarse = torch.randn(ids.numel(), 1, 3, 5, device=self.device)
                correlated = F.interpolate(coarse, size=self.capture_shape, mode="bilinear", align_corners=True).flatten(1)
                z = z + torch.randn_like(z)*std + correlated*cfg.correlated_height_std*scale[ids, None]
                z += pose_error[ids, 0, None]
                # Same roll/pitch estimate error affects gravity and measured map slope.
                safe_hits = torch.where(torch.isfinite(hits[ids]).all(-1)[..., None], hits[ids], base[ids, None])
                points = safe_hits.clone()
                points[..., 2] = z
                jacobian = tilt_height_jacobian(points, base[ids], rotation[ids])
                z += (jacobian*pose_error[ids, None, 4:6]).sum(-1)
                delta = (self.capture_grid[None]-self.occlusion_center[ids, None]).abs()
                hole = (delta[..., 0] <= cfg.occlusion_width_m/2) & (delta[..., 1] <= cfg.occlusion_height_m/2)
                valid &= ~(hole & self.occluded[ids, None])
                valid &= torch.rand_like(z) >= (cfg.point_dropout*scale[ids, None]).clamp_max(1)
                artifact = torch.rand(ids.numel(), device=self.device) < (cfg.map_artifact_probability*scale[ids]).clamp_max(1)
                z += hole*artifact[:, None]*torch.randn(ids.numel(), 1, device=self.device)*cfg.map_artifact_height_std*scale[ids, None]
                # Artifact is intentionally not an oracle low-confidence flag.
                height_variance = std.square() + ((cfg.height_bias_std**2+cfg.correlated_height_std**2)*scale[ids, None].square())
                xy_std = math.hypot(cfg.xy_bias_std, cfg.pose_drift_xy_std)*scale[ids]
                yaw_std = math.hypot(cfg.yaw_bias_std, cfg.pose_drift_yaw_std)*scale[ids]
                measured_points = safe_hits.clone()
                measured_points[..., 2] = z
                quality_jacobian = tilt_height_jacobian(measured_points, base[ids], rotation[ids])
                quality_tilt = torch.nan_to_num(quality_jacobian).square().sum(-1)
                variance = height_variance + pose_height_variance(z, valid, self.capture_grid, xy_std, yaw_std,
                                                                  cfg.roll_pitch_std*math.sqrt(1.25)*scale[ids], quality_tilt,
                                                                  cfg.confidence_step_ambiguity_weight,self.capture_shape)
                jitter = torch.exp(torch.randn(ids.numel(), 1, device=self.device)*cfg.confidence_log_std*scale[ids, None])
                overestimate = torch.rand(ids.numel(), 1, device=self.device) < (cfg.confidence_overestimate_probability*scale[ids, None]).clamp_max(1)
                variance *= jitter*torch.where(overestimate, cfg.confidence_overestimate_factor, 1.)
                self.packet_z[slots, ids] = z
                self.packet_valid[slots, ids] = valid
                self.packet_variance[slots, ids] = variance
                self.packet_anchor[slots, ids, :2] = base[ids, :2]+pose_error[ids, 1:3]
                self.packet_anchor[slots, ids, 2] = yaw[ids]+pose_error[ids, 3]
                self.packet_capture[slots, ids] = step
                # Hardware timing remains present at level=0.
                if cfg.continuous_delay:
                    requested = torch.empty(ids.numel(), device=self.device, dtype=torch.float64).uniform_(cfg.min_delay_s, cfg.max_delay_s)
                    self.requested_delay_s[ids] = torch.where(self.zero_delivery[ids], 0., requested)
                    delay = continuous_delivery_steps(requested, self.step_dt)
                else:
                    delay = torch.randint(math.ceil(cfg.min_delay_s/self.step_dt-1e-9),
                                          math.ceil(cfg.max_delay_s/self.step_dt)+1,
                                          (ids.numel(),), device=self.device)
                delay = torch.where(self.zero_delivery[ids],torch.zeros_like(delay),delay)
                if cfg.continuous_delay:
                    self.actual_delay_s[ids] = delay.to(torch.float64)*self.step_dt
                self.packet_arrival[slots, ids] = step+delay
                self.zero_delivery_packets[ids] += (delay == 0).long()
                self.delayed_delivery_packets[ids] += (delay > 0).long()
        current = torch.cat((base[:, :2], yaw[:, None]), -1)
        current = current[None].expand(self.packet_slots, -1, -1).reshape(-1, 3)
        query, inside = query_grid(current, self.packet_anchor.flatten(0, 1), self.grid,self.capture_shape)
        packets = torch.stack((self.packet_z, self.packet_valid.float(), self.packet_variance), 2).reshape(-1, 3, *self.capture_shape)
        sampled = F.grid_sample(packets, query, mode="nearest", padding_mode="zeros", align_corners=True)
        sampled = sampled.reshape(self.packet_slots, self.num_envs, 3, HEIGHT_DIM)
        age = (step-self.packet_capture)*self.step_dt
        delivered = (self.packet_capture >= 0) & (self.packet_arrival <= step) & (age <= cfg.max_map_age_s)
        valid = (sampled[:, :, 1] > .5) & inside.reshape(self.packet_slots, self.num_envs, HEIGHT_DIM) & delivered[..., None]
        captures = self.packet_capture[..., None].expand_as(valid)
        latest, slot = torch.where(valid, captures, -1).max(0)
        env_index = torch.arange(self.num_envs, device=self.device)[:, None]
        cell_index = torch.arange(HEIGHT_DIM, device=self.device)[None]
        z = sampled[slot, env_index, 0, cell_index]
        variance = sampled[slot, env_index, 2, cell_index]
        self.cell_age = (step-latest)*self.step_dt
        # Reapply in the CURRENT grid after all delayed/cached reprojection.
        valid = (latest >= 0) & self.output_mask
        variance += self.cell_age*cfg.variance_growth_per_s*scale[:, None].square()
        confidence = confidence_from_quality(valid, variance, self.cell_age, cfg)
        height = (base[:, 2:3]-z-HEIGHT_OFFSET).clamp(*HEIGHT_CLIP)
        height = torch.where(valid, height, 0.)
        self.capture = torch.where(delivered, self.packet_capture, -1).max(0).values
        self.last_channels = (height, valid.float(), confidence)
        self.last_visibility &= self.output_mask
        proprio = clean_proprio.clone()
        proprio[:, :3] += (torch.randn(self.num_envs, 3, device=self.device)*cfg.gyro_std_rad_s+self.gyro_bias)*scale[:, None]*.25
        tilt = torch.cat((pose_error[:, 4:6], torch.zeros(self.num_envs, 1, device=self.device)), -1)
        # If estimated body-to-world orientation is R*Exp(tilt), mapped height
        # gains roll*y-pitch*x while projected gravity uses Exp(-tilt)*g_body.
        gravity = proprio[:, 3:6]-torch.cross(tilt, proprio[:, 3:6], dim=-1)
        proprio[:, 3:6] = gravity/gravity.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        proprio[:, 9:21] += torch.randn(self.num_envs, 12, device=self.device)*cfg.joint_position_std_rad*scale[:, None]
        proprio[:, 21:25] = 0.
        proprio[:, 25:41] += torch.randn(self.num_envs, 16, device=self.device)*cfg.joint_velocity_std_rad_s*scale[:, None]*.05
        if self.command_transform is not None:
            proprio[:, 6:9] = self.command_transform(clean_proprio[:, 6:9], yaw+pose_error[:, 3],
                proprio[:, 2]/.25, self.last_channels, step)
        output = torch.cat((proprio, *self.last_channels, self.history.flatten(1)), -1)
        if self.history_steps > 1:
            updated_history = torch.cat((self.history[:, 1:], proprio[:, None]), 1)
            self.history[active] = updated_history[active]
        if self.last_obs is not None:
            output[~active] = self.last_obs[~active]
            self.last_channels = (output[:, 57:244], output[:, 244:431], output[:, 431:618])
        if output.shape != (self.num_envs, self.obs_dim) or not torch.isfinite(output).all():
            raise RuntimeError("Invalid v1 observation")
        self.last_step, self.last_obs = step, output.clone()
        self.observed_step.fill_(step)
        return output


class ActionDelayQueue:
    """Issued action history and actuator delay shared by training and deployment."""
    def __init__(self, num_envs, device, delay_steps):
        self.delay_steps = delay_steps
        self.queue = torch.zeros(delay_steps+1, num_envs, 16, device=device)
        self.cursor = 0
        self.previous_issued = torch.zeros(num_envs, 16, device=device)

    def push(self, issued):
        self.queue[self.cursor] = issued
        applied = self.queue[(self.cursor+1) % len(self.queue)].clone()
        self.cursor = (self.cursor+1) % len(self.queue)
        self.previous_issued.copy_(issued)
        return applied

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.queue[:, ids] = 0.
        self.previous_issued[ids] = 0.
