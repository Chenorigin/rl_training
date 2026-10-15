"""Shared M20 contact, landing and idle signals; no Isaac imports.

Used by quiet-landing/v2 training and MuJoCo measurements.
Keep all simulator-independent accounting in this one MDP module.
"""
from dataclasses import dataclass, field
import math

import torch


# Touchdown events and terrain clearance.
LEGS = ("fl", "fr", "hl", "hr")


@dataclass
class QuietLandingSettings:
    # Initial conservative settings, to be calibrated against the pinned baseline.
    velocity_reference: float = 0.5
    touchdown_weight: float = -0.02
    contact_on_force: float = 5.0
    contact_off_force: float = 2.0
    minimum_air_time: float = 0.01
    reset_grace_time: float = 0.1
    tread_force_ratio: float = 0.5
    wheel_radius: float = 0.09
    nearest_map_distance: float = 0.12
    approach_clearance: float = 0.06
    histogram_bins: int = 80
    histogram_max_speed: float = 4.0

    def validate(self):
        values = vars(self)
        if not all(math.isfinite(v) for v in values.values()):
            raise ValueError("Quiet landing settings must be finite")
        if (self.velocity_reference <= 0 or self.touchdown_weight > 0
                or not 0 <= self.contact_off_force < self.contact_on_force
                or self.minimum_air_time <= 0 or self.reset_grace_time < 0
                or not 0 <= self.tread_force_ratio <= 1
                or self.wheel_radius <= 0 or self.nearest_map_distance <= 0
                or self.approach_clearance <= 0 or self.histogram_bins < 2
                or self.histogram_max_speed <= 0):
            raise ValueError("Invalid quiet landing settings")


def map_clearance(wheels, hits, settings):
    """Nearest original scan hit; invalid data stays invalid, never zero clearance."""
    finite = torch.isfinite(hits).all(-1)
    distance = torch.linalg.vector_norm(wheels[:, :, None, :2] - hits[:, None, :, :2], dim=-1)
    distance = torch.where(finite[:, None], distance, torch.inf)
    nearest, index = distance.min(-1)
    ground = hits[..., 2].gather(1, index)
    valid = (nearest <= settings.nearest_map_distance) & torch.isfinite(ground)
    clearance = wheels[..., 2] - ground - settings.wheel_radius
    return torch.where(valid, clearance, torch.nan), valid


class TouchdownTracker:
    """Bounded per-episode histograms, reset-safe edges, per-policy-step costs.

    Call begin_step once, then update for every physics substep with velocities
    saved BEFORE advancing physics and forces obtained AFTER advancing physics.
    Initial settling is excluded by reset_grace_time, independently per env.
    Ground/map validity is diagnostic only; real touchdowns are never map-gated.
    """
    def __init__(self, num_envs, device, settings=None):
        self.settings = settings or QuietLandingSettings()
        self.settings.validate()
        shape = (num_envs, 4)
        self.contact = torch.zeros(shape, dtype=torch.bool, device=device)
        self.initialized = torch.zeros(shape, dtype=torch.bool, device=device)
        self.air_time = torch.zeros(shape, device=device)
        self.age = torch.zeros(num_envs, device=device)
        self.last_event = torch.zeros(shape, dtype=torch.bool, device=device)
        self.last_speed = torch.zeros(shape, device=device)
        self.last_pre_velocity = torch.zeros((*shape, 3), device=device)
        self.step_cost = torch.zeros(num_envs, device=device)
        self.step_events = torch.zeros(shape, device=device)
        self.count = torch.zeros(shape, device=device)
        self.speed_sum = torch.zeros(shape, device=device)
        self.speed_sq_sum = torch.zeros(shape, device=device)
        self.speed_max = torch.zeros(shape, device=device)
        self.raw_edges = torch.zeros(shape, device=device)
        self.non_tread_edges = torch.zeros(shape, device=device)
        self.map_samples = torch.zeros(shape, device=device)
        self.valid_map_samples = torch.zeros(shape, device=device)
        self.approach_samples = torch.zeros(shape, device=device)
        self.event_in_window = torch.zeros(shape, device=device)
        self.histogram = torch.zeros((*shape, self.settings.histogram_bins), device=device)
        self.distance = torch.zeros(num_envs, device=device)

    def begin_step(self):
        self.step_cost.zero_()
        self.step_events.zero_()

    def update(self, pre_velocity, post_force, dt, clearance=None, valid=None, normals=None):
        if pre_velocity.shape != self.last_pre_velocity.shape or post_force.shape != pre_velocity.shape:
            raise ValueError("Wheel velocity and force must have shape [env, 4, 3]")
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("Physics dt must be positive")
        if not torch.isfinite(pre_velocity).all() or not torch.isfinite(post_force).all():
            raise ValueError("Non-finite quiet landing physics sample")
        magnitude = torch.linalg.vector_norm(post_force, dim=-1)
        contact = torch.where(self.contact, magnitude > self.settings.contact_off_force,
                              magnitude >= self.settings.contact_on_force)
        edge = contact & ~self.contact & self.initialized
        eligible = self.age[:, None] >= self.settings.reset_grace_time
        self.raw_edges += (edge & eligible).float()
        if normals is None:
            normals = torch.zeros_like(pre_velocity)
            normals[..., 2] = 1
        if normals.shape != pre_velocity.shape or not torch.isfinite(normals).all():
            raise ValueError("Invalid estimated surface normals")
        normal_force = (post_force * normals).sum(-1)
        tread = normal_force >= self.settings.tread_force_ratio * magnitude
        event = edge & tread & eligible & (self.air_time >= self.settings.minimum_air_time)
        self.non_tread_edges += (edge & ~tread & eligible).float()
        speed = (-(pre_velocity * normals).sum(-1)).clamp_min(0)
        self.last_event.copy_(event)
        self.last_pre_velocity.copy_(pre_velocity)
        self.last_speed.copy_(torch.where(event, speed, self.last_speed))
        self.step_cost += (event * (speed / self.settings.velocity_reference).square()).sum(-1)
        self.step_events += event.float()
        self.count += event.float()
        self.speed_sum += event * speed
        self.speed_sq_sum += event * speed.square()
        self.speed_max = torch.maximum(self.speed_max, torch.where(event, speed, 0.0))
        bins = (speed / self.settings.histogram_max_speed * self.settings.histogram_bins).long()
        bins = bins.clamp(0, self.settings.histogram_bins - 1)
        self.histogram.scatter_add_(-1, bins[..., None], event.float()[..., None])
        if clearance is not None and valid is not None:
            approaching = valid & torch.isfinite(clearance) & (clearance <= self.settings.approach_clearance)
            approaching &= (clearance >= -self.settings.approach_clearance) & (speed > 0) & ~self.contact
            self.map_samples += eligible.float()
            self.valid_map_samples += (valid & eligible).float()
            self.approach_samples += (approaching & eligible).float()
            self.event_in_window += (event & approaching).float()
        self.air_time = torch.where(contact, 0.0, self.air_time + dt)
        self.contact.copy_(contact)
        self.initialized.fill_(True)
        self.age += dt
        return event, speed

    def metrics(self, env_ids):
        """Only emit conditional values when samples exist; counts remain explicit."""
        result = {}
        width = self.settings.histogram_max_speed / self.settings.histogram_bins
        for leg, indices in [("all", slice(None))] + [(name, i) for i, name in enumerate(LEGS)]:
            count = self.count[env_ids, indices].sum()
            prefix = f"QuietLanding/{leg}/"
            result[prefix + "touchdown_count"] = count
            result[prefix + "raw_contact_edges"] = self.raw_edges[env_ids, indices].sum()
            result[prefix + "non_tread_edges"] = self.non_tread_edges[env_ids, indices].sum()
            result[prefix + "has_samples"] = (count > 0).float()
            samples = self.map_samples[env_ids, indices].sum()
            if samples > 0:
                result[prefix + "map_valid_fraction"] = self.valid_map_samples[env_ids, indices].sum() / samples
            if count > 0:
                result[prefix + "pre_speed_mean_m_s"] = self.speed_sum[env_ids, indices].sum() / count
                result[prefix + "pre_speed_max_m_s"] = self.speed_max[env_ids, indices].amax()
                result[prefix + "specific_energy_mean_J_kg"] = 0.5 * self.speed_sq_sum[env_ids, indices].sum() / count
                result[prefix + "height_window_event_coverage"] = self.event_in_window[env_ids, indices].sum() / count
                hist = self.histogram[env_ids, indices].reshape(-1, self.settings.histogram_bins).sum(0)
                for label, quantile in (("p50", 0.5), ("p95", 0.95)):
                    index = torch.searchsorted(hist.cumsum(0), count * quantile).clamp_max(len(hist) - 1)
                    result[prefix + f"pre_speed_{label}_hist_m_s"] = (index + 0.5) * width
                result[prefix + "histogram_tail_count"] = hist[-1]
        distance = self.distance[env_ids].sum()
        result["QuietLanding/distance_m"] = distance
        if distance > 0:
            result["QuietLanding/events_per_m"] = self.count[env_ids].sum() / distance
            result["QuietLanding/specific_energy_per_m_J_kg_m"] = 0.5 * self.speed_sq_sum[env_ids].sum() / distance
        return result

    def reset(self, env_ids):
        for value in vars(self).values():
            if isinstance(value, torch.Tensor):
                value[env_ids] = 0


def touchdown_cost(env):
    # Isaac's RewardManager multiplies by step_dt: cancel it for discrete events.
    return env.quiet_tracker.step_cost / env.step_dt


# Impact, smoothness, support and stationary objectives.
@dataclass
class ExtendedQuietSettings:
    # Candidate scales/weights, not acoustically calibrated optimal parameters.
    weights: dict = field(default_factory=lambda: {
        "impact_peak": -0.005, "impact_impulse": -0.005, "loading_rate": -0.02,
        "wheel_jerk": -0.02, "base_angular_acceleration": -0.02,
        "base_vertical_acceleration": -0.02, "torque_rate": -0.01,
        "wheel_slip": -0.02, "swing_clearance": -0.1,
    })
    window_seconds: float = 0.04
    allowed_normal_force: float = 100.0
    force_reference: float = 200.0
    impulse_reference: float = 5.0
    loading_reference: float = 20000.0
    jerk_reference: float = 10000.0
    angular_acceleration_reference: float = 50.0
    vertical_acceleration_reference: float = 50.0
    torque_rate_reference: float = 2000.0
    slip_reference: float = 0.5
    clearance_target: float = 0.04
    map_fit_radius: float = 0.18
    map_surface_tolerance: float = 0.025
    stair_height_threshold: float = 0.08
    cost_cap: float = 4.0
    action_leg_weight: float = -0.01
    action_wheel_weight: float = -0.01
    impact_peak_mode: str = "capped_square"
    load_balance_window: float = 0.5
    load_balance_settle: float = 0.15
    load_balance_deadband: float = 0.05
    load_balance_reference: float = 0.2
    load_balance_flat_range: float = 0.025
    load_balance_vertical_speed: float = 0.1
    load_balance_angular_speed: float = 0.3
    load_balance_min_total_force: float = 100.0

    def validate(self):
        expected = {"impact_peak", "impact_impulse", "loading_rate", "wheel_jerk",
                    "base_angular_acceleration", "base_vertical_acceleration", "torque_rate",
                    "wheel_slip", "swing_clearance"}
        if set(self.weights) not in (expected, expected | {"load_balance"}):
            raise ValueError("Keep all extended reward keys; use weight=0 to disable a term")
        for name, value in vars(self).items():
            if name in ("weights", "impact_peak_mode"):
                continue
            if not math.isfinite(value) or ("weight" not in name and value <= 0):
                raise ValueError(f"Invalid extended quiet setting {name}")
        if any(not math.isfinite(w) or w > 0 for w in self.weights.values()):
            raise ValueError("Quiet cost weights must be finite and nonpositive")
        if self.action_leg_weight > 0 or self.action_wheel_weight > 0:
            raise ValueError("Action penalties must be nonpositive")
        if self.impact_peak_mode not in ("capped_square", "quadratic_linear"):
            raise ValueError("Unknown impact peak mode")
        if not 0 < self.load_balance_deadband < 1:
            raise ValueError("Load balance deadband must be in (0,1)")

    def peak_cost(self, normalized_excess):
        if self.impact_peak_mode == "capped_square":
            return normalized_excess.square().clamp_max(self.cost_cap)
        # Match V1 below its saturation knee, then continue with the tangent.
        # Continuous value/slope, no flat tail, linear growth limits outliers.
        knee = math.sqrt(self.cost_cap)
        return normalized_excess.clamp_max(knee).square() + 2*knee*(normalized_excess-knee).clamp_min(0)


def surface_normals(wheels, hits, settings):
    """Fit z=ax+by+c to nearby same-height scanner points; vertical fallback.

Filtering by height relative to the nearest hit prevents fitting a stair edge
as a fictitious ramp. A missing/collinear fit has an explicit false validity.
This is a scan estimate, not the simulator's actual contact normal.
"""
    finite = torch.isfinite(hits).all(-1)
    clean = torch.where(finite[..., None], hits, 0.0)
    offset = clean[:, None] - wheels[:, :, None]
    dist = torch.linalg.vector_norm(offset[..., :2], dim=-1)
    nearest, idx = torch.where(finite[:, None], dist, torch.inf).min(-1)
    nearest_z = clean[..., 2].gather(1, idx)
    mask = finite[:, None] & (dist <= settings.map_fit_radius)
    mask &= (clean[:, None, :, 2] - nearest_z[..., None]).abs() <= settings.map_surface_tolerance
    count = mask.sum(-1)
    mean = (offset * mask[..., None]).sum(-2) / count.clamp_min(1)[..., None]
    centered = (offset - mean[..., None, :]) * mask[..., None]
    x, y, z = centered.unbind(-1)
    xx, yy, xy = x.square().sum(-1), y.square().sum(-1), (x*y).sum(-1)
    xz, yz = (x*z).sum(-1), (y*z).sum(-1)
    det = xx*yy - xy.square()
    valid = (count >= 3) & (det > 1e-10) & (nearest <= settings.map_fit_radius)
    safe = det.clamp_min(1e-10)
    a, b = (xz*yy-yz*xy)/safe, (yz*xx-xz*xy)/safe
    normals = torch.stack((-a, -b, torch.ones_like(a)), -1)
    normals = normals / torch.linalg.vector_norm(normals, dim=-1, keepdim=True)
    fallback = torch.zeros_like(normals)
    fallback[..., 2] = 1
    return torch.where(valid[..., None], normals, fallback), valid


class ExtendedQuietTracker:
    def __init__(self, n, device, settings=None):
        self.settings = settings or ExtendedQuietSettings()
        self.settings.validate()
        shape = (n, 4)
        self.remaining = torch.zeros(shape, device=device)
        self.peak = torch.zeros(shape, device=device)
        self.impulse = torch.zeros(shape, device=device)
        self.previous_force = torch.zeros(shape, device=device)
        self.previous_acc = torch.zeros(shape, device=device)
        self.previous_omega = torch.zeros((n, 3), device=device)
        self.previous_velocity = torch.zeros((n, 3), device=device)
        self.previous_torque = torch.zeros((n, 16), device=device)
        self.has_history = torch.zeros(n, dtype=torch.bool, device=device)
        self.costs = {name: torch.zeros(n, device=device) for name in self.settings.weights}
        self.sample_time = torch.zeros(n, device=device)
        self.force_peak = torch.zeros(n, device=device)
        self.impulse_peak = torch.zeros(n, device=device)
        self.slip_integral = torch.zeros(n, device=device)
        self.loading_peak = torch.zeros(n, device=device)
        self.last = {}
        self.load_force_integral = torch.zeros(shape, device=device)
        self.load_contact_integral = torch.zeros(shape, device=device)
        self.balance_force_mean = torch.zeros(shape, device=device)
        self.balance_age = torch.zeros(n, device=device)
        self.balance_time = torch.zeros(n, device=device)
        self.balance_signed_integral = torch.zeros(n, device=device)
        self.balance_abs_integral = torch.zeros(n, device=device)

    def begin_step(self):
        for value in self.costs.values():
            value.zero_()

    def update(self, *, event, contact, eligible, force, normals, wheel_velocity,
               wheel_omega, radius, wheel_acceleration, base_omega, base_velocity,
               torque, clearance, map_valid, swing_gate, dt, balance_gate=None):
        values = (force, normals, wheel_velocity, wheel_omega, wheel_acceleration,
                  base_omega, base_velocity, torque)
        if dt <= 0 or not math.isfinite(dt) or any(not torch.isfinite(v).all() for v in values):
            raise ValueError("Invalid expanded quiet physics sample")
        s = self.settings
        square = lambda x: x.square().clamp_max(s.cost_cap)
        normal_force = (force * normals).sum(-1).clamp_min(0)
        self.remaining = torch.where(event, s.window_seconds, self.remaining)
        self.peak = torch.where(event, 0.0, self.peak)
        self.impulse = torch.where(event, 0.0, self.impulse)
        window = (self.remaining > 0) & contact & eligible[:, None]
        window_dt = self.remaining.clamp(min=0, max=dt) * window
        excess = (normal_force-s.allowed_normal_force).clamp_min(0)
        peak = torch.maximum(self.peak, torch.where(window, excess, 0.0))
        impulse = self.impulse + excess * window_dt
        self.costs["impact_peak"] += (s.peak_cost(peak/s.force_reference)-s.peak_cost(self.peak/s.force_reference)).sum(-1)
        self.costs["impact_impulse"] += (square(impulse/s.impulse_reference)-square(self.impulse/s.impulse_reference)).sum(-1)
        loading = ((normal_force-self.previous_force)/dt).clamp_min(0)
        self.costs["loading_rate"] += (square(loading/s.loading_reference)*window_dt).sum(-1)
        self.peak, self.impulse = peak, impulse
        self.remaining = (self.remaining-dt).clamp_min(0)
        # Compute differences in the world frame, not between rotating frames.
        history = eligible & self.has_history
        jerk = (wheel_acceleration-self.previous_acc)/dt
        angular = (base_omega-self.previous_omega)/dt
        vertical = (base_velocity[:, 2]-self.previous_velocity[:, 2])/dt
        torque_rate = (torque-self.previous_torque)/dt
        for name, value, reference in (
            ("wheel_jerk", jerk, s.jerk_reference),
            ("base_angular_acceleration", angular[:, :2], s.angular_acceleration_reference),
            ("torque_rate", torque_rate, s.torque_rate_reference),
        ):
            self.costs[name] += square(value/reference).mean(-1)*history*dt
        self.costs["base_vertical_acceleration"] += square(vertical/s.vertical_acceleration_reference)*history*dt
        # Static ground; approximate contact point at the circular wheel bottom.
        point_velocity = wheel_velocity + torch.linalg.cross(wheel_omega, -radius*normals, dim=-1)
        tangent = point_velocity - (point_velocity*normals).sum(-1, keepdim=True)*normals
        slip = torch.linalg.vector_norm(tangent, dim=-1)
        self.costs["wheel_slip"] += (square(slip/s.slip_reference)*contact*eligible[:, None]).sum(-1)*dt
        valid = map_valid & torch.isfinite(clearance)
        deficit = (s.clearance_target-torch.nan_to_num(clearance)).clamp_min(0)/s.clearance_target
        swing = swing_gate & ~contact & valid & eligible[:, None]
        self.costs["swing_clearance"] += (square(deficit)*swing).sum(-1)*dt
        self.last = {"normal_force_N": normal_force, "loading_rate_N_s": loading,
                     "window_peak_excess_N": peak, "window_excess_impulse_N_s": impulse,
                     "slip_m_s": slip, "wheel_jerk_rad_s3": jerk,
                     "base_angular_acceleration_rad_s2": angular,
                     "base_vertical_acceleration_m_s2": vertical, "torque_rate_Nm_s": torque_rate}
        self.sample_time += eligible*dt
        self.force_peak = torch.maximum(self.force_peak, (normal_force*window).amax(-1))
        self.impulse_peak = torch.maximum(self.impulse_peak, (impulse*window).amax(-1))
        self.loading_peak = torch.maximum(self.loading_peak, (loading*window).amax(-1))
        self.slip_integral += (slip*contact*eligible[:, None]).mean(-1)*dt
        self.previous_force.copy_(normal_force)
        self.load_force_integral += normal_force*eligible[:, None]*dt
        self.load_contact_integral += contact*eligible[:, None]*dt
        # No gate supplied (e.g. legacy MuJoCo adapter) means no balance reward.
        gate = torch.zeros_like(eligible) if balance_gate is None else balance_gate & eligible
        gate &= contact.all(-1) & ~event.any(-1)
        gate &= (normal_force > 5).all(-1) & (normal_force.sum(-1) >= s.load_balance_min_total_force)
        gate &= base_velocity[:, 2].abs() <= s.load_balance_vertical_speed
        gate &= torch.linalg.vector_norm(base_omega, dim=-1) <= s.load_balance_angular_speed
        was_stable = self.balance_age > 0
        self.balance_age = torch.where(gate, self.balance_age+dt, 0.0)
        alpha = -math.expm1(-dt/s.load_balance_window)
        smoothed = self.balance_force_mean + alpha*(normal_force-self.balance_force_mean)
        self.balance_force_mean = torch.where((gate & was_stable)[:, None], smoothed,
                                             torch.where(gate[:, None], normal_force, 0.0))
        mean = self.balance_force_mean
        diagonal = (mean[:, 0]+mean[:, 3]-mean[:, 1]-mean[:, 2])/mean.sum(-1).clamp_min(1e-6)
        active = gate & (self.balance_age >= s.load_balance_settle)
        if "load_balance" in self.costs:
            deficit = (diagonal.abs()-s.load_balance_deadband).clamp_min(0)/s.load_balance_reference
            self.costs["load_balance"] += square(deficit)*active*dt
        self.balance_time += active*dt
        self.balance_signed_integral += diagonal*active*dt
        self.balance_abs_integral += diagonal.abs()*active*dt
        self.last.update(load_diagonal_ratio=diagonal, load_balance_active=active)
        self.previous_acc.copy_(wheel_acceleration)
        self.previous_omega.copy_(base_omega)
        self.previous_velocity.copy_(base_velocity)
        self.previous_torque.copy_(torque)
        self.has_history.fill_(True)

    def metrics(self, ids):
        time = self.sample_time[ids].sum()
        if time <= 0:
            return {}
        result = {"QuietLanding/impact_window_force_max_N": self.force_peak[ids].amax(),
                "QuietLanding/impact_window_excess_impulse_max_N_s": self.impulse_peak[ids].amax(),
                "QuietLanding/impact_window_loading_rate_max_N_s": self.loading_peak[ids].amax(),
                "QuietLanding/contact_gated_slip_time_mean_m_s": self.slip_integral[ids].sum()/time}
        for i, leg in enumerate(("fl", "fr", "hl", "hr")):
            result[f"QuietLoad/{leg}/mean_normal_force_N"] = self.load_force_integral[ids, i].sum()/time
            result[f"QuietLoad/{leg}/contact_fraction"] = self.load_contact_integral[ids, i].sum()/time
        stable_time = self.balance_time[ids].sum()
        result["QuietLoad/stable_support_fraction"] = stable_time/time
        result["QuietLoad/stable_has_samples"] = (stable_time > 0).float()
        if stable_time > 0:
            result["QuietLoad/stable_diagonal_signed_mean"] = self.balance_signed_integral[ids].sum()/stable_time
            result["QuietLoad/stable_diagonal_abs_mean"] = self.balance_abs_integral[ids].sum()/stable_time
        return result

    def reset(self, ids):
        for value in vars(self).values():
            if isinstance(value, torch.Tensor):
                value[ids] = 0
        for value in self.costs.values():
            value[ids] = 0
        for value in self.last.values():
            value[ids] = 0


def extended_cost(env, name):
    # All costs are already either event increments or physical-time integrals.
    return env.quiet_extended_tracker.costs[name] / env.step_dt


def action_group_rate(env, group):
    ids = env._quiet_wheel_action_ids if group == "wheel" else env._quiet_leg_action_ids
    difference = env.action_manager.action[:, ids]-env.action_manager.prev_action[:, ids]
    # Equal mean-square pressure per group, independent of 12-versus-4 sizes.
    return difference.square().mean(-1)


def idle_stance_context(env, command_threshold=0.05, flat_range=0.025):
    """Zero requested motion on a known flat map; never gate on achieved speed.

    Gating on low measured speed/tilt would let the policy evade the standing
    penalty by moving or leaning. The loaded mask selects diagnostic samples;
    missing wheel support is penalized separately rather than disabling costs.
    """
    command = env.command_manager.get_command("base_velocity")
    hits = env.scene.sensors["height_scanner"].data.ray_hits_w
    finite = torch.isfinite(hits).all((-1, -2))
    z = torch.nan_to_num(hits[..., 2])
    flat = finite & ((z.amax(-1)-z.amin(-1)) <= flat_range)
    gate = flat & (command.abs().amax(-1) < command_threshold)
    gate &= env.quiet_tracker.age >= env.cfg.quiet_landing.reset_grace_time
    robot = env.scene["robot"]
    force = env.quiet_extended_tracker.last["normal_force_N"]
    loaded = gate & (force > 5).all(-1) & (force.sum(-1) >= 100)
    diagonal = (force[:, 0]+force[:, 3]-force[:, 1]-force[:, 2])/force.sum(-1).clamp_min(1e-6)
    return robot, gate, loaded, diagonal


def idle_stance_cost(env, component, command_threshold=0.05, flat_range=0.025,
                     linear_reference=0.05, yaw_reference=0.1,
                     joint_reference=0.1, wheel_reference=0.5,
                     balance_deadband=0.03, balance_reference=0.1):
    """Explicit stationary objective, independent of the locomotion balance gate."""
    robot, gate, _, diagonal = idle_stance_context(env, command_threshold, flat_range)
    cap = lambda x: x.square().clamp_max(4.0)
    if component == "motion":
        cost = cap(torch.linalg.vector_norm(robot.data.root_lin_vel_w[:, :2], dim=-1)/linear_reference)
        cost += cap(robot.data.root_ang_vel_w[:, 2]/yaw_reference)
    elif component == "pose":
        ids = env._quiet_leg_joint_ids
        delta = robot.data.joint_pos[:, ids]-robot.data.default_joint_pos[:, ids]
        cost = cap(delta/joint_reference).mean(-1)
        cost += cap(robot.data.joint_vel[:, env._quiet_wheel_joint_ids]/wheel_reference).mean(-1)
    elif component == "load":
        cost = cap((diagonal.abs()-balance_deadband).clamp_min(0)/balance_reference)
        # A lifted/unloaded wheel must not switch the entire penalty off.
        # Each fully unloaded wheel costs as much as a saturated diagonal
        # deficit; normal nonzero-command stepping is excluded by `gate`.
        force = env.quiet_extended_tracker.last["normal_force_N"]
        missing = ((5.0-force).clamp(min=0, max=5.0)/5.0).square().sum(-1)*4.0
        return (cost*(force.sum(-1) >= 100)+missing)*gate
    else:
        raise ValueError(f"Unknown idle stance component {component}")
    return cost*gate


# Historical 244D idle/stop fitting mask.
def idle_mask(policy, command_threshold=0.05, flat_range=0.025):
    """Select finite, unsaturated flat scans with a requested zero command.

    Achieved speed, contact and support do not disable the stopping target.
    This function is used during fitting, never in the deployment control loop.
    """
    if policy.ndim != 2 or policy.shape[1] != 244:
        raise ValueError("Idle stopping requires the V1 244D policy layout")
    command = policy[:, 6:9]
    scan = policy[:, 57:244]
    known = torch.isfinite(policy).all(-1) & (scan.abs() < 1.0).all(-1)
    flat = torch.nan_to_num(scan).amax(-1) - torch.nan_to_num(scan).amin(-1) <= flat_range
    return known & flat & (command.abs().amax(-1) < command_threshold)
