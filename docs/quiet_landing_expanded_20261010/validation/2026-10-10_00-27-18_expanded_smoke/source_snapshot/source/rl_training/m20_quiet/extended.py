"""Pure Torch quiet extensions. Static terrain, wheel-bottom contact approximation.

Event costs use incremental squared peak/impulse, so a window spanning policy
steps is neither paid twice nor lost if the episode terminates before it closes.
All other costs are time integrals and are averaged by the reward adapter.
"""
from dataclasses import dataclass, field
import math

import torch


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

    def validate(self):
        for name, value in vars(self).items():
            if name == "weights":
                continue
            if not math.isfinite(value) or ("weight" not in name and value <= 0):
                raise ValueError(f"Invalid extended quiet setting {name}")
        if any(not math.isfinite(w) or w > 0 for w in self.weights.values()):
            raise ValueError("Quiet cost weights must be finite and nonpositive")
        if self.action_leg_weight > 0 or self.action_wheel_weight > 0:
            raise ValueError("Action penalties must be nonpositive")


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
        self.last = {}

    def begin_step(self):
        for value in self.costs.values():
            value.zero_()

    def update(self, *, event, contact, eligible, force, normals, wheel_velocity,
               wheel_omega, radius, wheel_acceleration, base_omega, base_velocity,
               torque, clearance, map_valid, swing_gate, dt):
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
        self.costs["impact_peak"] += (square(peak/s.force_reference)-square(self.peak/s.force_reference)).sum(-1)
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
        self.previous_force.copy_(normal_force)
        self.previous_acc.copy_(wheel_acceleration)
        self.previous_omega.copy_(base_omega)
        self.previous_velocity.copy_(base_velocity)
        self.previous_torque.copy_(torque)
        self.has_history.fill_(True)

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
