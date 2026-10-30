"""Shared bounded landing reference controller, independent of either simulator.

This controls mechanical excitation, not sound pressure. Default numbers are
prototype hypotheses, not identified M20 parameters or a stability certificate.
Inputs use FL/FR/HL/HR order, SI units, and outward tread normals. The controller
only modifies leg references and gains; nominal policy action history is intact.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import torch


@dataclass
class LandingSettings:
    mode: str = "combined"
    landing_speed_m_s: float = 0.15
    available_deceleration_m_s2: float = 2.0
    latency_budget_s: float = 0.03
    clearance_margin_m: float = 0.015
    approach_distance_m: float = 0.09
    max_approach_distance_m: float = 0.18
    reference_response_s: float = 0.08
    max_cartesian_correction_m: float = 0.015
    max_joint_correction_rad: float = 0.04
    correction_rate_rad_s: float = 2.0
    jacobian_regularization_m: float = 0.01
    minimum_jacobian_norm_m: float = 0.015
    closing_threshold_m_s: float = 0.05
    penetration_tolerance_m: float = 0.025
    loading_reference_allowance_m: float = 0.003
    landing_kp_scale: float = 0.85
    landing_kd_scale: float = 1.0
    gain_rate_per_s: float = 3.0
    loading_time_s: float = 0.08
    contact_on_force_N: float = 12.0
    contact_off_force_N: float = 4.0
    contact_confirm_s: float = 0.01
    minimum_air_time_s: float = 0.025
    startup_grace_s: float = 0.1
    minimum_other_supports: int = 2
    terrain_max_distance_m: float = 0.12
    terrain_max_height_spread_m: float = 0.06
    # Match the current student map's declared expiry. A 0.10 s cutoff would
    # silently disable every ordinary 0.28-0.42 s delayed student map.
    terrain_max_age_s: float = 0.50
    # Retain previously observed static treads for rear wheels outside the
    # forward sensor view. This is measured sample age, not time since delivery.
    terrain_memory_s: float = 2.00
    terrain_memory_update_s: float = 0.10
    terrain_memory_frames: int = 16
    terrain_pose_jump_m: float = 0.25
    terrain_yaw_jump_rad: float = 0.35
    terrain_memory_cell_m: float = 0.025
    contact_delay_steps: int = 0
    contact_dropout_probability: float = 0.0
    contact_max_age_s: float = 0.04
    command_active_threshold: float = 0.05

    def validate(self):
        if self.mode not in ("off", "precontact", "phase", "combined"):
            raise ValueError("Unknown landing control mode")
        for name, value in asdict(self).items():
            if name != "mode" and (isinstance(value, bool) or not math.isfinite(value) or value < 0):
                raise ValueError(f"Invalid landing setting: {name}")
        for name in ("available_deceleration_m_s2", "reference_response_s",
                     "max_cartesian_correction_m", "max_joint_correction_rad",
                     "correction_rate_rad_s", "jacobian_regularization_m",
                     "minimum_jacobian_norm_m", "gain_rate_per_s", "loading_time_s",
                     "contact_confirm_s", "terrain_max_distance_m", "terrain_max_age_s",
                     "contact_max_age_s", "terrain_memory_update_s", "terrain_pose_jump_m",
                     "terrain_memory_cell_m", "terrain_yaw_jump_rad"):
            if getattr(self, name) <= 0:
                raise ValueError(f"Landing {name} must be positive")
        if not 0 < self.landing_kp_scale <= 1 or not 0 < self.landing_kd_scale <= 2:
            raise ValueError("Landing gain scale outside prototype bounds")
        if not self.contact_off_force_N < self.contact_on_force_N:
            raise ValueError("Contact hysteresis requires off < on")
        if self.max_approach_distance_m < self.approach_distance_m:
            raise ValueError("Maximum approach distance is below nominal approach distance")
        if type(self.minimum_other_supports) is not int or not 1 <= self.minimum_other_supports <= 3:
            raise ValueError("Expected one to three other supporting legs")
        if type(self.contact_delay_steps) is not int or self.contact_delay_steps < 0:
            raise ValueError("Contact delay must be a nonnegative integer")
        if type(self.terrain_memory_frames) is not int or not 1 <= self.terrain_memory_frames <= 32:
            raise ValueError("Terrain memory requires 1..32 frames")
        if not 0 <= self.contact_dropout_probability <= 1:
            raise ValueError("Invalid contact dropout probability")


def control_contract(settings: LandingSettings):
    settings.validate()
    return {
        "schema": "stair_quiet_v1.landing_control.v1",
        "settings": asdict(settings),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "memory_source_sha256": hashlib.sha256(
            Path(__file__).with_name("terrain_memory.py").read_bytes()).hexdigest(),
        "action_semantics": "nominal_16d_before_execution_control_and_actuator_delay",
        "geometry_scope": "horizontal_treads; uncertain_edges_bypass_and_count",
        "contact_requirement": "causal_support_contact_with_validity; simulator_force_is_oracle",
        "deployment_status": "requires_identified_actuator_and_validated_contact_inputs",
        "actual_acoustic_pass": False,
    }


def contract_hash(contract):
    return hashlib.sha256(json.dumps(contract, sort_keys=True, allow_nan=False).encode()).hexdigest()


def sample_tread_surface(wheel_xy, points_w, valid=None, *,
                         max_distance=0.12, max_height_spread=0.06):
    """Sample only observed horizontal patches; never replace unknowns with z=0.

    Accept a nearest sample within max_distance, but reject a wheel footprint
    crossing a height discontinuity. This does not detect unobserved risers.
    An invalid height is NaN intentionally: users must carry the validity mask.
    """
    if wheel_xy.ndim != 3 or wheel_xy.shape[1:] != (4, 2):
        raise ValueError("Expected [N,4,2] wheel coordinates")
    if points_w.ndim != 3 or points_w.shape[0] != wheel_xy.shape[0] or points_w.shape[-1] != 3:
        raise ValueError("Expected [N,M,3] terrain samples")
    if points_w.shape[1] == 0 or max_distance <= 0 or max_height_spread < 0:
        raise ValueError("Invalid terrain sampling domain")
    finite = torch.isfinite(points_w).all(-1)
    if valid is not None:
        if valid.shape != finite.shape:
            raise ValueError("Terrain validity shape mismatch")
        finite &= valid.bool()
    dist2 = (wheel_xy[:, :, None, :] - points_w[:, None, :, :2]).square().sum(-1)
    usable = finite[:, None, :] & torch.isfinite(dist2) & (dist2 <= max_distance**2)
    # All samples in the footprint must agree. Do not select whichever height
    # makes the controller's landing look easiest.
    heights = points_w[:, None, :, 2].expand(-1, 4, -1)
    lo = torch.where(usable, heights, torch.inf).amin(-1)
    hi = torch.where(usable, heights, -torch.inf).amax(-1)
    nearest = torch.where(usable, dist2, torch.inf).argmin(-1)
    chosen = heights.gather(-1, nearest[..., None]).squeeze(-1)
    ok = usable.any(-1) & ((hi - lo) <= max_height_spread) & torch.isfinite(wheel_xy).all(-1)
    return torch.where(ok, chosen, torch.nan), ok


class LandingController:
    """Causal per-wheel contact FSM and damped normal-reference projection.

    Phase codes: 0 swing, 1 approach, 2 loading, 3 support, 4 invalid input,
    5 insufficient support or singular geometry. Invalid observations disable
    new corrections. Existing offsets/gains return to nominal at bounded rates.
    """

    def __init__(self, num_envs, device, settings=None):
        self.settings = settings or LandingSettings()
        self.settings.validate()
        self.num_envs = num_envs
        self.device = torch.device(device)
        shape = (num_envs, 4)
        self.contact = torch.zeros(shape, dtype=torch.bool, device=device)
        self.pending_time = torch.zeros(shape, device=device)
        self.air_time = torch.zeros(shape, device=device)
        self.contact_age = torch.full(shape, 1e6, device=device)
        self.elapsed = torch.zeros((num_envs, 1), device=device)
        self.kp_scale = torch.ones(shape, device=device)
        self.kd_scale = torch.ones(shape, device=device)
        self.correction = torch.zeros((*shape, 3), device=device)
        self.counts = torch.zeros(6, dtype=torch.long, device=device)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.contact[ids] = False
        self.pending_time[ids] = 0
        self.air_time[ids] = 0
        self.contact_age[ids] = 1e6
        self.elapsed[ids] = 0
        self.kp_scale[ids] = 1
        self.kd_scale[ids] = 1
        self.correction[ids] = 0

    def summary(self):
        return dict(zip(("wheel_samples", "modified_samples", "invalid_samples",
                         "infeasible_samples", "touchdown_events", "approach_samples"),
                        self.counts.detach().cpu().tolist()))

    @torch.no_grad()
    def step(self, *, nominal_leg_target, nominal_wheel_velocity, joint_position,
             joint_velocity, normal_jacobian, normal_velocity, clearance,
             terrain_valid, contact_force, contact_valid, command_active, dt,
             point_jacobian=None, surface_normal=None):
        s = self.settings
        if not math.isfinite(dt) or not 0 < dt <= 0.1:
            raise ValueError("Landing controller dt must be in (0, 0.1] s")
        scalar_shape = (self.num_envs, 4)
        vector_shape = (*scalar_shape, 3)
        vectors = (nominal_leg_target, joint_position, joint_velocity, normal_jacobian)
        scalars = (nominal_wheel_velocity, normal_velocity, clearance, terrain_valid,
                   contact_force, contact_valid)
        if any(v.shape != vector_shape for v in vectors) or any(v.shape != scalar_shape for v in scalars):
            raise ValueError("Landing controller input shape mismatch")
        if command_active.shape != (self.num_envs,):
            raise ValueError("Expected one command-active bit per environment")
        # Never hide a corrupt outgoing nominal command or corrupt proprioception.
        if not torch.stack([torch.isfinite(v).all() for v in vectors[:3] + (nominal_wheel_velocity,)]).all():
            raise ValueError("Non-finite nominal command or joint state")
        geometric = (terrain_valid.bool() & torch.isfinite(clearance)
                     & torch.isfinite(normal_velocity) & torch.isfinite(normal_jacobian).all(-1))
        if (point_jacobian is None) != (surface_normal is None):
            raise ValueError("Full point Jacobian and surface normal must be provided together")
        if point_jacobian is not None:
            if point_jacobian.shape != (*scalar_shape, 3, 3) or surface_normal.shape != vector_shape:
                raise ValueError("Full point Jacobian/normal shape mismatch")
            geometric &= (torch.isfinite(point_jacobian).all(dim=(-2, -1))
                          & torch.isfinite(surface_normal).all(-1)
                          & ((surface_normal.square().sum(-1) - 1.).abs() < 1e-4))
        contact_ok = contact_valid.bool() & torch.isfinite(contact_force) & (contact_force >= 0)
        valid = geometric & contact_ok
        j = torch.nan_to_num(normal_jacobian, nan=0., posinf=0., neginf=0.)
        vn = torch.nan_to_num(normal_velocity, nan=0., posinf=0., neginf=0.)
        height = torch.nan_to_num(clearance, nan=0., posinf=0., neginf=0.)
        force = torch.nan_to_num(contact_force, nan=0., posinf=0., neginf=0.)
        self.elapsed += dt
        old_contact = self.contact.clone()
        wants_contact = contact_ok & torch.where(old_contact,
                            force >= s.contact_off_force_N, force >= s.contact_on_force_N)
        self.pending_time = torch.where(wants_contact, self.pending_time + dt, 0.)
        contact = wants_contact & (old_contact | (self.pending_time >= s.contact_confirm_s))
        touchdown = contact & ~old_contact & (self.air_time >= s.minimum_air_time_s)
        self.contact_age = torch.where(touchdown, 0., self.contact_age + dt)
        # Missing contact feedback is not evidence of flight.
        self.air_time = torch.where(contact_ok & ~contact, self.air_time + dt, 0.)
        self.contact = contact
        ready = (command_active.bool()[:, None] & (self.elapsed >= s.startup_grace_s))
        # Unknown wheel contacts cannot contribute supporting legs.
        others = contact.sum(-1, keepdim=True) - contact.to(torch.long)
        support_ok = others >= s.minimum_other_supports
        j2 = j.square().sum(-1)
        jac_ok = j2 >= s.minimum_jacobian_norm_m**2
        enabled = ready & valid & support_ok & jac_ok
        stopping = (vn.abs() * s.latency_budget_s + vn.square()
                    / (2 * s.available_deceleration_m_s2) + s.clearance_margin_m)
        trigger_distance = torch.clamp(stopping, min=s.approach_distance_m,
                                       max=s.max_approach_distance_m)
        approaching = (~contact & (self.air_time >= s.minimum_air_time_s)
                       & (vn < -s.closing_threshold_m_s)
                       & (height <= trigger_distance) & (height >= -s.penetration_tolerance_m))
        loading = contact & (self.contact_age < s.loading_time_s)
        remaining = (height - s.clearance_margin_m - vn.abs() * s.latency_budget_s).clamp_min(0.)
        safe_speed = torch.sqrt(s.landing_speed_m_s**2
                                + 2 * s.available_deceleration_m_s2 * remaining)
        desired_displacement = (j * (nominal_leg_target - joint_position)).sum(-1)
        # Include body/base motion, rather than treating all world vertical
        # wheel velocity as joint motion.
        base_velocity = vn - (j * joint_velocity).sum(-1)
        lower_displacement = (-safe_speed - base_velocity) * s.reference_response_s
        pre_correction = (lower_displacement - desired_displacement).clamp_min(0.)
        load_correction = (-s.loading_reference_allowance_m - desired_displacement).clamp_min(0.)
        magnitude = torch.zeros_like(height)
        if s.mode in ("precontact", "combined"):
            magnitude = torch.where(approaching & enabled, pre_correction, magnitude)
        if s.mode in ("phase", "combined"):
            magnitude = torch.where(loading & enabled, load_correction, magnitude)
        requested = magnitude.clone()
        magnitude = magnitude.clamp_max(s.max_cartesian_correction_m)
        if point_jacobian is None:
            proposed = j * (magnitude / (j2 + s.jacobian_regularization_m**2))[..., None]
        else:
            full_j = torch.nan_to_num(point_jacobian, nan=0., posinf=0., neginf=0.)
            normal = torch.nan_to_num(surface_normal, nan=0., posinf=0., neginf=0.)
            desired_cartesian = normal * magnitude[..., None]
            metric = full_j @ full_j.transpose(-1, -2)
            metric = metric + torch.eye(3, device=metric.device, dtype=metric.dtype) * s.jacobian_regularization_m**2
            proposed = (full_j.transpose(-1, -2) @ torch.linalg.solve(
                metric, desired_cartesian[..., None])).squeeze(-1)
        # Scale the whole correction, rather than clipping each joint and
        # rotating the intended correction into a different direction.
        ratio = (s.max_joint_correction_rad / proposed.abs().amax(-1).clamp_min(1e-12)).clamp_max(1.)
        proposed *= ratio[..., None]
        correction_step = s.correction_rate_rad_s * dt
        difference = proposed - self.correction
        rate_ratio = (correction_step / difference.abs().amax(-1).clamp_min(1e-12)).clamp_max(1.)
        self.correction += difference * rate_ratio[..., None]
        soften = enabled & (approaching | loading) & (s.mode in ("phase", "combined"))
        recovery = (self.contact_age / s.loading_time_s).clamp(0., 1.)
        ramp_kp = torch.where(loading, s.landing_kp_scale + (1 - s.landing_kp_scale) * recovery,
                              s.landing_kp_scale)
        target_kp = torch.where(soften, ramp_kp, 1.)
        target_kd = torch.where(soften, s.landing_kd_scale, 1.)
        gain_step = s.gain_rate_per_s * dt
        self.kp_scale += (target_kp - self.kp_scale).clamp(-gain_step, gain_step)
        self.kd_scale += (target_kd - self.kd_scale).clamp(-gain_step, gain_step)
        if s.mode == "off":
            self.correction.zero_()
            self.kp_scale.fill_(1.)
            self.kd_scale.fill_(1.)
        invalid = ready & ~valid
        limited = ((requested > s.max_cartesian_correction_m + 1e-6)
                   | ((j * self.correction).sum(-1) + 1e-6 < magnitude))
        infeasible = ready & valid & (approaching | loading) & (
            ~support_ok | ~jac_ok | limited)
        phase = torch.where(contact, 3, 0)
        phase = torch.where(approaching & enabled, 1, phase)
        phase = torch.where(loading & enabled, 2, phase)
        phase = torch.where(invalid, 4, phase)
        phase = torch.where(infeasible, 5, phase)
        modified = ((self.correction.abs().amax(-1) > 1e-7)
                    | ((self.kp_scale - 1.).abs() > 1e-7) | ((self.kd_scale - 1.).abs() > 1e-7))
        self.counts += torch.stack((torch.ones_like(phase).sum(), modified.sum(), invalid.sum(),
                                    infeasible.sum(), touchdown.sum(), (phase == 1).sum()))
        return {
            "leg_target": nominal_leg_target + self.correction,
            # Wheel synchronisation is deliberately not guessed from an
            # uncertain tread. A later separately evaluated mode can add it.
            "wheel_velocity": nominal_wheel_velocity.clone(),
            "kp_scale": self.kp_scale.clone(), "kd_scale": self.kd_scale.clone(),
            "phase": phase, "modified": modified, "invalid": invalid,
            "infeasible": infeasible, "safe_speed": safe_speed,
            "correction": self.correction.clone(),
        }
