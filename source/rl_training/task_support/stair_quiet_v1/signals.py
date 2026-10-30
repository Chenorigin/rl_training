"""Substep impact proxies for stair_quiet_v1; no simulator or acoustic model.

Force/velocity costs are training candidates. They are not calibrated sound
pressure estimates and must never be reported as dB or as acoustic acceptance.
"""
from dataclasses import dataclass, field
import math

import torch


LEGS = ("fl", "fr", "hl", "hr")
COST_NAMES = ("touchdown", "loading_rate", "window_peak", "window_impulse", "air_descent",
              "contact_work", "reimpact", "coincident_impact")


@dataclass
class StairQuietSettings:
    """Single source of candidate physical scales, not acoustically calibrated."""

    weights: dict = field(default_factory=lambda: {
        "touchdown": -0.03, "loading_rate": -0.02, "window_peak": -0.04,
        "window_impulse": -0.04, "air_descent": -0.02,
        "contact_work": -0.015, "reimpact": -0.02, "coincident_impact": -0.005,
    })
    contact_on_force: float = 5.0
    contact_off_force: float = 2.0
    reset_grace_time: float = 0.1
    minimum_air_time: float = 0.005
    velocity_reference: float = 0.5
    descent_free_speed: float = 0.2
    window_seconds: float = 0.04
    allowed_force: float = 100.0
    force_reference: float = 200.0
    impulse_reference: float = 5.0
    loading_reference: float = 20000.0
    quadratic_linear_knee: float = 2.0
    # Default M20.usd wheel collision is a capsule, not the URDF cylinder.
    # Geometry is checked against the ACTIVE USD before physics can be stepped.
    wheel_radius: float = 0.09
    wheel_capsule_half_length: float = 0.027
    wheel_axis_local: tuple = (0.0, 1.0, 0.0)
    geometry_tolerance: float = 1e-5
    axis_projection_tolerance: float = 1e-5
    vector_norm_tolerance: float = 1e-4
    numerical_epsilon: float = 1e-8
    contact_work_reference: float = 5.0
    reimpact_window_seconds: float = 0.15
    coincident_window_seconds: float = 0.02
    curriculum_initial_scale: float = 0.1
    curriculum_warmup_iterations: int = 200
    curriculum_ramp_iterations: int = 1000

    def validate(self):
        if set(self.weights) != set(COST_NAMES):
            raise ValueError("All stair quiet costs must be configured; weight=0 disables a cost")
        if any(not math.isfinite(v) or v > 0 for v in self.weights.values()):
            raise ValueError("Stair quiet weights must be finite and nonpositive")
        for name, value in vars(self).items():
            if name not in ("weights", "wheel_axis_local") and (not math.isfinite(value) or value < 0):
                raise ValueError(f"Invalid stair quiet setting: {name}")
        for name in ("velocity_reference", "window_seconds", "force_reference",
                     "impulse_reference", "loading_reference", "quadratic_linear_knee",
                     "wheel_radius", "wheel_capsule_half_length", "geometry_tolerance",
                     "vector_norm_tolerance", "numerical_epsilon", "contact_work_reference",
                     "reimpact_window_seconds", "coincident_window_seconds", "curriculum_ramp_iterations"):
            if getattr(self, name) <= 0:
                raise ValueError(f"Stair quiet {name} must be positive")
        if not self.contact_off_force < self.contact_on_force:
            raise ValueError("Contact thresholds require off < on")
        if (len(self.wheel_axis_local) != 3 or
                any(not math.isfinite(v) for v in self.wheel_axis_local) or
                not math.isclose(sum(v * v for v in self.wheel_axis_local), 1.,
                                 abs_tol=self.vector_norm_tolerance)):
            raise ValueError("Wheel local axis must be a finite unit vector")
        if not 0 <= self.curriculum_initial_scale <= 1:
            raise ValueError("Curriculum initial scale must be in [0, 1]")
        for key in ("curriculum_warmup_iterations", "curriculum_ramp_iterations"):
            if type(getattr(self, key)) is not int:
                raise ValueError("Curriculum iteration counts must be integers")


def scheduled_quiet_scale(schedule, common_step_counter, settings):
    """Resolve the runner's quiet-task clock; never use the source teacher iter."""
    if not isinstance(schedule, dict):
        raise RuntimeError("Stair quiet schedule is not installed by the task runner")
    keys = ("iteration_offset", "step_offset", "steps_per_iteration")
    if any(key not in schedule or type(schedule[key]) is not int for key in keys):
        raise ValueError("Stair quiet schedule requires integer offsets and rollout length")
    offset, step_offset, rollout = (schedule[key] for key in keys)
    if (type(common_step_counter) is not int or offset < 0 or step_offset < 0
            or rollout <= 0 or common_step_counter < step_offset):
        raise ValueError("Invalid or rewound stair quiet schedule")
    iteration = offset + (common_step_counter - step_offset) // rollout
    fraction = min(1., max(0., (iteration - settings.curriculum_warmup_iterations)
                           / settings.curriculum_ramp_iterations))
    smooth = fraction * fraction * (3. - 2. * fraction)
    scale = settings.curriculum_initial_scale + (1. - settings.curriculum_initial_scale) * smooth
    return iteration, scale


def wheel_axis_world(quaternion_wxyz, settings, *, validate=True):
    """Rotate the PRE-physics link pose; public calls validate by default.

    The env uses validate=False and passes the same quaternion snapshot into
    Tracker.update, where all physics inputs share one device-to-host check.
    """
    if quaternion_wxyz.shape[-1:] != (4,):
        raise ValueError("Wheel quaternions must be finite wxyz vectors")
    if validate:
        norm = torch.linalg.vector_norm(quaternion_wxyz, dim=-1)
        valid = (torch.isfinite(quaternion_wxyz).all()
                 & ((norm - 1.).abs() <= settings.vector_norm_tolerance).all())
        if not valid:
            raise ValueError("Wheel quaternions must be finite and normalized")
    local = quaternion_wxyz.new_tensor(settings.wheel_axis_local).expand(*quaternion_wxyz.shape[:-1], 3)
    imaginary = quaternion_wxyz[..., 1:]
    twice_cross = 2. * torch.linalg.cross(imaginary, local, dim=-1)
    return local + quaternion_wxyz[..., :1] * twice_cross + torch.linalg.cross(imaginary, twice_cross, dim=-1)


def _validate_contact_inputs(pre_velocity, pre_omega, normal_force, wheel_axis,
                             link_minus_com, settings, wheel_quaternion=None):
    """One device reduction/scalar transfer for all runtime tensor invariants."""
    shape = pre_velocity.shape
    vectors = (pre_velocity, pre_omega, normal_force, wheel_axis, link_minus_com)
    if shape[-1:] != (3,) or any(value.shape != shape for value in vectors):
        raise ValueError("Contact kinematics require matching [..., 3] vectors")
    checks = [torch.isfinite(value).all() for value in vectors]
    checks.append(((torch.linalg.vector_norm(wheel_axis, dim=-1) - 1.).abs()
                   <= settings.vector_norm_tolerance).all())
    if wheel_quaternion is not None:
        if wheel_quaternion.shape != (*shape[:-1], 4):
            raise ValueError("Wheel quaternion shape must match wheel kinematics")
        checks.extend((
            torch.isfinite(wheel_quaternion).all(),
            ((torch.linalg.vector_norm(wheel_quaternion, dim=-1) - 1.).abs()
             <= settings.vector_norm_tolerance).all(),
        ))
    # Keep every finite/norm check device-side until this single branch. The
    # error deliberately names both failure classes without additional reads.
    if not torch.stack(checks).all():
        raise ValueError("Non-finite contact physics sample or wheel axes/quaternions not normalized")


def contact_point_kinematics(pre_velocity, pre_omega, normal_force, wheel_axis, link_minus_com, settings,
                             *, validate=True):
    """Approximate capsule support-point velocity, not a measured contact point.

    A capsule is a line segment plus a sphere. Its support point against outward
    normal n is -R*n - half_length*sign(n.axis)*axis, relative to link origin.
    The segment midpoint is used when n.axis is nearly zero (nonunique support).
    Actual COM offset is included before applying v_point = v_com + omega x r.
    Net NORMAL force gives the estimated n. Multiple contacts are aggregated,
    so this does not recover every individual contact normal or contact point.
    Only normal closing speed is penalized; tangential rolling is excluded.
    """
    if validate:
        _validate_contact_inputs(pre_velocity, pre_omega, normal_force, wheel_axis, link_minus_com, settings)
    magnitude = torch.linalg.vector_norm(normal_force, dim=-1)
    normal = normal_force / magnitude.clamp_min(settings.numerical_epsilon)[..., None]
    projection = (normal * wheel_axis).sum(-1)
    end = torch.where(projection.abs() > settings.axis_projection_tolerance,
                      projection.sign(), 0.)
    offset = (link_minus_com - settings.wheel_radius * normal
              - settings.wheel_capsule_half_length * end[..., None] * wheel_axis)
    velocity = pre_velocity + torch.linalg.cross(pre_omega, offset, dim=-1)
    approach = (-(velocity * normal).sum(-1)).clamp_min(0)
    return velocity, approach, offset


def validate_usd_wheel_geometry(usd_path, settings):
    """Read the active USD on CPU and reject unsupported/changed wheel geometry."""
    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise ValueError(f"Could not inspect wheel collision geometry: {usd_path}")
    if not math.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1., abs_tol=settings.geometry_tolerance):
        raise ValueError("Stair quiet requires wheel geometry authored in metres")
    root = stage.GetDefaultPrim()
    cache = UsdGeom.XformCache()
    measured = []
    for leg in LEGS:
        body = stage.GetPrimAtPath(root.GetPath().AppendChild(f"{leg}_wheel"))
        if not body.IsValid():
            raise ValueError(f"Missing wheel body in active USD: {leg}")
        colliders = [p for p in Usd.PrimRange(body, Usd.TraverseInstanceProxies())
                     if p.GetAttribute("physics:collisionEnabled").Get() is True]
        if len(colliders) != 1 or not colliders[0].IsA(UsdGeom.Capsule):
            raise ValueError(f"Stair quiet only supports the verified single-capsule wheel: {leg}")
        prim = colliders[0]
        capsule = UsdGeom.Capsule(prim)
        radius, half_length = capsule.GetRadiusAttr().Get(), capsule.GetHeightAttr().Get() / 2.
        transform, _ = cache.ComputeRelativeTransform(prim, body)
        axis_local = {"X": Gf.Vec3d(1, 0, 0), "Y": Gf.Vec3d(0, 1, 0), "Z": Gf.Vec3d(0, 0, 1)}[capsule.GetAxisAttr().Get()]
        axis = transform.TransformDir(axis_local)
        expected_axis = Gf.Vec3d(*settings.wheel_axis_local)
        body_world = cache.GetLocalToWorldTransform(body)
        basis = [Gf.Vec3d(*(1 if j == i else 0 for j in range(3))) for i in range(3)]
        unit_scale = all(
            math.isclose(matrix.TransformDir(vector).GetLength(), 1., abs_tol=settings.geometry_tolerance)
            for matrix in (transform, body_world) for vector in basis)
        if (not math.isclose(radius, settings.wheel_radius, abs_tol=settings.geometry_tolerance)
                or not math.isclose(half_length, settings.wheel_capsule_half_length, abs_tol=settings.geometry_tolerance)
                or transform.Transform(Gf.Vec3d(0)).GetLength() > settings.geometry_tolerance
                or not math.isclose(abs(Gf.Dot(axis, expected_axis)), 1., abs_tol=settings.geometry_tolerance)
                or not unit_scale):
            raise ValueError(f"Active USD wheel geometry differs from StairQuietSettings: {leg}")
        measured.append({"leg": leg, "collision_path": str(prim.GetPath()), "shape": "capsule",
                         "radius_m": radius, "half_length_m": half_length, "axis_link": list(axis)})
    return {"usd_path": str(usd_path), "wheels": measured}


def _quadratic_linear(value, knee):
    """Quadratic near zero, linear tail; large impacts never reach a flat cap."""
    return value.clamp_max(knee).square() + 2 * knee * (value - knee).clamp_min(0)


class StairQuietTracker:
    """Reset-safe, per-wheel physical accounting and policy-step cost integrals.

    ``pre_velocity`` MUST be read before physics and ``post_force`` after that
    same physics substep. Magnitude-based force costs include riser collisions,
    ongoing contact and rapid contact chatter; they have no terrain/command/
    normal-estimation gate. Only touchdown speed requires an actual new contact.
    The measured net normal force direction approximates a contact normal.
    Wheel-point normal velocity includes rotation and actual COM offset, while
    airborne descent retains COM downward speed before any contact is present.
    """

    def __init__(self, num_envs, device, settings=None):
        self.settings = settings or StairQuietSettings()
        self.settings.validate()
        shape = (num_envs, 4)
        self.contact = torch.zeros(shape, dtype=torch.bool, device=device)
        self.initialized = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.age = torch.zeros(num_envs, device=device)
        self.air_time = torch.zeros(shape, device=device)
        self.previous_force = torch.zeros(shape, device=device)
        self.step_events = torch.zeros(shape, device=device)
        self.costs = {name: torch.zeros(num_envs, device=device) for name in COST_NAMES}
        self.event_count = torch.zeros(shape, device=device)
        self.speed_sum = torch.zeros(shape, device=device)
        self.speed_sq_sum = torch.zeros(shape, device=device)
        self.speed_max = torch.zeros(shape, device=device)
        self.force_peak = torch.zeros(shape, device=device)
        self.force_integral = torch.zeros(shape, device=device)
        self.loading_peak = torch.zeros(shape, device=device)
        self.window_impulse_peak = torch.zeros(shape, device=device)
        self.contact_work = torch.zeros(shape, device=device)
        self.reimpact_count = torch.zeros(shape, device=device)
        self.last_impact_time = torch.zeros(shape, device=device)
        self.last_impact_amplitude = torch.zeros(shape, device=device)
        self.had_impact = torch.zeros(shape, dtype=torch.bool, device=device)
        self.coincident_pair_cost = torch.zeros(num_envs, device=device)
        self.sample_time = torch.zeros(num_envs, device=device)
        self.distance = torch.zeros(num_envs, device=device)
        self.force_history = None
        self.history_cursor = 0
        self.physics_dt = None
        self.last = {}

    def begin_step(self):
        self.step_events.zero_()
        for value in self.costs.values():
            value.zero_()

    def update(self, pre_velocity, post_force, dt, *, pre_omega, wheel_axis, link_minus_com,
               wheel_quaternion=None):
        shape = (*self.contact.shape, 3)
        if pre_velocity.shape != shape or post_force.shape != shape:
            raise ValueError("Wheel velocity and force must have shape [num_envs, 4, 3]")
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("Physics dt must be finite and positive")
        s = self.settings
        _validate_contact_inputs(pre_velocity, pre_omega, post_force, wheel_axis, link_minus_com, s,
                                 wheel_quaternion=wheel_quaternion)
        if self.physics_dt is None:
            self.physics_dt = dt
            bins = max(1, math.ceil(s.window_seconds / dt))
            self.force_history = post_force.new_zeros((bins, *self.contact.shape))
        elif not math.isclose(self.physics_dt, dt, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("Physics dt cannot change while impact history is active")

        force = torch.linalg.vector_norm(post_force, dim=-1)
        contact = torch.where(self.contact, force > s.contact_off_force,
                              force >= s.contact_on_force)
        eligible = self.initialized & (self.age >= s.reset_grace_time)
        active = eligible[:, None]
        event = contact & ~self.contact & active & (self.air_time >= s.minimum_air_time)
        point_velocity, approach, point_offset = contact_point_kinematics(
            pre_velocity, pre_omega, post_force, wheel_axis, link_minus_com, s, validate=False)
        downward = (-pre_velocity[..., 2]).clamp_min(0)
        loading = ((force - self.previous_force) / dt).clamp_min(0)

        # Trailing windows cover every force sample, including reloading without
        # a contact edge. Partial last bins integrate exactly window_seconds.
        self.force_history[self.history_cursor].copy_(force)
        self.history_cursor = (self.history_cursor + 1) % len(self.force_history)
        window_peak = self.force_history.amax(0)
        excess_history = (self.force_history - s.allowed_force).clamp_min(0)
        impulse = excess_history.sum(0) * dt
        overhang = len(self.force_history) * dt - s.window_seconds
        impulse -= excess_history[self.history_cursor] * overhang
        impulse = impulse.clamp_min(0)
        penalty = lambda value: _quadratic_linear(value, s.quadratic_linear_knee)
        self.costs["touchdown"] += (event * (approach / s.velocity_reference).square()).sum(-1)
        self.costs["loading_rate"] += (penalty(loading / s.loading_reference) * active).sum(-1) * dt
        self.costs["window_peak"] += (
            penalty((window_peak - s.allowed_force).clamp_min(0) / s.force_reference) * active
        ).sum(-1) * dt
        self.costs["window_impulse"] += (
            penalty(impulse / s.impulse_reference) * active
        ).sum(-1) * dt
        # No height-map gate: missing map data must not eliminate the objective.
        # The airborne phase uses PRE-substep contact to include the impact step.
        descent = ((downward - s.descent_free_speed).clamp_min(0) / s.velocity_reference).square()
        self.costs["air_descent"] += (descent * ~self.contact * active).sum(-1) * dt

        # Force x PRE-step normal closing speed x dt is an incoming contact-work
        # proxy. It is not measured dissipation: post-step impulse and pre-step
        # velocity differ in time. Continuous reloading is included without an
        # edge gate; free rolling and normal separation contribute no work.
        work = force * approach * active * dt
        self.costs["contact_work"] += work.sum(-1) / s.contact_work_reference
        since_impact = self.age[:, None] - self.last_impact_time
        repeat = event & self.had_impact & (since_impact <= s.reimpact_window_seconds)
        amplitude = event * approach / s.velocity_reference
        self.costs["reimpact"] += (repeat * (approach / s.velocity_reference).square()).sum(-1)
        # Penalize only temporally clustered IMPACTS. Static multi-foot support,
        # sequential gait outside the window and multiple samples of one
        # landing have no cross-foot cost. Same-substep pairs are counted once.
        recent = (1. - since_impact / s.coincident_window_seconds).clamp(0., 1.)
        history = self.last_impact_amplitude * self.had_impact * recent
        other_history = history.sum(-1, keepdim=True) - history
        same_step = (amplitude.sum(-1).square() - amplitude.square().sum(-1)) * 0.5
        pair_cost = (amplitude * other_history).sum(-1) + same_step
        self.costs["coincident_impact"] += pair_cost
        self.contact_work += work
        self.reimpact_count += repeat
        self.coincident_pair_cost += pair_cost
        self.last_impact_time = torch.where(event, self.age[:, None], self.last_impact_time)
        self.last_impact_amplitude = torch.where(event, amplitude, self.last_impact_amplitude)
        self.had_impact |= event
        self.step_events += event
        self.event_count += event
        self.speed_sum += approach * event
        self.speed_sq_sum += approach.square() * event
        self.speed_max = torch.maximum(self.speed_max, approach * event)
        self.force_peak = torch.maximum(self.force_peak, force * active)
        self.force_integral += force * active * dt
        self.loading_peak = torch.maximum(self.loading_peak, loading * active)
        self.window_impulse_peak = torch.maximum(self.window_impulse_peak, impulse * active)
        self.sample_time += eligible * dt
        self.last = {
            "event": event, "eligible": eligible,
            "pre_velocity_m_s": pre_velocity.clone(),
            "pre_contact_point_velocity_m_s": point_velocity,
            "estimated_contact_offset_m": point_offset,
            "approach_speed_m_s": approach, "force_N": force,
            "loading_rate_N_s": loading, "window_peak_N": window_peak,
            "window_excess_impulse_N_s": impulse,
            "incoming_contact_work_J": work, "reimpact": repeat,
            "coincident_pair_cost": pair_cost,
        }
        self.previous_force.copy_(force)
        self.air_time = torch.where(contact, 0.0, self.air_time + dt)
        self.contact.copy_(contact)
        self.initialized.fill_(True)
        self.age += dt
        return event

    def metrics(self, env_ids):
        """Physical proxies with explicit opportunity counts, never acoustic dB.

        Mean/RMS metadata is consumed by the inherited runner to pool raw sums
        and denominators across reset batches. Ordinary maximum values are
        averaged by its logger, so their names explicitly describe the mean of
        reset-batch peaks; they are NOT global maxima over an iteration/run.
        """
        result = {}

        def statistic(key, total, count, kind="weighted"):
            value = total / count
            result[key] = value.sqrt() if kind == "rms" else value
            result[f"__{kind}_total__/{key}"] = total
            result[f"__{kind}_count__/{key}"] = count

        time = self.sample_time[env_ids].sum()
        groups = [("all", slice(None)), ("front", slice(0, 2)), ("rear", slice(2, 4))]
        groups += [(name, slice(i, i + 1)) for i, name in enumerate(LEGS)]
        for name, indices in groups:
            prefix = f"StairQuiet/{name}/"
            count = self.event_count[env_ids, indices].sum()
            result[prefix + "touchdown_count"] = count
            result[prefix + "reimpact_count"] = self.reimpact_count[env_ids, indices].sum()
            result[prefix + "has_touchdowns"] = (count > 0).float()
            result[prefix + "has_physics_samples"] = (time > 0).float()
            if time > 0:
                result[prefix + "mean_reset_batch_force_max_N"] = self.force_peak[env_ids, indices].amax()
                result[prefix + "mean_reset_batch_loading_rate_max_N_s"] = self.loading_peak[env_ids, indices].amax()
                result[prefix + "mean_reset_batch_window_excess_impulse_max_N_s"] = self.window_impulse_peak[env_ids, indices].amax()
                wheels = self.force_integral[env_ids, indices].shape[-1]
                statistic(prefix + "force_per_wheel_time_mean_N",
                          self.force_integral[env_ids, indices].sum(), time * wheels)
                statistic(prefix + "incoming_contact_work_rate_W",
                          self.contact_work[env_ids, indices].sum(), time)
            if count > 0:
                speed_sum = self.speed_sum[env_ids, indices].sum()
                speed_sq_sum = self.speed_sq_sum[env_ids, indices].sum()
                statistic(prefix + "preimpact_speed_mean_m_s", speed_sum, count)
                statistic(prefix + "preimpact_speed_rms_m_s", speed_sq_sum, count, "rms")
                result[prefix + "mean_reset_batch_preimpact_speed_max_m_s"] = self.speed_max[env_ids, indices].amax()
                statistic(prefix + "specific_impact_energy_mean_J_kg", 0.5 * speed_sq_sum, count)
        result["StairQuiet/eligible_environment_seconds"] = time
        distance = self.distance[env_ids].sum()
        result["StairQuiet/distance_m"] = distance
        if distance > 0:
            statistic("StairQuiet/specific_impact_energy_per_m_J_kg_m",
                      0.5 * self.speed_sq_sum[env_ids].sum(), distance)
        return result

    def reset(self, env_ids):
        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                if name == "force_history":
                    value[:, env_ids] = 0
                else:
                    value[env_ids] = 0
        for value in self.costs.values():
            value[env_ids] = 0
        for value in self.last.values():
            value[env_ids] = 0


def stair_quiet_cost(env, name):
    """Cancel RewardManager's dt factor for already integrated/discrete costs."""
    scale = getattr(env, "stair_quiet_reward_scale", None)
    if scale is None or not math.isfinite(scale) or not 0 <= scale <= 1:
        raise RuntimeError("Stair quiet reward curriculum was not resolved for this step")
    return env.stair_quiet_tracker.costs[name] * scale / env.step_dt
