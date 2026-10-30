"""Pure tensor stability signals for StairTrot-v1; no simulator or optimizer.

Wheel velocity is measured at the link origin, where the collision capsule is
centred. Isaac's ``body_lin_vel_w`` is a COM velocity and is not interchangeable
with ``body_link_lin_vel_w`` when forming a contact-point velocity.
"""
import math

import torch


def rolling_slip(
    link_vel, ang_vel, axle_w, normal_force_w,
    radius=0.09, capsule_half_length=0.027, load_min=15.0,
    upward_cos_min=0.5, axis_normal_deadband=1.e-5,
):
    """Return ``(tangential_speed_squared, upward_loaded)`` per wheel.

    Vector inputs have shape ``(..., wheels, 3)`` in the same world frame.
    ``axle_w`` is the unit capsule axis, or a nonzero vector parallel to it.
    ``normal_force_w`` contains normal contact forces, not friction forces.

    The actual M20 USD uses capsules. Their support point along a normal is
    ``-radius * normal - half_length * sign(axis.dot(normal)) * axis``.
    At perpendicular axes, the contact segment's midpoint is used. This is a
    geometric contact-velocity proxy, not a measured individual contact point.
    Airborne wheels, downward/near-horizontal contacts, or invalid inputs have
    a false validity mask and zero cost. They must not be logged as zero-slip
    support samples.
    """
    if radius <= 0 or capsule_half_length < 0 or load_min <= 0:
        raise ValueError("Wheel radius/load threshold must be positive and half length nonnegative")
    if not 0 < upward_cos_min <= 1 or axis_normal_deadband < 0:
        raise ValueError("Invalid upward support cosine or axis deadband")
    if link_vel.shape[-1] != 3 or any(
        value.shape != link_vel.shape for value in (ang_vel, axle_w, normal_force_w)
    ):
        raise ValueError("Wheel vectors must have identical (..., wheels, 3) shapes")
    finite = torch.stack([
        torch.isfinite(value).all(-1)
        for value in (link_vel, ang_vel, axle_w, normal_force_w)
    ]).all(0)
    velocity, omega, axis, force = (
        torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        for value in (link_vel, ang_vel, axle_w, normal_force_w)
    )
    magnitude = torch.linalg.vector_norm(force, dim=-1)
    axis_norm = torch.linalg.vector_norm(axis, dim=-1)
    normal = force / magnitude.clamp_min(1.e-8)[..., None]
    axis = axis / axis_norm.clamp_min(1.e-8)[..., None]
    loaded = (
        finite & (axis_norm > 1.e-8) & (force[..., 2] >= load_min)
        & (normal[..., 2] >= upward_cos_min)
    )
    dot = (axis * normal).sum(-1)
    endpoint = torch.where(dot.abs() > axis_normal_deadband, dot.sign(), 0.0)
    lever = -radius * normal - capsule_half_length * endpoint[..., None] * axis
    contact_velocity = velocity + torch.linalg.cross(omega, lever, dim=-1)
    tangent = contact_velocity - (contact_velocity * normal).sum(-1)[..., None] * normal
    squared = tangent.square().sum(-1)
    return torch.where(loaded, squared, 0.0), loaded


def posture_errors(projected_gravity_b, front_ground_z, rear_ground_z, axle_distance):
    """Return roll and nose-up pitch error in radians, with shape ``(envs,)``.

    For body-to-world rotation ``Rz(yaw) Ry(theta) Rx(roll)``, projected gravity
    is ``(sin(theta), -sin(roll)*cos(theta), -cos(roll)*cos(theta))``.
    Positive ``theta`` pitches the body x-axis *down*, so nose-up pitch is its
    negative. A higher front tread therefore requires a negative gravity x.

    Ground heights and axle distance must refer to the same forward route.
    Callers must mask missing/ambiguous tread estimates. ``axle_distance`` is
    the horizontal distance between the axle centres, not their 3-D distance.
    """
    if projected_gravity_b.shape[-1] != 3:
        raise ValueError("Projected gravity must have shape (..., 3)")
    gravity = projected_gravity_b
    horizontal_gravity = torch.linalg.vector_norm(gravity[..., 1:], dim=-1)
    roll = torch.atan2(-gravity[..., 1], -gravity[..., 2])
    nose_up = torch.atan2(-gravity[..., 0], horizontal_gravity)
    span = torch.as_tensor(
        axle_distance, dtype=gravity.dtype, device=gravity.device
    ).clamp_min(1.e-6)
    target = torch.atan2(front_ground_z - rear_ground_z, span)
    return roll, nose_up - target


def posture_cost(
    projected_gravity_b, roll_rate, front_ground_z, rear_ground_z, axle_distance,
    roll_tolerance=math.radians(4.0), pitch_tolerance=math.radians(12.0),
    angle_scale=math.radians(10.0), roll_rate_tolerance=0.35,
    roll_rate_scale=0.8, pitch_weight=1.0, rate_weight=0.25,
):
    """Terrain-relative pitch band plus roll and roll-rate excess, per env."""
    if angle_scale <= 0 or roll_rate_scale <= 0:
        raise ValueError("Posture normalization scales must be positive")
    if min(roll_tolerance, pitch_tolerance, roll_rate_tolerance, pitch_weight, rate_weight) < 0:
        raise ValueError("Posture tolerances and weights must be nonnegative")
    roll, pitch_error = posture_errors(
        projected_gravity_b, front_ground_z, rear_ground_z, axle_distance
    )
    roll_cost = ((roll.abs() - roll_tolerance).clamp_min(0) / angle_scale).square()
    pitch_cost = ((pitch_error.abs() - pitch_tolerance).clamp_min(0) / angle_scale).square()
    rate_cost = ((roll_rate.abs() - roll_rate_tolerance).clamp_min(0) / roll_rate_scale).square()
    return roll_cost + pitch_weight * pitch_cost + rate_weight * rate_cost


def action_jerk_cost(action, previous_action, previous_previous_action, episode_age, scale=1.0):
    """Mean squared second action difference; valid after two prior steps.

    ``episode_age`` counts completed history steps, starting at zero on reset.
    Numeric zero is an ordinary action value, never a history-validity flag.
    ``scale`` may be a scalar or a positive per-action tensor. This is target
    smoothness in action units, not physical jerk in metres per second cubed.
    """
    if action.shape != previous_action.shape or action.shape != previous_previous_action.shape:
        raise ValueError("All three action histories must have the same shape")
    if isinstance(scale, (int, float)) and scale <= 0:
        raise ValueError("Action normalization scale must be positive")
    normalization = torch.as_tensor(scale, dtype=action.dtype, device=action.device)
    second_difference = (action - 2 * previous_action + previous_previous_action) / normalization
    cost = second_difference.square().mean(-1)
    return torch.where(episode_age >= 2, cost, 0.0)


def touchdown_impact_proxy(
    previous_link_velocity, first_contact, normal_force_history, episode_age,
    free_fall_speed=0.1, approach_scale=0.5, force_threshold=150.0,
    force_scale=200.0, load_min=15.0, upward_cos_min=0.5, force_weight=1.0,
):
    """Sum touchdown approach/force costs over wheels, returning ``(envs,)``.

    History is ``(envs, physics_history, wheels, 3)``; previous velocity and
    first-contact flags are ``(envs, wheels, 3)`` and ``(envs, wheels)``.
    The normal at the largest upward load in the short sensor history defines
    approach direction. A pre-contact control sample survives velocity removal
    by the contact solver. Historical force peaks catch impacts missed between
    control samples. Both are proxies: this is neither exact impact velocity
    nor a force impulse, and no exact substep timing is claimed.

    Only first contacts with upward load pay the cost. ``episode_age >= 1``
    excludes a stale velocity sample immediately after an environment reset.
    """
    if approach_scale <= 0 or force_scale <= 0 or load_min <= 0:
        raise ValueError("Impact normalization scales and load threshold must be positive")
    if min(free_fall_speed, force_threshold, force_weight) < 0 or not 0 < upward_cos_min <= 1:
        raise ValueError("Invalid impact threshold, weight or support cosine")
    if (
        previous_link_velocity.ndim != 3 or previous_link_velocity.shape[-1] != 3
        or normal_force_history.ndim != 4 or normal_force_history.shape[1] < 1
        or normal_force_history.shape[0] != previous_link_velocity.shape[0]
        or normal_force_history.shape[2:] != previous_link_velocity.shape[1:]
        or first_contact.shape != previous_link_velocity.shape[:-1]
    ):
        raise ValueError("Impact inputs must be env/wheel vectors and env/history/wheel forces")
    force = torch.nan_to_num(normal_force_history, nan=0.0, posinf=0.0, neginf=0.0)
    magnitude = torch.linalg.vector_norm(force, dim=-1)
    normal = force / magnitude.clamp_min(1.e-8)[..., None]
    upward = (
        torch.isfinite(normal_force_history).all(-1)
        & (force[..., 2] >= load_min) & (normal[..., 2] >= upward_cos_min)
    )
    upward_magnitude = torch.where(upward, magnitude, 0.0)
    peak, indices = upward_magnitude.max(dim=1)
    gather = indices[:, None, :, None].expand(-1, 1, -1, 3)
    peak_normal = normal.gather(1, gather).squeeze(1)
    approach = (-(previous_link_velocity * peak_normal).sum(-1) - free_fall_speed).clamp_min(0)
    cost = (approach / approach_scale).square()
    cost += force_weight * ((peak - force_threshold).clamp_min(0) / force_scale).square()
    valid = (
        first_contact.bool() & upward.any(dim=1)
        & (episode_age >= 1)[:, None] & torch.isfinite(previous_link_velocity).all(-1)
    )
    return torch.where(valid, cost, 0.0).sum(-1)
