"""Longitudinal teacher scan and separate, privileged four-wheel safety input."""
import torch
from task_support.stair_quiet_v1.signals import wheel_axis_world
from task_support.stair_quiet_v1_width.geometry import (
    foothold_regions, capsule_contact_velocity, wheel_margins)
from .settings import SAFETY_DIM, LATERAL_SAMPLES


def longitudinal_scan(raw_height, local_x, course_y, width, lateral_samples=LATERAL_SAMPLES):
    """Replace ONLY rays outside the physical stair by the nearest lane ray.

    The same longitudinal sample is used. Real stair heights and every
    in-bounds measurement are retained; this is not a physical terrain edit.
    """
    if raw_height.shape[-1]%lateral_samples:
        raise ValueError("Longitudinal scan requires the rectangular grid layout")
    # Work per longitudinal grid row, O(N*187), avoiding an N*187*187
    # distance tensor at the actual 4096-environment training batch size.
    order = local_x.argsort(-1)
    heights = raw_height.gather(1, order).reshape(len(width), -1, lateral_samples)
    y = course_y.gather(1, order).reshape_as(heights)
    inside = y.abs() <= width[:, None, None]/2
    centre = y.abs().masked_fill(~inside, torch.inf).argmin(-1, keepdim=True)
    replacement = heights.gather(-1, centre).expand_as(heights)
    # Entirely off-course rows have no longitudinal evidence to invent.
    filtered = torch.where(inside | ~inside.any(-1, keepdim=True), heights, replacement)
    return torch.empty_like(raw_height).scatter_(1, order, filtered.flatten(1))


def locomotion_scan(env):
    from .course import course_state
    sensor = env.scene.sensors["height_scanner"]
    hits = sensor.data.ray_hits_w
    raw = (sensor.data.pos_w[:, 2:3]-hits[..., 2]-.5).clamp(-1., 1.)
    root = env.scene["robot"].data.root_pos_w
    q = env.scene["robot"].data.root_quat_w
    yaw = torch.atan2(2*(q[:, 0]*q[:, 3]+q[:, 1]*q[:, 2]), 1-2*(q[:, 2].square()+q[:, 3].square()))
    delta = hits[..., :2]-root[:, None, :2]
    local_x = delta[..., 0]*yaw.cos()[:, None]+delta[..., 1]*yaw.sin()[:, None]
    course_y = hits[..., 1]-env.scene.env_origins[:, None, 1]
    return longitudinal_scan(raw, local_x, course_y, course_state(env)["width"])


def physical_sample(env):
    from .course import course_state
    data, state = env.scene["robot"].data, course_state(env)
    # ObservationManager determines dimensions before the parent environment
    # installs its wheel indices. Resolve the same order during that phase.
    if not hasattr(env, "_quiet_body_ids"):
        names = [f"{leg}_wheel" for leg in ("fl", "fr", "hl", "hr")]
        env._quiet_body_ids, actual = env.scene["robot"].find_bodies(names, preserve_order=True)
        env._quiet_contact_ids, contact_names = env.scene.sensors["contact_forces"].find_bodies(names, preserve_order=True)
        if actual != names or contact_names != names:
            raise ValueError("Safe teacher wheel order changed")
    ids = env._quiet_body_ids
    q = data.root_quat_w
    yaw = torch.atan2(2*(q[:, 0]*q[:, 3]+q[:, 1]*q[:, 2]), 1-2*(q[:, 2].square()+q[:, 3].square()))
    axis = wheel_axis_world(data.body_link_quat_w[:, ids], env.cfg.stair_quiet, validate=False)
    omega = data.body_ang_vel_w[:, ids]
    velocity = data.body_lin_vel_w[:, ids]+torch.linalg.cross(
        omega, data.body_link_pos_w[:, ids]-data.body_com_pos_w[:, ids], dim=-1)
    return dict(position=data.body_link_pos_w[:, ids]-env.scene.env_origins[:, None, :],
        axis=axis, velocity=velocity, omega=omega,
        forces=env.scene.sensors["contact_forces"].data.net_forces_w[:, env._quiet_contact_ids],
        root=data.root_pos_w-env.scene.env_origins, root_velocity=data.root_lin_vel_w,
        yaw=yaw, upright=-data.projected_gravity_b[:, 2], height=state["height"], depth=state["depth"],
        width=state["width"], steps=state["steps"], dt=env.physics_dt)


def safety_geometry(d, s):
    pos, axis = d["position"], d["axis"]
    forward, side, top = foothold_regions(pos, axis, d["height"], d["depth"], d["width"], d["steps"], s)
    future = pos+d["velocity"]*s.prediction_s
    predicted_forward, _, _ = foothold_regions(future, axis, d["height"], d["depth"], d["width"], d["steps"], s)
    _, margin = wheel_margins(s, pos)
    extent = s.wheel_radius+s.wheel_half_length*axis[..., 1].abs()+margin
    half = d["width"][:, None]/2
    left, right = half-pos[..., 1]-extent, half+pos[..., 1]-extent
    future_left, future_right = half-future[..., 1]-extent, half+future[..., 1]-extent
    bottom = pos[..., 2]-s.wheel_radius-s.wheel_half_length*axis[..., 2].abs()
    f = d["forces"]
    supported = (f[..., 2] > s.support_force_N) & (
        f[..., 2] > torch.linalg.vector_norm(f[..., :2], dim=-1)) & ((bottom-top).abs() < .05)
    slip_speed = torch.linalg.vector_norm(capsule_contact_velocity(
        d["velocity"], d["omega"], axis, s.wheel_radius, s.wheel_half_length)[..., :2], dim=-1)
    return dict(forward=forward, side=side, left=left, right=right, top=top, clearance=bottom-top,
                predicted_forward=predicted_forward, predicted_left=future_left, predicted_right=future_right,
                supported=supported, slip_speed=slip_speed)


def safety_features(d, s):
    g = safety_geometry(d, s)
    feet = torch.stack((g["forward"]/.1, g["left"]/.1, g["right"]/.1,
        g["predicted_forward"]/.1, g["predicted_left"]/.1, g["predicted_right"]/.1,
        g["supported"].float(), g["slip_speed"]/.3, d["forces"][..., 2]/500.,
        d["velocity"][..., 0], d["velocity"][..., 1]), -1).flatten(1)
    body = torch.stack((d["root"][:, 1]/(d["width"]/2), d["root_velocity"][:, 1],
                        d["yaw"].sin(), d["yaw"].cos()), -1)
    result = torch.cat((feet, body), -1).clamp(-10., 10.)
    if result.shape[-1] != SAFETY_DIM:
        raise ValueError("Safe teacher feature layout changed")
    return result


def teacher_safety(env):
    return safety_features(physical_sample(env), env.cfg.stair_tail)
