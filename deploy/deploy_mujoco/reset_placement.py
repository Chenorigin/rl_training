"""Place a reset pose just above its first actual terrain collision.

Only kinematics/collision detection runs here: there is no settling rollout,
controller update, or change to the model's joint reference pose.
"""

from __future__ import annotations

import mujoco
import numpy as np


RESET_CLEARANCE = 0.001
SEARCH_STEP = 0.002
MAX_SEARCH_STEPS = 4096
BISECTION_STEPS = 24


def _world_bounds(model, data, geom):
    """MuJoCo's local collision AABB, transformed conservatively to world."""
    rotation = data.geom_xmat[geom].reshape(3, 3)
    local = model.geom_aabb[geom]
    center = data.geom_xpos[geom] + rotation @ local[:3]
    extent = np.abs(rotation) @ local[3:]
    return center - extent, center + extent


def _overlaps_xy(lower, upper, robot_lower, robot_upper):
    return bool(np.all(upper[:2] >= robot_lower[:2])
                and np.all(lower[:2] <= robot_upper[:2]))


def place_on_terrain(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    root_body_id: int,
    root_qpos_adr: int,
    terrain_geom_ids,
    terrain_flex_ids=(),
    clearance: float = RESET_CLEARANCE,
) -> float:
    """Translate the supplied robot pose vertically to its first terrain contact.

    The caller sets the desired joints, XY and orientation before calling. The
    whole robot's collision geometry is used, including the belly in a folded
    pose. Visual-only geoms and robot self-contact never determine the height.
    Collision masks, primitive/mesh geometry, ramps and fixed terrain flexes
    are handled by MuJoCo's own collision detector.

    A conservative AABB provides a collision-free starting height. A bounded
    downward search brackets the first contact, then bisection removes the
    search-step error. Search increments are capped by the smallest robot
    collision AABB thickness; this routine targets the volumetric robot geoms
    used by deployment, not zero-thickness robot surfaces. No support, an
    obstructing vertical wall, or exhaustion of the search bound raises an
    error instead of returning an airborne reset pose.

    Only the free root's Z coordinate changes. In particular, model.qpos0 must
    never be overwritten with reset joint angles: it also defines joint zero
    references. The viewer's built-in reset must be routed through the caller.
    """
    if not np.isfinite(clearance) or clearance < 0:
        raise ValueError("Reset clearance must be finite and non-negative")
    terrain_geoms = frozenset(int(g) for g in terrain_geom_ids
                              if model.geom_contype[g] or model.geom_conaffinity[g])
    terrain_flexes = frozenset(int(f) for f in terrain_flex_ids)
    robot_bodies = {int(root_body_id)}
    for body in range(root_body_id + 1, model.nbody):
        if int(model.body_parentid[body]) in robot_bodies:
            robot_bodies.add(body)
    robot_geoms = frozenset(g for g in range(model.ngeom)
                           if int(model.geom_bodyid[g]) in robot_bodies
                           and (model.geom_contype[g] or model.geom_conaffinity[g]))
    if not robot_geoms:
        raise RuntimeError("Reset robot has no collision geometry")
    if not terrain_geoms and not terrain_flexes:
        raise RuntimeError("Reset has no collision terrain")

    original_z = float(data.qpos[root_qpos_adr + 2])
    mujoco.mj_fwdPosition(model, data)
    bounds = [_world_bounds(model, data, g) for g in robot_geoms]
    robot_lower = np.min([lo for lo, _ in bounds], axis=0)
    robot_upper = np.max([hi for _, hi in bounds], axis=0)
    positive_thicknesses = [float(hi[2] - lo[2]) for lo, hi in bounds if hi[2] > lo[2]]
    if not positive_thicknesses:
        raise RuntimeError("Reset requires volumetric robot collision geometry")
    search_step = min(SEARCH_STEP, min(positive_thicknesses) / 4)

    terrain_tops = []
    for geom in terrain_geoms:
        if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_PLANE:
            normal = data.geom_xmat[geom].reshape(3, 3)[:, 2]
            if normal[2] <= 1e-6:
                continue  # A vertical/downward-facing plane is not support.
            point = data.geom_xpos[geom]
            for x in (robot_lower[0], robot_upper[0]):
                for y in (robot_lower[1], robot_upper[1]):
                    terrain_tops.append(float(point[2] -
                        (normal[0] * (x - point[0]) + normal[1] * (y - point[1])) / normal[2]))
        else:
            lower, upper = _world_bounds(model, data, geom)
            if _overlaps_xy(lower, upper, robot_lower, robot_upper):
                terrain_tops.append(float(upper[2]))
    for flex in terrain_flexes:
        start, count = int(model.flex_vertadr[flex]), int(model.flex_vertnum[flex])
        if not count:
            continue
        vertices = data.flexvert_xpos[start:start + count]
        radius = float(model.flex_radius[flex])
        lower, upper = vertices.min(axis=0) - radius, vertices.max(axis=0) + radius
        if _overlaps_xy(lower, upper, robot_lower, robot_upper):
            terrain_tops.append(float(upper[2]))
    if not terrain_tops:
        raise RuntimeError("No collision terrain beneath the reset robot footprint")

    def contact_state(z):
        data.qpos[root_qpos_adr + 2] = z
        mujoco.mj_fwdPosition(model, data)
        touching, supported = False, False
        for contact in data.contact:
            if contact.dist > 0:
                continue
            for robot_side in (0, 1):
                other = 1 - robot_side
                if (int(contact.geom[robot_side]) in robot_geoms
                        and (int(contact.geom[other]) in terrain_geoms
                             or int(contact.flex[other]) in terrain_flexes)):
                    touching = True
                    # Contact normals point from side 0 towards side 1.
                    upward = float(contact.frame[2]) * (1 if robot_side == 1 else -1)
                    supported = supported or upward > 1e-6
        return touching, supported

    try:
        high = max(terrain_tops) - float(robot_lower[2] - original_z) + clearance + search_step
        if contact_state(high)[0]:
            raise RuntimeError("Could not find a collision-free reset starting height")
        low = high
        for _ in range(MAX_SEARCH_STEPS):
            low -= search_step
            touching, supported = contact_state(low)
            if touching:
                break
            high = low
        else:
            raise RuntimeError("No reset terrain contact within the bounded placement search")
        for _ in range(BISECTION_STEPS):
            middle = (low + high) / 2
            if contact_state(middle)[0]:
                low = middle
            else:
                high = middle
        if not contact_state(low)[1]:
            raise RuntimeError("Reset pose hits vertical terrain without upward support")
        data.qpos[root_qpos_adr + 2] = high + clearance
        mujoco.mj_forward(model, data)
        return float(data.qpos[root_qpos_adr + 2])
    except Exception:
        data.qpos[root_qpos_adr + 2] = original_z
        mujoco.mj_forward(model, data)
        raise
