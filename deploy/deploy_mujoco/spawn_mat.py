"""A visible physical starting mat shared by every MuJoCo deployment entry."""

import math

import mujoco
import numpy as np

from terrain_loader import is_terrain_geom, static_terrain_flex_ids, terrain_ray


# Full dimensions in metres. Terrain XML may override with spawn_mat_size.
DEFAULT_SIZE = (1.4, 1.1, 0.02)
MAT_NAME = "terrain_spawn_mat"
# A rigid mat can bridge shallow floorboard/tile joints, not steps or pits.
MAX_SUPPORT_RELIEF = 0.005


def _clip(polygon, normal, limit):
    """Clip a convex polygon to normal @ xy <= limit."""
    output = []
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        da, db = normal @ a-limit, normal @ b-limit
        if da <= 0:
            output.append(a)
        if (da <= 0) != (db <= 0):
            output.append(a+(b-a)*da/(da-db))
    return np.asarray(output).reshape(-1, 2)


def _area(polygon):
    if len(polygon) < 3:
        return 0.
    return abs(np.sum(polygon[:, 0]*np.roll(polygon[:, 1], -1)
                      -polygon[:, 1]*np.roll(polygon[:, 0], -1)))/2


def _check_coverage(model, data, flexes, xy, rotation, size, base):
    """Subtract exact static support footprints; ray spacing cannot hide holes.

    Certification uses planes, box faces and fixed triangular flexes. Complex
    meshes/heightfields may be present elsewhere; place the mat on a supported
    primitive patch (or disable it) when those are the only spawn surface.
    """
    rectangle = np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])*size[:2]/2
    uncovered = [rectangle]
    faces = []
    for geom in range(model.ngeom):
        if not is_terrain_geom(model, geom) or model.geom_bodyid[geom] != 0:
            continue
        transform = data.geom_xmat[geom].reshape(3, 3)
        point = data.geom_xpos[geom].copy()
        if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_PLANE:
            faces.append((point, transform[:, 2], None))
        elif model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_BOX:
            k = int(np.abs(transform[2]).argmax())
            normal = transform[:, k]*np.sign(transform[2, k])
            point += normal*model.geom_size[geom, k]
            others = [i for i in range(3) if i != k]
            half = transform[:, others]*model.geom_size[geom, others]
            polygon = np.array([point+half @ signs for signs in
                                ((-1, -1), (1, -1), (1, 1), (-1, 1))])[:, :2]
            faces.append((point, normal, polygon))
    for flex in flexes:
        start = int(model.flex_elemdataadr[flex])
        if model.flex_dim[flex] != 2 or model.flex_radius[flex] != 0:
            continue
        vertices = (int(model.flex_vertadr[flex])
                    + model.flex_elem[start:start+3*int(model.flex_elemnum[flex])].reshape(-1, 3))
        for triangle in data.flexvert_xpos[vertices]:
            normal = np.cross(triangle[1]-triangle[0], triangle[2]-triangle[0])
            if normal[2] < 0:
                normal = -normal
            faces.append((triangle[0], normal, triangle[:, :2]))
    for point, normal, polygon in faces:
        if normal[2] <= 1.e-10:
            continue
        support = rectangle.copy() if polygon is None else (polygon-xy) @ rotation
        if _area(support) < 1.e-12:
            continue
        if np.sum(support[:, 0]*np.roll(support[:, 1], -1)
                  -support[:, 1]*np.roll(support[:, 0], -1)) < 0:
            support = support[::-1]
        for n, bound in (([1., 0.], size[0]/2), ([-1., 0.], size[0]/2),
                         ([0., 1.], size[1]/2), ([0., -1.], size[1]/2)):
            support = _clip(support, np.asarray(n), bound)
        height = point[2]-normal[:2] @ (xy-point[:2])/normal[2]
        slope = -(normal[:2] @ rotation)/normal[2]
        support = _clip(support, slope, base+1.e-7-height)
        support = _clip(support, -slope, height-base+MAX_SUPPORT_RELIEF+1.e-7)
        if _area(support) < 1.e-12:
            continue
        remainder = []
        for region in uncovered:
            inside = region
            for a, b in zip(support, np.roll(support, -1, axis=0)):
                edge = b-a
                # Clipping at an existing vertex can repeat that endpoint.
                # A zero edge has no half-space and must not restore inside.
                if np.linalg.norm(edge) < 1.e-12:
                    continue
                n = np.array([edge[1], -edge[0]])
                outside = _clip(inside, -n, -n @ a)
                if _area(outside) > 1.e-12:
                    remainder.append(outside)
                inside = _clip(inside, n, n @ a)
                if not len(inside):
                    break
        uncovered = remainder
        if sum(map(_area, uncovered)) <= 1.e-8:
            return
    raise ValueError("Spawn mat needs continuous fixed box/plane/flex support; its footprint "
                     "crosses a gap or unsupported surface. Move --spawn-mat-xy or use --no-spawn-mat")


def add_spawn_arguments(parser):
    parser.add_argument("--spawn-xy", type=float, nargs=2, metavar=("X", "Y"),
                        help="Robot reset X/Y (metres); use --spawn-mat-xy to move only the mat")
    parser.add_argument("--spawn-mat-xy", type=float, nargs=2, metavar=("X", "Y"),
                        help="Mat centre X/Y only (metres), leaving robot spawn unchanged; overrides XML spawn_mat_xy")
    parser.add_argument("--spawn-mat-yaw-deg", type=float,
                        help="Rotate only the mat (degrees); overrides XML spawn_mat_yaw_deg")
    parser.add_argument("--spawn-mat-size", type=float, nargs=3, metavar=("LENGTH", "WIDTH", "THICKNESS"),
                        help="Mat full dimensions (metres); thickness sets the height difference above its support")
    parser.add_argument("--no-spawn-mat", action="store_true",
                        help="Disable the starting mat and use the original terrain support")


def add_spawn_mat(spec, terrain, args):
    """Place a thin, fixed mat on an existing level, static support surface.

    Probe the actual compiled collision world, including open-stair flex faces.
    Do not turn a sloping, gapped or movable support into a floating platform.
    Mat placement never writes the robot's reset position or yaw. When mat
    settings are omitted, initially centre it on the robot for compatibility.
    """
    if getattr(args, "no_spawn_mat", False):
        return
    def setting(name, default):
        value = getattr(args, name, None)
        return terrain.parameters.get(name, default) if value is None else value

    mat_xy = np.asarray(setting("spawn_mat_xy", terrain.spawn_xy), dtype=float)
    mat_yaw = np.asarray(setting("spawn_mat_yaw_deg", [math.degrees(terrain.spawn_yaw)]), dtype=float).reshape(-1)
    if mat_xy.shape != (2,) or not np.isfinite(mat_xy).all():
        raise ValueError("spawn_mat_xy must contain finite x y")
    if mat_yaw.shape != (1,) or not np.isfinite(mat_yaw).all():
        raise ValueError("spawn_mat_yaw_deg must contain one finite angle")
    mat_yaw = math.radians(float(mat_yaw[0]))
    size = np.asarray(setting("spawn_mat_size", DEFAULT_SIZE), dtype=float)
    if size.shape != (3,) or not np.isfinite(size).all() or (size <= 0).any():
        raise ValueError("spawn_mat_size must contain positive finite length width thickness")
    if any(g.name == MAT_NAME for g in spec.geoms):
        raise ValueError(f"Terrain XML uses reserved name {MAT_NAME}")
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    groups = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
    flexes = static_terrain_flex_ids(model)
    c, s = math.cos(mat_yaw), math.sin(mat_yaw)
    rotation = np.array([[c, -s], [s, c]])
    heights = []
    # Include all corners/edges and centre; bound setup work for arbitrary XML sizes.
    for x in np.linspace(-size[0]/2, size[0]/2, 9):
        for y in np.linspace(-size[1]/2, size[1]/2, 9):
            xy = mat_xy + rotation @ [x, y]
            origin = np.array([*xy, terrain.ray_top])
            distance, geom, flex = terrain_ray(model, data, origin,
                                               np.array([0., 0., -1.]), groups, flexes)
            if (distance < 0 or (geom >= 0 and (not is_terrain_geom(model, geom)
                                               or model.geom_bodyid[geom] != 0))
                    or (geom < 0 and flex < 0)):
                raise ValueError("Spawn mat needs fixed terrain underneath its whole footprint; "
                                 "move --spawn-mat-xy to a level platform or use --no-spawn-mat")
            heights.append(terrain.ray_top-distance)
    if np.ptp(heights) > MAX_SUPPORT_RELIEF:
        raise ValueError("Spawn mat crosses a slope, step or gap; move --spawn-mat-xy to a level "
                         "platform, reduce XML spawn_mat_size or use --no-spawn-mat")
    base = max(heights)
    _check_coverage(model, data, flexes, mat_xy, rotation, size, base)
    centre = [*mat_xy, base+size[2]/2]
    spec.worldbody.add_geom(name=MAT_NAME, type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=centre, size=size/2,
        quat=[math.cos(mat_yaw/2), 0., 0., math.sin(mat_yaw/2)],
        rgba=[0.10, 0.55, 0.65, 1.], contype=1, conaffinity=1, condim=3, group=0,
        friction=terrain.parameters["friction"])
    terrain.ray_top = max(terrain.ray_top, base+size[2]+1.)
    terrain.spawn_mat = dict(centre_xyz=centre, size=size.tolist(), yaw=mat_yaw,
                             top_z=base+size[2])


def validate_spawn_mat(model, terrain):
    """Reject intervening obstacles missed between the support ray samples."""
    if terrain.spawn_mat is None:
        return
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    mat = model.geom(MAT_NAME).id
    # Fixed geoms on body 0 do not produce solver contact pairs with each
    # other. Query pair distances explicitly instead of reading data.contact.
    for other in range(model.ngeom):
        if other == mat or not is_terrain_geom(model, other):
            continue
        distance = mujoco.mj_geomDistance(model, data, mat, other, 1.e-4, None)
        if distance < -1.e-4:
            raise ValueError("Spawn mat intersects terrain; move --spawn-mat-xy to a clear level platform")
