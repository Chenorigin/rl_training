"""Deployment-only Safe geometry for box/plane and fixed triangle terrain.

Keep the checkpoint's safety48 order/scales and capsule footprint conventions.
Support faces come from current collision poses, never an overall bounding box.
Coplanar faces merge only when their union is convex; positive gaps remain gaps.
Nonconvex regions retain conservative piece boundaries. Before reading margins,
erode every edge by a rectangle enclosing the checkpoint's support footprint;
separate X/Y intercepts alone would miss an overhang across a rotated edge.
Prediction transports each face with its body's current rigid velocity. This is
a constant-velocity prediction, not a second physics solve or a terrain edit.
"""
from dataclasses import dataclass

import mujoco
import numpy as np
import torch

from task_support.stair_quiet_v1_width.geometry import capsule_contact_velocity, wheel_margins


@dataclass
class _Face:
    geom: int
    body: int
    point: np.ndarray
    normal: np.ndarray
    polygon: np.ndarray | None

    def __post_init__(self):
        if self.polygon is not None:
            p = self.polygon
            if np.sum(p[:, 0]*np.roll(p[:, 1], -1)-p[:, 1]*np.roll(p[:, 0], -1)) < 0:
                p = self.polygon = p[::-1].copy()
            edges = np.roll(p, -1, axis=0)-p
            self.normals = np.column_stack((edges[:, 1], -edges[:, 0]))
            self.normals /= np.linalg.norm(self.normals, axis=1)[:, None]
            self.limits = np.sum(self.normals*p, axis=1)

    def height(self, xy):
        return float(self.point[2]-np.dot(self.normal[:2], xy-self.point[:2])/self.normal[2])

    def contains(self, xy):
        return self.polygon is None or np.all(self.normals @ xy <= self.limits+1.e-8)

    def closest(self, xy):
        if self.contains(xy):
            return np.array(xy), 0.
        a = self.polygon
        edges = np.roll(a, -1, axis=0)-a
        t = np.clip(np.sum((xy-a)*edges, axis=1)/np.sum(edges*edges, axis=1), 0., 1.)
        points = a+t[:, None]*edges
        distances = np.linalg.norm(points-xy, axis=1)
        i = distances.argmin()
        return points[i], float(distances[i])

    def interval(self, xy, axis):
        """Intersection with a world X/Y line, expressed as absolute coordinates."""
        if self.polygon is None:
            return -np.inf, np.inf
        distance, direction = self.limits-self.normals @ xy, self.normals[:, axis]
        parallel = np.abs(direction) < 1.e-10
        if np.any(distance[parallel] < -1.e-8):
            return None
        pos, neg = direction > 1.e-10, direction < -1.e-10
        hi = np.min(distance[pos]/direction[pos], initial=np.inf)
        lo = np.max(distance[neg]/direction[neg], initial=-np.inf)
        return (xy[axis]+lo, xy[axis]+hi) if lo <= hi+1.e-8 else None


def _span(faces, xy, axis):
    intervals = sorted(i for f in faces if (i := f.interval(xy, axis)) is not None)
    if not intervals:
        # Outside every footprint: return negative clearance, never an AABB.
        distance = min(f.closest(xy)[1] for f in faces)
        return -distance, -distance
    merged = []
    for lo, hi in intervals:
        if merged and lo <= merged[-1][1]+1.e-8:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    lo, hi = max(merged, key=lambda span: min(xy[axis]-span[0], span[1]-xy[axis]))
    return xy[axis]-lo, hi-xy[axis]


def _coplanar(faces, reference):
    return [f for f in faces if np.dot(f.normal, reference.normal) > 1.-1.e-7
            and abs(np.dot(reference.normal, f.point-reference.point)) < 1.e-5]


def _area(polygon):
    return abs(np.sum(polygon[:, 0]*np.roll(polygon[:, 1], -1)
                      -polygon[:, 1]*np.roll(polygon[:, 0], -1)))/2 if len(polygon) >= 3 else 0.


def _clip(polygon, normal, limit):
    if not len(polygon):
        return polygon
    output = []
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        da, db = np.dot(normal, a)-limit, np.dot(normal, b)-limit
        if da <= 1.e-10:
            output.append(a)
        if (da <= 1.e-10) != (db <= 1.e-10):
            output.append(a+(b-a)*da/(da-db))
    return np.asarray(output).reshape(-1, 2)


def _hull(polygons):
    points = sorted(set(map(tuple, np.concatenate(polygons))))
    def turn(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    lower, upper = [], []
    for point in points:
        while len(lower) > 1 and turn(lower[-2], lower[-1], point) <= 1.e-12:
            lower.pop()
        lower.append(point)
    for point in reversed(points):
        while len(upper) > 1 and turn(upper[-2], upper[-1], point) <= 1.e-12:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1]+upper[:-1])


def _safe_regions(faces, extent_x, extent_y):
    if any(f.polygon is None for f in faces):
        return [next(f for f in faces if f.polygon is None)]
    polygons = [f.polygon for f in faces]
    reference = faces[0]
    # A convex hull is allowed ONLY if it adds no area to the true union.
    # This removes real shared seams without filling a gap or concave corner.
    while True:
        merged = False
        for i, a in enumerate(polygons):
            for j in range(i+1, len(polygons)):
                b = polygons[j]
                if np.any(a.max(axis=0) < b.min(axis=0)-1.e-10) or np.any(b.max(axis=0) < a.min(axis=0)-1.e-10):
                    continue
                boundary = _Face(-1, -1, reference.point, reference.normal, b)
                intersection = a
                for n, limit in zip(boundary.normals, boundary.limits):
                    intersection = _clip(intersection, n, limit)
                hull = _hull([a, b])
                union_area = _area(a)+_area(b)-_area(intersection)
                if abs(_area(hull)-union_area) <= 1.e-10*max(1., _area(hull)):
                    polygons[i] = hull
                    polygons.pop(j)
                    merged = True
                    break
            if merged:
                break
        if not merged:
            break
    result = []
    for polygon in polygons:
        boundary = _Face(-1, -1, reference.point, reference.normal, polygon)
        eroded = polygon
        for n, limit in zip(boundary.normals, boundary.limits):
            eroded = _clip(eroded, n, limit-abs(n[0])*extent_x-abs(n[1])*extent_y)
        if _area(eroded) > 1.e-12:
            result.append(_Face(-1, -1, reference.point, reference.normal, eroded))
    return result


class NativeTerrainGeometry:
    def __init__(self, model, geom_ids, flex_ids=()):
        self.model, self.geom_ids = model, tuple(sorted(geom_ids))
        self.flex_ids = tuple(sorted(flex_ids))
        if any(model.geom_type[g] not in (mujoco.mjtGeom.mjGEOM_BOX,
                                         mujoco.mjtGeom.mjGEOM_PLANE) for g in self.geom_ids):
            raise ValueError("Safe native geometry supports box/plane collision terrain only")
        for flex in self.flex_ids:
            start, count = int(model.flex_vertadr[flex]), int(model.flex_vertnum[flex])
            if (model.flex_dim[flex] != 2 or model.flex_radius[flex] != 0
                    or model.flex_group[flex] != 0 or not model.flex_rigid[flex]
                    or not count or not np.all(model.flex_vertbodyid[start:start+count] == 0)
                    or not (model.flex_contype[flex] or model.flex_conaffinity[flex])):
                raise ValueError("Safe native flex terrain requires world-fixed group-0 zero-radius collision triangles")
        if not self.geom_ids and not self.flex_ids:
            raise ValueError("Safe native geometry requires collision terrain")

    def _faces(self, data, horizon=0.):
        faces, twists = [], {}
        for geom in self.geom_ids:
            body = int(self.model.geom_bodyid[geom])
            if body not in twists:
                twists[body] = np.zeros(6)
                if body:
                    mujoco.mj_objectVelocity(self.model, data, mujoco.mjtObj.mjOBJ_BODY,
                                             body, twists[body], 0)
            omega, velocity = twists[body][:3], twists[body][3:]
            angle = np.linalg.norm(omega)*horizon
            delta = np.eye(3)
            if angle > 1.e-12:
                axis = omega/np.linalg.norm(omega)
                skew = np.array([[0., -axis[2], axis[1]], [axis[2], 0., -axis[0]],
                                 [-axis[1], axis[0], 0.]])
                delta += np.sin(angle)*skew+(1.-np.cos(angle))*(skew @ skew)
            centre = data.xipos[body] if body else np.zeros(3)
            point = centre+velocity*horizon+delta @ (data.geom_xpos[geom]-centre)
            rotation = delta @ data.geom_xmat[geom].reshape(3, 3)
            if self.model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_PLANE:
                normal, polygon = rotation[:, 2], None
            else:
                k = int(np.abs(rotation[2]).argmax())
                normal = rotation[:, k]*np.sign(rotation[2, k])
                point = point+normal*self.model.geom_size[geom, k]
                other = [i for i in range(3) if i != k]
                half = rotation[:, other]*self.model.geom_size[geom, other]
                polygon = np.array([point+half @ signs for signs in
                                    ((-1, -1), (1, -1), (1, 1), (-1, 1))])[:, :2]
            # Vertical risers/legs are not horizontal support surfaces.
            if normal[2] >= .8:
                faces.append(_Face(geom, body, point, normal, polygon))
        if self.flex_ids:
            twists.setdefault(0, np.zeros(6))
        for flex in self.flex_ids:
            start = int(self.model.flex_elemdataadr[flex])
            count = int(self.model.flex_elemnum[flex])
            vertices = int(self.model.flex_vertadr[flex]) + self.model.flex_elem[start:start+3*count].reshape(-1, 3)
            for index, triangle in enumerate(data.flexvert_xpos[vertices]):
                normal = np.cross(triangle[1]-triangle[0], triangle[2]-triangle[0])
                length = np.linalg.norm(normal)
                if length <= 1.e-12:
                    raise ValueError("Safe native flex terrain contains a degenerate triangle")
                normal /= length
                if normal[2] < 0:
                    normal = -normal  # Collision triangles support from either side.
                if normal[2] >= .8:
                    # Negative global element IDs cannot collide with geom IDs;
                    # they also associate the same fixed face across prediction.
                    face_id = -1-int(self.model.flex_elemadr[flex])-index
                    faces.append(_Face(face_id, 0, triangle[0].copy(), normal, triangle[:, :2].copy()))
        if not faces:
            raise ValueError("Safe native geometry has no upward support face")
        return faces, twists

    @staticmethod
    def _select(faces, position, axis, settings):
        xy = position[:2]
        bottom = position[2]-settings.wheel_radius-settings.wheel_half_length*abs(axis[2])
        reach = settings.wheel_radius+settings.wheel_half_length*np.linalg.norm(axis[:2])
        near, below = [], []
        for face in faces:
            closest, distance = face.closest(xy)
            top = face.height(closest)
            if distance == 0. and top <= bottom+.05:
                below.append((top, face))
            if distance > reach or abs(bottom-top) >= .05:
                continue
            # Do not select a hidden frame/leg top underneath the deck.
            if any(other.contains(closest) and top+1.e-5 < other.height(closest) <= bottom+.05
                   for other in faces):
                continue
            near.append((abs(bottom-top), distance, face))
        if near:
            return min(near, key=lambda item: item[:2])[2]
        if below:
            return max(below, key=lambda item: item[0])[1]
        return min(faces, key=lambda f: f.height(xy))

    def features(self, data, sample, settings):
        faces, twists = self._faces(data)
        future_faces, _ = self._faces(data, settings.prediction_s)
        future_by_geom = {f.geom: f for f in future_faces}
        position, axis, velocity, forces = (sample[k][0].numpy() for k in
                                            ("position", "axis", "velocity", "forces"))
        tread, lateral = (v.numpy() for v in wheel_margins(settings, sample["position"]))
        rolling = capsule_contact_velocity(sample["velocity"], sample["omega"], sample["axis"],
                                            settings.wheel_radius, settings.wheel_half_length)[0].numpy()
        feet, region_cache = [], {}
        for i in range(4):
            face = self._select(faces, position[i], axis[i], settings)
            regions = _coplanar(faces, face)
            extent_x = settings.contact_patch_radius+settings.wheel_half_length*abs(axis[i, 0])+tread[i]
            extent_y = settings.wheel_radius+settings.wheel_half_length*abs(axis[i, 1])+lateral[i]

            def margins(regions, xy):
                key = (tuple(id(f) for f in regions), float(extent_x), float(extent_y))
                if key not in region_cache:
                    region_cache[key] = _safe_regions(regions, extent_x, extent_y)
                safe = region_cache[key]
                if not safe:
                    return -1., -1., -1.
                back, front = _span(safe, xy, 0)
                right, left = _span(safe, xy, 1)
                return min(back, front), left, right

            current = margins(regions, position[i, :2])
            future = position[i]+velocity[i]*settings.prediction_s
            if face.geom in future_by_geom:
                reference = future_by_geom[face.geom]
                predicted = margins(_coplanar(future_faces, reference), future[:2])
            else:
                predicted = (-1., -1., -1.)
            bottom = position[i, 2]-settings.wheel_radius-settings.wheel_half_length*abs(axis[i, 2])
            supported = (forces[i, 2] > settings.support_force_N
                         and forces[i, 2] > np.linalg.norm(forces[i, :2])
                         and abs(bottom-face.height(position[i, :2])) < .05)
            offset = -settings.wheel_half_length*np.sign(axis[i, 2])*axis[i]
            offset[2] -= settings.wheel_radius
            omega, translation = twists[face.body][:3], twists[face.body][3:]
            centre = data.xipos[face.body] if face.body else np.zeros(3)
            support_velocity = translation+np.cross(omega, position[i]+offset-centre)
            relative = rolling[i]-support_velocity
            slip = np.linalg.norm(relative-np.dot(relative, face.normal)*face.normal)
            feet.extend([*(np.array(current)/.1), *(np.array(predicted)/.1),
                         float(supported), slip/.3, forces[i, 2]/500., *velocity[i, :2]])
        body = [sample["root"][0, 1]/(sample["width"][0]/2), sample["root_velocity"][0, 1],
                sample["yaw"][0].sin(), sample["yaw"][0].cos()]
        result = torch.tensor([*feet, *body], dtype=torch.float32).reshape(1, 48).clamp(-10., 10.)
        return result
