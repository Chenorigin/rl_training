"""Parameterized terrain MJCFs and native worldbody terrain fragments.

All sizes are metres; angles suffixed _deg are degrees. The XML custom
numerics are the source of truth. Geometries are rebuilt on each launch.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


TERRAIN_DIR = Path(__file__).resolve().parent / "terrains"
PRESETS = {
    "flat": "flat.xml", "ascent": "stairs_ascent.xml", "descent": "stairs_descent.xml",
    "ascent_open": "stairs_ascent_open.xml", "ascent_walls": "stairs_ascent_walls.xml",
    "platform": "platform.xml",
    "slope_up": "slope_up.xml", "slope_down": "slope_down.xml", "rough": "rough.xml",
    "obstacles": "obstacles.xml", "hurdles": "hurdles.xml", "mixed": "mixed.xml",
}
KINDS = (set(PRESETS) - {"ascent_open", "ascent_walls"}) | {"custom"}
STAIR_OVERRIDES = {"stair_height": "step_height", "tread_depth": "tread_depth",
                   "stair_count": "step_count", "stair_start": "start_x"}


@dataclass
class Terrain:
    path: Path
    kind: str
    parameters: dict[str, list[float]]
    spawn_xy: tuple[float, float]
    spawn_yaw: float
    goal_x: float
    course_width: float
    ray_top: float
    spawn_mat: dict | None = None

    def summary(self) -> dict:
        result = {"terrain": self.kind, "terrain_xml": str(self.path),
                "terrain_parameters": self.parameters, "goal_x": self.goal_x,
                "course_width": self.course_width}
        result["spawn_xy"] = list(self.spawn_xy)
        result["spawn_mat"] = self.spawn_mat
        if self.parameters.get("open_risers") == [1.0]:
            result["open_risers"] = True
            result["surface_collision"] = "treads: world-fixed zero-radius triangular flex; no riser/side/bottom faces"
        if "walls_enabled" in self.parameters:
            result["walls_enabled"] = self.parameters["walls_enabled"] == [1.0]
        return result


def static_terrain_flex_ids(model):
    """Only group-0 flexes whose vertices are all fixed to the world."""
    result = []
    for flex in range(getattr(model, "nflex", 0)):
        start, count = int(model.flex_vertadr[flex]), int(model.flex_vertnum[flex])
        if model.flex_group[flex] == 0 and count and np.all(model.flex_vertbodyid[start:start+count] == 0):
            result.append(flex)
    return tuple(result)


def terrain_ray(model, data, origin, direction, groups, flex_ids=None, *, normal=None):
    """Nearest ordinary geom OR actual static flex face; mj_ray omits flexes."""
    geom = np.full(1, -1, dtype=np.int32)
    if normal is None:
        distance = mujoco.mj_ray(model, data, origin, direction, groups, True, -1, geom)
    else:
        normal[:] = np.nan
        distance = mujoco.mj_ray(model, data, origin, direction, groups, True, -1, geom, normal)
    hit_geom, hit_flex = int(geom[0]), -1
    for flex in static_terrain_flex_ids(model) if flex_ids is None else flex_ids:
        if normal is None:
            value = mujoco.mj_rayFlex(model, data, 0, False, False, True, False,
                                     flex, origin, direction)
        else:
            face_normal = np.full(3, np.nan)
            value = mujoco.mj_rayFlex(model, data, 0, False, False, True, False,
                                     flex, origin, direction, None, face_normal)
        if value >= 0 and (distance < 0 or value < distance):
            distance, hit_geom, hit_flex = value, -1, flex
            if normal is not None:
                normal[:] = face_normal
    return distance, hit_geom, hit_flex


def is_terrain_geom(model, geom):
    """Collision terrain, including passive bodies attached with our prefix.

    Body 0 remains the contact-reader sentinel for terrain, not a claim that
    the physical object is fixed. Robot bodies must never become terrain.
    """
    return (geom >= 0 and model.geom_group[geom] == 0
            and bool(model.geom_contype[geom] or model.geom_conaffinity[geom])
            and (model.geom_bodyid[geom] == 0
                 or model.geom(geom).name.startswith("terrain_")))


def terrain_contact_bodies(model, contact, flex_ids):
    """Map fixed/movable terrain to sentinel 0; preserve robot body IDs."""
    bodies = []
    for side in range(2):
        geom = int(contact.geom[side])
        if geom >= 0:
            body = int(model.geom_bodyid[geom])
            bodies.append(0 if is_terrain_geom(model, geom) else (body or None))
        elif int(contact.flex[side]) in flex_ids:
            bodies.append(0)
        else:
            bodies.append(None)
    return tuple(bodies)


def read_config(path: Path, overrides: dict | None = None):
    root = ET.parse(path).getroot()
    if root.tag != "mujoco":
        raise ValueError(f"Terrain XML must have a <mujoco> root: {path}")
    kind_element = root.find("./custom/text[@name='terrain_kind']")
    kind = kind_element.get("data") if kind_element is not None else "custom"
    if kind not in KINDS:
        raise ValueError(f"Unknown terrain_kind {kind!r} in {path}")
    parameters = {}
    for node in root.findall("./custom/numeric"):
        name = node.get("name")
        if not name or name in parameters:
            raise ValueError(f"Missing/duplicate numeric name in {path}: {name}")
        values = [float(v) for v in node.get("data", "").split()]
        if not values or not np.isfinite(values).all():
            raise ValueError(f"Invalid numeric {name} in {path}")
        parameters[name] = values
    for name, value in (overrides or {}).items():
        if value is not None:
            if name == "spawn_xy":
                if len(value) != 2 or not np.isfinite(value).all():
                    raise ValueError("spawn_xy must contain finite x y")
                parameters[name] = [float(v) for v in value]
                continue
            if kind not in {"ascent", "descent"}:
                raise ValueError("Stair CLI overrides require an ascent/descent terrain XML")
            if not math.isfinite(value):
                raise ValueError(f"Stair override {name} must be finite")
            parameters[STAIR_OVERRIDES[name]] = [float(value)]
    return kind, parameters


def attach_terrain(robot_spec: mujoco.MjSpec, path: Path, overrides: dict | None = None) -> Terrain:
    path = path.resolve()
    kind, p = read_config(path, overrides)
    flag = p.get("open_risers", [0.0])
    if len(flag) != 1 or flag[0] not in (0.0, 1.0):
        raise ValueError("open_risers must be a single 0 or 1")
    open_risers = flag[0] == 1.0
    if open_risers and kind != "ascent":
        raise ValueError("open_risers requires terrain_kind=ascent")
    wall_flag = p.get("walls_enabled", [0.0])
    if len(wall_flag) != 1 or wall_flag[0] not in (0.0, 1.0):
        raise ValueError("walls_enabled must be a single 0 or 1")
    walls_enabled = wall_flag[0] == 1.0
    if walls_enabled and kind != "ascent":
        raise ValueError("walls_enabled requires terrain_kind=ascent")
    pending_surface = None

    def scalar(name: str, *, positive: bool = False, integer: bool = False) -> float:
        values = p.get(name)
        if values is None or len(values) != 1:
            raise ValueError(f"{path.name}: {name} must contain one number")
        value = values[0]
        if not math.isfinite(value) or (positive and value <= 0) or (integer and value != int(value)):
            raise ValueError(f"{path.name}: invalid {name}={value}")
        return value

    width = scalar("width", positive=True)
    friction = p.get("friction", [])
    if len(friction) != 3 or min(friction) < 0 or friction[0] <= 0:
        raise ValueError("friction must contain positive sliding and nonnegative torsional/rolling values")
    spawn = p.get("spawn_xy", [])
    if len(spawn) != 2:
        raise ValueError("spawn_xy must contain x y")
    yaw = math.radians(scalar("spawn_yaw_deg"))
    if any(item.name.startswith("terrain_") for item in (*robot_spec.geoms, *robot_spec.bodies)):
        raise ValueError("Robot XML uses reserved terrain_ body/geom namespace")
    terrain_spec = mujoco.MjSpec.from_file(str(path))
    components = scalar("dynamic_components", positive=True, integer=True) if "dynamic_components" in p else 0
    internal_count = (scalar("passive_internal_joints", positive=True, integer=True)
                      if "passive_internal_joints" in p else 0)
    if terrain_spec.worldbody.bodies or components or internal_count:
        if kind != "custom" or len(terrain_spec.worldbody.bodies) != components:
            raise ValueError("Movable terrain requires custom kind and matching dynamic_components count")
        # Each assembly remains free relative to world. Explicitly declared
        # internal spring hinges approximate support compliance, never anchoring
        # the assembly to the world or prescribing its motion.
        bodies = list(terrain_spec.bodies)[1:]
        roots = list(terrain_spec.worldbody.bodies)
        if any(
                len(b.joints) != 1 or b.joints[0].type != mujoco.mjtJoint.mjJNT_FREE
                for b in roots) or any(b.mocap or b.gravcomp != 0 for b in bodies):
            raise ValueError("Each terrain component requires one free root, gravity and no mocap driver")
        if any(np.any(np.asarray(b.joints[0].stiffness) != 0)
               or np.any(np.asarray(b.joints[0].damping) != 0)
               or b.joints[0].frictionloss != 0 for b in roots):
            raise ValueError("Terrain free roots must have zero stiffness, damping and frictionloss")
        # Uncompiled MjsBody.id is -1 for every non-world body.
        children = [b for b in bodies if b.parent != terrain_spec.worldbody]
        joints = [j for b in children for j in b.joints]
        if len(joints) != internal_count or any(not b.joints for b in children):
            raise ValueError("Terrain internal joints require matching passive_internal_joints count")
        if any(j.type != mujoco.mjtJoint.mjJNT_HINGE
               or j.limited != mujoco.mjtLimited.mjLIMITED_TRUE
               or not (j.range[0] < 0 < j.range[1])
               or np.asarray(j.stiffness).flat[0] <= 0 or np.asarray(j.damping).flat[0] <= 0
               or j.springref != 0 for j in joints):
            raise ValueError("Terrain internal joints require bounded zero-rest spring hinges with positive damping")
        if terrain_spec.actuators or terrain_spec.equalities or terrain_spec.tendons:
            raise ValueError("Movable terrain must be passive, without actuators, equality constraints or tendons")
    for geom in terrain_spec.geoms:
        collides = bool(geom.contype or geom.conaffinity)
        if components and collides and not geom.name:
            raise ValueError("Movable terrain collision geoms require names for sensor/contact identification")
        if (collides and geom.group != 0) or (not collides and geom.group != 2):
            raise ValueError("Terrain collision geoms require group 0; display-only geoms require group 2")
    # Replace the robot file's floor, keeping its bodies, assets and lighting.
    for geom in list(robot_spec.worldbody.geoms):
        if geom.group == 0 and (geom.contype or geom.conaffinity):
            robot_spec.delete(geom)
    common = dict(contype=1, conaffinity=1, condim=3, group=0, friction=friction)
    if open_risers and terrain_spec.worldbody.geoms:
        raise ValueError("Open stairs generate their own surfaces; native geoms could hide risers")
    floor_z = scalar("floor_height") if open_risers else 0.0
    if floor_z > 0:
        raise ValueError("floor_height must not exceed the approach tread height (zero)")
    if not any(g.type == mujoco.mjtGeom.mjGEOM_PLANE for g in terrain_spec.worldbody.geoms):
        terrain_spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                                       size=[100, 100, 0.1], pos=[0, 0, floor_z],
                                       rgba=[0.32, 0.35, 0.38, 1], **common)
    highest = 0.0

    def box(name, x0, x1, top, y=0.0, box_width=width, color=(0.48, 0.52, 0.59, 1)):
        nonlocal highest
        if top <= 0:
            return
        highest = max(highest, top)
        terrain_spec.worldbody.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[(x0 + x1) / 2, y, top / 2], size=[(x1 - x0) / 2, box_width / 2, top / 2],
            rgba=color, **common)

    def stairs(prefix, start, h, depth, count, descending=False):
        for i in range(count):
            level = (count - i - 1 if descending else i + 1) * h
            box(f"{prefix}_{i + 1}", start + i * depth, start + (i + 1) * depth, level)
        return start + count * depth

    def ramp(name, start, length, z0, z1, thickness):
        nonlocal highest
        angle = math.atan2(z1 - z0, length)
        highest = max(highest, z0, z1)
        terrain_spec.worldbody.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[start + length / 2 + math.sin(angle) * thickness / 2,
                 0, (z0 + z1) / 2 - math.cos(angle) * thickness / 2],
            size=[math.hypot(length, z1 - z0) / 2, width / 2, thickness / 2],
            quat=[math.cos(angle / 2), 0, -math.sin(angle / 2), 0],
            rgba=[0.46, 0.57, 0.43, 1], **common)

    def rough(prefix, start, length, cell, gap, hmin, hmax, seed):
        if gap < 0 or hmin < 0 or hmax < hmin:
            raise ValueError("Rough terrain requires gap>=0 and 0<=height_min<=height_max")
        rng = np.random.default_rng(seed)
        nx, ny = int(math.ceil(length / (cell + gap))), int(math.ceil(width / (cell + gap)))
        if nx * ny > 10000:
            raise ValueError("Rough terrain exceeds 10000 boxes; increase cell_size")
        for ix in range(nx):
            x0 = start + ix * (cell + gap)
            for iy in range(ny):
                y0 = -width / 2 + iy * (cell + gap)
                w = min(cell, width / 2 - y0)
                box(f"{prefix}_{ix}_{iy}", x0, min(x0 + cell, start + length),
                    rng.uniform(hmin, hmax), y=y0 + w / 2, box_width=w,
                    color=(0.53, 0.46, 0.36, 1))
        return start + length

    if kind == "custom" and "corridor_length" in p:
        length = scalar("corridor_length", positive=True)
        height = scalar("wall_height", positive=True)
        thickness = scalar("wall_thickness", positive=True)
        if terrain_spec.worldbody.geoms and any(
                g.type != mujoco.mjtGeom.mjGEOM_PLANE for g in terrain_spec.worldbody.geoms):
            raise ValueError("Parameterized corridor must not contain native obstacles/walls")
        for name, sign in (("wall_left", 1), ("wall_right", -1)):
            box(name, -length / 2, length / 2, height,
                y=sign * (width + thickness) / 2, box_width=thickness)
        # Default mat follows the actual clear width. Explicit XML/CLI sizes
        # retain their existing precedence and intersection validation.
        p.setdefault("spawn_mat_size", [1.4, min(1.1, width * 0.8), 0.02])
        goal = length / 2
    elif kind in {"flat", "custom"}:
        goal = scalar("goal_x")
    else:
        start = scalar("start_x")
        if kind in {"ascent", "descent"}:
            h, depth = scalar("step_height", positive=True), scalar("tread_depth", positive=True)
            count = int(scalar("step_count", positive=True, integer=True))
            if count > 10000:
                raise ValueError("Staircase exceeds 10000 steps")
            platform = scalar("platform_length", positive=True)
            if open_risers:
                approach = scalar("approach_length", positive=True)
                if not (start-approach <= spawn[0] < start and abs(spawn[1]) < width/2):
                    raise ValueError("Open-stair spawn_xy must lie on the approach surface")
                goal = start + count * depth
                spans = [(start-approach, start, 0.0)]
                spans += [(start+i*depth, start+(i+1)*depth, (i+1)*h) for i in range(count)]
                spans += [(goal, goal+platform, count*h)]
                vertices, elements = [], []
                visual_half_thickness = 1.e-6
                for surface_index, (x0, x1, z) in enumerate(spans):
                    i = len(vertices)//3
                    vertices += [x0, -width/2, z, x1, -width/2, z,
                                 x1, width/2, z, x0, width/2, z]
                    elements += [i, i+1, i+2, i, i+2, i+3]
                    # Radius-zero rigid flex is invisible in the default skin
                    # view. A site is display-only by construction. Do not use
                    # disabled plane geoms: MuJoCo 3.10's plane/flex path can
                    # still create contacts against those static planes.
                    terrain_spec.worldbody.add_site(
                        name=f"open_visual_{surface_index}", type=mujoco.mjtGeom.mjGEOM_BOX,
                        pos=[(x0+x1)/2, 0, z-visual_half_thickness],
                        size=[(x1-x0)/2, width/2, visual_half_thickness],
                        group=2, rgba=[.48, .52, .59, 1])
                pending_surface = dict(name="terrain_open_stair_surface", dim=2, radius=0.0,
                                       vertbody=["world"], vert=vertices, elem=elements,
                                       internal=0, selfcollide=0, rgba=[.48, .52, .59, 0], **common)
                highest = count*h
            elif kind == "descent":
                if not (start - platform <= spawn[0] < start and abs(spawn[1]) < width / 2):
                    raise ValueError("descent spawn_xy must lie on the elevated approach platform")
                box("approach", start - platform, start, count * h)
            if not open_risers:
                goal = stairs("stair", start, h, depth, count, kind == "descent")
                if kind == "ascent":
                    box("exit", goal, goal + platform, count * h)
            if walls_enabled:
                approach = scalar("approach_length", positive=True)
                wall_height = scalar("wall_height", positive=True)
                thickness = scalar("wall_thickness", positive=True)
                gap = scalar("wall_gap")
                if gap < 0:
                    raise ValueError("wall_gap must be nonnegative")
                # Continuous side walls from approach to the far end of the
                # landing. Their inner faces never intrude into the tread.
                # Wall tops track each adjacent tread; bottoms reach the floor,
                # also when open treads sit above a lower catch plane.
                wall_spans = [(start-approach, start, 0.0)]
                wall_spans += [(start+i*depth, start+(i+1)*depth, (i+1)*h)
                               for i in range(count)]
                wall_spans += [(goal, goal+platform, count*h)]
                for side, sign in (("left", 1), ("right", -1)):
                    for i, (x0, x1, tread_z) in enumerate(wall_spans):
                        top = tread_z + wall_height
                        highest = max(highest, top)
                        terrain_spec.worldbody.add_geom(
                            name=f"wall_{side}_{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
                            pos=[(x0+x1)/2, sign*(width/2+gap+thickness/2), (floor_z+top)/2],
                            size=[(x1-x0)/2, thickness/2, (top-floor_z)/2],
                            rgba=[.78, .76, .70, 1], **common)
        elif kind == "platform":
            length = scalar("platform_length", positive=True)
            height = scalar("platform_height", positive=True)
            goal = start + length
            box("platform", start, goal, height)
        elif kind in {"slope_up", "slope_down"}:
            length = scalar("ramp_length", positive=True)
            angle = scalar("slope_deg", positive=True)
            if angle >= 60:
                raise ValueError("slope_deg must be below 60 degrees")
            height = length * math.tan(math.radians(angle))
            platform = scalar("platform_length", positive=True)
            if kind == "slope_up":
                ramp("ramp", start, length, 0, height, scalar("ramp_thickness", positive=True))
                box("exit", start + length, start + length + platform, height)
            else:
                if not (start - platform <= spawn[0] < start and abs(spawn[1]) < width / 2):
                    raise ValueError("slope_down spawn_xy must lie on the elevated approach platform")
                box("approach", start - platform, start, height)
                ramp("ramp", start, length, height, 0, scalar("ramp_thickness", positive=True))
            goal = start + length
        elif kind == "rough":
            goal = rough("rough", start, scalar("length", positive=True), scalar("cell_size", positive=True),
                         scalar("gap"), scalar("height_min"), scalar("height_max"),
                         int(scalar("seed", integer=True)))
        elif kind == "obstacles":
            values = p.get("boxes", [])
            if not values or len(values) % 5:
                raise ValueError("boxes must contain rows of center_x center_y height length width")
            goal = start
            for i, (x, y, h, length, w) in enumerate(np.asarray(values).reshape(-1, 5)):
                if min(h, length, w) <= 0:
                    raise ValueError("Obstacle height/length/width must be positive")
                box(f"obstacle_{i}", x - length / 2, x + length / 2, h, y, w,
                    color=(0.62, 0.44, 0.34, 1))
                goal = max(goal, x + length / 2)
        elif kind == "hurdles":
            count = int(scalar("obstacle_count", positive=True, integer=True))
            spacing = scalar("spacing", positive=True)
            thickness = scalar("obstacle_depth", positive=True)
            h = scalar("height", positive=True)
            increment = scalar("height_increment")
            if h + min(0, (count - 1) * increment) <= 0 or thickness >= spacing:
                raise ValueError("Hurdles need positive heights and obstacle_depth < spacing")
            for i in range(count):
                box(f"hurdle_{i}", start + i * spacing, start + i * spacing + thickness,
                    h + i * increment, color=(0.62, 0.44, 0.34, 1))
            goal = start + (count - 1) * spacing + thickness
        elif kind == "mixed":
            h = scalar("step_height", positive=True)
            depth = scalar("tread_depth", positive=True)
            count = int(scalar("step_count", positive=True, integer=True))
            end = stairs("up", start, h, depth, count)
            plateau = scalar("plateau_length", positive=True)
            box("plateau", end, end + plateau, count * h)
            end = stairs("down", end + plateau, h, depth, count, True)
            gap = scalar("section_gap", positive=True)
            end = rough("rough", end + gap, scalar("rough_length", positive=True),
                scalar("cell_size", positive=True), scalar("gap"), scalar("height_min"),
                scalar("height_max"), int(scalar("seed", integer=True)))
            length = scalar("ramp_length", positive=True)
            angle = scalar("slope_deg", positive=True)
            if angle >= 60:
                raise ValueError("slope_deg must be below 60 degrees")
            height = length * math.tan(math.radians(angle))
            ramp("ramp_up", end + gap, length, 0, height, scalar("ramp_thickness", positive=True))
            end += gap + length
            plateau = scalar("ramp_plateau_length", positive=True)
            box("ramp_plateau", end, end + plateau, height)
            ramp("ramp_down", end + plateau, length, height, 0, scalar("ramp_thickness", positive=True))
            end += plateau + length + gap
            box("final_hurdle", end, end + scalar("obstacle_depth", positive=True),
                scalar("obstacle_height", positive=True))
            goal = end + scalar("obstacle_depth", positive=True)

    # Use world poses: movable-component geom positions are body-local.
    terrain_model = terrain_spec.compile()
    terrain_data = mujoco.MjData(terrain_model)
    mujoco.mj_forward(terrain_model, terrain_data)
    for i in range(terrain_model.ngeom):
        if terrain_model.geom_group[i] != 0:
            continue
        if terrain_model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE:
            highest = max(highest, float(terrain_data.geom_xpos[i, 2]))
            continue
        extent = float(np.linalg.norm(terrain_model.geom_size[i]))
        highest = max(highest, float(terrain_data.geom_xpos[i, 2]) + extent)
    ray_top = highest + 20.0
    if goal <= spawn[0]:
        raise ValueError("goal_x must be beyond spawn_x for the +X traversal summary")
    frame = robot_spec.worldbody.add_frame()
    robot_spec.attach(terrain_spec, prefix="terrain_", frame=frame)
    if pending_surface is not None:
        # MjSpec.attach copies the world tree, but can omit a flex attached
        # only to world. Add directly to the final model so collision survives.
        robot_spec.add_flex(**pending_surface)
    return Terrain(path, kind, p, tuple(spawn), yaw, goal, width, ray_top)


def select_terrain(args) -> Path:
    if getattr(args, "terrain_xml", None) is not None:
        if getattr(args, "terrain", None) is not None:
            raise ValueError("Choose either --terrain or --terrain-xml")
        return Path(args.terrain_xml).resolve()
    return TERRAIN_DIR / PRESETS[getattr(args, "terrain", None) or "ascent"]
