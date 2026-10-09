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
    "slope_up": "slope_up.xml", "slope_down": "slope_down.xml", "rough": "rough.xml",
    "obstacles": "obstacles.xml", "hurdles": "hurdles.xml", "mixed": "mixed.xml",
}
KINDS = set(PRESETS) | {"custom"}
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

    def summary(self) -> dict:
        return {"terrain": self.kind, "terrain_xml": str(self.path),
                "terrain_parameters": self.parameters, "goal_x": self.goal_x,
                "course_width": self.course_width}


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
            if kind not in {"ascent", "descent"}:
                raise ValueError("Stair CLI overrides require an ascent/descent terrain XML")
            if not math.isfinite(value):
                raise ValueError(f"Stair override {name} must be finite")
            parameters[STAIR_OVERRIDES[name]] = [float(value)]
    return kind, parameters


def attach_terrain(robot_spec: mujoco.MjSpec, path: Path, overrides: dict | None = None) -> Terrain:
    path = path.resolve()
    kind, p = read_config(path, overrides)

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
    terrain_spec = mujoco.MjSpec.from_file(str(path))
    if terrain_spec.worldbody.bodies:
        raise ValueError("Terrain XML must use worldbody geoms, without robot/dynamic bodies")
    if any(g.group != 0 or not (g.contype or g.conaffinity) for g in terrain_spec.worldbody.geoms):
        raise ValueError("Terrain geoms must be collision geoms in group 0 (height scan group)")
    # Replace the robot file's floor, keeping its bodies, assets and lighting.
    for geom in list(robot_spec.worldbody.geoms):
        if geom.group == 0 and (geom.contype or geom.conaffinity):
            robot_spec.delete(geom)
    common = dict(contype=1, conaffinity=1, condim=3, group=0, friction=friction)
    if not any(g.type == mujoco.mjtGeom.mjGEOM_PLANE for g in terrain_spec.worldbody.geoms):
        terrain_spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                                       size=[100, 100, 0.1], rgba=[0.32, 0.35, 0.38, 1], **common)
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

    if kind in {"flat", "custom"}:
        goal = scalar("goal_x")
    else:
        start = scalar("start_x")
        if kind in {"ascent", "descent"}:
            h, depth = scalar("step_height", positive=True), scalar("tread_depth", positive=True)
            count = int(scalar("step_count", positive=True, integer=True))
            platform = scalar("platform_length", positive=True)
            if kind == "descent":
                if not (start - platform <= spawn[0] < start and abs(spawn[1]) < width / 2):
                    raise ValueError("descent spawn_xy must lie on the elevated approach platform")
                box("approach", start - platform, start, count * h)
            goal = stairs("stair", start, h, depth, count, kind == "descent")
            if kind == "ascent":
                box("exit", goal, goal + platform, count * h)
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

    # Static native geometries can be taller than generated ones.
    terrain_model = terrain_spec.compile()
    for i in range(terrain_model.ngeom):
        if terrain_model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE:
            highest = max(highest, float(terrain_model.geom_pos[i, 2]))
            continue
        extent = float(np.linalg.norm(terrain_model.geom_size[i]))
        highest = max(highest, float(terrain_model.geom_pos[i, 2]) + extent)
    ray_top = highest + 20.0
    if goal <= spawn[0]:
        raise ValueError("goal_x must be beyond spawn_x for the +X traversal summary")
    frame = robot_spec.worldbody.add_frame()
    robot_spec.attach(terrain_spec, prefix="terrain_", frame=frame)
    return Terrain(path, kind, p, tuple(spawn), yaw, goal, width, ray_top)


def select_terrain(args) -> Path:
    if getattr(args, "terrain_xml", None) is not None:
        if getattr(args, "terrain", None) is not None:
            raise ValueError("Choose either --terrain or --terrain-xml")
        return Path(args.terrain_xml).resolve()
    return TERRAIN_DIR / PRESETS[getattr(args, "terrain", None) or "ascent"]
