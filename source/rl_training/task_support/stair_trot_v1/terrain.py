"""Straight ascent terrain with matching solid and genuinely open risers.

The open variant consists only of upward horizontal triangles. It contains no
vertical collision faces, including at the start/end platforms and safety floor.
This is an idealised zero-thickness tread, not a model of a manufactured stair:
triangle-edge contacts and PhysX contact offsets still need physical validation.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import trimesh

from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from isaaclab.utils import configclass


@dataclass(frozen=True)
class StairDimensions:
    """One staircase's geometry, in the generator's uncentred local frame."""

    step_height: float
    step_depth: float
    start_x: float
    end_x: float
    top_z: float


def stair_dimensions(difficulty: float, cfg: AscentStairsTerrainCfg) -> StairDimensions:
    """Validate the complete parameter range, then sample one regular staircase.

    Height follows curriculum difficulty. Tread depth is sampled independently
    using NumPy's global RNG, following IsaacLab's built-in terrain convention.
    A fixed depth range supports explicit short/medium/long terrain columns.
    The whole range must fit, so a bad configuration cannot fail only on some
    random seeds or silently lose its final landing.
    """
    if not math.isfinite(difficulty) or not 0.0 <= difficulty <= 1.0:
        raise ValueError("Stair difficulty must be finite and in [0, 1].")
    if len(cfg.size) != 2 or any(not math.isfinite(v) or v <= 0.0 for v in cfg.size):
        raise ValueError("Stair size must contain two finite positive lengths.")
    for name in ("step_height_range", "step_depth_range"):
        values = getattr(cfg, name)
        if (
            len(values) != 2
            or any(not math.isfinite(v) or v <= 0.0 for v in values)
            or values[0] > values[1]
        ):
            raise ValueError(f"{name} must be an ordered pair of positive finite values.")
    if isinstance(cfg.num_steps, bool) or not isinstance(cfg.num_steps, int) or cfg.num_steps < 1:
        raise ValueError("num_steps must be a positive integer.")
    for name in ("start_platform_length", "end_platform_length"):
        value = getattr(cfg, name)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive.")
    if not math.isfinite(cfg.safety_floor_depth) or cfg.safety_floor_depth <= 0.0:
        raise ValueError("safety_floor_depth must be finite and positive.")
    if not isinstance(cfg.riser, bool):
        raise ValueError("riser must be True (solid) or False (open).")
    required = (
        cfg.start_platform_length + cfg.num_steps * cfg.step_depth_range[1] + cfg.end_platform_length
    )
    if required > cfg.size[0] + 1e-9:
        raise ValueError(
            f"Staircase needs at least {required:g} m in x including both landings; "
            f"size[0]={cfg.size[0]:g} m."
        )
    step_height = cfg.step_height_range[0] + difficulty * (
        cfg.step_height_range[1] - cfg.step_height_range[0]
    )
    step_depth = float(np.random.uniform(*cfg.step_depth_range))
    return StairDimensions(
        step_height=step_height,
        step_depth=step_depth,
        start_x=cfg.start_platform_length,
        end_x=cfg.start_platform_length + cfg.num_steps * step_depth,
        top_z=cfg.num_steps * step_height,
    )


def _horizontal_patch(x0: float, x1: float, width: float, z: float) -> trimesh.Trimesh:
    """Two upward triangles without side/bottom faces or duplicate back faces."""
    return trimesh.Trimesh(
        vertices=np.array(((x0, 0.0, z), (x1, 0.0, z), (x1, width, z), (x0, width, z))),
        faces=np.array(((0, 1, 2), (0, 2, 3))),
        process=False,
    )


def ascent_stairs_terrain(
    difficulty: float, cfg: AscentStairsTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """IsaacLab ``SubTerrainBaseCfg.function``: rise along +X from a low landing.

    Bounds are ``[0, size_x] x [0, size_y]``. The origin is on the low landing,
    not the centre/top of the staircase. TerrainGenerator shifts both mesh and
    origin; callers must not centre either here a second time.

    All steps span the full tile width, so there is no flat lateral bypass
    within a stair tile. The final landing begins after ``num_steps`` treads;
    it shares the last tread height and is at least ``end_platform_length``
    long even at the longest configured depth.

    For open stairs a separate lower horizontal floor catches a failed ascent.
    Reaching that floor must count as failure in the task; it is not a tread.
    """
    shape = stair_dimensions(difficulty, cfg)
    size_x, size_y = cfg.size
    floor_z = -cfg.safety_floor_depth
    spans = [(0.0, shape.start_x, 0.0)]
    spans.extend(
        (
            shape.start_x + index * shape.step_depth,
            shape.start_x + (index + 1) * shape.step_depth,
            (index + 1) * shape.step_height,
        )
        for index in range(cfg.num_steps)
    )
    spans.append((shape.end_x, size_x, shape.top_z))
    if cfg.riser:
        meshes = []
        for x0, x1, top_z in spans:
            mesh = trimesh.creation.box(extents=(x1 - x0, size_y, top_z - floor_z))
            mesh.apply_translation(((x0 + x1) / 2.0, size_y / 2.0, (top_z + floor_z) / 2.0))
            meshes.append(mesh)
    else:
        meshes = [_horizontal_patch(x0, x1, size_y, z) for x0, x1, z in spans]
        meshes.append(_horizontal_patch(0.0, size_x, size_y, floor_z))
    origin = np.array((shape.start_x / 2.0, size_y / 2.0, 0.0), dtype=np.float64)
    return meshes, origin


@configclass
class AscentStairsTerrainCfg(SubTerrainBaseCfg):
    """Parameters shared by the solid/open matched staircase variants.

    These ranges are training coverage, not a claim that the robot can already
    traverse them. All lengths are metres. ``size`` is overridden by IsaacLab's
    TerrainGenerator and must still accommodate the longest staircase.
    """

    function = ascent_stairs_terrain
    size: tuple[float, float] = (8.0, 8.0)
    step_height_range: tuple[float, float] = (0.10, 0.25)
    step_depth_range: tuple[float, float] = (0.25, 0.50)
    num_steps: int = 6
    start_platform_length: float = 2.0
    end_platform_length: float = 2.0
    safety_floor_depth: float = 0.50
    riser: bool = True
