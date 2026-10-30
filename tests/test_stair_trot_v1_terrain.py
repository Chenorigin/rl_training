"""CPU collision-mesh checks; no Isaac application, simulation or training."""
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

import numpy as np
import trimesh


ROOT = Path(__file__).resolve().parents[1]


def _load_geometry():
    # Only the Isaac configuration containers are stubbed. Every tested vertex,
    # face, triangle normal and origin comes from the real production generator.
    base_module = ModuleType("isaaclab.terrains.sub_terrain_cfg")
    utils_module = ModuleType("isaaclab.utils")

    @dataclass
    class SubTerrainBaseCfg:
        proportion: float = 1.0
        size: tuple = (10.0, 10.0)

    base_module.SubTerrainBaseCfg = SubTerrainBaseCfg
    utils_module.configclass = dataclass
    spec = importlib.util.spec_from_file_location(
        "_stair_trot_terrain_geometry_test",
        ROOT / "source/rl_training/task_support/stair_trot_v1/terrain.py",
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {
        spec.name: module,
        "isaaclab.terrains.sub_terrain_cfg": base_module,
        "isaaclab.utils": utils_module,
    }):
        spec.loader.exec_module(module)
    return module


terrain = _load_geometry()


def upward_triangles(meshes):
    mesh = trimesh.util.concatenate(meshes)
    return mesh.triangles[mesh.face_normals[:, 2] > 0.999999]


def support_height(meshes, x, y):
    """Independent vertical-ray/triangle intersection without an R-tree."""
    triangles = upward_triangles(meshes)
    point = np.array((x, y))
    matches = []
    for triangle in triangles:
        xy = triangle[:, :2]
        matrix = np.column_stack((xy[1] - xy[0], xy[2] - xy[0]))
        u, v = np.linalg.solve(matrix, point - xy[0])
        if u >= -1e-9 and v >= -1e-9 and u + v <= 1.0 + 1e-9:
            matches.append(triangle[0, 2])
    return max(matches, default=np.nan)


class AscentStairGeometryTests(unittest.TestCase):
    def setUp(self):
        self.rng_state = np.random.get_state()

    def tearDown(self):
        np.random.set_state(self.rng_state)

    def cfg(self, **kwargs):
        return terrain.AscentStairsTerrainCfg(**kwargs)

    def test_open_collision_contains_no_vertical_faces_including_landings(self):
        cfg = self.cfg(riser=False)
        meshes, origin = terrain.ascent_stairs_terrain(1.0, cfg)
        mesh = trimesh.util.concatenate(meshes)
        np.testing.assert_allclose(mesh.face_normals, np.tile((0, 0, 1), (len(mesh.faces), 1)))
        self.assertTrue((mesh.area_faces > 0).all())
        self.assertEqual(len(mesh.faces), 2 * (cfg.num_steps + 3))
        self.assertFalse(mesh.is_watertight)
        self.assertAlmostEqual(mesh.bounds[0, 2], -cfg.safety_floor_depth)
        np.testing.assert_allclose(origin, (cfg.start_platform_length / 2, cfg.size[1] / 2, 0))

    def test_detector_rejects_solid_stairs_as_no_riser(self):
        meshes, _ = terrain.ascent_stairs_terrain(0.5, self.cfg(riser=True))
        mesh = trimesh.util.concatenate(meshes)
        self.assertTrue((np.abs(mesh.face_normals[:, 2]) < 0.01).any())
        self.assertTrue(all(part.is_watertight for part in meshes))

    def test_solid_and_open_have_identical_upward_walking_surfaces(self):
        for depth in (0.25, 0.35, 0.50):
            cfg = self.cfg(step_depth_range=(depth, depth))
            solid, solid_origin = terrain.ascent_stairs_terrain(0.4, cfg)
            cfg.riser = False
            opened, open_origin = terrain.ascent_stairs_terrain(0.4, cfg)
            np.testing.assert_array_equal(solid_origin, open_origin)
            for x in np.linspace(0.1, cfg.size[0] - 0.1, 41):
                for y in (0.05, cfg.size[1] / 2, cfg.size[1] - 0.05):
                    self.assertAlmostEqual(support_height(solid, x, y), support_height(opened, x, y))

    def test_low_spawn_ascent_and_full_exit_landing(self):
        cfg = self.cfg(riser=False, step_depth_range=(0.50, 0.50))
        meshes, origin = terrain.ascent_stairs_terrain(1.0, cfg)
        for dx in (-0.6, 0.6):
            for dy in (-0.35, 0.35):
                self.assertEqual(support_height(meshes, origin[0] + dx, origin[1] + dy), 0.0)
        for index in range(cfg.num_steps):
            x = cfg.start_platform_length + (index + 0.5) * cfg.step_depth_range[0]
            self.assertAlmostEqual(
                support_height(meshes, x, origin[1]), (index + 1) * cfg.step_height_range[1]
            )
        end_x = cfg.start_platform_length + cfg.num_steps * cfg.step_depth_range[1]
        self.assertGreaterEqual(cfg.size[0] - end_x, cfg.end_platform_length)
        for x in (end_x + 0.1, cfg.size[0] - 0.1):
            self.assertAlmostEqual(support_height(meshes, x, origin[1]), cfg.num_steps * cfg.step_height_range[1])

    def test_generator_recentering_preserves_spawn_to_step_relation(self):
        cfg = self.cfg(riser=False, step_depth_range=(0.35, 0.35))
        meshes, origin = terrain.ascent_stairs_terrain(0.5, cfg)
        mesh = trimesh.util.concatenate(meshes)
        offset = np.array((-cfg.size[0] / 2, -cfg.size[1] / 2, 0))
        mesh.apply_translation(offset)
        centred_origin = origin + offset
        self.assertEqual(support_height([mesh], *centred_origin[:2]), 0.0)
        self.assertLess(centred_origin[0], 0)
        np.testing.assert_allclose(mesh.bounds[:, :2], (-np.array(cfg.size) / 2, np.array(cfg.size) / 2))
        self.assertAlmostEqual(
            support_height([mesh], cfg.start_platform_length + 0.1 + offset[0], 0),
            (cfg.step_height_range[0] + cfg.step_height_range[1]) / 2,
        )

    def test_height_curriculum_and_depth_randomization_are_independent(self):
        cfg = self.cfg()
        depths = []
        for seed in range(30):
            np.random.seed(seed)
            easy = terrain.stair_dimensions(0.0, cfg)
            np.random.seed(seed)
            hard = terrain.stair_dimensions(1.0, cfg)
            self.assertEqual(easy.step_depth, hard.step_depth)
            self.assertEqual(easy.step_height, cfg.step_height_range[0])
            self.assertEqual(hard.step_height, cfg.step_height_range[1])
            self.assertGreaterEqual(easy.step_depth, cfg.step_depth_range[0])
            self.assertLessEqual(easy.step_depth, cfg.step_depth_range[1])
            depths.append(easy.step_depth)
        self.assertGreater(np.ptp(depths), 0.15)

    def test_fixed_short_medium_long_depths_and_both_riser_types(self):
        for depth in (0.25, 0.35, 0.50):
            for riser in (True, False):
                for difficulty in (0, 0.5, 1):
                    cfg = self.cfg(riser=riser, step_depth_range=(depth, depth))
                    meshes, _ = terrain.ascent_stairs_terrain(difficulty, cfg)
                    z = support_height(meshes, cfg.start_platform_length + depth / 2, cfg.size[1] / 2)
                    self.assertAlmostEqual(z, terrain.stair_dimensions(difficulty, cfg).step_height)

    def test_invalid_configuration_fails_before_generation(self):
        for kwargs in (
            {"size": (5.0, 8.0)}, {"size": (8.0, 0.0)}, {"num_steps": 0}, {"num_steps": 1.5},
            {"step_depth_range": (0.5, 0.25)}, {"step_height_range": (0.0, 0.2)},
            {"start_platform_length": -1}, {"end_platform_length": 0},
            {"safety_floor_depth": 0}, {"safety_floor_depth": float("nan")}, {"riser": "open"},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                terrain.ascent_stairs_terrain(0.5, self.cfg(**kwargs))
        for difficulty in (-0.1, 1.1, float("nan")):
            with self.subTest(difficulty=difficulty), self.assertRaises(ValueError):
                terrain.ascent_stairs_terrain(difficulty, self.cfg())


if __name__ == "__main__":
    unittest.main()
