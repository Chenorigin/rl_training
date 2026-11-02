"""Width-only edits must move real walls and keep the default mat clear."""
from pathlib import Path
from types import SimpleNamespace
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy/deploy_mujoco"))
from terrain_loader import TERRAIN_DIR, attach_terrain
from spawn_mat import add_spawn_mat, validate_spawn_mat, MAT_NAME


def scene(tmp_path, width, **parameters):
    tree = ET.parse(TERRAIN_DIR / "straight_corridor.xml")
    tree.find("./custom/numeric[@name='width']").set("data", str(width))
    for key, value in parameters.items():
        node = tree.find(f"./custom/numeric[@name='{key}']")
        if node is None:
            node = ET.SubElement(tree.find("custom"), "numeric", name=key)
        node.set("data", value)
    path = tmp_path / "corridor.xml"
    tree.write(path)
    spec = mujoco.MjSpec.from_string('<mujoco/>')
    terrain = attach_terrain(spec, path)
    return spec, terrain


@pytest.mark.parametrize("width", [0.6, 1.0, 2.0, 3.0])
def test_width_moves_collision_walls_and_fits_mat(tmp_path, width):
    spec, terrain = scene(tmp_path, width)
    add_spawn_mat(spec, terrain, SimpleNamespace())
    model = spec.compile()
    validate_spawn_mat(model, terrain)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for name, sign in (("left", 1), ("right", -1)):
        wall = model.geom(f"terrain_wall_{name}")
        assert abs(wall.pos[1]) - wall.size[1] == pytest.approx(width / 2)
        np.testing.assert_allclose(wall.size * 2, [20, .2, 1.2])
        hit = np.array([-1], dtype=np.int32)
        distance = mujoco.mj_ray(model, data, np.array([0., 0., .5]),
                                np.array([0., float(sign), 0.]), None, True, -1, hit)
        assert distance == pytest.approx(width / 2)
    assert model.geom(MAT_NAME).size[1] * 2 < width
    assert terrain.spawn_xy == (0., 0.)
    assert terrain.goal_x == 10


def test_reproduce_old_oversized_mat_and_keep_explicit_override(tmp_path):
    spec, terrain = scene(tmp_path, 1., spawn_mat_size="1.4 1.1 0.02")
    with pytest.raises(ValueError, match="Spawn mat"):
        add_spawn_mat(spec, terrain, SimpleNamespace())


def test_length_moves_wall_ends_and_summary_goal(tmp_path):
    spec, terrain = scene(tmp_path, 1., corridor_length="6")
    model = spec.compile()
    assert model.geom("terrain_wall_left").size[0] == 3
    assert terrain.goal_x == 3


@pytest.mark.parametrize("width", [0, -1])
def test_invalid_width_rejected(tmp_path, width):
    with pytest.raises(ValueError, match="width"):
        scene(tmp_path, width)
