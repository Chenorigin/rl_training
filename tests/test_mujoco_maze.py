"""The wall-sensitivity maze has a passable route and real collision walls."""

from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deploy/deploy_mujoco"))
from terrain_loader import attach_terrain, read_config

XML = ROOT / "deploy/deploy_mujoco/terrains/maze.xml"


def test_maze_is_standalone_static_and_loads_through_shared_terrain_path():
    standalone = mujoco.MjModel.from_xml_path(str(XML))
    assert standalone.nq == standalone.nv == standalone.nu == 0
    assert np.all(standalone.geom_bodyid == 0)
    assert np.all(standalone.geom_group == 0)
    assert np.all(standalone.geom_contype != 0) and np.all(standalone.geom_conaffinity != 0)
    spec = mujoco.MjSpec()
    terrain = attach_terrain(spec, XML)
    model = spec.compile()
    assert model.ngeom == standalone.ngeom
    assert np.count_nonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE) == 1
    assert terrain.spawn_xy == (-1., 0.) and terrain.spawn_yaw == 0.
    assert terrain.kind == "custom"
    assert model.geom("terrain_entry_left").id >= 0


def test_reference_route_has_robot_width_clearance_on_flat_ground():
    model = mujoco.MjModel.from_xml_path(str(XML))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    _, parameters = read_config(XML)
    route = np.asarray(parameters['maze_route_xy']).reshape(-1, 2)
    np.testing.assert_allclose(route[0], parameters['spawn_xy'])
    assert route[-1, 0] == parameters['goal_x'][0]
    walls = np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_BOX)
    for a, b in zip(route, route[1:]):
        for xy in np.linspace(a, b, int(np.ceil(np.linalg.norm(b-a)/.05))+1):
            outside = np.maximum(np.abs(xy-data.geom_xpos[walls, :2])-model.geom_size[walls, :2], 0.)
            # A robot-width disk can pass; yaw manoeuvres remain the user's control.
            assert np.linalg.norm(outside, axis=1).min() > .30
            geom = np.full(1, -1, dtype=np.int32)
            distance = mujoco.mj_ray(model, data, np.array([*xy, 2.]), np.array([0., 0., -1.]),
                                      None, True, -1, geom)
            assert abs(distance-2.) < 1.e-10


def test_walls_are_measured_and_collide_while_doorway_is_clear():
    spec = mujoco.MjSpec.from_file(str(XML))
    probe = spec.worldbody.add_body(name="probe", pos=[-1., 0., .5])
    probe.add_freejoint()
    probe.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[.08, 0., 0.], mass=1., group=1)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    groups = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
    for direction, expected in (([0., 1., 0.], "entry_left"), ([0., -1., 0.], "entry_right")):
        geom = np.full(1, -1, dtype=np.int32)
        distance = mujoco.mj_ray(model, data, np.array([-1., 0., .5]), np.asarray(direction),
                                  groups, True, -1, geom)
        assert abs(distance-.6) < 1.e-10
        assert model.geom(int(geom[0])).name == expected
    # Positive physical obstruction, then the real opening in the same wall.
    for position, blocked in (([.4, 0., .5], True), ([.4, 1.65, .5], False)):
        data.qpos[:3] = position
        mujoco.mj_forward(model, data)
        assert any(c.dist < 0. for c in data.contact) == blocked


def test_wall_junctions_overlap_without_unintended_slits():
    model = mujoco.MjModel.from_xml_path(str(XML))
    joins = [('entry_left', 'boundary_west_upper'),
             ('entry_right', 'boundary_west_lower'),
             ('boundary_west_upper', 'boundary_north'),
             ('boundary_west_lower', 'boundary_south'),
             ('boundary_east', 'boundary_north'),
             ('divider_1', 'boundary_south'), ('divider_2', 'boundary_north'),
             ('divider_3', 'boundary_south'), ('divider_4', 'boundary_north')]
    for first, second in joins:
        a, b = model.geom(first), model.geom(second)
        overlap = np.minimum(a.pos+a.size, b.pos+b.size)-np.maximum(a.pos-a.size, b.pos-b.size)
        # Positive overlap closes physical and visual seams, rather than merely
        # placing two independently rounded endpoints nominally edge-to-edge.
        assert np.all(overlap > .01), (first, second, overlap)


def test_compact_corridors_keep_wall_proximity_without_closing_passages():
    model = mujoco.MjModel.from_xml_path(str(XML))
    north, south = model.geom('boundary_north'), model.geom('boundary_south')
    interior_width = (north.pos[1]-north.size[1])-(south.pos[1]+south.size[1])
    assert interior_width <= 4.4+1.e-10
    walls = [model.geom(f'divider_{i}') for i in range(1, 5)]
    for a, b in zip(walls, walls[1:]):
        width = (b.pos[0]-b.size[0])-(a.pos[0]+a.size[0])
        assert .9 <= width <= 1.6
    for i, wall in enumerate(walls):
        width = ((north.pos[1]-north.size[1])-(wall.pos[1]+wall.size[1]) if i%2 == 0
                 else (wall.pos[1]-wall.size[1])-(south.pos[1]+south.size[1]))
        assert .89 <= width <= 1.11
