"""Parameter edits must change real MuJoCo collision, rays and support."""
from pathlib import Path
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'deploy/deploy_mujoco'))
from terrain_loader import TERRAIN_DIR, attach_terrain, select_terrain, terrain_ray
from test_mujoco_reset_placement import m20_path

GROUPS = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)


def edited(tmp_path, name, **parameters):
    tree = ET.parse(TERRAIN_DIR / name)
    for key, value in parameters.items():
        node = tree.find(f"./custom/numeric[@name='{key}']")
        if node is None:
            node = ET.SubElement(tree.find('./custom'), 'numeric', name=key)
        node.set('data', str(value))
    path = tmp_path / name
    tree.write(path)
    return path


def load(path):
    spec = mujoco.MjSpec.from_string(
        '<mujoco><option timestep=".002"/><worldbody>'
        '<body name="probe" pos="-1 0 .5"><freejoint/>'
        '<geom name="probe" type="sphere" size=".03" mass="1" group="1"/>'
        '</body></worldbody></mujoco>')
    terrain = attach_terrain(spec, path)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data, terrain


def ray(model, data, origin, direction):
    return terrain_ray(model, data, np.array(origin, dtype=float),
                       np.array(direction, dtype=float), GROUPS)[0]


@pytest.mark.parametrize('template', ['stairs_ascent.xml', 'stairs_ascent_open.xml',
                                      'stairs_ascent_walls.xml'])
@pytest.mark.parametrize('open_risers', [0, 1])
@pytest.mark.parametrize('walls_enabled', [0, 1])
def test_all_templates_switch_real_risers_and_side_walls_independently(
        tmp_path, template, open_risers, walls_enabled):
    path = edited(tmp_path, template, open_risers=open_risers, walls_enabled=walls_enabled,
                  step_height=.2, tread_depth=.4, step_count=3, width=1.6,
                  platform_length=1.3, wall_gap=.1, wall_thickness=.25, wall_height=1.7)
    model, data, terrain = load(path)
    assert model.nflex == open_risers
    assert terrain.summary()['walls_enabled'] == bool(walls_enabled)
    assert terrain.goal_x == pytest.approx(.25+3*.4)
    # Approach, every tread and landing have the intended actual support.
    for x, z in [(-1., 0.), (.45, .2), (.85, .4), (1.25, .6), (1.8, .6)]:
        assert terrain.ray_top-ray(model, data, [x, 0, terrain.ray_top], [0, 0, -1]) == pytest.approx(z)
        for sign in (-1, 1):
            distance = ray(model, data, [x, 0, z+.7], [0, sign, 0])
            assert distance == pytest.approx(.9) if walls_enabled else distance < 0
    # Riser test avoids horizontal treads and both side walls.
    data.qpos[:3] = [.25-.015, 0, .1]
    mujoco.mj_forward(model, data)
    assert (data.ncon > 0) == (not open_risers)
    # Side walls are real collision, not just decorative or ray-only geometry.
    data.qpos[:3] = [.45, .9-.015, .7]
    mujoco.mj_forward(model, data)
    assert (data.ncon > 0) == bool(walls_enabled)
    if walls_enabled:
        # Both entry and exit remain open; no end wall and no hidden roof.
        assert ray(model, data, [-1, 0, 1], [-1, 0, 0]) < 0
        assert ray(model, data, [terrain.goal_x+.7, 0, 1.3], [1, 0, 0]) < 0
        assert ray(model, data, [.45, 0, .7], [0, 0, 1]) < 0
        wall = model.geom('terrain_wall_left_1')
        assert wall.size[1]*2 == pytest.approx(.25)
        assert wall.pos[2]+wall.size[2] == pytest.approx(.2+1.7)
        assert wall.pos[2]-wall.size[2] == pytest.approx(-.5 if open_risers else 0.)


@pytest.mark.parametrize('height,length,width,start', [(.12, 1.5, .8, .2), (.7, 4.2, 2.6, 1.1)])
def test_platform_dimensions_rays_and_physical_support(tmp_path, height, length, width, start):
    path = edited(tmp_path, 'platform.xml', platform_height=height, platform_length=length,
                  width=width, start_x=start)
    model, data, terrain = load(path)
    geom = model.geom('terrain_platform')
    np.testing.assert_allclose(geom.size*2, [length, width, height])
    np.testing.assert_allclose(geom.pos, [start+length/2, 0, height/2])
    assert terrain.goal_x == pytest.approx(start+length)
    assert terrain.course_width == width
    for x, y, expected in [(start+length/2, 0, height), (start-.1, 0, 0),
                            (start+length+.1, 0, 0), (start+.1, width/2+.1, 0)]:
        assert 10-ray(model, data, [x, y, 10], [0, 0, -1]) == pytest.approx(expected)
    # Front vertical face exists; this is a platform, not a ramp or a thin top.
    data.qpos[:3] = [start-.015, 0, height/2]
    mujoco.mj_forward(model, data)
    assert data.ncon > 0
    data.qpos[:3] = [start+length/2, 0, height+.3]
    data.qvel[:] = 0
    for _ in range(700):
        mujoco.mj_step(model, data)
    assert data.qpos[2] == pytest.approx(height+.03, abs=.003)
    assert not np.any(data.warning.number)


@pytest.mark.parametrize('open_risers', [0, 1])
def test_stairwell_tread_bears_load_in_both_modes(tmp_path, open_risers):
    path = edited(tmp_path, 'stairs_ascent_walls.xml', open_risers=open_risers,
                  step_height=.2, tread_depth=.5, step_count=3)
    model, data, _ = load(path)
    data.qpos[:3] = [.5, 0, .5]
    for _ in range(700):
        mujoco.mj_step(model, data)
    assert data.qpos[2] == pytest.approx(.23, abs=.003)
    assert not np.any(data.warning.number)


@pytest.mark.parametrize('name,parameters', [
    ('stairs_ascent_walls.xml', dict(walls_enabled=2)),
    ('stairs_ascent_walls.xml', dict(walls_enabled='1 0')),
    ('stairs_ascent_walls.xml', dict(open_risers=-1)),
    ('stairs_ascent_walls.xml', dict(wall_height=0)),
    ('stairs_ascent_walls.xml', dict(wall_thickness=-.1)),
    ('stairs_ascent_walls.xml', dict(wall_gap=-.1)),
    ('stairs_ascent_walls.xml', dict(approach_length=0)),
    ('stairs_ascent_walls.xml', dict(step_count=2.5)),
    ('stairs_ascent_walls.xml', dict(step_count=10001)),
    ('stairs_ascent_walls.xml', dict(open_risers=1, spawn_xy='0.4 0')),
    ('platform.xml', dict(platform_height=0)),
    ('platform.xml', dict(platform_height='nan')),
    ('platform.xml', dict(platform_length=-1)),
    ('platform.xml', dict(width=0)),
    ('platform.xml', dict(walls_enabled=1)),
    ('platform.xml', dict(open_risers=1)),
])
def test_invalid_parameters_fail(tmp_path, name, parameters):
    with pytest.raises(ValueError):
        load(edited(tmp_path, name, **parameters))


@pytest.mark.parametrize('preset,filename', [('ascent_walls', 'stairs_ascent_walls.xml'),
                                           ('platform', 'platform.xml')])
def test_new_presets_select_loadable_templates(preset, filename):
    path = select_terrain(SimpleNamespace(terrain=preset, terrain_xml=None))
    assert path == TERRAIN_DIR/filename
    model, _, _ = load(path)
    assert model.ngeom > 1


@pytest.mark.parametrize('template,parameters,spawn_height', [
    ('stairs_ascent_walls.xml', {}, 0.),
    ('stairs_ascent_walls.xml', dict(open_risers=1), 0.),
    ('platform.xml', {}, 0.),
    ('platform.xml', dict(spawn_xy='1.5 0', platform_height=.3), .3),
])
@pytest.mark.parametrize('prone', [False, True])
@pytest.mark.parametrize('no_spawn_mat', [False, True])
def test_actual_m20_load_reset_and_scan(tmp_path, m20_path, template, parameters,
                                       spawn_height, prone, no_spawn_mat):
    import deploy_mujoco as runtime
    path = edited(tmp_path, template, **parameters)
    args = SimpleNamespace(terrain=None, terrain_xml=path, no_spawn_mat=no_spawn_mat)
    model = runtime.load_model(m20_path, args)
    contract = runtime.M20Contract(model, args.terrain_config)
    data = mujoco.MjData(model)
    contract.reset(data, prone=prone)
    expected = spawn_height if no_spawn_mat else contract.terrain.spawn_mat['top_z']
    assert contract.ground_height(data, *contract.terrain.spawn_xy) == pytest.approx(expected)
    if template == 'stairs_ascent_walls.xml':
        # The actual deployment scan ray must include the wall, not drop it.
        assert contract.ground_height(data, .4, .75) == pytest.approx(2.17)
    assert contract.height_scan(data).shape == (187,)
    assert np.isfinite(contract.height_scan(data)).all()
    assert all(contact.dist >= -1.e-7 for contact in data.contact
               if any(int(g) in contract.terrain_geom_ids for g in contact.geom)
               or any(int(f) in contract.terrain_flex_ids for f in contact.flex))
    first = data.qpos.copy()
    data.qpos[2] += 3
    data.qvel[:] = 2
    contract.reset(data, prone=prone)
    np.testing.assert_allclose(data.qpos, first, atol=1.e-12, rtol=0)
    assert np.all(data.qvel == 0)


@pytest.mark.parametrize('open_risers,walls_enabled', [(0, 0), (0, 1), (1, 0), (1, 1)])
@pytest.mark.parametrize('prone', [False, True])
def test_original_narrow_ascent_switches_with_default_mat(
        tmp_path, m20_path, open_risers, walls_enabled, prone):
    import deploy_mujoco as runtime
    from spawn_mat import MAT_NAME, DEFAULT_SIZE
    # Reproduce the user's exact operation: only change the two switches.
    # Do not widen the course or disable the default mat to make this pass.
    path = edited(tmp_path, 'stairs_ascent.xml', open_risers=open_risers,
                  walls_enabled=walls_enabled)
    args = SimpleNamespace(terrain=None, terrain_xml=path)
    model = runtime.load_model(m20_path, args)
    contract = runtime.M20Contract(model, args.terrain_config)
    data = mujoco.MjData(model)
    contract.reset(data, prone=prone)
    mat = model.geom(MAT_NAME)
    np.testing.assert_allclose(mat.size*2, [1.4, .9, .02])
    assert mat.pos[2]-mat.size[2] == pytest.approx(0.)
    assert contract.terrain.course_width == 1.
    np.testing.assert_allclose(data.qpos[:2], [-1, 0])
    assert np.isfinite(contract.height_scan(data)).all()
    for x in (-1.7, -.3):
        for y in (-.45, .45):
            assert contract.ground_height(data, x, y) == pytest.approx(.02)
    # Oversized explicit overrides remain rejected; no weakened geometry guard.
    if open_risers or walls_enabled:
        args.spawn_mat_size = DEFAULT_SIZE
        with pytest.raises(ValueError, match='Spawn mat crosses'):
            runtime.load_model(m20_path, args)
