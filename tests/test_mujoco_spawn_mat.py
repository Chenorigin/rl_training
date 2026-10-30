"""Starting mat geometry, reset support and all shared parser connections."""

from pathlib import Path
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deploy/deploy_mujoco"))
import deploy_mujoco as runtime
from spawn_mat import DEFAULT_SIZE, MAT_NAME, _check_coverage
from terrain_loader import TERRAIN_DIR, PRESETS
from test_quiet_mujoco_runtime import robot_fixture


def scene(tmp_path, terrain="flat", *, spread_wheels=False, **kwargs):
    robot = tmp_path / "robot.xml"
    robot_fixture(robot)
    if spread_wheels:
        tree = ET.parse(robot)
        for name, (x, y) in zip(runtime.JOINT_NAMES[12:], ((.3, .2), (.3, -.2), (-.3, .2), (-.3, -.2))):
            body = tree.find(f".//body[@name='{name.removesuffix('_joint')}']")
            body.set("pos", f"{x} {y} -.3")
        tree.write(robot)
    args = SimpleNamespace(terrain=terrain if isinstance(terrain, str) else None,
                           terrain_xml=terrain if isinstance(terrain, Path) else None, **kwargs)
    model = runtime.load_model(robot, args)
    contract = runtime.M20Contract(model, args.terrain_config)
    data = mujoco.MjData(model)
    contract.reset(data)
    return model, data, contract


@pytest.mark.parametrize("name", [*PRESETS, "sample_stairs.xml", "all_terrain.xml",
                                  "realistic_outdoor.xml", "real_world_final.xml", "custom.xml"])
def test_every_shipped_terrain_has_visible_physical_mat_at_reset(tmp_path, name):
    terrain = TERRAIN_DIR / name if name.endswith(".xml") else name
    model, data, contract = scene(tmp_path, terrain)
    mat = model.geom(MAT_NAME).id
    assert model.geom_bodyid[mat] == 0 and model.geom_group[mat] == 0
    assert model.geom_contype[mat] and model.geom_conaffinity[mat]
    assert model.geom_rgba[mat, 3] == 1.
    np.testing.assert_allclose(model.geom_size[mat]*2, DEFAULT_SIZE)
    np.testing.assert_allclose(data.qpos[:2], model.geom_pos[mat, :2])
    assert mat in contract.terrain_geom_ids
    assert contract.ground_height(data, *contract.terrain.spawn_xy) == pytest.approx(contract.terrain.spawn_mat["top_z"])
    contract.height_scan(data)
    # A central measured ray sees the actual top, not the old underlying ground.
    central = np.linalg.norm(contract.last_scan_hits[:, :2]-data.qpos[:2], axis=1).argmin()
    assert contract.last_scan_hits[central, 2] == pytest.approx(contract.terrain.spawn_mat["top_z"])


@pytest.mark.parametrize("prone", [False, True])
def test_relocation_and_repeated_reset_keep_close_support_without_penetration(tmp_path, prone):
    model, data, contract = scene(tmp_path, spawn_xy=[-2., .4])
    baseline, base_data, old = scene(tmp_path, spawn_xy=[-2., .4], no_spawn_mat=True)
    old.reset(base_data, prone=prone)
    for _ in range(2):
        data.qpos[:3] = [20., 20., 5.]
        data.qvel[:] = 10.
        contract.reset(data, prone=prone)
        np.testing.assert_allclose(data.qpos[:2], [-2., .4])
        assert data.qpos[2]-base_data.qpos[2] == pytest.approx(DEFAULT_SIZE[2], abs=2.e-7)
        assert not any(c.dist < -1.e-7 for c in data.contact
                       if any(g in contract.terrain_geom_ids for g in c.geom))
        assert np.all(data.qvel == 0.)
    # Release from reset: support must acquire a real contact within a bounded drop.
    mat = model.geom(MAT_NAME).id
    for _ in range(20):
        mujoco.mj_step(model, data)
    assert any(mat in c.geom and c.efc_address >= 0 for c in data.contact)
    assert np.isfinite(data.qpos).all()


def write_flat(tmp_path, *, extras="", spawn="-1 0", yaw="0", size=None):
    tree = ET.parse(TERRAIN_DIR / "flat.xml")
    tree.find("./custom/numeric[@name='spawn_xy']").set("data", spawn)
    tree.find("./custom/numeric[@name='spawn_yaw_deg']").set("data", yaw)
    if size is not None:
        ET.SubElement(tree.find("custom"), "numeric", name="spawn_mat_size", data=size)
    if extras:
        world = ET.SubElement(tree.getroot(), "worldbody")
        world.append(ET.fromstring(extras))
    path = tmp_path / "flat.xml"
    tree.write(path)
    return path


def test_xml_position_yaw_size_and_cli_precedence(tmp_path):
    path = write_flat(tmp_path, spawn="-2 .3", yaw="45", size="1.8 1.2 .04")
    tree = ET.parse(path)
    ET.SubElement(tree.find("custom"), "numeric", name="spawn_mat_xy", data="-2.1 .2")
    ET.SubElement(tree.find("custom"), "numeric", name="spawn_mat_yaw_deg", data="10")
    tree.write(path)
    model, data, contract = scene(tmp_path, path)
    np.testing.assert_allclose(model.geom(MAT_NAME).size, [.9, .6, .02])
    np.testing.assert_allclose(data.qpos[:2], [-2., .3])
    np.testing.assert_allclose(model.geom(MAT_NAME).pos[:2], [-2.1, .2])
    np.testing.assert_allclose(model.geom(MAT_NAME).quat, [np.cos(np.deg2rad(5)), 0, 0, np.sin(np.deg2rad(5))])
    assert not np.allclose(model.geom(MAT_NAME).quat, data.qpos[3:7])
    assert contract.ground_height(data, -2., .3) == pytest.approx(.04)
    original_pose = data.qpos.copy()
    model, data, contract = scene(tmp_path, path, spawn_mat_xy=[-3., -.2],
                                 spawn_mat_yaw_deg=80., spawn_mat_size=[1.2, 1., .08])
    np.testing.assert_allclose(model.geom(MAT_NAME).pos[:2], [-3., -.2])
    np.testing.assert_allclose(data.qpos[:2], original_pose[:2])
    np.testing.assert_allclose(data.qpos[3:], original_pose[3:])
    np.testing.assert_allclose(model.geom(MAT_NAME).size, [.6, .5, .04])
    assert contract.terrain.spawn_mat['yaw'] == pytest.approx(np.deg2rad(80))
    assert contract.terrain.spawn_xy == (-2., .3)
    assert contract.terrain.parameters['spawn_xy'] == [-2., .3]


@pytest.mark.parametrize("prone", [False, True])
def test_moving_only_mat_preserves_robot_xy_orientation_and_joints_on_reset(tmp_path, prone):
    path = write_flat(tmp_path, spawn="-1 0", yaw="20")
    for xy, yaw in (([-1., 0.], 0.), ([-1.2, -.2], 45.), ([2., 2.], -70.)):
        model, data, contract = scene(tmp_path, path, spawn_mat_xy=xy, spawn_mat_yaw_deg=yaw)
        for _ in range(2):
            data.qpos[:3] = [20., 20., 5.]
            data.qvel[:] = 10.
            contract.reset(data, prone=prone)
            np.testing.assert_allclose(data.qpos[:2], [-1., 0.])
            np.testing.assert_allclose(data.qpos[3:7], [np.cos(np.deg2rad(10)), 0, 0, np.sin(np.deg2rad(10))])
            np.testing.assert_allclose(data.qpos[contract.qpos_ids],
                                       runtime.PRONE_JOINT_POS if prone else runtime.DEFAULT_JOINT_POS)
            np.testing.assert_allclose(model.geom(MAT_NAME).pos[:2], xy)
            assert not any(c.dist < -1.e-7 for c in data.contact
                           if any(g in contract.terrain_geom_ids for g in c.geom))


def test_three_wheels_on_mat_one_off_without_moving_robot(tmp_path):
    thickness = .08
    model, data, contract = scene(tmp_path, spread_wheels=True,
        spawn_mat_xy=[-1.-np.sqrt(.125), -np.sqrt(.125)], spawn_mat_yaw_deg=45.,
        spawn_mat_size=[1.4, 1.1, thickness])
    centres = data.xpos[contract.wheel_body_ids]
    heights = np.array([contract.ground_height(data, *p[:2]) for p in centres])
    np.testing.assert_allclose(heights, [0., thickness, thickness, thickness])
    np.testing.assert_allclose(data.qpos[:2], [-1., 0.])
    np.testing.assert_allclose(data.qpos[3:7], [1., 0., 0., 0.])
    clearance = centres[:, 2]-.09-heights
    np.testing.assert_allclose(clearance, [thickness+.001, .001, .001, .001], atol=2.e-7)
    assert not any(c.dist < -1.e-7 for c in data.contact
                   if any(g in contract.terrain_geom_ids for g in c.geom))


@pytest.mark.parametrize("size", ["0 1 .02", "1 -1 .02", "1 1", "1 1 nan"])
def test_invalid_size_rejected(tmp_path, size):
    with pytest.raises(ValueError):
        scene(tmp_path, write_flat(tmp_path, size=size))


def test_mat_can_be_disabled_without_removing_spawn_override(tmp_path):
    model, data, contract = scene(tmp_path, no_spawn_mat=True, spawn_xy=[-2., .3])
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, MAT_NAME) == -1
    assert contract.terrain.spawn_mat is None
    np.testing.assert_allclose(data.qpos[:2], [-2., .3])
    assert contract.ground_height(data, -2., .3) == pytest.approx(0.)


def test_platform_edge_and_movable_support_rejected(tmp_path):
    with pytest.raises(ValueError, match="Spawn mat crosses"):
        scene(tmp_path, "ascent_open", spawn_mat_xy=[0., 0.])
    with pytest.raises(ValueError, match="fixed terrain"):
        scene(tmp_path, TERRAIN_DIR / "sample_stairs.xml", spawn_mat_xy=[1.03, 0.])


def test_narrow_obstacle_between_support_rays_is_rejected(tmp_path):
    path = write_flat(tmp_path, extras='<geom name="pin" type="box" pos="-1.06 .06 .01" size=".005 .005 .01" group="0"/>')
    with pytest.raises(ValueError, match="intersects terrain"):
        scene(tmp_path, path)


def test_narrow_deep_gap_between_support_rays_is_rejected(tmp_path):
    path = write_flat(tmp_path, spawn="0 0")
    tree = ET.parse(path)
    world = ET.SubElement(tree.getroot(), "worldbody")
    for text in ('<geom type="plane" pos="0 0 -1" size="10 10 .1"/>',
                 '<geom type="box" pos="-.475 0 -.1" size=".525 2 .1"/>',
                 '<geom type="box" pos=".56 0 -.1" size=".44 2 .1"/>'):
        world.append(ET.fromstring(text))
    tree.write(path)
    with pytest.raises(ValueError, match="continuous fixed"):
        scene(tmp_path, path)


def test_diagonal_support_seam_at_clipped_corners_is_not_a_gap():
    spec = mujoco.MjSpec()
    spec.add_flex(name="support", dim=2, radius=0., group=0,
        vertbody=["world"], vert=[-2., -2., 0., 2., -2., 0., 2., 2., 0., -2., 2., 0.],
        elem=[0, 1, 2, 0, 2, 3], contype=1, conaffinity=1)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    _check_coverage(model, data, (0,), np.zeros(2), np.eye(2), np.array([2., 2., .02]), 0.)


@pytest.mark.parametrize("module", ["deploy_mujoco", "deploy_quiet_landing", "deploy_student_v0", "deploy_student_v1"])
def test_each_independent_cli_accepts_shared_options(tmp_path, monkeypatch, module):
    import importlib
    parser = importlib.import_module(module).parse_args
    monkeypatch.setattr(sys, "argv", [module, "--model", str(tmp_path / "robot.xml"),
        "--checkpoint", str(tmp_path / "actor.pt"), "--terrain", "flat",
        "--spawn-xy", "-1.2", "0.3", "--spawn-mat-xy", "-1.4", "-.3",
        "--spawn-mat-yaw-deg", "45", "--spawn-mat-size", "1.4", "1.1", ".08", "--no-spawn-mat"])
    args = parser()
    assert args.spawn_xy == [-1.2, .3] and args.no_spawn_mat
    assert args.spawn_mat_xy == [-1.4, -.3] and args.spawn_mat_yaw_deg == 45.
    assert args.spawn_mat_size == [1.4, 1.1, .08]


@pytest.mark.parametrize("options", [dict(spawn_mat_xy=[float('nan'), 0.]),
    dict(spawn_mat_yaw_deg=float('inf')), dict(spawn_mat_size=[1., 1., -1.])])
def test_invalid_mat_options_rejected(tmp_path, options):
    with pytest.raises(ValueError):
        scene(tmp_path, **options)


@pytest.mark.parametrize("xy", [[float("nan"), 0.], [0., float("inf")]])
def test_invalid_spawn_override_rejected(tmp_path, xy):
    with pytest.raises(ValueError, match="finite"):
        scene(tmp_path, spawn_xy=xy)
