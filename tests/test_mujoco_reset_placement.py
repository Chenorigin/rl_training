"""Geometry-based reset regressions; CPU only, with strictly bounded physics."""

import os
from pathlib import Path
import sys

import mujoco
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deploy/deploy_mujoco"))
import reset_placement as placement


def robot_scene(terrain, folded=False, collision_mask="1"):
    wheel_z = "0" if folded else "-.45"
    return mujoco.MjModel.from_xml_string(f'''<mujoco>
      <option timestep=".002"/>
      <worldbody>{terrain}
        <body name="robot" pos="0 0 1">
          <freejoint name="root"/>
          <geom name="belly" type="box" size=".3 .15 .08" mass="10"
                contype="{collision_mask}" conaffinity="1"/>
          <geom name="visual" type="sphere" pos="0 0 -3" size="1"
                contype="0" conaffinity="0" mass="0"/>
          <body pos=".25 .15 {wheel_z}">
            <joint name="hinge" ref=".2"/>
            <geom name="wheel" type="cylinder" size=".06 .02"
                  quat=".707106781 .707106781 0 0" mass=".2"
                  contype="{collision_mask}" conaffinity="1"/>
          </body>
        </body>
      </worldbody>
    </mujoco>''')


def place(model, data, **kwargs):
    return placement.place_on_terrain(model, data,
        root_body_id=model.body("robot").id,
        root_qpos_adr=int(model.joint("root").qposadr[0]),
        terrain_geom_ids=[g for g in range(model.ngeom) if model.geom_bodyid[g] == 0],
        **kwargs)


def support_plane_clearance(model, data, robot_geoms, terrain_geom):
    """Independent analytic oracle for these fixtures' large, planar supports.

    MuJoCo 3.10's mj_geomDistance can return zero for separated box pairs when
    queried with a large distmax. These exact primitive support functions do
    not rely on that API, contacts, or the placement routine's AABB bounds.
    """
    normal = data.geom_xmat[terrain_geom].reshape(3, 3)[:, 2]
    point = data.geom_xpos[terrain_geom].copy()
    if model.geom_type[terrain_geom] == mujoco.mjtGeom.mjGEOM_BOX:
        point += normal * model.geom_size[terrain_geom, 2]
    distances = []
    for geom in robot_geoms:
        rotation = data.geom_xmat[geom].reshape(3, 3)
        size = model.geom_size[geom]
        if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_BOX:
            radius = np.abs(normal @ rotation) @ size
        elif model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_CYLINDER:
            cosine = float(normal @ rotation[:, 2])
            radius = size[0] * np.sqrt(max(0., 1. - cosine**2)) + size[1] * abs(cosine)
        else:
            raise AssertionError("Oracle fixture must use boxes or cylinders")
        distances.append(float(normal @ (data.geom_xpos[geom] - point) - radius))
    return min(distances)


@pytest.mark.parametrize("terrain", [
    '<geom name="floor" type="plane" size="0 0 .1"/>',
    '<geom name="platform" type="box" pos="0 0 .3" size="2 2 .3"/>',
    '<geom name="slope" type="box" pos="0 0 .3" size="2 2 .1" euler="0 15 0"/>',
    '<geom name="slope" type="plane" pos="0 0 .2" size="0 0 .1" euler="0 15 0"/>',
])
@pytest.mark.parametrize("folded", [False, True])
def test_reset_has_close_support_without_penetration_and_preserves_pose(terrain, folded):
    model = robot_scene(terrain, folded)
    data = mujoco.MjData(model)
    data.qpos[:2] = [.1, -.1]
    data.qpos[3:7] = [np.cos(.3), 0., 0., np.sin(.3)]
    before, qpos0 = data.qpos.copy(), model.qpos0.copy()
    z = place(model, data)
    actual = support_plane_clearance(model, data,
                                    [model.geom("belly").id, model.geom("wheel").id], 0)
    assert 0 < actual <= placement.RESET_CLEARANCE + 1e-7
    assert z == data.qpos[2]
    np.testing.assert_array_equal(data.qpos[np.arange(model.nq) != 2], before[np.arange(model.nq) != 2])
    np.testing.assert_array_equal(model.qpos0, qpos0)
    assert data.time == 0
    np.testing.assert_array_equal(data.qvel, 0)


def test_analytic_oracle_rejects_airborne_and_penetrating_poses():
    model = robot_scene('<geom type="box" pos="0 0 .3" size="2 2 .3"/>')
    data = mujoco.MjData(model)
    geoms = [model.geom("belly").id, model.geom("wheel").id]
    for z, expected in [(1.21, .1), (1.01, -.1)]:
        data.qpos[2] = z
        mujoco.mj_fwdPosition(model, data)
        assert support_plane_clearance(model, data, geoms, 0) == pytest.approx(expected)


def test_prone_uses_belly_and_ignores_visual_only_geometry():
    model = robot_scene('<geom type="plane" size="0 0 .1"/>', folded=True)
    data = mujoco.MjData(model)
    assert place(model, data) == pytest.approx(.08 + placement.RESET_CLEARANCE, abs=1e-7)


def test_accepts_collision_affinity_without_contype():
    model = robot_scene('<geom type="plane" size="0 0 .1"/>', collision_mask="0")
    data = mujoco.MjData(model)
    assert place(model, data) == pytest.approx(.51 + placement.RESET_CLEARANCE, abs=1e-7)


def test_raised_support_under_foot_overrides_ground_under_root():
    model = robot_scene('''<geom type="plane" size="0 0 .1"/>
      <geom type="box" pos=".25 .15 .2" size=".08 .08 .2"/>''')
    data = mujoco.MjData(model)
    # The root is over the low floor; the wheel is over the raised block.
    assert place(model, data) == pytest.approx(.4 + .51 + placement.RESET_CLEARANCE, abs=1e-7)


def test_missing_terrain_fails_without_moving_robot():
    model = robot_scene('<geom type="box" pos="10 0 0" size="1 1 1"/>')
    data = mujoco.MjData(model)
    before = data.qpos.copy()
    with pytest.raises(RuntimeError, match="No collision terrain beneath"):
        place(model, data)
    np.testing.assert_array_equal(data.qpos, before)


def test_failed_collision_masks_are_bounded_and_restore_pose(monkeypatch):
    model = robot_scene('<geom type="plane" size="0 0 .1" contype="2" conaffinity="2"/>')
    data = mujoco.MjData(model)
    before = data.qpos.copy()
    monkeypatch.setattr(placement, "MAX_SEARCH_STEPS", 4)
    with pytest.raises(RuntimeError, match="bounded placement search"):
        place(model, data)
    np.testing.assert_array_equal(data.qpos, before)


def test_prone_does_not_undergo_the_old_free_fall():
    model = robot_scene('<geom type="plane" size="0 0 .1"/>', folded=True)
    good, old = mujoco.MjData(model), mujoco.MjData(model)
    place(model, good)
    old.qpos[2] = .2
    mujoco.mj_forward(model, old)
    good_z, old_z = good.qpos[2], old.qpos[2]
    # Exactly 40 physics steps per case, no policy, optimizer or graphics.
    for _ in range(40):
        mujoco.mj_step(model, good)
        mujoco.mj_step(model, old)
    assert good_z - good.qpos[2] < .005
    assert old_z - old.qpos[2] > .02
    assert np.isfinite(good.qpos).all()


@pytest.fixture(scope="module")
def m20_path():
    # The model documented by docs/m20_student_v0_mujoco.md. Allow other
    # machines to select their actual deployment model explicitly.
    documented = (Path.home() / "桌面/m20_1/M20_perception_Lidar_rl_2/"
                  "M20_perception_Lidar_rl/deep_robotics_model/M20/mjcf/M20.xml")
    path = Path(os.environ.get("M20_MODEL_PATH", str(documented)))
    if not path.is_file():
        pytest.skip("Actual M20 deployment model is unavailable; set M20_MODEL_PATH")
    return path


@pytest.mark.parametrize("terrain_name", ["flat", "ascent", "descent", "slope_up", "slope_down", "ascent_open"])
@pytest.mark.parametrize("prone", [False, True])
def test_actual_m20_contract_reset(m20_path, terrain_name, prone):
    from types import SimpleNamespace
    import deploy_mujoco as runtime

    args = SimpleNamespace(terrain=terrain_name, terrain_xml=None)
    model = runtime.load_model(m20_path, args)
    contract = runtime.M20Contract(model, args.terrain_config)
    data = mujoco.MjData(model)
    original_qpos0 = model.qpos0.copy()
    contract.reset(data, prone=prone)
    first = data.qpos.copy()
    robot_geoms = [g for g in range(model.ngeom)
                   if model.geom_bodyid[g] != 0
                   and g not in contract.terrain_geom_ids
                   and (model.geom_contype[g] or model.geom_conaffinity[g])]
    def terrain_contact_distances():
        return [float(contact.dist) for contact in data.contact
                if any(int(contact.geom[side]) in robot_geoms and
                       (int(contact.geom[1-side]) in contract.terrain_geom_ids or
                        int(contact.flex[1-side]) in contract.terrain_flex_ids)
                       for side in (0, 1))]

    assert all(distance >= 0 for distance in terrain_contact_distances())
    # Moving down by twice the clearance must encounter actual support,
    # including zero-radius flex surfaces (mj_geomDistance excludes flexes).
    data.qpos[2] -= 2 * placement.RESET_CLEARANCE
    mujoco.mj_fwdPosition(model, data)
    assert any(distance < 0 for distance in terrain_contact_distances())
    data.qpos[:] = first
    mujoco.mj_forward(model, data)
    np.testing.assert_array_equal(model.qpos0, original_qpos0)
    # Repeat after corrupting a running state to test the actual reset caller.
    data.qpos[2] += 2
    data.qvel[:] = 3
    data.ctrl[:] = 4
    data.time = 10
    contract.reset(data, prone=prone)
    np.testing.assert_allclose(data.qpos, first, atol=1e-12, rtol=0)
    np.testing.assert_array_equal(data.qvel, 0)
    np.testing.assert_array_equal(data.ctrl, 0)
    assert data.time == 0
