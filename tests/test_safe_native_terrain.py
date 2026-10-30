"""Native Safe geometry regressions; bounded CPU physics, no optimization."""
from pathlib import Path
import sys

import mujoco
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deploy/deploy_mujoco"))
sys.path.insert(0, str(ROOT / "source/rl_training"))
from safe_native_geometry import NativeTerrainGeometry
from task_support.stair_quiet_v1_safe.settings import SafeSettings
from task_support.stair_quiet_v1_safe.observations import safety_features
from test_safe_mujoco_runtime import saved_safe, scene
import deploy_quiet_landing as runtime

XML = ROOT / "deploy/deploy_mujoco/terrains/sample_stairs.xml"


@pytest.fixture
def native():
    model = mujoco.MjModel.from_xml_path(str(XML))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data, NativeTerrainGeometry(model, range(model.ngeom)), SafeSettings()


def sample(position, velocity=(0., 0., 0.), omega=(0., 0., 0.), force=(0., 0., 100.)):
    return {k: torch.as_tensor(np.asarray(v), dtype=torch.float32).unsqueeze(0) for k, v in dict(
        position=[position]*4, velocity=[velocity]*4, omega=[omega]*4,
        axis=[[0., 1., 0.]]*4, forces=[force]*4, root=[1.03, 0., .8],
        root_velocity=[0., 0., 0.], yaw=0., width=1.).items()}


def feet(native, observation):
    _, data, geometry, settings = native
    result = geometry.features(data, observation, settings)
    assert result.shape == (1, 48) and torch.isfinite(result).all()
    return result[0, :44].reshape(4, 11).numpy()


def translate(model, data, body, delta):
    adr = int(model.joint(body+"_free").qposadr[0])
    data.qpos[adr:adr+3] += delta
    mujoco.mj_forward(model, data)


def test_white_dark_gap_floor_and_vertical_force_semantics(native):
    s = native[3]
    white = feet(native, sample([1.03, 0., .54]))
    assert white[0, 0] == pytest.approx((.275-s.contact_patch_radius-s.front_tread_margin)/.1)
    assert white[0, 1] == pytest.approx((.45-s.wheel_radius-s.wheel_half_length-s.front_lateral_margin)/.1)
    assert np.all(white[:, 6] == 1)
    dark = feet(native, sample([1.03, -.705, .49]))
    assert dark[0, 1] == pytest.approx((.2-s.wheel_radius-s.wheel_half_length-s.front_lateral_margin)/.1, abs=1.e-5)
    gap = feet(native, sample([1.03, -.4775, .54]))
    assert np.all(gap[:, 2] < 0), 'Gap must not become one white/dark bounding rectangle'
    # Riser-only force and mismatched wheel-bottom height never mean support.
    assert not feet(native, sample([1.03, -.4775, .54], force=(200., 0., 20.)))[:, 6].any()
    assert not feet(native, sample([1.03, 0., .69]))[:, 6].any()
    floor = feet(native, sample([-1., 0., .09]))
    np.testing.assert_array_equal(floor[:, :3], 10.)


def test_high_tread_lip_can_support_wheel_whose_centre_is_behind_edge(native):
    model, data, geometry, s = native
    observation = sample([-.02, .25, .24])
    faces, _ = geometry._faces(data)
    face = geometry._select(faces, np.array([-.02, .25, .24]), np.array([0., 1., 0.]), s)
    assert model.geom(face.geom).name == 'stairs_25_left_tread_1'
    f = feet(native, observation)
    assert np.all(f[:, 0] < 0) and np.all(f[:, 6] == 1)


def test_touching_small_stairs_merge_but_sliding_opens_real_seam(native):
    model, data, _, _ = native
    observation = sample([.375, -.02, .39])
    original = feet(native, observation)
    assert np.all(original[:, 1:3] > 0)
    joint = model.joint('stairs_25_right_free')
    data.qvel[int(joint.dofadr[0])+1] = -.3
    mujoco.mj_forward(model, data)
    predicted = feet(native, observation)
    assert np.all(predicted[:, 4] < 0), 'Moving seam must open in constant-velocity prediction'
    data.qvel[:] = 0
    translate(model, data, 'stairs_25_right', [0., -.04, 0.])
    moved = feet(native, observation)
    # In the gap, the nearest tread has an unsafe inner edge and a safe far edge.
    assert np.all(moved[:, 1:3].min(axis=1) < 0)


def test_platform_translation_yaw_and_tilt_use_current_oriented_faces(native):
    model, data, geometry, s = native
    original = feet(native, sample([1.03, 0., .54]))
    adr = int(model.joint('white_platform_free').qposadr[0])
    data.qpos[adr+3:adr+7] = [np.sqrt(.5), 0., 0., np.sqrt(.5)]
    mujoco.mj_forward(model, data)
    rotated = feet(native, sample([1.03, 0., .54]))
    assert rotated[0, 0] > original[0, 0]+1.
    assert rotated[0, 1] < original[0, 1]-1.
    data.qpos[adr+3:adr+7] = [1., 0., 0., 0.]
    translate(model, data, 'white_platform', [0., .1, 0.])
    moved = feet(native, sample([1.03, 0., .54]))
    assert moved[0, 1] == pytest.approx(original[0, 1]+1., abs=1.e-5)
    data.qpos[adr+3:adr+7] = [np.cos(.04), np.sin(.04), 0., 0.]
    mujoco.mj_forward(model, data)
    geom = model.geom('white_deck').id
    top = data.geom_xpos[geom]+data.geom_xmat[geom].reshape(3, 3)[:, 2]*model.geom_size[geom, 2]
    faces, _ = geometry._faces(data)
    face = geometry._select(faces, top+[0., 0., s.wheel_radius], np.array([0., 1., 0.]), s)
    assert face.geom == geom
    assert face.height(top[:2]) == pytest.approx(top[2])
    assert np.all(feet(native, sample(top+[0., 0., s.wheel_radius]))[:, 6] == 1)


def test_matching_platform_velocity_has_no_false_slip_or_edge_approach(native):
    model, data, _, _ = native
    dof = int(model.joint('white_platform_free').dofadr[0])
    data.qvel[dof+1] = .2
    mujoco.mj_forward(model, data)
    moving_together = feet(native, sample([1.03, 0., .54], velocity=(0., .2, 0.)))
    np.testing.assert_allclose(moving_together[:, 7], 0., atol=1.e-6)
    np.testing.assert_allclose(moving_together[:, :3], moving_together[:, 3:6], atol=1.e-5)
    stationary_wheel = feet(native, sample([1.03, 0., .54]))
    np.testing.assert_allclose(stationary_wheel[:, 7], .2/.3, atol=1.e-6)
    data.qvel[:] = 0
    data.qvel[dof+5] = .4
    mujoco.mj_forward(model, data)
    position = np.array([1.13, .1, .54])
    velocity = np.cross([0., 0., .4], position-data.xpos[model.body('white_platform').id])
    rotating_together = feet(native, sample(position, velocity=velocity, omega=(0., 0., .4)))
    np.testing.assert_allclose(rotating_together[:, 7], 0., atol=1.e-6)


def test_diagonal_edge_checks_full_support_envelope_not_only_axis_intercepts():
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><geom type="plane" size="2 2 .1"/>'
        '<geom type="box" pos="0 0 .285" size=".5 .5 .015" '
        'quat=".9238795325 0 0 .3826834324"/></worldbody></mujoco>')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    geometry = NativeTerrainGeometry(model, range(model.ngeom))
    # Both world-axis intercepts are .16 m away, but the support envelope
    # crosses the true NE diagonal edge. All-positive margins are false safe.
    result = geometry.features(data, sample([.27355339, .27355339, .39]), SafeSettings())
    f = result[0, :44].reshape(4, 11).numpy()
    assert np.all(f[:, :3].min(axis=1) < 0)


def test_unsupported_native_collider_still_fails_explicitly():
    model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="sphere" size="1"/></worldbody></mujoco>')
    with pytest.raises(ValueError, match='box/plane'):
        NativeTerrainGeometry(model, [0])


def test_native_packing_matches_checkpoint_shared_formula_on_same_tread(tmp_path, monkeypatch, saved_safe):
    _, _, model, contract, data, execution = scene(tmp_path, monkeypatch, saved_safe, spawn_mat=False)
    safe = execution.safe
    d = safe.physical_sample(data)
    world_x = safe.origin[0]+safe.settings.start_length/2+safe.depth/2
    d['position'][:] = torch.tensor([world_x-safe.origin[0], .2, safe.height+.09])
    d['axis'][:] = torch.tensor([0., 1., 0.])
    d['forces'][:] = torch.tensor([0., 0., 100.])
    d['velocity'][:] = torch.tensor([.1, .1, 0.])
    d['omega'].zero_()
    expected = safety_features(d, safe.settings)
    d['position'][..., 0] += safe.origin[0]
    geometry = NativeTerrainGeometry(model, contract.terrain_geom_ids)
    actual = geometry.features(data, d, safe.settings)
    torch.testing.assert_close(actual, expected, atol=1.e-5, rtol=1.e-5)


@pytest.mark.parametrize('profile', ['teacher', 'front_only_clean'])
def test_custom_runtime_preserves_delivered_scan_and_exports_native_safe48(tmp_path, monkeypatch, saved_safe, profile):
    args, actor, _, contract, data, execution = scene(tmp_path, monkeypatch, saved_safe, XML, profile)
    perception = runtime.make_noise_adapter(args, contract)
    obs = (contract.observe(data, np.zeros(3), np.zeros(16)) if perception is None
           else perception.observe(data, np.zeros(3), np.zeros(16), 0))
    # Unique values expose any accidental straight-stair lane replacement.
    obs[57:] = np.linspace(-.7, .7, 187)
    inputs = execution.policy_inputs(obs, np.zeros(3), actor, data, perception)
    expected = obs[57:].copy()
    if perception is not None:
        expected[perception.map_adapter.last_channels[1] <= 0] = 0.
    np.testing.assert_array_equal(inputs['locomotion_scan'][0], expected)
    assert inputs['safety'].shape == (1, 48)
    args.output = tmp_path/'native.npz'
    monkeypatch.setattr(runtime.baseline, 'heading_controller', lambda *args: None)
    result = runtime.main(args)
    assert result['safe_runtime']['geometry_source'] == 'current_native_collision_poses'
    assert result['safe_runtime']['policy_samples'] == args.steps
    trace = np.load(args.output)
    assert trace['safe_safety'].shape == (args.steps, 48) and np.isfinite(trace['safe_safety']).all()
    assert result['controller_execution']['applied_physics_substeps'] == 4*args.steps
