"""Safe deployment contract, physical inputs, noisy map and runtime wiring."""
from copy import deepcopy
from contextlib import nullcontext
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import mujoco
import numpy as np
import pytest
import torch
from tensordict import TensorDict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deploy/deploy_mujoco"))
import deploy_quiet_landing as runtime
import safe_actor_adapter as bridge
from task_support.stair_quiet_v1_safe.actor import SafeResidualActor
from task_support.stair_quiet_v1_safe.settings import SafeSettings
from task_support.stair_quiet_v1_safe.observations import safety_features
from task_support.stair_quiet_v1.landing import LandingSettings, control_contract
from test_quiet_mujoco_runtime import robot_fixture

torch.set_num_threads(1)


@pytest.fixture
def saved_safe():
    config = dict(class_name=bridge.CLASS, hidden_dims=[512, 256, 128], activation="elu",
                  obs_normalization=False, residual_hidden_dims=[128, 64],
                  distribution_cfg=dict(class_name="GaussianDistribution", init_std=1., std_type="log"))
    kwargs = deepcopy(config)
    kwargs.pop("class_name")
    obs = TensorDict({k: torch.zeros(1, n) for k, n in bridge.GROUPS.items()}, batch_size=[1])
    actor = SafeResidualActor(obs, {"actor": ["policy"]}, "actor", 16, **kwargs)
    # Exercise a learned safety branch rather than the zero-output initializer.
    with torch.no_grad():
        actor.safety_mlp[-1].weight.fill_(.01)
    contract = {"environment": {"stair_tail": asdict(SafeSettings()),
                                "landing_control": asdict(LandingSettings())}}
    manifest = dict(refinement_version="m20_stair_quiet_v1_safe.v1", actor_input_groups=bridge.GROUPS,
        actor_model_config=config, motion_control_version="m20_raw_command_gate_20261017",
        control_profile="legacy_v1", m20_profile="idle_stop_motion_v1",
        noise_refinement=dict(task="stair_quiet_v1_safe", version=1, contract=contract,
            contract_sha256=hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest(),
            source_sha256={"source/rl_training/task_support/"+p: hashlib.sha256(
                (ROOT / "source/rl_training/task_support" / p).read_bytes()).hexdigest()
                for p in bridge.SHARED_SOURCES}, control_contract=control_contract(LandingSettings())))
    return dict(actor_state_dict=actor.state_dict(), m20_training_manifest=manifest)


def scene(tmp_path, monkeypatch, saved, terrain="ascent", profile="teacher", steps=4, spawn_mat=True):
    checkpoint, model = tmp_path / "safe.pt", tmp_path / "robot.xml"
    torch.save(saved, checkpoint)
    robot_fixture(model)
    terrain_flags = ["--terrain-xml", str(terrain)] if isinstance(terrain, Path) else ["--terrain", terrain]
    if not spawn_mat:
        terrain_flags.append("--no-spawn-mat")
    monkeypatch.setattr(sys, "argv", ["deploy_quiet_landing.py", "--checkpoint", str(checkpoint),
        "--model", str(model), *terrain_flags, "--noise-profile", profile, "--steps", str(steps)])
    args = runtime.parse_args()
    actor, document = runtime.load_policy(checkpoint)
    physical = runtime.baseline.load_model(model, args)
    contract = runtime.baseline.M20Contract(physical, args.terrain_config)
    data = mujoco.MjData(physical)
    contract.reset(data)
    execution = runtime.execution_factory(document, actor)(physical, contract)
    return args, actor, physical, contract, data, execution


def test_complete_actor_matches_training_and_keeps_safety_head(saved_safe):
    actor = bridge.load_safe_actor(saved_safe["actor_state_dict"], saved_safe["m20_training_manifest"])
    kwargs = deepcopy(saved_safe["m20_training_manifest"]["actor_model_config"])
    kwargs.pop("class_name")
    obs = TensorDict({k: torch.randn(1, n) for k, n in bridge.GROUPS.items()}, batch_size=[1])
    original = SafeResidualActor(obs, {"actor": ["policy"]}, "actor", 16, **kwargs)
    original.load_state_dict(saved_safe["actor_state_dict"], strict=True)
    with torch.inference_mode():
        expected = original(obs)
        torch.testing.assert_close(actor(obs), expected, rtol=0, atol=0)
        actor.safety_mlp[-1].weight.zero_()
        assert not torch.allclose(actor(obs), expected)


@pytest.mark.parametrize("corruption", ["missing_head", "nan", "groups", "digest", "source"])
def test_corrupt_safe_is_rejected(saved_safe, corruption):
    saved = deepcopy(saved_safe)
    state, manifest = saved["actor_state_dict"], saved["m20_training_manifest"]
    if corruption == "missing_head":
        del state["safety_mlp.0.weight"]
    elif corruption == "nan":
        state["safety_mlp.0.weight"][0, 0] = torch.nan
    elif corruption == "groups":
        manifest["actor_input_groups"] = {"policy": 244, "motion_control": 3}
    elif corruption == "digest":
        manifest["noise_refinement"]["contract_sha256"] = "0"*64
    else:
        manifest["noise_refinement"]["source_sha256"] = {}
    with pytest.raises((ValueError, RuntimeError)):
        bridge.load_safe_actor(state, manifest)


def test_legacy_entry_still_refuses_safe(tmp_path, saved_safe):
    path = tmp_path / "safe.pt"
    torch.save(saved_safe, path)
    with pytest.raises(ValueError, match="cannot execute"):
        runtime.baseline.load_actor(path)


def test_real_xml_geometry_and_identical_shared_features(tmp_path, monkeypatch, saved_safe):
    args, actor, model, contract, data, execution = scene(tmp_path, monkeypatch, saved_safe, spawn_mat=False)
    safe = execution.safe
    p = args.terrain_config.parameters
    assert safe.width == p["width"][0]
    assert safe.height == p["step_height"][0]
    assert safe.steps == int(p["step_count"][0])
    assert safe.origin[0]+safe.settings.start_length/2 == p["start_x"][0]
    assert safe.origin[0]+safe.settings.tile_length-safe.settings.start_length/2 == pytest.approx(
        p["start_x"][0]+safe.steps*safe.depth+p["platform_length"][0])
    obs = contract.observe(data, np.zeros(3), np.zeros(16))
    packed = execution.policy_inputs(obs, np.array([.5, 0, 0], dtype=np.float32), actor, data)
    torch.testing.assert_close(packed["safety"], safety_features(safe.physical_sample(data), safe.settings))
    assert {k: v.shape[-1] for k, v in packed.items()} == bridge.GROUPS


def test_wheel_link_velocity_and_axis_are_not_com_velocity(tmp_path, monkeypatch, saved_safe):
    _, _, model, contract, data, execution = scene(tmp_path, monkeypatch, saved_safe)
    wheel = int(contract.wheel_body_ids[0])
    model.body_ipos[wheel] = [.04, 0, 0]
    data.qvel.fill(0.)
    data.qvel[5] = 1.3
    mujoco.mj_forward(model, data)
    sample = execution.safe.physical_sample(data)
    twist = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, wheel, twist, 0)
    link = twist[3:]+np.cross(twist[:3], data.xpos[wheel]-data.xipos[wheel])
    np.testing.assert_allclose(sample["velocity"][0, 0], link, atol=1e-6)
    assert np.linalg.norm(link-twist[3:]) > .01
    np.testing.assert_allclose(sample["axis"][0, 0], data.xmat[wheel].reshape(3, 3)[:, 1], atol=1e-6)


def test_supported_rolling_and_loaded_spin_have_different_slip(tmp_path, monkeypatch, saved_safe):
    _, _, _, _, data, execution = scene(tmp_path, monkeypatch, saved_safe, spawn_mat=False)
    d = execution.safe.physical_sample(data)
    d["position"][..., 0] = .2  # approach before the first riser
    d["position"][..., 2] = .09
    d["axis"][:] = torch.tensor([0., 1., 0.])
    d["forces"][:] = torch.tensor([0., 0., 100.])
    d["velocity"][:] = torch.tensor([.9, 0., 0.])
    d["omega"][:] = torch.tensor([0., 10., 0.])
    feet = safety_features(d, execution.safe.settings)[0, :44].reshape(4, 11)
    assert torch.all(feet[:, 6] == 1)
    torch.testing.assert_close(feet[:, 7], torch.zeros(4), atol=1e-6, rtol=0)
    d["velocity"].zero_()
    assert torch.all(safety_features(d, execution.safe.settings)[0, :44].reshape(4, 11)[:, 7] > 2.9)


def test_root_velocity_matches_isaac_com_semantics(tmp_path, monkeypatch, saved_safe):
    _, _, model, contract, data, execution = scene(tmp_path, monkeypatch, saved_safe)
    model.body_ipos[contract.base_id] = [.07, .02, 0.]
    model.body_sameframe[contract.base_id] = 0
    mujoco.mj_setConst(model, data)
    data.qvel.fill(0.)
    data.qvel[5] = 1.3
    mujoco.mj_forward(model, data)
    sample = execution.safe.physical_sample(data)
    # Pure yaw about the link origin moves the offset COM laterally.
    np.testing.assert_allclose(sample["root_velocity"][0], [-.026, .091, 0.], atol=1e-6)
    features = safety_features(sample, execution.safe.settings)
    assert features[0, 45].item() == pytest.approx(.091, abs=1e-6)


def test_one_metre_stair_front_rear_margins_and_prediction(tmp_path, monkeypatch, saved_safe):
    _, _, _, _, data, execution = scene(tmp_path, monkeypatch, saved_safe, spawn_mat=False)
    s = replace(execution.safe.settings, front_lateral_margin=.04, rear_lateral_margin=.06,
                front_tread_margin=.03, rear_tread_margin=.05)
    d = execution.safe.physical_sample(data)
    d["width"][:] = 1.
    d["height"][:] = .15
    d["depth"][:] = .30
    d["position"][:] = torch.tensor([1.15, .2, .24])
    d["axis"][:] = torch.tensor([0., 1., 0.])
    d["velocity"][:] = torch.tensor([.1, .1, 0.])
    d["omega"].zero_()
    feet = safety_features(d, s)[0, :44].reshape(4, 11)
    # First tread [1,1.3], bottom z=.15; radius=.09, half axle=.027.
    np.testing.assert_allclose(feet[:, 0], [1., 1., .8, .8], atol=1e-5)
    np.testing.assert_allclose(feet[:, 1], [1.43, 1.43, 1.23, 1.23], atol=1e-5)
    np.testing.assert_allclose(feet[:, 2], [5.43, 5.43, 5.23, 5.23], atol=1e-5)
    np.testing.assert_allclose(feet[:, 4], feet[:, 1]-.12, atol=1e-5)


def test_open_approach_length_is_read_from_actual_xml(tmp_path, monkeypatch, saved_safe):
    args, actor, _, _, _, _ = scene(tmp_path, monkeypatch, saved_safe, "ascent_open", spawn_mat=False)
    source = ROOT / "deploy/deploy_mujoco/terrains/stairs_ascent_open.xml"
    custom = tmp_path / "open.xml"
    custom.write_text(source.read_text().replace('name="approach_length" data="2.0"',
                                               'name="approach_length" data="3.0"'))
    args.terrain, args.terrain_xml = None, custom
    model = runtime.baseline.load_model(args.model, args)
    contract = runtime.baseline.M20Contract(model, args.terrain_config)
    safe = bridge.SafeMujocoAdapter(model, contract, actor)
    assert safe.settings.start_length == 3.
    assert safe.origin[0]-safe.settings.start_length/2 == pytest.approx(
        args.terrain_config.parameters["start_x"][0]-3.)


@pytest.mark.parametrize("terrain", ["flat", "ascent", "ascent_open"])
def test_spawn_mat_safe_reads_real_height_and_edges(tmp_path, monkeypatch, saved_safe, terrain):
    args, actor, model, contract, data, execution = scene(tmp_path, monkeypatch, saved_safe, terrain)
    safe = execution.safe
    assert args.terrain_config.spawn_mat is not None
    assert safe.native is not None
    assert safe.metadata()["geometry_source"] == "current_native_collision_poses"
    mat = model.geom("terrain_spawn_mat").id
    # Move the actual mat and robot together vertically. The regression must
    # read current collision poses rather than a z=0 or XML-only support guess.
    model.geom_pos[mat, 2] += .2
    data.qpos[int(contract.root_joint.qposadr[0])+2] += .2
    mujoco.mj_forward(model, data)
    safe.forces[:] = [0., 0., 100.]  # Explicit measured-force input fixture.
    sample = safe.physical_sample(data)
    faces, _ = safe.native._faces(data)
    for wheel in range(4):
        face = safe.native._select(faces, sample["position"][0, wheel].numpy(),
                                  sample["axis"][0, wheel].numpy(), safe.settings)
        assert face.geom == mat
        assert face.height(sample["position"][0, wheel, :2].numpy()) == pytest.approx(
            data.geom_xpos[mat, 2]+model.geom_size[mat, 2])
    observation = contract.observe(data, np.zeros(3), np.zeros(16))
    packed = execution.policy_inputs(observation, np.zeros(3), actor, data)
    feet = packed["safety"][0, :44].reshape(4, 11)
    assert torch.all(feet[:, 6] == 1)
    assert torch.all((feet[:, 0] > 0) & (feet[:, 0] < 10))
    torch.testing.assert_close(packed["locomotion_scan"][0], torch.from_numpy(observation[57:]))


def _two_fixed_flex_model():
    spec = mujoco.MjSpec()
    for index, (x, z) in enumerate(((0., 0.), (2., 1.))):
        spec.add_flex(name=f"surface_{index}", dim=2, radius=0., group=0,
            vertbody=["world"], vert=[x, -1., z, x+1., -1., z, x+1., 1., z, x, 1., z],
            elem=[0, 1, 2, 0, 2, 3], contype=1, conaffinity=1)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def test_native_flex_addresses_and_shared_triangle_seam_are_physical():
    from safe_native_geometry import NativeTerrainGeometry

    model, data = _two_fixed_flex_model()
    geometry = NativeTerrainGeometry(model, (), (0, 1))
    faces, _ = geometry._faces(data)
    assert len(faces) == 4 and len({f.geom for f in faces}) == 4
    assert all(f.body == 0 and f.geom < 0 for f in faces)
    assert sum(f.contains([.5, 0.]) for f in faces) == 2
    assert sum(f.contains([2.5, 0.]) for f in faces) == 2
    assert not any(f.contains([1.5, 0.]) for f in faces), "The real gap must remain empty"
    settings = SafeSettings()
    sample = {k: torch.as_tensor(np.asarray(v), dtype=torch.float32).unsqueeze(0) for k, v in dict(
        position=[[2.5, 0., 1.+settings.wheel_radius]]*4, velocity=[[0., 0., 0.]]*4,
        omega=[[0., 0., 0.]]*4, axis=[[0., 1., 0.]]*4, forces=[[0., 0., 100.]]*4,
        root=[2.5, 0., 1.5], root_velocity=[0., 0., 0.], yaw=0., width=2.).items()}
    result = geometry.features(data, sample, settings)[0, :44].reshape(4, 11)
    assert torch.isfinite(result).all()
    assert torch.all(result[:, 6] == 1)
    # The internal triangle diagonal is not an edge of the physical deck.
    expected = (.5-settings.contact_patch_radius-settings.front_tread_margin)/.1
    assert result[0, 0].item() == pytest.approx(expected, abs=1.e-6)
    torch.testing.assert_close(result[:, :3], result[:, 3:6])


@pytest.mark.parametrize("invalid", ["radius", "group", "dimension", "moving", "nonrigid", "visual"])
def test_native_flex_rejects_unsupported_surfaces(invalid):
    from safe_native_geometry import NativeTerrainGeometry

    model, _ = _two_fixed_flex_model()
    if invalid == "radius":
        model.flex_radius[0] = .01
    elif invalid == "group":
        model.flex_group[0] = 2
    elif invalid == "dimension":
        model.flex_dim[0] = 1
    elif invalid == "moving":
        model.flex_vertbodyid[0] = 1
    elif invalid == "nonrigid":
        model.flex_rigid[0] = False
    else:
        model.flex_contype[0] = model.flex_conaffinity[0] = 0
    with pytest.raises(ValueError, match="world-fixed group-0 zero-radius"):
        NativeTerrainGeometry(model, (), (0,))


def test_physical_contact_feedback_then_reset(tmp_path, monkeypatch, saved_safe):
    _, _, model, contract, data, execution = scene(tmp_path, monkeypatch, saved_safe, "flat")
    data.qpos[contract.root_joint.qposadr[0]+2] = .38
    data.qpos[contract.qpos_ids] = 0.
    data.qvel.fill(0.)
    mujoco.mj_forward(model, data)
    mujoco.mj_step1(model, data)
    mujoco.mj_step2(model, data)
    execution.after_step(data)
    assert np.all(execution.safe.forces[:, 2] > 0)
    assert execution.safe.force_time == data.time
    before = execution.safe.forces.copy()
    mujoco.mj_fwdPosition(model, data)
    mujoco.mj_fwdVelocity(model, data)
    np.testing.assert_array_equal(execution.safe.forces, before)
    execution.reset()
    assert execution.safe.force_time is None
    assert not execution.safe.forces.any()


def test_front_only_holes_remain_missing_and_oracle_scope_is_explicit(tmp_path, monkeypatch, saved_safe):
    args, actor, _, contract, data, execution = scene(tmp_path, monkeypatch, saved_safe, profile="front_only_clean")
    perception = runtime.make_noise_adapter(args, contract)
    observation = perception.observe(data, np.zeros(3), np.zeros(16), 0)
    missing = perception.map_adapter.last_channels[1] == 0
    assert missing.any() and not missing.all()
    monkeypatch.setattr(contract, "height_scan", lambda *args: pytest.fail("oracle map refill"))
    packed = execution.policy_inputs(observation, np.array([.5, 0, 0]), actor, data, perception)
    assert not packed["locomotion_scan"][0, missing].any()
    assert not packed["policy"][0, 57:][missing].any()
    assert execution.metadata()["safe_runtime"]["safety_observation"] == "simulation_privileged_safety48"


def test_safe_headless_executes_and_exports_actual_inputs(tmp_path, monkeypatch, saved_safe):
    args, _, _, _, _, _ = scene(tmp_path, monkeypatch, saved_safe, profile="front_only_clean")
    args.output = tmp_path / "trace.npz"
    # The standalone synthetic checkpoint has no heading tuning; disable that
    # unrelated controller instead of pretending to validate it with mock data.
    monkeypatch.setattr(runtime.baseline, "heading_controller", lambda *args: None)
    result = runtime.main(args)
    assert result["safe_runtime"]["policy_samples"] == args.steps
    assert result["controller_execution"]["applied_physics_substeps"] == 4*args.steps
    trace = np.load(args.output)
    assert trace["safe_safety"].shape == (args.steps, 48)
    assert trace["safe_locomotion_scan"].shape == (args.steps, 187)
    assert np.isfinite(trace["safe_safety"]).all()
    np.testing.assert_allclose(trace["safe_policy_time_s"][1:], trace["safe_contact_sample_time_s"][1:])
    assert np.isnan(trace["safe_contact_sample_time_s"][0])


def test_unknown_geometry_fails_explicitly(tmp_path, monkeypatch, saved_safe):
    with pytest.raises(ValueError, match="unsupported terrain"):
        scene(tmp_path, monkeypatch, saved_safe, "descent")


@pytest.mark.parametrize("terrain", ["flat", ROOT / "deploy/deploy_mujoco/terrains/sample_stairs.xml"])
def test_manual_viewer_reaches_complete_safe_actor_and_resets(tmp_path, monkeypatch, saved_safe, terrain):
    args, _, _, _, _, _ = scene(tmp_path, monkeypatch, saved_safe, terrain, "front_only_clean", steps=12)
    args.viewer = True
    args.output = tmp_path / "manual.npz"
    # This test exercises handover, whose real M20 ready gate requires a
    # standing base above .42 m. Give the rigid fixture actual leg clearance;
    # its old short legs only passed because reset suspended it in mid-air.
    robot_fixture(args.model, leg_length=.45)
    b = runtime.baseline
    real_reset = b.M20Contract.reset
    # This toy fixture has no articulated legs. Start it upright to exercise
    # the real viewer loop/physics without claiming to test its stand-up skill.
    monkeypatch.setattr(b.M20Contract, "reset", lambda self, data, prone=False: real_reset(self, data))
    monkeypatch.setattr(b, "heading_controller", lambda *args: None)
    monkeypatch.setattr(b, "STAND_PHASE_DURATION", .005)
    monkeypatch.setattr(b.time, "sleep", lambda *args: None)
    events = iter([["c"], ["z"], ["c"], [], ["r"], ["c"], ["z"], ["c"], [], ["escape"]])
    keys = Mock(_window_keys=None)
    keys.take_events.side_effect = lambda: next(events)
    keys.velocity_command.return_value = np.array([.7, 0, 0], dtype=np.float32)
    terminal = Mock()
    terminal.start.return_value = False
    viewer = SimpleNamespace(lock=nullcontext, cam=SimpleNamespace(), is_running=lambda: True, sync=lambda: None)
    monkeypatch.setattr(b, "TeleopKeyboard", lambda: keys)
    monkeypatch.setattr(b, "TerminalInput", lambda *args: terminal)
    monkeypatch.setattr(b, "launch_viewer", lambda *args, **kwargs: (viewer, []))
    monkeypatch.setattr(b, "close_viewer", lambda *args: None)
    monkeypatch.setattr(runtime.QuietMujocoMetrics, "draw", lambda *args: None)
    monkeypatch.setattr(runtime.QuietMujocoMetrics, "snapshot", lambda *args: None)
    import teacher_noise_adapter
    monkeypatch.setattr(teacher_noise_adapter.CleanFrontTeacherAdapter, "draw", lambda *args: None)
    result = runtime.main(args)
    assert [t["mode"] for t in result["transitions"]].count("policy") == 2
    assert result["safe_runtime"]["policy_samples"] == 4
    trace = np.load(tmp_path / "manual_quiet.npz")
    assert trace["safe_safety"].shape == (4, 48)
    # Both handovers discard the previous policy/contact sample.
    assert np.isnan(trace["safe_contact_sample_time_s"][[0, 2]]).all()
    trajectory = np.load(args.output)
    modes = trajectory["mode"]
    np.testing.assert_array_equal(trajectory["requested_command"][modes != "policy"], 0)
    np.testing.assert_allclose(trajectory["requested_command"][modes == "policy"], [[.7, 0, 0]]*4)
