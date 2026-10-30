"""Dedicated teacher runtime: contract rejection and causal physical execution."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import sys
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from tensordict import TensorDict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deploy/deploy_mujoco"))
import deploy_quiet_landing as runtime
from task_support.stair_quiet_v1.landing import LandingSettings, control_contract

torch.set_num_threads(1)


@pytest.fixture
def saved_teacher():
    obs = TensorDict({"policy": torch.zeros(1, 244)}, batch_size=[1])
    actor = runtime.baseline.MLPModel(
        obs, {"actor": ["policy"]}, "actor", 16, hidden_dims=[512, 256, 128],
        activation="elu", obs_normalization=False,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1., "std_type": "log"})
    return {"actor_state_dict": actor.state_dict()}


def with_control(saved, settings=None):
    saved = deepcopy(saved)
    settings = settings or LandingSettings()
    saved["m20_training_manifest"] = {"noise_refinement": {
        "control_contract": control_contract(settings),
        "contract": {"environment": {"landing_control": asdict(settings)}}}}
    return saved


def robot_fixture(path, leg_length=.3):
    """Small physical robot in shuffled storage order; no external MJCF needed."""
    bodies, motors = [], []
    for name in reversed(runtime.baseline.JOINT_NAMES):
        body = name.removesuffix("_joint")
        shape = ('type="cylinder" size=".09 .027" quat=".70710678 .70710678 0 0"'
                 if body.endswith("_wheel") else 'type="sphere" size=".02"')
        bodies.append(f'<body name="{body}" pos="0 0 -{leg_length}"><joint name="{name}" axis="0 1 0"/>'
                      f'<geom {shape} mass=".1" group="1"/></body>')
        motors.append(f'<motor name="{name}" joint="{name}"/>')
    path.write_text('<mujoco><worldbody><body name="base_link"><freejoint name="floating_base"/>'
                    '<geom type="box" size=".2 .1 .1" mass="10" group="1"/>'
                    + ''.join(bodies) + '</body></worldbody><actuator>'
                    + ''.join(motors) + '</actuator></mujoco>')


def arguments(monkeypatch, tmp_path, saved, steps=8):
    checkpoint, model = tmp_path / "teacher.pt", tmp_path / "robot.xml"
    torch.save(saved, checkpoint)
    robot_fixture(model)
    monkeypatch.setattr(sys, "argv", ["deploy_quiet_landing.py", "--checkpoint", str(checkpoint),
                                     "--model", str(model), "--terrain", "flat", "--steps", str(steps)])
    return runtime.parse_args()


@pytest.mark.parametrize("settings", [LandingSettings(), LandingSettings(mode="precontact", landing_kp_scale=.9)])
def test_checkpoint_settings_select_actual_runtime(monkeypatch, tmp_path, saved_teacher, settings):
    args = arguments(monkeypatch, tmp_path, with_control(saved_teacher, settings))
    actor, document = runtime.load_policy(args.checkpoint)
    assert document == control_contract(settings)
    with torch.inference_mode():
        assert actor(TensorDict({"policy": torch.zeros(1, 244)}, batch_size=[1])).shape == (1, 16)
    # The ordinary loader must continue to refuse this same checkpoint.
    with pytest.raises(ValueError, match="cannot execute"):
        runtime.baseline.load_actor(args.checkpoint)


@pytest.mark.parametrize("corruption", ["source", "missing", "null", "conflicting", "settings"])
def test_invalid_contract_rejected_before_actor_loading(tmp_path, corruption):
    saved = with_control({"actor_state_dict": {}})
    noise = saved["m20_training_manifest"]["noise_refinement"]
    if corruption == "source":
        noise["control_contract"]["source_sha256"] = "0" * 64
    elif corruption == "missing":
        del noise["control_contract"]
    elif corruption == "null":
        noise["control_contract"] = None
    elif corruption == "conflicting":
        saved["quiet_execution_control"] = control_contract(LandingSettings(mode="off"))
    else:
        noise["contract"]["environment"]["landing_control"]["mode"] = "off"
    path = tmp_path / "bad.pt"
    torch.save(saved, path)
    with pytest.raises(ValueError, match="control|landing|execution|Conflicting"):
        runtime.load_policy(path)


def test_physical_substeps_execute_controller_and_keep_nominal_history(monkeypatch, tmp_path, saved_teacher):
    args = arguments(monkeypatch, tmp_path, with_control(saved_teacher))
    args.output = tmp_path / "signals.npz"
    applied, observations, captured = [], [], {}
    real_apply, real_inputs = runtime.MujocoLanding.apply, runtime.baseline.policy_inputs

    def apply(self, data, action, force, valid, active, dt, feedback_age):
        applied.append((action.copy(), force.copy(), valid.copy(), float(data.time)))
        return real_apply(self, data, action, force, valid, active, dt, feedback_age)

    def inputs(observation, command, actor):
        observations.append(observation.copy())
        return real_inputs(observation, command, actor)

    def save(self, path, metadata):
        captured.update(self.extended_records)

    def unexpected_nominal_pd(*args):
        pytest.fail("Marked checkpoint bypassed its landing controller")

    monkeypatch.setattr(runtime.MujocoLanding, "apply", apply)
    monkeypatch.setattr(runtime.baseline, "policy_inputs", inputs)
    monkeypatch.setattr(runtime.baseline.M20Contract, "apply_pd", unexpected_nominal_pd)
    monkeypatch.setattr(runtime.QuietMujocoMetrics, "save", save)
    result = runtime.main(args)
    assert result["controller_execution"]["applied_physics_substeps"] == 32
    assert result["controller_execution"]["wheel_samples"] == 128
    assert result["duration_s"] == pytest.approx(.16)
    assert not applied[0][2].any()  # No completed contact solve exists at startup.
    assert all(row[2].all() for row in applied[1:])
    np.testing.assert_array_equal(observations[0][41:57], np.zeros(16))
    for i in range(1, 8):
        np.testing.assert_array_equal(observations[i][41:57], applied[4 * (i - 1)][0])
    assert len(captured["landing_leg_target"]) == 32
    assert np.isfinite(captured["landing_leg_target"]).all()


def test_unmarked_teacher_retains_nominal_pd(monkeypatch, tmp_path, saved_teacher):
    args = arguments(monkeypatch, tmp_path, saved_teacher)
    calls = []
    original = runtime.baseline.M20Contract.apply_pd

    def pd(self, data, action):
        calls.append(action.copy())
        return original(self, data, action)

    def unexpected_controller(*args):
        pytest.fail("Unmarked legacy checkpoint acquired a new controller")

    monkeypatch.setattr(runtime.baseline.M20Contract, "apply_pd", pd)
    monkeypatch.setattr(runtime, "MujocoLanding", unexpected_controller)
    result = runtime.main(args)
    assert len(calls) == 32
    assert result["controller_execution"]["enabled"] is False
    assert "quiet_execution_control" not in result


def test_viewer_defaults_to_unlimited_manual_route(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["deploy_quiet_landing.py", "--model", "unused.xml", "--viewer"])
    args = runtime.parse_args()
    assert args.steps is None
    actor, document = object(), control_contract(LandingSettings())
    monkeypatch.setattr(runtime, "load_policy", lambda path: (actor, document))
    seen = {}

    def interactive(args, **kwargs):
        seen.update(kwargs)
        assert args.quiet_landing is True
        return "manual"

    monkeypatch.setattr(runtime.baseline, "interactive_rollout", interactive)
    assert runtime.main(args) == "manual"
    assert seen["actor"] is actor
    assert callable(seen["execution_factory"])


@pytest.mark.parametrize('reset_source', [None, 'key', 'native'])
def test_manual_handover_commands_release_and_reentry(monkeypatch, tmp_path, reset_source):
    b = runtime.baseline
    data = SimpleNamespace(time=0., xpos=np.array([[0., 0., .5]]), xmat=np.eye(3).reshape(1, 9),
                           qpos=np.zeros(16), qvel=np.zeros(16), ctrl=np.zeros(16))
    observations, applied = [], []

    def observe(data, command, previous):
        obs = np.zeros(244, dtype=np.float32)
        obs[6:9], obs[41:57] = command, previous
        observations.append(obs.copy())
        return obs

    contract = SimpleNamespace(base_id=0, qpos_ids=np.arange(16), actuator_ids=np.arange(16),
                               reset=Mock(), height_scan=Mock(), observe=observe,
                               ground_height=lambda *args: 0., apply_pose_pd=Mock(), apply_damping=Mock(),
                               apply_pd=Mock(side_effect=AssertionError("quiet policy bypass")),
                               crossing_summary=lambda *args: {})
    model = SimpleNamespace(opt=SimpleNamespace(timestep=.005))
    args = SimpleNamespace(checkpoint=tmp_path / "teacher.pt", model=tmp_path / "robot.xml",
                           output=None, snapshot=None, steps=20, viewer=True, live_plot=False,
                           quiet_landing=True, stop_on_fall=False, print_every=100,
                           command_x=9.99, terrain_config=SimpleNamespace(summary=lambda: {}))
    # C before Z cannot move. Queued C after Z enters policy. R interrupts;
    # C cannot leave damping, and a new Z/C starts clean controller/history.
    events = iter([[], ["c"], ["z"], ["c"], [], [], [], ["r"], ["c"],
                   ["z"], ["c"], [], ["x"], ["r"],
                   ["reset"] if reset_source == 'key' else [], ["escape"]])
    commands = np.array([[0, 0, 0], [.7, 0, 0], [0, 0, .6], [0, 0, 0],
                         [0, 0, 0], [0, .5, 0]], dtype=np.float32)
    keys = Mock()
    keys.take_events.side_effect = lambda: next(events)
    keys.velocity_command.side_effect = list(commands)
    terminal = Mock(); terminal.start.return_value = False
    viewer = SimpleNamespace(lock=nullcontext, cam=SimpleNamespace(), is_running=lambda: True, sync=lambda: None)
    sync_count = 0
    def sync():
        nonlocal sync_count
        sync_count += 1
        if reset_source == 'native' and sync_count == 15:
            data.time = 0.
            data.qpos[:] = 99.  # The XML default must never reach a physics step.
    viewer.sync = sync
    def reset(data, prone=False):
        data.time = 0.
        data.qpos[:] = 0.
    contract.reset.side_effect = reset
    metrics = SimpleNamespace(tracker=SimpleNamespace(distance=torch.zeros(1), begin_step=Mock()),
                              kinematics=data, records={}, extended_records={}, events=[],
                              pre_step=Mock(), post_step=Mock(), draw=Mock(), reset=Mock(), summary=lambda: {})
    execution = Mock()
    execution.apply.side_effect = lambda data, action, command: applied.append(action.copy()) or np.zeros(16)
    execution.metadata.return_value = {"controller_execution": {"enabled": True}}
    actor = lambda obs: torch.ones(1, 16)
    import quiet_landing_metrics
    monkeypatch.setattr(quiet_landing_metrics, "QuietMujocoMetrics", lambda *args: metrics)
    monkeypatch.setattr(b, "load_model", lambda *args: model)
    monkeypatch.setattr(b, "M20Contract", lambda *args: contract)
    monkeypatch.setattr(b.mujoco, "MjData", lambda model: data)
    monkeypatch.setattr(b, "TeleopKeyboard", lambda: keys)
    monkeypatch.setattr(b, "TerminalInput", lambda keys: terminal)
    monkeypatch.setattr(b, "launch_viewer", lambda *args, **kwargs: (viewer, []))
    monkeypatch.setattr(b, "close_viewer", Mock())
    monkeypatch.setattr(b, "draw_height_scan", Mock())
    monkeypatch.setattr(b, "STAND_PHASE_DURATION", .009)
    monkeypatch.setattr(b.time, "sleep", lambda *args: None)
    for name in ("mj_step1", "mj_fwdPosition", "mj_fwdVelocity"):
        monkeypatch.setattr(b.mujoco, name, lambda *args: None)

    def integrate(*args):
        assert not np.any(data.qpos == 99.)
        data.time += .005

    monkeypatch.setattr(b.mujoco, "mj_step2", integrate)
    result = b.interactive_rollout(args, actor=actor, execution_factory=lambda *args: execution)
    assert len(observations) == 6
    np.testing.assert_array_equal(np.array(observations)[:, 6:9], commands)
    for index in (0, 4):
        np.testing.assert_array_equal(observations[index][41:57], np.zeros(16))
        np.testing.assert_array_equal(applied[4 * index], np.zeros(16))
    assert np.abs(observations[2][41:57]).sum() > 0
    assert execution.apply.call_count == 24
    assert execution.reset.call_count == 9 + bool(reset_source)
    assert contract.reset.call_count == 1 + bool(reset_source)
    assert metrics.reset.call_count == bool(reset_source)
    assert result["final_mode"] == ("prone" if reset_source else "damping")
    assert [t["mode"] for t in result["transitions"]] == [
        "prone", "standing", "ready", "policy", "damping", "standing", "ready", "policy", "lying", "damping"] + (["prone"] if reset_source else [])
    keys.close.assert_called_once()
    terminal.close.assert_called_once()


def test_keyboard_hold_release_and_focus_loss(monkeypatch):
    b = runtime.baseline
    held = set()
    window = SimpleNamespace(pressed=lambda: set(held), focused=lambda: True, close=lambda: None)
    monkeypatch.setattr(b.X11WindowKeys, "open", lambda: window)
    keys = b.TeleopKeyboard()
    np.testing.assert_array_equal(keys.velocity_command(), np.zeros(3))
    held.add("w")
    first = keys.velocity_command()
    assert 0 < first[0] < .7
    for _ in range(round(b.TELEOP_RISE_TIME / b.POLICY_DT)):
        command = keys.velocity_command()
    np.testing.assert_allclose(command, [.7, 0, 0])
    held.clear()  # Release, or X11 reports no keys after focus loss.
    for _ in range(round(b.TELEOP_STOP_TIME / b.POLICY_DT) + 1):
        keys.velocity_command()
    np.testing.assert_array_equal(keys.velocity_command(), np.zeros(3))
    held.add('w')
    assert keys.velocity_command()[0] > 0
    window.focused = lambda: False
    np.testing.assert_array_equal(keys.velocity_command(), np.zeros(3))
    keys.close()


def test_reset_clears_old_quiet_contact_events(monkeypatch, tmp_path, saved_teacher):
    from quiet_landing_metrics import QuietMujocoMetrics
    args = arguments(monkeypatch, tmp_path, saved_teacher)
    b = runtime.baseline
    model = b.load_model(args.model, args)
    contract = b.M20Contract(model, args.terrain_config)
    metrics = QuietMujocoMetrics(model, contract)
    metrics.last_event_time[:] = 100.
    metrics.tracker.contact[:] = True
    metrics.tracker.age[:] = 100.
    metrics.events.append({'time': 100.})
    metrics.records['time'].append(100.)
    metrics.reset()
    assert np.isneginf(metrics.last_event_time).all()
    assert not metrics.tracker.contact.any()
    assert not metrics.tracker.age.any()
    assert not metrics.events and not metrics.records['time']


def test_controller_handover_clears_contact_and_map_but_keeps_counters(monkeypatch, tmp_path, saved_teacher):
    args = arguments(monkeypatch, tmp_path, saved_teacher)
    b = runtime.baseline
    model = b.load_model(args.model, args)
    contract = b.M20Contract(model, args.terrain_config)
    execution = runtime.QuietExecution(model, contract, control_contract(LandingSettings(contact_delay_steps=2)))
    data = b.mujoco.MjData(model)
    contract.reset(data)
    contract.height_scan(data)
    execution.observe(data)
    b.mujoco.mj_step1(model, data)
    execution.apply(data, np.zeros(16), np.zeros(3))
    assert execution.adapter.control.contact.is_inference()
    with torch.inference_mode():
        execution.adapter.control.contact.fill_(True)
        execution.adapter.control.elapsed.fill_(3.)
        execution.adapter.control.correction.fill_(.01)
        execution.adapter.control.counts.fill_(7)
    execution.adapter.memory = object()
    execution.feedback_force.fill(100.)
    execution.feedback_valid.fill(True)
    execution.reset()
    assert not execution.adapter.control.contact.any()
    assert not execution.adapter.control.correction.any()
    assert not execution.adapter.control.elapsed.any()
    assert execution.adapter.control.counts.tolist() == [7] * 6
    assert not execution.feedback_valid.any()
    assert not execution.feedback_force.any()
    assert execution.adapter.memory is None
    assert len(execution.adapter.contact_history) == 2


def test_quiet_runtime_delivers_noisy_map_and_ages_to_controller(monkeypatch, tmp_path, saved_teacher):
    args = arguments(monkeypatch, tmp_path, with_control(saved_teacher))
    args.noise_profile = "front_only"
    observed = []
    original = runtime.MujocoLanding.observe

    def observe(self, data, adapter):
        assert adapter is not None
        observed.append((adapter.last_channels[1].copy(), adapter.map.cell_age.clone()))
        return original(self, data, adapter)

    monkeypatch.setattr(runtime.MujocoLanding, "observe", observe)
    result = runtime.main(args)
    assert len(observed) == 8
    assert all((mask == 0).any() for mask, age in observed)
    assert all(age.shape == (1, 187) for mask, age in observed)
    assert result["noise_profile"] == "front_only"
    assert result["noise"]["perception"]["front_only"] is True
    assert result["controller_execution"]["applied_physics_substeps"] == 32
