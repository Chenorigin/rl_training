"""Run the actual playback adapter; physical fixture tests are bounded on CPU."""
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from test_quiet_mujoco_runtime import robot_fixture, runtime
from task_support.cts.configuration import default_settings
from task_support.cts.joint_limits_mujoco import MujocoJointLimits, PositionLimitError
from task_support.stair_quiet_v1.landing import LandingSettings, control_contract


def fixture(tmp_path):
    cfg = default_settings()['joint_limits']
    path = tmp_path/'robot.xml'
    robot_fixture(path)
    xml = ET.parse(path)
    ET.SubElement(xml.getroot(), 'compiler', angle='radian')
    for joint in xml.iter('joint'):
        name = joint.get('name')
        if name in cfg['joint_names']:
            i = cfg['joint_names'].index(name)
            joint.set('range', ' '.join(map(str, cfg['feedback_bounds_rad'][i])))
            joint.set('limited', 'true')
    xml.write(path)
    import sys
    from unittest.mock import patch
    with patch.object(sys, "argv", ["test", "--model", str(path), "--terrain", "flat"]):
        args = runtime.parse_args()
    model = runtime.baseline.load_model(path, args)
    model.opt.timestep = .005
    contract = runtime.baseline.M20Contract(model, args.terrain_config)
    data = mujoco.MjData(model)
    contract.reset(data)
    actor = SimpleNamespace(joint_limit_contract=cfg)
    execution = runtime.QuietExecution(model, contract, control_contract(LandingSettings()), actor)
    return cfg, model, contract, data, execution


def test_reversed_physical_joint_order_maps_by_name_and_checks_safe_action(tmp_path):
    _, _, contract, data, execution = fixture(tmp_path)
    assert list(contract.qpos_ids[:12]) != sorted(contract.qpos_ids[:12])
    execution.check_issued(data, np.zeros(16))
    # Wheel positions are deliberately not subject to finite leg-angle ranges.
    data.qpos[contract.qpos_ids[12:]] = 1.e6
    execution.check_issued(data, np.zeros(16))


def test_nominal_fault_blocks_controller_and_does_not_write_actuation(tmp_path, monkeypatch):
    _, _, _, data, execution = fixture(tmp_path)
    monkeypatch.setattr(execution.adapter, 'apply', lambda *a: pytest.fail('invalid target reached controller'))
    action = np.zeros(16); action[6] = -4.818743705749512
    before = data.ctrl.copy()
    with pytest.raises(PositionLimitError) as caught:
        execution.apply(data, action, np.zeros(3))
    assert caught.value.record['reason'] == 'nominal_target'
    assert caught.value.record['violations'][0]['joint'] == 'hl_hipx_joint'
    np.testing.assert_array_equal(data.ctrl, before)


def test_corrected_target_is_checked_before_effort_is_returned(tmp_path, monkeypatch):
    cfg, _, _, data, execution = fixture(tmp_path)
    target = np.array(cfg['default_position_rad']); target[0] = .8
    monkeypatch.setattr(execution.adapter, 'apply', lambda *a: (np.ones(16), {'landing_leg_target': target}))
    with pytest.raises(PositionLimitError) as caught:
        execution.apply(data, np.zeros(16), np.zeros(3))
    assert caught.value.record['reason'] == 'corrected_target'
    assert execution.applied_substeps == 0


def test_feedback_gate_runs_after_physics_including_last_substep(tmp_path):
    _, _, contract, data, execution = fixture(tmp_path)
    data.qpos[contract.qpos_ids[0]] = .612
    with pytest.raises(PositionLimitError) as caught:
        execution.after_step(data)
    assert caught.value.record['reason'] == 'measured_position'


def test_xml_range_or_mapping_drift_is_rejected(tmp_path):
    cfg, model, contract, _, _ = fixture(tmp_path)
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, cfg['joint_names'][0])
    model.jnt_range[jid, 1] += .1
    with pytest.raises(ValueError, match='XML range'):
        MujocoJointLimits(cfg, model, contract, runtime.baseline)


def test_bounded_real_mujoco_steps_with_guard_enabled(tmp_path):
    _, model, contract, data, execution = fixture(tmp_path)
    # Isolate admissible-target mechanics from this synthetic fixture's falling
    # free base. This is eight physical substeps, not a locomotion benchmark.
    model.opt.gravity[:] = 0
    action = np.zeros(16)
    for _ in range(8):
        execution.check_issued(data, action)
        mujoco.mj_step1(model, data)
        execution.observe(data)
        data.ctrl[contract.actuator_ids] = execution.apply(data, action, np.zeros(3))
        mujoco.mj_step2(model, data)
        execution.after_step(data)
    assert execution.applied_substeps == 8
    assert data.time == pytest.approx(.04)


def test_idle_feedback_is_sampled_but_policy_takeover_still_rejects_bad_pose(tmp_path):
    _, _, contract, data, execution = fixture(tmp_path)
    data.qpos[contract.qpos_ids[0]] = -.438
    execution.after_idle_step(data)
    assert execution.sample_time == data.time
    with pytest.raises(PositionLimitError, match='measured_position'):
        execution.check_issued(data, np.zeros(16))
    with pytest.raises(PositionLimitError, match='measured_position'):
        execution.after_step(data)


@pytest.mark.parametrize('events,expected_mode', [
    ([[], []], 'prone'), ([['z'], []], 'standing'),
    ([['r'], []], 'damping'), ([['z'], ['x']], 'lying'),
    ([['z'], ['reset']], 'prone'),
])
def test_viewer_nonpolicy_modes_do_not_run_policy_feedback_gate(tmp_path, monkeypatch, events, expected_mode):
    # Exercise the real manual loop, real pose reset and real MuJoCo substeps.
    # Only display, keyboard IO and unused perception are replaced.
    from contextlib import nullcontext
    from unittest.mock import Mock
    _, model, contract, _, execution = fixture(tmp_path)
    b = runtime.baseline
    args = SimpleNamespace(checkpoint=tmp_path/'unused.pt', model=tmp_path/'robot.xml',
        terrain_config=SimpleNamespace(summary=lambda: {}), steps=len(events),
        output=None, snapshot=None, quiet_landing=False, print_every=100,
        stop_on_fall=False)
    viewer = SimpleNamespace(lock=nullcontext, cam=SimpleNamespace(),
                             is_running=lambda: True, sync=lambda: None)
    keys = Mock(); keys._window_keys = None; keys.take_events.side_effect = events
    terminal = Mock(); terminal.start.return_value = False
    monkeypatch.setattr(b, 'load_model', lambda *a: model)
    monkeypatch.setattr(b, 'M20Contract', lambda *a: contract)
    monkeypatch.setattr(b, 'make_noise_adapter', lambda *a: None)
    monkeypatch.setattr(b, 'make_trot_controller', lambda *a: None)
    monkeypatch.setattr(b, 'heading_controller', lambda *a: None)
    monkeypatch.setattr(b, 'noise_metadata', lambda *a: {})
    monkeypatch.setattr(b, 'TeleopKeyboard', lambda: keys)
    monkeypatch.setattr(b, 'TerminalInput', lambda *a: terminal)
    monkeypatch.setattr(b, 'launch_viewer', lambda *a, **kw: (viewer, []))
    monkeypatch.setattr(b, 'close_viewer', Mock())
    monkeypatch.setattr(b, 'draw_height_scan', Mock())
    monkeypatch.setattr(b.time, 'sleep', lambda *a: None)
    actor = lambda *a: pytest.fail('policy ran before takeover')
    result = b.interactive_rollout(args, actor=actor, execution_factory=lambda *a: execution)
    assert result['steps_executed'] == len(events)
    assert result['final_mode'] == expected_mode
    assert execution.applied_substeps == 0
    terminal.close.assert_called_once()
