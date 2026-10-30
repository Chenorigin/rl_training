"""CPU metric/protocol checks; no MuJoCo, checkpoint, optimizer or hardware."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1]/"scripts/tools/evaluate_cts_motion_stability.py"
spec = importlib.util.spec_from_file_location("motion_stability_evaluation", PATH)
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def row(t=0., phase="handover"):
    return dict(time_s=t, phase=phase, rpy=np.zeros(3), velocity_body=np.zeros(3),
                omega_body=np.zeros(3), position=np.zeros(3), target_q=np.zeros(12),
                wheel_target=np.zeros(4), joint_margin=np.ones(12), target_margin=np.ones(12),
                effort=np.zeros(16), effort_limit=np.ones(16), requested=np.zeros(3),
                effective_command=np.zeros(3), support_z=np.zeros(4), fell=False,
                handoff_target_q=np.zeros(12))


def test_handover_window_and_increment_include_the_standing_target():
    rows = [row(0), row(1), row(2), row(2.1, "left")]
    rows[0]["target_q"][0] = .3
    rows[1]["target_q"][0] = .4
    rows[2]["target_q"][0] = .5
    rows[3]["rpy"][0] = 2.
    rows[1]["rpy"][0] = .2
    rows[2]["omega_body"][1] = .7
    result = evaluation.summarize(rows, policy_dt=.02, completed=True)
    early = result["handover_first_2s"]
    assert early["peak_nominal_leg_target_step_rad"] == pytest.approx(.3)
    assert early["peak_abs_roll_pitch_rad"] == pytest.approx([.2, 0])
    assert early["peak_abs_body_angular_rate_rad_s"] == pytest.approx([0, .7, 0])
    assert result["status"] == "COMPLETED"
    assert result["hardware_validation"] == "NOT_RUN"


def test_phase_displacement_is_in_start_heading_and_unwraps_yaw():
    rows = [row(0, "left"), row(1, "left")]
    rows[0]["rpy"][2] = np.pi/2
    rows[1]["rpy"][2] = np.pi/2
    rows[1]["position"][:2] = [-.5, .1]
    result = evaluation.summarize(rows, policy_dt=.02, completed=True)
    assert result["phases"]["left"]["displacement_in_phase_start_heading_m"] == pytest.approx([.1, .5])
    rows[0]["rpy"][2], rows[1]["rpy"][2] = 3.1, -3.1
    result = evaluation.summarize(rows, policy_dt=.02, completed=True)
    assert result["phases"]["left"]["yaw_change_rad"] == pytest.approx(2*np.pi-6.2)


def test_physical_failures_are_separate_from_nominal_target_and_effort_clip():
    rows = [row()]
    rows[0]["target_margin"][0] = -.2
    rows[0]["effort"][0] = 1.
    result = evaluation.summarize(rows, policy_dt=.02, completed=True)
    assert result["nominal_target_outside_mjcf_range"]
    assert result["status"] == "FAIL"  # SDK rejects the target before publication
    assert result["actuator_effort_at_total_clip_fraction"] == pytest.approx(1/16)
    rows[0]["joint_margin"][0] = -.001
    assert evaluation.summarize(rows, policy_dt=.02, completed=True)["status"] == "FAIL"
    rows[0]["joint_margin"][0] = 1.
    rows[0]["fell"] = True
    assert evaluation.summarize(rows, policy_dt=.02, completed=True)["status"] == "FAIL"
    rows[0]["position"][0] = np.nan
    assert evaluation.summarize(rows, policy_dt=.02, completed=True)["reason"] == "nonfinite physical telemetry"


def test_missing_policy_is_not_success_and_failure_preserves_reason():
    assert evaluation.summarize([], policy_dt=.02, completed=False)["status"] == "NOT_RUN"
    assert evaluation.summarize([], policy_dt=.02, completed=False, fault="loader mismatch")["status"] == "FAIL"
    assert evaluation.summarize([row()], policy_dt=.02, completed=False)["status"] == "INCOMPLETE"


def test_telemetry_samples_and_period_are_separate_from_policy_period():
    rows = [row(tick*.005) for tick in range(8)]
    result = evaluation.summarize(rows, policy_dt=.02, completed=True)
    assert result["telemetry_samples"] == 8
    assert result["telemetry_dt_s"] == pytest.approx(.005)
    assert result["policy_dt_s"] == .02
    assert "policy_samples" not in result
    assert evaluation.summarize([row()], policy_dt=.02, completed=False)["telemetry_dt_s"] is None


def test_protocol_has_start_stop_left_right_and_yaw_add_withdraw_reverse():
    speed = (.4, .25, .3)
    for scenario in evaluation.SCENARIOS:
        np.testing.assert_array_equal(evaluation.protocol(scenario, 1.99, speed)[2], 0.)
    np.testing.assert_array_equal(evaluation.protocol("left", 3, speed)[2], [0, .25, 0])
    np.testing.assert_array_equal(evaluation.protocol("right", 3, speed)[2], [0, -.25, 0])
    expected = [(3, "forward", .0), (5, "add_yaw", .3),
                (7, "withdraw_yaw", .0), (9, "reverse_yaw", -.3)]
    for t, phase, yaw in expected:
        actual, _, command = evaluation.protocol("forward_yaw", t, speed)
        assert actual == phase
        np.testing.assert_array_equal(command, [.4, 0, yaw])
    assert evaluation.protocol("forward_yaw", 11, speed)[0] == "stop"


def test_scripted_input_queues_z_c_once_and_uses_existing_keyboard_when_selected():
    calls = []
    window = SimpleNamespace(held=set())
    keyboard = SimpleNamespace(velocity_command=lambda: calls.append(window.held.copy()) or np.array([.01, .02, .03]),
                               clear_motion=lambda: calls.append("clear"), close=lambda: None,
                               on_keycode=lambda k: None)
    keys = evaluation.ScriptedInput(keyboard, window, "left", .02, (.4, .25, .3), "keyboard")
    assert keys.take_events() == ["z", "c"]
    assert keys.take_events() == []
    keys.calls = 100
    np.testing.assert_allclose(keys.velocity_command(), [.01, .02, .03])
    assert calls == [{"a"}]
    keys.clear_motion()
    assert calls[-1] == "clear"
    keys.mode = "step"
    np.testing.assert_allclose(keys.velocity_command(), [0, .25, 0])
    assert calls[-1] == "clear"  # step mode deliberately bypasses only keyboard ramp


def test_execution_proxy_calls_real_controller_and_after_step_before_sampling():
    calls = []
    delegate = SimpleNamespace(reset=lambda: calls.append("reset"),
                               apply=lambda data, action, command: calls.append("apply") or action*2,
                               after_step=lambda data: calls.append("after"))
    recorder = SimpleNamespace(action=None, sample=lambda data, effort: calls.append(("sample", effort[0])))
    proxy = evaluation.RecordedExecution(delegate, recorder)
    proxy.reset()
    action = np.ones(16)
    np.testing.assert_array_equal(proxy.apply(None, action, np.zeros(3)), action*2)
    proxy.after_step(None)
    assert calls == ["reset", "apply", "after", ("sample", 2.)]
    np.testing.assert_array_equal(recorder.action, action)


@pytest.mark.parametrize("extra", [["--policy-steps", "0"], ["--startup-steps", "5001"],
                                  ["--speed-x", "nan"], ["--yaw-rate", ".7"],
                                  ["--spawn-mat-size", "1", "0", ".03"]])
def test_cli_rejects_unbounded_or_invalid_inputs(extra):
    with pytest.raises(SystemExit):
        evaluation.parse_args(["--checkpoint", "a.pt", "--model", "a.xml", "--output", "out", *extra])


def test_mat_is_explicitly_not_run_without_geometry(tmp_path):
    args = evaluation.parse_args(["--checkpoint", "a.pt", "--model", "a.xml", "--output", str(tmp_path)])
    result = evaluation.run_case(args, None, None, None, None, "handover", "mat", tmp_path/"missing_mat")
    assert result["status"] == "NOT_RUN"
    assert "--spawn-mat-xy" in result["reason"]
    assert (tmp_path/"missing_mat/result.json").is_file()
