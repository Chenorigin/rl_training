"""CPU unit tests only: no MuJoCo simulation, model loading or training."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source/rl_training"))
from task_support.stair_quiet_v1 import benchmark
from task_support.stair_quiet_v1.acoustics import STANDARD_PATH, read_json
from task_support.stair_quiet_v1.baselines import TEACHER_SHA256, STUDENT_SHA256
from task_support.stair_quiet_v1.compare import comparison_reasons, validate_candidate, control_reasons, read_rollout
from task_support.stair_quiet_v1.landing import LandingSettings, control_contract, contract_hash
from task_support.stair_quiet_v1.mujoco_landing import (
    MujocoLanding, map_points, pd_effort, upward_support_force,
    validate_checkpoint_control, validate_role,
)
from task_support.stair_quiet_v1.stress import StressChain, canonical_hash, load_stress


def base_summary():
    settings = LandingSettings(mode="off")
    control = control_contract(settings)
    return dict(
        role="baseline-teacher", completed=True, window_valid=True, fell=False,
        ever_outside_course=False, measurement_duration_s=8., max_stationary_interval_s=.1,
        optimizer_updates=0, weights_updated=False, compiled_model_sha256="a" * 64,
        command=[.5, 0, 0], physics_dt_s=.005, policy_dt_s=.02, seed=42,
        event_extraction={}, traversal_settings={}, source_sha256={
            "controller": "a",
            "source/rl_training/task_support/stair_quiet_v1/landing.py": control["source_sha256"],
            "source/rl_training/task_support/stair_quiet_v1/terrain_memory.py": control["memory_source_sha256"]},
        landing_control="off", control_contract=control,
        control_contract_sha256=contract_hash(control),
        stress=StressChain(load_stress("nominal"), .005, 42).summary())


def controlled_summary():
    summary = base_summary()
    control = control_contract(LandingSettings(mode="combined"))
    summary.update(role="control-teacher", landing_control="combined",
                   control_contract=control, control_contract_sha256=contract_hash(control),
                   checkpoint_sha256=TEACHER_SHA256, pinned_actor_identity_checked=True,
                   candidate_kind="baseline_actor_plus_controller",
                   controller_execution=dict(enabled=True, wheel_samples=100, modified_wheel_samples=10))
    return summary


def test_explicit_roles_cannot_relabel_baseline_or_silent_off():
    validate_role("baseline-teacher", "off")
    validate_role("control-student", "phase")
    for role, mode in (("baseline-teacher", "phase"), ("control-student", "off"),
                       ("candidate-student", "unknown"), ("another-role", "off")):
        with pytest.raises(ValueError):
            validate_role(role, mode)
    expected = [{"sha256": TEACHER_SHA256}, {"sha256": STUDENT_SHA256}]
    good = controlled_summary()
    validate_candidate(good, expected)
    for key, value in (("candidate_kind", "trained_actor"), ("checkpoint_sha256", STUDENT_SHA256),
                       ("landing_control", "off"), ("pinned_actor_identity_checked", False)):
        with pytest.raises(ValueError):
            validate_candidate({**good, key: value}, expected)
    with pytest.raises(ValueError):
        validate_candidate({**good, "role": "candidate-teacher"}, expected)


def test_cli_default_is_off_and_control_requires_explicit_mode(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(benchmark, "run", lambda args: seen.append(args))
    flags = ["--role", "baseline-teacher", "--model", "unused.xml", "--output", str(tmp_path / "new")]
    benchmark.main(flags)
    assert seen[0].landing_control == "off"
    assert seen[0].stress_profile == "nominal"
    with pytest.raises(SystemExit):
        benchmark.main(flags + ["--landing-control", "combined"])
    flags[1] = "control-teacher"
    with pytest.raises(SystemExit):
        benchmark.main(flags)
    benchmark.main(flags + ["--landing-control", "combined"])
    assert seen[-1].role == "control-teacher"


def test_saved_contract_cannot_be_silently_disabled_or_changed():
    settings = LandingSettings(mode="combined")
    document = control_contract(settings)
    saved = {"m20_training_manifest": {"noise_refinement": {
        "contract": {"environment": {"landing_control": asdict(settings)}},
        "control_contract": document}}}
    assert validate_checkpoint_control(saved, settings, document)
    with pytest.raises(ValueError):
        validate_checkpoint_control(saved, LandingSettings(mode="off"),
                                    control_contract(LandingSettings(mode="off")))
    changed = {**document, "source_sha256": "0" * 64}
    with pytest.raises(ValueError):
        validate_checkpoint_control(saved, settings, changed)
    assert not validate_checkpoint_control({}, settings, document)


def test_self_consistent_contract_cannot_disagree_with_recorded_sources():
    bad = controlled_summary()
    bad["control_contract"]["source_sha256"] = "0" * 64
    bad["control_contract_sha256"] = contract_hash(bad["control_contract"])
    assert any("landing.py" in reason for reason in control_reasons(bad))
    bad = controlled_summary()
    bad["control_contract"]["memory_source_sha256"] = "0" * 64
    bad["control_contract_sha256"] = contract_hash(bad["control_contract"])
    assert any("terrain_memory.py" in reason for reason in control_reasons(bad))


def test_same_profile_and_actual_execution_required_for_comparison():
    baseline, candidate = base_summary(), controlled_summary()
    standard = read_json(STANDARD_PATH)
    assert comparison_reasons(baseline, candidate, standard) == []
    for profile in ("terrain_high", "contact_delay_loss", "actuator_chain", "combined"):
        c = deepcopy(candidate)
        c["stress"] = StressChain(load_stress(profile), .005, 42).summary()
        assert comparison_reasons(baseline, c, standard)
        b = deepcopy(baseline)
        b["stress"] = deepcopy(c["stress"])
        assert comparison_reasons(b, c, standard) == []
    for mutation in (
            {"controller_execution": dict(enabled=True, wheel_samples=100, modified_wheel_samples=0)},
            {"control_contract_sha256": "0" * 64},
            {"measurement_duration_s": 12.},
            {"max_stationary_interval_s": 1.},
            {"completed": False}):
        assert comparison_reasons(baseline, {**candidate, **mutation}, standard)
    bad = deepcopy(candidate)
    bad["stress"]["profile"]["terrain_height_bias_m"] = 2.
    assert comparison_reasons(baseline, bad, standard)
    assert control_reasons({**candidate, "landing_control": "off"})


def test_stress_is_deterministic_causal_and_nominal_exact():
    x = np.linspace(-76.4, 76.4, 16)
    nominal = StressChain(load_stress("nominal"), .005, 10)
    np.testing.assert_array_equal(nominal.effort(x), x)
    np.testing.assert_array_equal(nominal.effort(-x), -x)
    a = StressChain(load_stress("combined"), .005, 10)
    b = StressChain(load_stress("combined"), .005, 10)
    seen_valid = []
    for step in range(20):
        fa, va = a.contact(np.full(4, step), np.ones(4, bool))
        fb, vb = b.contact(np.full(4, step), np.ones(4, bool))
        np.testing.assert_array_equal(fa, fb)
        np.testing.assert_array_equal(va, vb)
        seen_valid.append(va.copy())
        np.testing.assert_array_equal(a.effort(x), b.effort(x))
        if step < 3:
            assert not va.any()
        elif va.any():
            assert np.all(fa[va] == step - 3)
    assert np.any(seen_valid[3:]) and not np.all(seen_valid[3:])
    assert a.summary()["contact_total_age_s"] == pytest.approx(.02)
    assert a.summary()["profile_sha256"] == canonical_hash(load_stress("combined").document())
    delayed = StressChain(load_stress("actuator_chain"), .005, 1)
    assert np.all(delayed.effort(x) == 0)
    assert np.all(delayed.effort(x) == 0)
    first = delayed.effort(x)
    assert np.all(np.abs(first) < np.abs(x))


def test_support_force_excludes_risers_and_wrong_side_normals():
    assert upward_support_force([0, 0, -1], 20, True) == 20
    assert upward_support_force([0, 0, 1], 20, False) == 20
    assert upward_support_force([1, 0, 0], 20, True) == 0
    assert upward_support_force([0, 0, 1], 20, True) == 0
    assert upward_support_force([0, 0, -1], 20, False) == 0
    assert upward_support_force([0, 0, 1], -5, False) == 0


def test_nominal_pd_is_original_expression_including_limits():
    rng = np.random.default_rng(9)
    q, dq, target, wheel = [rng.normal(size=n) for n in (16, 16, 12, 4)]
    constants = SimpleNamespace(LEG_EFFORT_LIMIT=76.4, WHEEL_EFFORT_LIMIT=21.6)
    result = pd_effort(q, dq, target.reshape(4, 3), wheel, np.ones(4), np.ones(4), constants)
    expected = np.r_[np.clip(80. * (target - q[:12]) - 2. * dq[:12], -76.4, 76.4),
                     np.clip(.6 * (wheel - dq[12:]), -21.6, 21.6)]
    np.testing.assert_array_equal(result, expected)


def fake_contract_and_data():
    grid = np.array([[.1, 0.], [.2, 0.], [.3, 0.], [.4, 0.]])
    c = SimpleNamespace(base_id=0, grid=grid, qpos_ids=np.arange(16), qvel_ids=np.arange(16),
                        wheel_body_ids=np.arange(1, 5))
    rotation = np.array([[0, -1., 0], [1., 0, 0], [0, 0, 1.]])
    d = SimpleNamespace(xpos=np.array([[1., 2., .8]]), xmat=rotation.reshape(1, 9))
    return c, d


def test_student_uses_delivered_map_holes_and_saturation_not_oracle():
    c, d = fake_contract_and_data()
    # There deliberately is no last_scan_hits or ground_height on the contract.
    adapter = SimpleNamespace(last_channels=(
        np.array([.1, .2, 1., .3]), np.array([1., 0., 1., 1.]), np.array([1., 1., 1., 0.])))
    points, valid = map_points(c, d, adapter)
    np.testing.assert_allclose(points[:, :2], np.c_[np.ones(4), [2.1, 2.2, 2.3, 2.4]])
    np.testing.assert_allclose(points[:, 2], [.2, .1, -.7, 0.], atol=1e-15)
    np.testing.assert_array_equal(valid, [True, False, False, False])
    adapter.last_channels = None
    assert len(map_points(c, d, adapter)[0]) == 0


def test_real_adapter_calls_shared_core_without_physics_and_bypasses_missing_map(monkeypatch):
    # Only mj_jac is faked; no MuJoCo model or simulation is constructed.
    def jac(_model, _data, out, _rot, point, body):
        out[:] = 0.
        out[:, 3 * (body - 1):3 * body] = np.eye(3) * .2

    monkeypatch.setitem(sys.modules, "mujoco",
                        SimpleNamespace(mjtGeom=SimpleNamespace(mjGEOM_CYLINDER=5), mj_jac=jac))
    model = SimpleNamespace(ngeom=4, geom_bodyid=np.arange(1, 5), geom_contype=np.ones(4),
                            geom_conaffinity=np.ones(4), geom_type=np.full(4, 5),
                            geom_size=np.tile([.09, .025, 0.], (4, 1)), nv=16)
    c, data = fake_contract_and_data()
    wheel_rotation = np.array([[1., 0, 0], [0, 0, 1.], [0, -1., 0]])
    data.geom_xmat = np.tile(wheel_rotation.reshape(1, 9), (4, 1))
    data.geom_xpos = np.c_[np.arange(4), np.zeros(4), np.full(4, .3)]
    data.qpos = np.linspace(-.1, .2, 16)
    data.qvel = np.zeros(16)
    data.time = 0.
    constants = SimpleNamespace(DEFAULT_JOINT_POS=np.linspace(-.3, .3, 16),
                                ACTION_SCALE=np.ones(16) * .125,
                                LEG_EFFORT_LIMIT=76.4, WHEEL_EFFORT_LIMIT=21.6)
    adapter = MujocoLanding(model, c, LandingSettings(mode="combined"), constants)
    student = SimpleNamespace(last_channels=(np.zeros(4), np.zeros(4), np.zeros(4)),
                              map=SimpleNamespace(cell_age=torch.zeros(1, 4)))
    adapter.observe(data, student)
    action = np.linspace(-.2, .3, 16)
    effort, info = adapter.apply(data, action, np.zeros(4), np.zeros(4, bool), True, .005, .005)
    nominal_target = constants.DEFAULT_JOINT_POS[:12] + action[:12] * constants.ACTION_SCALE[:12]
    expected = pd_effort(data.qpos, data.qvel, nominal_target.reshape(4, 3),
                         action[12:] * constants.ACTION_SCALE[12:], np.ones(4), np.ones(4), constants)
    np.testing.assert_array_equal(effort, expected)
    assert not info["landing_modified"].any()
    assert not info["landing_terrain_valid"].any()
    assert not info["landing_contact_valid"].any()
    assert adapter.control.summary()["wheel_samples"] == 4
    assert adapter.contract_sha256 == contract_hash(control_contract(adapter.settings))


def test_student_map_age_is_frozen_then_expires(monkeypatch):
    # Reuse a lightweight fake model; inspect validity via the actual adapter.
    def jac(_m, _d, out, _r, _point, body):
        out[:] = 0
        out[:, 3 * (body - 1):3 * body] = np.eye(3) * .2
    monkeypatch.setitem(sys.modules, "mujoco",
                        SimpleNamespace(mjtGeom=SimpleNamespace(mjGEOM_CYLINDER=5), mj_jac=jac))
    c, data = fake_contract_and_data()
    model = SimpleNamespace(ngeom=4, geom_bodyid=np.arange(1, 5), geom_contype=np.ones(4),
                            geom_conaffinity=np.ones(4), geom_type=np.full(4, 5),
                            geom_size=np.tile([.09, .025, 0.], (4, 1)), nv=16)
    data.geom_xmat = np.tile(np.array([[1., 0, 0], [0, 0, 1.], [0, -1., 0]]).reshape(1, 9), (4, 1))
    data.geom_xpos = np.c_[np.ones(4), [2.1, 2.2, 2.3, 2.4], np.full(4, .4)]
    data.qpos, data.qvel, data.time = np.zeros(16), np.zeros(16), 0.
    const = SimpleNamespace(DEFAULT_JOINT_POS=np.zeros(16), ACTION_SCALE=np.ones(16),
                            LEG_EFFORT_LIMIT=76.4, WHEEL_EFFORT_LIMIT=21.6)
    control = MujocoLanding(model, c, LandingSettings(
        mode="combined", terrain_max_age_s=.10, terrain_memory_s=.10), const)
    student = SimpleNamespace(last_channels=(np.zeros(4), np.ones(4), np.ones(4)),
                              map=SimpleNamespace(cell_age=torch.full((1, 4), .09)))
    control.observe(data, student)
    cached = control.points.copy()
    _, now = control.apply(data, np.zeros(16), np.zeros(4), np.ones(4, bool), True, .005, .005)
    assert now["landing_terrain_valid"].all()
    data.time = .02
    data.xpos[0, 0] += .2
    _, later = control.apply(data, np.zeros(16), np.zeros(4), np.ones(4, bool), True, .005, .005)
    assert not later["landing_terrain_valid"].any()
    np.testing.assert_array_equal(control.points, cached)


def write_controlled_rollout(folder):
    folder.mkdir()
    summary = controlled_summary()
    summary.update(schema="stair_quiet_v1.mujoco_impact_rollout.v1",
                   measurement_start_s=0., measurement_end_s=.01, duration_s=.01,
                   measurement_duration_s=.01)
    summary["controller_execution"].update(wheel_samples=8, modified_wheel_samples=1)
    events = {"duration_s": .01, "measurement_window_s": [0., .01], "events": [],
              "provenance": {"checkpoint_sha256": summary["checkpoint_sha256"],
                             "control_contract_sha256": summary["control_contract_sha256"],
                             "stress_profile_sha256": summary["stress"]["profile_sha256"]}}
    benchmark.write_json(folder / "summary.json", summary)
    benchmark.write_json(folder / "events.json", events)
    arrays = dict(time_s=np.array([0., .005]), landing_leg_target=np.zeros((2, 4, 3)),
                  landing_kp_scale=np.ones((2, 4)), landing_kd_scale=np.ones((2, 4)),
                  landing_phase=np.zeros((2, 4), dtype=int),
                  landing_modified=np.array([[True, False, False, False], [False] * 4]),
                  landing_invalid=np.zeros((2, 4), bool), landing_contact_valid=np.ones((2, 4), bool),
                  effort_requested_Nm=np.zeros((2, 16)), effort_applied_Nm=np.zeros((2, 16)))
    np.savez(folder / "contacts.npz", **arrays)
    np.savez(folder / "contact_points.npz", values=np.zeros((0, 11)))
    return arrays


def test_raw_controller_evidence_requires_actual_shapes_masks_and_counts(tmp_path):
    folder = tmp_path / "recorded"
    arrays = write_controlled_rollout(folder)
    assert read_rollout(folder)["summary"]["role"] == "control-teacher"
    for key, value in (
            ("landing_modified", np.zeros((2, 4), bool)),
            ("landing_modified", arrays["landing_modified"].astype(float)),
            ("landing_leg_target", np.zeros((2, 12))),
            ("landing_phase", np.full((2, 4), 8)),
            ("time_s", np.array([0., .006])),
            ("landing_kp_scale", np.full((2, 4), np.nan))):
        np.savez(folder / "contacts.npz", **{**arrays, key: value})
        with pytest.raises(ValueError):
            read_rollout(folder)
    np.savez(folder / "contacts.npz", **{k: v for k, v in arrays.items() if k != "landing_phase"})
    with pytest.raises(ValueError):
        read_rollout(folder)
