"""Independent invariants and counterexamples for the shared landing controller."""
from dataclasses import replace

import pytest
import torch

from task_support.stair_quiet_v1.landing import (
    LandingController, LandingSettings, contract_hash, control_contract, sample_tread_surface,
)


def inputs():
    j = torch.tensor([[[0., 0.2, 0.1]]]).repeat(1, 4, 1)
    return dict(
        nominal_leg_target=torch.tensor([[[0., -0.15, -0.075]]]).repeat(1, 4, 1),
        nominal_wheel_velocity=torch.full((1, 4), 4.),
        joint_position=torch.zeros(1, 4, 3), joint_velocity=torch.zeros(1, 4, 3),
        normal_jacobian=j, normal_velocity=torch.tensor([[-0.6, 0., 0., 0.]]),
        clearance=torch.tensor([[0.035, 0., 0., 0.]]),
        terrain_valid=torch.ones(1, 4, dtype=torch.bool),
        contact_force=torch.tensor([[0., 100., 100., 100.]]),
        contact_valid=torch.ones(1, 4, dtype=torch.bool),
        command_active=torch.tensor([True]), dt=0.005,
    )


def run(controller, values, count=40):
    for _ in range(count):
        result = controller.step(**values)
    return result


def test_off_is_exact_nominal_including_gains_and_wheel_speeds():
    x = inputs()
    c = LandingController(1, "cpu", LandingSettings(mode="off"))
    y = run(c, x)
    assert torch.equal(y["leg_target"], x["nominal_leg_target"])
    assert torch.equal(y["wheel_velocity"], x["nominal_wheel_velocity"])
    assert torch.equal(y["kp_scale"], torch.ones(1, 4))
    assert not y["modified"].any()


def test_precontact_reduces_inward_reference_and_preserves_tangent():
    x = inputs()
    # Whole-body velocity includes the moving joint contribution here.
    x["joint_velocity"][0, 0] = torch.tensor([0., -2.4, -1.2])
    c = LandingController(1, "cpu", LandingSettings(mode="precontact"))
    y = run(c, x)
    correction = y["leg_target"] - x["nominal_leg_target"]
    normal_change = (correction * x["normal_jacobian"]).sum(-1)
    assert normal_change[0, 0] > 0
    assert torch.allclose(correction[0, 0, 1] / correction[0, 0, 2], torch.tensor(2.))
    assert correction[0, 0, 0] == 0  # No arbitrary lateral correction.
    assert not y["modified"][0, 1:].any()  # Do not soften the support legs.
    assert correction.abs().max() <= c.settings.max_joint_correction_rad + 1e-7
    assert torch.equal(y["kp_scale"], torch.ones(1, 4))


def test_approach_softens_then_recovers_after_real_contact():
    x = inputs()
    c = LandingController(1, "cpu")
    y = run(c, x)
    assert c.settings.landing_kp_scale <= y["kp_scale"][0, 0] < 1.
    assert torch.equal(y["kp_scale"][0, 1:], torch.ones(3))
    x["contact_force"][0, 0] = 100
    x["normal_velocity"][0, 0] = 0
    y = run(c, x, 3)
    assert y["phase"][0, 0] in (2, 5)  # Limits are reported, not silently claimed feasible.
    y = run(c, x, 80)
    assert y["phase"][0, 0] == 3
    assert torch.allclose(y["kp_scale"], torch.ones(1, 4))
    assert torch.allclose(y["correction"], torch.zeros(1, 4, 3))


@pytest.mark.parametrize("kind", ["unknown_map", "unknown_contact", "nan_clearance", "infinite_jacobian"])
def test_unknown_observation_never_creates_a_control_correction(kind):
    x = inputs()
    if kind == "unknown_map":
        x["terrain_valid"][0, 0] = False
    elif kind == "unknown_contact":
        x["contact_valid"][0, 0] = False
    elif kind == "nan_clearance":
        x["clearance"][0, 0] = float("nan")
    else:
        x["normal_jacobian"][0, 0] = float("inf")
    c = LandingController(1, "cpu")
    y = run(c, x)
    assert y["invalid"][0, 0]
    assert torch.isfinite(y["leg_target"]).all()
    assert torch.equal(y["leg_target"], x["nominal_leg_target"])
    assert c.summary()["invalid_samples"] > 0


def test_missing_contact_is_not_airtime_evidence():
    x = inputs()
    x["contact_valid"].fill_(False)
    c = LandingController(1, "cpu")
    run(c, x)
    x["contact_valid"].fill_(True)
    x["contact_force"].fill_(100.)
    y = run(c, x, 3)
    assert c.summary()["touchdown_events"] == 0
    assert not (y["phase"] == 2).any()


@pytest.mark.parametrize("kind", ["one_support", "singular", "zero_command"])
def test_controller_does_not_spend_unavailable_support_authority(kind):
    x = inputs()
    if kind == "one_support":
        x["contact_force"][0, 2:] = 0
    elif kind == "singular":
        x["normal_jacobian"][0, 0] = 0
    else:
        x["command_active"].fill_(False)
    y = run(LandingController(1, "cpu"), x)
    assert torch.equal(y["leg_target"], x["nominal_leg_target"])
    assert torch.equal(y["kp_scale"], torch.ones(1, 4))
    if kind != "zero_command":
        assert y["infeasible"][0, 0]


def test_target_and_gain_change_rate_and_reset():
    x = inputs()
    c = LandingController(1, "cpu")
    previous = torch.zeros(1, 4, 3)
    prev_kp = torch.ones(1, 4)
    for _ in range(50):
        y = c.step(**x)
        assert (y["correction"] - previous).abs().max() <= c.settings.correction_rate_rad_s*x["dt"] + 1e-7
        assert (y["kp_scale"] - prev_kp).abs().max() <= c.settings.gain_rate_per_s*x["dt"] + 1e-7
        previous, prev_kp = y["correction"], y["kp_scale"]
    c.reset(torch.tensor([0]))
    y = c.step(**x)
    assert not y["modified"].any()
    assert c.elapsed.item() == pytest.approx(x["dt"])


def test_nonfinite_nominal_command_fails_instead_of_sending_it():
    x = inputs()
    x["nominal_leg_target"][0, 0, 1] = float("nan")
    with pytest.raises(ValueError, match="Non-finite nominal"):
        LandingController(1, "cpu").step(**x)


def test_surface_sampler_rejects_unknown_far_and_discontinuous_patches():
    xy = torch.zeros(1, 4, 2)
    points = torch.tensor([[[0., 0., 0.17], [0.05, 0., 0.17]]])
    h, ok = sample_tread_surface(xy, points)
    assert ok.all() and torch.allclose(h, torch.full((1, 4), .17))
    points[0, 1, 2] = 0.
    h, ok = sample_tread_surface(xy, points)
    assert not ok.any() and torch.isnan(h).all()
    points[:, :, :2] += 2
    h, ok = sample_tread_surface(xy, points)
    assert not ok.any()
    points.fill_(float("nan"))
    h, ok = sample_tread_surface(xy, points)
    assert not ok.any() and torch.isnan(h).all()


def test_latency_budget_changes_allowable_approach_speed_in_expected_direction():
    x = inputs()
    a = run(LandingController(1, "cpu", LandingSettings(latency_budget_s=0.)), x)
    b = run(LandingController(1, "cpu", LandingSettings(latency_budget_s=.08)), x)
    assert b["safe_speed"][0, 0] < a["safe_speed"][0, 0]


def test_full_jacobian_avoids_tangential_reference_motion():
    x = inputs()
    matrix = torch.tensor([[.2, .05, 0.], [0., .3, .04], [0., .2, .1]])
    x["point_jacobian"] = matrix[None, None].repeat(1, 4, 1, 1)
    x["surface_normal"] = torch.tensor([[[0., 0., 1.]]]).repeat(1, 4, 1)
    y = run(LandingController(1, "cpu", LandingSettings(mode="precontact")), x)
    cartesian = matrix @ y["correction"][0, 0]
    assert cartesian[2] > 0.
    assert cartesian[:2].abs().max() < cartesian[2] * .03


def test_contract_tracks_settings_and_source_and_preserves_unverified_acoustics():
    s = LandingSettings()
    a = control_contract(s)
    b = control_contract(replace(s, landing_kp_scale=.9))
    assert a["actual_acoustic_pass"] is False
    assert len(a["source_sha256"]) == 64
    assert contract_hash(a) != contract_hash(b)
