"""Safety instruments must distinguish actual dangerous/benign trajectories."""
from pathlib import Path
import sys
from copy import deepcopy
from types import SimpleNamespace
import torch
import pytest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/rl_training"))
from task_support.stair_quiet_v1_width.settings import WidthSettings, GEOMETRIES, PROFILES
from task_support.stair_quiet_v1_width.geometry import (
    stair_mesh, foothold_regions, capsule_contact_velocity, face_profile_ids)
from task_support.stair_quiet_v1_width.signals import TailTracker, failure_mask, failure_reward, tail_cost
from task_support.stair_quiet_v1_width.runner import validate_warm_start, DEFAULT_START, DEFAULT_START_SHA256
from task_support.stair_quiet_v1.baselines import TEACHER, TEACHER_SHA256, checked_path
from task_support.idle_stop_motion_v1.runner import file_sha256

torch.set_num_threads(1)


def sample(n=1, x=0., y=.25, force=100., speed=0., spin=None):
    pos = torch.tensor([[[x, y, .09], [x, -y, .09], [x, y, .09], [x, -y, .09]]]).repeat(n, 1, 1)
    axis = torch.zeros_like(pos)
    axis[..., 1] = 1.
    vel = torch.zeros_like(pos)
    vel[..., 0] = speed
    omega = torch.zeros_like(pos)
    omega[..., 1] = speed/.09 if spin is None else spin
    f = torch.zeros_like(pos)
    f[..., 2] = force
    return dict(position=pos, axis=axis, velocity=vel, omega=omega, forces=f,
                root=torch.zeros(n, 3), root_velocity=torch.zeros(n, 3), yaw=torch.zeros(n),
                upright=torch.ones(n), height=torch.full((n,), .15), depth=torch.full((n,), .2),
                width=torch.ones(n), steps=torch.full((n,), 6), dt=.005)


def test_reference_short_tread_region_and_side_margin():
    s = WidthSettings()
    data = sample(x=1.10)
    x, y, _ = foothold_regions(data["position"], data["axis"], data["height"], data["depth"],
                              data["width"], data["steps"], s)
    assert torch.all(x > .049)  # 20 cm tread has a nonempty usable centre region
    assert torch.all(y > 0)
    data["position"][..., 0] = 1.02
    data["position"][..., 1] = .36
    x, y, _ = foothold_regions(data["position"], data["axis"], data["height"], data["depth"],
                              data["width"], data["steps"], s)
    assert torch.all(x < 0) and torch.all(y < 0)


def test_front_and_hind_foothold_safety_are_independent_and_both_enforced():
    s = WidthSettings(rear_tread_margin=.06, rear_lateral_margin=.08)
    d = sample(x=1.075, y=.32)
    d["position"][..., 2] += .15
    forward, side, _ = foothold_regions(d["position"], d["axis"], d["height"], d["depth"],
                                        d["width"], d["steps"], s)
    assert (forward[:, :2] > 0).all() and (side[:, :2] > 0).all()
    assert (forward[:, 2:] < 0).all() and (side[:, 2:] < 0).all()
    t = TailTracker(1, "cpu", s)
    for _ in range(30):
        t.update(**d)
    assert t.cost["foothold"].item() > 0 and t.unsafe_seconds.item() > 0
    # Swap only front limits: front feet must fail while rear feet stay safe.
    s.front_tread_margin, s.rear_tread_margin = .06, .03
    s.front_lateral_margin, s.rear_lateral_margin = .08, .04
    forward, side, _ = foothold_regions(d["position"], d["axis"], d["height"], d["depth"],
                                        d["width"], d["steps"], s)
    assert (forward[:, :2] < 0).all() and (side[:, :2] < 0).all()
    assert (forward[:, 2:] > 0).all() and (side[:, 2:] > 0).all()


def test_rolling_is_not_slip_but_loaded_spinning_is():
    good = sample(speed=.5)
    v = capsule_contact_velocity(good["velocity"], good["omega"], good["axis"])
    assert v.abs().max() < 1e-6
    a, b = TailTracker(1, "cpu", WidthSettings()), TailTracker(1, "cpu", WidthSettings())
    for _ in range(20):
        a.update(**good)
        b.update(**sample(speed=0., spin=10.))
    assert a.cost["slip"].item() == 0 and not a.recovering.item()
    assert b.cost["slip"].item() > 0 and b.recovering.item()
    airborne = TailTracker(1, "cpu", WidthSettings())
    for _ in range(20):
        airborne.update(**sample(speed=0., spin=10., force=0.))
    assert airborne.cost["slip"].item() == 0


def test_side_escape_latches_failure_even_after_return():
    t = TailTracker(1, "cpu", WidthSettings())
    t.update(**sample(y=.25))
    assert not t.failed().item()
    t.update(**sample(y=.4))
    assert t.edge_failed.item() and t.cost["edge"].item() > 0
    t.update(**sample(y=.25))
    assert t.failed().item()


def test_tread_transition_grace_does_not_allow_indefinite_edge_loading():
    t = TailTracker(1, "cpu", WidthSettings())
    d = sample(x=1.02)
    d["position"][..., 2] += .15
    for _ in range(8):
        t.update(**d)
    assert t.cost["foothold"].item() == 0
    for _ in range(20):
        t.update(**d)
    assert t.cost["foothold"].item() > 0


def test_front_or_hind_wheel_perched_behind_next_lip_cannot_evade_support_check():
    for foot in range(4):
        d = sample()
        d["position"][0, foot, 0] = .98  # Centre behind first riser x=1.0.
        d["position"][0, foot, 2] += .15  # But vertical load is on its upper lip.
        t = TailTracker(1, "cpu", WidthSettings())
        for _ in range(30):
            t.update(**d)
        assert t.last["supported"][0, foot]
        assert t.last["forward_margin"][0, foot] < 0
        assert t.cost["foothold"].item() > 0
        assert t.unsafe_foot_seconds[0, foot] > 0


def test_tail_no_saturation_no_plateau_double_charge_and_reset_isolation():
    t = TailTracker(2, "cpu", WidthSettings())
    d = sample(n=2, force=1000.)
    t.update(**d)
    first = t.cost["tail_force"].clone()
    for _ in range(8):
        t.update(**d)
    assert torch.equal(first, t.cost["tail_force"])
    d["forces"][0, :, 2] = 2000.
    t.update(**d)
    assert t.cost["tail_force"][0] > 4*t.cost["tail_force"][1]
    retained = t.window[:, 1].clone()
    t.reset(torch.tensor([0]), torch.tensor([0.]))
    assert torch.equal(t.window[:, 1], retained) and t.window[:, 0].count_nonzero() == 0
    assert t.force_peak[0] == 0 and t.force_peak[1] == 1000


def test_recovery_tracks_robot_after_slipping_wheel_handoff():
    t = TailTracker(1, "cpu", WidthSettings())
    for i in range(20):
        d = sample()
        d["omega"][0, i % 2, 1] = 12.
        t.update(**d)
    assert t.recovering.item() and t.recovery_events.item() == 1
    for _ in range(50):
        t.update(**sample())
    assert t.recovering.item()  # static stance without resumed progress is not recovery
    d = sample()
    d["root"][:, 0] = .05
    for _ in range(50):
        t.update(**d)
    assert not t.recovering.item()


def test_progressing_loaded_slip_costs_are_not_an_elapsed_recovery_failure():
    s = WidthSettings()
    t = TailTracker(1, "cpu", s)
    # Exceed the former deadline with genuine net progress and support.
    for i in range(650):
        x = i*.005*.15
        d = sample(x=x, speed=.15, spin=10.)
        d["root"][:, 0] = x
        t.update(**d)
    assert t.recovering.item() and t.recovery_age.item() > s.recovery_timeout_s
    assert t.cost["slip"].item() > 0 and t.slip_seconds.item() > 0
    assert not t.failed().item()


@pytest.mark.parametrize("rocking", [False, True])
def test_spinning_without_new_net_progress_still_fails(rocking):
    s = WidthSettings()
    t = TailTracker(1, "cpu", s)
    for i in range(350):
        d = sample(spin=10.)
        d["root"][:, 0] = .009*np.sin(i*.1) if rocking else 0.
        # A reported velocity alone cannot renew the progress grace period.
        d["root_velocity"][:, 0] = .6
        t.update(**d)
    assert t.recovery_failed.item() and t.failed().item()
    assert t.recovery_no_progress_age.item() > s.recovery_timeout_s


def test_forward_motion_does_not_mask_sustained_recovery_support_loss():
    s = WidthSettings()
    t = TailTracker(1, "cpu", s)
    for _ in range(20):
        t.update(**sample(spin=10.))
    for i in range(120):
        d = sample(spin=10.)
        d["root"][:, 0] = i*.005*.2
        d["forces"][:, 1:, 2] = 0.
        t.update(**d)
        if i == 50:
            assert not t.failed().item()  # Brief load transfer is allowed.
    assert t.recovery_failed.item()
    assert t.recovery_support_loss_age.item() > s.recovery_support_loss_s
    assert t.recovery_no_progress_age.item() < s.recovery_timeout_s


def test_brief_support_transfers_do_not_accumulate_into_support_loss_failure():
    t = TailTracker(1, "cpu", WidthSettings())
    for i in range(600):
        x = i*.005*.15
        d = sample(x=x, speed=.15, spin=10.)
        d["root"][:, 0] = x
        if i % 40 < 20:
            d["forces"][:, 1:, 2] = 0.
        t.update(**d)
    assert t.recovering.item() and not t.failed().item()


def test_new_recovery_and_partial_reset_get_fresh_watchdog_state():
    t = TailTracker(2, "cpu", WidthSettings())
    for _ in range(190):
        t.update(**sample(n=2, spin=10.))
    for _ in range(55):
        d = sample(n=2)
        d["root"][:, 0] = .05
        t.update(**d)
    assert not t.recovering.any()
    for _ in range(20):
        d = sample(n=2, spin=10.)
        d["root"][:, 0] = .05
        t.update(**d)
    assert t.recovering.all() and not t.recovery_failed.any()
    assert (t.recovery_no_progress_age < .15).all()
    kept = {k: getattr(t, k)[1].clone() for k in (
        "recovery_no_progress_age", "recovery_progress_x", "recovery_support_loss_age", "recovering")}
    t.reset(torch.tensor([0]), torch.tensor([.4]))
    for name, value in kept.items():
        assert torch.equal(getattr(t, name)[1], value)
        assert getattr(t, name)[0] == 0


def test_stopping_cannot_be_quiet_success_and_riser_force_cannot_support():
    t = TailTracker(1, "cpu", WidthSettings())
    for _ in range(810):
        t.update(**sample())
    assert t.failed().item() and not t.success.item()
    d = sample()
    d["forces"][..., 0] = 200.
    t.update(**d)
    assert not t.last["supported"].any()


def test_completion_requires_all_four_top_touches_and_final_capture():
    t = TailTracker(1, "cpu", WidthSettings())
    d = sample(x=2.35)
    d["position"][..., 2] += .9
    d["root"][:, 0], d["root"][:, 2] = 2.35, 1.3
    d["forces"][:, 2:, 2] = 0  # Rear feet never touched: no completion.
    for _ in range(60):
        t.update(**d)
    assert not t.success.item()
    d["forces"][:, 2:, 2] = 100
    for _ in range(10):
        t.update(**d)
    assert not t.success.item()
    for _ in range(65):
        t.update(**d)
    assert t.success.item() and not t.failed().item()


def test_completion_is_not_failure_and_ablation_switch_is_live():
    terms = {"tail_success": torch.tensor([True]), "illegal_contact": torch.tensor([False]),
             "time_out": torch.tensor([False])}
    manager = SimpleNamespace(active_terms=list(terms), get_term=lambda k: terms[k],
                              get_term_cfg=lambda k: SimpleNamespace(time_out=k == "time_out"))
    env = SimpleNamespace(num_envs=1, device="cpu", termination_manager=manager, step_dt=.02,
                          cfg=SimpleNamespace(stair_tail=WidthSettings()), tail_stage=2,
                          tail_tracker=SimpleNamespace(cost={"tail_force": torch.ones(1)}))
    assert not failure_mask(env).item() and failure_reward(env).item() == 0
    terms["illegal_contact"][0] = True
    assert failure_mask(env).item() and failure_reward(env).item() == 50
    assert tail_cost(env, "tail_force").item() == 50
    env.cfg.stair_tail.objective = "control"  # Hydra override after cfg creation.
    assert tail_cost(env, "tail_force").item() == 0


def test_finite_board_geometry_has_void_and_floor_below_approach():
    s = WidthSettings()
    for g in GEOMETRIES:
        mesh = stair_mesh(g, s)
        # Every elevated triangle stays inside the true 1 m staircase width.
        elevated = mesh.triangles[(mesh.triangles[..., 2] > -.1).any(-1)]
        assert np.max(np.abs(elevated[..., 1])) <= .5+1e-10
        assert mesh.bounds[0, 2] < -s.floor_depth
        if not g.closed_riser:
            # Each board is a distinct watertight connected component.
            assert len(mesh.split(only_watertight=False)) == g.steps+3
    assert {g.depth for g in GEOMETRIES} >= {.2, .3, .25, .35}
    assert len({p.stiffness for p in PROFILES}) > 2


def test_real_v1_complete_warm_start_and_reject_changed_base():
    teacher = torch.load(checked_path(TEACHER, TEACHER_SHA256), weights_only=True, map_location="cpu")
    path = ROOT / DEFAULT_START
    assert file_sha256(path) == DEFAULT_START_SHA256
    saved = torch.load(path, weights_only=True, map_location="cpu")
    validate_warm_start(saved, teacher)
    altered = deepcopy(saved)
    key = next(k for k in altered["actor_state_dict"] if k.startswith("mlp.") and k.endswith("weight"))
    altered["actor_state_dict"][key].view(-1)[0] += 1.
    with pytest.raises(ValueError, match="frozen"):
        validate_warm_start(altered, teacher)
