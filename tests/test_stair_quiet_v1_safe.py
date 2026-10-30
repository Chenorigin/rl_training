"""Safety counterexamples, real width weight migration and full safe resume."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace as NS
import sys
import runpy
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"source/rl_training"))
sys.path.insert(0, str(ROOT/"tests"))
from test_idle_stop_motion_v1 import ObservationEnv, train_cfg
from test_stair_quiet_v1_width_mechanics import sample
from task_support.stair_quiet_v1_safe.settings import SafeSettings
from task_support.stair_quiet_v1_safe.observations import longitudinal_scan, safety_features
from task_support.stair_quiet_v1_safe.signals import SafeTailTracker
from task_support.stair_quiet_v1_safe.course import record_outcomes, advance_materials, validate_counts, case_assignment, SCALARS, FEET
from task_support.stair_quiet_v1_width.settings import cases, PROFILES
from task_support.stair_quiet_v1_width.curriculum import WidthCurriculum
from task_support.stair_quiet_v1.landing import LandingSettings
from task_support.stair_quiet_v1_safe import runner as runtime


def state_fixture(n=4):
    s = SafeSettings()
    state = {k: torch.zeros(len(cases()), dtype=torch.long) for k in (
        "episodes", "successes", "stage_episodes", "stage_successes")}
    state.update(assign_count=torch.zeros(n, dtype=torch.long),
        width_curriculum=WidthCurriculum(n, "cpu", s),
        material_practice=torch.arange(n)%s.material_practice_every == 0,
        stage_table=torch.tensor([p.stage for _, p in cases()]),
        width_statistics={k: torch.zeros(10, len(cases()), dtype=torch.long) for k in (
            "episodes", "successes", "edge_failures", "recovery_timeouts", "stall_timeouts", "other_failures")},
        case_metrics={k: torch.zeros(len(cases())) for k in (
            "force_peak_sum", "force_peak_max", "impulse_peak_sum", "impulse_peak_max", "slip_seconds_sum",
            "unsafe_seconds_sum", "recovery_events_sum", "edge_failures_sum")},
        mechanics={k: torch.zeros((10, len(cases()), *((4,) if k in FEET else ())), dtype=torch.float64)
                   for k in (*SCALARS, *FEET)})
    state["width_curriculum"].level[state["material_practice"]] = s.material_practice_level
    return state


def test_only_side_void_is_filtered_and_real_stair_profile_is_retained():
    x = torch.tensor([[0., 0., 0., 1., 1., 1.]])
    y = torch.tensor([[-.7, 0., .7, -.7, 0., .7]])
    raw = torch.tensor([[1., .2, 1., 1., .05, 1.]])
    out = longitudinal_scan(raw, x, y, torch.ones(1), lateral_samples=3)
    assert torch.equal(out, torch.tensor([[.2, .2, .2, .05, .05, .05]]))
    assert torch.equal(longitudinal_scan(raw, x, y, torch.full((1,), 4.), lateral_samples=3), raw)
    # Entire scanner outside staircase: do not manufacture an in-bounds ray.
    assert torch.equal(longitudinal_scan(raw, x, y+3., torch.ones(1), lateral_samples=3), raw)
    assert torch.equal(raw, torch.tensor([[1., .2, 1., 1., .05, 1.]]))


def test_raw_safety_input_still_distinguishes_left_right_and_all_four_feet():
    d = sample(x=1.1, y=.25)
    d["position"][..., 2] += .15
    a = safety_features(d, SafeSettings())
    d["position"][:, 0, 1] += .1
    b = safety_features(d, SafeSettings())
    assert a.shape == (1, 48) and not torch.equal(a[:, :11], b[:, :11])
    assert torch.equal(a[:, 11:44], b[:, 11:44])
    assert a[0, 1] < a[0, 2] and a[0, 12] > a[0, 13]


def test_unsafe_first_load_is_charged_before_old_grace_and_contact_breaks():
    s = SafeSettings()
    good, bad = SafeTailTracker(1, "cpu", s), SafeTailTracker(1, "cpu", s)
    for tracker, x in ((good, 1.1), (bad, 1.02)):
        d = sample(x=x)
        d["position"][..., 2] += .15
        tracker.update(**d)
    assert good.unsafe_seconds.item() == 0 and good.cost["landing_risk"].item() == 0
    assert bad.cost["foothold"].item() == 0  # old grace is still present
    assert bad.cost["landing_risk"].item() > 0
    assert bad.touchdown_unsafe_count.sum() == 4
    assert bad.unsafe_foot_seconds.sum() > 0
    assert bad.unsafe_seconds.item() == pytest.approx(.005)
    bad.reset(torch.tensor([0]), torch.tensor([2.]))
    assert not bad.previous_supported.any() and bad.touchdown_unsafe_count.sum() == 0
    assert bad.previous_x.item() == 2.


def test_prediction_shaping_acts_before_contact_without_faking_a_failure():
    d = sample(x=1.10, force=0.)
    d["position"][..., 2] += .15+.04
    d["velocity"][..., 2] = -.1
    good, bad = SafeTailTracker(1, "cpu", SafeSettings()), SafeTailTracker(1, "cpu", SafeSettings())
    good.update(**d)
    d["velocity"][..., 0] = 1.
    bad.update(**d)
    assert good.cost["landing_risk"].item() == 0
    assert bad.cost["landing_risk"].item() > 0 and bad.predicted_unsafe_foot_seconds.sum() > 0
    assert not bad.failed().item()
    # A physical side escape remains a hard failure, even without a contact.
    d["position"][..., 1] = .45
    bad.update(**d)
    assert bad.edge_failed.item()


def test_normal_rolling_is_free_but_loaded_spin_has_slip_distance():
    a, b = SafeTailTracker(1, "cpu", SafeSettings()), SafeTailTracker(1, "cpu", SafeSettings())
    a.update(**sample(speed=.5))
    b.update(**sample(spin=10.))
    assert a.slip_foot_distance_m.sum() == 0 and b.slip_foot_distance_m.sum() > 0


def test_front_and_hind_early_landing_limits_are_independent():
    tracker = SafeTailTracker(1, "cpu", SafeSettings(rear_tread_margin=.06))
    d = sample(x=1.075)
    d["position"][..., 2] += .15
    tracker.update(**d)
    assert tracker.touchdown_unsafe_count[0].tolist() == [0., 0., 1., 1.]
    assert tracker.unsafe_seconds.item() == pytest.approx(.005)
    tracker.support_age[0, 2] = .1  # early and stable unsafe contacts coexist
    tracker.update(**d)
    assert tracker.unsafe_seconds.item() == pytest.approx(.010)


def test_wider_stairs_unlock_materials_but_narrow_success_cannot_fabricate_evidence():
    state, s = state_fixture(), SafeSettings(stage_min_episodes=1)
    env = NS(cfg=NS(stair_tail=s), tail_stage=0)
    types = torch.arange(0, len(cases()), len(PROFILES))
    yes, no = torch.ones(len(types), dtype=torch.bool), torch.zeros(len(types), dtype=torch.bool)
    record_outcomes(state, types, torch.full_like(types, 9), yes, no, no, no, s)
    advance_materials(env, state)
    assert env.tail_stage == 0 and state["stage_episodes"].sum() == 0
    record_outcomes(state, types, torch.full_like(types, 7), yes, no, no, no, s)
    validate_counts(state, s)
    advance_materials(env, state)
    assert env.tail_stage == 1 and state["stage_episodes"].sum() == 0
    assert state["width_statistics"]["successes"][9].sum() == len(types)


def test_bad_wide_contact_case_blocks_material_promotion():
    state, s = state_fixture(), SafeSettings(stage_min_episodes=1)
    env = NS(cfg=NS(stair_tail=s), tail_stage=0)
    ids = torch.arange(0, len(cases()), len(PROFILES))
    yes = torch.ones(len(ids), dtype=torch.bool)
    yes[-1] = False
    no = torch.zeros_like(yes)
    record_outcomes(state, ids, torch.full_like(ids, 7), yes, no, no, no, s)
    advance_materials(env, state)
    assert env.tail_stage == 0
    validate_counts(state, s)
    state["stage_successes"][-len(PROFILES)] += 1
    with pytest.raises(ValueError):
        validate_counts(state, s)


@pytest.mark.parametrize("allowed", [6, 18, 30])
def test_default_4096_practice_envs_cover_every_geometry_and_material(allowed):
    ids = torch.arange(4096)
    selected = ids%4 == 0
    seen = torch.stack([case_assignment(ids, torch.full_like(ids, count), allowed, 4, 42)
                        for count in range(allowed)])
    assert set(seen[:, selected].flatten().tolist()) == set(range(allowed))
    assert set(seen[:, ~selected].flatten().tolist()) == set(range(allowed))
    # Even a single practice env eventually sees every case independently.
    assert set(seen[:, 0].tolist()) == set(range(allowed))


@pytest.fixture
def make_runner(monkeypatch):
    monkeypatch.setattr(runtime, "course_state", lambda e: e.safe_state)
    def make():
        env = ObservationEnv()
        env.obs["locomotion_scan"] = env.obs["policy"][:, 57:].clone()
        env.obs["safety"] = torch.zeros(4, 48)
        env.cfg.stair_tail = SafeSettings()
        env.cfg.landing_control = LandingSettings()
        env.stair_quiet_geometry, env.tail_wheel_material_receipt = {}, {}
        env.stair_quiet_landing = NS(contract=lambda: {})
        env.scene = NS(terrain=NS(contact_receipt=[]))
        env.safe_state, env.tail_stage, env.resets = state_fixture(), 0, 0
        def reset():
            env.resets += 1
        env.reset = reset
        cfg = train_cfg()
        cfg.update(descent_guide_iters=0, descent_guide_decay_iters=0)
        with redirect_stdout(StringIO()):
            r = runtime.StairQuietSafeRunner(env, cfg, device="cpu")
        r.logger.writer = None
        return r
    return make


def test_real_width_weights_and_progress_preserved_new_branch_zero_and_gradients_work(make_runner):
    r = make_runner()
    r.initialize_task({})
    source = torch.load(ROOT/runtime.DEFAULT_START, map_location="cpu", weights_only=True)
    for kind in ("actor", "critic"):
        actual = getattr(r.alg, kind).state_dict()
        assert all(torch.equal(actual[k], v) for k, v in source[kind+"_state_dict"].items())
    actor = r.alg.actor
    assert actor.safety_mlp[-1].weight.count_nonzero() == 0
    assert r._initialization["source_iteration"] == 9500
    assert r.env.safe_state["width_curriculum"].level[0] == 7
    levels = source["m20_training_manifest"]["width_curriculum"]["level"]
    assert r.env.safe_state["width_curriculum"].level[1:].tolist() == [levels[i] for i in (1024, 2048, 3072)]
    assert not r.alg.optimizer.state and r.alg.guard_updates == 0 and r.env.resets == 1
    assert r.env.safe_state["width_statistics"]["episodes"].sum() == 0
    assert all(not p.requires_grad for p in actor.mlp.parameters())
    assert all(not p.requires_grad for p in actor.distribution.parameters())
    assert all(p.requires_grad for p in actor.safety_mlp.parameters())
    actor(r.env.obs).square().mean().backward()
    assert actor.safety_mlp[-1].weight.grad.abs().sum() > 0
    assert actor.residual_mlp[-1].weight.grad is not None
    assert all(p.grad is None for p in actor.mlp.parameters())
    obs = r.env.obs.clone()
    obs["motion_control"][:] = 0
    a = actor(obs)
    obs["locomotion_scan"][:] += .3
    obs["safety"][:] = 5
    assert torch.equal(actor(obs), a)
    actor(obs, stochastic_output=True)
    assert actor.get_output_log_prob(a).isfinite().all()


def test_full_safe_resume_preserves_new_head_optimizer_counters_and_evidence(make_runner, tmp_path):
    r = make_runner()
    r.initialize_task({})
    r.alg.guard_updates, r.current_learning_iteration = 12, 11
    with torch.no_grad():
        r.alg.actor.safety_mlp[-1].bias.fill_(.03)
    parameter = next(r.alg.actor.safety_mlp.parameters())
    r.alg.optimizer.state[parameter] = dict(step=torch.tensor(7.), exp_avg=torch.zeros_like(parameter),
                                          exp_avg_sq=torch.zeros_like(parameter))
    r.env.safe_state["mechanics"]["slip_seconds"][7, 0] = .3
    path = tmp_path/"safe.pt"
    r.save(path)
    resumed = make_runner()
    resumed.load(path)
    resumed.initialize_task({"resume": True})
    assert resumed.current_learning_iteration == resumed.alg.guard_updates == 12
    assert resumed.env.safe_state["mechanics"]["slip_seconds"][7, 0] == .3
    assert resumed.alg.optimizer.state_dict()["state"]
    assert torch.equal(r.alg.actor.safety_mlp[-1].bias, resumed.alg.actor.safety_mlp[-1].bias)
    assert torch.equal(r.env.safe_state["width_curriculum"].level, resumed.env.safe_state["width_curriculum"].level)
    assert torch.equal(r.alg.actor(r.env.obs), resumed.alg.actor(r.env.obs))
    bad = torch.load(path, weights_only=True)
    bad["m20_training_manifest"]["noise_refinement"]["contract_sha256"] = "0"*64
    torch.save(bad, path)
    with pytest.raises(ValueError, match="contract"):
        make_runner().load(path)


def test_registration_is_discovered_without_touching_old_registry_and_live_hashes_still_match():
    import gymnasium as gym
    registration = ROOT/"source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/stair_quiet_v1_safe/__init__.py"
    runpy.run_path(str(registration))
    assert gym.spec("Rough-Deeprobotics-M20-StairQuiet-v1-Safe").entry_point.endswith("stair_quiet_v1_safe_env:M20StairQuietV1SafeEnv")
    saved = torch.load(ROOT/runtime.DEFAULT_START, map_location="cpu", weights_only=True)
    import hashlib
    for name, digest in saved["m20_training_manifest"]["noise_refinement"]["source_sha256"].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest


@pytest.mark.parametrize("kind,key,value", [
    ("actor_state_dict", "safety_mlp.4.bias", float("nan")),
    ("critic_state_dict", "mlp.0.bias", float("inf")),
    ("preservation_reference_actor", "safety_mlp.0.bias", float("nan")),
])
def test_resume_rejects_nonfinite_models_before_loading(make_runner, tmp_path, kind, key, value):
    r = make_runner()
    r.initialize_task({})
    path = tmp_path/"safe.pt"
    r.save(path)
    saved = torch.load(path, map_location="cpu", weights_only=True)
    saved[kind][key][0] = value
    torch.save(saved, path)
    with pytest.raises(ValueError, match="model"):
        make_runner().load(path)


def test_width_checkpoint_cannot_be_used_as_full_safe_resume(make_runner):
    with pytest.raises(ValueError, match="contract"):
        make_runner().load(ROOT/runtime.DEFAULT_START)


def test_resume_rejects_nonfinite_optimizer_scalar(make_runner, tmp_path):
    r = make_runner()
    r.initialize_task({})
    path = tmp_path/"safe.pt"
    r.save(path)
    saved = torch.load(path, weights_only=True)
    saved["optimizer_state_dict"]["param_groups"][0]["betas"] = (.9, float("nan"))
    torch.save(saved, path)
    with pytest.raises(ValueError, match="optimizer"):
        make_runner().load(path)


def test_physics_reset_accepts_inference_caches(make_runner):
    r = make_runner()
    with torch.inference_mode():
        cache = torch.zeros(4)
    r.env._m20_zero_hold_heading_state = {"was_moving": cache}
    r.env.reset = lambda: cache.zero_()
    r._reset_physics()
    normal = r.env._m20_zero_hold_heading_state["was_moving"]
    assert not normal.is_inference()
    normal.zero_()


def test_full_scan_batch_preserves_grid_rows_without_pairwise_distance_memory():
    x = torch.linspace(-.8, .8, 17).repeat_interleave(11)[None].expand(4096, -1)
    y = torch.linspace(-.5, .5, 11).repeat(17)[None].expand_as(x)+.12
    width = torch.ones(4096)
    raw = torch.where(y.abs() <= .5, x, torch.ones_like(x))
    result = longitudinal_scan(raw, x, y, width)
    assert result.shape == (4096, 187) and torch.equal(result, x)


@pytest.mark.parametrize("changes", [{"material_practice_every": 1}, {"material_practice_level": 9},
    {"prediction_clearance_m": 0.}, {"landing_risk_weight": 1.}])
def test_invalid_safe_settings_rejected(changes):
    with pytest.raises(ValueError):
        SafeSettings(**changes).validate()
