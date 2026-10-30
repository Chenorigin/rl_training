"""Causal expiry/recovery through the actual sensor queue, without optimization."""
import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from task_support.cts.configuration import default_settings, validate_config
from task_support.cts.map_recovery import RecoveryObservationState, configure, validate_recovery
from task_support.student_v1.configuration import StudentV1PerceptionCfg
from task_support.student_v1.perception import StudentObservationState, terrain_visibility
from test_cts_all_terrain_perception import scene


def recovery(**overrides):
    cfg = deepcopy(default_settings()['map_recovery'])
    cfg.update(episode_probability=1., interval_s=[.2, .2], blackout_extra_s=[.1, .1],
               recovery_s=[.2, .2], initial_keep_fraction=[.15, .15], ramp_probability=0.)
    cfg.update(overrides)
    return cfg


def sensor(**overrides):
    cfg = dict(continuous_delay=True, min_delay_s=.04, max_delay_s=.04,
               max_map_age_s=.1, map_update_period_s=.02, update_jitter_s=0.,
               terrain_visibility=False, lidar_horizontal_fov_deg=360.,
               vertical_fov_min_deg=-90., vertical_fov_max_deg=90.,
               level=0., clean_probability=1., stress_probability=0.)
    cfg.update(overrides)
    return StudentV1PerceptionCfg(**cfg)


def test_old_cells_expire_then_sparse_returns_wait_for_delivery_with_original_age():
    torch.manual_seed(84)
    state = RecoveryObservationState(2, 'cpu', .02, sensor(), 4, recovery=recovery())
    args = scene(2)
    outputs = [state.observe(*args, step) for step in range(33)]
    # First outage starts at .20, last original capture at .18. Masking does
    # not immediately erase the cache, nor refresh its original timestamps.
    assert outputs[10][:, 244:431].sum() > 0
    assert outputs[14][:, 244:431].sum() > 0  # age == TTL is valid
    for step in range(15, 22):
        assert torch.count_nonzero(outputs[step][:, 57:618]) == 0
    # Recovery capture at .40 cannot appear until its .04 delivery delay.
    counts = outputs[22][:, 244:431].sum(-1)
    assert ((counts > 0) & (counts < 187)).all()
    # Restoring the ordinary capture at .60 likewise arrives at .64.
    assert (outputs[31][:, 244:431].sum(-1) < 187).all()
    assert (outputs[32][:, 244:431].sum(-1) == 187).all()
    assert state.recovery_events.tolist() == [1, 1]
    assert state.recovery_transitions.tolist() == [2, 2]  # initial packet + restoration
    assert torch.allclose(state.cell_age, torch.full_like(state.cell_age, .04))
    # Every unknown has zero height AND confidence, including mixed sparse maps.
    for obs in outputs:
        unknown = obs[:, 244:431] == 0
        assert (obs[:, 57:244][unknown] == 0).all()
        assert (obs[:, 431:618][unknown] == 0).all()
        assert torch.isfinite(obs).all() and obs.shape == (2, 789)
    assert torch.isfinite(args[1]).all()  # caller's terrain geometry was not mutated


@pytest.mark.parametrize('disabled', ['enabled', 'probability'])
def test_disabled_path_preserves_observations_queue_and_rng_exactly(disabled):
    cfg = recovery(**({'enabled': False} if disabled == 'enabled' else {'episode_probability': 0.}))
    results = []
    for kind in (StudentObservationState, RecoveryObservationState):
        torch.manual_seed(401)
        kwargs = {'recovery': cfg} if kind is RecoveryObservationState else {}
        state = kind(4, 'cpu', .02, sensor(level=1., clean_probability=0.), 4, **kwargs)
        outputs = [state.observe(*scene(4), step) for step in range(50)]
        results.append((torch.stack(outputs), state.packet_capture, state.packet_valid, torch.get_rng_state()))
    assert all(torch.equal(a, b) for a, b in zip(*results))


def test_same_tick_cache_and_partial_reset_do_not_advance_neighbors_or_rng():
    state = RecoveryObservationState(2, 'cpu', .02, sensor(), 4, recovery=recovery())
    for step in range(25):
        original = state.observe(*scene(2), step)
    rng, history, next_event = torch.get_rng_state(), state.history.clone(), state.recovery_next.clone()
    assert torch.equal(state.observe(*scene(2), 24), original)
    assert torch.equal(torch.get_rng_state(), rng)
    assert torch.equal(state.history, history)
    state.reset(torch.tensor([0]))
    assert state.recovery_previous_count[0] == -1 and state.recovery_phase[0] == 0
    output = state.observe(*scene(2), 24)
    assert torch.equal(output[1], original[1])
    assert torch.equal(state.history[1], history[1])
    assert state.recovery_next[1] == next_event[1]
    assert torch.count_nonzero(output[0, 57:618]) == 0
    assert (state.packet_capture[:, 0] <= 24).all()
    assert state.recovery_events[0] == 1  # lifetime counter is intentionally retained


def test_ramped_recovery_adds_observed_cells_without_rewriting_old_capture_times():
    torch.manual_seed(17)
    state = RecoveryObservationState(1, 'cpu', .02, sensor(), 4, recovery=recovery(ramp_probability=1.))
    counts = []
    for step in range(32):
        obs = state.observe(*scene(1), step)
        if 22 <= step <= 31:
            counts.append(int(obs[:, 244:431].sum()))
            assert (state.cell_age[obs[:, 244:431].bool()] >= .04 - 1e-6).all()
    assert counts == sorted(counts) and counts[-1] > counts[0] > 0


def test_thinning_an_occluding_surface_does_not_expose_hidden_geometry():
    cfg = sensor(terrain_visibility=True)
    state = RecoveryObservationState(1, 'cpu', .02, cfg, 4, recovery=recovery())
    proprio, hits, base, yaw, rotation = scene(1)
    # A tall strip between a sensor and the remaining ground.
    hits[:, (state.grid[:, 0] > .3) & (state.grid[:, 0] < .5), 2] = 1.2
    visible = terrain_visibility(hits, base, yaw, rotation, cfg)
    assert visible.any() and (~visible).any()  # positive and negative controls
    state.recovery_needs_reset.zero_()
    state.recovery_selected.fill_(True)
    state.recovery_next.fill_(100)
    state.recovery_start.fill_(0)
    state.recovery_end.fill_(100)
    state.recovery_keep.fill_(.5)
    state.recovery_rank.uniform_()
    masked = state._acquisition_hits(hits, base, yaw, rotation, 1, torch.tensor([True]))
    assert not torch.isfinite(masked).all(-1)[~visible].any()


@pytest.mark.parametrize('key', ['episode_probability', 'ramp_probability', 'interval_s',
                                 'blackout_extra_s', 'recovery_s', 'initial_keep_fraction'])
@pytest.mark.parametrize('bad', [True, float('nan'), float('inf'), -1.])
def test_invalid_sampling_contract_fails_closed(key, bad):
    cfg = recovery()
    cfg[key] = [bad, 1.] if isinstance(cfg[key], list) else bad
    with pytest.raises(ValueError):
        validate_recovery(cfg)


def test_default_v15_and_pre_feature_v15_are_readable_without_silent_enable():
    cfg = default_settings()
    validate_config(cfg)
    cfg.pop('map_recovery')
    validate_config(cfg)
    assert 'map_recovery' not in cfg
    for bad in ({}, {**recovery(), 'unused': 1}, {**recovery(), 'enabled': 1},
                {**recovery(), 'initial_keep_fraction': [.1, 1.]},
                {**recovery(), 'interval_s': [2., 1.]}):
        with pytest.raises(ValueError):
            validate_recovery(bad)


@pytest.mark.parametrize('yaw_rate,stairs', [(-.6, False), (.6, False), (-.6, True), (.6, True)])
def test_turning_and_stair_returns_keep_age_and_unknown_semantics(yaw_rate, stairs):
    from task_support.student_v1.perception import grid_points
    torch.manual_seed(917)
    cfg = sensor(max_map_age_s=.5, map_capture_padding_m=.6)
    state = RecoveryObservationState(2, 'cpu', .02, cfg, 4, recovery=recovery())
    proprio, _, base, yaw, rotation = scene(2)
    grid = grid_points('cpu', .6)
    empty_frames, sparse_frames = 0, 0
    for step in range(130):
        yaw.fill_(step * .02 * yaw_rate)
        co, si = yaw[0].cos(), yaw[0].sin()
        rotation[:, :2, :2] = torch.stack((co, -si, si, co)).reshape(2, 2)
        base[:, 0] = step * .02 * .2
        proprio[:, 6] = .2
        proprio[:, 8] = yaw_rate
        xy = grid[None].expand(2, -1, -1) @ rotation[:, :2, :2].transpose(1, 2) + base[:, None, :2]
        z = (xy[..., :1] / .3).floor() * .05 if stairs else torch.zeros_like(xy[..., :1])
        obs = state.observe(proprio, torch.cat((xy, z), -1), base, yaw, rotation, step)
        valid = obs[:, 244:431].bool()
        empty_frames += int((valid.sum(-1) == 0).sum())
        sparse_frames += int(((valid.sum(-1) > 0) & (valid.sum(-1) < 187)).sum())
        assert obs.shape == (2, 789) and torch.isfinite(obs).all()
        assert (obs[:, 57:244][~valid] == 0).all()
        assert (obs[:, 431:618][~valid] == 0).all()
        assert (state.cell_age[valid] <= cfg.max_map_age_s).all()
    assert empty_frames > 0 and sparse_frames > 0
    assert (state.recovery_events > 1).all()


def test_memory_reserve_covers_allocated_recovery_state_and_disabled_cost_is_zero():
    from dataclasses import asdict
    from task_support.cts.map_recovery import memory_reserve
    cfg = sensor(map_capture_padding_m=.6)
    state = RecoveryObservationState(4, 'cpu', .02, cfg, 4, recovery=recovery())
    live = sum(v.numel() * v.element_size() for k, v in vars(state).items()
               if k.startswith('recovery_') and isinstance(v, torch.Tensor))
    assert memory_reserve(recovery(), asdict(cfg), 4) > live
    assert memory_reserve(recovery(enabled=False), asdict(cfg), 4) == 0


def test_actual_environment_constructor_installs_state_before_parent_uses_sensor_factory():
    # Execute the production constructor with only the simulator parent replaced.
    # This catches install-after-super, wrong config wiring, or a second sensor.
    from task_support.student_v1.environment import sensor_state
    root = Path(__file__).resolve().parents[1]
    path = root / 'source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/cts_env.py'
    node = next(x for x in ast.parse(path.read_text()).body
                if isinstance(x, ast.ClassDef) and x.name == 'M20CTSAllTerrainEnv')
    init = next(x for x in node.body if isinstance(x, ast.FunctionDef) and x.name == '__init__')
    # Joint-limit installation requires Isaac and is independently tested.
    init.body = init.body[:-2]
    node.body = [init]
    class Parent:
        def __init__(self, cfg, **kwargs):
            self.num_envs, self.device, self.step_dt, self.cfg = 2, 'cpu', .02, cfg
            self.observation_manager = SimpleNamespace(group_obs_dim={'teacher_critic': (285,)})
            self.observed_sensor = sensor_state(self)
            assert isinstance(self.observed_sensor, RecoveryObservationState)
            assert sensor_state(self) is self.observed_sensor
            self.observed_sensor.observe(*scene(2), 0)
    namespace = {'M20StairQuietV1StudentEnv': Parent, 'torch': torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    cfg = SimpleNamespace(scene=SimpleNamespace(num_envs=2), sim=SimpleNamespace(device='cpu', dt=.005),
                          decimation=4, student_perception=sensor(), student_history_steps=4)
    configure(cfg, recovery())
    env = namespace['M20CTSAllTerrainEnv'](cfg)
    assert env.observed_sensor.last_obs.shape == (2, 789)
