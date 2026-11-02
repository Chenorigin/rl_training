"""Physical-state handover gates and real course wiring, with deterministic fake physics."""
from copy import deepcopy
from types import SimpleNamespace
import pytest
import torch
from test_cts_wall_hold import source as legacy_source, setup, step
from task_support.cts.c_handover import validate


@pytest.fixture(scope='module')
def source(legacy_source):
    settings, atlas = deepcopy(legacy_source)
    settings['all_terrain']['wall_hold_practice']['handover'] = dict(
        fraction=.5, standing_s_range=[.6, 1.4], blend_s=.3)
    return settings, atlas


@pytest.fixture
def transition(setup, monkeypatch):
    env, state, hold, _, _, _ = setup
    from task_support.student_v1 import environment as sensor
    from task_support.student_v3_5 import v3_5_environment as feedback
    from task_support.cts import all_terrain_terms
    calls = []
    def record(name):
        return lambda *a: calls.append((name, a[-1].clone()))
    monkeypatch.setattr(sensor, 'reset_student', record('sensor_delay'))
    monkeypatch.setattr(sensor, 'sensor_state', lambda _: SimpleNamespace(history=torch.zeros(env.num_envs, 3, 57)))
    monkeypatch.setattr(sensor, 'action_delay', lambda _: SimpleNamespace(previous_issued=torch.zeros(env.num_envs, 16)))
    monkeypatch.setattr(feedback, 'reset_feedback', record('student_heading'))
    monkeypatch.setattr(all_terrain_terms, 'reset_clean_commands', record('teacher_heading'))
    env.stair_quiet_landing = SimpleNamespace(before_reset=record('restore_gains'), reset=record('landing'))
    env.action_manager.reset = record('action_history')
    return env, state, hold, calls


def test_independent_cohorts_geometry_and_wait_exclusion(setup):
    env, s, h, _, _, _ = setup
    assert h.num_groups == 2
    for group in [0, 1]:
        for teacher in [False, True]:
            ids = (h.active & (h.group == group) & (s.teacher == teacher)).nonzero().flatten()
            assert len(ids)
            assert all(s.atlas.records[int(s.tile[i])]['width'] == 1. for i in ids)
    assert not s.tags()['cts_policy_active'][h.handover.active].any()
    for k in range(200):
        step(env, s, .3+k*.02)
    assert not h.ready[h.handover.active].any()
    assert not s.success[h.handover.active].any()
    assert h.totals['frames'][2:].sum() == 0
    assert h.totals['frames'][:2].sum() > 0


def test_c_is_measured_once_no_teleport_and_resets_only_selected(transition):
    env, s, h, calls = transition
    hand = h.handover
    ids = hand.active.nonzero().flatten()
    first = ids[:1]
    hand.wait_until[ids] = 20.
    hand.wait_until[first] = .6
    env.episode_length_buf[:] = 29
    root = env.scene.robot.data.root_pos_w.clone()
    anchor_before = h.anchor.clone()
    switched = hand.after_physics()
    torch.testing.assert_close(switched, first)
    torch.testing.assert_close(env.scene.robot.data.root_pos_w, root)
    assert [name for name, _ in calls] == ['sensor_delay', 'student_heading', 'teacher_heading', 'restore_gains', 'landing', 'action_history']
    assert all(torch.equal(actual, first) for _, actual in calls)
    other = torch.ones(env.num_envs, dtype=torch.bool);other[first] = False
    torch.testing.assert_close(h.anchor[other], anchor_before[other])
    assert not hand.after_physics().numel()  # No second C / no re-anchoring.
    assert s.tags()['cts_policy_active'][first].all()
    assert not s.tags()['cts_policy_active'][ids[1:]].any()


def test_issued_ramp_preserves_ppo_sample_and_first_step_zero(transition):
    env, s, h, calls = transition
    hand = h.handover
    hand.wait_until[:] = .6
    env.episode_length_buf[:] = 29
    ids = hand.after_physics()
    raw = torch.full((env.num_envs, 16), 2.)
    saved = raw.clone()
    env.episode_length_buf[:] = 30
    issued = hand.issued_action(raw)
    assert not issued[ids].any()
    torch.testing.assert_close(issued[~hand.active], raw[~hand.active])
    env.episode_length_buf[:] = 35  # C + .1s, not .12s
    torch.testing.assert_close(hand.issued_action(raw)[ids], raw[ids]/3)
    env.episode_length_buf[:] = 45
    torch.testing.assert_close(hand.issued_action(raw), raw)
    torch.testing.assert_close(raw, saved)


def test_post_c_escape_never_becomes_success(transition):
    env, s, h, _ = transition
    hand = h.handover
    hand.wait_until[:] = .6
    env.episode_length_buf[:] = 29
    ids = hand.after_physics()
    i = int(ids[0])
    anchor = h.anchor[i].clone()
    env.scene.robot.data.root_pos_w[i, 0] -= .03
    step(env, s, .62)  # Still in blending grace; escape must already latch.
    assert h.escaped[i]
    env.scene.robot.data.root_pos_w[i] = anchor
    for k in range(180):step(env, s, .9+k*.02)
    assert not s.success[i]
    torch.testing.assert_close(h.anchor[i], anchor)


def test_separate_ledger_and_snapshot(setup):
    env, s, h, _, _, _ = setup
    ids = h.handover.active.nonzero().flatten()[:1]
    i = int(ids[0]);s.success[i] = False
    h.record(ids, torch.zeros(env.num_envs, dtype=torch.bool))
    assert h.totals['episodes'][:2].sum() == 0
    assert h.totals['episodes'][2:].sum() == 1
    saved = h.snapshot();h.checked_snapshot(saved)
    assert len(saved['curriculum']['states']) == 2
    assert 'c_handover/student/episodes' in h.diagnostics()


@pytest.mark.parametrize('key,value', [('fraction',0),('fraction',1),('blend_s',.2),('standing_s_range',[1,.5])])
def test_invalid_handover_rejected(key, value):
    c = dict(fraction=.5, standing_s_range=[.6,1.4], blend_s=.3);c[key] = value
    with pytest.raises(ValueError):validate(c)


def test_wait_frames_excluded_from_actual_ppo_and_distillation(monkeypatch):
    from test_cts_algorithm import fixture
    models, alg, obs, *_ = fixture()
    active = torch.ones(8, dtype=torch.bool)
    # Keep both cohorts represented while marking a unique inactive observation.
    teacher = int(alg.teacher_mask.nonzero()[0]);student = int((~alg.teacher_mask).nonzero()[0])
    active[[teacher, student]] = False
    obs['policy'][~active, 0] = 123.
    with torch.inference_mode():
        for _ in range(4):
            raw = alg.act(obs);before = raw.clone()
            alg.process_env_step(obs, torch.ones(8), torch.zeros(8, dtype=torch.bool), {'cts_policy_active':active})
            torch.testing.assert_close(alg.rollout[-1]['actions'], before)
        alg.compute_returns(obs)
    assert not alg.advantages[:, ~active].any()
    original_mean = models.mean
    def mean(values, mask, **kwargs):
        assert not (values['policy'][:, 0] == 123).any()
        return original_mean(values, mask, **kwargs)
    monkeypatch.setattr(models, 'mean', mean)
    assert alg.ppo_update()['ppo_minibatches'] > 0
    original_student = models.student_latent
    def latent(values):
        assert not (values['policy'][:, 0] == 123).any()
        return original_student(values)
    monkeypatch.setattr(models, 'student_latent', latent)
    assert alg.distill_update()['student_minibatches'] > 0


def test_ready_execution_restores_only_waiting_targets_and_gains():
    from task_support.cts.c_handover import install_standing_execution
    calls=[]
    pos=torch.full((3,12), 7.);vel=torch.full((3,4),8.);gains=torch.ones(3)
    default=torch.arange(48,dtype=torch.float32).reshape(3,16)/100
    def original(dt):pos.fill_(7.);vel.fill_(8.);gains.fill_(.5)
    def setpos(value,*,joint_ids,env_ids):pos[env_ids]=value
    def setvel(value,*,joint_ids,env_ids):vel[env_ids]=value
    def restore(ids):gains[ids]=1.;calls.append(ids.clone())
    landing=SimpleNamespace(apply=original,before_reset=restore,leg_ids=torch.arange(12).reshape(4,3),wheel_ids=torch.arange(12,16))
    robot=SimpleNamespace(data=SimpleNamespace(default_joint_pos=default),set_joint_position_target=setpos,set_joint_velocity_target=setvel)
    hand=SimpleNamespace(active=torch.tensor([True,True,False]),started=torch.tensor([False,True,False]))
    env=SimpleNamespace(stair_quiet_landing=landing,scene={'robot':robot},cts_all_terrain=SimpleNamespace(wall_hold=SimpleNamespace(handover=hand)))
    install_standing_execution(env);landing.apply(.005)
    torch.testing.assert_close(pos[0],default[0,:12]);assert not vel[0].any();assert gains[0]==1
    assert (pos[1:]==7).all() and (vel[1:]==8).all() and (gains[1:]==.5).all()
    hand.started[0]=True;landing.apply(.005)
    assert (pos==7).all() and (vel==8).all()  # No residual controller lock after C.


def test_ready_controller_installed_inside_final_target_guard():
    import ast
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / 'source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/cts_env.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'M20CTSAllTerrainEnv')
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    calls = {n.func.id: n.lineno for n in ast.walk(init) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert calls['install_standing_execution'] < calls['install']
