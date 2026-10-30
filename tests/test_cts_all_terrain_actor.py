"""Bounded idle adaptation, strict migration and actual CTS anchor gradients."""
from copy import deepcopy
from unittest.mock import patch

import pytest
import torch

from test_cts_algorithm import fixture
from task_support.cts.algorithm import CTSAlgorithm, domain_anchor_loss
from task_support.cts.model import build_models, transfer_state
from task_support.cts.straight import straight_teacher_actions


def terrain_fixture():
    legacy, algorithm, obs, teacher, student, critic, saved_teacher, saved_student = fixture()
    all_terrain = {'actor': {'idle_leg_scale': .25, 'new_domain_anchor_scale': .1}}
    models = build_models(saved_teacher, saved_student, algorithm.cfg, all_terrain_cfg=all_terrain)
    algorithm = CTSAlgorithm(models, algorithm.cfg, 8, 4, all_terrain_cfg=all_terrain)
    return models, algorithm, obs, legacy, saved_teacher, saved_student, all_terrain


def collect(algorithm, obs, known=None):
    known = torch.arange(8) % 2 == 0 if known is None else known
    with torch.inference_mode():
        for step in range(4):
            actions = algorithm.act(obs)
            rewards = torch.arange(8, dtype=torch.float32)*.1 + step*.2 - actions.square().sum(-1)
            algorithm.process_env_step(obs, rewards, torch.zeros(8, dtype=torch.bool),
                                       {'cts_anchor': known, 'cts_course': torch.arange(8) % 3})
        algorithm.compute_returns(obs)


def test_optional_architecture_exactly_preserves_legacy_idle_and_moving_outputs():
    models, algorithm, obs, legacy, *_ = terrain_fixture()
    obs['motion_control'][::2] = 0
    for method in ('teacher_latent', 'student_latent'):
        torch.testing.assert_close(getattr(models, method)(obs), getattr(legacy, method)(obs), rtol=0, atol=0)
    torch.testing.assert_close(models.mean(obs, algorithm.teacher_mask)[0],
                               legacy.mean(obs, algorithm.teacher_mask)[0], rtol=0, atol=0)
    assert not any('idle_mlp' in key for key in legacy.state_dict())
    with torch.no_grad():
        models.teacher.idle_mlp[-1].bias.fill_(10.)
        models.decoder.idle_mlp[-1].bias.fill_(10.)
    moving = obs['motion_control'].abs().amax(-1) > 1.e-4
    torch.testing.assert_close(models.mean(obs, algorithm.teacher_mask)[0][moving],
                               legacy.mean(obs, algorithm.teacher_mask)[0][moving], rtol=0, atol=0)


def test_idle_teacher_gradients_and_saturated_bounds_do_not_change_wheels():
    models, _, obs, legacy, *_ = terrain_fixture()
    obs['motion_control'].zero_()
    action = models.teacher_latent(obs)
    action[:, :12].sum().backward()
    assert models.teacher.idle_mlp[-1].weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in models.teacher.mlp.parameters())
    assert all(parameter.grad is None or not parameter.grad.any()
               for module in (models.teacher.residual_mlp, models.teacher.safety_mlp)
               for parameter in module.parameters())
    with torch.no_grad():
        models.teacher.idle_mlp[-1].bias.fill_(100.)
        delta = models.teacher_latent(obs)-legacy.teacher_latent(obs)
    torch.testing.assert_close(delta[:, :12], delta.new_tensor([.18, .35, .35]*4).expand(8, -1)*.25)
    torch.testing.assert_close(delta[:, 12:], torch.zeros(8, 4), rtol=0, atol=0)


def test_idle_decoder_export_has_real_gradient_same_abi_and_no_privileged_input(tmp_path):
    models, _, obs, *_ = terrain_fixture()
    obs['motion_control'].zero_()
    latent = models.student_latent(obs).detach()
    output = models.decoder(obs['policy'][:, :57], latent, obs['motion_control'])
    output[:, :12].sum().backward()
    assert models.decoder.idle_mlp[-1].weight.grad.abs().sum() > 0
    assert all(parameter.grad is None or not parameter.grad.any() for parameter in models.decoder.network.parameters())
    with torch.no_grad():
        models.decoder.idle_mlp[-1].bias.fill_(100.)
    action = models.student_action(obs['policy'], obs['motion_control'])
    torch.testing.assert_close(action[:, 12:], latent[:, 12:], rtol=0, atol=0)
    torch.testing.assert_close(action[:, :12]-latent[:, :12], models.decoder.bounds[:12].expand(8, -1)*.25)
    exported = torch.jit.script(models.student_export())
    target = tmp_path/'student.pt'
    torch.jit.save(exported, str(target))
    reloaded = torch.jit.load(str(target))
    torch.testing.assert_close(reloaded(obs['policy'], obs['motion_control']), action, rtol=0, atol=0)
    obs['teacher'].fill_(1000.)
    obs['safety'].fill_(1000.)
    torch.testing.assert_close(reloaded(obs['policy'], obs['motion_control']), action, rtol=0, atol=0)


def test_legacy_transfer_only_adds_zero_output_heads_and_rejects_partial_heads():
    models, algorithm, obs, legacy, *_ = terrain_fixture()
    with torch.no_grad():
        legacy.decoder.network[-1].bias.fill_(.1)
    source = deepcopy(legacy.state_dict())
    merged = transfer_state(models, source)
    algorithm_state = algorithm.save()
    algorithm_state['models_state_dict'] = merged
    algorithm.load(algorithm_state)
    obs['motion_control'][::2] = 0
    torch.testing.assert_close(models.mean(obs, algorithm.teacher_mask)[0],
                               legacy.mean(obs, algorithm.teacher_mask)[0], rtol=0, atol=0)
    partial = deepcopy(merged)
    partial.pop('teacher.idle_mlp.0.weight')
    with pytest.raises(ValueError, match='complete'): transfer_state(models, partial)
    corrupted = deepcopy(source)
    corrupted.pop('teacher.residual_mlp.0.weight')
    with pytest.raises(ValueError, match='complete'): transfer_state(models, corrupted)
    corrupted = deepcopy(source)
    corrupted['decoder.network.0.weight'] = corrupted['decoder.network.0.weight'].double()
    with pytest.raises(ValueError, match='tensor'): transfer_state(models, corrupted)
    with torch.no_grad(): models.teacher.idle_mlp[-1].bias.fill_(.01)
    with pytest.raises(ValueError, match='zero-output'): transfer_state(models, source)


def test_anchor_weight_is_absolute_per_sample_and_changes_real_gradients():
    residual = torch.ones(4, requires_grad=True)
    known = torch.tensor([True, True, False, False])
    loss = domain_anchor_loss(residual.square(), known, .1)
    torch.testing.assert_close(loss, torch.tensor(.55))
    loss.backward()
    torch.testing.assert_close(residual.grad, torch.tensor([.5, .5, .05, .05]))
    torch.testing.assert_close(domain_anchor_loss(residual.square(), torch.zeros(4, dtype=torch.bool), .1),
                               torch.tensor(.1))


def test_rollout_requires_pre_step_tags_and_clones_them_before_reset():
    _, algorithm, obs, *_ = terrain_fixture()
    algorithm.act(obs)
    rewards, dones = torch.ones(8), torch.zeros(8, dtype=torch.bool)
    with pytest.raises(ValueError, match='cts_anchor'): algorithm.process_env_step(obs, rewards, dones, {})
    known, course = torch.ones(8, dtype=torch.bool), torch.arange(8) % 3
    for bad in (course.float(), torch.full((8,), -1, dtype=torch.long),
                torch.full((8,), algorithm.num_courses, dtype=torch.long), course[:4]):
        with pytest.raises(ValueError, match='cts_course'):
            algorithm.process_env_step(obs, rewards, dones, {'cts_anchor': known, 'cts_course': bad})
    algorithm.process_env_step(obs, rewards, dones, {'cts_anchor': known, 'cts_course': course})
    known.zero_(); course.fill_(10)
    assert algorithm.rollout[0]['anchor'].all()
    assert (algorithm.rollout[0]['course'] < 3).all()


def test_actual_ppo_wires_anchor_tags_and_updates_new_idle_parameters():
    models, algorithm, obs, *_ = terrain_fixture()
    obs['motion_control'].zero_()
    collect(algorithm, obs)
    before = deepcopy(models.state_dict())
    with patch('task_support.cts.algorithm.domain_anchor_loss', wraps=domain_anchor_loss) as anchor_call:
        result = algorithm.ppo_update()
    assert result['ppo_minibatches'] > 0 and anchor_call.call_count > 0
    seen_known = torch.cat([call.args[1] for call in anchor_call.call_args_list])
    assert seen_known.any() and (~seen_known).any()
    assert all(call.args[2] == .1 for call in anchor_call.call_args_list)
    after = models.state_dict()
    for prefix in ('teacher.idle_mlp.', 'decoder.idle_mlp.'):
        assert any(not torch.equal(value, after[key]) for key, value in before.items() if key.startswith(prefix))
    for prefix in ('student.', 'teacher.mlp.', 'teacher.residual_mlp.', 'teacher.safety_mlp.', 'reference_teacher.'):
        assert all(torch.equal(value, after[key]) for key, value in before.items() if key.startswith(prefix))


def test_roundtrip_preserves_new_parameters_optimizer_and_contract():
    models, algorithm, obs, _, teacher, student, cfg = terrain_fixture()
    obs['motion_control'][::2] = 0
    collect(algorithm, obs)
    metrics = algorithm.update()
    assert metrics['online_teacher_new_domain_fraction'] + metrics['online_teacher_known_domain_fraction'] == 1.
    assert metrics['online_student_new_domain_fraction'] + metrics['online_student_known_domain_fraction'] == 1.
    saved = deepcopy(algorithm.save())
    rebuilt = build_models(teacher, student, algorithm.cfg, all_terrain_cfg=cfg)
    restored = CTSAlgorithm(rebuilt, algorithm.cfg, 8, 4, all_terrain_cfg=cfg)
    restored.load(saved)
    assert restored.optimizer.state and restored.student_optimizer.state
    for key, value in models.state_dict().items():
        torch.testing.assert_close(rebuilt.state_dict()[key], value, rtol=0, atol=0)
    bad = deepcopy(saved); bad['all_terrain_contract']['actor']['idle_leg_scale'] = .5
    before = deepcopy(rebuilt.state_dict())
    with pytest.raises(ValueError, match='contract'): restored.load(bad)
    assert all(torch.equal(value, rebuilt.state_dict()[key]) for key, value in before.items())


def test_clean_straight_target_stays_independent_of_noisy_student_common_input():
    models, _, obs, *_ = terrain_fixture()
    with torch.no_grad(): models.decoder.network[-1].weight.fill_(.02)
    before = straight_teacher_actions(models, obs)
    obs['policy'].add_(2.)
    torch.testing.assert_close(straight_teacher_actions(models, obs), before, rtol=0, atol=0)


def test_complete_teacher_rollout_and_targets_ignore_student_noise_but_student_uses_it():
    models, algorithm, obs, *_ = terrain_fixture()
    obs['motion_control'][::2] = 0
    with torch.no_grad():
        models.decoder.network[-1].weight.fill_(.02)
        models.decoder.idle_mlp[-1].weight.fill_(.02)
        before = models.mean(obs,algorithm.teacher_mask)[0]
        target = models.teacher_action(obs)
        old_common = models.decoder(obs['policy'][:, :57],models.teacher_latent(obs),obs['motion_control'])
        obs['policy'].add_(.7)
        after = models.mean(obs,algorithm.teacher_mask)[0]
        changed_target = models.teacher_action(obs)
        wrong_common = models.decoder(obs['policy'][:, :57],models.teacher_latent(obs),obs['motion_control'])
    torch.testing.assert_close(before[algorithm.teacher_mask],after[algorithm.teacher_mask],rtol=0,atol=0)
    torch.testing.assert_close(target,changed_target,rtol=0,atol=0)
    assert (before[~algorithm.teacher_mask]-after[~algorithm.teacher_mask]).abs().max() > 1.e-5
    # A negative oracle proves this catches the previous contaminated decoder.
    assert (old_common-wrong_common).abs().max() > 1.e-5


def test_new_contract_cannot_be_silently_enabled_on_a_legacy_architecture():
    models, algorithm, _, legacy, *_ = terrain_fixture()
    with pytest.raises(ValueError, match='architecture'):
        CTSAlgorithm(legacy, algorithm.cfg, 8, 4, all_terrain_cfg=algorithm.all_terrain_cfg)
    with pytest.raises(ValueError, match='architecture'):
        CTSAlgorithm(models, algorithm.cfg, 8, 4)
