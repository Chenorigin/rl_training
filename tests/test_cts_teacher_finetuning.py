"""Full teacher updates, isolated optimizer groups and strict resume semantics."""
from copy import deepcopy

import pytest
import torch

from test_cts_algorithm import fixture
from test_cts_all_terrain_actor import collect
from task_support.cts.algorithm import CTSAlgorithm, validate_config
from task_support.cts.configuration import default_settings, validate_config as validate_document
from task_support.cts.model import build_models


def setup():
    _, legacy, obs, _, _, _, teacher, student = fixture()
    cfg = {**legacy.cfg, 'train_teacher_base': True, 'teacher_learning_rate': 3.e-6}
    terrain = {'actor': {'idle_leg_scale': .25, 'new_domain_anchor_scale': .1}}
    models = build_models(teacher, student, cfg, all_terrain_cfg=terrain)
    algorithm = CTSAlgorithm(models, cfg, 8, 4, all_terrain_cfg=terrain)
    obs['motion_control'][::2] = 0
    return models, algorithm, obs, teacher, student, terrain


def same(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected: same(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected): same(a, b)
    else:
        assert actual == expected


def test_all_teacher_policy_layers_update_and_reference_and_noise_stay_fixed():
    models, algorithm, obs, *_ = setup()
    teacher_ids = {id(p) for name, p in models.teacher.named_parameters()
                   if not name.startswith('distribution.')}
    assert teacher_ids == {id(p) for p in algorithm.optimizer.param_groups[0]['params']}
    assert all(p.requires_grad for name, p in models.teacher.named_parameters()
               if not name.startswith('distribution.'))
    assert not any(p.requires_grad for p in models.reference_teacher.parameters())
    groups = algorithm.optimizer.param_groups
    assert [(g['name'], g['lr']) for g in groups] == [
        ('teacher', algorithm.cfg['teacher_learning_rate']), ('decoder_critic', algorithm.cfg['learning_rate'])]
    all_ids = [id(p) for g in groups for p in g['params']]
    assert len(all_ids) == len(set(all_ids)) == len(algorithm.ppo_parameters)
    collect(algorithm, obs)
    before = deepcopy(models.state_dict())
    metrics = algorithm.ppo_update()
    assert metrics['ppo_minibatches'] > 0
    assert metrics['teacher_learning_rate'] == algorithm.cfg['teacher_learning_rate']
    for key, value in before.items():
        if key.startswith('teacher.mlp.'):
            assert not torch.equal(value, models.state_dict()[key]), key
        if key.startswith(('student.', 'reference_teacher.', 'teacher.distribution.')) or key == 'log_std':
            same(value, models.state_dict()[key])
    for prefix in ('teacher.residual_mlp.', 'teacher.safety_mlp.', 'teacher.idle_mlp.'):
        assert any(not torch.equal(v, models.state_dict()[k]) for k, v in before.items() if k.startswith(prefix))
    after_ppo = deepcopy(models.state_dict())
    algorithm.distill_update()
    for key, value in after_ppo.items():
        if not key.startswith('student.'): same(value, models.state_dict()[key])


def test_full_teacher_resume_restores_changed_base_and_both_learning_rates_atomically():
    models, algorithm, obs, teacher, student, terrain = setup()
    collect(algorithm, obs)
    algorithm.update()
    saved = deepcopy(algorithm.save())
    rebuilt = build_models(teacher, student, algorithm.cfg, all_terrain_cfg=terrain)
    restored = CTSAlgorithm(rebuilt, algorithm.cfg, 8, 4, all_terrain_cfg=terrain)
    restored.load(saved)
    same(restored.save(), saved)
    # A bad rate in either group or changed frozen reference must reject before mutation.
    for group in (0, 1):
        bad = deepcopy(saved)
        bad['optimizer_state_dict']['param_groups'][group]['lr'] *= 2
        before = deepcopy(restored.save())
        with pytest.raises(ValueError): restored.load(bad)
        same(restored.save(), before)
    bad = deepcopy(saved)
    key = next(k for k in bad['models_state_dict'] if k.startswith('reference_teacher.mlp.'))
    bad['models_state_dict'][key] += 1
    before = deepcopy(restored.save())
    with pytest.raises(ValueError): restored.load(bad)
    same(restored.save(), before)


def test_v15_config_remains_frozen_and_v16_requires_explicit_paired_settings():
    cfg = default_settings()
    validate_document(cfg)
    legacy = deepcopy(cfg)
    for k in ('train_teacher_base', 'teacher_learning_rate'): legacy['algorithm'].pop(k)
    validate_document(legacy, legacy=True)
    with pytest.raises(ValueError): validate_document(legacy)
    _, algorithm, *_ = fixture()
    assert len(algorithm.optimizer.param_groups) == 1
    assert not any(p.requires_grad for p in algorithm.models.teacher.mlp.parameters())
    for override in ({'train_teacher_base': 'yes'}, {'teacher_learning_rate': 0},
                     {'teacher_learning_rate': float('nan')}, {'teacher_learning_rate': True}):
        bad = {**cfg['algorithm'], **override}
        with pytest.raises(ValueError): validate_config(bad)
    for key in ('train_teacher_base', 'teacher_learning_rate'):
        bad = deepcopy(cfg['algorithm']); bad.pop(key)
        with pytest.raises(ValueError): validate_config(bad)
