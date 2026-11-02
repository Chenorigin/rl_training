"""All-terrain contracts and deployment architecture selection without Isaac."""
import ast
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from task_support.cts.all_terrain_config import validate_all_terrain
from task_support.cts.all_terrain_curriculum import CourseCurriculum
from task_support.cts.all_terrain_geometry import DEFAULTS

ROOT = Path(__file__).resolve().parents[1]


def config():
    cfg = json.loads((ROOT / 'configs/distillation/cts.json').read_text())['all_terrain']
    # Geometry defaults have one owner; fixtures also work while JSON is being
    # upgraded by the implementation's separate integration owner.
    cfg['geometry'] = {**deepcopy(DEFAULTS), **cfg['geometry']}
    return cfg


def numeric_paths(value, prefix=()):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from numeric_paths(child, (*prefix, key))
    elif isinstance(value, list):
        for key, child in enumerate(value):
            yield from numeric_paths(child, (*prefix, key))
    elif type(value) in (int, float):
        yield prefix


def put(cfg, path, value):
    target = cfg
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value


def test_complete_configuration_constructs_all_wall_and_legacy_curriculum_groups():
    cfg = config()
    assert validate_all_terrain(cfg) is cfg
    assert set(cfg['geometry']) == set(DEFAULTS)
    assert sum(cfg['fractions'][:3]) == pytest.approx(.65)
    curriculum = CourseCurriculum(cfg['curriculum'], num_courses=140)
    curriculum.restore(curriculum.snapshot())


@pytest.mark.parametrize('path', list(numeric_paths(config())))
@pytest.mark.parametrize('invalid', [float('nan'), float('inf'), True])
def test_every_numeric_setting_rejects_nonfinite_and_boolean_values(path, invalid):
    cfg = config()
    put(cfg, path, invalid)
    with pytest.raises(ValueError):
        validate_all_terrain(cfg)


@pytest.mark.parametrize('section', ['geometry', 'curriculum', 'domain', 'commands', 'rewards', 'actor', 'simulation'])
@pytest.mark.parametrize('kind', ['missing', 'extra', 'wrong_type'])
def test_each_nested_contract_rejects_unknown_missing_or_malformed_fields(section, kind):
    cfg = config()
    if kind == 'missing':
        cfg[section].pop(next(iter(cfg[section])))
    elif kind == 'extra':
        cfg[section]['unwired_typo'] = .1
    else:
        cfg[section] = []
    with pytest.raises(ValueError):
        validate_all_terrain(cfg)


@pytest.mark.parametrize('path,value', [
    (('enabled',), 1), (('fractions',), [.1] * 9),
    (('domain', 'dynamic_ratio'), [.8, 1.1]),
    (('domain', 'mass_scale'), [1.1, .9]),
    (('domain', 'initial_scale'), 1.1),
    (('domain', 'com_offset_m'), -.01),
    (('geometry', 'geometry_levels'), 8),
    (('geometry', 'steps_range'), [8.5, 20]),
    (('geometry', 'initial_steps_max'), 30),
    (('geometry', 'height_range'), [.15, .15]),
    (('geometry', 'width_range'), [1., 7.]),
    (('geometry', 'start_backward'), 10.),
    (('geometry', 'pit_x_range'), [1.6, 8.]),
    (('geometry', 'obstacle_x_range'), [.9, 6.5]),
    (('geometry', 'mixed_obstacle_x_range'), [2., 6.5]),
    (('commands', 'relative_tracking_tolerance'), 1.),
    (('commands', 'terrain_exposure_m'), 8.),
    (('commands', 'success_hold_s'), 2.),
    (('commands', 'backward_speed'), [-.2, .1]),
    (('rewards', 'flat_clearance_scale_m'), 0.),
    (('rewards', 'min_upright'), 1.1),
    (('rewards', 'boundary_margin'), 3.),
    (('rewards', 'slip'), -.1),
    (('actor', 'idle_leg_scale'), 0.),
    (('actor', 'new_domain_anchor_scale'), 1.1),
    (('simulation', 'collision_stack_mib'), 63),
    (('simulation', 'collision_stack_mib'), 2049),
    (('simulation', 'collision_stack_mib'), 512.),
])
def test_invalid_physical_intervals_and_unreachable_course_contracts_fail(path, value):
    cfg = config()
    put(cfg, path, value)
    with pytest.raises(ValueError):
        validate_all_terrain(cfg)


def test_valid_ablation_and_different_continuous_intervals_are_not_overconstrained():
    cfg = config()
    cfg['domain']['mass_scale'] = [1., 1.]
    cfg['domain']['com_offset_m'] = 0.
    cfg['domain']['initial_scale'] = 0.
    cfg['actor']['new_domain_anchor_scale'] = 0.
    cfg['rewards']['slip'] = cfg['rewards']['flat_air_cost'] = 0.
    cfg['geometry']['height_range'] = [.16, .24]
    cfg['geometry']['samples_per_level'] = 3
    cfg['geometry']['geometry_levels'] = cfg['curriculum']['geometry_levels'] = 1
    assert validate_all_terrain(cfg) is cfg


@pytest.mark.parametrize('stack_mib', [None, 64, 512, 2048])
def test_preflight_counts_collision_stack_outside_baseline_and_reports_exact_bytes(stack_mib):
    # Execute the actual preflight accounting statements with other measured
    # costs set to zero. No simulator, model, retention dataset or GPU allocation.
    path = ROOT / 'source/rl_training/task_support/cts/entry.py'
    function = next(node for node in ast.parse(path.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == 'preflight')
    assignments = {node.targets[0].id: node for node in function.body
                   if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
    report = deepcopy(assignments['report'])
    fields = {'planned_increment_bytes', 'collision_stack_reserve_bytes'}
    selected = [(key, value) for key, value in zip(report.value.keys, report.value.values)
                if isinstance(key, ast.Constant) and key.value in fields]
    report.value = ast.Dict(keys=[key for key, _ in selected], values=[value for _, value in selected])
    module = ast.Module(body=[deepcopy(assignments['collision_stack_bytes']),
                             deepcopy(assignments['allowance']), report], type_ignores=[])
    cfg = config()
    all_terrain = None
    if stack_mib is not None:
        cfg['simulation']['collision_stack_mib'] = stack_mib
        validate_all_terrain(cfg)
        all_terrain = cfg
    namespace = dict(all_terrain=all_terrain, n=0, rollout=0, model_reserve=0,
                     mesh_bytes=0, auxiliary_budget={'auxiliary_reserve_bytes': 0},
                     capture_budget={'extra_capture_reserve_bytes': 0},
                     straight_target_bytes=0, course_tag_bytes=0, recovery_bytes=0, surface_bytes=0)
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
    expected = 0 if stack_mib is None else stack_mib * 1024**2
    assert namespace['report'] == dict(planned_increment_bytes=8 * 1024**3 + expected,
                                     collision_stack_reserve_bytes=expected)


@pytest.mark.parametrize('variant', ['enabled', 'disabled', 'legacy'])
def test_actual_deployment_loader_selects_saved_architecture_without_simulator_import(variant):
    # Compile the real loader and actor only. Deployment's viewer/physics imports
    # are irrelevant to checkpoint architecture wiring and are not executed.
    path = ROOT / 'deploy/deploy_mujoco/deploy_cts_student.py'
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                 and node.name in ('CTSStudentActor', 'load_policy')]
    all_terrain = config()
    all_terrain['enabled'] = variant == 'enabled'
    bounds = [1.] * 16
    cfg = {'algorithm': {'decoder_bound': bounds}}
    if variant != 'legacy':
        cfg['all_terrain'] = all_terrain
    meta = dict(student_inputs={'policy': 789, 'motion_control': 3}, config=cfg,
                command_feedback={'version': 'legacy_v1_v3_5', 'tuning': {}},
                teacher_manifest={'stair_tuning': {}}, version='test_version')
    saved = dict(cts=meta, initialization={'teacher': {}, 'student': {}},
                 algorithm=dict(cts_version='test_algorithm', algorithm_config=cfg['algorithm'],
                                models_state_dict={'weight': torch.zeros(1)}))

    class Export(torch.nn.Module):
        def forward(self, obs, command):
            return obs.new_zeros(obs.shape[0], 16)

    models = SimpleNamespace(load_state_dict=Mock(), decoder=SimpleNamespace(bounds=torch.tensor(bounds)),
                             student_export=lambda: Export())
    build = Mock(return_value=models)
    namespace = dict(torch=torch, Path=Path, build_models=build,
                     validate_metadata=lambda value, **_: value,
                     execution_contract=lambda *_, **__: {}, algorithm_version=lambda _: 'test_algorithm',
                     LEGACY_VERSION='old')
    exec(compile(tree, str(path), 'exec'), namespace)
    with patch.object(torch, 'load', return_value=saved):
        actor, _ = namespace['load_policy']('unused.pt')
    expected = all_terrain if variant == 'enabled' else None
    assert build.call_args.kwargs['all_terrain_cfg'] == expected
    assert actor.student_provenance == meta
    models.load_state_dict.assert_called_once_with(saved['algorithm']['models_state_dict'], strict=True)
