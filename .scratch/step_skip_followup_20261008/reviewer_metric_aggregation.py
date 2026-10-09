#!/usr/bin/env python3
"""Independent CPU-only verification of newly added stair event ratios."""
import ast
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/tools'))
import torch
from check_gait_safety import load
from check_stair_rewards import make_env

SOURCE = ROOT / 'source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/stair_teacher.py'
BEFORE = Path(__file__).with_name('before_stair_teacher.py')
CONFIG = ROOT / 'source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/stair_teacher_env_cfg.py'
parser = argparse.ArgumentParser()
parser.add_argument('--allow-transition-payment-change', action='store_true')
parser.add_argument('--output', type=Path)
args = parser.parse_args()
torch.set_num_threads(1)
ns = load(SOURCE)


def serial(metrics):
    return {k: float(v) for k, v in metrics.items()
            if k.startswith(('up_front_skip', 'up_rear_skip', 'up_front_same_tread_fraction',
                             'up_rear_same_tread_fraction', 'up_front_visited', 'up_rear_visited'))}


def sample_env(n):
    env = make_env()
    env.num_envs = n
    env.episode_length_buf = torch.full((n,), 2, dtype=torch.long)
    env._m20_ascent_state = ns['_new_state'](env)
    return env, env._m20_ascent_state


def assert_ratios(metrics):
    for name in ('front', 'rear'):
        for kind in ('skip', 'same_tread'):
            value = metrics[f'up_{name}_{kind}_fraction']
            assert torch.isfinite(value) and 0 <= value <= 1, (name, kind, value)


env, state = sample_env(3)
# env0: two visited intermediate treads, both duplicated, one skip event.
# env1: eight visited intermediate treads, no duplicates and no skip event.
# env2: no observed geometry/landing. Each nonempty map also includes future
# unvisited intermediate treads and an unvisited final platform.
for i, (known, visited, duplicates, skips) in enumerate(((5, 2, 2, 1), (12, 8, 0, 0))):
    state['map_count'][i] = known
    state['map_intermediate'][i, :known - 1] = True
    state['map_flight'][i, :known] = 1
    state['map_z'][i, :known] = torch.arange(1, known + 1) * .1
    for axle in (0, 1):
        state['map_paid'][i, :visited, axle] = True
        state['map_correct'][i, :visited, axle] = True
        state['map_duplicate_paid'][i, :duplicates, axle] = True
        state['physical_total'][i, axle] = visited
        state['skip_total'][i, axle] = skips

metrics = ns['stair_step_completion_metrics'](env, slice(None))
assert_ratios(metrics)
for name in ('front', 'rear'):
    assert abs(float(metrics[f'up_{name}_skip_fraction']) - .1) < 1e-7
    assert abs(float(metrics[f'up_{name}_same_tread_fraction']) - .2) < 1e-7
    assert metrics[f'up_{name}_visited_intermediate_risers'] == 10
    assert metrics[f'up_{name}_registered_intermediate_risers'] == 15
    assert abs(float(metrics[f'up_{name}_landing_coverage']) - 10 / 17) < 1e-7
out = {'event_weighted_batch': serial(metrics)}

# Independently indexed subset: an empty environment must not dilute event
# ratios, despite per-environment count means intentionally having a mean.
subset = ns['stair_step_completion_metrics'](env, torch.tensor([0, 2]))
assert_ratios(subset)
for name in ('front', 'rear'):
    assert subset[f'up_{name}_skip_fraction'] == .5
    assert subset[f'up_{name}_same_tread_fraction'] == 1
    assert subset[f'up_{name}_visited_intermediate_risers'] == 2
out['indexed_subset_with_empty_env'] = serial(subset)

# Visiting the final platform adds a physical landing, but no intermediate
# tread to the same-tread denominator. This is the intended exemption.
state['map_paid'][0, 4, :] = True
state['map_correct'][0, 4, :] = True
state['physical_total'][0, :] += 1
platform = ns['stair_step_completion_metrics'](env, slice(None))
assert_ratios(platform)
for name in ('front', 'rear'):
    assert abs(float(platform[f'up_{name}_skip_fraction']) - 1 / 11) < 1e-7
    assert abs(float(platform[f'up_{name}_same_tread_fraction']) - .2) < 1e-7
    assert platform[f'up_{name}_visited_intermediate_risers'] == 10
out['final_platform_denominator'] = serial(platform)

zero_env, _ = sample_env(4)
empty = ns['stair_step_completion_metrics'](zero_env, slice(None))
assert_ratios(empty)
for name in ('front', 'rear'):
    assert empty[f'up_{name}_skip_fraction'] == 0
    assert empty[f'up_{name}_same_tread_fraction'] == 0
    assert empty[f'up_{name}_visited_intermediate_risers'] == 0
out['allocated_no_samples'] = serial(empty)

# Ratio bounds rely on the real ledger invariant: a skip is a subset of first
# physical landings, and duplicate treads are a subset of visited treads.
# Exercise valid ledgers spanning both boundaries on heterogeneous batches.
generator = torch.Generator().manual_seed(381)
for _ in range(100):
    e, s = sample_env(7)
    for i in range(7):
        known = int(torch.randint(1, 25, (), generator=generator))
        s['map_count'][i] = known
        s['map_intermediate'][i, :known - 1] = True
        s['map_flight'][i, :known] = 1
        s['map_z'][i, :known] = torch.arange(known) * .1
        for axle in (0, 1):
            visited = int(torch.randint(0, known + 1, (), generator=generator))
            dup = int(torch.randint(0, min(visited, known - 1) + 1, (), generator=generator))
            skip = int(torch.randint(0, visited + 1, (), generator=generator))
            s['map_paid'][i, :visited, axle] = True
            s['map_correct'][i, :visited, axle] = True
            s['map_duplicate_paid'][i, :dup, axle] = True
            s['physical_total'][i, axle] = visited
            s['skip_total'][i, axle] = skip
    m = ns['stair_step_completion_metrics'](e, slice(None))
    assert_ratios(m)
    for axle, name in enumerate(('front', 'rear')):
        count = s['physical_total'][:, axle].sum()
        expected = s['skip_total'][:, axle].sum() / count.clamp_min(1)
        assert torch.equal(m[f'up_{name}_skip_fraction'], expected)
        visited = s['map_paid'][:, :, axle] & s['map_intermediate']
        expected = (s['map_duplicate_paid'][:, :, axle] & visited).sum() / visited.sum().clamp_min(1)
        assert torch.equal(m[f'up_{name}_same_tread_fraction'], expected)
out['valid_randomized_batch_cases'] = 100


def function_ast(path):
    tree = ast.parse(path.read_text())
    return {node.name: ast.dump(node, include_attributes=False)
            for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}

old, new = function_ast(BEFORE), function_ast(SOURCE)
changed = sorted(k for k in old.keys() | new.keys() if old.get(k) != new.get(k))
expected_changes = ['_new_state', 'ascent_state', 'stair_step_completion_metrics'] if args.allow_transition_payment_change else ['stair_step_completion_metrics']
assert changed == expected_changes, changed
old_module = ast.parse(BEFORE.read_text())
new_module = ast.parse(SOURCE.read_text())
for tree in (old_module, new_module):
    tree.body = [node for node in tree.body
                 if not isinstance(node, ast.FunctionDef) or node.name not in expected_changes]
assert ast.dump(old_module, include_attributes=False) == ast.dump(new_module, include_attributes=False)
before_ns = load(BEFORE)
assert before_ns['TUNING'] == ns['TUNING']
out['production_functions_changed'] = changed
out['reward_tuning_unchanged'] = True
out['whole_module_except_expected_changes_ast_unchanged'] = True
out['source_sha256'] = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
out['config_sha256'] = hashlib.sha256(CONFIG.read_bytes()).hexdigest()
out['reviewer_family'] = 'Codex / GPT; same family as implementer, independent execution'
(args.output or Path(__file__).with_suffix('.json')).write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps(out, indent=2))
