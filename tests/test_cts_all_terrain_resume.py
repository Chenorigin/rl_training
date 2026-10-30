"""CPU resume checks using the real runner and real course reset.

Only simulator writes and the optimizer are fake. Metadata validation has its
own tests; this suite isolates ordering and saved RNG effects on a new episode.
"""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from task_support.cts.runner import CTSRunner
from task_support.cts import entry
from test_cts_all_terrain_course import fake, source  # noqa: F401 (shared fixture)


def test_resume_restores_rng_before_first_reset_and_repeats_first_commands(source, monkeypatch, tmp_path):
    with torch.random.fork_rng(devices=[]):
        env, course = fake(source)
        metadata = {'config': {'all_terrain': {'enabled': True}}}
        monkeypatch.setattr(entry, 'validate_metadata', lambda saved: saved)
        calls = []
        real_restore = course.restore

        def restore(saved):
            calls.append('course.restore')
            real_restore(saved)

        monkeypatch.setattr(course, 'restore', restore)

        def load_algorithm(saved):
            calls.append('alg.load')
            torch.set_rng_state(saved['torch_rng_state'])

        def reset():
            calls.append('env.reset')
            assert torch.is_inference_mode_enabled()
            env.episode_length_buf.zero_()
            course.reset(torch.arange(env.num_envs))

        runner = CTSRunner(SimpleNamespace(unwrapped=env, reset=reset),
                           SimpleNamespace(load=load_algorithm), metadata, {}, tmp_path, {})
        course.update_id = 7
        torch.manual_seed(73129)
        saved = {'cts': metadata, 'algorithm': {'num_updates': 7, 'torch_rng_state': torch.get_rng_state()},
                 'all_terrain_course': deepcopy(course.snapshot())}
        results = []
        for unrelated_caller_seed in (111, 987654):
            torch.manual_seed(unrelated_caller_seed)
            calls.clear()
            runner.load(saved)
            assert calls == ['course.restore', 'alg.load', 'env.reset']
            # Initial hold commands are always zero, so inspect the first
            # active command to avoid a vacuous deterministic-zero assertion.
            hold = course.cfg['commands']['initial_hold_s']
            env.episode_length_buf.fill_(round(hold/env.step_dt)+1)
            results.append((course.raw_command().clone(), course.tile.clone(),
                            env.scene.robot.data.root_pos_w.clone(),
                            env.scene.robot.data.root_quat_w.clone()))
        assert results[0][0].abs().sum() > 0
        for first, second in zip(*results):
            torch.testing.assert_close(first, second, rtol=0, atol=0)

        # Counterexample: changing only the checkpoint's Torch RNG must change
        # real first command draws, while its private course RNG retains tiles.
        different_rng = deepcopy(saved)
        torch.manual_seed(73130)
        different_rng['algorithm']['torch_rng_state'] = torch.get_rng_state()
        runner.load(different_rng)
        env.episode_length_buf.fill_(round(hold/env.step_dt)+1)
        assert not torch.equal(results[0][0], course.raw_command())
        torch.testing.assert_close(results[0][1], course.tile, rtol=0, atol=0)


def test_resume_iteration_mismatch_fails_before_restore_or_reset(source, monkeypatch, tmp_path):
    env, course = fake(source)
    metadata = {'config': {'all_terrain': {'enabled': True}}}
    monkeypatch.setattr(entry, 'validate_metadata', lambda saved: saved)
    calls = []
    monkeypatch.setattr(course, 'restore', lambda saved: calls.append('course.restore'))
    runner = CTSRunner(SimpleNamespace(unwrapped=env, reset=lambda: calls.append('env.reset')),
                       SimpleNamespace(load=lambda saved: calls.append('alg.load')), metadata, {}, tmp_path, {})
    saved = {'cts': metadata, 'algorithm': {'num_updates': 2}, 'all_terrain_course': {'update_id': 1}}
    with pytest.raises(ValueError, match='iteration/course'):
        runner.load(saved)
    assert calls == []
