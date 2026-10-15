"""Exercise manual v1 handover with mocked physics/viewer, on CPU only."""
from contextlib import nullcontext
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'deploy/deploy_mujoco'))
import deploy_student_v1 as entry
from student_v1_adapter import StudentMapAdapter, SensorExtras, profile_config
from m20_student_v1.perception import grid_points


class HandoverTests(unittest.TestCase):
    def adapter(self):
        points = grid_points('cpu').numpy()
        proprio = np.zeros(57, np.float32)
        proprio[5] = -1.
        def observe(data, command, previous):
            p = proprio.copy()
            p[6:9] = command
            p[41:57] = previous
            return np.concatenate((p, np.zeros(187, np.float32)))
        contract = SimpleNamespace(base_id=0, qpos_ids=np.arange(16),
                                   last_scan_hits=np.column_stack((points, np.zeros(187))),
                                   observe=observe, ground_height=lambda *args: 0.,
                                   apply_pose_pd=Mock(), apply_damping=Mock(), apply_pd=Mock(),
                                   crossing_summary=lambda *args: {})
        cfg, _ = profile_config('clean', {'contract': {'history_steps': 4}})
        adapter = StudentMapAdapter(contract, cfg, SensorExtras(1, 4))
        data = SimpleNamespace(time=0., xpos=np.array([[0., 0., .5]]),
                               xmat=np.eye(3).reshape(1, 9), qpos=np.zeros(16), qvel=np.zeros(16))
        return adapter, contract, data

    def exercise_handover(self, force_fall=False):
        engine = entry.runtime()
        adapter, contract, data = self.adapter()
        model = SimpleNamespace(opt=SimpleNamespace(timestep=.02))
        args = SimpleNamespace(output_dir=None, snapshot=None, steps=8, continue_after_fall=False,
                               model=Path('/tmp/model.xml'), noise_profile='clean', seed=42,
                               friction_scale=1., terrain_config=SimpleNamespace(summary=lambda: {}))
        # Queued C while standing -> policy; C while in policy -> retain history;
        # R -> damping -> Z -> ready -> C -> a fresh policy observation.
        events = iter([['z'], ['c'], ['c'], [], ['r'], ['z'], [], ['c']])
        keys = Mock()
        keys.take_events.side_effect = lambda: next(events)
        keys.velocity_command.return_value = np.zeros(3, np.float32)
        keys._window_keys = object()
        terminal = Mock(); terminal.start.return_value = False
        viewer = SimpleNamespace(lock=nullcontext, cam=SimpleNamespace(), is_running=lambda: True, sync=lambda: None)
        samples = []
        def policy(obs):
            samples.append(obs.numpy().copy())
            return torch.ones(1, 16)
        def step(*args):
            data.time += .02
            if force_fall and .079 < data.time < .081:
                data.xpos[0, 2] = .1
        contract.apply_pose_pd.side_effect = lambda *args: data.xpos.__setitem__((0, 2), .5)
        torch.set_num_threads(1)
        with patch.object(engine, 'create_simulation', return_value=(model, contract, data, adapter, adapter.extras)), patch.object(engine, 'prepare_output'), patch.object(engine, 'sha256', return_value='unused'), patch.object(engine, 'draw_student_map'), patch.object(engine.baseline, 'TeleopKeyboard', return_value=keys), patch.object(engine.baseline, 'TerminalInput', return_value=terminal), patch.object(engine.baseline, 'launch_viewer', return_value=(viewer, [])), patch.object(engine.baseline, 'close_viewer'), patch.object(engine.baseline, 'STAND_PHASE_DURATION', .009), patch.object(engine.time, 'sleep'), patch.object(engine.mujoco, 'mj_step', side_effect=step), patch.object(engine.mujoco, 'mj_forward'), patch('sys.stdout', new_callable=io.StringIO):
            result = engine.interactive_rollout(args, policy, {}, {})
        self.assertEqual(result['student_policy_steps'], 4)
        self.assertEqual(result['fell'], force_fall)
        self.assertEqual(len(samples), 4)
        for index in (0, 3):
            self.assertTrue(np.array_equal(samples[index][0, 618:], np.zeros(171)))
            self.assertTrue(np.array_equal(samples[index][0, 41:57], np.zeros(16)))
        self.assertGreater(np.abs(samples[1][0, 618:]).sum(), 0.)
        # The first handover action is zero because of the existing 0.3 s blend.
        np.testing.assert_allclose(samples[2][0, 41:57], np.full(16, .02/.3), rtol=1e-6)
        # Each re-entry receives a zero delayed action before the new queue fills.
        self.assertEqual(contract.apply_pd.call_count, 4)
        self.assertTrue(np.array_equal(contract.apply_pd.call_args_list[0].args[1], np.zeros(16)))
        np.testing.assert_allclose(contract.apply_pd.call_args_list[2].args[1], np.full(16, .02/.3), rtol=1e-6)
        self.assertTrue(np.array_equal(contract.apply_pd.call_args_list[3].args[1], np.zeros(16)))

    def test_standing_handover_and_reentry_clear_history_and_action_delay(self):
        self.exercise_handover()

    def test_fall_recovery_handover_starts_a_fresh_history(self):
        self.exercise_handover(force_fall=True)

    def test_reset_rejects_prior_observation_and_pending_map_packets(self):
        adapter, _, data = self.adapter()
        for step in range(5):
            old = adapter.observe(data, np.zeros(3), np.ones(16), step)
        self.assertGreater(np.abs(old[618:]).sum(), 0.)
        self.assertTrue(adapter.map.packet_valid.any())
        adapter.reset_for_policy_handover()
        self.assertFalse(adapter.map.packet_valid.any())
        self.assertEqual(adapter.map.history.abs().sum(), 0.)
        self.assertIsNone(adapter.last_channels)
        self.assertFalse(adapter.last_visibility.any())
        self.assertFalse(adapter.last_blackout)
        # Even at the same frame index the cached old observation must not return.
        fresh = adapter.observe(data, np.zeros(3), np.zeros(16), 4)
        self.assertTrue(np.array_equal(fresh[618:], np.zeros(171)))
        self.assertTrue(np.array_equal(fresh[41:57], np.zeros(16)))
        self.assertTrue(np.isfinite(fresh).all())


if __name__ == '__main__':
    unittest.main()
