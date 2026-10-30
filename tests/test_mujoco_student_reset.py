"""Native viewer and explicit reset must clear manual student episode state."""
from contextlib import nullcontext
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'deploy/deploy_mujoco'))
import deploy_student_v0 as engine


@pytest.mark.parametrize('native', [False, True])
@pytest.mark.parametrize('version', [0, 1])
def test_student_reset_clears_sensors_and_restarts_prone(monkeypatch, tmp_path, native, version):
    from student_v0_adapter import StudentMapAdapter as V0, SensorExtras as Extras0, profile_config as config0
    from student_v1_adapter import StudentMapAdapter as V1, SensorExtras as Extras1, profile_config as config1
    Adapter, config, Extras = (V0, config0, Extras0) if version == 0 else (V1, config1, Extras1)
    grid = engine.baseline.scan_grid()
    observed_previous = []
    def observe(data, command, previous):
        observed_previous.append(previous.copy())
        obs = np.zeros(244, dtype=np.float32)
        obs[5] = -1.
        obs[6:9], obs[41:57] = command, previous
        return obs
    data = SimpleNamespace(time=0., xpos=np.array([[0., 0., .5]]), xmat=np.eye(3).reshape(1, 9),
                           qpos=np.zeros(16), qvel=np.zeros(16))
    def reset(data, prone=False):
        assert prone
        data.time = 0.
        data.qpos[:] = 0.
    contract = SimpleNamespace(base_id=0, qpos_ids=np.arange(16),
        last_scan_hits=np.column_stack((grid, np.zeros(187))), observe=observe,
        ground_height=lambda *a: 0., apply_pose_pd=Mock(), apply_damping=Mock(), apply_pd=Mock(),
        crossing_summary=lambda *a: {}, reset=Mock(side_effect=reset))
    cfg, extras = config('clean', {'contract': {'history_steps': 4}})
    adapter = Adapter(contract, cfg, extras, 42)
    reset_sensor = Mock(wraps=adapter.reset) if version == 1 else None
    if reset_sensor is not None:
        adapter.reset = reset_sensor
    made = []
    def make(*args):
        made.append(Adapter(*args))
        return made[-1]
    monkeypatch.setattr(engine, 'StudentMapAdapter', make)
    model = SimpleNamespace(opt=SimpleNamespace(timestep=.02))
    args = SimpleNamespace(output_dir=None, snapshot=None, steps=5, continue_after_fall=False,
        model=tmp_path/'robot.xml', noise_profile='clean', seed=42, friction_scale=1.,
        terrain_config=SimpleNamespace(summary=lambda: {}))
    events = iter([['z'], ['c'], [], [] if native else ['reset'], []])
    keys = Mock()
    keys.take_events.side_effect = lambda: next(events)
    keys.velocity_command.return_value = np.zeros(3, np.float32)
    terminal = Mock(); terminal.start.return_value = False
    sync_count = 0
    def sync():
        nonlocal sync_count
        sync_count += 1
        if native and sync_count == 3:
            data.time = 0.
            data.qpos[:] = 99.
    viewer = SimpleNamespace(lock=nullcontext, cam=SimpleNamespace(), is_running=lambda: True, sync=sync)
    def step(*args):
        assert not np.any(data.qpos == 99.)
        data.time += .02
    monkeypatch.setattr(engine, 'create_simulation', lambda *a, **k: (model, contract, data, adapter, extras))
    for name in ('prepare_output', 'draw_student_map'):
        monkeypatch.setattr(engine, name, Mock())
    monkeypatch.setattr(engine, 'sha256', lambda *a: 'unused')
    monkeypatch.setattr(engine.baseline, 'TeleopKeyboard', lambda: keys)
    monkeypatch.setattr(engine.baseline, 'TerminalInput', lambda *a: terminal)
    monkeypatch.setattr(engine.baseline, 'launch_viewer', lambda *a, **k: (viewer, []))
    monkeypatch.setattr(engine.baseline, 'close_viewer', Mock())
    monkeypatch.setattr(engine.baseline, 'STAND_PHASE_DURATION', .009)
    monkeypatch.setattr(engine.time, 'sleep', Mock())
    monkeypatch.setattr(engine.mujoco, 'mj_step', step)
    monkeypatch.setattr(engine.mujoco, 'mj_forward', Mock())
    result = engine.interactive_rollout(args, lambda obs: torch.ones(1, 16), {}, {})
    contract.reset.assert_called_once_with(data, prone=True)
    assert result['final_mode'] == 'prone'
    assert result['student_policy_steps'] == 2
    np.testing.assert_array_equal(observed_previous[-1], 0.)
    if version == 0:
        assert len(made) == 1 and made[0] is not adapter
    else:
        assert reset_sensor.call_count >= 2  # Handover, then reset.
