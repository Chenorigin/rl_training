"""Autoplay must sanitize native viewer resets before the next control tick.

Five bounded ticks per case, with physics and rendering replaced by assertions.
The real CPU sensor adapters exercise packet/history and delayed-action state.
"""
from contextlib import nullcontext
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy/deploy_mujoco"))
import deploy_mujoco as baseline
import deploy_student_v0 as student
from student_v0_adapter import StudentMapAdapter as V0Adapter, profile_config as v0_config
from student_v1_adapter import StudentMapAdapter as V1Adapter, profile_config as v1_config
from teacher_noise_adapter import TeacherNoiseAdapter


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    data = SimpleNamespace(time=0., xpos=np.array([[0., 0., .5]]),
        xmat=np.eye(3).reshape(1, 9), xquat=np.array([[1., 0., 0., 0.]]),
        qpos=np.zeros(16), qvel=np.zeros(16), ctrl=np.zeros(16))
    model = SimpleNamespace(opt=SimpleNamespace(timestep=baseline.POLICY_DT),
        body_mass=np.array([10.]), nu=16, ngeom=0)
    previous, applied, physics = [], [], []

    def reset(state, prone=False):
        assert not prone  # Fixed-command playback restarts in its standing pose.
        state.time = 0.
        state.qpos.fill(0.)
        state.qvel.fill(0.)
        state.ctrl.fill(0.)

    def observe(state, command, previous_action):
        assert not np.any(state.qpos == 99.)
        previous.append(previous_action.copy())
        observation = np.zeros(244, dtype=np.float32)
        observation[5] = -1.
        observation[6:9], observation[41:57] = command, previous_action
        return observation

    contract = SimpleNamespace(base_id=0, qpos_ids=np.arange(16),
        last_scan_hits=np.column_stack((baseline.scan_grid(), np.zeros(187))),
        observe=observe, reset=Mock(side_effect=reset),
        ground_height=lambda *args: 0., crossing_summary=lambda *args: {},
        apply_pd=lambda state, action: applied.append(action.copy()))

    def integrate(*args):
        assert not np.any(data.qpos == 99.), "XML reset pose reached physics"
        physics.append(data.qpos.copy())
        data.time += baseline.POLICY_DT

    sync_count = 0

    def sync():
        nonlocal sync_count
        sync_count += 1
        if sync_count == 3:
            data.time = 0.
            data.qpos.fill(99.)  # Simulate pending native Backspace/UI reset.

    viewer = SimpleNamespace(lock=nullcontext, cam=SimpleNamespace(),
        is_running=lambda: True, sync=sync)
    terrain = SimpleNamespace(summary=lambda: {}, course_width=10.)
    args = SimpleNamespace(checkpoint=tmp_path / "unused.pt", model=tmp_path / "unused.xml",
        terrain_config=terrain, viewer=True, autoplay=True, steps=5,
        command_x=.4, command_y=.2, command_yaw=.1, print_every=100,
        output=None, output_dir=None, snapshot=None, seed=42, noise_profile="clean",
        quiet_landing=False, track_course=False, stop_on_fall=False,
        continue_after_fall=False, friction_scale=1., warmup_s=0.)
    monkeypatch.setattr(baseline, "PHYSICS_DT", baseline.POLICY_DT)
    monkeypatch.setattr(baseline, "load_model", lambda *args: model)
    monkeypatch.setattr(baseline, "M20Contract", lambda *args: contract)
    monkeypatch.setattr(baseline, "launch_viewer", lambda *args, **kwargs: (viewer, []))
    monkeypatch.setattr(baseline, "close_viewer", Mock())
    monkeypatch.setattr(baseline, "draw_height_scan", Mock())
    monkeypatch.setattr(baseline, "make_trot_controller", lambda *args: None)
    monkeypatch.setattr(baseline, "heading_controller", lambda *args: None)
    monkeypatch.setattr(baseline, "noise_metadata", lambda *args: {})
    monkeypatch.setattr(baseline.time, "sleep", Mock())
    monkeypatch.setattr(baseline.mujoco, "MjData", lambda *args: data)
    monkeypatch.setattr(baseline.mujoco, "mj_step", integrate)
    for name in ("mj_forward", "mj_fwdPosition", "mj_fwdVelocity", "mj_objectVelocity"):
        monkeypatch.setattr(baseline.mujoco, name, Mock())
    return SimpleNamespace(args=args, model=model, contract=contract, data=data,
        previous=previous, applied=applied, physics=physics)


def sensor_factory(version, contract, snapshots, adapters):
    """Record the real adapter's state immediately before each observation."""
    Adapter, config = (V0Adapter, v0_config) if version == 0 else (V1Adapter, v1_config)
    cfg, extras = config("clean", {"contract": {"history_steps": 4}})
    cfg.map_update_period_s = .02
    cfg.max_delay_s = .04
    extras.action_delay_steps = 2
    if version == 1:
        cfg.min_delay_s = .04
        cfg.action_delay_steps = 2
        cfg.update_jitter_s = 0.

    def make(contract, cfg=cfg, extras=extras, seed=42):
        adapter = Adapter(contract, cfg, extras, seed)
        original = adapter.observe

        def observe(data, command, previous_action, step):
            queue = adapter.map.queue_capture if version == 0 else adapter.map.packet_capture
            snapshots.append({"captures": queue.clone(),
                "history": adapter.map.history.clone() if version == 1 else None})
            return original(data, command, previous_action, step)

        adapter.observe = observe
        adapters.append(adapter)
        return adapter

    return make, cfg, extras


def assert_restart(runtime, snapshots):
    assert len(runtime.physics) == len(runtime.previous) == len(snapshots) == 5
    assert runtime.contract.reset.call_count == 2  # Startup plus native reset.
    np.testing.assert_array_equal(runtime.previous[2], 1.)
    np.testing.assert_array_equal(runtime.previous[3], 0.)
    # A two-tick action delay must begin empty again after the native reset.
    np.testing.assert_array_equal(runtime.applied, np.array([0., 0., 1., 0., 0.])[:, None] * np.ones((1, 16)))
    assert torch.any(snapshots[2]["captures"] >= 0), "Pre-reset sensor state was never populated"
    assert torch.all(snapshots[3]["captures"] == -1), "Old sensor packets survived reset"
    if snapshots[3]["history"] is not None:
        assert torch.any(snapshots[2]["history"] != 0)
        assert torch.all(snapshots[3]["history"] == 0)


def test_teacher_autoplay_native_reset_clears_pose_actions_and_sensor_history(monkeypatch, runtime):
    snapshots, adapters = [], []
    make, cfg, _ = sensor_factory(1, runtime.contract, snapshots, adapters)
    # Use the real teacher delay/reset implementation with the real v1 map.
    perception = TeacherNoiseAdapter.__new__(TeacherNoiseAdapter)
    perception.map_adapter = make(runtime.contract)
    perception.cfg = cfg
    perception.steps = perception.valid_samples = 0
    perception.reset()
    monkeypatch.setattr(perception, "draw", Mock())
    monkeypatch.setattr(baseline, "make_noise_adapter", lambda *args: perception)
    monkeypatch.setattr(baseline, "load_actor", lambda *args: lambda obs: torch.ones(1, 16))
    monkeypatch.setattr(baseline, "policy_inputs", lambda observation, *args: observation)
    result = baseline.rollout(runtime.args)
    assert result["steps_executed"] == 5
    assert_restart(runtime, snapshots)


@pytest.mark.parametrize("version", [0, 1])
def test_student_autoplay_native_reset_clears_pose_actions_and_sensor_history(monkeypatch, runtime, version):
    snapshots, adapters, saved = [], [], {}
    make, cfg, extras = sensor_factory(version, runtime.contract, snapshots, adapters)
    adapter = make(runtime.contract)

    def create(*args):
        runtime.contract.reset(runtime.data)
        return runtime.model, runtime.contract, runtime.data, adapter, extras

    monkeypatch.setattr(student, "create_simulation", create)
    monkeypatch.setattr(student, "StudentMapAdapter", make)
    monkeypatch.setattr(student, "sha256", lambda *args: "unused")
    for name in ("prepare_output", "draw_student_map", "write_json"):
        monkeypatch.setattr(student, name, Mock())
    runtime.args.output_dir = runtime.args.model.parent / "output"
    monkeypatch.setattr(student.np, "savez_compressed", lambda path, **values: saved.update(values))
    result = student.rollout(runtime.args, lambda observation: torch.ones(1, 16), {}, {})
    assert result["control_steps"] == 5
    assert_restart(runtime, snapshots)
    assert len(adapters) == (2 if version == 0 else 1)
    # Reset must not mutate already-recorded nominal actions through an alias.
    np.testing.assert_array_equal(saved["action_issued"], np.ones((5, 16)))
