"""Bounded CPU tests for genuine observed terrain history; no physics."""
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source/rl_training"))
from task_support.stair_quiet_v1.landing import LandingSettings
from task_support.stair_quiet_v1.terrain_memory import TerrainMemory


def make(n=1, m=1, **settings):
    return TerrainMemory(n, m, "cpu", LandingSettings(**settings))


def update(memory, points, t=0., age=0., valid=True, pose=None, yaw=None):
    n, m = memory.num_envs, memory.num_points
    points = torch.as_tensor(points, dtype=torch.float32).reshape(n, m, 3)
    ages = torch.as_tensor(age, dtype=torch.float32).expand(n, m).clone()
    mask = torch.as_tensor(valid, dtype=torch.bool).expand(n, m).clone()
    memory.update(points_w=points, valid=mask, age_s=ages,
                  pose_w=torch.zeros(n, 3) if pose is None else torch.tensor(pose, dtype=torch.float32).reshape(n, 3),
                  yaw_w=torch.zeros(n) if yaw is None else torch.tensor(yaw, dtype=torch.float32).reshape(n),
                  time_s=t)


def sample(memory, xy=(0., 0.), t=0.):
    query = torch.tensor(xy, dtype=torch.float32).reshape(1, 1, 2).expand(memory.num_envs, 4, 2)
    return memory.sample(query, time_s=t)


def test_unknown_current_map_preserves_only_previously_observed_terrain():
    mem = make()
    update(mem, [[.5, 0., .17]], t=0., age=.3)
    assert not sample(mem, xy=(-.5, 0.))[1].any()
    # Move incrementally: no pose jump and no new terrain observations.
    for i in range(1, 6):
        update(mem, [[0., 0., 0.]], t=.1*i, valid=False, pose=[.1*i, 0., 0.])
    height, valid, age = sample(mem, xy=(.5, 0.), t=.5)
    assert valid.all()
    torch.testing.assert_close(height, torch.full((1, 4), .17))
    torch.testing.assert_close(age, torch.full((1, 4), .8))
    assert not sample(mem, xy=(2., 0.), t=.5)[1].any()


def test_original_capture_expiry_is_not_renewed_by_redelivery():
    mem = make()
    update(mem, [[0., 0., .17]], t=0., age=.3)
    update(mem, [[0., 0., .17]], t=.1, age=.4)
    assert sample(mem, t=1.69)[1].all()
    assert not sample(mem, t=1.71)[1].any()
    for invalid_age in (.51, -1., float("nan")):
        fresh = make()
        update(fresh, [[0., 0., .17]], age=invalid_age)
        assert not sample(fresh)[1].any()


def test_newer_capture_replaces_old_and_stale_delivery_cannot_overwrite():
    mem = make()
    update(mem, [[0., 0., .1]], t=0.)
    update(mem, [[0., 0., .11]], t=.2)
    update(mem, [[0., 0., .4]], t=.3, age=.3)
    height, valid, age = sample(mem, t=.3)
    assert valid.all()
    torch.testing.assert_close(height, torch.full((1, 4), .11))
    torch.testing.assert_close(age, torch.full((1, 4), .1), atol=1e-6, rtol=0)


def test_contradictory_height_invalidates_footprint_without_averaging():
    mem = make()
    update(mem, [[0., 0., 0.]], t=0.)
    update(mem, [[0., 0., .2]], t=.2)
    height, valid, age = sample(mem, t=.21)
    assert not valid.any() and torch.isnan(height).all() and torch.isnan(age).all()
    # The older conflicting capture, not each delivery, fixes the quarantine.
    update(mem, [[0., 0., .2]], t=.3, age=.1)
    assert not sample(mem, t=1.99)[1].any()
    height, valid, age = sample(mem, t=2.01)
    assert valid.all()
    torch.testing.assert_close(height, torch.full((1, 4), .2))
    assert (age > 1.8).all()


def test_same_capture_conflict_and_stair_edge_remain_unknown():
    conflict = make(m=2)
    update(conflict, [[0., 0., 0.], [.001, 0., .2]])
    assert not sample(conflict)[1].any()
    update(conflict, [[0., 0., 0.], [0., 0., 0.]], t=.2, valid=False)
    assert not sample(conflict, t=.2)[1].any()
    edge = make(m=2)
    update(edge, [[-.05, 0., 0.], [.05, 0., .17]])
    assert not sample(edge)[1].any()


def test_older_and_newer_captures_in_one_packet_retain_conflict_after_old_cell_disappears():
    mem = make(m=2)
    update(mem, [[.001, 0., 0.], [.001, 0., .2]], t=1., age=[[.1, 0.]])
    assert not sample(mem, xy=(.001, 0.), t=1.)[1].any()
    update(mem, [[.001, 0., 0.], [.001, 0., .2]],
           t=1.02, age=[[.12, .02]], valid=[[False, True]])
    assert not sample(mem, xy=(.001, 0.), t=1.02)[1].any()
    assert not sample(mem, xy=(.001, 0.), t=2.89)[1].any()
    assert sample(mem, xy=(.001, 0.), t=2.91)[1].all()


def test_hash_eviction_cannot_release_conflict_early():
    mem = make()
    update(mem, [[.001, 0., 0.]], t=1.)
    update(mem, [[.001, 0., .2]], t=1.1)
    assert not sample(mem, xy=(.001, 0.), t=1.1)[1].any()
    # These cell coordinates share a slot at capacity 16.
    update(mem, [[.401, 0., .2]], t=1.2)
    update(mem, [[.001, 0., .2]], t=1.3)
    assert not sample(mem, xy=(.001, 0.), t=1.3)[1].any()
    assert not sample(mem, xy=(.001, 0.), t=2.99)[1].any()
    assert sample(mem, xy=(.001, 0.), t=3.01)[1].all()


@pytest.mark.parametrize("kind", ["position", "yaw", "clock"])
def test_jump_clears_and_rejects_jump_frame_and_delayed_old_epoch(kind):
    mem = make()
    update(mem, [[0., 0., .17]], t=1.)
    pose = [.4, 0., 0.] if kind == "position" else [0., 0., 0.]
    yaw = [.5] if kind == "yaw" else [0.]
    now = .5 if kind == "clock" else 1.1
    update(mem, [[0., 0., .17]], t=now, pose=pose, yaw=yaw)
    assert not sample(mem, t=now)[1].any()
    update(mem, [[0., 0., .17]], t=now+.2, age=.3, pose=pose, yaw=yaw)
    assert not sample(mem, t=now+.2)[1].any()
    update(mem, [[0., 0., .17]], t=now+.3, age=0., pose=pose, yaw=yaw)
    assert sample(mem, t=now+.3)[1].all()


def test_wrapped_yaw_is_continuous_and_partial_reset_isolated():
    mem = make(n=2)
    update(mem, [[[0., 0., .1]], [[0., 0., .2]]], yaw=[3.13, 3.13])
    update(mem, [[[0., 0., .1]], [[0., 0., .2]]], t=.1, age=.1, yaw=[-3.13, -3.13])
    assert sample(mem, t=.1)[1].all()
    mem.reset(torch.tensor([0]))
    _, valid, _ = sample(mem, t=.1)
    assert not valid[0].any() and valid[1].all()
    mem.reset()
    assert not sample(mem, t=.1)[1].any()


def test_hash_collision_can_drop_evidence_but_never_alias_locations():
    mem = make(terrain_memory_frames=1)
    # Capacity one forces every coordinate to collide.
    update(mem, [[0., 0., .1]])
    update(mem, [[2., 0., .2]], t=.1)
    assert not sample(mem, xy=(0., 0.), t=.1)[1].any()
    height, valid, _ = sample(mem, xy=(2., 0.), t=.1)
    assert valid.all()
    torch.testing.assert_close(height, torch.full((1, 4), .2))
    assert mem.summary()["counts"]["hash_evictions"] == 1


def test_live_map_has_no_history_update_period_latency():
    mem = make()
    update(mem, [[0., 0., .1]])
    # This new cell is delivered before the next 100 ms history admission.
    update(mem, [[.5, 0., .2]], t=.02)
    height, valid, age = sample(mem, xy=(.5, 0.), t=.02)
    assert valid.all() and (age == 0).all()
    torch.testing.assert_close(height, torch.full((1, 4), .2))


def test_nonfinite_inputs_and_time_cannot_become_flat_ground():
    mem = make(m=2)
    update(mem, [[float("nan"), 0., .2], [0., 0., float("inf")]])
    assert not sample(mem)[1].any()
    update(mem, [[0., 0., .2], [.1, 0., .2]], t=.1)
    assert not sample(mem, t=float("nan"))[1].any()
    update(mem, [[0., 0., .2], [.1, 0., .2]], t=.2, yaw=[float("nan")])
    assert not sample(mem, t=.2)[1].any()


def test_memory_capacity_and_storage_are_explicit():
    memory = make(n=4, m=187)
    assert memory.capacity == 16 * 187
    assert memory.summary()["persistent_storage_bytes"] == memory.storage_bytes
    assert memory.storage_bytes < 1024 * 1024
    assert memory.summary()["query_cells_per_support"] == 121
