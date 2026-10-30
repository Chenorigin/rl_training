"""Bounded memory of observed static treads, never a terrain oracle.

Each environment owns ``frames * input_points`` spatial-hash slots. Hash
collisions evict evidence; exact cell keys are checked on every lookup so a
collision cannot invent terrain at another location. New history cells are
admitted at the declared period; the current live map is usable immediately.
Original capture time is retained, including when a delayed cell is redelivered.

Updates use O(N*(capacity + input_points)) storage and an input-only M log M
grouping pass, not all-pairs history matching. Queries gather only a bounded square of hash cells around four
supports, then use the same tread-discontinuity test as the controller.
Pose is supplied explicitly. Current simulator adapters use simulation pose;
this is not evidence of real LIO accuracy or complete LIO reset detection.
"""
from __future__ import annotations

import math
import torch


class TerrainMemory:
    def __init__(self, num_envs, num_points, device, settings):
        if type(num_envs) is not int or type(num_points) is not int or min(num_envs, num_points) <= 0:
            raise ValueError("Terrain memory requires positive fixed dimensions")
        self.num_envs, self.num_points = num_envs, num_points
        self.device, self.settings = torch.device(device), settings
        self.capacity = settings.terrain_memory_frames * num_points
        self.cell_m = settings.terrain_memory_cell_m
        if not math.isfinite(self.cell_m) or self.cell_m <= 0 or self.capacity <= 0:
            raise ValueError("Invalid memory capacity or spatial resolution")
        n, c, m = num_envs, self.capacity, num_points
        self.points = torch.zeros((n, c, 3), device=device)
        self.device = self.points.device
        self.keys = torch.zeros((n, c), dtype=torch.long, device=device)
        self.capture = torch.full((n, c), -torch.inf, device=device)
        self.occupied = torch.zeros((n, c), dtype=torch.bool, device=device)
        self.conflict_until = torch.full((n, c), -torch.inf, device=device)
        self.live_points = torch.zeros((n, m, 3), device=device)
        self.live_valid = torch.zeros((n, m), dtype=torch.bool, device=device)
        self.live_capture = torch.full((n, m), -torch.inf, device=device)
        self.last_pose = torch.zeros((n, 3), device=device)
        self.last_yaw = torch.zeros(n, device=device)
        self.last_time = torch.zeros(n, device=device)
        self.last_admission = torch.full((n,), -torch.inf, device=device)
        self.capture_not_before = torch.full((n,), -torch.inf, device=device)
        self.seen_pose = torch.zeros(n, dtype=torch.bool, device=device)
        self.counts = torch.zeros(6, dtype=torch.long, device=device)
        radius = math.ceil(settings.terrain_max_distance_m / self.cell_m)
        if radius > 16:
            raise ValueError("Terrain query would exceed the bounded cell neighborhood")
        axis = torch.arange(-radius, radius + 1, device=device)
        xx, yy = torch.meshgrid(axis, axis, indexing="ij")
        self.offsets = torch.stack((xx.flatten(), yy.flatten()), -1)

    @property
    def storage_bytes(self):
        return sum(x.numel() * x.element_size() for x in vars(self).values() if isinstance(x, torch.Tensor))

    def summary(self):
        names = ("updates", "admitted_cells", "hash_evictions", "conflicting_cells",
                 "pose_jumps", "clock_or_invalid_pose_resets")
        return {
            "capacity_per_environment": self.capacity,
            "query_cells_per_support": len(self.offsets),
            "persistent_storage_bytes": self.storage_bytes,
            "counts": dict(zip(names, self.counts.detach().cpu().tolist())),
            "capture_clock": "original observation time; redelivery does not renew age",
            "pose_scope": "position/yaw jump and clock reversal detection; explicit reset required for other session changes",
            "collision_policy": "exact key lookup; collisions discard positive evidence but retain conservative bucket quarantine",
            "hardware_pose_validated": False,
        }

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.occupied[ids] = False
        self.live_valid[ids] = False
        self.capture[ids] = -torch.inf
        self.conflict_until[ids] = -torch.inf
        self.live_capture[ids] = -torch.inf
        self.last_admission[ids] = -torch.inf
        self.capture_not_before[ids] = -torch.inf
        self.seen_pose[ids] = False

    def _time(self, time_s):
        value = torch.as_tensor(time_s, device=self.device, dtype=torch.float32)
        if value.ndim == 0:
            value = value.expand(self.num_envs)
        if value.shape != (self.num_envs,):
            raise ValueError("Expected one memory timestamp per environment")
        return value

    def _cells(self, xy):
        cells = torch.floor(torch.nan_to_num(xy, nan=0., posinf=0., neginf=0.) / self.cell_m).long()
        # Full signed x/y coordinates remain in the key; slot collisions cannot
        # make a distant point look like a nearby one.
        key = (cells[..., 0] << 32) | (cells[..., 1] & 0xffffffff)
        slot = torch.remainder((cells[..., 0] * 73856093) ^ (cells[..., 1] * 19349663),
                               self.capacity)
        return cells, key, slot

    def _input_quarantine(self, key, slot, z, capture, fresh):
        """Group only current M points by exact cell, including older captures.

        Negative evidence must survive even when another cell wins the same
        hash bucket. Sorting M inputs avoids a history-by-input distance matrix.
        """
        n, m = key.shape
        order = key.argsort(dim=1, stable=True)
        keys = key.gather(1, order)
        new_group = torch.cat((torch.ones((n, 1), dtype=torch.bool, device=self.device),
                               keys[:, 1:] != keys[:, :-1]), 1)
        group = new_group.long().cumsum(1) - 1
        height, time, usable = z.gather(1, order), capture.gather(1, order), fresh.gather(1, order)
        lo = torch.full((n, m), torch.inf, device=self.device)
        hi = torch.full((n, m), -torch.inf, device=self.device)
        newest = torch.full((n, m), -torch.inf, device=self.device)
        lo.scatter_reduce_(1, group, torch.where(usable, height, torch.inf), reduce="amin")
        hi.scatter_reduce_(1, group, torch.where(usable, height, -torch.inf), reduce="amax")
        newest.scatter_reduce_(1, group, torch.where(usable, time, -torch.inf), reduce="amax")
        latest = usable & (time >= newest.gather(1, group) - 1e-6)
        chosen = torch.full((n, m), -1, dtype=torch.long, device=self.device)
        chosen.scatter_reduce_(1, group, torch.where(
            latest, torch.arange(m, device=self.device).expand(n, -1), -1), reduce="amax")
        newest_height = height.gather(1, chosen.clamp_min(0))
        contradicts = usable & ((height - newest_height.gather(1, group)).abs()
                                > self.settings.terrain_max_height_spread_m)
        contrary_time = torch.full((n, m), -torch.inf, device=self.device)
        contrary_time.scatter_reduce_(1, group, torch.where(contradicts, time, -torch.inf), reduce="amax")
        # A newest height between two contradictory extremes is conservatively
        # held to its capture expiry; never release before an opposing capture.
        expiry = torch.where(torch.isfinite(contrary_time), contrary_time, newest)
        expiry += self.settings.terrain_memory_s
        expiry = torch.where(hi - lo > self.settings.terrain_max_height_spread_m, expiry, -torch.inf)
        buckets = torch.full((n, self.capacity), -torch.inf, device=self.device)
        buckets.scatter_reduce_(1, slot.gather(1, order), expiry.gather(1, group), reduce="amax")
        return buckets

    @torch.no_grad()
    def update(self, *, points_w, valid, age_s, pose_w, yaw_w, time_s):
        n, m, c, s = self.num_envs, self.num_points, self.capacity, self.settings
        if (points_w.shape != (n, m, 3) or valid.shape != (n, m)
                or age_s.shape != (n, m) or pose_w.shape != (n, 3)
                or yaw_w.shape != (n,)):
            raise ValueError("Terrain memory observation shape mismatch")
        if any(t.device != self.device for t in (points_w, valid, age_s, pose_w, yaw_w)):
            raise ValueError("Terrain memory inputs must share its device")
        now = self._time(time_s)
        pose_ok = torch.isfinite(pose_w).all(-1) & torch.isfinite(yaw_w) & torch.isfinite(now)
        position_jump = (pose_w - self.last_pose).norm(dim=-1) > s.terrain_pose_jump_m
        delta_yaw = torch.atan2(torch.sin(yaw_w - self.last_yaw), torch.cos(yaw_w - self.last_yaw)).abs()
        pose_jump = self.seen_pose & pose_ok & (position_jump | (delta_yaw > s.terrain_yaw_jump_rad))
        clock_jump = self.seen_pose & (now < self.last_time - 1e-6)
        reset = pose_jump | clock_jump | ~pose_ok
        self.occupied &= ~reset[:, None]
        self.live_valid &= ~reset[:, None]
        self.conflict_until = torch.where(reset[:, None], -torch.inf, self.conflict_until)
        safe_now = torch.where(torch.isfinite(now), now, self.last_time)
        self.capture_not_before = torch.where(reset, safe_now, self.capture_not_before)
        self.last_admission = torch.where(reset, safe_now, self.last_admission)
        due = safe_now - self.last_admission >= s.terrain_memory_update_s - 1e-6
        self.last_admission = torch.where(due & pose_ok & ~reset, safe_now, self.last_admission)
        capture = safe_now[:, None] - age_s
        fresh = (valid.bool() & torch.isfinite(points_w).all(-1) & torch.isfinite(age_s)
                 & (age_s >= 0) & (age_s <= s.terrain_max_age_s)
                 & (points_w[..., :2].abs().amax(-1) < self.cell_m * 2**30)
                 & (capture >= self.capture_not_before[:, None] - 1e-6)
                 & pose_ok[:, None] & ~reset[:, None])
        self.live_points.copy_(torch.nan_to_num(points_w))
        self.live_valid.copy_(fresh)
        self.live_capture.copy_(torch.where(fresh, capture, -torch.inf))
        _, key, slot = self._cells(points_w[..., :2])
        old_key = self.keys.gather(1, slot)
        old_capture = self.capture.gather(1, slot)
        newer_cached = (self.occupied.gather(1, slot) & (old_key == key)
                        & (safe_now[:, None] - old_capture <= s.terrain_memory_s)
                        & (capture < old_capture - 1e-6))
        self.live_valid &= ~newer_cached
        # Per hash bucket, choose its latest actually captured input. A stale
        # packet delivered later cannot replace newer evidence.
        newest = torch.full((n, c), -torch.inf, device=self.device)
        newest.scatter_reduce_(1, slot, torch.where(fresh, capture, -torch.inf), reduce="amax")
        latest = fresh & (capture >= newest.gather(1, slot) - 1e-6)
        source_index = torch.arange(m, device=self.device).expand(n, -1)
        chosen = torch.full((n, c), -1, dtype=torch.long, device=self.device)
        chosen.scatter_reduce_(1, slot, torch.where(latest, source_index, -1), reduce="amax")
        index = chosen.clamp_min(0)
        new_points = points_w.gather(1, index[..., None].expand(-1, -1, 3))
        new_keys = key.gather(1, index)
        input_quarantine = self._input_quarantine(key, slot, points_w[..., 2], capture, fresh)
        old_alive = self.occupied & ((safe_now[:, None] - self.capture) <= s.terrain_memory_s)
        same = old_alive & (self.keys == new_keys)
        accept = ((chosen >= 0) & (due[:, None] | same)
                  & (~old_alive | (newest > self.capture + 1e-6)
                     | (same & (newest >= self.capture - 1e-6))))
        # Contradictory heights are not averaged into a fictitious flat tread.
        # Keep the new measurement, but block that footprint until the older
        # contradicting observation expires. Same-capture conflicts expire
        # with that capture, not with each repeated delivery.
        contradiction = same & ((new_points[..., 2] - self.points[..., 2]).abs()
                                > s.terrain_max_height_spread_m)
        # Negative evidence belongs to the bucket even after a hash collision
        # evicts its positive sample. Exact-key occupancy alone cannot protect
        # against a later new observation silently recreating that old cell.
        until = torch.maximum(self.conflict_until, input_quarantine)
        until = torch.where(accept & contradiction,
                            torch.maximum(until, self.capture + s.terrain_memory_s), until)
        new_conflicts = until > self.conflict_until
        evictions = accept & old_alive & ~same
        self.points = torch.where(accept[..., None], new_points, self.points)
        self.keys = torch.where(accept, new_keys, self.keys)
        self.capture = torch.where(accept, newest, self.capture)
        self.conflict_until = until
        self.occupied |= accept
        self.last_pose = torch.where(pose_ok[:, None], pose_w, self.last_pose)
        self.last_yaw = torch.where(pose_ok, yaw_w, self.last_yaw)
        self.last_time = safe_now
        self.seen_pose = pose_ok
        self.counts += torch.stack((torch.ones_like(now, dtype=torch.long).sum(), accept.sum(),
                                    evictions.sum(), new_conflicts.sum(),
                                    pose_jump.sum(), (clock_jump | ~pose_ok).sum()))

    @torch.no_grad()
    def sample(self, wheel_xy, *, time_s):
        from .landing import sample_tread_surface
        if wheel_xy.shape != (self.num_envs, 4, 2):
            raise ValueError("Terrain memory query requires four XY support points")
        now, s = self._time(time_s), self.settings
        cells, _, _ = self._cells(wheel_xy)
        neighbor = cells[:, :, None, :] + self.offsets[None, None]
        key = (neighbor[..., 0] << 32) | (neighbor[..., 1] & 0xffffffff)
        slot = torch.remainder((neighbor[..., 0] * 73856093) ^ (neighbor[..., 1] * 19349663),
                               self.capacity).flatten(1)
        key = key.flatten(1)
        p = self.points.gather(1, slot[..., None].expand(-1, -1, 3))
        capture = self.capture.gather(1, slot)
        age = now[:, None] - capture
        known = (self.occupied.gather(1, slot) & (self.keys.gather(1, slot) == key)
                 & (age >= -1e-6) & (age <= s.terrain_memory_s))
        conflict = known & (self.conflict_until.gather(1, slot) > now[:, None])
        # A quarantined bucket cannot be freed by evicting its original key.
        # Treat any queried cell hashing to it conservatively as unknown.
        cell_low = neighbor * self.cell_m
        cell_high = cell_low + self.cell_m
        query = wheel_xy[:, :, None, :]
        cell_distance = torch.maximum(cell_low - query, query - cell_high).clamp_min(0.)
        footprint_cell = cell_distance.square().sum(-1) <= s.terrain_max_distance_m**2
        quarantined = (self.conflict_until.gather(1, slot).reshape(
            self.num_envs, 4, -1) > now[:, None, None]) & footprint_cell
        live_age = now[:, None] - self.live_capture
        live_valid = self.live_valid & (live_age >= -1e-6) & (live_age <= s.terrain_max_age_s)
        points = torch.cat((p, self.live_points), 1)
        usable = torch.cat((known & ~conflict, live_valid), 1)
        ages = torch.cat((age, live_age), 1)
        height, valid = sample_tread_surface(
            wheel_xy, points, usable, max_distance=s.terrain_max_distance_m,
            max_height_spread=s.terrain_max_height_spread_m)
        dist2 = (wheel_xy[:, :, None] - points[:, None, :, :2]).square().sum(-1)
        nearest = torch.where(usable[:, None] & torch.isfinite(dist2), dist2, torch.inf).argmin(-1)
        capture_age = ages.gather(1, nearest)
        valid &= ~quarantined.any(-1) & torch.isfinite(now)[:, None] & (now >= self.last_time - 1e-6)[:, None]
        return (torch.where(valid, height, torch.nan), valid,
                torch.where(valid, capture_age, torch.nan))
