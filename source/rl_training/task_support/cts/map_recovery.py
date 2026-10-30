"""Training-only acquisition outages; the existing packet queue owns age/quality.

No actor channels, output interpolation, reward weights or deployment gates are
changed. Missing returns cannot refresh old cells. Restored returns still pass
the ordinary visibility, packet delay, geometry/noise and confidence pipeline.
"""
from copy import deepcopy
import math

import torch

from task_support.student_v1.perception import StudentObservationState, terrain_visibility
from task_support.student_v1.contract import HEIGHT_DIM


def validate_recovery(cfg):
    keys = {'enabled', 'episode_probability', 'interval_s', 'blackout_extra_s',
            'recovery_s', 'initial_keep_fraction', 'ramp_probability'}
    if not isinstance(cfg, dict) or set(cfg) != keys or type(cfg['enabled']) is not bool:
        raise ValueError('Unknown or missing CTS map recovery setting')
    for key in ('episode_probability', 'ramp_probability'):
        value = cfg[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f'CTS map recovery {key} must be finite in [0,1]')
    for key in ('interval_s', 'blackout_extra_s', 'recovery_s', 'initial_keep_fraction'):
        value = cfg[key]
        if (not isinstance(value, (list, tuple)) or len(value) != 2
                or any(type(x) not in (int, float) or not math.isfinite(x) for x in value)
                or not 0 < value[0] <= value[1]):
            raise ValueError(f'CTS map recovery {key} requires positive ordered endpoints')
    if cfg['initial_keep_fraction'][1] >= 1:
        raise ValueError('CTS map recovery must start sparse')
    return cfg


class RecoveryObservationState(StudentObservationState):
    def __init__(self, *args, recovery, **kwargs):
        super().__init__(*args, **kwargs)
        self.recovery_cfg = deepcopy(validate_recovery(recovery))
        n, device = self.num_envs, self.device
        self.recovery_needs_reset = torch.ones(n, dtype=torch.bool, device=device)
        self.recovery_selected = torch.zeros(n, dtype=torch.bool, device=device)
        self.recovery_next = torch.zeros(n, dtype=torch.long, device=device)
        self.recovery_start = torch.full_like(self.recovery_next, -1)
        self.recovery_end = torch.full_like(self.recovery_next, -1)
        self.recovery_keep = torch.zeros(n, device=device)
        self.recovery_ramp = torch.zeros(n, dtype=torch.bool, device=device)
        self.recovery_rank = torch.zeros(n, len(self.capture_grid), device=device)
        self.recovery_phase = torch.zeros(n, dtype=torch.long, device=device)
        self.recovery_events = torch.zeros(n, dtype=torch.long, device=device)
        self.recovery_transitions = torch.zeros_like(self.recovery_events)
        self.recovery_previous_count = torch.full_like(self.recovery_events, -1)
        self.recovery_new_cells = torch.zeros(n, device=device)
        self.recovery_previous_valid = torch.zeros(n, HEIGHT_DIM, dtype=torch.bool, device=device)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self.recovery_needs_reset[ids] = True
        self.recovery_selected[ids] = False
        self.recovery_phase[ids] = 0
        self.recovery_start[ids] = self.recovery_end[ids] = -1
        self.recovery_previous_count[ids] = -1
        self.recovery_previous_valid[ids] = False
        self.recovery_new_cells[ids] = 0
        # Event/transition counters are lifetime diagnostics, like delay counters.

    def _draw(self, key, count):
        lo, hi = self.recovery_cfg[key]
        return torch.empty(count, device=self.device).uniform_(lo, hi)

    def _ticks(self, key, count, extra=0.):
        return ((self._draw(key, count) + extra) / self.step_dt).ceil().long()

    def _acquisition_hits(self, hits, base, yaw, rotation, step, active):
        cfg = self.recovery_cfg
        ids = (self.recovery_needs_reset & active).nonzero().flatten()
        if ids.numel():
            self.recovery_selected[ids] = torch.rand(len(ids), device=self.device) < cfg['episode_probability']
            self.recovery_next[ids] = step + self._ticks('interval_s', len(ids))
            self.recovery_needs_reset[ids] = False
        ids = (active & self.recovery_selected & (step >= self.recovery_next)).nonzero().flatten()
        if ids.numel():
            # Longer than TTL: even the newest pre-outage capture must expire.
            self.recovery_start[ids] = step + self._ticks('blackout_extra_s', len(ids), self.cfg.max_map_age_s)
            self.recovery_end[ids] = self.recovery_start[ids] + self._ticks('recovery_s', len(ids))
            self.recovery_next[ids] = self.recovery_end[ids] + self._ticks('interval_s', len(ids))
            self.recovery_keep[ids] = self._draw('initial_keep_fraction', len(ids))
            self.recovery_ramp[ids] = torch.rand(len(ids), device=self.device) < cfg['ramp_probability']
            self.recovery_rank[ids] = torch.rand(len(ids), len(self.capture_grid), device=self.device)
            self.recovery_events[ids] += 1
        blackout = self.recovery_selected & (step < self.recovery_start)
        restoring = self.recovery_selected & (step >= self.recovery_start) & (step < self.recovery_end)
        phase = blackout.long() + 2 * restoring.long()
        self.recovery_phase[active] = phase[active]
        affected = active & (phase != 0)
        if not affected.any():
            return hits
        ids = affected.nonzero().flatten()
        progress = ((step - self.recovery_start[ids]).float()
                    / (self.recovery_end[ids] - self.recovery_start[ids]).clamp_min(1)).clamp(0, 1)
        keep = self.recovery_keep[ids]
        keep = keep + (1 - keep) * progress * self.recovery_ramp[ids]
        retained = (self.recovery_rank[ids] < keep[:, None]) & ~blackout[ids, None]
        # Evaluate occlusion using the COMPLETE surface before removing returns:
        # dropping an occluder must not reveal cells behind it in the surrogate.
        retained &= terrain_visibility(hits[ids], base[ids], yaw[ids], rotation[ids],
                                       self.cfg, self.capture_shape)
        result = hits.clone()
        result[ids] = torch.where(retained[..., None], hits[ids], torch.nan)
        return result

    def observe(self, clean_proprio, hits, base, yaw, rotation, step):
        if self.last_step == step and self.last_obs is not None:
            return self.last_obs.clone()
        enabled = self.recovery_cfg['enabled'] and self.recovery_cfg['episode_probability'] > 0
        active = self.observed_step != step
        if enabled:
            hits = self._acquisition_hits(hits, base, yaw, rotation, step, active)
        output = super().observe(clean_proprio, hits, base, yaw, rotation, step)
        valid = self.last_channels[1].bool()
        count = valid.sum(-1)
        self.recovery_transitions += (active & (self.recovery_previous_count == 0) & (count > 0)).long()
        self.recovery_new_cells[active] = (valid & ~self.recovery_previous_valid).float().mean(-1)[active]
        self.recovery_previous_count[active] = count[active]
        self.recovery_previous_valid[active] = valid[active]
        return output

    def diagnostics(self):
        return {
            'selected_fraction': self.recovery_selected.float().mean(),
            'blackout_fraction': (self.recovery_phase == 1).float().mean(),
            'recovery_fraction': (self.recovery_phase == 2).float().mean(),
            'empty_fraction': (self.recovery_previous_count == 0).float().mean(),
            'new_cell_fraction': self.recovery_new_cells.mean(),
            'events_total': self.recovery_events.sum(),
            'empty_to_known_total': self.recovery_transitions.sum(),
        }


def install(env, cfg):
    """Install before parent manager construction/reset first asks for sensors."""
    recovery = getattr(cfg, 'cts_map_recovery', None)
    if recovery is None:
        return
    if hasattr(env, '_m20_student_v1_state'):
        raise RuntimeError('CTS map recovery must be installed before sensor creation')
    env._m20_student_v1_state = RecoveryObservationState(
        cfg.scene.num_envs, cfg.sim.device, cfg.sim.dt * cfg.decimation,
        cfg.student_perception, cfg.student_history_steps, recovery=recovery)


def configure(cfg, recovery):
    cfg.cts_map_recovery = deepcopy(validate_recovery(recovery))


def memory_reserve(recovery, perception, num_envs):
    """Conservative extra live tensors plus full-geometry visibility workspace."""
    if not recovery or not recovery['enabled'] or not recovery['episode_probability']:
        return 0
    from task_support.student_v1.perception import capture_grid_shape
    rows, cols = capture_grid_shape(perception.get('map_capture_padding_m', 0.))
    # Includes hit clone/ranks/masks, path XYZ/query/sample/occlusion temporaries
    # with headroom, output masks/history counters. Base queue is already budgeted.
    return num_envs * (rows * cols * (128 + 64 * perception['visibility_samples'])
                       + HEIGHT_DIM * 32 + 256)
