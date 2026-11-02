"""Bounded, per-wheel ascent credit. No force/torque bonus or prescribed gait.

Positive credit uses physical tread support, never the perceptual wall surface.
Episode history is reset on every reset; only aggregate evidence is resumed.
"""
import math
import torch


def validate(cfg, terrain):
    keys = {'enabled', 'settle_s', 'slip_tolerance_m_s', 'slip_scale_m_s',
            'landing_slip_m_s', 'edge_margin_m', 'step_reward_budget',
            'progress_reward_budget', 'slip_weight', 'min_settle_samples', 'settle_tread_fraction'}
    if not isinstance(cfg, dict) or set(cfg) != keys or type(cfg['enabled']) is not bool:
        raise ValueError('Invalid hind ascent configuration')
    for key in keys - {'enabled'}:
        v = cfg[key]
        if type(v) not in (int, float) or not math.isfinite(v) or v <= 0:
            raise ValueError('Invalid hind ascent value: ' + key)
    if type(cfg['min_settle_samples']) is not int or cfg['min_settle_samples'] < 2:
        raise ValueError('Hind landing requires consecutive support observations')
    if cfg['settle_tread_fraction'] > 1:
        raise ValueError('Hind settling fraction exceeds the available tread')
    if cfg['landing_slip_m_s'] < cfg['slip_tolerance_m_s']:
        raise ValueError('Hind landing slip threshold is below the slip deadband')
    if cfg['edge_margin_m'] >= terrain['geometry']['depth_range'][0] / 2:
        raise ValueError('Hind landing margin consumes the tread')
    if cfg['enabled'] and not terrain.get('hind_clearance', {}).get('enabled'):
        raise ValueError('Hind ascent requires the existing clearance contract')


def rear_measurement(g, radius, half_length):
    """Wheel material-point velocity already includes angular velocity cross r."""
    pos, axis = g['position'][:, 2:], g['axis'][:, 2:]
    velocity, normal = g['velocity'][:, 2:], g['normal'][:, 2:]
    finite = torch.isfinite(torch.cat((pos, axis, velocity, normal, g['force'][:, 2:]), -1)).all(-1)
    finite &= torch.isfinite(g['height'][:, 2:])
    tangent = velocity - (velocity * normal).sum(-1, keepdim=True) * normal
    slip = torch.nan_to_num(tangent.norm(dim=-1), nan=torch.inf, posinf=torch.inf)
    bottom = pos[..., 2] - radius - half_length * axis[..., 2].abs()
    reach = radius + half_length * axis[..., 0].abs()
    lateral = radius + half_length * axis[..., 1].abs()
    return dict(finite=finite, slip=slip, bottom=bottom, reach=reach, lateral=lateral,
                supported=g['supported'][:, 2:] & finite)


class HindAscent:
    def __init__(self, state):
        from .hind_clearance import rising_edges
        self.state, self.cfg = state, state.cfg['hind_ascent']
        self.edges, self.valid = rising_edges(state.surface.rectangles, state.cfg['hind_clearance']['min_rise_m'])
        self.valid &= self.edges[..., 0] >= state.cfg['geometry']['route_start'] - 1.e-5
        self.ends = state.surface.rectangles[..., 1]
        self.first = torch.where(self.valid, self.edges[..., 0], torch.inf).amin(-1)
        self.last = torch.where(self.valid, self.edges[..., 0], -torch.inf).amax(-1) + self.cfg['edge_margin_m']
        self.edge_count = self.valid.sum(-1).clamp_min(1)
        n, device = state.env.num_envs, state.env.device
        shape = (n, 2, self.edges.shape[1])
        self.armed = torch.zeros(shape, dtype=torch.bool, device=device)
        self.paid = torch.zeros_like(self.armed)
        self.settled = torch.zeros(shape, device=device)
        self.best = torch.zeros(n, 2, device=device)
        self.last_x = torch.zeros(n, 2, device=device)
        self.initialized = torch.zeros(n, dtype=torch.bool, device=device)
        self.values = {k:torch.zeros(n, device=device) for k in ('step', 'progress', 'slip')}
        self.tick = -1
        # Cohort x rear wheel. Units are explicit; forces are observation only.
        self.totals = {k:torch.zeros(2, 2, dtype=torch.float64, device=device) for k in
                       ('confirmed_steps', 'step_reward', 'progress_reward', 'slip_cost',
                        'support_s', 'slip_distance_m', 'normal_force_impulse_Ns')}

    def reset(self, ids):
        self.armed[ids] = self.paid[ids] = False
        self.settled[ids] = 0.
        self.best[ids] = 0.
        self.last_x[ids] = 0.
        self.initialized[ids] = False
        for v in self.values.values(): v[ids] = 0.
        # advance is driven by the course once after physics, never by reward calls.

    def advance(self, g):
        s, c, dt = self.state, self.cfg, self.state.env.step_dt
        if self.tick == s.env.common_step_counter: return
        self.tick = s.env.common_step_counter
        tail = s.env.cfg.stair_tail
        m = rear_measurement(g, tail.wheel_radius, tail.wheel_half_length)
        ascent = (s.course_ids == 0) | (s.course_ids == 2)
        s.failed |= ascent & ~m['finite'].all(-1)
        moving = ascent & ~s.failed & ~s.reached & (g['command'][:, 0] > s.cfg['hind_clearance']['command_min_m_s'])
        pos = g['position'][:, 2:].clone()
        pos[..., :2] -= s.origins[s.tile, None, :2]  # absolute Z, including elevated spawns
        edges, valid = self.edges[s.tile], self.valid[s.tile]
        x, ymin, ymax, top = (edges[:, None, :, j] for j in (0, 1, 2, 4))
        rear_x = pos[..., 0] - m['reach']
        finite = m['finite'][..., None]
        initial = ~self.initialized
        baseline = ((rear_x[..., None] >= x) |
                    ((pos[..., 0, None] >= x) & m['supported'][..., None]
                     & (g['height'][:, 2:, None] >= top)))
        self.paid |= initial[:, None, None] & finite & baseline
        # A spawn on an upper tread establishes a baseline and earns no event.
        self.armed |= (finite & ascent[:, None, None] & valid[:, None]
                       & (rear_x[..., None] < x) & (m['bottom'][..., None] < top))
        tol = s.cfg['hind_clearance']['support_tolerance_m']
        landing = (valid[:, None] & finite & m['supported'][..., None]
                   & (rear_x[..., None] >= x + c['edge_margin_m'])
                   & (pos[..., 0, None] < self.ends[s.tile, None])
                   & (pos[..., 1, None] - m['lateral'][..., None] >= ymin)
                   & (pos[..., 1, None] + m['lateral'][..., None] <= ymax)
                   & ((m['bottom'][..., None] - top).abs() <= tol)
                   & ((g['height'][:, 2:, None] - top).abs() <= tol)
                   & (m['slip'][..., None] <= c['landing_slip_m_s'])
                   & moving[:, None, None])
        self.settled = torch.where(landing, self.settled + dt, 0.)
        # Narrow treads at the commanded speed must admit a rolling landing.
        speed = torch.maximum(g['command'][:, 0, None].abs(), (pos[..., 0] - self.last_x).abs() / dt)
        speed = torch.where(initial[:, None], g['command'][:, 0, None].abs(), speed)
        speed = speed.clamp_min(s.cfg['hind_clearance']['command_min_m_s'])
        available = (self.ends[s.tile, None] - x - m['reach'][..., None] - c['edge_margin_m']).clamp_min(0.)
        required = (c['settle_tread_fraction'] * available / speed[..., None]).clamp(
            max=c['settle_s']).clamp_min(c['min_settle_samples'] * dt)
        events = self.armed & ~self.paid & (self.settled + 1.e-7 >= required)
        self.paid |= events
        credit = events.sum(-1) * (c['step_reward_budget'] / (2 * self.edge_count[s.tile, None]))
        # Consume all finite physical advance, even during swing/pause/slip, so
        # landing later cannot retroactively cash in unsupported displacement.
        first, last = self.first[s.tile, None], self.last[s.tile, None]
        span = (last - first).clamp_min(c['edge_margin_m'])
        coordinate = torch.nan_to_num(((rear_x - first) / span).clamp(0., 1.), nan=0.)
        delta = (coordinate - self.best).clamp_min(0.)
        initial = ~self.initialized
        delta[initial] = 0.
        good = landing.any(-1)  # Same wheel, same physical ascending tread.
        progress = .5 * c['progress_reward_budget'] * delta * good
        self.best = torch.where(m['finite'], torch.maximum(self.best, coordinate), self.best)
        self.last_x = torch.where(m['finite'],pos[..., 0],self.last_x)
        self.initialized |= m['finite'].all(-1)
        # The new cost supplements the existing all-wheel cost only in ascent.
        zone = (rear_x >= first - s.cfg['hind_clearance']['lookahead_m']) & (rear_x <= last)
        loaded = m['supported'] & moving[:, None] & zone
        excess = ((m['slip'] - c['slip_tolerance_m_s']).clamp_min(0.) / c['slip_scale_m_s']).clamp(max=4.)
        cost = .5 * c['slip_weight'] * torch.where(loaded, excess.square(), 0.)
        self.values['step'] = credit.sum(-1) / dt
        self.values['progress'] = progress.sum(-1) / dt
        self.values['slip'] = -cost.sum(-1)
        cohort = (~s.teacher).long()
        updates = dict(confirmed_steps=events.sum(-1), step_reward=credit, progress_reward=progress,
                       slip_cost=cost * dt, support_s=loaded * dt,
                       slip_distance_m=torch.where(loaded, m['slip'], 0.) * dt,
                       normal_force_impulse_Ns=torch.where(loaded, g['force_n'][:, 2:].clamp_min(0.), 0.) * dt)
        for name, value in updates.items(): self.totals[name].index_add_(0, cohort, value.double())

    def diagnostics(self):
        out = {}
        for name, tensor in self.totals.items():
            values = tensor.cpu().tolist()
            for i, cohort in enumerate(('teacher', 'student')):
                for j, leg in enumerate(('left', 'right')): out[f'{cohort}/{leg}/{name}'] = values[i][j]
        return out

    def snapshot(self):
        return {k:v.cpu().clone() for k,v in self.totals.items()}

    def checked_snapshot(self, saved):
        if not isinstance(saved, dict) or saved.keys() != self.totals.keys():
            raise ValueError('Hind ascent ledger keys changed')
        for k,v in saved.items():
            target = self.totals[k]
            if (not isinstance(v, torch.Tensor) or v.shape != target.shape or v.dtype != target.dtype
                    or not torch.isfinite(v).all() or (v < 0).any()):
                raise ValueError('Invalid hind ascent ledger: ' + k)
        return {k:v.to(self.state.env.device) for k,v in saved.items()}


def hind_stable_step(env):
    return env.cts_all_terrain.hind_ascent.values['step']


def hind_supported_progress(env):
    return env.cts_all_terrain.hind_ascent.values['progress']


def hind_ascent_slip(env):
    return env.cts_all_terrain.hind_ascent.values['slip']
