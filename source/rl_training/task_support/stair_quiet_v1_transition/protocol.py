"""Command experiments are training inputs, not a universal fixed-pause assist."""
import torch


class EntryProtocol:
    def __init__(self, n, device, settings):
        self.s = settings
        self.mode = torch.zeros(n, dtype=torch.long, device=device)
        self.phase = torch.zeros_like(self.mode)  # approach, pause/ramp, finished
        self.started_at = torch.zeros(n, device=device)
        self.trigger = torch.zeros(n, device=device)
        self.pause = torch.zeros(n, device=device)
        self.ramp = torch.ones(n, device=device)
        self.stopped = torch.zeros(n, dtype=torch.bool, device=device)

    def reset(self, ids, counts):
        self.mode[ids] = (ids+counts) % 4
        self.phase[ids] = 0
        self.started_at[ids] = 0
        self.stopped[ids] = False
        for name, bounds in (("trigger", self.s.entry_trigger_range_m),
                             ("pause", self.s.entry_pause_range_s), ("ramp", self.s.entry_ramp_range_s)):
            getattr(self, name)[ids] = torch.empty(len(ids), device=self.mode.device).uniform_(*bounds)

    def command(self, ids, now, gap, crossed, desired_speed):
        self.stopped[ids] |= crossed[ids]
        mode, phase = self.mode[ids], self.phase[ids]
        begin = (phase == 0) & ((mode == 1) | (mode == 2)) & (gap[ids] <= self.trigger[ids])
        self.started_at[ids] = torch.where(begin, now, self.started_at[ids])
        phase = torch.where(begin, 1, phase)
        age = (now-self.started_at[ids]).clamp_min(0)
        waiting = (mode == 1) & (phase == 1) & (age < self.pause[ids])
        pause_ramp = ((age-self.pause[ids])/self.ramp[ids]).clamp(0, 1)
        # Ramp examples enter at half of their sampled command, never a forced
        # universal stop. Both kinds use physical acceleration under the actor.
        moving_ramp = .5+.5*(age/self.ramp[ids]).clamp(0, 1)
        scale = torch.where((mode == 1) & (phase > 0), pause_ramp, 1.)
        scale = torch.where(mode == 2, torch.where(phase > 0, moving_ramp, .5), scale)
        finished = ((mode == 1) & (age >= self.pause[ids]+self.ramp[ids])) | ((mode == 2) & (age >= self.ramp[ids]))
        self.phase[ids] = torch.where((phase == 1) & finished, 2, phase)
        return torch.where(waiting | self.stopped[ids], 0., desired_speed[ids]*scale)
