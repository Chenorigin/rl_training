"""Simulator-free per-environment promotion/demotion and final-width gate."""
import torch


class WidthCurriculum:
    def __init__(self, n, device, settings):
        self.s = settings
        self.level = torch.full((n,), settings.initial_width_level, device=device, dtype=torch.long)
        self.success_streak = torch.zeros_like(self.level)

    def update(self, ids, *, actual_levels, succeeded):
        """Only completed episodes call this. Unrelated envs remain unchanged.

        Actual levels describe the mesh just traversed. The new target is
        applied only by the next reset; in-progress meshes never change.
        """
        if not self.s.width_curriculum_enabled:
            return
        streak = torch.where(succeeded, self.success_streak[ids]+1, 0)
        promote = succeeded & (streak >= self.s.consecutive_successes_to_narrow)
        next_level = actual_levels + promote.long() - (~succeeded).long()
        self.level[ids] = next_level.clamp(0, len(self.s.widths_m)-1)
        self.success_streak[ids] = torch.where(promote, 0, streak)

    def state_dict(self):
        return {"level": self.level.cpu().tolist(), "success_streak": self.success_streak.cpu().tolist()}

    def load_state_dict(self, state):
        validated = {}
        for name in ("level", "success_streak"):
            raw = state[name]
            if not isinstance(raw, list) or any(type(v) is not int for v in raw):
                raise ValueError("Curriculum state must contain integer arrays")
            value = torch.tensor(raw, device=self.level.device, dtype=torch.long)
            upper = len(self.s.widths_m) if name == "level" else self.s.consecutive_successes_to_narrow
            if value.shape != self.level.shape or (value < 0).any() or (value >= upper).any():
                raise ValueError("Invalid width curriculum state or changed environment count")
            validated[name] = value
        for name, value in validated.items():
            getattr(self, name).copy_(value)


def final_width_mask(actual_levels, settings):
    return actual_levels == len(settings.widths_m)-1


def record_outcomes(state, types, rows, succeeded, edge, recovery, stall, settings):
    """Count completed episodes on their actual mesh, before any promotion.

    Failure-reason tags may overlap; they are not a partition of failures.
    Only final-width episodes are evidence for expanding contact profiles.
    """
    one = torch.ones_like(types)
    for key, value in (("episodes", one), ("successes", succeeded.long())):
        state[key].index_add_(0, types, value)
    target = final_width_mask(rows, settings)
    state["stage_episodes"].index_add_(0, types[target], one[target])
    state["stage_successes"].index_add_(0, types[target], succeeded[target].long())
    index = rows*len(state["episodes"])+types
    for key, value in (("episodes", one), ("successes", succeeded.long()), ("edge_failures", edge.long()),
                       ("recovery_timeouts", recovery.long()), ("stall_timeouts", stall.long()),
                       ("other_failures", (~succeeded & ~edge & ~recovery & ~stall).long())):
        state["width_statistics"][key].view(-1).index_add_(0, index, value)


def validate_outcome_counts(counts, statistics):
    if (not torch.equal(statistics["episodes"].sum(0), counts["episodes"])
            or not torch.equal(statistics["successes"].sum(0), counts["successes"])
            or any((v > statistics["episodes"]).any() for k, v in statistics.items() if k != "episodes")
            or (counts["stage_successes"] > counts["stage_episodes"]).any()
            or (counts["stage_successes"] > statistics["successes"][-1]).any()
            or ((counts["stage_episodes"]-counts["stage_successes"])
                > (statistics["episodes"][-1]-statistics["successes"][-1])).any()):
        raise ValueError("Inconsistent width/contact episode counts")
