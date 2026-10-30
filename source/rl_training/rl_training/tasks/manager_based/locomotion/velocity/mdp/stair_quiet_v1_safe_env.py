"""Safe teacher: reuse width physics, own predictive tracker and reset evidence."""
from .stair_quiet_v1_width_env import M20StairQuietV1WidthEnv
from .stair_quiet_v1_env import M20StairQuietV1Env
from task_support.stair_quiet_v1_safe.course import course_state, record_episodes
from task_support.stair_quiet_v1_safe.observations import physical_sample
from task_support.stair_quiet_v1_safe.signals import SafeTailTracker


class M20StairQuietV1SafeEnv(M20StairQuietV1WidthEnv):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        course_state(self)
        self.tail_tracker = SafeTailTracker(self.num_envs, self.device, cfg.stair_tail)
        self.tail_tracker.reset(slice(None), self.scene["robot"].data.root_pos_w[:, 0]-self.scene.env_origins[:, 0])

    def _after_quiet_physics_sample(self):
        self.tail_tracker.update(**physical_sample(self))

    def _reset_idx(self, env_ids):
        tracker = getattr(self, "tail_tracker", None)
        metrics = (record_episodes(self, env_ids) if isinstance(tracker, SafeTailTracker)
                   and not getattr(self, "_tail_skip_episode_record", False) else {})
        if tracker is not None and hasattr(tracker, "last"):
            tracker.last["crossed"][env_ids] = False
        # Bypass the width recorder: it gates material changes on 1 m success.
        M20StairQuietV1Env._reset_idx(self, env_ids)
        self.extras.setdefault("log", {}).update(metrics)
        if tracker is not None:
            tracker.reset(env_ids, self.scene["robot"].data.root_pos_w[env_ids, 0]-self.scene.env_origins[env_ids, 0])
