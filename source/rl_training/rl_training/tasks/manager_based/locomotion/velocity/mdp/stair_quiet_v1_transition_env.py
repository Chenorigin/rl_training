"""Own 35-case initialization/recording, reuse the original physical step loop."""
from .stair_quiet_v1_env import M20StairQuietV1Env
from task_support.stair_quiet_v1_width.terrain import neutral_wheel_material
from task_support.stair_quiet_v1_transition.course import course_state, record_episodes
from task_support.stair_quiet_v1_transition.observations import physical_sample
from task_support.stair_quiet_v1_transition.signals import TransitionTracker


class M20StairQuietV1TransitionEnv(M20StairQuietV1Env):
    def __init__(self, cfg, **kwargs):
        # Width/Safe constructors hard-code 30 columns: do not call them with
        # 35 physical cases. Observation and command bootstraps own the table.
        super().__init__(cfg, **kwargs)
        course_state(self)
        neutral_wheel_material(self)
        self.tail_tracker = TransitionTracker(self.num_envs, self.device, cfg.stair_tail)
        self.tail_tracker.reset(slice(None), self.scene["robot"].data.root_pos_w[:, 0]-self.scene.env_origins[:, 0])

    def step(self, action):
        self.tail_tracker.begin_step()
        return super().step(action)

    def _after_quiet_physics_sample(self):
        state = course_state(self)
        protocol = state["protocol"]
        command_paused = (protocol.mode == 1) & (protocol.phase == 1) & ~protocol.stopped
        term = self.command_manager.get_term("base_velocity")
        command_paused &= term.vel_command_b[:, 0].abs() < 1.e-4
        self.tail_tracker.update(stopped=protocol.stopped, closed_riser=state["closed"],
                                 command_paused=command_paused, **physical_sample(self))

    def _reset_idx(self, env_ids):
        tracker = getattr(self, "tail_tracker", None)
        metrics = (record_episodes(self, env_ids) if tracker is not None
                   and not getattr(self, "_tail_skip_episode_record", False) else {})
        if tracker is not None and hasattr(tracker, "last"):
            tracker.last["crossed"][env_ids] = False
        super()._reset_idx(env_ids)
        self.extras.setdefault("log", {}).update(metrics)
        if tracker is not None:
            tracker.reset(env_ids, self.scene["robot"].data.root_pos_w[env_ids, 0]-self.scene.env_origins[env_ids, 0])
