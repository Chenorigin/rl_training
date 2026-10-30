"""Isaac command adapter; keep simulator imports out of pure state/runner tools."""
import torch
from task_support.stair_quiet_v1_width.course import WidthAscentCommand
from .course import course_state


class TransitionAscentCommand(WidthAscentCommand):
    def _resample_command(self, env_ids):
        # Episode-owned speed/protocol, no unrelated ten-second resampling jump.
        self._apply_course(env_ids)

    def _apply_course(self, ids):
        env, state = self._env, course_state(self._env)
        ids = torch.arange(env.num_envs, device=env.device)[ids]
        tracker = getattr(env, "tail_tracker", None)
        crossed = (tracker.last["crossed"] if tracker is not None and hasattr(tracker, "last")
                   else torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
        robot = env.scene["robot"]
        if hasattr(env, "_quiet_body_ids"):
            leading = (robot.data.body_link_pos_w[:, env._quiet_body_ids[:2], 0]-env.scene.env_origins[:, None, 0]).amax(-1)
        else:
            leading = robot.data.root_pos_w[:, 0]-env.scene.env_origins[:, 0]+.32
        gap = env.cfg.stair_tail.start_length/2-leading
        speed = state["protocol"].command(ids, env.common_step_counter*env.step_dt, gap, crossed, state["speed"])
        self.vel_command_b[ids] = 0.
        self.vel_command_b[ids, 0] = speed
        self.is_heading_env[ids] = False
        self.is_standing_env[ids] = speed <= 1.e-4
        self.heading_target[ids] = 0.
        for name in ("_stair_zero_transition_env", "_flat_zero_transition_env", "_stair_exit_zero_env", "_entry_pause_pending", "_v4_preview_pending"):
            getattr(self, name)[ids] = False

