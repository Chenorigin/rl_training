"""Original ascent commands with explicit student waiting/start/stop examples."""
import torch
from task_support.stair_quiet_v1_width.course import WidthAscentCommand
from .start_stop import environment_masks


class StudentStartStopCommand(WidthAscentCommand):
    def __init__(self,cfg,env):
        from .terrain_course import course_state
        course_state(env)
        super().__init__(cfg,env)

    def _apply_course(self, ids):
        super()._apply_course(ids)
        _, paused = environment_masks(self._env)
        self.vel_command_b[ids] = torch.where(paused[ids, None], 0., self.vel_command_b[ids])
        self.is_standing_env[ids] |= paused[ids]
