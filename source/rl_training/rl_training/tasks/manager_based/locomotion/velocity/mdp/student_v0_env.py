"""V0 student environment integration."""
from m20_student_v0.contract import STUDENT_OBS_DIM, TEACHER_OBS_DIM
from .quiet_landing_env import M20QuietLandingEnv


class M20StudentV0Env(M20QuietLandingEnv):
    POLICY_OBSERVATION_DIM = STUDENT_OBS_DIM

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        if self.observation_manager.group_obs_dim["teacher"] != (TEACHER_OBS_DIM,):
            raise RuntimeError("V0 distillation requires a clean 244D teacher observation")
