"""V3.5 adds residual-teacher commands without changing the v3 sensor ABI."""
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg, EventTermCfg
from isaaclab.utils import configclass
from m20_student_v1 import v3_5_environment as glue
from m20_student_v1.configuration import StudentV1PerceptionCfg
from m20_student_v1.v3_5_configuration import student_v3_5_settings
from .student_v3_env_cfg import DeeproboticsM20StudentV3EnvCfg


@configclass
class MotionControlCfg(ObservationGroupCfg):
    raw_command = ObservationTermCfg(func=glue.raw_motion_control)
    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class DeeproboticsM20StudentV35EnvCfg(DeeproboticsM20StudentV3EnvCfg):
    motion_teacher_tuning: dict = {}
    def __post_init__(self):
        super().__post_init__()
        settings = student_v3_5_settings()
        self.student_perception = StudentV1PerceptionCfg(**settings['perception'])
        self.student_perception.validate()
        self.student_history_steps = settings['history_steps']
        self.observations.motion_control = MotionControlCfg()
        self.observations.teacher.velocity_commands.func = glue.effective_velocity_commands
        self.observations.policy.observation.func = glue.student_observation
        self.observations.critic.observation.func = glue.student_observation
        self.events.student_v3_5_heading_reset = EventTermCfg(func=glue.reset_feedback, mode='reset')
