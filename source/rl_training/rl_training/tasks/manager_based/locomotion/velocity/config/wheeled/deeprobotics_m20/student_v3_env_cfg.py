"""Front-only student v3; physical task remains the parking teacher's task."""
from isaaclab.utils import configclass

from m20_student_v1.configuration import StudentV1PerceptionCfg
from m20_student_v1.v3_configuration import student_v3_settings
from .student_v1_env_cfg import DeeproboticsM20StudentV1EnvCfg


@configclass
class DeeproboticsM20StudentV3EnvCfg(DeeproboticsM20StudentV1EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        settings = student_v3_settings()
        self.student_perception = StudentV1PerceptionCfg(**settings["perception"])
        self.student_perception.validate()
        self.student_history_steps = settings["history_steps"]
