"""V3.5 retains v3 sensors, 789D student and the existing distillation backend."""
from dataclasses import dataclass
from pathlib import Path
import json
from m20_student_v1.v3_configuration import DeeproboticsM20StudentV3DistillationCfg

V3_5_CONFIG_PATH = Path(__file__).resolve().parents[4] / 'configs/distillation/student_v3_5.json'


def student_v3_5_settings():
    return json.loads(V3_5_CONFIG_PATH.read_text())


@dataclass
class DeeproboticsM20StudentV35DistillationCfg(DeeproboticsM20StudentV3DistillationCfg):
    def __post_init__(self):
        super().__post_init__()
        settings = student_v3_5_settings()
        self.history_steps = settings['history_steps']
        for name, value in settings['training'].items():
            current = getattr(self, name)
            if isinstance(current, dict):
                current.update(value)
            else:
                setattr(self, name, value)
        self.teacher.update(class_name='m20_student_v1.v3_5_teacher.MotionTeacherModel', residual_hidden_dims=[128, 64])
        self.obs_groups['teacher'] = ['teacher', 'motion_control']
