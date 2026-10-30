"""Task identity and defaults; numerical settings live in the JSON ledger."""
from dataclasses import dataclass, field
import json
from pathlib import Path

from m20_student_v1.configuration import DeeproboticsM20StudentV1DistillationCfg

ROOT = Path(__file__).resolve().parents[4]
TASK = "Rough-Deeprobotics-M20-Student-v3-Safe"
CONFIG_PATH = ROOT / "configs/distillation/student_v3_safe.json"
TEACHER_CLASS = "task_support.student_v3_safe.teacher.SafeTeacherModel"
TEACHER_GROUPS = ["teacher", "motion_control", "locomotion_scan", "safety"]
ALGORITHM_CLASS = "task_support.student_v3_safe.gait_policy_gradient.GaitStudentDistillation"


def default_settings():
    return json.loads(CONFIG_PATH.read_text())


@dataclass
class DeeproboticsM20StudentV3SafeDistillationCfg(DeeproboticsM20StudentV1DistillationCfg):
    idle_practice: dict = field(default_factory=dict)
    diagonal_reward: dict = field(default_factory=dict)
    terrain_course: dict = field(default_factory=dict)

    def __post_init__(self):
        settings = default_settings()
        self.history_steps = settings["history_steps"]
        for key, value in settings["training"].items():
            current = getattr(self, key)
            if isinstance(current, dict):
                current.update(value)
            else:
                setattr(self, key, value)
        self.teacher.update(class_name=TEACHER_CLASS, residual_hidden_dims=[128, 64])
        self.algorithm["class_name"] = ALGORITHM_CLASS
        self.obs_groups["teacher"] = list(TEACHER_GROUPS)
