"""Student v3 task defaults using the existing 789D-capable student backend."""
from dataclasses import dataclass
import json
from pathlib import Path

from m20_student_v1.configuration import DeeproboticsM20StudentV1DistillationCfg


V3_CONFIG_PATH = Path(__file__).resolve().parents[4] / "configs/distillation/student_v3_front_only.json"


def student_v3_settings():
    return json.loads(V3_CONFIG_PATH.read_text())


@dataclass
class DeeproboticsM20StudentV3DistillationCfg(DeeproboticsM20StudentV1DistillationCfg):
    def __post_init__(self):
        settings = student_v3_settings()
        self.history_steps = settings["history_steps"]
        for name, value in settings["training"].items():
            current = getattr(self, name)
            if isinstance(current, dict):
                current.update(value)
            else:
                setattr(self, name, value)
