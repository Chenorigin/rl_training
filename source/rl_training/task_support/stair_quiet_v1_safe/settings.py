"""Additional parameters; physical widths and margins retain the width ledger."""
from dataclasses import dataclass
from task_support.stair_quiet_v1_width.settings import WidthSettings

SAFETY_DIM = 48
SCAN_DIM = 187
LATERAL_SAMPLES = 11


@dataclass
class SafeSettings(WidthSettings):
    material_practice_every: int = 4
    material_practice_level: int = 7  # 1.6667 m, still a real finite staircase
    landing_risk_weight: float = -2.
    prediction_clearance_m: float = .25
    prediction_descent_m_s: float = .02

    def validate(self):
        super().validate()
        if type(self.material_practice_every) is not int or self.material_practice_every < 2:
            raise ValueError("Material practice must leave width-learning environments")
        if (type(self.material_practice_level) is not int
                or not 0 <= self.material_practice_level < len(self.widths_m)-1):
            raise ValueError("Material practice needs a wider physical staircase")
        if self.landing_risk_weight >= 0 or min(self.prediction_clearance_m, self.prediction_descent_m_s) <= 0:
            raise ValueError("Invalid predictive landing settings")
