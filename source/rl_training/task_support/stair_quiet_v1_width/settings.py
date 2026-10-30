"""Single ledger of engineering candidates; none are measured wood properties."""
from dataclasses import dataclass, asdict
import math


@dataclass(frozen=True)
class ContactProfile:
    name: str
    static_friction: float
    dynamic_friction: float
    restitution: float
    stiffness: float
    damping: float
    stage: int


# Force springs in N/m and N s/m. A 200 N load gives 0.5--5 mm
# nominal spring displacement. Static geometry does not simulate board modes.
PROFILES = (
    ContactProfile("reference_effective", .85, .70, .0, 150000., 1500., 0),
    ContactProfile("lower_grip", .55, .40, .0, 150000., 1500., 1),
    ContactProfile("compliant_damped", .85, .70, .0, 40000., 1400., 1),
    ContactProfile("stiff_less_damped", 1.05, .85, .0, 400000., 500., 2),
    ContactProfile("low_grip_compliant", .40, .28, .0, 60000., 700., 2),
)


@dataclass(frozen=True)
class Geometry:
    name: str
    height: float
    depth: float
    width: float = 1.0
    steps: int = 6
    closed_riser: bool = False


GEOMETRIES = (
    Geometry("reference_15_30_open", .15, .30),
    Geometry("reference_15_20_open", .15, .20),
    Geometry("easy_12_30_open", .12, .30),
    Geometry("varied_18_25_open", .18, .25),
    Geometry("varied_15_25_closed", .15, .25, closed_riser=True),
    Geometry("varied_17_35_open", .17, .35),
)


@dataclass
class WidthSettings:
    # Floor is below the start landing. No stair-height side catch platform.
    tile_length: float = 8.0
    start_length: float = 2.0
    board_thickness: float = .035
    floor_depth: float = .5
    # FL/FR and HL/HR are configured independently. Positive values move the
    # safe region inward from each physical edge; all four feet are checked.
    front_lateral_margin: float = .04
    rear_lateral_margin: float = .04
    front_tread_margin: float = .03
    rear_tread_margin: float = .03
    body_lateral_margin: float = .04
    # Effective finite contact patch, NOT the full wheel's horizontal diameter.
    # A wheel can be supported on a 20 cm tread without its whole disk above it.
    contact_patch_radius: float = .02
    wheel_radius: float = .09
    wheel_half_length: float = .027
    landing_grace_s: float = .08
    support_force_N: float = 15.
    slip_speed_m_s: float = .12
    slip_on_s: float = .06
    recovery_hold_s: float = .20
    # Failure clocks measure missing progress/support, not elapsed slip time.
    recovery_timeout_s: float = 1.5
    recovery_support_loss_s: float = .50
    forward_progress_m: float = .02
    no_progress_timeout_s: float = 4.
    min_progress_speed_m_s: float = .04
    prediction_s: float = .12
    body_half_length: float = .40
    body_half_width: float = .16
    success_hold_s: float = .25
    # Match the source teacher's dedicated straight-stair command practice.
    # Slow travel is a separate skill; it is not a proxy for light touchdown.
    speed_range: tuple = (.40, .80)
    reset_y_range: float = .04
    reset_yaw_range: float = .04
    # Candidate physical scales, not baseline percentiles or calibrated sound.
    tail_force_N: float = 650.
    tail_force_scale_N: float = 400.
    tail_impulse_N_s: float = 12.
    tail_impulse_scale_N_s: float = 8.
    tail_window_s: float = .04
    stage_min_episodes: int = 128
    stage_success_fraction: float = .85
    # Negative cost weights; completion/failure are task events.
    edge_weight: float = -4.
    foothold_weight: float = -2.
    slip_weight: float = -1.
    recovery_impact_weight: float = -.08
    tail_force_weight: float = -.08
    tail_impulse_weight: float = -.06
    failure_weight: float = -20.
    success_weight: float = 8.
    progress_weight: float = 2.
    # 0: safety + stability; 1: contact diversity; 2: full diversity + tails.
    initial_stage: int = 0
    automatic_stages: bool = True
    objective: str = "stable_light"  # "control" disables new mechanical costs only

    widths_m: tuple = tuple(4.0-i/3 for i in range(10))
    tile_width: float = 5.0
    initial_width_level: int = 0
    width_curriculum_enabled: bool = True
    consecutive_successes_to_narrow: int = 2
    # Width starts at the wide row, contact stage stays at reference. New
    # contact stages still require measured performance on the final width.

    def validate(self):
        for name, value in asdict(self).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool) and not math.isfinite(value):
                raise ValueError(f"Nonfinite WidthSettings.{name}")
        if self.objective not in ("stable_light", "control") or self.initial_stage not in (0, 1, 2):
            raise ValueError("Invalid objective or initial stage")
        for key in ("front_lateral_margin", "rear_lateral_margin", "front_tread_margin", "rear_tread_margin",
                    "body_lateral_margin", "contact_patch_radius", "tail_force_scale_N",
                    "tail_impulse_scale_N_s", "tail_window_s", "landing_grace_s", "no_progress_timeout_s",
                    "recovery_timeout_s", "recovery_support_loss_s", "recovery_hold_s", "forward_progress_m"):
            if getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive")
        if not 0 < self.stage_success_fraction <= 1 or self.stage_min_episodes < 1:
            raise ValueError("Invalid stage gate")
        if (len(self.speed_range) != 2 or any(not math.isfinite(v) or v <= 0 for v in self.speed_range)
                or self.speed_range[0] > self.speed_range[1]):
            raise ValueError("Invalid forward command speed range")
        for g in GEOMETRIES:
            if g.depth <= 2 * (self.contact_patch_radius + max(self.front_tread_margin, self.rear_tread_margin)):
                raise ValueError("Empty longitudinal safe foothold region")
            if g.width <= 2 * (self.wheel_radius + self.wheel_half_length
                              + max(self.front_lateral_margin, self.rear_lateral_margin)):
                raise ValueError("Empty lateral safe foothold region")

        if (not self.widths_m or any(not math.isfinite(w) or w <= 0 for w in self.widths_m)
                or any(a <= b for a, b in zip(self.widths_m, self.widths_m[1:]))
                or not math.isclose(self.widths_m[-1], 1.0, abs_tol=1e-9)):
            raise ValueError("Widths must strictly decrease to the 1 m target")
        if self.widths_m[0] >= self.tile_width:
            raise ValueError("Tile must leave a real lateral drop beside the widest staircase")
        if type(self.initial_width_level) is not int or not 0 <= self.initial_width_level < len(self.widths_m):
            raise ValueError("Invalid initial width level")
        if type(self.consecutive_successes_to_narrow) is not int or self.consecutive_successes_to_narrow < 1:
            raise ValueError("Width promotion requires a positive integer success count")


def cases():
    return tuple((g, p) for g in GEOMETRIES for p in PROFILES)


def width_for_difficulty(difficulty, settings):
    """Installed Isaac generator gives row i difficulty in [i/L,(i+1)/L)."""
    if not math.isfinite(difficulty) or not 0 <= difficulty <= 1:
        raise ValueError("Width-row difficulty must be finite in [0,1]")
    row = min(int(difficulty*len(settings.widths_m)), len(settings.widths_m)-1)
    return settings.widths_m[row]
