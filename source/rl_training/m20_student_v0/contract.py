"""Deployment observation contract shared by simulation and export."""

PROPRIO_DIM = 57
GRID_ROWS, GRID_COLS = 11, 17
GRID_RESOLUTION = 0.1
HEIGHT_DIM = GRID_ROWS * GRID_COLS
TEACHER_OBS_DIM = PROPRIO_DIM + HEIGHT_DIM
STUDENT_OBS_DIM = PROPRIO_DIM + 3 * HEIGHT_DIM
ACTION_DIM = 16
HEIGHT_OFFSET = 0.5
HEIGHT_CLIP = (-1.0, 1.0)


def observation_contract():
    return {
        "version": "m20_student_v0_618",
        "student_observation_dim": STUDENT_OBS_DIM,
        "teacher_observation_dim": TEACHER_OBS_DIM,
        "action_dim": ACTION_DIM,
        "channels": ["proprio", "height", "validity", "confidence"],
        "slices": {"proprio": [0, 57], "height": [57, 244],
                   "validity": [244, 431], "confidence": [431, 618]},
        "grid": {"rows_y": GRID_ROWS, "columns_x": GRID_COLS,
                 "resolution_m": GRID_RESOLUTION, "x_range_m": [-0.8, 0.8],
                 "y_range_m": [-0.5, 0.5], "order": "y rows, x columns, ascending",
                 "alignment": "gravity aligned, robot yaw"},
        "height": {"formula": "clip(base_z - terrain_z - offset, min, max)",
                   "offset_m": HEIGHT_OFFSET, "clip": list(HEIGHT_CLIP)},
        "unknown": {"height": 0.0, "validity": 0.0, "confidence": 0.0},
        "confidence": "validity / (1 + variance / reference_std**2) * exp(-age / age_tau)",
        "confidence_is_calibrated_probability": False,
        "normalization": "teacher proprio scales retained; no running observation normalization",
        "actions": "12 leg position residuals, then 4 wheel velocity commands; teacher scaling retained",
    }
