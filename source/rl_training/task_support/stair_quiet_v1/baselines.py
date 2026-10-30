"""Immutable input identities; these are source checkpoints, never output paths."""
from pathlib import Path
import hashlib

PROJECT_ROOT = Path(__file__).resolve().parents[4]
TEACHER = "deploy/models/idle_stop_motion_v1/idle_stop_motion_v1_46400.pt"
STUDENT = "deploy/models/student_v3_5/model_21000.pt"
TEACHER_SHA256 = "2326cd0aa65219334ed1e2fd7a63dbc880d351f61b512d34ebcaa24ed6bafe8c"
STUDENT_SHA256 = "1f6f14857776b64200f6a2d4c81ad7af48e738604bbcec67acb8c7113a6dae73"
TASK = "Rough-Deeprobotics-M20-StairQuiet-v1"


def checked_path(value, expected):
    path = Path(value).expanduser()
    path = (path if path.is_absolute() else PROJECT_ROOT / path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError(f"StairQuiet baseline SHA256 mismatch: {path}")
    return path
