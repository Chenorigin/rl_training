"""Keep archived student class paths available without importing Isaac tasks."""
from pathlib import Path

_support = Path(__file__).resolve().parents[1] / "task_support"
__path__ = [str(_support / task) for task in (
    "student_v1", "student_v3", "student_v3_5",
)]
