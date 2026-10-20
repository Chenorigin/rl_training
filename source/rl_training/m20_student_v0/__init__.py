"""Keep archived v0 class paths available from the task support directory."""
from pathlib import Path

__path__ = [str(Path(__file__).resolve().parents[1] / "task_support/student_v0")]
