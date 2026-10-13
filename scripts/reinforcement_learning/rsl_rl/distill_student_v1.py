#!/usr/bin/env python3
"""Compatibility alias. Implementation lives in task-selected distill_student.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from distill_student import (
    main as unified_main, parse_args, settings_v1 as settings,
    preflight_v1 as preflight, train_or_validate_v1 as train_or_validate,
)


def main(argv=None):
    return unified_main(argv, default_task="Rough-Deeprobotics-M20-Student-v1")


if __name__ == "__main__":
    main()
