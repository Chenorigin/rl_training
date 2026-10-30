#!/usr/bin/env python3
"""Train the M20 CTS teacher and student; preflight/validation never optimize."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'source/rl_training'))

from task_support.cts.entry import main

if __name__ == '__main__':
    main()
