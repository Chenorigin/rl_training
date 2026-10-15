"""Finish the remaining recordings, one CPU physics process at a time."""
from pathlib import Path
import json
import os
import subprocess
import sys

base = Path(__file__).resolve().parent
env = dict(os.environ, MUJOCO_GL="osmesa", LP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
           OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
jobs = [("fixed", "零命令", "fixed_zero"),
        ("baseline", "平地启停", "baseline_stop"),
        ("fixed", "平地启停", "fixed_stop"),
        ("baseline", "仅w", "baseline_stairs"),
        ("fixed", "仅w", "fixed_stairs")]
results = []
for label, case, log_name in jobs:
    print(f"Recording {label} / {case}", flush=True)
    log = base / (log_name + ".log")
    with log.open("x") as stream:
        result = subprocess.run(["/usr/bin/time", "-v", "timeout", "240", sys.executable,
                                 str(base / "record.py"), label, case],
                                env=env, stdout=stream, stderr=subprocess.STDOUT)
    results.append(dict(label=label, case=case, returncode=result.returncode, log=str(log)))
    (base / "record_remaining_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
    if result.returncode:
        raise SystemExit(f"Recording failed: {log}, rc={result.returncode}")
    print(f"Completed {label} / {case}", flush=True)
