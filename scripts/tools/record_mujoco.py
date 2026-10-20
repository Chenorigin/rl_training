#!/usr/bin/env python3
"""Launch MuJoCo deployment and automatically record only its viewer window.

Example:
    python scripts/tools/record_mujoco.py \
      --output logs/evaluation/mujoco_record/ascent.mp4 -- \
      --model /path/to/M20.xml --checkpoint logs/run/model.pt \
      --terrain-xml deploy/deploy_mujoco/terrains/stairs_ascent.xml \
      --viewer --output logs/evaluation/mujoco_record/ascent.npz

The deployment arguments follow ``--``. Recording starts only after the
MuJoCo viewer belonging to the launched deployment process is found, and the
MP4 is finalized when that process exits.
"""

from __future__ import annotations

import argparse
import os
import re
import signal
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEPLOY_SCRIPT = ROOT / "deploy/deploy_mujoco/deploy_mujoco.py"
WINDOW_WAIT_SECONDS = 30.0
WINDOW_POLL_SECONDS = 0.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="MP4 destination; defaults to logs/evaluation/mujoco_record/mujoco_<timestamp>.mp4",
    )
    parser.add_argument(
        "deployment_args",
        nargs=argparse.REMAINDER,
        help="Arguments for deploy_mujoco.py; put them after -- and include --viewer",
    )
    args = parser.parse_args()
    if args.deployment_args and args.deployment_args[0] == "--":
        args.deployment_args = args.deployment_args[1:]
    if "--viewer" not in args.deployment_args:
        parser.error("deployment arguments must include --viewer")
    if "--autoplay" in args.deployment_args:
        parser.error("--autoplay is not supported; use the interactive --viewer mode")
    if not DEPLOY_SCRIPT.is_file():
        parser.error(f"deployment entry not found: {DEPLOY_SCRIPT}")
    return args


def find_viewer_window(pid: int, display: str) -> str | None:
    """Return the MuJoCo X11 window ID owned by pid, if it is visible in tree."""
    try:
        tree = subprocess.run(
            ["xwininfo", "-root", "-tree"],
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "DISPLAY": display},
            timeout=3,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None

    for line in tree.splitlines():
        match = re.match(r"\s*(0x[0-9a-fA-F]+)\s+\"([^\"]*)\"", line)
        if not match or "mujoco" not in match.group(2).lower():
            continue
        window_id = match.group(1)
        try:
            props = subprocess.run(
                ["xprop", "-id", window_id, "_NET_WM_PID"],
                check=True,
                capture_output=True,
                text=True,
                env={**os.environ, "DISPLAY": display},
                timeout=3,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            continue
        pid_match = re.search(r"_NET_WM_PID\s*\(CARDINAL\)\s*=\s*(\d+)", props)
        if pid_match and int(pid_match.group(1)) == pid:
            return window_id
    return None


def stop_process(process: subprocess.Popen[bytes] | None, timeout: float = 15.0) -> int | None:
    if process is None or process.poll() is not None:
        return None if process is None else process.returncode
    process.send_signal(signal.SIGINT)
    try:
        return process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            return process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.kill()
            return process.wait()


def main() -> int:
    args = parse_args()
    if not os.environ.get("DISPLAY"):
        print("ERROR: DISPLAY is not set; run this from the graphical desktop session.", file=sys.stderr)
        return 2
    if not shutil.which("ffmpeg") or not shutil.which("xwininfo") or not shutil.which("xprop"):
        print("ERROR: required command missing; install ffmpeg and x11-utils.", file=sys.stderr)
        return 2

    display = os.environ["DISPLAY"]
    output = args.output or (
        ROOT / "logs/evaluation/mujoco_record" / f"mujoco_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    )
    output = output if output.is_absolute() else ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        print(f"ERROR: refusing to overwrite existing video: {output}", file=sys.stderr)
        return 2

    command = [sys.executable, str(DEPLOY_SCRIPT), *args.deployment_args]
    print(f"[LAUNCH] {' '.join(command)}", flush=True)
    deployment: subprocess.Popen[bytes] | None = None
    recorder: subprocess.Popen[bytes] | None = None
    def request_stop(_signum: int, _frame: object) -> None:
        if deployment is not None and deployment.poll() is None:
            deployment.send_signal(signal.SIGINT)

    old_sigint = signal.signal(signal.SIGINT, request_stop)
    old_sigterm = signal.signal(signal.SIGTERM, request_stop)
    try:
        deployment = subprocess.Popen(command, cwd=ROOT)
        deadline = time.monotonic() + WINDOW_WAIT_SECONDS
        window_id = None
        while deployment.poll() is None and time.monotonic() < deadline:
            window_id = find_viewer_window(deployment.pid, display)
            if window_id:
                break
            time.sleep(WINDOW_POLL_SECONDS)

        if deployment.poll() is not None:
            return deployment.returncode or 0
        if not window_id:
            print(
                f"ERROR: MuJoCo viewer window for PID {deployment.pid} was not found "
                f"within {WINDOW_WAIT_SECONDS:.0f}s; stopping deployment.",
                file=sys.stderr,
                flush=True,
            )
            deployment.send_signal(signal.SIGINT)
            deployment.wait(timeout=10)
            return 2

        print(f"[RECORD] window={window_id} pid={deployment.pid} -> {output}", flush=True)
        recorder_command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-n",
            "-f",
            "x11grab",
            "-framerate",
            "30",
            "-window_id",
            window_id,
            "-i",
            display,
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(output),
        ]
        recorder = subprocess.Popen(recorder_command, cwd=ROOT, stdin=subprocess.DEVNULL)

        while deployment.poll() is None:
            if recorder.poll() is not None:
                print(
                    f"ERROR: ffmpeg stopped unexpectedly (exit {recorder.returncode}); stopping deployment.",
                    file=sys.stderr,
                    flush=True,
                )
                deployment.send_signal(signal.SIGINT)
                deployment.wait(timeout=10)
                return 2
            time.sleep(0.25)

        deployment_status = deployment.returncode
        return deployment_status if deployment_status is not None else 0
    except KeyboardInterrupt:
        return 130
    except subprocess.TimeoutExpired:
        print("ERROR: deployment did not stop after SIGINT; stopping it.", file=sys.stderr)
        if deployment is not None and deployment.poll() is None:
            deployment.terminate()
            deployment.wait(timeout=5)
        return 2
    finally:
        if deployment is not None and deployment.poll() is None:
            deployment.send_signal(signal.SIGINT)
            try:
                deployment.wait(timeout=10)
            except subprocess.TimeoutExpired:
                deployment.terminate()
                deployment.wait(timeout=5)
        if recorder is not None:
            stop_process(recorder)
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)
        if output.exists():
            print(f"[SAVED] {output}", flush=True)
if __name__ == "__main__":
    raise SystemExit(main())
