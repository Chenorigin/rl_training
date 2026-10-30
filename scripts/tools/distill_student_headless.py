#!/usr/bin/env python3
"""Run the existing M20 distillation CLI without remote command debug arrows.

All original CLI arguments and checkpoint source checks are retained. Only the
velocity command's display flag is disabled before environment construction.
The original source files remain byte-identical for strict Safe resume checks.
"""
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]


def launch_without_command_markers(launch_app, args):
    app = launch_app(args)
    import isaaclab_tasks.utils as task_utils

    parse_env_cfg = task_utils.parse_env_cfg

    def parse_without_command_markers(*positional, **keywords):
        cfg = parse_env_cfg(*positional, **keywords)
        cfg.commands.base_velocity.debug_vis = False
        print("HEADLESS_COMMAND_DEBUG_VIS_DISABLED=base_velocity", flush=True)
        return cfg

    task_utils.parse_env_cfg = parse_without_command_markers
    return app


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--headless" not in arguments:
        arguments.append("--headless")
    path = ROOT / "scripts/reinforcement_learning/rsl_rl/distill_student.py"
    spec = importlib.util.spec_from_file_location("m20_headless_distillation_entry", path)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    original_launch_app = cli.launch_app
    cli.launch_app = lambda args: launch_without_command_markers(original_launch_app, args)
    return cli.main(arguments)


if __name__ == "__main__":
    main()
