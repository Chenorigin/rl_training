"""Student-only start/stop practice; no changes to teacher physics or inputs."""
import math
import torch


def validate_practice(settings, stall_timeout):
    if set(settings) != {"every", "initial_hold_s", "move_s", "stop_s"}:
        raise ValueError("Invalid student idle-practice settings")
    if type(settings["every"]) is not int or settings["every"] < 1:
        raise ValueError("idle-practice every must be a positive integer")
    for key in ("initial_hold_s", "move_s", "stop_s"):
        value = settings[key]
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid idle-practice {key}")
    # Preserve the original failure threshold, even if the dog never moves
    # during the brief forward pulse. Do not mask stalls or physical failures.
    if sum(settings[k] for k in ("initial_hold_s", "move_s", "stop_s")) >= stall_timeout:
        raise ValueError("Student start/stop practice must finish before the original stall timeout")


def practice_masks(ids, reset_counts, elapsed_s, seed, settings):
    """Rotate practice episodes across envs without changing material routing."""
    selected = (ids + reset_counts + seed) % settings["every"] == 0
    first_end = settings["initial_hold_s"]
    second_start = first_end + settings["move_s"]
    second_end = second_start + settings["stop_s"]
    paused = selected & ((elapsed_s < first_end) | ((elapsed_s >= second_start) & (elapsed_s < second_end)))
    return selected, paused


def environment_masks(env):
    ids = torch.arange(env.num_envs, device=env.device)
    return practice_masks(ids, env._tail_course["assign_count"], env.episode_length_buf * env.step_dt,
                          int(env.cfg.seed), env.cfg.student_idle_practice)
