"""Student observation wiring and completed-episode course persistence."""
import torch

from m20_student_v1 import environment as common
from m20_student_v1 import v3_5_environment as motion

__all__ = ["common", "motion", "completed_failure", "sampling_tags", "sensor_metrics",
           "course_snapshot", "restore_course", "initialize_course", "checked_tree"]


def completed_failure(env):
    """A completed Safe episode failed unless it genuinely finished ascent."""
    from task_support.stair_quiet_v1_width.signals import failure_mask
    failed = failure_mask(env)
    done = env.termination_manager.terminated | env.termination_manager.time_outs
    succeeded = env.tail_tracker.success & ~failed
    return done & ~succeeded


def sampling_tags(env):
    robot = env.scene["robot"]
    return common.classify_state(
        env.scene.sensors["height_scanner"].data.ray_hits_w,
        robot.data.projected_gravity_b, env.command_manager.get_command("base_velocity"),
        torch.where(env.stair_quiet_landing.terrain_valid, env.stair_quiet_landing.clearance, torch.inf),
        robot.data.body_lin_vel_w[:, env._quiet_body_ids, 2], env.stair_quiet_tracker.contact,
        env.scene.terrain.terrain_levels,
    )


def sensor_metrics(env, obs):
    state = common.sensor_state(env)
    usable = obs["policy"][:, 244:431] > .5
    return {
        "StudentSensor/valid_fraction": usable.float().mean(),
        "StudentSensor/confidence_mean": obs["policy"][:, 431:618].mean(),
        "StudentSensor/visible_fraction": state.last_visibility.float().mean(),
        "StudentSensor/oldest_valid_age_s": torch.where(usable, state.cell_age, 0.).max(),
        "StudentSensor/level": float(state.cfg.level),
    }


def clone_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: clone_tree(v) for k, v in value.items()}
    raise ValueError("Unsupported course state")


def checked_tree(saved, target):
    """Validate the whole snapshot before mutating any live environment state."""
    if isinstance(target, dict):
        if not isinstance(saved, dict) or set(saved) != set(target):
            raise ValueError("Student course state keys changed")
        return {k: checked_tree(saved[k], target[k]) for k in target}
    if (not isinstance(saved, torch.Tensor) or saved.shape != target.shape
            or saved.dtype != target.dtype or not torch.isfinite(saved).all()):
        raise ValueError("Student course tensor shape/dtype/nonfinite mismatch")
    return saved.to(target.device).clone()


def course_snapshot(env):
    from .terrain_course import course_state
    state = course_state(env)
    return {"stage": int(env.tail_stage), "width": state["width_curriculum"].state_dict(),
            "state": clone_tree({k: v for k, v in state.items() if k != "width_curriculum"})}


def reset_physics(env):
    env._tail_skip_episode_record = True
    try:
        with torch.inference_mode():
            env.reset()
    finally:
        env._tail_skip_episode_record = False


def restore_course(env, saved):
    from .terrain_course import course_state, validate_counts
    from task_support.stair_quiet_v1_width.curriculum import WidthCurriculum
    state = course_state(env)
    if type(saved.get("stage")) is not int or saved["stage"] not in (0, 1, 2):
        raise ValueError("Invalid student material stage")
    values = checked_tree(saved.get("state"), {k: v for k, v in state.items() if k != "width_curriculum"})
    # A persisted ledger must never redefine the mesh that was just imported.
    for name in state:
        if name.endswith('_table') or name == 'width_values':
            if not torch.equal(values[name], state[name]):
                raise ValueError("Student terrain geometry/material table changed: " + name)
    curriculum = WidthCurriculum(env.num_envs, env.device, env.cfg.stair_tail)
    curriculum.load_state_dict(saved["width"])
    validate_counts(values, env.cfg.stair_tail)
    if not torch.equal(values["material_practice"], state["material_practice"]):
        raise ValueError("Student material-practice allocation changed")
    for name in ("assign_count", "episodes", "successes", "stage_episodes", "stage_successes"):
        if (values[name] < 0).any():
            raise ValueError("Negative student course counters")
    state.update(values)
    state["width_curriculum"].load_state_dict(saved["width"])
    env.tail_stage = saved["stage"]
    reset_physics(env)


def initialize_course(env, manifest):
    """Keep teacher difficulty coverage; start independent student statistics."""
    from .terrain_course import course_state
    state = course_state(env)
    levels = manifest.get("width_curriculum", {}).get("level", [])
    if not levels or any(type(x) is not int or not 0 <= x < len(env.cfg.stair_tail.widths_m) for x in levels):
        raise ValueError("Safe teacher has invalid width targets")
    stage = manifest.get("tail_curriculum", {}).get("stage")
    if type(stage) is not int or stage not in (0, 1, 2):
        raise ValueError("Safe teacher has invalid material stage")
    source = torch.tensor(levels, device=env.device)
    index = torch.arange(env.num_envs, device=env.device) * len(source) // env.num_envs
    state["width_curriculum"].level.copy_(source[index])
    state["width_curriculum"].level[state["material_practice"]] = env.cfg.stair_tail.material_practice_level
    state["width_curriculum"].success_streak.zero_()
    env.tail_stage = stage
    reset_physics(env)
