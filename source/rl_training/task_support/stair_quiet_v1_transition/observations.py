"""Bootstrap the 35-case table before any inherited geometry observation."""
from task_support.stair_quiet_v1_safe import observations as shared
from .course import course_state


def physical_sample(env):
    course_state(env)
    return shared.physical_sample(env)


def locomotion_scan(env):
    course_state(env)
    return shared.locomotion_scan(env)


def teacher_safety(env):
    return shared.safety_features(physical_sample(env), env.cfg.stair_tail)
