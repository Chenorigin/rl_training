"""V3 sensor task with a separate raw-command group for the residual teacher."""
import torch
from .v3_5_feedback import SensorHeadingFeedback
from m20_student_v1 import environment as base
from m20_student_v1.perception import quaternion_rotation


def raw_motion_control(env):
    return env.command_manager.get_command('base_velocity').clone()


def feedback(env):
    if not hasattr(env, '_v3_5_heading'):
        env._v3_5_heading = SensorHeadingFeedback(env.num_envs, env.device, env.cfg.motion_teacher_tuning)
        base.sensor_state(env).command_transform = env._v3_5_heading
    return env._v3_5_heading


def effective_velocity_commands(env, command_name='base_velocity'):
    if command_name != 'base_velocity':
        raise ValueError('V3.5 feedback is bound to base_velocity')
    return student_observation(env)[:, 6:9]


def student_observation(env):
    feedback(env)
    robot = env.scene['robot']
    rotation = quaternion_rotation(robot.data.root_quat_w)
    yaw = torch.atan2(rotation[:, 1, 0], rotation[:, 0, 0])
    return base.sensor_state(env).observe(base.clean_teacher_proprio(env, raw_motion_control(env)),
        base.capture_hits(env), robot.data.root_pos_w, yaw, rotation, int(env.common_step_counter))


def reset_feedback(env, env_ids):
    feedback(env).reset(env_ids)
