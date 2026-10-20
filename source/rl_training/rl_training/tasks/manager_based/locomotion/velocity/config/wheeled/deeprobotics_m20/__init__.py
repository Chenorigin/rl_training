# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause
# 
# # Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""M20 tasks; each canonical task exposes a consistently named configuration trio."""
import gymnasium as gym

from . import agents

_MDP = "rl_training.tasks.manager_based.locomotion.velocity.mdp"

# (Gym ID, canonical task name, Env class, EnvCfg class, PPO/distillation class)
_TASKS = (
    ('Rough-Deeprobotics-M20-IdleStopMotion-v1', 'idle_stop_motion_v1', 'M20IdleStopMotionV1Env', 'DeeproboticsM20IdleStopMotionV1EnvCfg', 'DeeproboticsM20IdleStopMotionV1PPORunnerCfg'),
    ('Rough-Deeprobotics-M20-Student-v3.5', 'student_v3_5', 'M20StudentV35Env', 'DeeproboticsM20StudentV35EnvCfg', 'DeeproboticsM20StudentV35DistillationCfg'),
    ('Rough-Deeprobotics-M20-Student-v3', 'student_v3', 'M20StudentV3Env', 'DeeproboticsM20StudentV3EnvCfg', 'DeeproboticsM20StudentV3DistillationCfg'),
    ('Rough-Deeprobotics-M20-Student-v1', 'student_v1', 'M20StudentV1Env', 'DeeproboticsM20StudentV1EnvCfg', 'DeeproboticsM20StudentV1DistillationCfg'),
    ('Rough-Deeprobotics-M20-Student-v0', 'student_v0', 'M20StudentV0Env', 'DeeproboticsM20StudentV0EnvCfg', 'DeeproboticsM20StudentV0DistillationCfg'),
    ('Rough-Deeprobotics-M20-GaitRefinement-v2', 'gait_refinement_v2', 'M20GaitRefinementV2Env', 'DeeproboticsM20GaitRefinementV2EnvCfg', 'DeeproboticsM20GaitRefinementV2PPORunnerCfg'),
    ('Rough-Deeprobotics-M20-QuietLanding-v0', 'quiet_landing', 'M20QuietLandingEnv', 'DeeproboticsM20QuietLandingEnvCfg', 'DeeproboticsM20QuietLandingPPORunnerCfg'),
    ('Flat-Deeprobotics-M20-v0', 'flat', 'M20FlatEnv', 'DeeproboticsM20FlatEnvCfg', 'DeeproboticsM20FlatPPORunnerCfg'),
    ('Rough-Deeprobotics-M20-v0', 'rough', 'M20RoughEnv', 'DeeproboticsM20RoughEnvCfg', 'DeeproboticsM20RoughPPORunnerCfg'),
    ('Rough-Deeprobotics-M20-StairTeacher-v0', 'stair_teacher', 'M20StairTeacherEnv', 'DeeproboticsM20StairTeacherEnvCfg', 'DeeproboticsM20StairTeacherPPORunnerCfg'),
    ('Rough-Deeprobotics-M20-PreTeacher-v0', 'pre_teacher', 'M20PreTeacherEnv', 'DeeproboticsM20PreTeacherEnvCfg', 'DeeproboticsM20PreTeacherPPORunnerCfg'),
    ('Rough-Deeprobotics-M20-StairStudent-v0', 'stair_student', 'M20StairStudentEnv', 'DeeproboticsM20StairStudentEnvCfg', 'DeeproboticsM20StairStudentPPORunnerCfg'),
)

for task_id, task_name, env_class, env_cfg_class, ppo_cfg_class in _TASKS:
    entry_point = f"{_MDP}.{task_name}_env:{env_class}"
    kwargs = {
        "env_cfg_entry_point": f"{__name__}.{task_name}_env_cfg:{env_cfg_class}",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.{task_name}_ppo_cfg:{ppo_cfg_class}",
    }
    gym.register(id=task_id, entry_point=entry_point, disable_env_checker=True, kwargs=kwargs)
