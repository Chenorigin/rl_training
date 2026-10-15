# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause
# 
# # Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym

from . import agents

for task_id, env_class in (
    ("Rough-Deeprobotics-M20-Student-v1", "DeeproboticsM20StudentV1EnvCfg"),
    ("Rough-Deeprobotics-M20-StudentV15-v1", "DeeproboticsM20StudentV15V1EnvCfg"),
):
    gym.register(
        id=task_id,
        entry_point=f"{__name__}.student_v1_env_cfg:M20StudentV1Env",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.student_v1_env_cfg:{env_class}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.student_v1_distillation_cfg:DeeproboticsM20StudentV1DistillationCfg",
        },
    )

for task_id, env_class in (
    ("Rough-Deeprobotics-M20-Student-v0", "DeeproboticsM20StudentV0EnvCfg"),
    ("Rough-Deeprobotics-M20-StudentV15-v0", "DeeproboticsM20StudentV15V0EnvCfg"),
):
    gym.register(
        id=task_id,
        entry_point=f"{__name__}.student_v0_env_cfg:M20StudentV0Env",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.student_v0_env_cfg:{env_class}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.student_v0_distillation_cfg:DeeproboticsM20StudentV0DistillationCfg",
        },
    )

gym.register(
    id="Rough-Deeprobotics-M20-GaitRefinement-v2",
    entry_point="rl_training.tasks.manager_based.locomotion.velocity.mdp.gait_refinement_v2_env:M20GaitRefinementV2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gait_refinement_v2_env_cfg:DeeproboticsM20GaitRefinementV2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.gait_refinement_v2_ppo_cfg:DeeproboticsM20GaitRefinementV2PPORunnerCfg",
    },
)

gym.register(
    id="Rough-Deeprobotics-M20-QuietLanding-v0",
    entry_point="rl_training.tasks.manager_based.locomotion.velocity.mdp.quiet_landing_traversal:M20QuietLandingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.quiet_landing_env_cfg:DeeproboticsM20QuietLandingEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.quiet_landing_ppo_cfg:DeeproboticsM20QuietLandingPPORunnerCfg",
    },
)

##
# Register Gym environments.
##

gym.register(
    id="Rough-Deeprobotics-M20-QuietLanding-V1_5-v0",
    entry_point="rl_training.tasks.manager_based.locomotion.velocity.mdp.quiet_landing_traversal:M20QuietLandingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.quiet_landing_env_cfg:DeeproboticsM20QuietLandingV15EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.quiet_landing_ppo_cfg:DeeproboticsM20QuietLandingV15PPORunnerCfg",
    },
)

gym.register(
    id="Flat-Deeprobotics-M20-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:DeeproboticsM20FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsM20FlatPPORunnerCfg",
    },
)

gym.register(
    id="Rough-Deeprobotics-M20-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:DeeproboticsM20RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DeeproboticsM20RoughPPORunnerCfg",
    },
)


gym.register(
    id="Rough-Deeprobotics-M20-StairTeacher-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stair_teacher_env_cfg:DeeproboticsM20StairTeacherEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.stair_teacher_ppo_cfg:DeeproboticsM20StairTeacherPPORunnerCfg",
    },
)


gym.register(
    id="Rough-Deeprobotics-M20-PreTeacher-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pre_teacher_env_cfg:DeeproboticsM20PreTeacherEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.pre_teacher_ppo_cfg:DeeproboticsM20PreTeacherPPORunnerCfg",
    },
)

gym.register(
    id="Rough-Deeprobotics-M20-StairStudent-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stair_student_env_cfg:DeeproboticsM20StairStudentEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.stair_student_ppo_cfg:DeeproboticsM20StairStudentPPORunnerCfg",
    },
)
