"""Discoverable registration without changing existing teacher source hashes."""
import gymnasium as gym
from task_support.student_v3_safe.configuration import TASK

_CFG = "rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20"
gym.register(
    id=TASK,
    entry_point="rl_training.tasks.manager_based.locomotion.velocity.mdp.student_v3_safe_env:M20StudentV3SafeEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": _CFG + ".student_v3_safe_env_cfg:DeeproboticsM20StudentV3SafeEnvCfg",
            "rsl_rl_cfg_entry_point": _CFG + ".agents.student_v3_safe_ppo_cfg:DeeproboticsM20StudentV3SafeDistillationCfg"},
)
