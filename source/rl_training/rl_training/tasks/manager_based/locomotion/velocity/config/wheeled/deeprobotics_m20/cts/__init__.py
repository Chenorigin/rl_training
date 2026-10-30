"""Recursively discovered CTS registration; existing tasks remain independent."""
import gymnasium as gym

_CFG = "rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20"

gym.register(
    id="Rough-Deeprobotics-M20-CTS",
    entry_point="rl_training.tasks.manager_based.locomotion.velocity.mdp.cts_env:M20CTSEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": _CFG + ".cts_env_cfg:DeeproboticsM20CTSEnvCfg",
        "rsl_rl_cfg_entry_point": _CFG + ".agents.cts_ppo_cfg:CTSConfig",
    },
)
