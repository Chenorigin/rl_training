"""Discovered by Isaac's package walker, without modifying live width hashes."""
import gymnasium as gym

_CFG = "rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20"
gym.register(
    id="Rough-Deeprobotics-M20-StairQuiet-v1-Safe",
    entry_point="rl_training.tasks.manager_based.locomotion.velocity.mdp.stair_quiet_v1_safe_env:M20StairQuietV1SafeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": _CFG+".stair_quiet_v1_safe_env_cfg:DeeproboticsM20StairQuietV1SafeEnvCfg",
        "rsl_rl_cfg_entry_point": _CFG+".agents.stair_quiet_v1_safe_ppo_cfg:DeeproboticsM20StairQuietV1SafePPORunnerCfg",
    },
)
