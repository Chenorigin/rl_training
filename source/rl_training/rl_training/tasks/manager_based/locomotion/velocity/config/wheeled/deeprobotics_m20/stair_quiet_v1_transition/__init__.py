"""Task discovery adds a version without changing any old checkpoint dependency."""
import gymnasium as gym

_CFG = "rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20"
gym.register(
    id="Rough-Deeprobotics-M20-StairQuiet-v1-Transition",
    entry_point="rl_training.tasks.manager_based.locomotion.velocity.mdp.stair_quiet_v1_transition_env:M20StairQuietV1TransitionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": _CFG+".stair_quiet_v1_transition_env_cfg:DeeproboticsM20StairQuietV1TransitionEnvCfg",
        "rsl_rl_cfg_entry_point": _CFG+".agents.stair_quiet_v1_transition_ppo_cfg:DeeproboticsM20StairQuietV1TransitionPPORunnerCfg",
    },
)
