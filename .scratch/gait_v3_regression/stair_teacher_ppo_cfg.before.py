# Copyright (c) 2026 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .rsl_rl_ppo_cfg import DeeproboticsM20RoughPPORunnerCfg


@configclass
class DeeproboticsM20StairTeacherPPORunnerCfg(DeeproboticsM20RoughPPORunnerCfg):
    """PPO runner for the privileged M20 stair teacher."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "deeprobotics_m20_stair_teacher"
