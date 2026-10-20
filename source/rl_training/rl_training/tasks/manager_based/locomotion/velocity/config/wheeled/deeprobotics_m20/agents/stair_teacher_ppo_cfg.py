# Copyright (c) 2026 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg

# Re-export the implementation for existing checkpoint callable paths.
from task_support.stair_teacher.ppo import StairGuardedPPO
from .rough_ppo_cfg import DeeproboticsM20RoughPPORunnerCfg


@configclass
class StairPreservationAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = f"{StairGuardedPPO.__module__}:StairGuardedPPO"
    anchor_coefficient: float = 0.1
    critic_warmup_iterations: int = 50
    reference_std_multiplier: float = 1.05
    anchor_uneven_weight: float = 0.1
    actor_update_mask_key: str | None = None


@configclass
class DeeproboticsM20StairTeacherPPORunnerCfg(DeeproboticsM20RoughPPORunnerCfg):
    """PPO runner for the privileged M20 stair teacher."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "deeprobotics_m20_stair_teacher"
        # PreTeacher inherits this class; keep its original PPO configuration.
        if type(self) is not DeeproboticsM20StairTeacherPPORunnerCfg:
            return
        settings = self.algorithm.to_dict()
        settings.update(class_name=f"{StairGuardedPPO.__module__}:StairGuardedPPO",
                        learning_rate=1.0e-5, schedule="fixed", entropy_coef=0.0,
                        clip_param=0.1)
        self.algorithm = StairPreservationAlgorithmCfg(**settings)
