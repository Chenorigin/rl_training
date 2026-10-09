# Copyright (c) 2026 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .stair_teacher_ppo_cfg import DeeproboticsM20StairTeacherPPORunnerCfg


@configclass
class DeeproboticsM20PreTeacherPPORunnerCfg(DeeproboticsM20StairTeacherPPORunnerCfg):
    """Same actor/critic architecture as the stair teacher, separate logs."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "deeprobotics_m20_pre_teacher"
