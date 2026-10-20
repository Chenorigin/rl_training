# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from .rough_ppo_cfg import DeeproboticsM20RoughPPORunnerCfg


@configclass
class DeeproboticsM20FlatPPORunnerCfg(DeeproboticsM20RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 5000
        self.experiment_name = "deeprobotics_m20_flat"
