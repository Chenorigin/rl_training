"""PPO fine-tuning configuration; this is not the distillation runner."""

from copy import deepcopy
from isaaclab.utils import configclass

from .rsl_rl_ppo_cfg import DeeproboticsM20RoughPPORunnerCfg


@configclass
class DeeproboticsM20StairStudentPPORunnerCfg(DeeproboticsM20RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "deeprobotics_m20_stair_student"
        self.max_iterations = 5000
        self.empirical_normalization = False
        self.obs_groups = {"actor": ["policy"], "critic": ["critic"]}
        self.algorithm = deepcopy(self.algorithm)
        self.algorithm.learning_rate = 1.0e-4
        self.algorithm.schedule = "adaptive"
