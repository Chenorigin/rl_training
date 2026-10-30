"""Safe refinement identity, initialized from the current width checkpoint."""
from isaaclab.utils import configclass
from .stair_quiet_v1_ppo_cfg import DeeproboticsM20StairQuietV1PPORunnerCfg


@configclass
class DeeproboticsM20StairQuietV1SafePPORunnerCfg(DeeproboticsM20StairQuietV1PPORunnerCfg):
    class_name = "task_support.stair_quiet_v1_safe.runner:StairQuietSafeRunner"
    experiment_name = "deeprobotics_m20_stair_quiet_v1_safe"
    run_name = "teacher_safe"
    max_iterations = 30000
