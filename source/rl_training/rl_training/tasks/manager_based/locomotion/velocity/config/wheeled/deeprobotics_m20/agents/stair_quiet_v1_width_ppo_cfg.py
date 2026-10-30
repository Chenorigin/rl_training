"""Original residual PPO with an independent width-curriculum run identity."""
from isaaclab.utils import configclass
from .stair_quiet_v1_ppo_cfg import DeeproboticsM20StairQuietV1PPORunnerCfg


@configclass
class DeeproboticsM20StairQuietV1WidthPPORunnerCfg(DeeproboticsM20StairQuietV1PPORunnerCfg):
    class_name = "task_support.stair_quiet_v1_width.runner:StairQuietWidthRunner"
    experiment_name = "deeprobotics_m20_stair_quiet_v1_width"
    run_name = "teacher_width_curriculum"

    max_iterations = 10000

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.critic_warmup_iterations = 50
