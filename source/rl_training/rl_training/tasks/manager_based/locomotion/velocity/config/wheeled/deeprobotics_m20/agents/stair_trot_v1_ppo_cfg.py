"""PPO entry for ascent-only refinement; source policy remains frozen."""
from isaaclab.utils import configclass
from .idle_stop_motion_v1_ppo_cfg import DeeproboticsM20IdleStopMotionV1PPORunnerCfg


@configclass
class DeeproboticsM20StairTrotV1PPORunnerCfg(DeeproboticsM20IdleStopMotionV1PPORunnerCfg):
    class_name = "task_support.stair_trot_v1.runner:StairTrotRunner"
    experiment_name = "deeprobotics_m20_stair_trot_v1"
    run_name = "stair_trot_v1"
    baseline_checkpoint = "deploy/models/idle_stop_motion_v1/idle_stop_motion_v1_46400.pt"
    max_iterations = 20000
    obs_groups = {"actor": ["policy", "motion_control", "trot_control"],
                  "critic": ["critic", "motion_control", "trot_control"]}

    def __post_init__(self):
        self.algorithm.critic_warmup_iterations = 20
        self.algorithm.motion_regularization_coefficient = 0.005
