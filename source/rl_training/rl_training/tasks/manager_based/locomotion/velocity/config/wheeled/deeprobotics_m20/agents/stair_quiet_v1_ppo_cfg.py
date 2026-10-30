"""Conservative complete-teacher refinement; start long runs explicitly."""
from isaaclab.utils import configclass
from task_support.stair_quiet_v1.baselines import TEACHER
from .idle_stop_motion_v1_ppo_cfg import DeeproboticsM20IdleStopMotionV1PPORunnerCfg
from .student_v3_5_ppo_cfg import DeeproboticsM20StudentV35DistillationCfg


@configclass
class DeeproboticsM20StairQuietV1PPORunnerCfg(DeeproboticsM20IdleStopMotionV1PPORunnerCfg):
    class_name = "task_support.stair_quiet_v1.runner:StairQuietRunner"
    experiment_name = "deeprobotics_m20_stair_quiet_v1"
    run_name = "stair_quiet_v1"
    max_iterations = 10000
    save_interval = 100
    baseline_checkpoint = TEACHER
    descent_guide_iters = 0
    descent_guide_decay_iters = 0

    def __post_init__(self):
        self.algorithm.learning_rate = 3.e-5
        self.algorithm.anchor_coefficient = 0.5
        self.algorithm.anchor_stair_weight = 0.1
        self.algorithm.anchor_stop_weight = 1.0
        self.algorithm.motion_regularization_coefficient = 0.0


@configclass
class DeeproboticsM20StairQuietV1StudentDistillationCfg(DeeproboticsM20StudentV35DistillationCfg):
    experiment_name = "deeprobotics_m20_stair_quiet_v1_student"
    run_name = "stair_quiet_v1_student"
