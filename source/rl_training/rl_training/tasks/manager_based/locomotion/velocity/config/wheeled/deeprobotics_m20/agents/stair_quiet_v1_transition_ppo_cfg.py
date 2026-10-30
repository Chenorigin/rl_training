"""Initialize from complete Safe weights; training is started by the user."""
from isaaclab.utils import configclass
from .stair_quiet_v1_safe_ppo_cfg import DeeproboticsM20StairQuietV1SafePPORunnerCfg


@configclass
class DeeproboticsM20StairQuietV1TransitionPPORunnerCfg(DeeproboticsM20StairQuietV1SafePPORunnerCfg):
    class_name = "task_support.stair_quiet_v1_transition.runner:StairQuietTransitionRunner"
    experiment_name = "deeprobotics_m20_stair_quiet_v1_transition"
    run_name = "teacher_transition_from_safe"
    max_iterations = 30000
