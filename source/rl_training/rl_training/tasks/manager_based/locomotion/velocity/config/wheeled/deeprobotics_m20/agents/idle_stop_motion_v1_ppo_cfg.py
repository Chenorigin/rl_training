"""Original idle-stop motion residual PPO as an independent M20 task."""
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg

from .rough_ppo_cfg import DeeproboticsM20RoughPPORunnerCfg


@configclass
class IdleStopMotionAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "task_support.idle_stop_motion_v1.training:StairGuardedPPO"
    anchor_coefficient: float = 0.1
    critic_warmup_iterations: int = 20
    reference_std_multiplier: float = 1.02
    anchor_stair_weight: float = 1.0
    anchor_stop_weight: float = 0.02
    anchor_stop_command_epsilon: float = 1.e-4
    stop_teacher_coefficient: float = 0.0
    stop_regularization_coefficient: float = 0.0
    motion_regularization_coefficient: float = 0.02


@configclass
class DeeproboticsM20IdleStopMotionV1PPORunnerCfg(DeeproboticsM20RoughPPORunnerCfg):
    class_name = "task_support.idle_stop_motion_v1.runner:IdleStopMotionRunner"
    experiment_name = "deeprobotics_m20_idle_stop_motion_v1"
    run_name = "idle_stop_motion_v1"
    max_iterations = 150000
    save_interval = 100
    baseline_checkpoint: str = "deploy/models/idle_stop_v1/model_257100_idle_stop.pt"
    initial_critic_checkpoint: str | None = None
    descent_guide_iters: int = 20000
    descent_guide_decay_iters: int = 10000
    obs_groups = {"actor": ["policy", "motion_control"], "critic": ["critic", "motion_control"]}
    algorithm = IdleStopMotionAlgorithmCfg(
        value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.1,
        entropy_coef=1.e-4, num_learning_epochs=5, num_mini_batches=4,
        learning_rate=1.e-4, schedule="fixed", gamma=0.99, lam=0.95,
        desired_kl=0.01, max_grad_norm=1.0,
    )
