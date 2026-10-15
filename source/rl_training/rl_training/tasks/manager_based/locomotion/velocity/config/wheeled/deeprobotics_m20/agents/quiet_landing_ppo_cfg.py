"""Independent quiet V1 PPO configuration; baseline architecture and optimizer."""
import torch
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils.logger import Logger

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from .stair_teacher_ppo_cfg import StairGuardedPPO


class QuietLandingLogger(Logger):
    def log(self, *args, **kwargs):
        iteration = kwargs.get("it", args[0] if args else 0)
        if self.writer is not None:
            keys = sorted({key for info in self.ep_extras for key in info})
            for key in keys:
                samples = [torch.as_tensor(info[key], device=self.device, dtype=torch.float32).reshape(-1)
                           for info in self.ep_extras if key in info]
                values = torch.cat(samples)
                finite = values[torch.isfinite(values)]
                if finite.numel():
                    self.writer.add_scalar(key, finite.mean(), iteration)
        # Parent handles PPO/train/performance cards. It must not re-log extras
        # using ep_extras[0], which silently misses conditional touchdown keys.
        self.ep_extras.clear()
        return super().log(*args, **kwargs)


class QuietLandingRunner(OnPolicyRunner):
    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        super().__init__(env, train_cfg, log_dir, device)
        self.logger = QuietLandingLogger(
            log_dir=log_dir, cfg=self.cfg, env_cfg=self.env.cfg, num_envs=self.env.num_envs,
            is_distributed=self.is_distributed, gpu_world_size=self.gpu_world_size,
            gpu_global_rank=self.gpu_global_rank, device=self.device,
        )


@configclass
class QuietLandingAlgorithmCfg(RslRlPpoAlgorithmCfg):
    # Reuse the proven preservation algorithm, without inheriting teacher config.
    class_name = f"{StairGuardedPPO.__module__}:StairGuardedPPO"
    anchor_coefficient: float = 0.1
    critic_warmup_iterations: int = 50
    reference_std_multiplier: float = 1.05
    anchor_uneven_weight: float = 0.1
    actor_update_mask_key: str | None = None


@configclass
class DeeproboticsM20QuietLandingPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name = f"{__name__}:QuietLandingRunner"
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 100
    experiment_name = "deeprobotics_m20_quiet_landing_v1"
    run_name = "quiet_landing_v1"
    empirical_normalization = False
    clip_actions = 100
    logger = "tensorboard"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0, noise_std_type="log", actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128], activation="elu",
    )
    algorithm = QuietLandingAlgorithmCfg(
        value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.1,
        entropy_coef=0.0, num_learning_epochs=5, num_mini_batches=4,
        learning_rate=1.0e-5, schedule="fixed", gamma=0.99, lam=0.95,
        desired_kl=0.01, max_grad_norm=1.0,
    )


@configclass
class DeeproboticsM20QuietLandingV15PPORunnerCfg(DeeproboticsM20QuietLandingPPORunnerCfg):
    experiment_name = "deeprobotics_m20_quiet_landing_v1_5"
    run_name = "quiet_landing_v1_5"
