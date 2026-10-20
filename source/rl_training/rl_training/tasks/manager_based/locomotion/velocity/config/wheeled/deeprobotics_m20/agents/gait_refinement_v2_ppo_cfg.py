"""Gait refinement v2 PPO configuration for the standard training entrypoint."""
from isaaclab.utils import configclass

# Re-export runtime classes for existing checkpoint callable paths.
from task_support.gait_refinement_v2.training import GaitRefinementV2PPO, GaitRefinementV2Runner
from .quiet_landing_ppo_cfg import QuietLandingAlgorithmCfg, DeeproboticsM20QuietLandingPPORunnerCfg


@configclass
class GaitRefinementV2AlgorithmCfg(QuietLandingAlgorithmCfg):
    class_name = f"{GaitRefinementV2PPO.__module__}:GaitRefinementV2PPO"
    actor_update_mask_key = "policy"
    anchor_coefficient = 1.0
    anchor_uneven_weight = 0.10
    critic_warmup_iterations = 10
    replay_coefficient: float = 5.0
    replay_batch_size: int = 256
    freeze_actor_front: bool = True
    # Reuse the existing physical idle/stop input set; no new manifest tree.
    correction_replay_manifest: str | None = "docs/m20_idle_stop_257100_20261012/reproduction_manifest.json"


@configclass
class DeeproboticsM20GaitRefinementV2PPORunnerCfg(DeeproboticsM20QuietLandingPPORunnerCfg):
    class_name = f"{GaitRefinementV2Runner.__module__}:GaitRefinementV2Runner"
    experiment_name = "deeprobotics_m20_gait_refinement_v2"
    run_name = "v2_idle_stop_stair_heading"
    save_interval = 50
    algorithm = GaitRefinementV2AlgorithmCfg(
        value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.1,
        entropy_coef=0.0, num_learning_epochs=5, num_mini_batches=4,
        learning_rate=3.0e-5, schedule="fixed", gamma=0.99, lam=0.95,
        desired_kl=0.01, max_grad_norm=1.0)

