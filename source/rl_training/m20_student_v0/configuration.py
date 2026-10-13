"""Native RSL-RL v5 student/teacher config; no PPO actor/critic conversion."""

from copy import deepcopy
from dataclasses import dataclass, field


def model_cfg():
    return {"class_name": "rsl_rl.models.MLPModel", "hidden_dims": [512, 256, 128],
            "activation": "elu", "obs_normalization": False,
            "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "log"}}


@dataclass
class DeeproboticsM20StudentV0DistillationCfg:
    seed: int = 42
    num_steps_per_env: int = 32
    max_iterations: int = 5000
    save_interval: int = 100
    experiment_name: str = "deeprobotics_m20_student_v0"
    logger: str = "tensorboard"
    check_for_nan: bool = True
    clip_actions: float = 100.0
    student: dict = field(default_factory=model_cfg)
    teacher: dict = field(default_factory=model_cfg)
    obs_groups: dict = field(default_factory=lambda: {"student": ["policy"], "teacher": ["teacher"]})
    algorithm: dict = field(default_factory=lambda: {
        "class_name": "m20_student_v0.distillation.StudentV0Distillation",
        "learning_rate": 1e-4, "num_learning_epochs": 2, "max_grad_norm": 1.0,
        "batch_size": 1024, "replay_rollouts": 4, "teacher_ratio": 0.5,
        "rnd_cfg": None, "symmetry_cfg": None,
    })

    def to_dict(self):
        return deepcopy(self.__dict__)
