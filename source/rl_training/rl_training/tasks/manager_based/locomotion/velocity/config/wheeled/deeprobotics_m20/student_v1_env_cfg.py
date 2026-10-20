"""V1 keeps the pinned quiet-landing physical task and replaces sensor observations."""
from copy import deepcopy
from dataclasses import MISSING

from isaaclab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg
from isaaclab.utils import configclass

from m20_student_v1 import environment as glue
from m20_student_v1.configuration import StudentV1PerceptionCfg
from .quiet_landing_env_cfg import DeeproboticsM20QuietLandingEnvCfg
from rl_training.tasks.manager_based.locomotion.velocity.mdp.student_v1_env import StudentV1VelocityCommand




@configclass
class PackedPolicyCfg(ObservationGroupCfg):
    observation = ObservationTermCfg(func=glue.student_observation)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class StudentV1ObservationsCfg:
    teacher: ObservationGroupCfg = MISSING
    policy: ObservationGroupCfg = PackedPolicyCfg()
    critic: ObservationGroupCfg = PackedPolicyCfg()


def install_student(cfg):
    teacher = deepcopy(cfg.observations.policy)
    teacher.enable_corruption = False
    teacher.actions.func = glue.last_issued_action
    cfg.observations = StudentV1ObservationsCfg(teacher=teacher)
    cfg.events.student_v1_reset = EventTermCfg(func=glue.reset_student, mode="reset")
    cfg.commands.base_velocity.class_type = StudentV1VelocityCommand
    cfg.commands.base_velocity.heading_command = False
    cfg.commands.base_velocity.rel_heading_envs = 0.
    # Preserve teacher ranges; balance actual exposure inside those ranges.
    cfg.curriculum.command_levels = None
    # Warm-started student sees the complete terrain level range from the start.
    cfg.curriculum.terrain_levels = None
    cfg.scene.terrain.max_init_terrain_level = cfg.scene.terrain.terrain_generator.num_rows-1


@configclass
class DeeproboticsM20StudentV1EnvCfg(DeeproboticsM20QuietLandingEnvCfg):
    student_perception: StudentV1PerceptionCfg = StudentV1PerceptionCfg()
    student_history_steps: int = 1

    def __post_init__(self):
        super().__post_init__()
        install_student(self)
