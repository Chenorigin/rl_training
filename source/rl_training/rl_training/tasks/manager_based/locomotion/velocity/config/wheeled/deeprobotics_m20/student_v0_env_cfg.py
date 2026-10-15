"""V0 student: Quiet Landing physical task with a separate sensor-limited policy."""

from copy import deepcopy
from dataclasses import MISSING

from isaaclab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from m20_student_v0 import perception
from m20_student_v0.contract import STUDENT_OBS_DIM, TEACHER_OBS_DIM
from .quiet_landing_env_cfg import DeeproboticsM20QuietLandingEnvCfg, DeeproboticsM20QuietLandingV15EnvCfg


@configclass
class StudentV0ObservationsCfg:
    policy: ObservationGroupCfg = MISSING
    critic: ObservationGroupCfg = MISSING
    teacher: ObservationGroupCfg = MISSING


def install_student_observations(cfg):
    cfg.student_perception.validate()
    teacher = deepcopy(cfg.observations.policy)
    teacher.enable_corruption = False
    policy = deepcopy(teacher)
    params = {"sensor_cfg": SceneEntityCfg("height_scanner")}
    policy.height_scan = ObservationTermCfg(func=perception.student_height, params=deepcopy(params))
    policy.height_validity = ObservationTermCfg(func=perception.student_validity, params=deepcopy(params))
    policy.height_confidence = ObservationTermCfg(func=perception.student_confidence, params=deepcopy(params))
    # Critic is reserved for future PPO; pure distillation never constructs it.
    cfg.observations = StudentV0ObservationsCfg(policy=policy, critic=deepcopy(policy), teacher=teacher)
    cfg.events.student_v0_reset = EventTermCfg(func=perception.reset_student_map, mode="reset")


@configclass
class DeeproboticsM20StudentV0EnvCfg(DeeproboticsM20QuietLandingEnvCfg):
    student_perception: perception.StudentV0PerceptionCfg = perception.StudentV0PerceptionCfg()

    def __post_init__(self):
        super().__post_init__()
        install_student_observations(self)


@configclass
class DeeproboticsM20StudentV15V0EnvCfg(DeeproboticsM20QuietLandingV15EnvCfg):
    student_perception: perception.StudentV0PerceptionCfg = perception.StudentV0PerceptionCfg()

    def __post_init__(self):
        super().__post_init__()
        install_student_observations(self)


# Keep the parent's integration, reward tracking and physical stepping path.
from rl_training.tasks.manager_based.locomotion.velocity.mdp.quiet_landing_traversal import M20QuietLandingEnv


class M20StudentV0Env(M20QuietLandingEnv):
    POLICY_OBSERVATION_DIM = STUDENT_OBS_DIM

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        if self.observation_manager.group_obs_dim["teacher"] != (TEACHER_OBS_DIM,):
            raise RuntimeError("V0 distillation requires a clean 244D teacher observation")
