"""618D student scaffold sharing the stair teacher's physical task/rewards."""

from copy import deepcopy
from dataclasses import MISSING

from isaaclab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from rl_training.tasks.manager_based.locomotion.velocity.mdp import stair_student
from .stair_teacher_env_cfg import DeeproboticsM20StairTeacherEnvCfg


@configclass
class M20StairStudentObservationsCfg:
    policy: ObservationGroupCfg = MISSING
    critic: ObservationGroupCfg = MISSING
    # Clean 244D label-query view. RSL PPO selects policy/critic, never this group.
    teacher: ObservationGroupCfg = MISSING


@configclass
class DeeproboticsM20StairStudentEnvCfg(DeeproboticsM20StairTeacherEnvCfg):
    student_perception: stair_student.StairStudentPerceptionCfg = stair_student.StairStudentPerceptionCfg()

    def __post_init__(self):
        # First resolve the exact teacher proprioceptive bindings/scales and
        # action/terrain/reward task, then construct independent student groups.
        super().__post_init__()
        self.student_perception.validate()
        teacher = deepcopy(self.observations.policy)
        teacher.enable_corruption = False
        policy = deepcopy(teacher)
        params = {"sensor_cfg": SceneEntityCfg("height_scanner")}
        policy.height_scan = ObservationTermCfg(
            func=stair_student.student_height_scan, params=deepcopy(params), noise=None)
        policy.height_validity = ObservationTermCfg(
            func=stair_student.student_height_validity, params=deepcopy(params), noise=None)
        policy.height_confidence = ObservationTermCfg(
            func=stair_student.student_height_confidence, params=deepcopy(params), noise=None)
        # No hidden clean-map/force/friction/acceleration observation in critic.
        self.observations = M20StairStudentObservationsCfg(
            policy=policy, critic=deepcopy(policy), teacher=teacher)
        self.events.student_perception_reset = EventTermCfg(
            func=stair_student.reset_student_perception, mode="reset")
