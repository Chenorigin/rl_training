"""Safe physics/course retained; only the student's observation is restricted."""
from copy import deepcopy

from isaaclab.managers import EventTermCfg, ObservationGroupCfg, RewardTermCfg
from isaaclab.utils import configclass

from m20_student_v1 import environment as common, v3_5_environment as motion
from m20_student_v1.configuration import StudentV1PerceptionCfg
from task_support.student_v3_safe.configuration import default_settings
from task_support.student_v3_safe.practice_command import StudentStartStopCommand
from task_support.student_v3_safe.gait_reward import diagonal_reward
from .stair_quiet_v1_safe_env_cfg import DeeproboticsM20StairQuietV1SafeEnvCfg, LongitudinalScanCfg, TeacherSafetyCfg
from .student_v1_env_cfg import PackedPolicyCfg
from .student_v3_5_env_cfg import MotionControlCfg


@configclass
class SafeStudentObservationsCfg:
    policy: ObservationGroupCfg = PackedPolicyCfg()
    critic: ObservationGroupCfg = PackedPolicyCfg()
    teacher: ObservationGroupCfg = None
    motion_control: ObservationGroupCfg = MotionControlCfg()
    locomotion_scan: ObservationGroupCfg = LongitudinalScanCfg()
    safety: ObservationGroupCfg = TeacherSafetyCfg()


@configclass
class DeeproboticsM20StudentV3SafeEnvCfg(DeeproboticsM20StairQuietV1SafeEnvCfg):
    stair_quiet_student: bool = True
    student_perception: object = None
    student_history_steps: int = 4
    motion_teacher_tuning: dict = {}
    student_idle_practice: dict = {}
    student_diagonal_reward: dict = {}
    student_terrain_course: dict = {}

    def __post_init__(self):
        super().__post_init__()
        settings = default_settings()
        self.student_perception = StudentV1PerceptionCfg(**settings["perception"])
        self.student_perception.validate()
        self.student_history_steps = settings["history_steps"]
        self.student_idle_practice = deepcopy(settings["training"]["idle_practice"])
        self.student_diagonal_reward = deepcopy(settings["training"]["diagonal_reward"])
        self.rewards.student_diagonal_alternation = RewardTermCfg(
            func=diagonal_reward, weight=self.student_diagonal_reward["weight"])
        self.commands.base_velocity.class_type = StudentStartStopCommand
        teacher = deepcopy(self.observations.policy)
        teacher.enable_corruption = False
        teacher.actions.func = common.last_issued_action
        teacher.velocity_commands.func = motion.effective_velocity_commands
        self.observations = SafeStudentObservationsCfg(teacher=teacher)
        self.observations.policy.observation.func = motion.student_observation
        self.observations.critic.observation.func = motion.student_observation
        self.events.student_v3_safe_reset = EventTermCfg(func=common.reset_student, mode="reset")
        self.events.student_v3_safe_heading_reset = EventTermCfg(func=motion.reset_feedback, mode="reset")
        from task_support.student_v3_safe.terrain_importer import configure_course
        configure_course(self,settings['training']['terrain_course'])
        # Do not call install_student(): retain the Safe physical course/reset.
        # StudentStartStopCommand delegates to the original width command first.
