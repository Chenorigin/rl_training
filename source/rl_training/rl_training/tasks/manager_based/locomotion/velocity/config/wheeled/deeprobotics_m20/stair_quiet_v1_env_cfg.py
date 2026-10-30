"""Stair-quiet candidate on the exact idle-stop-motion-v1 environment contract."""
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from task_support.stair_quiet_v1.signals import StairQuietSettings, stair_quiet_cost
from task_support.stair_quiet_v1.landing import LandingSettings
from .idle_stop_motion_v1_env_cfg import DeeproboticsM20IdleStopMotionV1EnvCfg


@configclass
class DeeproboticsM20StairQuietV1EnvCfg(DeeproboticsM20IdleStopMotionV1EnvCfg):
    """Preserve baseline traversal, power, termination, observations and control.

    Physical proxy scales are candidates, not evidence of acoustic reduction.
    Both stair directions and all baseline mixed-terrain coverage are retained.
    Quiet costs are ramped by the runner's task-iteration clock; baseline
    traversal costs and physical materials stay fixed. The task control layer
    modifies targets/gains before the existing explicit delayed PD actuator.
    """

    stair_quiet: StairQuietSettings = StairQuietSettings()
    landing_control: LandingSettings = LandingSettings()
    stair_quiet_student: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.stair_quiet.validate()
        self.landing_control.validate()
        # The parent profile removes unfamiliar rewards. Add ours only AFTER it
        # has restored the pinned baseline objectives and observation contract.
        for name, weight in self.stair_quiet.weights.items():
            setattr(self.rewards, "stair_quiet_" + name, RewTerm(
                func=stair_quiet_cost, weight=weight, params={"name": name},
            ))
        # Fresh force samples are essential at each unchanged physics timestep.
        if self.scene.contact_forces.update_period != self.sim.dt:
            raise ValueError("Stair quiet requires the baseline substep contact sensor period")


@configclass
class DeeproboticsM20StairQuietV1StudentEnvCfg(DeeproboticsM20StairQuietV1EnvCfg):
    """The quiet physical task with the established v3.5 student sensor ABI."""

    stair_quiet_student: bool = True
    student_perception: object = None
    student_history_steps: int = 0
    motion_teacher_tuning: dict = {}

    def __post_init__(self):
        super().__post_init__()
        from isaaclab.managers import EventTermCfg
        from m20_student_v1.configuration import StudentV1PerceptionCfg
        from m20_student_v1.v3_5_configuration import student_v3_5_settings
        from m20_student_v1 import v3_5_environment as glue
        from .student_v1_env_cfg import install_student
        from .student_v3_5_env_cfg import MotionControlCfg
        settings = student_v3_5_settings()
        self.student_perception = StudentV1PerceptionCfg(**settings["perception"])
        self.student_perception.validate()
        self.student_history_steps = settings["history_steps"]
        install_student(self)
        self.observations.motion_control = MotionControlCfg()
        self.observations.teacher.velocity_commands.func = glue.effective_velocity_commands
        self.observations.policy.observation.func = glue.student_observation
        self.observations.critic.observation.func = glue.student_observation
        self.events.student_v3_5_heading_reset = EventTermCfg(func=glue.reset_feedback, mode="reset")
