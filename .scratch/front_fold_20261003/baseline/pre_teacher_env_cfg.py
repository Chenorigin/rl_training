# Copyright (c) 2026 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause

"""M20 wheel-driven locomotion pretraining with the stair teacher's model I/O."""

from copy import deepcopy
import os

from isaaclab.envs.mdp import height_scan
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains.trimesh import MeshPlaneTerrainCfg
from isaaclab.utils import configclass

from .rough_env_cfg import DeeproboticsM20RoughEnvCfg
from .stair_teacher_env_cfg import DEFAULT_M20_USD_PATH, M20StairTeacherObservationsCfg


@configclass
class DeeproboticsM20PreTeacherEnvCfg(DeeproboticsM20RoughEnvCfg):
    """Learn flat-ground wheel locomotion before adding stair-specific rewards.

    The policy, critic, action, and PPO architectures match StairTeacher.
    Rewards are inherited from the official M20 Rough task, except rewards
    that explicitly request wheel lift or diagonal stepping on flat ground.
    """

    observations: M20StairTeacherObservationsCfg = M20StairTeacherObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        # Keep the same USD as StairTeacher; allow relocation through one override.
        model_usd = os.path.realpath(os.environ.get("RL_TRAINING_M20_USD_PATH", DEFAULT_M20_USD_PATH))
        if not os.path.isfile(model_usd):
            raise FileNotFoundError(
                f"M20 pre-teacher USD does not exist: {model_usd}. "
                "Set RL_TRAINING_M20_USD_PATH if the model was moved."
            )
        self.scene.robot.spawn.usd_path = model_usd

        # RoughEnvCfg removes the actor scan. Restore the exact StairTeacher
        # term (order, clip, scale, and absence of scan noise) for transfer.
        self.observations.policy.height_scan = ObsTerm(
            func=height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=None,
            clip=(-1.0, 1.0),
            scale=1.0,
        )
        for term_name in ("joint_pos", "joint_vel"):
            getattr(self.observations.critic, term_name).params["asset_cfg"].joint_names = self.joint_names

        # Wheels should handle this phase. Gentle ramps expose varied scan
        # values without teaching a stair-stepping gait before the next phase.
        generator = deepcopy(self.scene.terrain.terrain_generator)
        for terrain in generator.sub_terrains.values():
            terrain.proportion = 0.0
        generator.sub_terrains["flat"] = MeshPlaneTerrainCfg(proportion=0.80)
        for name in ("hf_pyramid_slope", "hf_pyramid_slope_inv"):
            generator.sub_terrains[name].proportion = 0.10
            generator.sub_terrains[name].slope_range = (0.0, 0.10)
        generator.curriculum = False
        self.scene.terrain.terrain_generator = generator
        self.scene.terrain.max_init_terrain_level = None
        self.curriculum.terrain_levels = None

        # Preserve the StairTeacher command mix so translation is practiced
        # often, while still sampling stopping and in-place turning.
        self.commands.base_velocity.rel_zero_vel_envs = 0.05
        self.commands.base_velocity.rel_only_ang_z_envs = 0.05

        # The original M20 turn-gait rewards explicitly pay for wheel air time
        # and alternating diagonal contact. Keep velocity/stance/effort terms,
        # but learn flat-ground turning on wheels rather than lifting legs.
        self.disable_zero_weight_rewards()
        for name in (
            "feet_air_time_ang_z_M20",
            "rotation_gait_status",
            "rotation_gait_symmetry",
            "feet_slide_ang_z_cmd",
        ):
            setattr(self.rewards, name, None)
        self.curriculum.gait_level = None
