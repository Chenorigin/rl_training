"""CTS-only terrain, command and reset wiring; imports require a live Isaac app."""
import torch
import trimesh

from isaaclab.managers import RewardTermCfg, TerminationTermCfg
from isaaclab.utils import configclass
from task_support.student_v3_safe.terrain_importer import StudentTerrainImporter, StudentStairsCfg
from task_support.student_v3_safe.practice_command import StudentStartStopCommand
from task_support.student_v3_safe.terrain_course import reset_course as student_reset
from .platform import platform_patch_mesh, platform_reward, platform_failure, platform_timeout


@configclass
class CTSStairsCfg(StudentStairsCfg):
    # Isaac replaces dataclasses per tile; dynamic attributes are lost there.
    cts_sample_stairs: dict = None
    cts_legacy_stairs: dict = None


class CTSTerrainImporter(StudentTerrainImporter):
    def import_mesh(self, name, mesh):
        platform, platform_origins = platform_patch_mesh(self.cfg.terrain_generator,
            self.cfg.cts_platform_practice)
        from .motion import motion_patch_mesh
        motion, motion_origins = motion_patch_mesh(self.cfg.terrain_generator,self.cfg.cts_motion_practice)
        from .straight import straight_patch_mesh
        straight, straight_origins = straight_patch_mesh(self.cfg.terrain_generator,
            self.cfg.cts_motion_practice,self.cfg.cts_straight_practice)
        # Both raycasting's main mesh and the existing per-profile physical
        # collision meshes include these same triangles. Material assignment
        # uses the original Y columns, independently of the new X rows.
        super().import_mesh(name, trimesh.util.concatenate((mesh, platform, motion, straight)))
        self.cts_platform_origins = torch.as_tensor(platform_origins, device=self.device)
        self.cts_motion_origins = torch.as_tensor(motion_origins, device=self.device)
        self.cts_straight_origins = torch.as_tensor(straight_origins, device=self.device)


class CTSCommand(StudentStartStopCommand):
    def _apply_course(self, ids):
        super()._apply_course(ids)
        self._env.cts_platform.command(self, ids)
        self._env.cts_motion.command(self, ids)


def reset_course(env, env_ids, pose_range, velocity_range, asset_cfg=None):
    student_reset(env, env_ids, pose_range, velocity_range, asset_cfg)
    env.cts_sample.reset(env_ids)
    env.cts_platform.reset(env_ids)
    env.cts_motion.reset(env_ids)


def configure_practice(cfg, algorithm, platform, sample, legacy, motion, mixture, straight):
    from copy import deepcopy
    from .perception import validate_mixture
    cfg.student_perception.zero_delay_probability = validate_mixture(mixture)['zero_delay_probability']
    cfg.student_perception.validate()
    cfg.cts_cohort_config = deepcopy(algorithm)
    from .platform import validate_platform, locomotion_scan, teacher_safety
    cfg.cts_platform_practice = deepcopy(validate_platform(platform))
    from .sample_stairs import validate_sample, training_stairs
    cfg.cts_sample_stairs = deepcopy(validate_sample(sample))
    from .terrain import validate_legacy
    cfg.cts_legacy_stairs = deepcopy(validate_legacy(legacy))
    from .motion import validate_motion, motion_reward, motion_boundary, motion_timeout
    cfg.cts_motion_practice = deepcopy(validate_motion(motion))
    from .straight import validate_straight
    cfg.cts_straight_practice = deepcopy(validate_straight(straight,motion))
    from dataclasses import replace, fields
    cfg.cts_sample_automatic_stages = cfg.stair_tail.automatic_stages
    cfg.stair_tail = replace(cfg.stair_tail,automatic_stages=False)
    generator = cfg.scene.terrain.terrain_generator
    for name,terrain in tuple(generator.sub_terrains.items()):
        values = {field.name:deepcopy(getattr(terrain,field.name)) for field in fields(terrain) if field.init}
        values.update(function=training_stairs,cts_sample_stairs=deepcopy(sample),cts_legacy_stairs=deepcopy(legacy),settings=cfg.stair_tail)
        generator.sub_terrains[name] = CTSStairsCfg(**values)
    from .sample_stairs import sample_route_reward
    cfg.rewards.cts_sample_route = RewardTermCfg(func=sample_route_reward,weight=1.)
    cfg.scene.terrain.class_type = CTSTerrainImporter
    cfg.scene.terrain.cts_platform_practice = deepcopy(platform)
    cfg.scene.terrain.cts_motion_practice = deepcopy(motion)
    cfg.scene.terrain.cts_straight_practice = deepcopy(straight)
    cfg.commands.base_velocity.class_type = CTSCommand
    cfg.events.randomize_reset_base.func = reset_course
    cfg.rewards.cts_platform = RewardTermCfg(func=platform_reward,weight=1.)
    cfg.terminations.cts_platform_failure = TerminationTermCfg(func=platform_failure)
    cfg.terminations.cts_platform_timeout = TerminationTermCfg(func=platform_timeout,time_out=True)
    cfg.rewards.cts_motion = RewardTermCfg(func=motion_reward,weight=1.)
    cfg.terminations.cts_motion_boundary = TerminationTermCfg(func=motion_boundary)
    cfg.terminations.cts_motion_timeout = TerminationTermCfg(func=motion_timeout,time_out=True)
    cfg.observations.locomotion_scan.scan.func = locomotion_scan
    cfg.observations.safety.safety.func = teacher_safety
