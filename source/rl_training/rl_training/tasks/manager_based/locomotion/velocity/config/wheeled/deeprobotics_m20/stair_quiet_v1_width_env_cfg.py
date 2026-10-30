"""Owned width curriculum, effective contacts and four-wheel safety."""
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg, TerminationTermCfg
from task_support.stair_quiet_v1_width.settings import WidthSettings, GEOMETRIES, PROFILES
from task_support.stair_quiet_v1_width.terrain import WidthStairsCfg, WidthTerrainImporter
from task_support.stair_quiet_v1_width.course import WidthAscentCommand, reset_course
from task_support.stair_quiet_v1_width.observations import effective_friction
from task_support.stair_quiet_v1_width.signals import (
    tail_cost, tail_failure, tail_success, success_reward, failure_reward)
from .stair_quiet_v1_env_cfg import DeeproboticsM20StairQuietV1EnvCfg


@configclass
class DeeproboticsM20StairQuietV1WidthEnvCfg(DeeproboticsM20StairQuietV1EnvCfg):
    stair_tail: WidthSettings = WidthSettings()

    def __post_init__(self):
        super().__post_init__()
        s = self.stair_tail
        s.validate()
        self.scene.num_envs = 1024
        self.episode_length_s = 20.
        # Perfect teacher scan; actor dimensions and frozen base stay unchanged.
        self.observations.policy.enable_corruption = False
        self.observations.critic.enable_corruption = False
        self.observations.critic.material_friction.func = effective_friction
        terrain = self.scene.terrain
        terrain.class_type = WidthTerrainImporter
        terrain.max_init_terrain_level = 0
        gen = terrain.terrain_generator
        gen.num_rows, gen.num_cols = len(s.widths_m), len(GEOMETRIES)*len(PROFILES)
        gen.difficulty_range = (0., 1.)
        gen.size = (s.tile_length, s.tile_width)
        gen.curriculum, gen.use_cache = True, False
        gen.border_width = 0.
        gen.sub_terrains = {
            f"{g.name}__{p.name}": WidthStairsCfg(
                proportion=1., geometry_index=gi, profile_index=pi, settings=s)
            for gi, g in enumerate(GEOMETRIES) for pi, p in enumerate(PROFILES)}
        # Single exact static ray mesh, distinct collision meshes/materials.
        self.scene.height_scanner.mesh_prim_paths = ["/World/ground/terrain/mesh"]
        if getattr(self.scene, "height_scanner_base", None) is not None:
            self.scene.height_scanner_base.mesh_prim_paths = ["/World/ground/terrain/mesh"]
        self.curriculum.terrain_levels = None
        # Parent per-terrain telemetry names refer to the old mixed terrain
        # dictionary; our per-case counters replace those obsolete selectors.
        for name in tuple(vars(self.curriculum)):
            if name.startswith("terrain_level_"):
                setattr(self.curriculum, name, None)
        self.commands.base_velocity.class_type = WidthAscentCommand
        self.events.randomize_reset_base.func = reset_course
        self.events.randomize_rigid_body_material = None
        # Retain stage-0 actuator/mass randomization; do not broaden them when
        # expanding stair contacts. No external pushes in this teacher task.
        for name in ("randomize_push_robot", "randomize_apply_external_force_torque"):
            if hasattr(self.events, name):
                setattr(self.events, name, None)
        self.terminations.tail_failure = TerminationTermCfg(func=tail_failure)
        # New starts are near the low X end of an 8 m tile; the inherited 3 m
        # whole-map buffer would immediately time out every episode there.
        self.terminations.terrain_out_of_bounds.params["distance_buffer"] = .1
        # A completed finite ascent is terminal: no time-limit value bootstrap.
        self.terminations.tail_success = TerminationTermCfg(func=tail_success)
        self.rewards.tail_success = RewardTermCfg(func=success_reward, weight=s.success_weight)
        self.rewards.tail_failure = RewardTermCfg(func=failure_reward, weight=s.failure_weight)
        for name in ("edge", "foothold", "slip", "recovery_impact", "tail_force", "tail_impulse", "progress"):
            weight = getattr(s, name+"_weight")
            if s.objective == "control" and name in ("slip", "recovery_impact", "tail_force", "tail_impulse"):
                weight = 0.
            setattr(self.rewards, "tail_"+name, RewardTermCfg(func=tail_cost, weight=weight, params={"name": name}))
        self.m20_refinement_profile = "stair_quiet_v1_width"
