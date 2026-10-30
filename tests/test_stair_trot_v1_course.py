"""CPU tests of the actual course and active parent command/curriculum code."""
import ast
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/rl_training"))
from task_support.stair_trot_v1.signals import quat_rotate
from task_support.idle_stop_motion_v1.profile import TUNING

torch.set_num_threads(1)


@dataclass
class TerrainCfg:
    step_depth_range: tuple = (0.25, 0.25)
    step_height_range: tuple = (0.08, 0.24)
    num_steps: int = 6
    start_platform_length: float = 2.0
    proportion: float = 0.10


class LowerCommand:
    """Isaac's low-level sampling is stubbed; the active teacher is real AST."""
    def __init__(self, cfg, env):
        self.cfg, self._env = cfg, env
        self.device, self.num_envs = env.device, env.num_envs
        self.robot = env.scene["robot"]
        self.vel_command_b = torch.zeros(env.num_envs, 3)
        self.is_standing_env = torch.ones(env.num_envs, dtype=torch.bool)
        self.is_heading_env = torch.ones(env.num_envs, dtype=torch.bool)
        self.heading_target = torch.ones(env.num_envs)
        self.time_left = torch.ones(env.num_envs)
        self.metrics = {}
        self._metric_step_counter = torch.ones(env.num_envs, dtype=torch.long)

    def _resample_command(self, ids):
        self.vel_command_b[ids] = torch.tensor([-.4, .2, .3])
        self.is_standing_env[ids] = False
        self.is_heading_env[ids] = False

    def _update_command(self):
        self.vel_command_b[self.is_standing_env] = 0

    def _update_metrics(self):
        pass


class SceneEntityCfg:
    def __init__(self, name):
        self.name = name


def _load_course():
    events = ModuleType("isaaclab.envs.mdp.events")
    events.reset_root_state_uniform = lambda env, ids, pose, velocity, cfg: env.normal_resets.append(
        (ids.clone(), pose, velocity)
    )
    managers = ModuleType("isaaclab.managers")
    managers.SceneEntityCfg = SceneEntityCfg
    terrain = ModuleType("task_support.stair_trot_v1.terrain")
    terrain.AscentStairsTerrainCfg = TerrainCfg
    teacher = ModuleType("task_support.idle_stop_motion_v1.rewards")
    source = ast.parse((ROOT / "source/rl_training/task_support/idle_stop_motion_v1/rewards.py").read_text())
    definitions = [node for node in source.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and
                   node.name in ("StairTeacherVelocityCommand", "stair_distance_terrain_levels")]
    teacher.__dict__.update(
        torch=torch, np=np, UniformThresholdVelocityCommand=LowerCommand,
        TUNING=TUNING, SceneEntityCfg=SceneEntityCfg, quat_apply=quat_rotate, yaw_quat=lambda q: q,
        _stair_activity_mask=lambda env: torch.zeros(env.num_envs, dtype=torch.bool),
        _flat_surface_mask=lambda env, cfg: torch.zeros(env.num_envs, dtype=torch.bool),
        __package__="task_support.idle_stop_motion_v1",
    )
    exec(compile(ast.Module(body=definitions, type_ignores=[]), "<actual_idle_stop_parent>", "exec"),
         teacher.__dict__)
    regularizers = ModuleType("task_support.idle_stop_motion_v1.regularizers")
    regularizers.update_gait_level_from_terrain_mean = lambda level: None
    mocks = {
        events.__name__: events, managers.__name__: managers, terrain.__name__: terrain,
        teacher.__name__: teacher, regularizers.__name__: regularizers,
    }
    name = "task_support.stair_trot_v1._course_cpu_tests"
    spec = importlib.util.spec_from_file_location(name, ROOT / "source/rl_training/task_support/stair_trot_v1/course.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {**mocks, name: module}):
        spec.loader.exec_module(module)
    return module, mocks


course, MOCKS = _load_course()


class Asset:
    def __init__(self, n):
        self.data = SimpleNamespace(
            root_pos_w=torch.tensor([[0., 0., .55]]).repeat(n, 1),
            root_quat_w=torch.tensor([[1., 0., 0., 0.]]).repeat(n, 1),
            projected_gravity_b=torch.tensor([[0., 0., -1.]]).repeat(n, 1),
            root_lin_vel_w=torch.zeros(n, 3), root_ang_vel_w=torch.zeros(n, 3),
            body_pos_w=torch.zeros(n, 4, 3),
            default_root_state=torch.tensor([[0., 0., .55, 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]).repeat(n, 1),
        )
        self.poses = []
        self.velocities = []

    def find_bodies(self, names, preserve_order):
        return list(range(4)), list(names)

    def write_root_pose_to_sim(self, pose, env_ids):
        self.poses.append((env_ids.clone(), pose.clone()))
        self.data.root_pos_w[env_ids] = pose[:, :3]
        self.data.root_quat_w[env_ids] = pose[:, 3:7]

    def write_root_velocity_to_sim(self, velocity, env_ids):
        self.velocities.append((env_ids.clone(), velocity.clone()))


class Scene:
    def __getitem__(self, name):
        return self.robot if name == "robot" else self.sensors[name]


def fixture():
    n = 10
    sub = {"flat": SimpleNamespace(proportion=.40)}
    for riser in ("solid", "open"):
        for name, depth in (("short", .25), ("medium", .35), ("long", .50)):
            sub[f"{riser}_{name}"] = TerrainCfg(step_depth_range=(depth, depth))
    generator = SimpleNamespace(curriculum=True, num_cols=n, size=(8., 8.), sub_terrains=sub)
    scene = Scene()
    scene.robot = Asset(n)
    scene.env_origins = torch.zeros(n, 3)
    scene.terrain = SimpleNamespace(
        cfg=SimpleNamespace(terrain_generator=generator), terrain_types=torch.arange(n),
        terrain_levels=torch.ones(n, dtype=torch.long),
    )
    scene.terrain.updates = []

    def update_origins(ids, up, down):
        scene.terrain.updates.append((ids.clone(), up.clone(), down.clone()))
        scene.terrain.terrain_levels[ids] += up.long() - down.long()
        scene.env_origins[ids, 0] += (up.float() - down.float()) * 10
    scene.terrain.update_env_origins = update_origins
    sensor = SimpleNamespace(
        data=SimpleNamespace(net_forces_w=torch.zeros(n, 4, 3)),
        find_bodies=lambda names, preserve_order: (list(range(4)), list(names)),
    )
    scene.sensors = {"contact_forces": sensor}
    env = SimpleNamespace(
        num_envs=n, device="cpu", scene=scene, common_step_counter=10, step_dt=.02,
        episode_length_buf=torch.ones(n, dtype=torch.long),
        cfg=SimpleNamespace(stair_course=course.StairCourseSettings()),
        termination_manager=SimpleNamespace(terminated=torch.zeros(n, dtype=torch.bool),
                                             time_outs=torch.zeros(n, dtype=torch.bool)),
        normal_resets=[],
    )
    command = course.StairAscentVelocityCommand(SimpleNamespace(resampling_time_range=(1., 2.)), env)
    env.command_manager = SimpleNamespace(get_term=lambda name: command, get_command=lambda name: command.vel_command_b)
    return env, command


def place_on_landing(env, ids):
    state = course.course_state(env)
    for i in ids:
        origin = env.scene.env_origins[i]
        top = origin[2] + state["min_top_z_offset"][i]
        env.scene.robot.data.body_pos_w[i, :, 0] = state["goal_x"][i] + .2
        env.scene.robot.data.body_pos_w[i, :, 1] = origin[1]
        env.scene.robot.data.body_pos_w[i, :, 2] = top + .09
        env.scene.robot.data.root_pos_w[i] = torch.tensor([state["goal_x"][i] + .6, origin[1], top + .55])
        env.scene.sensors["contact_forces"].data.net_forces_w[i, :, 2] = 80


class StairCourseTests(unittest.TestCase):
    def setUp(self):
        self.modules = patch.dict(sys.modules, MOCKS)
        self.modules.start()
        self.env, self.command = fixture()

    def tearDown(self):
        self.modules.stop()

    def advance(self, count=10):
        for _ in range(count):
            self.env.common_step_counter += 1
            course.stair_course_failure(self.env)

    def test_exact_column_mapping_all_six_types_and_world_goal(self):
        env = self.env
        state = course.course_state(env)
        self.assertEqual(state["mask"].tolist(), [False] * 4 + [True] * 6)
        np.testing.assert_array_equal(course.terrain_column_indices(env.scene.terrain.cfg.terrain_generator),
                                      [0, 0, 0, 0, 1, 2, 3, 4, 5, 6])
        torch.testing.assert_close(state["goal_x_offset"][4:], torch.tensor([2.5, 3.1, 4., 2.5, 3.1, 4.]))
        env.scene.env_origins[4, 0] = 13
        self.assertAlmostEqual(course.course_state(env)["goal_x"][4].item(), 15.5)

    def test_wrong_random_mode_or_unrecorded_random_depth_fails(self):
        generator = self.env.scene.terrain.cfg.terrain_generator
        generator.curriculum = False
        with self.assertRaises(ValueError):
            course.course_state(self.env)
        generator.curriculum = True
        generator.sub_terrains["solid_short"].step_depth_range = (.25, .5)
        with self.assertRaises(ValueError):
            course.course_state(self.env)

    def test_active_parent_statistics_and_normal_commands_preserved(self):
        self.command._resample_command(torch.arange(10))
        self.command._update_command()
        command = self.command.vel_command_b
        torch.testing.assert_close(command[:4], torch.tensor([[-.4, .2, .3]]).repeat(4, 1))
        self.assertTrue(((command[4:, 0] >= .25) & (command[4:, 0] <= .55)).all())
        self.assertTrue((command[4:, 1:] == 0).all())
        self.command._update_metrics()
        torch.testing.assert_close(
            self.command.episode_commanded_translation_m,
            torch.linalg.vector_norm(command[:, :2], dim=1) * self.env.step_dt)
        torch.testing.assert_close(self.command.episode_commanded_translation_w, command[:, :2] * self.env.step_dt)

    def test_supported_exit_latches_before_reset_and_stops_command(self):
        place_on_landing(self.env, [4])
        self.command._resample_command(torch.arange(10))
        self.advance()
        state = course.course_state(self.env)
        self.assertTrue(state["success"][4])
        self.command._update_command()
        self.assertTrue((self.command.vel_command_b[4] == 0).all())
        self.command._resample_command(torch.tensor([4]))
        self.assertTrue((self.command.vel_command_b[4] == 0).all())
        self.env.scene.sensors["contact_forces"].data.net_forces_w[4] = 0
        self.advance()
        self.assertTrue(state["success"][4])

    def test_rear_wheel_shortcut_flight_wall_load_floor_and_heading_do_not_pass(self):
        place_on_landing(self.env, list(range(4, 10)))
        state = course.course_state(self.env)
        self.env.scene.robot.data.body_pos_w[4, 3, 0] = state["goal_x"][4] - .01
        force = self.env.scene.sensors["contact_forces"].data.net_forces_w
        force[5] = 0  # Flight.
        force[6, :, 0] = 400  # Horizontal wall support dominates.
        self.env.scene.robot.data.body_pos_w[7, :, 2] = -.41
        self.env.scene.robot.data.root_pos_w[7, 2] = .05  # Catch-floor bypass.
        self.env.scene.robot.data.root_quat_w[8] = torch.tensor([.7071068, 0, 0, .7071068])
        self.env.scene.robot.data.root_pos_w[9, 1] = .5
        self.advance()
        self.assertFalse(state["success"].any())
        self.assertTrue(course.stair_course_failure(self.env)[7])

    def test_partial_reset_same_tick_cannot_add_support_dwell(self):
        place_on_landing(self.env, [4, 5])
        course.stair_course_failure(self.env)
        state = course.course_state(self.env)
        before = state["hold_time"][5].clone()
        course.reset_root_on_stair_course(self.env, torch.tensor([4]), {}, {})
        course.stair_course_failure(self.env)
        self.assertEqual(state["hold_time"][5], before)
        self.assertEqual(state["hold_time"][4], 0)
        self.assertFalse(state["stop"][4])

    def test_completion_reward_is_once_per_episode_and_not_consumed_by_reads(self):
        place_on_landing(self.env, [4])
        seen = []
        for _ in range(20):
            self.env.common_step_counter += 1
            first = course.course_completion_reward(self.env)
            repeated = course.course_completion_reward(self.env)
            torch.testing.assert_close(first, repeated)
            seen.append(first[4].item() * self.env.step_dt)
        self.assertAlmostEqual(sum(seen), 1.0)
        self.assertEqual(sum(value > 0 for value in seen), 1)
        course.reset_root_on_stair_course(self.env, torch.tensor([4]), {}, {})
        self.assertEqual(course.course_state(self.env)["success_tick"][4], -1)
        self.assertEqual(course.course_completion_reward(self.env)[4], 0)

    def test_reset_preserves_normal_rule_and_zeroes_ascent_velocity(self):
        pose, velocity = {"yaw": (-3, 3)}, {"x": (-1, 1)}
        course.reset_root_on_stair_course(self.env, torch.tensor([0, 4]), pose, velocity)
        self.assertEqual(self.env.normal_resets[0][0].tolist(), [0])
        self.assertIs(self.env.normal_resets[0][1], pose)
        ids, values = self.env.scene.robot.velocities[0]
        self.assertEqual(ids.tolist(), [4])
        self.assertTrue((values == 0).all())
        self.assertLessEqual(self.env.scene.robot.data.root_pos_w[4, :2].abs().max(), .10)

    def test_curriculum_uses_latch_failure_priority_and_pre_reset_metrics(self):
        place_on_landing(self.env, [4, 6])
        self.advance()
        self.env.termination_manager.time_outs[[4, 5, 6]] = True
        self.env.termination_manager.terminated[6] = True  # Later fall overrides success.
        self.command.episode_commanded_translation_m[0] = 2
        self.command.episode_commanded_translation_w[0, 0] = 2
        self.env.scene.robot.data.root_pos_w[0, 0] = 1.5
        course.stair_course_terrain_levels(self.env, torch.tensor([0, 4, 5, 6]))
        self.assertEqual(self.env.scene.terrain.terrain_levels[[0, 4, 5, 6]].tolist(), [2, 2, 0, 0])
        metrics = course.course_metrics(self.env, None)
        self.assertEqual(metrics["solid_short/success_fraction"], 1)
        self.assertEqual(metrics["solid_medium/success_fraction"], 0)
        self.assertEqual(metrics["solid_long/success_fraction"], 0)
        self.assertEqual(metrics["solid_short/episodes"], 1)
        course.stair_course_terrain_levels(self.env, torch.tensor([4, 5, 6]))
        self.assertEqual(course.course_metrics(self.env, None)["solid_short/episodes"], 1)

    def test_initial_reset_and_out_of_route_termination(self):
        self.env.episode_length_buf.zero_()
        course.stair_course_terrain_levels(self.env, torch.arange(10))
        self.assertEqual(course.course_metrics(self.env, None)["open_long/episodes"], 0)
        self.env.scene.robot.data.root_pos_w[4, 1] = 1
        self.env.scene.robot.data.root_pos_w[5, 2] = -.1
        self.env.scene.robot.data.root_pos_w[6, 0] = 7
        self.env.scene.robot.data.root_pos_w[0, 2] = -.1  # Normal tiles use their own terminations.
        self.assertEqual(course.stair_course_failure(self.env)[:7].tolist(), [False] * 4 + [True] * 3)


if __name__ == "__main__":
    unittest.main()
