"""CPU counterexamples for impact accounting; no simulator or optimizer starts."""
import importlib.util
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.utils._python_dispatch import TorchDispatchMode

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/rl_training"))
from task_support.stair_quiet_v1.signals import (
    COST_NAMES, StairQuietSettings, StairQuietTracker, stair_quiet_cost,
    contact_point_kinematics, scheduled_quiet_scale, wheel_axis_world,
)

torch.set_num_threads(1)


def kinematics(tracker):
    zeros = torch.zeros((*tracker.contact.shape, 3))
    axis = zeros.clone()
    axis[..., 1] = 1
    return dict(pre_omega=zeros, wheel_axis=axis, link_minus_com=zeros.clone())


class ImpactAccountingTests(unittest.TestCase):
    def tracker(self, n=1, **kwargs):
        return StairQuietTracker(n, "cpu", StairQuietSettings(reset_grace_time=0, **kwargs))

    def sample(self, tracker, force=0.0, speed=0.0, dt=0.005, axis=2):
        v = torch.zeros((*tracker.contact.shape, 3))
        f = torch.zeros_like(v)
        v[..., axis] = -speed
        f[..., axis] = force
        tracker.update(v, f, dt, **kinematics(tracker))

    def test_initial_contact_and_partial_reset_are_not_touchdowns(self):
        t = self.tracker(2)
        self.sample(t, 500, 3)
        self.assertEqual(t.event_count.sum(), 0)
        self.assertTrue(all(v.sum() == 0 for v in t.costs.values()))
        self.sample(t, 0)
        self.sample(t, 500, 1)
        previous = t.event_count[1].clone()
        history = t.force_history[:, 1].clone()
        t.reset(torch.tensor([0]))
        self.assertTrue(torch.equal(t.event_count[1], previous))
        self.assertTrue(torch.equal(t.force_history[:, 1], history))
        self.assertEqual(t.force_history[:, 0].sum(), 0)
        self.sample(t, 500, 3)
        self.assertEqual(t.event_count[0].sum(), 0)
        self.assertEqual(t.loading_peak[0].sum(), 0)
        self.assertTrue(torch.equal(t.event_count[1], previous))

    def test_preimpact_speed_square_and_riser_contact_are_not_gated(self):
        for axis in (0, 2):
            with self.subTest(axis=axis):
                t = self.tracker()
                self.sample(t)
                self.sample(t, 300, 2, axis=axis)
                self.assertEqual(t.event_count.sum(), 4)
                self.assertAlmostEqual(float(t.costs["touchdown"]), 4 * (2 / .5) ** 2)
                self.assertGreater(float(t.costs["loading_rate"]), 0)
                self.assertEqual(t.last["pre_velocity_m_s"][0, 0, axis], -2)

    def test_loading_rate_penalizes_continuous_contact_outside_touchdown_window(self):
        t = self.tracker()
        for _ in range(12):
            self.sample(t, 50)
        t.begin_step()
        self.sample(t, 600, axis=0)
        self.assertEqual(t.step_events.sum(), 0)
        for name in ("loading_rate", "window_peak", "window_impulse"):
            self.assertGreater(float(t.costs[name]), 0, name)

    def test_no_map_or_command_inputs_required_for_air_descent(self):
        t = self.tracker()
        self.sample(t)
        self.sample(t, speed=2)
        self.assertGreater(float(t.costs["air_descent"]), 0)
        self.assertEqual(t.costs["touchdown"].sum(), 0)
        self.assertEqual(t.costs["loading_rate"].sum(), 0)

    def test_zero_contact_does_not_claim_zero_measured_touchdown_speed(self):
        t = self.tracker()
        self.sample(t)
        self.sample(t)
        metrics = t.metrics(torch.tensor([0]))
        self.assertEqual(metrics["StairQuiet/all/has_touchdowns"], 0)
        self.assertNotIn("StairQuiet/all/preimpact_speed_rms_m_s", metrics)
        self.assertEqual(metrics["StairQuiet/all/has_physics_samples"], 1)

    def test_rolling_window_integrates_duration_and_forgets_expired_peak(self):
        t = self.tracker(window_seconds=.012)
        self.sample(t)
        for _ in range(4):
            self.sample(t, 200)
        self.assertTrue(torch.allclose(t.last["window_excess_impulse_N_s"], torch.full((1, 4), 1.2)))
        for _ in range(3):
            self.sample(t)
        self.assertEqual(t.last["window_peak_N"].sum(), 0)
        self.assertEqual(t.last["window_excess_impulse_N_s"].sum(), 0)

    def test_large_impacts_have_no_flat_penalty_tail(self):
        costs = []
        for force in (1000, 10000):
            t = self.tracker()
            self.sample(t)
            self.sample(t, force, 1)
            costs.append({name: float(value) for name, value in t.costs.items()})
        for name in ("loading_rate", "window_peak", "window_impulse"):
            self.assertGreater(costs[1][name], costs[0][name], name)

    def test_grace_maintains_history_without_fake_loading_after_settling(self):
        t = StairQuietTracker(1, "cpu", StairQuietSettings(reset_grace_time=.01))
        for _ in range(5):
            self.sample(t, 80, 2)
        self.assertEqual(t.event_count.sum(), 0)
        self.assertEqual(t.costs["loading_rate"].sum(), 0)
        self.assertEqual(t.costs["touchdown"].sum(), 0)
        self.assertEqual(t.costs["air_descent"].sum(), 0)
        self.assertGreater(t.sample_time.sum(), 0)

    def test_front_and_rear_metrics_keep_independent_evidence(self):
        t = self.tracker()
        self.sample(t)
        v = torch.zeros((1, 4, 3))
        f = torch.zeros_like(v)
        v[:, :2, 2], f[:, :2, 2] = -1, 300
        t.update(v, f, .005, **kinematics(t))
        m = t.metrics(torch.tensor([0]))
        self.assertEqual(m["StairQuiet/front/touchdown_count"], 2)
        self.assertEqual(m["StairQuiet/rear/touchdown_count"], 0)
        self.assertEqual(m["StairQuiet/front/preimpact_speed_rms_m_s"], 1)

    def test_inherited_runner_pools_raw_touchdowns_instead_of_batch_rms(self):
        from task_support.idle_stop_motion_v1.runner import pool_episode_metrics, episode_means

        rows = []
        for count, speed, force in ((1, 10., 500.), (100, 1., 100.)):
            t = self.tracker()
            # Populate the sufficient statistics for two deliberately unequal
            # reset batches; run the real inherited aggregation implementation.
            t.event_count[0, 0] = count
            t.speed_sum[0, 0] = count * speed
            t.speed_sq_sum[0, 0] = count * speed * speed
            t.speed_max[0, 0] = speed
            t.sample_time[0] = count
            t.force_peak[0, 0] = force
            t.force_integral[0, 0] = count * force
            t.distance[0] = count
            rows.append(t.metrics(torch.tensor([0])))
        pooled = episode_means(pool_episode_metrics(rows))
        prefix = "StairQuiet/all/"
        self.assertAlmostEqual(pooled[prefix + "preimpact_speed_rms_m_s"], math.sqrt(200 / 101), places=6)
        self.assertAlmostEqual(pooled[prefix + "preimpact_speed_mean_m_s"], 110 / 101, places=6)
        self.assertAlmostEqual(pooled[prefix + "specific_impact_energy_mean_J_kg"], 100 / 101, places=6)
        self.assertAlmostEqual(pooled[prefix + "force_per_wheel_time_mean_N"], 10500 / (101 * 4), places=5)
        self.assertAlmostEqual(pooled["StairQuiet/specific_impact_energy_per_m_J_kg_m"], 100 / 101, places=6)
        self.assertEqual(pooled[prefix + "mean_reset_batch_force_max_N"], 300)
        self.assertNotIn(prefix + "force_max_N", pooled)
        self.assertNotIn(prefix + "preimpact_speed_max_m_s", pooled)
        self.assertFalse(any("__weighted_" in key or "__rms_" in key for key in pooled))

    def test_begin_step_and_reward_dt_do_not_change_episode_totals(self):
        t = self.tracker()
        self.sample(t)
        self.sample(t, 300, 1)
        total = t.event_count.clone()
        env = SimpleNamespace(stair_quiet_tracker=t, step_dt=.02, stair_quiet_reward_scale=1.)
        self.assertTrue(torch.equal(stair_quiet_cost(env, "touchdown") * .02, t.costs["touchdown"]))
        t.begin_step()
        self.assertTrue(torch.equal(t.event_count, total))
        self.assertTrue(all(value.sum() == 0 for value in t.costs.values()))
        self.assertEqual(t.step_events.sum(), 0)

    def test_bad_physics_and_setting_inputs_fail_closed(self):
        t = self.tracker()
        self.sample(t)
        with self.assertRaisesRegex(ValueError, "cannot change"):
            self.sample(t, dt=.01)
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            self.sample(t, force=float("nan"))
        with self.assertRaisesRegex(ValueError, "positive"):
            self.sample(t, dt=0)
        with self.assertRaisesRegex(ValueError, "shape"):
            t.update(torch.zeros(1, 3), torch.zeros(1, 3), .005, **kinematics(t))
        with self.assertRaisesRegex(ValueError, "weights"):
            StairQuietTracker(1, "cpu", StairQuietSettings(weights={k: 1 for k in COST_NAMES}))

    def test_pure_rolling_and_tangential_slip_have_no_normal_impact_cost(self):
        t = self.tracker()
        self.sample(t)
        k = kinematics(t)
        k["pre_omega"][..., 1] = 1. / t.settings.wheel_radius
        velocity = torch.zeros((1, 4, 3))
        velocity[..., 0] = 1.
        force = torch.zeros_like(velocity)
        force[..., 2] = 300.
        t.update(velocity, force, .005, **k)
        self.assertTrue(torch.allclose(t.last["pre_contact_point_velocity_m_s"], torch.zeros_like(velocity), atol=1e-6))
        self.assertEqual(t.costs["touchdown"].sum(), 0)
        self.assertEqual(t.costs["contact_work"].sum(), 0)
        velocity[..., 0] = 3.
        t.update(velocity, force, .005, **k)
        self.assertEqual(t.costs["contact_work"].sum(), 0)

    def test_capsule_tilt_and_actual_com_offset_change_normal_point_speed(self):
        s = StairQuietSettings()
        v, omega, force, axis, offset = (torch.zeros((1, 4, 3)) for _ in range(5))
        omega[..., 0] = 2.
        force[..., 2] = 300.
        axis[..., 1:] = 1 / math.sqrt(2)
        point, approach, r = contact_point_kinematics(v, omega, force, axis, offset, s)
        expected = 2 * s.wheel_capsule_half_length / math.sqrt(2)
        self.assertTrue(torch.allclose(approach, torch.full((1, 4), expected)))
        self.assertTrue(torch.allclose(point, v + torch.linalg.cross(omega, r, dim=-1)))
        offset[..., 1] = .01
        _, shifted, _ = contact_point_kinematics(v, omega, force, axis, offset, s)
        self.assertTrue(torch.allclose(shifted, approach - .02))

    def test_world_axis_rotation_and_invalid_kinematics(self):
        s = StairQuietSettings()
        q = torch.tensor([math.sqrt(.5), math.sqrt(.5), 0., 0.]).repeat(1, 4, 1)
        self.assertTrue(torch.allclose(wheel_axis_world(q, s), torch.tensor([0., 0., 1.]).repeat(1, 4, 1), atol=1e-6))
        with self.assertRaisesRegex(ValueError, "normalized"):
            wheel_axis_world(q * 2, s)
        t = self.tracker()
        k = kinematics(t)
        k["pre_omega"][0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            t.update(torch.zeros(1, 4, 3), torch.zeros(1, 4, 3), .005, **k)
        k = kinematics(t)
        k["wheel_axis"].zero_()
        with self.assertRaisesRegex(ValueError, "normalized"):
            t.update(torch.zeros(1, 4, 3), torch.zeros(1, 4, 3), .005, **k)

    def test_substep_combined_validation_has_one_scalar_transfer(self):
        class ScalarTransfers(TorchDispatchMode):
            def __init__(self):
                self.count = 0

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if func == torch.ops.aten._local_scalar_dense.default:
                    self.count += 1
                return func(*args, **(kwargs or {}))

        t = self.tracker(2)
        zeros = torch.zeros(2, 4, 3)
        quaternion = torch.tensor([1., 0., 0., 0.]).repeat(2, 4, 1)
        # Exercise both initial allocation and established substep history.
        for _ in range(3):
            with ScalarTransfers() as calls:
                axis = wheel_axis_world(quaternion, t.settings, validate=False)
                t.update(zeros, zeros, .005, pre_omega=zeros, wheel_axis=axis,
                         link_minus_com=zeros, wheel_quaternion=quaternion)
            self.assertEqual(calls.count, 1)

    def test_combined_validation_rejects_every_input_and_nonunit_quaternion(self):
        t = self.tracker()
        quaternion = torch.tensor([1., 0., 0., 0.]).repeat(1, 4, 1)
        for key in ("pre_velocity", "post_force", "pre_omega", "wheel_axis",
                    "link_minus_com", "wheel_quaternion"):
            inputs = dict(pre_velocity=torch.zeros(1, 4, 3), post_force=torch.zeros(1, 4, 3),
                          **kinematics(t), wheel_quaternion=quaternion.clone())
            inputs[key].reshape(-1)[0] = float("nan")
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "Non-finite"):
                t.update(dt=.005, **inputs)
            self.assertFalse(t.initialized.any())
        # q=(2,0,0,0) happens to rotate this axis to a unit vector, so checking
        # only the derived axis would silently miss the invalid input pose.
        bad_q = quaternion * 2
        axis = wheel_axis_world(bad_q, t.settings, validate=False)
        self.assertTrue(torch.allclose(axis.norm(dim=-1), torch.ones(1, 4)))
        with self.assertRaisesRegex(ValueError, "normalized"):
            t.update(torch.zeros(1, 4, 3), torch.zeros(1, 4, 3), .005,
                     pre_omega=torch.zeros(1, 4, 3), wheel_axis=axis,
                     link_minus_com=torch.zeros(1, 4, 3), wheel_quaternion=bad_q)

    def test_contact_work_integrates_physical_time_and_continuous_loading(self):
        totals = []
        for dt in (.005, .01):
            t = self.tracker()
            self.sample(t, 100, 0, dt)
            for _ in range(round(.1 / dt)):
                self.sample(t, 100, .5, dt)
            self.assertEqual(t.event_count.sum(), 0)  # continuous contact
            totals.append(float(t.contact_work.sum()))
        self.assertAlmostEqual(totals[0], 4 * 100 * .5 * .1, places=5)
        self.assertAlmostEqual(totals[0], totals[1], places=5)

    def test_secondary_impact_is_time_limited_and_reset_safe(self):
        for delay, expected in ((.01, 4), (.2, 0)):
            t = self.tracker()
            self.sample(t)
            self.sample(t, 300, 1)
            for _ in range(round(delay / .005) - 1):
                self.sample(t)
            self.sample(t, 300, 1)
            self.assertEqual(t.reimpact_count.sum(), expected)
            t.reset(torch.tensor([0]))
            self.sample(t)
            t.begin_step()
            self.sample(t, 300, 1)
            self.assertEqual(t.reimpact_count.sum(), 0)
            self.assertEqual(t.costs["reimpact"].sum(), 0)

    def test_cross_foot_coincidence_is_phase_sensitive_without_support_penalty(self):
        def event(t, feet):
            v, f = torch.zeros(1, 4, 3), torch.zeros(1, 4, 3)
            v[:, feet, 2], f[:, feet, 2] = -.5, 300
            t.update(v, f, .005, **kinematics(t))

        costs = []
        for delay in (0., .01, .04):
            t = self.tracker()
            self.sample(t)
            if delay == 0:
                event(t, [0, 2])
            else:
                event(t, [0])
                for _ in range(round(delay / .005) - 1):
                    self.sample(t)
                event(t, [2])
            costs.append(float(t.costs["coincident_impact"]))
        self.assertAlmostEqual(costs[0], 1.)
        self.assertAlmostEqual(costs[1], .5, places=5)
        self.assertEqual(costs[2], 0.)
        t = self.tracker()
        self.sample(t, 300)
        for _ in range(10):
            self.sample(t, 300)
        self.assertEqual(t.costs["coincident_impact"].sum(), 0)

    def test_curriculum_boundaries_resume_and_reward_scaling(self):
        s = StairQuietSettings()
        clock = dict(iteration_offset=0, step_offset=100, steps_per_iteration=24)
        warm, ramp = s.curriculum_warmup_iterations, s.curriculum_ramp_iterations
        for iteration, expected in ((0, s.curriculum_initial_scale), (warm, s.curriculum_initial_scale),
                                    (warm + ramp // 2, .5 * (1. + s.curriculum_initial_scale)),
                                    (warm + ramp, 1.), (warm + ramp + 100, 1.)):
            actual, scale = scheduled_quiet_scale(clock, 100 + 24 * iteration, s)
            self.assertEqual(actual, iteration)
            self.assertAlmostEqual(scale, expected)
        resume = dict(iteration_offset=warm + ramp // 2, step_offset=0, steps_per_iteration=24)
        self.assertEqual(scheduled_quiet_scale(resume, 0, s),
                         scheduled_quiet_scale(clock, 100 + 24 * (warm + ramp // 2), s))
        with self.assertRaisesRegex(RuntimeError, "not installed"):
            scheduled_quiet_scale(None, 0, s)
        for bad in ({**clock, "steps_per_iteration": 0}, {**clock, "iteration_offset": -1},
                    {**clock, "step_offset": 1.5}):
            with self.assertRaises(ValueError):
                scheduled_quiet_scale(bad, 100, s)
        with self.assertRaisesRegex(ValueError, "rewound"):
            scheduled_quiet_scale(clock, 99, s)
        t = self.tracker()
        t.costs["contact_work"].fill_(2.)
        env = SimpleNamespace(stair_quiet_tracker=t, step_dt=.02, stair_quiet_reward_scale=.1)
        self.assertAlmostEqual(float(stair_quiet_cost(env, "contact_work") * env.step_dt), .2, places=6)
        self.assertEqual(t.costs["contact_work"], 2)  # raw diagnostics never ramped
        del env.stair_quiet_reward_scale
        with self.assertRaisesRegex(RuntimeError, "not resolved"):
            stair_quiet_cost(env, "contact_work")


class EnvironmentWiringTests(unittest.TestCase):
    def test_configuration_preserves_parent_profile_and_adds_costs_after_filter(self):
        """Run the real child post_init against a parent that removes unknowns."""
        package = ModuleType("stair_quiet_cfg_fixture")
        package.__path__ = []
        parent = ModuleType(package.__name__ + ".idle_stop_motion_v1_env_cfg")
        managers, utils = ModuleType("isaaclab.managers"), ModuleType("isaaclab.utils")
        managers.RewardTermCfg = lambda **kwargs: SimpleNamespace(**kwargs)
        utils.configclass = lambda cls: cls

        class ParentCfg:
            def __post_init__(self):
                self.rewards = SimpleNamespace(
                    traversal=SimpleNamespace(weight=3), power=SimpleNamespace(weight=-.1))
                self.observations = object()
                self.scene = SimpleNamespace(contact_forces=SimpleNamespace(update_period=.005))
                self.sim = SimpleNamespace(dt=.005)
                self.decimation = 4

        parent.DeeproboticsM20IdleStopMotionV1EnvCfg = ParentCfg
        file = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/stair_quiet_v1_env_cfg.py"
        spec = importlib.util.spec_from_file_location(package.__name__ + ".stair_quiet_v1_env_cfg", file)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {
            package.__name__: package, parent.__name__: parent,
            "isaaclab.managers": managers, "isaaclab.utils": utils,
        }):
            spec.loader.exec_module(module)
        cfg = module.DeeproboticsM20StairQuietV1EnvCfg()
        cfg.__post_init__()
        self.assertEqual(cfg.rewards.traversal.weight, 3)
        self.assertEqual(cfg.rewards.power.weight, -.1)
        self.assertEqual(cfg.sim.dt, .005)
        self.assertEqual(cfg.decimation, 4)
        for name in COST_NAMES:
            term = getattr(cfg.rewards, "stair_quiet_" + name)
            self.assertIs(term.func, stair_quiet_cost)
            self.assertEqual(term.params, {"name": name})
            self.assertEqual(term.weight, cfg.stair_quiet.weights[name])

    def test_real_task_step_pairs_substeps_and_keeps_terminal_step_evidence(self):
        """Execute the shipped step with a deterministic fake physics adapter.

        This proves hook ordering/reset wiring, not Isaac physics or climbing.
        """
        calls = []
        base_module = ModuleType("isaaclab.envs")

        class BaseEnv:
            def _reset_idx(self, ids):
                calls.append("reset")
                self.extras["log"] = {"baseline/power": 123}

        base_module.ManagerBasedRLEnv = BaseEnv
        file = ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/stair_quiet_v1_env.py"
        spec = importlib.util.spec_from_file_location("stair_quiet_env_cpu_fixture", file)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"isaaclab.envs": base_module}):
            spec.loader.exec_module(module)
        env = module.M20StairQuietV1Env.__new__(module.M20StairQuietV1Env)
        env.num_envs, env.device = 2, "cpu"
        env.physics_dt, env.step_dt = .005, .01
        env.cfg = SimpleNamespace(decimation=2, sim=SimpleNamespace(render_interval=2))
        env._sim_step_counter, env.common_step_counter = 0, 0
        env._m20_stair_quiet_schedule = dict(iteration_offset=0, step_offset=0, steps_per_iteration=24)
        env._quiet_body_ids = env._quiet_contact_ids = [0, 1, 2, 3]
        env.episode_length_buf = torch.zeros(2, dtype=torch.long)
        env.extras = {}
        env.stair_quiet_tracker = StairQuietTracker(2, "cpu", StairQuietSettings(reset_grace_time=0))
        robot = SimpleNamespace(data=SimpleNamespace(
            root_pos_w=torch.zeros(2, 3), body_lin_vel_w=torch.zeros(2, 4, 3),
            body_ang_vel_w=torch.zeros(2, 4, 3),
            body_link_pos_w=torch.zeros(2, 4, 3), body_com_pos_w=torch.zeros(2, 4, 3),
            body_link_quat_w=torch.tensor([1., 0., 0., 0.]).repeat(2, 4, 1)))
        sensor = SimpleNamespace(data=SimpleNamespace(net_forces_w=torch.zeros(2, 4, 3)))

        class Scene(dict):
            sensors = {"contact_forces": sensor}

            def write_data_to_sim(self):
                calls.append("write")

            def update(self, dt):
                calls.append("scene_update")
                robot.data.body_lin_vel_w[..., 2] = -2 if env._sim_step_counter == 1 else 0
                sensor.data.net_forces_w[..., 2] = 0 if env._sim_step_counter == 1 else 300

        env.scene = Scene(robot=robot)
        env.action_manager = SimpleNamespace(
            process_action=lambda x: calls.append("process"),
            apply_action=lambda: calls.append("apply"),
        )
        env.sim = SimpleNamespace(
            has_gui=lambda: False, has_rtx_sensors=lambda: False,
            step=lambda **kw: calls.append("physics"),
        )
        env.recorder_manager = SimpleNamespace(
            record_pre_step=lambda: None, record_post_physics_decimation_step=lambda: None,
            active_terms=[], record_pre_reset=lambda ids: None, record_post_reset=lambda ids: None,
        )
        env.termination_manager = SimpleNamespace(
            compute=lambda: torch.tensor([True, False]),
            terminated=torch.tensor([True, False]), time_outs=torch.tensor([False, False]),
        )

        def reward(dt):
            calls.append("reward")
            return env.stair_quiet_tracker.costs["touchdown"].clone()

        env.reward_manager = SimpleNamespace(compute=reward)
        env.command_manager = SimpleNamespace(compute=lambda **kw: calls.append("command"))
        env.event_manager = SimpleNamespace(
            available_modes=["interval"], apply=lambda **kw: calls.append("interval"))
        env.observation_manager = SimpleNamespace(compute=lambda **kw: calls.append("observe") or {})
        _, rewards, _, _, extras = env.step(torch.zeros(2, 16))
        self.assertTrue(torch.equal(rewards, torch.full((2,), 64.)))
        self.assertEqual(env.stair_quiet_step_events.sum(), 8)
        self.assertEqual(env.stair_quiet_tracker.event_count[0].sum(), 0)
        self.assertEqual(env.stair_quiet_tracker.event_count[1].sum(), 4)
        self.assertEqual(extras["log"]["baseline/power"], 123)
        self.assertEqual(extras["log"]["StairQuiet/front/preimpact_speed_rms_m_s"], 2)
        self.assertLess(calls.index("reward"), calls.index("reset"))
        self.assertEqual(calls[-4:], ["reset", "command", "interval", "observe"])
        self.assertEqual(calls[:9], ["process", "apply", "write", "physics", "scene_update",
                                    "apply", "write", "physics", "scene_update"])
        env.termination_manager.compute = lambda: torch.tensor([False, False])
        _, _, _, _, next_extras = env.step(torch.zeros(2, 16))
        self.assertNotIn("log", next_extras)


if __name__ == "__main__":
    unittest.main()
