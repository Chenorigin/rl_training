"""Analytic CPU counterexamples for stability signals; no simulation/training."""
import math
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/rl_training"))
from task_support.stair_trot_v1.stability import (
    action_jerk_cost,
    posture_cost,
    posture_errors,
    rolling_slip,
    touchdown_impact_proxy,
)

torch.set_num_threads(1)


class RollingContactTests(unittest.TestCase):
    def setUp(self):
        self.velocity = torch.tensor([[[0.9, 0.0, 0.0]]], dtype=torch.float64)
        self.omega = torch.tensor([[[0.0, 10.0, 0.0]]], dtype=torch.float64)
        self.axle = torch.tensor([[[0.0, -1.0, 0.0]]], dtype=torch.float64)
        self.force = torch.tensor([[[0.0, 0.0, 100.0]]], dtype=torch.float64)

    def test_pure_rolling_is_free_but_spinning_and_locked_skidding_are_not(self):
        pure, support = rolling_slip(self.velocity, self.omega, self.axle, self.force)
        spin, _ = rolling_slip(self.velocity * 0, self.omega, self.axle, self.force)
        skid, _ = rolling_slip(self.velocity, self.omega * 0, self.axle, self.force)
        self.assertTrue(support.item())
        self.assertAlmostEqual(pure.item(), 0.0, places=12)
        self.assertAlmostEqual(spin.item(), 0.81, places=12)
        self.assertAlmostEqual(skid.item(), 0.81, places=12)

    def test_sideways_sliding_and_wrong_spin_direction_are_detected(self):
        velocity = self.velocity + torch.tensor([[[0.0, 0.2, 0.0]]])
        sideways, _ = rolling_slip(velocity, self.omega, self.axle, self.force)
        reverse, _ = rolling_slip(self.velocity, -self.omega, self.axle, self.force)
        self.assertAlmostEqual(sideways.item(), 0.04, places=7)
        self.assertAlmostEqual(reverse.item(), 3.24, places=12)

    def test_world_yaw_rotation_and_force_scaling_preserve_the_signal(self):
        yaw = 0.77
        rotation = torch.tensor([
            [math.cos(yaw), -math.sin(yaw), 0],
            [math.sin(yaw), math.cos(yaw), 0],
            [0, 0, 1],
        ], dtype=torch.float64)
        velocity = self.velocity + torch.tensor([[[0.0, 0.2, 0.0]]])
        expected, _ = rolling_slip(velocity, self.omega, self.axle, self.force)
        actual, valid = rolling_slip(
            velocity @ rotation.T, self.omega @ rotation.T,
            self.axle @ rotation.T, 3 * self.force @ rotation.T,
        )
        torch.testing.assert_close(actual, expected)
        self.assertTrue(valid.item())

    def test_capsule_endpoint_with_camber_and_body_roll_uses_link_velocity(self):
        # At 30-degree camber the lower cap centre is 0.027 * sin(30) lower.
        axle = torch.tensor([[[0.0, -math.sqrt(3) / 2, -0.5]]], dtype=torch.float64)
        omega = torch.tensor([[[2.0, 0.0, 0.0]]], dtype=torch.float64)
        velocity = torch.tensor(
            [[[0.0, -2 * (0.09 + 0.027 * 0.5), 0.027 * math.sqrt(3)]]],
            dtype=torch.float64,
        )
        correct, _ = rolling_slip(velocity, omega, axle, self.force)
        sphere, _ = rolling_slip(velocity, omega, axle, self.force, capsule_half_length=0)
        self.assertAlmostEqual(correct.item(), 0.0, places=12)
        self.assertGreater(sphere.item(), 0.0007)

    def test_normal_velocity_is_not_tangential_slip(self):
        velocity = self.velocity + torch.tensor([[[0.0, 0.0, -0.3]]])
        slip, valid = rolling_slip(velocity, self.omega, self.axle, self.force)
        self.assertTrue(valid.item())
        self.assertAlmostEqual(slip.item(), 0.0, places=12)

    def test_airborne_riser_side_contacts_and_nonfinite_data_are_not_support(self):
        forces = (
            self.force * 0,
            -self.force,
            torch.tensor([[[100.0, 0.0, 5.0]]], dtype=torch.float64),
            torch.full_like(self.force, float("nan")),
        )
        for force in forces:
            slip, valid = rolling_slip(self.velocity, self.omega, self.axle, force)
            self.assertFalse(valid.item())
            self.assertEqual(slip.item(), 0)


class PostureTests(unittest.TestCase):
    def test_uphill_pitch_sign_is_derived_from_rotated_gravity(self):
        pitch = math.radians(20)
        gravity = torch.tensor([[-math.sin(pitch), 0.0, -math.cos(pitch)]])
        front, rear = torch.tensor([math.tan(pitch) * 0.6]), torch.tensor([0.0])
        roll, error = posture_errors(gravity, front, rear, 0.6)
        torch.testing.assert_close(roll, torch.zeros_like(roll), atol=1.e-7, rtol=0)
        torch.testing.assert_close(error, torch.zeros_like(error), atol=1.e-7, rtol=0)
        cost = posture_cost(gravity, torch.zeros(1), front, rear, 0.6)
        wrong = gravity.clone()
        wrong[:, 0] *= -1
        bad = posture_cost(wrong, torch.zeros(1), front, rear, 0.6)
        self.assertEqual(cost.item(), 0)
        self.assertGreater(bad.item(), 5)

    def test_roll_is_separate_from_pitch_and_roll_rate_is_penalized(self):
        nose_up, roll = math.radians(15), math.radians(12)
        gravity = torch.tensor([[
            -math.sin(nose_up), -math.sin(roll) * math.cos(nose_up),
            -math.cos(roll) * math.cos(nose_up),
        ]])
        front, rear = torch.tensor([math.tan(nose_up) * 0.6]), torch.zeros(1)
        measured_roll, pitch_error = posture_errors(gravity, front, rear, 0.6)
        self.assertAlmostEqual(measured_roll.item(), roll, places=6)
        self.assertAlmostEqual(pitch_error.item(), 0, places=6)
        still = posture_cost(gravity, torch.zeros(1), front, rear, 0.6)
        moving = posture_cost(gravity, torch.ones(1), front, rear, 0.6)
        self.assertGreater(still.item(), 0)
        self.assertGreater(moving.item(), still.item())

    def test_pitch_band_accepts_flat_and_required_descent_without_forcing_level(self):
        for pitch in (0.0, -math.radians(25)):
            gravity = torch.tensor([[-math.sin(pitch), 0.0, -math.cos(pitch)]])
            cost = posture_cost(
                gravity, torch.zeros(1), torch.tensor([0.6 * math.tan(pitch)]),
                torch.zeros(1), 0.6,
            )
            self.assertEqual(cost.item(), 0)
        inverted = posture_cost(
            torch.tensor([[0.0, 0.0, 1.0]]), torch.zeros(1),
            torch.zeros(1), torch.zeros(1), 0.6,
        )
        self.assertGreater(inverted.item(), 100)


class SmoothnessAndImpactTests(unittest.TestCase):
    def test_zero_nonzero_oscillation_is_not_masked_but_reset_history_is(self):
        current = torch.tensor([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
        previous = torch.tensor([[1.0, 0.0]]).expand(3, -1)
        ages = torch.tensor([0, 1, 2])
        cost = action_jerk_cost(current, previous, current, ages)
        torch.testing.assert_close(cost, torch.tensor([0.0, 0.0, 4.0]))

    def test_constant_action_and_constant_action_rate_have_zero_second_difference(self):
        current = torch.tensor([[2.0, 0.0], [2.0, -2.0]])
        previous = torch.tensor([[2.0, 0.0], [1.0, -1.0]])
        oldest = torch.tensor([[2.0, 0.0], [0.0, 0.0]])
        cost = action_jerk_cost(current, previous, oldest, torch.tensor([4, 4]))
        torch.testing.assert_close(cost, torch.zeros(2))

    def test_previous_approach_catches_impact_after_solver_stops_velocity(self):
        # Current post-solver wheel velocity would be zero. The API deliberately
        # takes the earlier approaching velocity instead.
        previous = torch.tensor([[[0.0, 0.0, -0.8]]])
        contact = torch.tensor([[True]])
        force = torch.tensor([[[[0.0, 0.0, 60.0]], [[0.0, 0.0, 0.0]]]])
        cost = touchdown_impact_proxy(previous, contact, force, torch.tensor([3]))
        self.assertAlmostEqual(cost.item(), 1.96, places=6)

    def test_force_history_peak_catches_missing_approach_sample_once(self):
        previous = torch.zeros(1, 1, 3)
        history = torch.tensor([[[[0.0, 0.0, 60.0]], [[0.0, 0.0, 350.0]]]])
        first = torch.tensor([[True]])
        peak = touchdown_impact_proxy(previous, first, history, torch.tensor([3]))
        held = touchdown_impact_proxy(previous, ~first, history, torch.tensor([3]))
        reset = touchdown_impact_proxy(previous, first, history, torch.tensor([0]))
        self.assertEqual(peak.item(), 1)
        self.assertEqual(held.item(), 0)
        self.assertEqual(reset.item(), 0)

    def test_soft_touchdown_and_riser_face_contact_do_not_count_as_hard_tread_landings(self):
        first, age = torch.tensor([[True]]), torch.tensor([2])
        previous = torch.tensor([[[0.0, 0.0, -0.05]]])
        gentle = torch.tensor([[[[0.0, 0.0, 50.0]]]])
        side = torch.tensor([[[[500.0, 0.0, 0.0]]]])
        self.assertEqual(touchdown_impact_proxy(previous, first, gentle, age).item(), 0)
        self.assertEqual(touchdown_impact_proxy(previous, first, side, age).item(), 0)

    def test_nonfinite_force_cannot_create_an_apparently_valid_impact_sample(self):
        previous = torch.tensor([[[0.0, 0.0, -1.0]]])
        history = torch.tensor([[[[float("nan"), 0.0, 500.0]]]])
        cost = touchdown_impact_proxy(
            previous, torch.tensor([[True]]), history, torch.tensor([2])
        )
        self.assertEqual(cost.item(), 0)


if __name__ == "__main__":
    unittest.main()
