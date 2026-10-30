"""CTS auxiliary objectives: real CPU gradients, sensor ABI and data isolation.

These tests use tiny networks and bounded synthetic rollouts. They never
launch a simulator or a long-running optimizer loop.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from test_cts_algorithm import fixture
from task_support.cts.algorithm import CTSAlgorithm
from m20_student_v1.configuration import StudentV1PerceptionCfg
from m20_student_v1.retention import CLASSES
from m20_student_v1.stability import action_group_mse, temporal_residual_loss, perturb_sensors

torch.set_num_threads(1)


def aux_config(**overrides):
    result = dict(temporal_loss_weight=0., spatial_loss_weight=0., stability_batch_size=32,
                  spatial_noise_scale=.5, retention_loss_weight=0., retention_manifest=None,
                  retention_sha256=None, retention_command_manifest=None, retention_command_sha256=None,
                  retention_batch_size=10, leg_loss_weight=1., wheel_loss_weight=1.)
    result.update(overrides)
    return result


def make_algorithm(auxiliary=None, *, perception=None):
    models, original, obs, *_ = fixture()
    cfg = deepcopy(original.cfg)
    cfg.update(student_learning_epochs=1, student_batch_size=32)
    perception = deepcopy(perception or StudentV1PerceptionCfg())
    perception.gyro_std_rad_s = .4
    perception.height_noise_std = .1
    # Nonzero decoder makes encoder-only auxiliary losses observably wrong.
    with torch.no_grad():
        models.decoder.network[-1].weight.fill_(.4)
        models.decoder.network[-1].bias.fill_(.2)
    obs["policy"][:, 244:431] = 1.
    obs["policy"][:, 244:254] = 0.
    obs["policy"][:, 57:67] = 0.
    alg = CTSAlgorithm(models, cfg, 8, 4, "cpu", auxiliary_cfg=auxiliary, perception=vars(perception))
    return alg, obs


def collect(alg, obs, *, exclusions=False):
    with torch.inference_mode():
        for step in range(4):
            current = obs.clone()
            current["policy"][:, 0] += .1*step
            current["policy"][:, 8] += .01*step  # effective heading, not raw request
            current["teacher"][:, 0] -= .04*step
            phase = torch.zeros(8, dtype=torch.long)
            done = torch.zeros(8, dtype=torch.bool)
            if exclusions:
                if step == 0: done[1] = True
                if step == 1: phase[2] = 1
                if step >= 1: current["motion_control"][3, 0] = .8
            alg.act(current)
            alg.process_env_step(current, torch.zeros(8), done, {"student_v1": {"phase": phase}})
        alg.compute_returns(current)


def selection_field(selection, name):
    return selection[name] if isinstance(selection, dict) else getattr(selection, name)


def teacher_actions(alg, data):
    with torch.no_grad():
        latent = alg.models.teacher_latent(data["obs"])
        return alg.models.decoder(data["obs"]["policy"][:, :57], latent,
                                  data["obs"]["motion_control"]).detach()


def capture_distill_gradient(alg):
    gradients = []
    step = alg.student_optimizer.step
    def capture(*args, **kwargs):
        gradients.append(torch.cat([p.grad.detach().flatten().clone() for p in alg.student_parameters]))
        return step(*args, **kwargs)
    with patch.object(alg.student_optimizer, "step", capture):
        torch.manual_seed(146)
        metrics = alg.distill_update()
    return gradients[0], metrics


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assert_state_equal(test, actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        test.assertEqual(set(actual), set(expected))
        for key in expected: assert_state_equal(test, actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        test.assertEqual(type(actual), type(expected))
        test.assertEqual(len(actual), len(expected))
        for value, reference in zip(actual, expected): assert_state_equal(test, value, reference)
    else:
        test.assertEqual(actual, expected)


def retention_dataset(folder, *, heldout_shift=0., missing_class=False, startup=False):
    folder = Path(folder)
    random = np.random.default_rng(18)
    classes = np.tile(np.arange(5, dtype=np.int64), 6)
    if missing_class: classes[classes == 4] = 0
    heldout = np.arange(30) >= 15
    obs = random.normal(0., .1, (30, 789)).astype(np.float32)
    obs[:, 244:431] = 1.
    labels = random.normal(0., .5, (30, 16)).astype(np.float32)
    labels[heldout] += heldout_shift
    payload = folder / "samples.npz"
    np.savez_compressed(payload, observations=obs, labels=labels, classes=classes, holdout=heldout)
    manifest = folder / "manifest.json"
    manifest.write_text(json.dumps(dict(schema="m20_nonforward_retention_v1", obs_dim=789, action_dim=16,
                                       classes=list(CLASSES), sample_count=30, payload=payload.name,
                                       payload_sha256=file_hash(payload), source_checkpoint_sha256="b"*64)))
    if startup:
        metadata = json.loads(manifest.read_text())
        metadata['sampling'] = dict(mode='class_startup_balanced_v1', startup_fraction=.5,
                                    startup_policy_steps=1, startup_indices=list(range(5))+list(range(15,20)))
        manifest.write_text(json.dumps(metadata))
    commands = folder / "commands.json"
    commands.write_text(json.dumps(dict(schema="m20_cts_retention_commands_v1", classes=list(CLASSES),
                                       commands=[[-.7, 0., 0.], [0., .5, 0.], [0., -.5, 0.], [0., 0., .6], [0., 0., -.6]],
                                       retention_manifest_sha256=file_hash(manifest), payload_sha256=file_hash(payload),
                                       source_checkpoint_sha256="b"*64, source_plan="fixture-plan.json",
                                       source_plan_sha256="c"*64, semantics="Synthetic requested commands for tests")))
    return dict(retention_loss_weight=1., retention_manifest=str(manifest), retention_sha256=file_hash(manifest),
                retention_command_manifest=str(commands), retention_command_sha256=file_hash(commands))


class CTSAuxiliaryTests(unittest.TestCase):
    def test_each_stability_objective_changes_actual_student_gradient_only(self):
        results = []
        for cfg in (aux_config(), aux_config(temporal_loss_weight=.5), aux_config(spatial_loss_weight=.1)):
            with self.subTest(config=cfg):
                alg, obs = make_algorithm(cfg)
                collect(alg, obs)
                before = deepcopy(alg.models.state_dict())
                gradient, metrics = capture_distill_gradient(alg)
                self.assertTrue(any(not torch.equal(value, alg.models.state_dict()[key])
                                    for key, value in before.items() if key.startswith("student.mlp.")))
                for prefix in ("teacher.", "decoder.", "critic.", "reference_teacher."):
                    self.assertTrue(all(torch.equal(value, alg.models.state_dict()[key])
                                        for key, value in before.items() if key.startswith(prefix)), prefix)
                for module in (alg.models.teacher, alg.models.decoder, alg.models.critic):
                    self.assertTrue(all(parameter.grad is None for parameter in module.parameters()))
                results.append((gradient, metrics))
        self.assertEqual(results[0][1]["temporal_residual_mse"], 0.)
        self.assertEqual(results[0][1]["spatial_action_mse"], 0.)
        self.assertGreater(results[1][1]["temporal_residual_mse"], 0.)
        self.assertGreater(results[2][1]["spatial_action_mse"], 0.)
        self.assertFalse(torch.equal(results[0][0], results[1][0]))
        self.assertFalse(torch.equal(results[0][0], results[2][0]))

    def test_temporal_pairs_exclude_reset_changed_raw_command_and_both_nonregular_ends(self):
        alg, obs = make_algorithm(aux_config(temporal_loss_weight=.5))
        collect(alg, obs, exclusions=True)
        selection = alg.auxiliary.prepare(alg.rollout)
        current = selection_field(selection, "pairs_current")
        previous = selection_field(selection, "pairs_previous")
        self.assertEqual(set(current.tolist()), set(range(8, 32))-{9, 10, 11, 18})
        torch.testing.assert_close(previous, current-8, rtol=0, atol=0)
        # Effective heading changed on each step, but the human request did not.
        self.assertTrue({8, 16, 24}.issubset(set(current.tolist())))
        with self.assertRaises(ValueError): alg.auxiliary.regular_phase({}, 8)
        with self.assertRaises(ValueError): alg.auxiliary.regular_phase({"student_v1": {"phase": torch.zeros(7)}}, 8)

    def test_temporal_objective_uses_complete_nonidentity_decoder_actions(self):
        alg, obs = make_algorithm(aux_config(temporal_loss_weight=1., stability_batch_size=32))
        collect(alg, obs)
        data = alg._data()
        selection = alg.auxiliary.prepare(alg.rollout)
        current = selection_field(selection, "pairs_current")
        previous = selection_field(selection, "pairs_previous")
        target = teacher_actions(alg, data)
        policy, raw = data["obs"]["policy"], data["obs"]["motion_control"]
        full = alg.models.student_action(policy, raw)
        expected = temporal_residual_loss(full[current], full[previous], target[current], target[previous])
        encoded = alg.models.student_latent(data["obs"])
        wrong = temporal_residual_loss(encoded[current], encoded[previous], target[current], target[previous])
        self.assertGreater(abs(float(expected-wrong)), 1.e-10)
        # Select every valid pair once so this checks the objective itself,
        # independently of the implementation's with-replacement sampler.
        with patch("task_support.cts.auxiliary.torch.randint", return_value=torch.arange(len(current))):
            loss, metrics = alg.auxiliary.loss(data, torch.arange(len(full)), target, selection)
        torch.testing.assert_close(loss, expected, rtol=1.e-5, atol=1.e-8)
        self.assertAlmostEqual(float(metrics["temporal_residual_mse"]), float(expected), places=7)

    def test_spatial_objective_preserves_commands_masks_history_unknowns_and_raw_command(self):
        alg, obs = make_algorithm(aux_config(spatial_loss_weight=1., stability_batch_size=32))
        collect(alg, obs)
        data = alg._data()
        original = {key: value.clone() for key, value in data["obs"].items()}
        target = teacher_actions(alg, data)
        selection = alg.auxiliary.prepare(alg.rollout)
        calls = []
        student_action = alg.models.student_action
        def capture(policy, raw_command):
            calls.append((policy.detach().clone(), raw_command.detach().clone()))
            return student_action(policy, raw_command)
        with patch.object(alg.models, "student_action", capture):
            loss, metrics = alg.auxiliary.loss(data, torch.arange(32), target, selection)
        self.assertGreater(float(loss), 0.)
        self.assertGreater(float(metrics["spatial_action_mse"]), 0.)
        for key, value in original.items(): torch.testing.assert_close(data["obs"][key], value, rtol=0, atol=0)
        altered = [(policy, raw) for policy, raw in calls if len(policy) == 32 and not torch.equal(policy, original["policy"])]
        self.assertTrue(altered, "Actual CTS loss must run the perturbed complete student action path")
        for policy, raw in altered:
            for start, end in ((6, 9), (21, 25), (41, 57), (244, 789), (57, 67)):
                torch.testing.assert_close(policy[:, start:end], original["policy"][:, start:end], rtol=0, atol=0)
            torch.testing.assert_close(raw, original["motion_control"], rtol=0, atol=0)
            self.assertGreater(float((policy[:, :3]-original["policy"][:, :3]).abs().sum()), 0.)

    def test_spatial_loss_matches_complete_decoder_action_difference(self):
        alg, obs = make_algorithm(aux_config(spatial_loss_weight=1., stability_batch_size=32))
        collect(alg, obs)
        data = alg._data()
        policy, raw = data["obs"]["policy"], data["obs"]["motion_control"]
        torch.manual_seed(81)
        alternate = perturb_sensors(policy, alg.auxiliary.contract["sensor_noise_std"], .5)
        expected = action_group_mse(alg.models.student_action(alternate, raw)-alg.models.student_action(policy, raw), 1., 1.)
        wrong = action_group_mse(alg.models.student_latent({"policy": alternate})-alg.models.student_latent({"policy": policy}), 1., 1.)
        self.assertGreater(abs(float(expected-wrong)), 1.e-12)
        with patch("task_support.cts.auxiliary.perturb_sensors", return_value=alternate):
            loss, _ = alg.auxiliary.loss(data, torch.arange(32), teacher_actions(alg, data), alg.auxiliary.prepare(alg.rollout))
        torch.testing.assert_close(loss, expected, rtol=1.e-6, atol=1.e-9)

    def test_disabled_default_and_invalid_missing_phase_fail_closed(self):
        alg, _ = make_algorithm(None)
        self.assertEqual(alg.auxiliary.contract["version"], 1)
        for key in ("temporal_loss_weight", "spatial_loss_weight", "retention_loss_weight"):
            self.assertEqual(alg.auxiliary.contract["config"][key], 0.)
        for cfg in (aux_config(temporal_loss_weight=-1.), aux_config(spatial_loss_weight=float("nan"))):
            with self.assertRaises(ValueError): make_algorithm(cfg)
        active, obs = make_algorithm(aux_config(temporal_loss_weight=.5))
        active.act(obs)
        with self.assertRaises(ValueError): active.process_env_step(obs, torch.zeros(8), torch.zeros(8), {})

    def test_resume_auxiliary_configuration_and_derived_noise_reject_atomically(self):
        auxiliary = aux_config(temporal_loss_weight=.5, spatial_loss_weight=.1)
        trained, obs = make_algorithm(auxiliary)
        collect(trained, obs)
        trained.update()
        saved = deepcopy(trained.save())
        alg, _ = make_algorithm(auxiliary)
        original_state = deepcopy(alg.save())
        self.assertIn("auxiliary_contract", saved)
        for field in ("config", "sensor_noise_std"):
            with self.subTest(field=field):
                bad = deepcopy(saved)
                if field == "config": bad["auxiliary_contract"][field]["temporal_loss_weight"] = .9
                else: bad["auxiliary_contract"][field]["gyro"] += .01
                with self.assertRaises(ValueError): alg.load(bad)
                assert_state_equal(self, alg.save(), original_state)
        alg.load(saved)
        self.assertTrue(alg.optimizer.state)
        self.assertTrue(alg.student_optimizer.state)
        changed_perception = StudentV1PerceptionCfg(roll_pitch_std=.2)
        changed, _ = make_algorithm(auxiliary, perception=changed_perception)
        unchanged = deepcopy(changed.save())
        with self.assertRaises(ValueError): changed.load(saved)
        assert_state_equal(self, changed.save(), unchanged)


class CTSRetentionTests(unittest.TestCase):
    def test_actual_retention_bank_balances_classes_and_keeps_holdout_separate(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg = aux_config(**retention_dataset(folder))
            alg, _ = make_algorithm(cfg)
            bank = alg.auxiliary.retention
            for heldout in (False, True):
                obs, _, classes = bank.sample(23, heldout=heldout)
                self.assertEqual(torch.bincount(classes).tolist(), [5, 5, 5, 4, 4])
                population = bank.obs[15:] if heldout else bank.obs[:15]
                self.assertTrue((obs[:, None] == population[None]).all(-1).any(-1).all())
            self.assertEqual(alg.auxiliary.contract["retention"]["training_counts"], [3]*5)
            self.assertEqual(alg.auxiliary.contract["retention"]["heldout_counts"], [3]*5)
            self.assertGreater(alg.auxiliary.memory_bytes, 30*(789+16)*4)

    def test_retention_changes_gradient_but_heldout_labels_cannot_train_any_model(self):
        outcomes = []
        with tempfile.TemporaryDirectory() as folder:
            for weight, shift in ((0., 0.), (1., 0.), (1., 100.)):
                cfg = aux_config(**retention_dataset(folder, heldout_shift=shift))
                cfg["retention_loss_weight"] = weight
                alg, obs = make_algorithm(cfg)
                collect(alg, obs)
                before = deepcopy(alg.models.state_dict())
                gradient, metrics = capture_distill_gradient(alg)
                for key, value in before.items():
                    if not key.startswith("student."):
                        torch.testing.assert_close(alg.models.state_dict()[key], value, rtol=0, atol=0)
                for module in (alg.models.teacher, alg.models.decoder, alg.models.critic):
                    self.assertTrue(all(p.grad is None for p in module.parameters()))
                outcomes.append((gradient, metrics, deepcopy(alg.models.student.state_dict())))
        self.assertEqual(outcomes[0][1]["retention_mse"], 0.)
        self.assertGreater(outcomes[1][1]["retention_mse"], 0.)
        self.assertFalse(torch.equal(outcomes[0][0], outcomes[1][0]))
        torch.testing.assert_close(outcomes[1][0], outcomes[2][0], rtol=0, atol=0)
        for key, value in outcomes[1][2].items():
            torch.testing.assert_close(value, outcomes[2][2][key], rtol=0, atol=0)
        self.assertGreater(outcomes[2][1]["retention_heldout_mse"], outcomes[1][1]["retention_heldout_mse"])
        for command in CLASSES:
            self.assertGreater(outcomes[1][1][f"retention_{command}_mse"], 0.)

    def test_retention_compares_complete_decoder_output_with_pinned_raw_commands(self):
        with tempfile.TemporaryDirectory() as folder:
            alg, obs = make_algorithm(aux_config(**retention_dataset(folder)))
            collect(alg, obs)
            data = alg._data()
            retained_policy, labels, classes = alg.auxiliary.retention.sample(10)
            # Effective yaw differs deliberately from the requested command;
            # use the sidecar, never infer requested motion from policy input.
            retained_policy[:, 6:9] = 0.
            commands = alg.auxiliary.commands[classes]
            full = alg.models.student_action(retained_policy, commands)
            expected = action_group_mse(full-labels, 1., 1.)
            encoded = alg.models.student_latent({"policy": retained_policy})
            wrong = action_group_mse(encoded-labels, 1., 1.)
            self.assertGreater(abs(float(expected-wrong)), 1.e-6)
            with patch.object(alg.auxiliary.retention, "sample", return_value=(retained_policy, labels, classes)):
                with patch.object(alg.models, "student_action", wraps=alg.models.student_action) as call:
                    actual, metrics = alg.auxiliary.loss(data, torch.arange(32), teacher_actions(alg, data),
                                                          alg.auxiliary.prepare(alg.rollout))
            torch.testing.assert_close(actual, expected, rtol=1.e-6, atol=1.e-7)
            self.assertEqual(metrics["retention_training_samples"], 10)
            torch.testing.assert_close(call.call_args.args[1], commands, rtol=0, atol=0)

    def test_retention_sidecar_binding_classes_and_hashes_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            for corruption in ("binding", "direction", "missing_class", "manifest_hash"):
                with self.subTest(corruption=corruption):
                    cfg = aux_config(**retention_dataset(folder, missing_class=corruption == "missing_class"))
                    path = Path(cfg["retention_command_manifest"])
                    if corruption in ("binding", "direction"):
                        value = json.loads(path.read_text())
                        if corruption == "binding": value["payload_sha256"] = "0"*64
                        else: value["commands"][0] = [.7, 0., 0.]
                        path.write_text(json.dumps(value))
                        cfg["retention_command_sha256"] = file_hash(path)
                    if corruption == "manifest_hash": cfg["retention_sha256"] = "0"*64
                    with self.assertRaises(ValueError): make_algorithm(cfg)

    def test_resume_retention_receipt_drift_rejects_before_model_or_rng_mutation(self):
        with tempfile.TemporaryDirectory() as folder:
            alg, _ = make_algorithm(aux_config(**retention_dataset(folder)))
            saved = deepcopy(alg.save())
            before = deepcopy(alg.models.state_dict())
            rng = torch.get_rng_state().clone()
            bad = deepcopy(saved)
            bad["auxiliary_contract"]["retention"]["payload_sha256"] = "0"*64
            bad["models_state_dict"]["decoder.network.4.bias"].fill_(.9)
            with self.assertRaises(ValueError): alg.load(bad)
            for key, value in before.items(): torch.testing.assert_close(value, alg.models.state_dict()[key], rtol=0, atol=0)
            torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
            alg.load(saved)


class CTSBackwardProtectionTests(unittest.TestCase):
    def test_ppo_rehearsal_differentiates_decoder_only(self):
        with tempfile.TemporaryDirectory() as folder:
            alg, _ = make_algorithm(aux_config(**retention_dataset(folder, startup=True)))
            alg.models.zero_grad(set_to_none=True)
            loss, metrics = alg.auxiliary.retention_loss(detach_student=True)
            self.assertGreater(metrics['retention_backward_startup_mse'], 0.)
            loss.backward()
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in alg.models.decoder.parameters()))
            for module in (alg.models.student, alg.models.teacher, alg.models.reference_teacher, alg.models.critic):
                self.assertTrue(all(p.grad is None for p in module.parameters()))

    def test_actual_ppo_gradient_includes_retention(self):
        with tempfile.TemporaryDirectory() as folder:
            alg, obs = make_algorithm(aux_config(**retention_dataset(folder, startup=True)))
            alg.cfg.update(num_learning_epochs=1, num_mini_batches=1)
            collect(alg, obs)
            gradients = []
            def capture():
                gradients.append(torch.cat([p.grad.flatten().clone() for p in alg.models.decoder.parameters()]))
            objective = alg.auxiliary.retention_loss
            def disabled(**kwargs):
                value, metrics = objective(**kwargs)
                return value*0., metrics  # Same sampling/RNG and batch, only remove the gradient.
            with patch.object(alg.optimizer, 'step', side_effect=capture):
                torch.manual_seed(22)
                with patch.object(alg.auxiliary, 'retention_loss', side_effect=disabled):
                    alg.ppo_update()
                torch.manual_seed(22)
                metrics = alg.ppo_update()
            self.assertEqual(len(gradients), 2)
            self.assertFalse(torch.equal(gradients[0], gradients[1]))
            self.assertGreater(metrics['ppo_retention_backward_startup_mse'], 0.)
            self.assertTrue(all(p.grad is None for p in alg.student_parameters))

    def test_sampler_balances_classes_and_phases_without_split_leakage(self):
        with tempfile.TemporaryDirectory() as folder:
            alg, _ = make_algorithm(aux_config(**retention_dataset(folder, startup=True)))
            engine = alg.auxiliary
            for heldout in (False, True):
                torch.manual_seed(1)
                policy, labels, classes, startup = engine.sample_retention(heldout=heldout)
                self.assertEqual(len(policy), 10)
                self.assertEqual(int(startup.sum()), 5)
                for index in range(5):
                    self.assertEqual(int((classes == index).sum()), 2)
                    self.assertEqual(int(((classes == index) & startup).sum()), 1)
                allowed = torch.cat(engine.retention.pools['heldout' if heldout else 'train'])
                for row, target in zip(policy, labels):
                    match = (engine.retention.obs[allowed] == row).all(-1)
                    self.assertTrue(match.any())
                    torch.testing.assert_close(engine.retention.labels[allowed[match][0]], target)
            metrics = engine.evaluate()
            self.assertIn('retention_backward_startup_heldout_mse', metrics)
            self.assertIn('retention_backward_later_heldout_mse', metrics)

    def test_heldout_labels_cannot_influence_ppo_rehearsal_gradient(self):
        gradients = []
        for shift in (0., 1000.):
            with tempfile.TemporaryDirectory() as folder:
                cfg = aux_config(**retention_dataset(folder, startup=True, heldout_shift=shift))
                alg, _ = make_algorithm(cfg)
                torch.manual_seed(19)
                loss, _ = alg.auxiliary.retention_loss(detach_student=True)
                loss.backward()
                gradients.append(torch.cat([p.grad.flatten() for p in alg.models.decoder.parameters()]))
        torch.testing.assert_close(gradients[0], gradients[1], rtol=0, atol=0)

    def test_old_optimization_version_is_rejected_before_mutation(self):
        alg, _ = make_algorithm(aux_config())
        saved = deepcopy(alg.save())
        self.assertEqual(saved['cts_version'], 3)
        saved['cts_version'] = 2
        before = deepcopy(alg.save())
        with self.assertRaisesRegex(ValueError, 'contract changed'):
            alg.load(saved)
        assert_state_equal(self, alg.save(), before)

    def test_weight_transfer_preserves_models_and_starts_fresh_optimizers_and_counters(self):
        from task_support.cts.entry import initialize_cts_weights
        from task_support.cts.configuration import LEGACY_VERSION
        trained, obs = make_algorithm(aux_config())
        collect(trained, obs)
        trained.update()  # Tiny synthetic fixture only, no real checkpoint training.
        previous = deepcopy(trained.save())
        previous['cts_version'] = 2
        saved = {'cts': {'version': LEGACY_VERSION,
                         'config': {'algorithm': trained.cfg},
                         'auxiliary_contract': previous['auxiliary_contract']}, 'algorithm': previous}
        fresh, _ = make_algorithm(aux_config())
        initialize_cts_weights(fresh, saved)
        assert_state_equal(self, fresh.models.state_dict(), trained.models.state_dict())
        self.assertEqual(fresh.num_updates, 0)
        self.assertFalse(fresh.optimizer.state)
        self.assertFalse(fresh.student_optimizer.state)
        before = deepcopy(fresh.save())
        saved['algorithm']['models_state_dict']['teacher.mlp.0.weight'][0, 0] += 1.
        with self.assertRaisesRegex(ValueError, 'frozen'):
            initialize_cts_weights(fresh, saved)
        assert_state_equal(self, fresh.save(), before)


if __name__ == "__main__":
    unittest.main()
