"""Pinned motion rehearsal: balanced directions, split isolation and real gradient wiring."""
from pathlib import Path
import hashlib
import json
import tempfile
import unittest

import numpy as np
import torch

from test_student_stability import collect_fixture
from m20_student_v1.retention import MotionRetention, CLASSES, retention_bytes


def dataset(folder, *, heldout_shift=0., missing_class=False):
    rng = np.random.default_rng(18)
    groups = np.tile(np.arange(5, dtype=np.int64), 6)
    if missing_class:
        groups[groups == 4] = 0
    holdout = np.arange(30) >= 15
    obs = rng.normal(0., .1, (30, 789)).astype(np.float32)
    labels = rng.normal(0., .5, (30, 16)).astype(np.float32)
    labels[holdout] += heldout_shift
    np.savez_compressed(folder/'samples.npz', observations=obs, labels=labels, classes=groups, holdout=holdout)
    meta = {"schema": "m20_nonforward_retention_v1", "obs_dim": 789, "action_dim": 16,
            "classes": list(CLASSES), "sample_count": 30, "payload": "samples.npz",
            "payload_sha256": hashlib.sha256((folder/'samples.npz').read_bytes()).hexdigest(),
            "source_checkpoint_sha256": "b" * 64}
    path = folder/'manifest.json';path.write_text(json.dumps(meta))
    return str(path), hashlib.sha256(path.read_bytes()).hexdigest()


class RetentionTests(unittest.TestCase):
    def test_balanced_sampling_uses_all_directions_and_never_crosses_holdout(self):
        with tempfile.TemporaryDirectory() as name:
            path, sha = dataset(Path(name))
            bank = MotionRetention(path, sha, 'cpu')
            for heldout in (False, True):
                obs, labels, classes = bank.sample(256, heldout=heldout)
                self.assertEqual(torch.bincount(classes).tolist(), [52, 51, 51, 51, 51])
                population = bank.obs[15:] if heldout else bank.obs[:15]
                self.assertTrue((obs[:, None] == population[None]).all(-1).any(-1).all())
            self.assertEqual(bank.receipt['training_counts'], [3] * 5)
            self.assertEqual(retention_bytes(path, sha), 30*((789+16)*4+17))

    def test_hash_corruption_missing_class_and_small_batch_fail_closed(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name);path, sha = dataset(folder)
            with self.assertRaisesRegex(ValueError, 'manifest'):
                MotionRetention(path, '0'*64, 'cpu')
            with (folder/'samples.npz').open('ab') as f:
                f.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'payload hash'):
                MotionRetention(path, sha, 'cpu')
            path, sha = dataset(folder, missing_class=True)
            with self.assertRaisesRegex(ValueError, 'Every retention command'):
                MotionRetention(path, sha, 'cpu')
            path, sha = dataset(folder)
            with self.assertRaisesRegex(ValueError, 'all five commands'):
                MotionRetention(path, sha, 'cpu').sample(4)

    def test_real_update_has_retention_gradient_and_heldout_labels_cannot_train(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            outcomes = []
            for weight, heldout_shift in ((0., 0.), (1., 0.), (1., 100.)):
                path, sha = dataset(folder, heldout_shift=heldout_shift)
                alg = collect_fixture(retention_loss_weight=weight, retention_manifest=path,
                                      retention_sha256=sha, retention_batch_size=10)
                teacher = {k:v.clone() for k,v in alg.teacher.state_dict().items()}
                gradients = []
                original = alg.optimizer.step
                def step():
                    gradients.append(torch.cat([p.grad.flatten().clone() for p in alg.student.mlp.parameters()]))
                    original()
                alg.optimizer.step = step
                torch.manual_seed(29)
                stats = alg.update()
                self.assertTrue(all(torch.equal(v, alg.teacher.state_dict()[k]) for k,v in teacher.items()))
                outcomes.append((stats, gradients[0], alg))
            self.assertEqual(outcomes[0][0]['retention_mse'], 0.)
            self.assertGreater(outcomes[1][0]['retention_mse'], 0.)
            self.assertFalse(torch.equal(outcomes[0][1], outcomes[1][1]))
            self.assertTrue(torch.equal(outcomes[1][1], outcomes[2][1]), 'Heldout labels must not affect training gradients')
            self.assertGreater(outcomes[2][0]['retention_heldout_mse'], outcomes[1][0]['retention_heldout_mse'])
            for command in CLASSES:
                self.assertGreater(outcomes[1][0]['retention_'+command+'_mse'], 0.)
            saved = outcomes[1][2].save()
            path, sha = dataset(folder)
            restored = collect_fixture(retention_loss_weight=1., retention_manifest=path, retention_sha256=sha,
                                       retention_batch_size=10)
            restored.load(saved)
            self.assertEqual(restored.retention.receipt, outcomes[1][2].retention.receipt)
            saved['student_v1']['retention']['payload_sha256'] = '0'*64
            with self.assertRaisesRegex(ValueError, 'retention data changed'):
                restored.load(saved)


if __name__ == '__main__':
    unittest.main()
