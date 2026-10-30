"""Pinned, balanced rehearsal of real non-forward rollouts from a prior student.

This is training data only. It adds no deployed inputs, controller or policy
branch. One complete collection seed is held out from optimizer batches.
"""
import json
from pathlib import Path

import numpy as np
import torch

from .model_io import file_hash

ROOT = Path(__file__).resolve().parents[4]
CLASSES = ("backward", "left", "right", "yaw_left", "yaw_right")


def checked_manifest(path, expected_sha256):
    path = Path(path).expanduser()
    path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_file() or file_hash(path) != expected_sha256:
        raise ValueError("Motion retention manifest missing or hash changed")
    meta = json.loads(path.read_text())
    if (meta.get("schema") != "m20_nonforward_retention_v1" or meta.get("obs_dim") != 789
            or meta.get("action_dim") != 16 or meta.get("classes") != list(CLASSES)
            or type(meta.get("sample_count")) is not int or meta["sample_count"] < 10):
        raise ValueError("Invalid motion retention manifest/ABI")
    payload = (path.parent / meta["payload"]).resolve()
    if not payload.is_relative_to(path.parent) or not payload.is_file():
        raise ValueError("Invalid motion retention payload path")
    if file_hash(payload) != meta.get("payload_sha256"):
        raise ValueError("Motion retention payload hash changed")
    return path, payload, meta


def retention_bytes(path, sha256):
    _, _, meta = checked_manifest(path, sha256)
    # Obs/action float32, class int64, holdout bool and split index int64.
    return meta["sample_count"] * ((789 + 16) * 4 + 8 + 1 + 8)


class MotionRetention:
    def __init__(self, path, sha256, device):
        manifest, payload, meta = checked_manifest(path, sha256)
        with np.load(payload, allow_pickle=False) as data:
            if set(data.files) != {"observations", "labels", "classes", "holdout"}:
                raise ValueError("Invalid motion retention fields")
            arrays = {k: data[k].copy() for k in data.files}
        n = meta["sample_count"]
        for key, shape, dtype in (("observations", (n, 789), np.float32), ("labels", (n, 16), np.float32),
                                   ("classes", (n,), np.int64), ("holdout", (n,), np.bool_)):
            if arrays[key].shape != shape or arrays[key].dtype != dtype or not np.isfinite(arrays[key]).all():
                raise ValueError(f"Invalid motion retention {key}")
        classes = arrays["classes"]
        if not ((classes >= 0) & (classes < len(CLASSES))).all():
            raise ValueError("Invalid retention command class")
        self.obs = torch.from_numpy(arrays["observations"]).to(device)
        self.labels = torch.from_numpy(arrays["labels"]).to(device)
        self.classes = torch.from_numpy(classes).to(device)
        holdout = torch.from_numpy(arrays["holdout"]).to(device)
        self.pools = {}
        for split, mask in (("train", ~holdout), ("heldout", holdout)):
            pools = [(mask & (self.classes == k)).nonzero().flatten() for k in range(len(CLASSES))]
            if any(not p.numel() for p in pools):
                raise ValueError("Every retention command requires independent train and heldout samples")
            self.pools[split] = pools
        self.receipt = {"manifest": str(manifest), "manifest_sha256": sha256,
                        "payload_sha256": meta["payload_sha256"], "source_checkpoint_sha256": meta["source_checkpoint_sha256"],
                        "samples": n, "training_counts": [int(p.numel()) for p in self.pools["train"]],
                        "heldout_counts": [int(p.numel()) for p in self.pools["heldout"]]}

    def sample(self, count, *, heldout=False):
        if type(count) is not int or count < len(CLASSES):
            raise ValueError("Retention batch must cover all five commands")
        chunks = []
        for k, pool in enumerate(self.pools["heldout" if heldout else "train"]):
            size = count // len(CLASSES) + int(k < count % len(CLASSES))
            chunks.append(pool[torch.randint(pool.numel(), (size,), device=pool.device)])
        ids = torch.cat(chunks)
        return self.obs[ids], self.labels[ids], self.classes[ids]
