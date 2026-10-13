"""Strict weights-only warm start and actor-only export without simulator startup."""
from copy import deepcopy
import hashlib
from pathlib import Path
import re

import torch


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_checkpoint(path):
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        candidates = [(int(match.group(1)), item) for item in path.glob("model_*.pt")
                      if (match := re.fullmatch(r"model_(\d+)\.pt", item.name))]
        if not candidates:
            raise ValueError(f"No complete numbered checkpoints in {path}")
        path = max(candidates)[1]
    if not path.is_file():
        raise ValueError(f"Missing checkpoint: {path}")
    return path


def initialize_student(student, saved):
    """New optimizer/replay; history and quality columns zero when absent at source."""
    source = saved.get("student_state_dict", saved.get("actor_state_dict"))
    if source is None:
        raise ValueError("Checkpoint has no student/actor weights")
    source = deepcopy(source)
    first = source["mlp.0.weight"]
    target = student.state_dict()
    if first.shape[1] not in (244, 618, student.obs_dim):
        raise ValueError("Unsupported source observation dimension")
    if first.shape[0] != target["mlp.0.weight"].shape[0]:
        raise ValueError("Source/student architecture mismatch")
    if first.shape[1] != student.obs_dim:
        expanded = torch.zeros_like(target["mlp.0.weight"])
        expanded[:, :first.shape[1]] = first.to(expanded.device)
        source["mlp.0.weight"] = expanded
    student.load_state_dict(source, strict=True)
    return {"source_input_dim": first.shape[1], "target_input_dim": student.obs_dim,
            "optimizer_loaded": False, "replay_loaded": False}


def actor_from_state(state, input_dim, cfg):
    if cfg.get("obs_normalization") is not False or cfg.get("activation") != "elu":
        raise ValueError("Expected an unnormalized ELU student")
    sizes = [input_dim, *cfg["hidden_dims"], 16]
    modules = []
    for index, (left, right) in enumerate(zip(sizes, sizes[1:])):
        modules.append(torch.nn.Linear(left, right))
        if index < len(sizes)-2:
            modules.append(torch.nn.ELU())
    actor = torch.nn.Sequential(*modules)
    unsupported = [key for key in state if not key.startswith("mlp.") and key != "distribution.log_std_param"]
    if unsupported:
        raise ValueError(f"Unsupported student state keys: {unsupported}")
    actor.load_state_dict({key.removeprefix("mlp."): value for key, value in state.items() if key.startswith("mlp.")}, strict=True)
    return actor.eval().requires_grad_(False)
