"""Output budgeting and atomic checkpoints; never deletes existing run data."""

from dataclasses import dataclass
import math
import os
from pathlib import Path
import shutil
import tempfile


GIB = 1024**3


@dataclass(frozen=True)
class OutputStorageCfg:
    min_free_disk_gb: float = 2.0
    checkpoint_margin: float = 1.15
    event_bytes_per_iteration: int = 64*1024
    snapshot_and_metadata_bytes: int = 128*1024**2


def output_budget(cfg, num_envs, iterations, start_update=0, storage_cfg=None):
    """Conservative float32 MLP/Adam/replay serialization estimate, not a quota."""
    storage_cfg = storage_cfg or OutputStorageCfg()
    if num_envs < 1 or iterations < 1 or start_update < 0 or cfg["save_interval"] < 1:
        raise ValueError("Output budget requires positive sizes and save_interval")

    def parameters(input_dim, model):
        sizes = [input_dim, *model["hidden_dims"], 16]
        return sum((a+1)*b for a,b in zip(sizes,sizes[1:]))

    replay = num_envs*cfg["num_steps_per_env"]*cfg["algorithm"]["replay_rollouts"]*(618+16)*4
    # Student weights + two Adam moments, teacher weights, Gaussian stds, header.
    serialized = (3*parameters(618,cfg["student"])+parameters(244,cfg["teacher"])+32)*4 + replay + 64*1024
    checkpoint = math.ceil(serialized*storage_cfg.checkpoint_margin)
    end = start_update+iterations
    interval = cfg["save_interval"]
    count = (end-1)//interval-(start_update-1)//interval
    count += int((end-1)%interval != 0)
    reserve = math.ceil(storage_cfg.min_free_disk_gb*GIB)
    # One atomic temporary file and one early-stop recovery checkpoint.
    run_bytes = (count+2)*checkpoint + iterations*storage_cfg.event_bytes_per_iteration + storage_cfg.snapshot_and_metadata_bytes
    return {"checkpoint_estimate_bytes":checkpoint, "checkpoint_count":count,
            "run_output_estimate_bytes":run_bytes, "min_free_disk_bytes":reserve,
            "required_free_bytes":run_bytes+reserve,
            "basis":"float32 MLP weights + Adam moments + full replay, scheduled/final/recovery checkpoints, TensorBoard/code allowance; engineering estimate"}


class LowDiskSpace(RuntimeError):
    pass


def require_disk_space(directory, required_bytes):
    path = Path(directory).expanduser().resolve()
    while not path.exists():
        path = path.parent
    free = shutil.disk_usage(path).free
    if free < required_bytes:
        raise LowDiskSpace(f"Output disk {path}: free {free/GIB:.2f} GiB, required {required_bytes/GIB:.2f} GiB; free space or use --log-root on a verified writable disk")
    return free


def atomic_torch_save(state, target, save_fn=None):
    """A failed write leaves the previous checkpoint untouched."""
    if save_fn is None:
        import torch
        save_fn = torch.save
    target = Path(target)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.",suffix=".tmp",dir=target.parent)
    try:
        with os.fdopen(descriptor,"wb") as stream:
            save_fn(state,stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary,target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
