"""V1 tensor and output budget, including history, tags and critical replay."""
import math

from m20_student_v0.output_storage import OutputStorageCfg
from .contract import student_dim


def tensor_budget(cfg, num_envs, perception):
    dim = student_dim(cfg.get("history_steps", 1))
    rollout = num_envs*cfg["num_steps_per_env"]
    algorithm = cfg["algorithm"]
    regular = rollout*algorithm["replay_rollouts"]
    critical = rollout*algorithm["hard_replay_rollouts"]
    replay_bytes = (regular+critical)*((dim+16)*4+4*8+1)
    rollout_bytes = rollout*((dim*2+244+32+1)*4+3)
    minimum_period = max(.02, perception.map_update_period_s-perception.update_jitter_s)
    slots = math.ceil((perception.max_map_age_s+perception.max_delay_s)/minimum_period)+3
    map_bytes = slots*num_envs*(187*9+3*4+2*8)
    return {"replay_bytes": replay_bytes, "rollout_bytes": rollout_bytes, "map_bytes": map_bytes,
            "sample_buffer_bytes": replay_bytes+rollout_bytes+map_bytes,
            "sample_buffer_formula": "(regular+critical)*((actor_dim+16)*4+tag4*8+holdout1) + rollout + map packets"}


def output_budget(cfg, num_envs, iterations, perception, start_update=0, min_free_disk_gb=2.):
    settings = OutputStorageCfg(min_free_disk_gb=min_free_disk_gb)
    if iterations < 1 or cfg["save_interval"] < 1 or num_envs < 1:
        raise ValueError("Invalid run budget")
    def parameters(dim, model):
        sizes = [dim, *model["hidden_dims"], 16]
        return sum((a+1)*b for a, b in zip(sizes, sizes[1:]))
    budget = tensor_budget(cfg, num_envs, perception)
    serialized = 4*(3*parameters(student_dim(cfg.get("history_steps", 1)), cfg["student"])+parameters(244, cfg["teacher"])+32)
    save_replay = cfg["algorithm"].get("save_replay", True)
    if not isinstance(save_replay, bool):
        raise ValueError("save_replay must be a boolean")
    checkpoint_replay = budget["replay_bytes"] if save_replay else 0
    checkpoint = math.ceil((serialized+checkpoint_replay+128*1024)*settings.checkpoint_margin)
    end = start_update+iterations
    interval = cfg["save_interval"]
    count = (end-1)//interval-(start_update-1)//interval + int((end-1)%interval != 0)
    run_bytes = (count+2)*checkpoint+iterations*settings.event_bytes_per_iteration+settings.snapshot_and_metadata_bytes
    reserve = math.ceil(min_free_disk_gb*1024**3)
    return {**budget, "checkpoint_estimate_bytes": checkpoint, "checkpoint_count": count,
            "checkpoint_replay_bytes": checkpoint_replay, "save_replay": save_replay,
            "run_output_estimate_bytes": run_bytes, "min_free_disk_bytes": reserve,
            "required_free_bytes": run_bytes+reserve,
            "basis": "MLP weights, Adam moments, optional replay serialization, atomic/recovery checkpoints, event/code allowance; live replay always included in memory budget"}
