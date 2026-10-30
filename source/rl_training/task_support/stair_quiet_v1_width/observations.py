"""Privileged effective ground friction, preserving the two-dimensional ABI."""
import torch
from .settings import PROFILES


def effective_friction(env, asset_cfg=None):
    table = getattr(env, "_tail_friction_table", None)
    if table is None:
        table = torch.tensor([[p.static_friction, p.dynamic_friction] for p in PROFILES], device=env.device)
        env._tail_friction_table = table
    return table[env.scene.terrain.terrain_types % len(PROFILES)]
