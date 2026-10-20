"""M20 student observation contract and actor initialization helpers.

This first scaffold samples the simulation ray scan. It is NOT a LiDAR/map
simulator: world-map reprojection, age, occlusion and source calibration remain
future work. Policy and critic share one atomic terrain snapshot per env step.
"""

from dataclasses import dataclass
from collections.abc import Mapping

import torch

PROPRIO_DIM = 57
HEIGHT_DIM = 187
TEACHER_OBS_DIM = PROPRIO_DIM + HEIGHT_DIM
STUDENT_OBS_DIM = PROPRIO_DIM + 3 * HEIGHT_DIM
ACTION_DIM = 16
OBS_SLICES = {"proprio": slice(0, 57), "height": slice(57, 244),
              "validity": slice(244, 431), "confidence": slice(431, 618)}


@dataclass
class StairStudentPerceptionCfg:
    """Baseline corruption only; confidence is a synthetic source score.

    Zero defaults give the clean initialization baseline. A nonzero std uses
    the configured source variance, never error against privileged truth, to
    construct confidence. Do not interpret this score as a calibrated probability.
    """

    height_offset: float = 0.5
    height_noise_std: float = 0.0
    dropout_probability: float = 0.0
    confidence_std_reference: float = 0.05

    def validate(self):
        if not 0 <= self.dropout_probability <= 1:
            raise ValueError("dropout_probability must be in [0, 1]")
        if not 0 <= self.height_noise_std < float("inf"):
            raise ValueError("height_noise_std must be finite and nonnegative")
        if not 0 < self.confidence_std_reference < float("inf"):
            raise ValueError("confidence_std_reference must be positive and finite")
        if not abs(self.height_offset) < float("inf"):
            raise ValueError("height_offset must be finite")


def build_terrain_channels(ray_hits_w, base_pos_w, cfg):
    """Return height/usable-validity/reliability, each [N,187], in ray order.

    Grid ABI: x=-.8..+.8, y=-.5..+.5, .1m, y rows/x columns ascending.
    Invalid heights are finite zero placeholders; validity distinguishes unknown
    from a real zero. No height*confidence multiplication is performed.
    """
    cfg.validate()
    if ray_hits_w.ndim != 3 or tuple(ray_hits_w.shape[1:]) != (HEIGHT_DIM, 3):
        raise ValueError(f"Expected [N,187,3] hits, received {tuple(ray_hits_w.shape)}")
    if tuple(base_pos_w.shape) != (ray_hits_w.shape[0], 3):
        raise ValueError("Base pose and scan batch dimensions differ")
    valid = torch.isfinite(ray_hits_w).all(-1) & torch.isfinite(base_pos_w).all(-1)[:, None]
    height = base_pos_w[:, 2, None] - ray_hits_w[..., 2] - cfg.height_offset
    height = torch.where(valid, height, torch.zeros_like(height))
    if cfg.height_noise_std:
        height = height + torch.randn_like(height) * cfg.height_noise_std
    if cfg.dropout_probability:
        valid = valid & (torch.rand_like(height) >= cfg.dropout_probability)
    height = torch.where(valid, height.clamp(-1, 1), torch.zeros_like(height))
    validity = valid.to(height.dtype)
    source_score = torch.exp(height.new_tensor(
        -(cfg.height_noise_std / cfg.confidence_std_reference) ** 2))
    confidence = validity * source_score
    return height, validity, confidence


def _terrain_snapshot(env, sensor_cfg):
    # Episode identity catches reset followed by observation at the same global
    # step. Returning clones below keeps observation transforms out of the cache.
    cache = getattr(env, "_m20_student_terrain_cache", None)
    step = env.common_step_counter
    episodes = env.episode_length_buf
    if cache is None or cache["step"] != step or not torch.equal(cache["episodes"], episodes):
        sensor = env.scene.sensors[sensor_cfg.name]
        channels = build_terrain_channels(sensor.data.ray_hits_w, sensor.data.pos_w,
                                          env.cfg.student_perception)
        cache = {"step": step, "episodes": episodes.clone(), "channels": channels}
        env._m20_student_terrain_cache = cache
    return cache["channels"]


def student_height_scan(env, sensor_cfg):
    return _terrain_snapshot(env, sensor_cfg)[0].clone()


def student_height_validity(env, sensor_cfg):
    return _terrain_snapshot(env, sensor_cfg)[1].clone()


def student_height_confidence(env, sensor_cfg):
    return _terrain_snapshot(env, sensor_cfg)[2].clone()


def reset_student_perception(env, env_ids):
    """Invalidate even an explicit repeated reset at the same global step."""
    env._m20_student_terrain_cache = None


def initialize_student_actor(student_actor, teacher_actor_state: Mapping):
    """Expand a matching 244D MLP actor to 618D with zero new columns.

    Call from the future student training entrypoint, NOT train.py's existing
    --init_actor_from (which intentionally performs a same-shape strict load).
    Critic and optimizer are never loaded by this helper. Observation
    normalization must be disabled, as in this project's teacher and student.
    """
    target = student_actor.state_dict()
    key = "mlp.0.weight"
    source = teacher_actor_state[key]
    if source.shape != (512, TEACHER_OBS_DIM) or target[key].shape != (512, STUDENT_OBS_DIM):
        raise ValueError("Expected teacher 244D and student 618D, both with 512 hidden units")
    if set(target) != set(teacher_actor_state):
        raise ValueError("Actor state keys differ; normalization/architecture may not match")
    expanded = {name: value.detach().clone() for name, value in teacher_actor_state.items()}
    expanded[key] = source.new_zeros(target[key].shape)
    expanded[key][:, :TEACHER_OBS_DIM] = source
    student_actor.load_state_dict(expanded, strict=True)


def action_distillation_loss(student_actions, teacher_actions):
    """Deterministic normalized 16D actions; teacher labels receive no gradient."""
    if student_actions.shape != teacher_actions.shape or student_actions.shape[-1] != ACTION_DIM:
        raise ValueError("Distillation requires matching 16D action tensors")
    return (student_actions - teacher_actions.detach()).square().mean()
