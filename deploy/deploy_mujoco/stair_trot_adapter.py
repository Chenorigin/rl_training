"""CPU StairTrot-v1 inference and sensor-derived control state for MuJoCo.

No Isaac imports, optimizers, curriculum labels or privileged critic inputs.
The default gate uses training height rays; an explicit noise profile supplies
delivered map samples with missing cells, without clean-ray fallback.
"""
from __future__ import annotations

from dataclasses import fields
from copy import deepcopy
import math
from pathlib import Path
import sys

import mujoco
import numpy as np
import torch
from terrain_loader import static_terrain_flex_ids, terrain_contact_bodies

SOURCE = Path(__file__).resolve().parents[2] / "source/rl_training"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from task_support.stair_trot_v1.actor import StairTrotActor
from task_support.stair_trot_v1.signals import TrotSettings, ascent_geometry, command_gate
from task_support.idle_stop_motion_v1.profile import TUNING


def load_stair_trot_actor(obs, state, manifest):
    from motion_residual_adapter import load_motion_actor

    if manifest.get("refinement_version") != "m20_stair_trot_v1":
        raise ValueError("StairTrot weights require the StairTrot-v1 manifest")
    if manifest.get("actor_input_groups") != {"policy": 244, "motion_control": 3, "trot_control": 6}:
        raise ValueError("Expected StairTrot policy244 + raw command3 + control6")
    contract = manifest.get("task_contract", {})
    if contract.get("objective") != "stable_ascent" or contract.get("trot_control_size") != 6:
        raise ValueError("Unsupported StairTrot route/observation contract")
    saved_settings = manifest.get("trot_settings", {})
    if set(saved_settings) != {field.name for field in fields(TrotSettings)}:
        raise ValueError("Missing or unknown StairTrot settings; refusing guessed defaults")
    settings = TrotSettings(**saved_settings)
    config = deepcopy(manifest.get("actor_model_config", {}))
    if config.pop("class_name", None) != "task_support.stair_trot_v1.actor:StairTrotActor":
        raise ValueError("Unsupported StairTrot actor class")
    limits = [settings.hipx_residual_limit, settings.hipy_residual_limit,
              settings.knee_residual_limit] * 4 + [settings.wheel_residual_limit] * 4
    commands = [settings.forward_command_min, settings.lateral_command_max, settings.yaw_command_max]
    if config.get("correction_limits") != limits or config.get("command_limits") != commands:
        raise ValueError("StairTrot actor limits differ from saved gate/action settings")
    if not torch.equal(state.get("trot_limits", torch.empty(0)), torch.tensor(limits)):
        raise ValueError("StairTrot action-limit buffer differs from its manifest")
    # Validate the complete source ABI and its recorded actor source hash.
    base_state = {key: value for key, value in state.items()
                  if not key.startswith("trot_mlp.") and key != "trot_limits"}
    load_motion_actor(obs, base_state, deepcopy(manifest.get("source_teacher_manifest", {})))
    observations = obs.clone()
    observations["motion_control"] = torch.zeros(1, 3)
    observations["trot_control"] = torch.zeros(1, 6)
    actor = StairTrotActor(observations, {"actor": ["policy"]}, "actor", 16, **config)
    actor.load_state_dict(state, strict=True)
    actor.eval().requires_grad_(False)
    actor.m20_training_manifest = manifest
    return actor


class TrotControlState:
    """Mirror the observation-producing part of Isaac's per-env trot context.

    ``fixed_route`` contains (origin_xy, yaw) for straight ascent courses;
    other terrain latches the current root pose at the first active riser.
    Repeated reads at one physical time never advance phase or gate twice.
    """
    def __init__(self, settings, dt, fixed_route=None):
        self.cfg, self.dt, self.fixed_route = settings, dt, fixed_route
        self.reset()

    def reset(self):
        self.phase = 0.0
        self.gate = 0.0
        self.route = None
        self.last_time = None
        self.last_control = np.zeros(6, dtype=np.float32)

    def observe(self, hits, ray_xy, root_xy, yaw, wheels, forces, air_time, command, now):
        raw = torch.as_tensor(np.asarray(command, dtype=np.float32)).reshape(1, 3)
        if not torch.isfinite(raw).all():
            raise ValueError("StairTrot needs a finite raw command")
        if self.last_time is not None and now < self.last_time:
            self.reset()
        if now != self.last_time:
            cfg = self.cfg
            hits = torch.as_tensor(hits, dtype=torch.float32)
            rays = torch.as_tensor(ray_xy, dtype=torch.float32)
            wheels = torch.as_tensor(wheels, dtype=torch.float32)
            force = torch.as_tensor(forces, dtype=torch.float32)
            valid = torch.isfinite(hits).all(-1)
            distance = torch.linalg.vector_norm(wheels[:, None, :2] - torch.nan_to_num(hits)[None, :, :2], dim=-1)
            distance[:, ~valid] = float("inf")
            nearest, indices = distance.min(-1)
            ground_valid = nearest < cfg.ground_distance_max
            ground = torch.nan_to_num(hits[:, 2])[indices]
            radius = TUNING["wheel_radius"]
            clearance = torch.where(ground_valid, wheels[:, 2] - ground - radius, 0.0)
            contact = (force[:, 2] > cfg.contact_force) & (force[:, 2] > .5 * torch.linalg.vector_norm(force, dim=-1))
            air = (~contact & ground_valid & (clearance > cfg.clearance_min)
                   & (torch.as_tensor(air_time) >= cfg.air_time_min))
            raw_gate = float(ascent_geometry(hits[None, :, 2], rays, raw, cfg)[0])
            active = raw_gate > 0 and bool(ground_valid.all())
            entering = active and self.gate == 0
            frequency = float(np.clip(abs(float(raw[0, 0])) / cfg.nominal_stride,
                                      cfg.frequency_min, cfg.frequency_max))
            self.phase = (.5 if bool(air[1] & air[2]) else 0.0) if entering else (
                self.phase + self.dt * frequency * active) % 1.0
            self.gate = min(raw_gate, self.gate + self.dt / cfg.gate_rise_time) if active else 0.0
            if active and self.route is None:
                self.route = self.fixed_route if self.fixed_route is not None else (np.array(root_xy).copy(), yaw)
            cross_track = heading_error = 0.0
            if self.route is not None:
                origin, heading = self.route
                delta = np.asarray(root_xy) - origin
                cross_track = -delta[0] * math.sin(heading) + delta[1] * math.cos(heading)
                heading_error = math.atan2(math.sin(yaw-heading), math.cos(yaw-heading))
            self.last_control = np.array([
                self.gate, math.sin(2*math.pi*self.phase), math.cos(2*math.pi*self.phase),
                np.clip(cross_track, -2, 2), math.sin(heading_error), math.cos(heading_error),
            ], dtype=np.float32)
            self.last_time = now
        control = self.last_control.copy()
        # Isaac resamples commands between reward and observation: recheck now.
        control[0] *= float(command_gate(raw, self.cfg)[0])
        if control[0] <= 0:
            control.fill(0)
        return control


class StairTrotMujocoController:
    """Map solved terrain contact normals and ray hits to the six input values."""
    def __init__(self, contract, actor, dt):
        self.contract = contract
        self.settings = TrotSettings(**actor.m20_training_manifest["trot_settings"])
        self.flex_ids = getattr(contract, "terrain_flex_ids", static_terrain_flex_ids(contract.model))
        terrain = contract.terrain
        # Parameterized ascent stairs run along world +X, centred at world Y=0.
        # spawn_yaw/Y are initial approach errors, not a new desired route.
        route = (np.array([terrain.spawn_xy[0], 0.0]), 0.0) if terrain.kind == "ascent" else None
        self.state = TrotControlState(self.settings, dt, route)
        self.reset()

    def reset(self):
        self.state.reset()
        self.air_time = np.zeros(4, dtype=np.float32)
        self.forces = np.zeros((4, 3), dtype=np.float32)
        self.contact_time = None
        self.active_samples = self.samples = 0

    def sample_contacts(self, data):
        if self.contact_time is not None and data.time < self.contact_time:
            self.reset()
        if data.time == self.contact_time:
            return
        model = self.contract.model
        self.forces.fill(0)
        for i in range(data.ncon):
            contact = data.contact[i]
            b1, b2 = terrain_contact_bodies(model, contact, self.flex_ids)
            for leg, wheel in enumerate(self.contract.wheel_body_ids):
                if not ((b1 == wheel and b2 == 0) or (b2 == wheel and b1 == 0)):
                    continue
                wrench = np.zeros(6)
                mujoco.mj_contactForce(model, data, i, wrench)
                normal = np.asarray(contact.frame).reshape(3, 3)[0]
                self.forces[leg] += (1 if b2 == wheel else -1) * max(wrench[0], 0) * normal
        elapsed = 0.0 if self.contact_time is None else float(data.time-self.contact_time)
        # Isaac ContactSensorCfg.force_threshold default is 1 N, distinct from
        # the stronger upward-bearing threshold used to qualify gait support.
        touching = np.linalg.norm(self.forces, axis=-1) > 1.0
        self.air_time = np.where(touching, 0.0, self.air_time + elapsed).astype(np.float32)
        self.contact_time = float(data.time)

    def control(self, data, command, *, terrain_points=None):
        self.sample_contacts(data)
        contract = self.contract
        rotation = data.xmat[contract.base_id].reshape(3, 3)
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
        previous_time = self.state.last_time
        hits = contract.last_scan_hits if terrain_points is None else terrain_points
        values = self.state.observe(hits, contract.grid,
                                    data.xpos[contract.base_id, :2], yaw,
                                    data.xpos[contract.wheel_body_ids], self.forces,
                                    self.air_time, command, float(data.time))
        if previous_time != data.time:
            self.samples += 1
            self.active_samples += int(values[0] > 0)
        return values

    def summary(self):
        return {"input_groups": {"policy": 244, "motion_control": 3, "trot_control": 6},
                "control_samples": self.samples, "active_samples": self.active_samples,
                "acceptance": "interface/physics rollout only; gait performance requires separate evaluation"}


def make_trot_controller(contract, actor, dt):
    return StairTrotMujocoController(contract, actor, dt) if getattr(actor, "uses_trot_control", False) else None
