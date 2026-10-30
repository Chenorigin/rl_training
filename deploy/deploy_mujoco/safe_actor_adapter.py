"""Safe teacher inference and privileged MuJoCo observation bridge.

No training source is patched. Checkpoint-pinned functions define the feature
layout. Sensor profiles affect the delivered map/proprioception; safety48 is
explicitly simulator privileged, including when front_only_clean is selected.
"""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
from tensordict import TensorDict

from terrain_loader import terrain_contact_bodies

ROOT = Path(__file__).resolve().parents[2]
GROUPS = {"policy": 244, "motion_control": 3, "locomotion_scan": 187, "safety": 48}
CLASS = "task_support.stair_quiet_v1_safe.actor:SafeResidualActor"
SHARED_SOURCES = (
    "stair_quiet_v1_safe/actor.py", "stair_quiet_v1_safe/observations.py",
    "stair_quiet_v1_safe/settings.py", "stair_quiet_v1_width/geometry.py",
    "stair_quiet_v1_width/settings.py", "stair_quiet_v1/signals.py",
    "idle_stop_motion_v1/motion_residual_actor.py",
)


def load_safe_actor(state, manifest):
    """Fail closed on incomplete Safe weights, schema or shared formula drift."""
    noise = manifest.get("noise_refinement", {})
    config = manifest.get("actor_model_config", {})
    contract = noise.get("contract", {})
    digest = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    if (manifest.get("refinement_version") != "m20_stair_quiet_v1_safe.v1"
            or noise.get("task") != "stair_quiet_v1_safe" or noise.get("version") != 1
            or noise.get("contract_sha256") != digest):
        raise ValueError("Safe checkpoint manifest/contract digest mismatch")
    if (manifest.get("actor_input_groups") != GROUPS or config.get("class_name") != CLASS
            or manifest.get("motion_control_version") != "m20_raw_command_gate_20261017"
            or manifest.get("control_profile") != "legacy_v1"
            or manifest.get("m20_profile") != "idle_stop_motion_v1"):
        raise ValueError("Safe actor input groups or raw-command contract mismatch")
    if (config.get("hidden_dims") != [512, 256, 128]
            or config.get("residual_hidden_dims") != [128, 64]
            or config.get("activation") != "elu" or config.get("obs_normalization") is not False):
        raise ValueError("Unsupported Safe actor architecture")
    for relative in SHARED_SOURCES:
        key = "source/rl_training/task_support/" + relative
        if noise.get("source_sha256", {}).get(key) != hashlib.sha256((ROOT / key).read_bytes()).hexdigest():
            raise ValueError(f"Safe shared source hash mismatch: {relative}")
    if not all(torch.is_tensor(v) and torch.isfinite(v).all() for v in state.values()):
        raise ValueError("Safe actor state contains non-finite or non-tensor weights")
    from task_support.stair_quiet_v1_safe.actor import SafeResidualActor
    from task_support.stair_quiet_v1_safe.settings import SafeSettings
    settings = SafeSettings(**contract["environment"]["stair_tail"])
    settings.validate()
    observations = TensorDict({k: torch.zeros(1, n) for k, n in GROUPS.items()}, batch_size=[1])
    kwargs = deepcopy(config)
    kwargs.pop("class_name")
    actor = SafeResidualActor(observations, {"actor": ["policy"]}, "actor", 16, **kwargs)
    actor.load_state_dict(state, strict=True)
    actor.eval().requires_grad_(False)
    actor.uses_safe_control = True
    actor.m20_training_manifest, actor.safe_settings = manifest, settings
    print("[POLICY] Safe: complete base + motion residual + safety residual; "
          "policy244/raw_command3/locomotion187/safety48", flush=True)
    return actor


class SafeMujocoAdapter:
    """Current link kinematics plus contact forces from the preceding solve."""

    def __init__(self, model, contract, actor):
        from task_support.stair_quiet_v1_safe.observations import longitudinal_scan, safety_features
        self.longitudinal_scan, self.safety_features = longitudinal_scan, safety_features
        self.model, self.contract = model, contract
        self.settings = actor.safe_settings
        terrain = contract.terrain
        self.flat = terrain.kind == "flat"
        self.native = None
        if terrain.kind not in ("ascent", "flat", "custom"):
            raise ValueError("Safe geometry supports parameterized ascent/ascent_open, flat or native box/plane custom; "
                             f"unsupported terrain: {terrain.kind}")
        # A spawn mat introduces a real raised support and finite edges even
        # on a flat/parameterized course. Read those collision faces instead
        # of applying the original plane/stair-only analytic description.
        if terrain.kind == "custom" or getattr(terrain, "spawn_mat", None) is not None:
            from safe_native_geometry import NativeTerrainGeometry
            self.native = NativeTerrainGeometry(model, contract.terrain_geom_ids, contract.terrain_flex_ids)
        p = terrain.parameters
        self.width = float(terrain.course_width)
        self.origin = np.zeros(3)
        if self.flat or self.native is not None:
            self.height, self.depth, self.steps = 1., 1., 1
        else:
            self.height, self.depth = float(p["step_height"][0]), float(p["tread_depth"][0])
            self.steps = int(p["step_count"][0])
            start, platform = float(p["start_x"][0]), float(p["platform_length"][0])
            if p.get("open_risers") == [1.0]:
                self.settings = replace(self.settings, start_length=float(p["approach_length"][0]))
            self.origin[0] = start-self.settings.start_length/2
            self.settings = replace(self.settings, tile_length=(
                self.settings.start_length+self.steps*self.depth+platform))
        self.wheels = {int(body): i for i, body in enumerate(contract.wheel_body_ids)}
        # Training observes the wheel link's local +Y axis (not its COM frame).
        # Accept the existing cylinder collider but verify its size and axis.
        for body in self.wheels:
            geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == body
                     and (model.geom_contype[g] or model.geom_conaffinity[g])]
            if len(geoms) != 1:
                raise ValueError("Safe requires one physical wheel collider per wheel link")
            geom = geoms[0]
            axis = model.geom_quat[geom]
            rotation = np.empty(9)
            mujoco.mju_quat2Mat(rotation, axis)
            if (model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_CYLINDER
                    or not np.allclose(model.geom_size[geom, :2],
                                       [self.settings.wheel_radius, self.settings.wheel_half_length], atol=1e-5)
                    or not np.allclose(np.abs(rotation.reshape(3, 3)[:, 2]), [0, 1, 0], atol=1e-5)
                    or not np.allclose(model.geom_pos[geom], 0., atol=1e-5)):
                raise ValueError("Safe wheel collider size/axis/origin differs from the M20 link contract")
        self.policy_samples = 0
        self.reset()
        print("[SAFE] safety48=simulation privileged geometry/kinematics/contact; "
              "noise profiles affect policy244 and locomotion187 only", flush=True)

    def reset(self):
        self.forces = np.zeros((4, 3))
        self.force_time = None
        self.last_inputs = None
        self.last_record_time = None
        self.last_time = None

    def sample_contacts(self, data):
        forces = np.zeros((4, 3))
        for contact_id in range(data.ncon):
            contact = data.contact[contact_id]
            if contact.efc_address < 0:
                continue
            first, second = terrain_contact_bodies(self.model, contact, self.contract.terrain_flex_ids)
            if first in self.wheels and second == 0:
                wheel, sign = self.wheels[first], -1.
            elif second in self.wheels and first == 0:
                wheel, sign = self.wheels[second], 1.
            else:
                continue
            local = np.zeros(6)
            mujoco.mj_contactForce(self.model, data, contact_id, local)
            forces[wheel] += sign*(contact.frame.reshape(3, 3).T @ local[:3])
        self.forces, self.force_time = forces, float(data.time)

    def physical_sample(self, data):
        ids = self.contract.wheel_body_ids
        jacp, jacr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        velocity, omega = [], []
        for body in ids:
            mujoco.mj_jacBody(self.model, data, jacp, jacr, int(body))
            velocity.append(jacp @ data.qvel)
            omega.append(jacr @ data.qvel)
        # Isaac root_lin_vel_w aliases root COM velocity. Its wheel inputs,
        # in contrast, are explicitly converted to link-origin velocity.
        mujoco.mj_jacBodyCom(self.model, data, jacp, jacr, self.contract.base_id)
        root_velocity = jacp @ data.qvel
        rotation = data.xmat[self.contract.base_id].reshape(3, 3)
        sample = dict(position=data.xpos[ids]-self.origin,
            axis=data.xmat[ids].reshape(4, 3, 3)[:, :, 1],
            velocity=np.array(velocity[:4]), omega=np.array(omega[:4]), forces=self.forces,
            root=data.xpos[self.contract.base_id]-self.origin, root_velocity=root_velocity,
            yaw=np.arctan2(rotation[1, 0], rotation[0, 0]), upright=rotation[2, 2],
            height=self.height, depth=self.depth, width=self.width, steps=self.steps)
        return {k: torch.as_tensor(np.array(v, copy=True), dtype=(torch.long if k == "steps" else torch.float32))
                .unsqueeze(0) for k, v in sample.items()}

    def inputs(self, observation, requested_command, data, perception=None):
        raw, command = np.asarray(observation, dtype=np.float32), np.asarray(requested_command, dtype=np.float32)
        if raw.shape != (244,) or command.shape != (3,) or not np.isfinite(raw).all() or not np.isfinite(command).all():
            raise ValueError("Safe requires finite policy244 and raw requested command3")
        d = self.physical_sample(data)
        yaw = float(d["yaw"][0])
        grid = self.contract.grid
        course_y = data.xpos[self.contract.base_id, 1]+np.sin(yaw)*grid[:, 0]+np.cos(yaw)*grid[:, 1]
        scan = torch.from_numpy(raw[57:].copy()).reshape(1, 187)
        if self.native is None:
            scan = self.longitudinal_scan(scan,
                torch.as_tensor(grid[:, 0].copy(), dtype=torch.float32).reshape(1, 187),
                torch.as_tensor(course_y, dtype=torch.float32).reshape(1, 187), d["width"])
        if perception is not None:
            channels = perception.map_adapter.last_channels
            if channels is None:
                raise ValueError("Safe noisy map must be observed before policy inference")
            valid = torch.as_tensor(channels[1] > 0).reshape(1, 187)
            scan = torch.where(valid, scan, 0.)
        features = (self.native.features(data, d, self.settings) if self.native is not None
                    else self.safety_features(d, self.settings))
        if self.flat and self.native is None:
            # A plane has no longitudinal tread edge. The existing flat XML's
            # course_width defines a route corridor, not the infinite floor edge.
            feet = features[:, :44].reshape(1, 4, 11)
            feet[:, :, 0] = feet[:, :, 3] = 10.
            bottom = d["position"][..., 2]-self.settings.wheel_radius-self.settings.wheel_half_length*d["axis"][..., 2].abs()
            force = d["forces"]
            feet[:, :, 6] = ((force[..., 2] > self.settings.support_force_N)
                & (force[..., 2] > torch.linalg.vector_norm(force[..., :2], dim=-1)) & (bottom.abs() < .05)).float()
        values = {"policy": torch.from_numpy(raw.copy()).reshape(1, 244),
                  "motion_control": torch.from_numpy(command.copy()).reshape(1, 3),
                  "locomotion_scan": scan, "safety": features}
        if any(not torch.isfinite(v).all() for v in values.values()):
            raise RuntimeError("Non-finite Safe observation")
        self.last_inputs = TensorDict(values, batch_size=[1])
        self.last_time = float(data.time)
        self.force_time_at_policy = self.force_time
        self.policy_samples += 1
        return self.last_inputs

    def record(self, records):
        if self.last_inputs is None or self.last_record_time == self.last_time:
            return
        records.setdefault("safe_policy_time_s", []).append(self.last_time)
        for group, tensor in self.last_inputs.items():
            records.setdefault("safe_"+group, []).append(tensor[0].cpu().numpy().copy())
        records.setdefault("safe_contact_sample_time_s", []).append(
            np.nan if self.force_time_at_policy is None else self.force_time_at_policy)
        self.last_record_time = self.last_time

    def metadata(self):
        return dict(actor_input_groups=GROUPS, policy_samples=self.policy_samples,
            safety_observation="simulation_privileged_safety48",
            noise_scope="policy244_and_locomotion187; safety48_remains_oracle",
            contact_timing="previous_completed_physics_solve",
            wheel_features="training_capsule_envelope; physical_MJCF_cylinders_unchanged",
            geometry_source=("current_native_collision_poses" if self.native is not None
                             else "current_terrain_XML_and_CLI_overrides"),
            locomotion_scan=("delivered_map_without_lane_replacement" if self.native is not None
                             else "longitudinal_lane_replacement"),
            terrain_kind=self.contract.terrain.kind, width_m=self.width,
            height_m=None if self.flat or self.native is not None else self.height,
            depth_m=None if self.flat or self.native is not None else self.depth,
            steps=None if self.flat or self.native is not None else self.steps,
            course_origin=self.origin.tolist(),
            native_prediction="constant_rigid_body_velocity" if self.native is not None else None,
            native_envelope="conservative_world_axis_rectangle; convex_coplanar_unions" if self.native is not None else None,
            native_width_semantics="route_corridor; wheel_margins_use_actual_faces" if self.native is not None else None,
            flat_width_semantics="route_corridor" if self.flat else None)
