#!/usr/bin/env python3
"""MuJoCo sim-to-sim deployment of the M20 stair teacher actor.

With --viewer, the robot starts prone: Z stands, C enters the teacher policy,
W/S/A/D/Q/E set velocity commands, H toggles the height scan, X lies down,
R damps, and Esc quits.
Without --viewer, the script runs the original fixed-command batch rollout.
The checkpoint was trained with asymmetric PPO: actor=57D proprioception +
187D ground height scan, critic=actor information plus privileged signals.
Only actor_state_dict is loaded here. This script reproduces the actor input
and the saved action contract; it never reads critic-only observations.
idle_stop_motion_v1 checkpoints additionally receive their original raw command
in a separate 3D motion_control group and use their recorded legacy yaw feedback.
StairTrot-v1 additionally computes its six-dimensional ascent/phase/route input
from height rays and physical wheel contacts, and loads the complete new actor.

The observation/action constants below come from the checkpoint run's
params/env.yaml (2026-09-29_19-34-19), not from mutable current config.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import select
import sys
import termios
import threading
import time
from collections import deque
from pathlib import Path

import mujoco
import numpy as np
import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from keyboard_input import X11WindowKeys
from reset_placement import place_on_terrain
from spawn_mat import add_spawn_arguments, add_spawn_mat, validate_spawn_mat
from terrain_loader import (PRESETS, STAIR_OVERRIDES, Terrain, attach_terrain, select_terrain,
                            static_terrain_flex_ids, terrain_ray, is_terrain_geom)
from motion_residual_adapter import heading_controller, load_motion_actor, policy_inputs
from stair_trot_adapter import load_stair_trot_actor, make_trot_controller
from teacher_noise_adapter import add_noise_arguments, validate_noise_arguments, make_noise_adapter, noise_metadata


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "logs/rsl_rl/deeprobotics_m20_stair_teacher/2026-09-29_19-34-19/model_39300.pt"
)
JOINT_NAMES = (
    "fl_hipx_joint", "fl_hipy_joint", "fl_knee_joint",
    "fr_hipx_joint", "fr_hipy_joint", "fr_knee_joint",
    "hl_hipx_joint", "hl_hipy_joint", "hl_knee_joint",
    "hr_hipx_joint", "hr_hipy_joint", "hr_knee_joint",
    "fl_wheel_joint", "fr_wheel_joint", "hl_wheel_joint", "hr_wheel_joint",
)
DEFAULT_JOINT_POS = np.array(
    [0.0, -0.3, 0.6, 0.0, -0.3, 0.6, 0.0, 0.3, -0.6, 0.0, 0.3, -0.6, 0.0, 0.0, 0.0, 0.0],
    dtype=np.float64,
)
ACTION_SCALE = np.array([0.125, 0.25, 0.25] * 4 + [5.0] * 4, dtype=np.float64)
LEG_EFFORT_LIMIT = 76.4
WHEEL_EFFORT_LIMIT = 21.6
WHEEL_ARMATURE = 0.00243216
POLICY_DT = 0.02
PHYSICS_DT = 0.005
SCAN_OFFSET = 0.5
SCAN_HEIGHT = 20.0
WHEEL_RADIUS = 0.09
ACTOR_OBS_DIM = 244
ACTION_DIM = 16
PRONE_JOINT_POS = np.array(
    [-0.438, -1.16, 2.76, 0.438, -1.16, 2.76,
     -0.438, 1.16, -2.76, 0.438, 1.16, -2.76, 0.0, 0.0, 0.0, 0.0],
    dtype=np.float64,
)
PRE_STAND_JOINT_POS = np.array(
    [0.0, -0.6, 1.0, 0.0, -0.6, 1.0,
     0.0, 0.6, -1.0, 0.0, 0.6, -1.0, 0.0, 0.0, 0.0, 0.0],
    dtype=np.float64,
)
STAND_PHASE_DURATION = 2.5
TELEOP_KEY_TIMEOUT = 0.5
TELEOP_VELOCITIES = {"w": (0.7, 0.0, 0.0), "s": (-0.7, 0.0, 0.0),
                     "a": (0.0, 0.5, 0.0), "d": (0.0, -0.5, 0.0),
                     "q": (0.0, 0.0, 0.6), "e": (0.0, 0.0, -0.6)}
TELEOP_LIMITS = np.max(np.abs(list(TELEOP_VELOCITIES.values())), axis=0)
TELEOP_RISE_TIME = 1.0  # Seconds from zero to the requested speed.
TELEOP_STOP_TIME = 0.3  # Normal release; damping/reset/focus loss stop immediately.


def scan_grid() -> np.ndarray:
    """Match Isaac Lab GridPatternCfg(size=(1.6, 1.0), resolution=0.1, ordering='xy')."""
    x = np.arange(-0.8, 0.8 + 1.0e-9, 0.1, dtype=np.float64)
    y = np.arange(-0.5, 0.5 + 1.0e-9, 0.1, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(x, y, indexing="xy")
    points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    if points.shape != (187, 2):
        raise RuntimeError(f"Unexpected height grid shape: {points.shape}")
    return points


def load_actor(checkpoint: Path, *, allow_execution_control=False) -> MLPModel:
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    sys.path.insert(0, str(REPO_ROOT / "source/rl_training"))
    from task_support.stair_quiet_v1.deployment import execution_contract, reject_unsupported_quiet_checkpoint
    if allow_execution_control is True:
        execution_contract(saved)
    else:
        reject_unsupported_quiet_checkpoint(saved, "Legacy teacher MuJoCo runtime")
    state = saved.get("actor_state_dict")
    if not isinstance(state, dict):
        raise ValueError("Checkpoint does not contain actor_state_dict")
    first = state.get("mlp.0.weight")
    last = state.get("mlp.6.weight")
    if first is None or last is None or tuple(first.shape) != (512, ACTOR_OBS_DIM) or tuple(last.shape) != (ACTION_DIM, 128):
        raise ValueError("Checkpoint actor architecture does not match this 244D -> 16D contract")
    obs = TensorDict({"policy": torch.zeros((1, ACTOR_OBS_DIM), dtype=torch.float32)}, batch_size=[1])
    manifest = saved.get("m20_training_manifest") or {}
    if manifest.get("noise_refinement", {}).get("task") == "stair_quiet_v1_transition":
        from transition_actor_adapter import load_transition_actor
        return load_transition_actor(state, manifest)
    if (any(key.startswith("safety_mlp.") for key in state)
            or manifest.get("noise_refinement", {}).get("task") == "stair_quiet_v1_safe"):
        from safe_actor_adapter import load_safe_actor
        return load_safe_actor(state, manifest)
    if any(key.startswith("trot_mlp.") for key in state) or manifest.get("refinement_version") == "m20_stair_trot_v1":
        actor = load_stair_trot_actor(obs, state, manifest)
        print("[POLICY] StairTrot-v1: complete teacher + ascent residual; policy244/raw_command3/trot_control6", flush=True)
        return actor
    if any(key.startswith("residual_mlp.") for key in state) or manifest.get("m20_profile") == "idle_stop_motion_v1":
        return load_motion_actor(obs, state, manifest)
    actor = MLPModel(
        obs,
        {"actor": ["policy"]},
        "actor",
        ACTION_DIM,
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "log"},
    )
    actor.load_state_dict(state, strict=True)
    actor.eval()
    return actor


def load_model(model_path: Path, args: argparse.Namespace) -> mujoco.MjModel:
    model_path = model_path.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"MuJoCo M20 model not found: {model_path}")
    spec = mujoco.MjSpec.from_file(str(model_path))
    spec.option.timestep = PHYSICS_DT
    args.terrain_config = attach_terrain(spec, select_terrain(args),
        {name: getattr(args, name, None) for name in (*STAIR_OVERRIDES, "spawn_xy")})
    add_spawn_mat(spec, args.terrain_config, args)
    model = spec.compile()
    validate_spawn_mat(model, args.terrain_config)
    if model.nu != ACTION_DIM:
        raise ValueError(f"Expected {ACTION_DIM} M20 motor actuators, got {model.nu}")
    # The training actuator adds wheel-side armature absent from this MJCF.
    for name in JOINT_NAMES[12:]:
        model.dof_armature[model.joint(name).dofadr[0]] = WHEEL_ARMATURE
    return model


class M20Contract:
    """Resolve each Isaac joint by name, independent of MuJoCo XML storage order."""

    def __init__(self, model: mujoco.MjModel, terrain: Terrain):
        self.model = model
        self.terrain = terrain
        self.base_id = model.body("base_link").id
        self.root_joint = model.joint("floating_base")
        if self.root_joint.type != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("M20 root joint must be a free joint")
        self.qpos_ids = np.array([model.joint(name).qposadr[0] for name in JOINT_NAMES], dtype=np.int32)
        self.qvel_ids = np.array([model.joint(name).dofadr[0] for name in JOINT_NAMES], dtype=np.int32)
        self.actuator_ids = np.array([model.actuator(name).id for name in JOINT_NAMES], dtype=np.int32)
        self.wheel_body_ids = np.array(
            [model.body(name).id for name in ("fl_wheel", "fr_wheel", "hl_wheel", "hr_wheel")],
            dtype=np.int32,
        )
        if len(set(self.qpos_ids.tolist())) != ACTION_DIM or len(set(self.actuator_ids.tolist())) != ACTION_DIM:
            raise ValueError("Duplicate or missing M20 joint/actuator mapping")
        self.grid = scan_grid()
        self.last_scan_hits = np.empty((len(self.grid), 3), dtype=np.float64)
        self.terrain_group = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
        self.terrain_flex_ids = static_terrain_flex_ids(model)
        terrain_geoms = [
            i for i in range(model.ngeom)
            if is_terrain_geom(model, i)
        ]
        if not terrain_geoms:
            raise ValueError("Model has no group-0 collision terrain geometry for the height scan")
        self.terrain_geom_ids = frozenset(terrain_geoms)

    def reset(self, data: mujoco.MjData, prone: bool = False) -> None:
        mujoco.mj_resetData(self.model, data)
        mujoco.mj_forward(self.model, data)
        x, y = self.terrain.spawn_xy
        yaw = self.terrain.spawn_yaw
        root_qpos = int(self.root_joint.qposadr[0])
        data.qpos[root_qpos : root_qpos + 7] = [x, y, 0., math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        data.qpos[self.qpos_ids] = PRONE_JOINT_POS if prone else DEFAULT_JOINT_POS
        data.qvel[:] = 0.0
        data.ctrl[:] = 0.0
        place_on_terrain(self.model, data, root_body_id=self.base_id,
                         root_qpos_adr=root_qpos, terrain_geom_ids=self.terrain_geom_ids,
                         terrain_flex_ids=self.terrain_flex_ids)

    def ground_height(self, data: mujoco.MjData, x: float, y: float) -> float:
        """Query actual collision terrain, including ramps and native XML geoms."""
        origin = np.array([x, y, self.terrain.ray_top], dtype=np.float64)
        distance, geom_id, flex_id = terrain_ray(
            self.model, data, origin, np.array([0., 0., -1.]), self.terrain_group, self.terrain_flex_ids)
        if distance < 0 or (flex_id < 0 and geom_id not in self.terrain_geom_ids):
            raise RuntimeError(f"No collision terrain at {(x, y)}")
        return float(origin[2] - distance)

    def crossing_summary(self, data: mujoco.MjData, fell: bool) -> dict:
        """Geometric end-line crossing, not proof of traversing every obstacle."""
        wheels = data.xpos[self.wheel_body_ids]
        crossed = bool(np.all(wheels[:, 0] >= self.terrain.goal_x + WHEEL_RADIUS))
        inside = bool(np.all(np.abs(wheels[:, 1]) <= self.terrain.course_width / 2))
        cleared = crossed and inside and not fell
        return {"crossed_finish": crossed, "inside_course": inside,
                "cleared_terrain": cleared,
                "cleared_stairs": cleared if self.terrain.kind in {"ascent", "descent"} else False,
                "clearance_criterion": "all wheels beyond +X goal, inside width at final frame, no fall; bypass not checked"}

    def height_scan(self, data: mujoco.MjData) -> np.ndarray:
        """Raycast current terrain poses using Isaac's yaw-aligned grid."""
        rotation = data.xmat[self.base_id].reshape(3, 3)
        base_pos = data.xpos[self.base_id]
        ray_z = max(base_pos[2] + SCAN_HEIGHT, self.terrain.ray_top)
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
        world_xy = np.empty_like(self.grid)
        world_xy[:, 0] = base_pos[0] + cos_yaw * self.grid[:, 0] - sin_yaw * self.grid[:, 1]
        world_xy[:, 1] = base_pos[1] + sin_yaw * self.grid[:, 0] + cos_yaw * self.grid[:, 1]
        ray_origin = np.array([0.0, 0.0, ray_z], dtype=np.float64)
        ray_direction = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        scan = np.empty(len(self.grid), dtype=np.float32)
        for i, xy in enumerate(world_xy):
            ray_origin[:2] = xy
            distance, geom_id, flex_id = terrain_ray(
                self.model, data, ray_origin, ray_direction, self.terrain_group, self.terrain_flex_ids)
            if distance < 0 or (flex_id < 0 and geom_id not in self.terrain_geom_ids):
                raise RuntimeError(f"No terrain ray hit at scan index {i}, world_xy={xy.tolist()}")
            hit_z = ray_z - distance
            self.last_scan_hits[i] = (xy[0], xy[1], hit_z)
            scan[i] = np.clip(base_pos[2] - hit_z - SCAN_OFFSET, -1.0, 1.0)
        return scan

    def capture_height_scan(self, data, grid, *, return_normals=False):
        """Additional measured terrain rays; missed outer rays remain unknown."""
        rotation = data.xmat[self.base_id].reshape(3,3)
        base = data.xpos[self.base_id]
        yaw = math.atan2(rotation[1,0],rotation[0,0])
        co,si = math.cos(yaw),math.sin(yaw)
        xy = np.column_stack((base[0]+co*grid[:,0]-si*grid[:,1],
                              base[1]+si*grid[:,0]+co*grid[:,1]))
        hits = np.column_stack((xy,np.full(len(grid),np.nan)))
        normals = np.full_like(hits, np.nan) if return_normals else None
        origin = np.array([0.,0.,max(base[2]+SCAN_HEIGHT,self.terrain.ray_top)])
        for i,point in enumerate(xy):
            origin[:2] = point
            distance,geom,flex = terrain_ray(self.model,data,origin,np.array([0.,0.,-1.]),
                                             self.terrain_group,self.terrain_flex_ids,
                                             normal=normals[i] if return_normals else None)
            if distance >= 0 and (flex >= 0 or geom in self.terrain_geom_ids):
                hits[i,2] = origin[2]-distance
        return (hits, normals) if return_normals else hits

    def observe(self, data: mujoco.MjData, command: np.ndarray, previous_action: np.ndarray) -> np.ndarray:
        body_velocity = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self.model, data, mujoco.mjtObj.mjOBJ_BODY, self.base_id, body_velocity, 1
        )
        rotation = data.xmat[self.base_id].reshape(3, 3)
        projected_gravity = rotation.T @ np.array([0.0, 0.0, -1.0], dtype=np.float64)
        joint_pos = data.qpos[self.qpos_ids] - DEFAULT_JOINT_POS
        joint_pos[12:] = 0.0  # training's joint_pos_rel_without_wheel
        joint_vel = data.qvel[self.qvel_ids]
        obs = np.concatenate(
            (
                np.clip(body_velocity[:3], -100.0, 100.0) * 0.25,
                np.clip(projected_gravity, -100.0, 100.0),
                command,
                np.clip(joint_pos, -100.0, 100.0),
                np.clip(joint_vel, -100.0, 100.0) * 0.05,
                np.clip(previous_action, -100.0, 100.0),
                self.height_scan(data),
            )
        ).astype(np.float32)
        if obs.shape != (ACTOR_OBS_DIM,) or not np.isfinite(obs).all():
            raise RuntimeError(f"Invalid actor observation: shape={obs.shape}, finite={np.isfinite(obs).all()}")
        return obs

    def apply_pose_pd(self, data: mujoco.MjData, target_pos: np.ndarray) -> None:
        """Hold a transition pose with the saved joint limits and wheel damping."""
        joint_pos = data.qpos[self.qpos_ids]
        joint_vel = data.qvel[self.qvel_ids]
        effort = np.zeros(ACTION_DIM, dtype=np.float64)
        effort[:12] = np.clip(
            80.0 * (target_pos[:12] - joint_pos[:12]) - 2.0 * joint_vel[:12],
            -LEG_EFFORT_LIMIT, LEG_EFFORT_LIMIT,
        )
        effort[12:] = np.clip(-0.6 * joint_vel[12:], -WHEEL_EFFORT_LIMIT, WHEEL_EFFORT_LIMIT)
        data.ctrl[self.actuator_ids] = effort

    def apply_damping(self, data: mujoco.MjData) -> None:
        joint_vel = data.qvel[self.qvel_ids]
        effort = np.zeros(ACTION_DIM, dtype=np.float64)
        effort[:12] = np.clip(-2.0 * joint_vel[:12], -LEG_EFFORT_LIMIT, LEG_EFFORT_LIMIT)
        effort[12:] = np.clip(-0.6 * joint_vel[12:], -WHEEL_EFFORT_LIMIT, WHEEL_EFFORT_LIMIT)
        data.ctrl[self.actuator_ids] = effort

    def apply_pd(self, data: mujoco.MjData, action: np.ndarray) -> None:
        action = np.clip(action, -100.0, 100.0)
        joint_pos = data.qpos[self.qpos_ids]
        joint_vel = data.qvel[self.qvel_ids]
        target_pos = DEFAULT_JOINT_POS[:12] + action[:12] * ACTION_SCALE[:12]
        target_vel = action[12:] * ACTION_SCALE[12:]
        effort = np.empty(ACTION_DIM, dtype=np.float64)
        effort[:12] = np.clip(80.0 * (target_pos - joint_pos[:12]) - 2.0 * joint_vel[:12],
                              -LEG_EFFORT_LIMIT, LEG_EFFORT_LIMIT)
        effort[12:] = np.clip(0.6 * (target_vel - joint_vel[12:]),
                              -WHEEL_EFFORT_LIMIT, WHEEL_EFFORT_LIMIT)
        data.ctrl[self.actuator_ids] = effort


def smoothstep(fraction: float) -> float:
    fraction = float(np.clip(fraction, 0.0, 1.0))
    return fraction * fraction * (3.0 - 2.0 * fraction)


class TeleopKeyboard:
    """Share MuJoCo window and terminal key presses with the physics loop."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: deque[str] = deque()
        self._last_motion_press: dict[str, float] = {}
        self._window_keys = X11WindowKeys.open()
        self._command = np.zeros(3, dtype=np.float64)
        self._terminal_physical = False

    def on_keycode(self, keycode: int) -> None:
        with self._lock:
            if self._window_keys is not None:
                self._window_keys.note_viewer_event()
                # MuJoCo invokes this callback on PRESS, not GLFW_REPEAT/RELEASE.
                # Motion keys therefore come from physical state, not a timeout.
                if 65 <= keycode <= 90 and chr(keycode).lower() in TELEOP_VELOCITIES:
                    return
        if keycode == 256:  # GLFW escape
            self.on_char("escape")
        elif keycode == 259:  # GLFW backspace (also resets the native viewer)
            self.on_char("reset")
        elif 65 <= keycode <= 90:
            self.on_char(chr(keycode).lower())

    def on_char(self, key: str) -> None:
        key = "escape" if key == "\x1b" else "reset" if key in {"\x7f", "\x08"} else key.lower()
        with self._lock:
            if key in TELEOP_VELOCITIES:
                if not self._terminal_physical:
                    self._last_motion_press[key] = time.monotonic()
            elif key in {"z", "c", "x", "r", "h", "escape", "reset"}:
                self._events.append(key)

    def note_terminal_event(self) -> None:
        with self._lock:
            # Once verified, delayed terminal repeats after focus loss must
            # never re-enable the character-timeout fallback.
            self._terminal_physical |= (self._window_keys is not None
                                        and self._window_keys.note_terminal_event())
            if self._terminal_physical:
                self._last_motion_press.clear()

    def take_events(self) -> list[str]:
        with self._lock:
            events = list(self._events)
            self._events.clear()
            return events

    def clear_motion(self) -> None:
        with self._lock:
            self._last_motion_press.clear()
            self._command.fill(0.)

    def close(self) -> None:
        with self._lock:
            if self._window_keys is not None:
                self._window_keys.close()
                self._window_keys = None
            self._last_motion_press.clear()
            self._command.fill(0.)

    def velocity_command(self) -> np.ndarray:
        now = time.monotonic()
        with self._lock:
            self._last_motion_press = {
                key: seen for key, seen in self._last_motion_press.items()
                if now - seen <= TELEOP_KEY_TIMEOUT
            }
            if self._window_keys is not None and self._window_keys.focused():
                pressed = self._window_keys.pressed()
                self._last_motion_press.clear()
            elif self._terminal_physical or not self._last_motion_press:
                # Focus loss must not leave a decaying movement command behind.
                self._command.fill(0.)
                return self._command.astype(np.float32)
            else:
                pressed = set(self._last_motion_press)
            target = np.zeros(3, dtype=np.float64)
            for key in pressed:
                target += TELEOP_VELOCITIES[key]
            target = np.clip(target, -TELEOP_LIMITS, TELEOP_LIMITS)
            # Reverse through zero first. Each call is one policy tick, so a
            # paused viewer or slow render cannot jump straight to full speed.
            braking = (target * self._command < 0) | (target == 0)
            goal = np.where(braking, 0., target)
            rate = TELEOP_LIMITS / np.where(braking, TELEOP_STOP_TIME, TELEOP_RISE_TIME)
            self._command += np.clip(goal - self._command, -rate * POLICY_DT, rate * POLICY_DT)
            return self._command.astype(np.float32)


class TerminalInput:
    """Read the same keys from a focused terminal when stdin is a TTY."""

    def __init__(self, keyboard: TeleopKeyboard) -> None:
        self.keyboard = keyboard
        self.fd = 0
        self.saved_attrs = None
        self.saved_flags = None

    def start(self) -> bool:
        if not os.isatty(self.fd):
            return False
        self.saved_attrs = termios.tcgetattr(self.fd)
        self.saved_flags = fcntl.fcntl(self.fd, fcntl.F_GETFL)
        attrs = termios.tcgetattr(self.fd)
        attrs[3] &= ~(termios.ECHO | termios.ICANON)
        attrs[6][termios.VMIN] = 0
        attrs[6][termios.VTIME] = 0
        termios.tcsetattr(self.fd, termios.TCSANOW, attrs)
        fcntl.fcntl(self.fd, fcntl.F_SETFL, self.saved_flags | os.O_NONBLOCK)
        return True

    def poll(self) -> None:
        if self.saved_attrs is None:
            return
        while select.select([self.fd], [], [], 0.0)[0]:
            try:
                keys = os.read(self.fd, 64)
            except BlockingIOError:
                break
            if not keys:
                break
            self.keyboard.note_terminal_event()
            for key in keys:
                self.keyboard.on_char(chr(key))

    def close(self) -> None:
        if self.saved_attrs is not None:
            termios.tcsetattr(self.fd, termios.TCSANOW, self.saved_attrs)
            fcntl.fcntl(self.fd, fcntl.F_SETFL, self.saved_flags)
            self.saved_attrs = None


def launch_viewer(model: mujoco.MjModel, data: mujoco.MjData, key_callback=None):
    from mujoco import viewer as mj_viewer
    existing_threads = set(threading.enumerate())
    viewer = mj_viewer.launch_passive(model, data, key_callback=key_callback)
    viewer_threads = [thread for thread in threading.enumerate() if thread not in existing_threads]
    return viewer, viewer_threads


def close_viewer(viewer, viewer_threads: list[threading.Thread]) -> None:
    if viewer is None:
        return
    viewer.close()
    # MuJoCo 3.10 close() requests exit but does not join its GL thread.
    for thread in viewer_threads:
        thread.join(timeout=5.0)
    if any(thread.is_alive() for thread in viewer_threads):
        raise RuntimeError("MuJoCo viewer thread did not stop after close()")


def draw_height_scan(viewer, contract: M20Contract, visible: bool) -> None:
    """Draw the actor's cached ray hits in the passive viewer's user scene."""
    with viewer.lock():
        scene = viewer.user_scn
        scene.ngeom = 0
        if not visible:
            return
        hits = contract.last_scan_hits
        if scene.maxgeom < len(hits):
            raise RuntimeError(f"Viewer supports {scene.maxgeom} user geoms, need {len(hits)}")
        center_z = hits[len(hits) // 2, 2]
        size = np.array([0.018, 0.0, 0.0], dtype=np.float64)
        orientation = np.eye(3, dtype=np.float64).ravel()
        for i, hit in enumerate(hits):
            delta = float(hit[2] - center_z)
            strength = min(abs(delta) / 0.25, 1.0)
            if delta > 0.02:
                rgb = (0.2 + 0.8 * strength, 0.85 - 0.7 * strength, 0.2)
            elif delta < -0.02:
                rgb = (0.2, 0.85 - 0.55 * strength, 0.2 + 0.8 * strength)
            else:
                rgb = (0.2, 0.9, 0.25)
            rgba = np.array([*rgb, 0.9], dtype=np.float32)
            pos = hit.copy()
            pos[2] += 0.025
            mujoco.mjv_initGeom(scene.geoms[i], mujoco.mjtGeom.mjGEOM_SPHERE,
                                size, pos, orientation, rgba)
        scene.ngeom = len(hits)


def interactive_rollout(args: argparse.Namespace, *, actor=None, execution_factory=None) -> dict:
    """Start prone, stand with Z, then hand control to the teacher with C."""
    torch.set_num_threads(1)
    if actor is None:
        actor = load_actor(args.checkpoint)
    model = load_model(args.model, args)
    contract = M20Contract(model, args.terrain_config)
    data = mujoco.MjData(model)
    contract.reset(data, prone=True)
    perception = make_noise_adapter(args, contract)
    execution = execution_factory(model, contract) if execution_factory is not None else None
    trot = make_trot_controller(contract, actor, POLICY_DT)
    heading_hold = heading_controller(contract, actor, getattr(args, "disable_heading_hold", False))
    metrics = live = None
    if getattr(args, "quiet_landing", False):
        from quiet_landing_metrics import QuietMujocoMetrics, LiveSignals
        contract.height_scan(data)
        metrics = QuietMujocoMetrics(model, contract)
        live = LiveSignals() if args.live_plot else None
    print("[TERRAIN] " + json.dumps(args.terrain_config.summary()), flush=True)
    keys = TeleopKeyboard()
    terminal = TerminalInput(keys)
    viewer, viewer_threads = launch_viewer(model, data, key_callback=keys.on_keycode)
    try:
        with viewer.lock():
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = contract.base_id
            viewer.cam.distance = 3.2
            viewer.cam.elevation = -18.0
        terminal_active = terminal.start()
        print("[KEYS] Z stand | C teacher policy | X lie down | R damping | Backspace reset | H height scan | Esc quit", flush=True)
        print("[HEIGHTMAP] 187 points: green=local ground, red=higher, blue=lower", flush=True)
        print("[KEYS] W/S forward/back | A/D left/right | Q/E turn left/right", flush=True)
        print(f"[KEYS] Input: MuJoCo window{' or focused terminal' if terminal_active else ''}", flush=True)
        print(f"[KEYS] Motion: {'X11 held keys in viewer or verified local terminal; combine W+A/Q' if keys._window_keys is not None else 'limited press/timeout fallback; reliable held combinations require an X11 viewer'}", flush=True)
        print(f"[KEYS] Gradual speed: {TELEOP_RISE_TIME:g}s to full speed, {TELEOP_STOP_TIME:g}s release braking; R stops immediately", flush=True)
        print("[MODE] prone", flush=True)
    except BaseException:
        terminal.close()
        try:
            close_viewer(viewer, viewer_threads)
        finally:
            keys.close()
        raise

    mode = "prone"
    stand_start = 0.0
    stand_initial = None
    lie_start = 0.0
    lie_initial = None
    policy_start = 0.0
    pending_policy = False
    show_height_scan = True
    previous_action = np.zeros(ACTION_DIM, dtype=np.float32)
    records = {"time": [], "root_pos": [], "command": [], "requested_command": [], "mode": [],
               "action": [], "action_issued": []} if args.output else None
    if records is not None and trot is not None:
        records["trot_control"] = []
    transitions = [{"time": 0.0, "mode": "prone"}]
    start_x = float(data.xpos[contract.base_id, 0])
    max_x = start_x
    fell_ever = False
    substeps = round(POLICY_DT / PHYSICS_DT)
    step = 0
    steps_done = 0

    def enter(new_mode: str) -> None:
        nonlocal mode
        mode = new_mode
        keys.clear_motion()
        if heading_hold is not None:
            heading_hold.reset()
        if trot is not None:
            trot.reset()
        if execution is not None:
            execution.reset()
        if perception is not None:
            perception.reset()
        transitions.append({"time": round(float(data.time), 3), "mode": new_mode})
        print(f"[MODE] {new_mode} t={data.time:.2f}s", flush=True)

    def reset_session() -> None:
        nonlocal pending_policy, stand_initial, lie_initial
        contract.reset(data, prone=True)
        pending_policy = False
        stand_initial = lie_initial = None
        previous_action.fill(0.)
        if metrics is not None:
            metrics.reset()
        enter("prone")

    try:
        while viewer.is_running() and (args.steps is None or step < args.steps):
            tick_start = time.perf_counter()
            if execution is not None or perception is not None:
                mujoco.mj_fwdPosition(model, data)
                mujoco.mj_fwdVelocity(model, data)
            terminal.poll()
            quit_requested = False
            for key in keys.take_events():
                if key == "escape":
                    quit_requested = True
                elif key == "reset":
                    reset_session()
                elif key == "h":
                    show_height_scan = not show_height_scan
                    print(f"[HEIGHTMAP] {'visible' if show_height_scan else 'hidden'}", flush=True)
                elif key == "r":
                    pending_policy = False
                    previous_action.fill(0)
                    keys.clear_motion()
                    enter("damping")
                elif key == "z":
                    if mode in {"prone", "damping", "lying"}:
                        stand_start = float(data.time)
                        stand_initial = data.qpos[contract.qpos_ids].copy()
                        pending_policy = False
                        enter("standing")
                elif key == "c":
                    if mode == "standing":
                        pending_policy = True
                        print("[MODE] policy queued until standing finishes", flush=True)
                    elif mode == "ready":
                        policy_start = float(data.time)
                        previous_action.fill(0)
                        enter("policy")
                    elif mode != "policy":
                        print("[MODE] press Z and wait for ready before C", flush=True)
                elif key == "x" and mode not in {"prone", "lying"}:
                    lie_start = float(data.time)
                    lie_initial = data.qpos[contract.qpos_ids].copy()
                    pending_policy = False
                    previous_action.fill(0)
                    keys.clear_motion()
                    enter("lying")
            if quit_requested:
                break

            if mode == "standing" and data.time - stand_start >= 2.0 * STAND_PHASE_DURATION:
                position = data.xpos[contract.base_id]
                ground = contract.ground_height(data, float(position[0]), float(position[1]))
                upright = data.xmat[contract.base_id].reshape(3, 3)[2, 2] > 0.8
                if position[2] >= ground + 0.42 and upright:
                    enter("ready")
                    if pending_policy:
                        policy_start = float(data.time)
                        previous_action.fill(0)
                        enter("policy")
                elif step % args.print_every == 0:
                    print("[MODE] holding stand pose until body is upright", flush=True)
            if mode == "lying" and data.time - lie_start >= STAND_PHASE_DURATION:
                enter("prone")

            requested_command = keys.velocity_command() if mode == "policy" else np.zeros(3, dtype=np.float32)
            command = heading_hold.update(data, requested_command) if mode == "policy" and heading_hold is not None else requested_command.copy()
            if metrics is not None:
                metrics.command_xy = command[:2]
                metrics.tracker.begin_step()
                current = metrics.kinematics if data.time > 0 else data
                quiet_start_xy = current.xpos[contract.base_id, :2].copy()
            applied_action = np.zeros(ACTION_DIM, dtype=np.float32)
            issued_action = applied_action.copy()
            trot_control = np.zeros(6, dtype=np.float32)
            if mode == "policy":
                obs = (perception.observe(data, command, previous_action, step) if perception is not None
                       else contract.observe(data, command, previous_action))
                if execution is not None:
                    execution.observe(data, perception.map_adapter if perception is not None else None)
                if trot is not None:
                    trot_control = (trot.control(data, requested_command, terrain_points=perception.terrain_points(data))
                                    if perception is not None else trot.control(data, requested_command))
                with torch.inference_mode():
                    tensor_obs = (execution.policy_inputs(obs, requested_command, actor, data, perception, trot_control)
                                  if execution is not None and hasattr(execution, "policy_inputs")
                                  else policy_inputs(obs, requested_command, actor, trot_control))
                    raw_action = actor(tensor_obs).squeeze(0).cpu().numpy()
                if raw_action.shape != (ACTION_DIM,) or not np.isfinite(raw_action).all():
                    raise RuntimeError(f"Invalid actor action at step {step}: {raw_action}")
                blend = min(1.0, (data.time - policy_start) / 0.3)
                applied_action = np.clip(raw_action * blend, -100.0, 100.0).astype(np.float32)
                previous_action = applied_action.copy()
                issued_action = applied_action.copy()
                if execution is not None and hasattr(execution, 'check_issued'):
                    execution.check_issued(data, issued_action)
                if perception is not None:
                    applied_action = perception.action(issued_action)
            elif mode == "standing":
                elapsed = data.time - stand_start
                if elapsed < STAND_PHASE_DURATION:
                    blend = smoothstep(elapsed / STAND_PHASE_DURATION)
                    target = stand_initial + blend * (PRE_STAND_JOINT_POS - stand_initial)
                else:
                    blend = smoothstep((elapsed - STAND_PHASE_DURATION) / STAND_PHASE_DURATION)
                    target = PRE_STAND_JOINT_POS + blend * (DEFAULT_JOINT_POS - PRE_STAND_JOINT_POS)
            elif mode == "lying":
                blend = smoothstep((data.time - lie_start) / STAND_PHASE_DURATION)
                target = lie_initial + blend * (PRONE_JOINT_POS - lie_initial)
            elif mode == "ready":
                target = DEFAULT_JOINT_POS
            if mode != "policy" and (show_height_scan or metrics is not None):
                contract.height_scan(data)

            if records is not None:
                records["time"].append(float(data.time))
                records["root_pos"].append(data.xpos[contract.base_id].copy())
                records["command"].append(command.copy())
                records["requested_command"].append(requested_command.copy())
                records["mode"].append(mode)
                records["action"].append(applied_action.copy())
                records["action_issued"].append(issued_action.copy())
                if trot is not None:
                    records["trot_control"].append(trot_control.copy())
            for _ in range(substeps):
                if execution is not None:
                    mujoco.mj_step1(model, data)
                if mode == "policy":
                    if execution is not None:
                        data.ctrl[contract.actuator_ids] = execution.apply(data, applied_action, command)
                    else:
                        contract.apply_pd(data, applied_action)
                elif mode in {"standing", "ready", "lying"}:
                    contract.apply_pose_pd(data, target)
                else:
                    contract.apply_damping(data)
                if metrics is None:
                    if execution is None:
                        mujoco.mj_step(model, data)
                    else:
                        mujoco.mj_step2(model, data)
                else:
                    if execution is None:
                        mujoco.mj_step1(model, data)
                    snapshot = metrics.pre_step(data)
                    mujoco.mj_step2(model, data)
                    metrics.post_step(data, snapshot, record=args.output is not None or live is not None)
                    if args.output is None:
                        for group in (metrics.records, metrics.extended_records):
                            for values in group.values(): del values[:-1000]
                        del metrics.events[:-1000]
                if execution is not None:
                    if mode != "policy" and hasattr(execution, "after_idle_step"):
                        execution.after_idle_step(data)
                    else:
                        execution.after_step(data)
                    if metrics is not None and (args.output is not None or live is not None):
                        execution.record(metrics.extended_records)
                if trot is not None:
                    trot.sample_contacts(data)
            if execution is not None or perception is not None:
                mujoco.mj_fwdPosition(model, data)
                mujoco.mj_fwdVelocity(model, data)
                if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                    raise RuntimeError("Non-finite MuJoCo state")
            steps_done += 1
            if metrics is not None:
                metrics.tracker.distance[0] += float(np.linalg.norm(metrics.kinematics.xpos[contract.base_id, :2]-quiet_start_xy))
            if perception is not None:
                perception.draw(viewer, data, show_height_scan)
            else:
                draw_height_scan(viewer, contract, show_height_scan)
            if metrics is not None:
                metrics.draw(viewer, data)
            if live is not None and step % 5 == 0:
                live.update(metrics)
            before_sync = float(data.time)
            viewer.sync()
            if data.time < before_sync:
                # Native Backspace/UI Reset uses XML qpos0. Replace it before
                # any physics step, including when no key callback is emitted.
                reset_session()
            time.sleep(max(0.0, POLICY_DT - (time.perf_counter() - tick_start)))
            position = data.xpos[contract.base_id]
            max_x = max(max_x, float(position[0]))
            if mode == "policy":
                ground = contract.ground_height(data, float(position[0]), float(position[1]))
                upright = data.xmat[contract.base_id].reshape(3, 3)[2, 2] > 0.35
                fell = position[2] < ground + 0.22 or not upright
                fell_ever |= fell
                if fell and args.stop_on_fall:
                    print("[MODE] policy fall detected; stopping", flush=True)
                    break
            if step % args.print_every == 0:
                print(
                    f"step={step:04d} mode={mode} t={data.time:.2f}s "
                    f"x={position[0]:.3f} z={position[2]:.3f} cmd={command.tolist()}",
                    flush=True,
                )
            step += 1
    finally:
        try:
            terminal.close()
        finally:
            try:
                close_viewer(viewer, viewer_threads)
            finally:
                keys.close()
                if live is not None:
                    live.close()

    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "model": str(args.model.resolve()),
        **args.terrain_config.summary(),
        **contract.crossing_summary(data, fell_ever),
        "steps_executed": steps_done,
        "final_mode": mode,
        "transitions": transitions,
        "max_x": max_x,
        "final_x": float(data.xpos[contract.base_id, 0]),
        "final_z": float(data.xpos[contract.base_id, 2]),
        "fell_in_policy": bool(fell_ever),
        "critic_used": False,
    }
    if trot is not None:
        result["stair_trot"] = trot.summary()
        result["actor_input_dim"] = sum(result["stair_trot"]["input_groups"].values())
    if execution is not None:
        result.update(execution.metadata())
    result.update(noise_metadata(args, perception))
    if records is not None:
        for key in records:
            records[key] = np.asarray(records[key])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, **records, summary=json.dumps(result))
        result["trajectory"] = str(args.output.resolve())
    if metrics is not None:
        result["quiet"] = metrics.summary()
        if getattr(args, "snapshot", None):
            metrics.snapshot(data, args.snapshot)
        if args.output is not None:
            quiet_path = args.output.with_name(args.output.stem+"_quiet.npz")
            metrics.save(quiet_path, result)
            result["quiet_trajectory"] = str(quiet_path.resolve())
    print(json.dumps(result, indent=2), flush=True)
    return result


def rollout(args: argparse.Namespace) -> dict:
    torch.set_num_threads(1)
    actor = load_actor(args.checkpoint)
    model = load_model(args.model, args)
    contract = M20Contract(model, args.terrain_config)
    data = mujoco.MjData(model)
    contract.reset(data)
    perception = make_noise_adapter(args, contract)
    trot = make_trot_controller(contract, actor, POLICY_DT)
    heading_hold = heading_controller(contract, actor, getattr(args, "disable_heading_hold", False) or getattr(args, "track_course", False))
    metrics = live = None
    if getattr(args, "quiet_landing", False):
        from quiet_landing_metrics import QuietMujocoMetrics, LiveSignals
        contract.height_scan(data)
        metrics = QuietMujocoMetrics(model, contract)
        live = LiveSignals() if args.live_plot else None
    print("[TERRAIN] " + json.dumps(args.terrain_config.summary()), flush=True)
    requested_command = np.array([args.command_x, args.command_y, args.command_yaw], dtype=np.float32)
    command = requested_command.copy()
    previous_action = np.zeros(ACTION_DIM, dtype=np.float32)
    substeps = round(POLICY_DT / PHYSICS_DT)
    if not math.isclose(substeps * PHYSICS_DT, POLICY_DT):
        raise RuntimeError("Policy period is not an integer number of physics steps")
    records = {"time": [], "root_pos": [], "root_quat": [], "observation": [], "action": [],
               "action_applied": [], "command": [], "requested_command": []}
    if trot is not None:
        records["trot_control"] = []
    start_x = float(data.xpos[contract.base_id, 0])
    fell = False
    fell_ever = False
    max_x = start_x
    viewer = None
    viewer_threads = []
    if args.viewer:
        viewer, viewer_threads = launch_viewer(model, data)

    try:
        for step in range(args.steps if args.steps is not None else 1000):
            tic = time.perf_counter()
            if perception is not None:
                mujoco.mj_fwdPosition(model, data)
                mujoco.mj_fwdVelocity(model, data)
            if heading_hold is not None:
                command = heading_hold.update(data, requested_command)
            if getattr(args, "track_course", False):
                # Same external command generator for every policy. Keep the
                # course heading/center rather than requesting zero yaw rate.
                rotation = data.xmat[contract.base_id].reshape(3, 3)
                yaw = math.atan2(rotation[1, 0], rotation[0, 0])
                route_xy, target = (trot.state.fixed_route if trot is not None and trot.state.fixed_route is not None
                                    else (contract.terrain.spawn_xy, contract.terrain.spawn_yaw))
                lateral = float(np.clip(-0.5*(data.xpos[contract.base_id, 1]-route_xy[1]), -0.25, 0.25))
                world_x = args.command_x*math.cos(target)-lateral*math.sin(target)
                world_y = args.command_x*math.sin(target)+lateral*math.cos(target)
                command[:2] = (math.cos(yaw)*world_x+math.sin(yaw)*world_y,
                               -math.sin(yaw)*world_x+math.cos(yaw)*world_y)
                heading_error = math.atan2(math.sin(target-yaw), math.cos(target-yaw))
                command[2] = np.clip(0.5*heading_error, -0.5, 0.5)
            obs = (perception.observe(data, command, previous_action, step) if perception is not None
                   else contract.observe(data, command, previous_action))
            trot_control = (trot.control(data, requested_command, terrain_points=perception.terrain_points(data))
                            if trot is not None and perception is not None else
                            trot.control(data, requested_command) if trot is not None else None)
            with torch.inference_mode():
                tensor_obs = policy_inputs(obs, requested_command, actor, trot_control)
                action = actor(tensor_obs).squeeze(0).cpu().numpy()
            if action.shape != (ACTION_DIM,) or not np.isfinite(action).all():
                raise RuntimeError(f"Invalid actor action at step {step}: {action}")
            action = np.clip(action, -100.0, 100.0)
            applied_action = perception.action(action) if perception is not None else action
            if metrics is not None:
                metrics.command_xy = command[:2]
                metrics.tracker.begin_step()
                current = metrics.kinematics if data.time > 0 else data
                quiet_start_xy = current.xpos[contract.base_id, :2].copy()
            records["time"].append(float(data.time))
            records["root_pos"].append(data.xpos[contract.base_id].copy())
            records["root_quat"].append(data.xquat[contract.base_id].copy())
            records["observation"].append(obs)
            records["action"].append(action.copy())
            records["action_applied"].append(applied_action.copy())
            records["command"].append(command.copy())
            records["requested_command"].append(requested_command.copy())
            if trot is not None:
                records["trot_control"].append(trot_control.copy())
            for _ in range(substeps):
                contract.apply_pd(data, applied_action)
                if metrics is None:
                    mujoco.mj_step(model, data)
                else:
                    mujoco.mj_step1(model, data)
                    snapshot = metrics.pre_step(data)
                    mujoco.mj_step2(model, data)
                    metrics.post_step(data, snapshot, record=args.output is not None or live is not None)
                    if args.output is None:
                        for group in (metrics.records, metrics.extended_records):
                            for values in group.values(): del values[:-1000]
                        del metrics.events[:-1000]
                if trot is not None:
                    trot.sample_contacts(data)
            previous_action = action.astype(np.float32)
            if metrics is not None:
                metrics.tracker.distance[0] += float(np.linalg.norm(metrics.kinematics.xpos[contract.base_id, :2]-quiet_start_xy))
            if viewer is not None:
                if perception is not None:
                    perception.draw(viewer, data, True)
                else:
                    draw_height_scan(viewer, contract, True)
                if metrics is not None:
                    metrics.draw(viewer, data)
                before_sync = float(data.time)
                viewer.sync()
                if data.time < before_sync:
                    contract.reset(data)
                    previous_action.fill(0.)
                    command[:] = requested_command
                    for state in (perception, trot, heading_hold):
                        if state is not None:
                            state.reset()
                    if metrics is not None:
                        metrics.reset()
                time.sleep(max(0.0, POLICY_DT - (time.perf_counter() - tic)))
            if live is not None and step % 5 == 0:
                live.update(metrics)
            position = data.xpos[contract.base_id]
            gravity_z = float(data.xmat[contract.base_id].reshape(3, 3)[:, 2] @ np.array([0.0, 0.0, -1.0]))
            local_ground = contract.ground_height(data, float(position[0]), float(position[1]))
            fell = bool(position[2] < local_ground + 0.22 or gravity_z > -0.35)
            fell_ever |= fell
            max_x = max(max_x, float(position[0]))
            if step % args.print_every == 0 or fell:
                print(
                    f"step={step:04d} t={data.time:.2f}s x={position[0]:.3f} "
                    f"z={position[2]:.3f} ground={local_ground:.3f} fall={fell}",
                    flush=True,
                )
            if fell and args.stop_on_fall:
                break
            if getattr(args, "stop_at_goal", False) and contract.crossing_summary(data, fell_ever)["cleared_terrain"]:
                break
            if viewer is not None and not viewer.is_running():
                break
    finally:
        close_viewer(viewer, viewer_threads)
        if live is not None:
            live.close()

    for key in records:
        records[key] = np.asarray(records[key])
    final_pos = data.xpos[contract.base_id].copy()
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "model": str(args.model.resolve()),
        **args.terrain_config.summary(),
        **contract.crossing_summary(data, fell_ever),
        "steps_executed": len(records["time"]),
        "physics_enabled": True,
        "actor_input_dim": ACTOR_OBS_DIM,
        "critic_used": False,
        "start_x": start_x,
        "final_x": float(final_pos[0]),
        "final_z": float(final_pos[2]),
        "max_x": max_x,
        "forward_distance": float(final_pos[0] - start_x),
        "fell": fell_ever,
        "track_course": getattr(args, "track_course", False),
    }
    result.update(noise_metadata(args, perception))
    if trot is not None:
        result["stair_trot"] = trot.summary()
        result["actor_input_dim"] = sum(result["stair_trot"]["input_groups"].values())
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, **records, summary=json.dumps(result))
        result["trajectory"] = str(args.output.resolve())
    if metrics is not None:
        result["quiet"] = metrics.summary()
        if args.output is not None:
            quiet_path = args.output.with_name(args.output.stem+"_quiet.npz")
            metrics.save(quiet_path, result)
            result["quiet_trajectory"] = str(quiet_path.resolve())
    print(json.dumps(result, indent=2), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_noise_arguments(parser)
    add_spawn_arguments(parser)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model", type=Path, required=True, help="M20 MuJoCo XML (pin the tested model version)")
    terrain_choice = parser.add_mutually_exclusive_group()
    terrain_choice.add_argument("--terrain", choices=tuple(PRESETS),
                               help="Built-in terrain XML preset; defaults to ascent")
    terrain_choice.add_argument("--terrain-xml", type=Path,
                               help="Parameterized/native terrain XML, independent of --model")
    parser.add_argument("--stair-height", type=float, help="Override staircase XML step_height (metres)")
    parser.add_argument("--tread-depth", type=float, help="Override staircase XML tread_depth (metres)")
    parser.add_argument("--stair-count", type=int, help="Override staircase XML step_count")
    parser.add_argument("--stair-start", type=float, help="Override staircase XML start_x (metres)")
    parser.add_argument("--command-x", type=float, default=0.5)
    parser.add_argument("--command-y", type=float, default=0.0)
    parser.add_argument("--command-yaw", type=float, default=0.0)
    parser.add_argument("--steps", type=int, help="Limit to this many 50 Hz steps; viewer teleop runs until Esc by default")
    parser.add_argument("--print-every", type=int, default=10,
                        help="Print status every N policy steps (default: 10 = 0.2 simulated seconds)")
    parser.add_argument("--stop-on-fall", action="store_true")
    parser.add_argument("--track-course", action="store_true",
                        help="Autoplay only: shared heading/center-line velocity command feedback")
    parser.add_argument("--disable-heading-hold", action="store_true",
                        help="Disable checkpoint-specific legacy_v1 yaw feedback for residual models")
    parser.add_argument("--stop-at-goal", action="store_true",
                        help="Autoplay only: stop once all wheels cross the goal inside the course without falling")
    parser.add_argument("--viewer", action="store_true", help="Open interactive keyboard teleop")
    parser.add_argument("--autoplay", action="store_true", help="Run the old fixed-command rollout with --viewer")
    parser.add_argument("--output", type=Path, help="Optional compressed trajectory .npz")
    parser.add_argument("--quiet-landing", action="store_true",
                        help="Show four-wheel velocity/translation energy/terrain contact force overlays")
    parser.add_argument("--live-plot", action="store_true",
                        help="Live four-wheel speed/energy/force curves; enables --quiet-landing")
    args = parser.parse_args()
    validate_noise_arguments(args, parser)
    args.quiet_landing = args.quiet_landing or args.live_plot
    if ((args.steps is not None and args.steps <= 0)
        or any(value is not None and value <= 0 for value in
               (args.stair_count, args.stair_height, args.tread_depth))):
        parser.error("steps (when given), stair-count, stair-height, and tread-depth must be positive")
    if args.print_every <= 0:
        parser.error("print-every must be positive")
    return args


if __name__ == "__main__":
    cli_args = parse_args()
    if cli_args.viewer and not cli_args.autoplay:
        interactive_rollout(cli_args)
    else:
        rollout(cli_args)
