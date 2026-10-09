"""Drive the original GUI FSM through scripted terminal keys; observe only."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "deploy/deploy_mujoco"))
import deploy_mujoco as deploy
from gait_metrics import GaitMetrics
import imageio.v2 as imageio
import mujoco
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--speed", type=float, default=.7)
parser.add_argument("--label", default="interactive_07")
parser.add_argument("--tread-depth", type=float)
selection = parser.parse_args()
deploy.TELEOP_VELOCITIES["w"] = (selection.speed, 0., 0.)
OUT = Path(__file__).resolve().parent / selection.label
OUT.mkdir(exist_ok=True)
active = None
renderer = None
writer = None
records = []
collisions = {}
snapshots = {}
best = {"front_support_min": float("inf"), "rear_support_min": float("inf"),
        "front_swing_min": float("inf"), "pitch_abs_max": -1.}
input_events = []

class MeasuredContract(deploy.M20Contract):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        global active
        active = self
        self.metrics = GaitMetrics(self)
        self.policy = False
        self.command = np.zeros(3)
        self.action = np.zeros(16)
        self.substep = 0

    def reset(self, data, *args, **kwargs):
        self.data = data
        return super().reset(data, *args, **kwargs)

    def observe(self, data, command, previous_action):
        self.policy = True
        self.command = command.copy()
        return super().observe(data, command, previous_action)

    def apply_pd(self, data, action):
        self.action = action.copy()
        return super().apply_pd(data, action)

class ScriptedTerminal(deploy.TerminalInput):
    def __init__(self, keys):
        super().__init__(keys)
        self.sent = set()

    def poll(self):
        super().poll()
        now = active.data.time
        for when, key in ((.25,"z"), (5.8,"c"), (29.5,"escape")):
            if now >= when and key not in self.sent:
                self.keyboard.on_char(key)
                self.sent.add(key)
                input_events.append({"time":float(now),"key":key})
        if 6.5 <= now < 28.5:
            # Equivalent to terminal W repeat; same production command/FSM path.
            self.keyboard.on_char("w")

def render_frame(data, azimuth=100):
    global renderer
    if renderer is None:
        renderer = mujoco.Renderer(active.model, width=640, height=480)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = data.xpos[active.base_id]
    cam.distance, cam.azimuth, cam.elevation = 2.9, azimuth, -16
    renderer.update_scene(data, camera=cam)
    return renderer.render().copy()

real_step = mujoco.mj_step
def measured_step(model, data, *args, **kwargs):
    global writer
    real_step(model, data, *args, **kwargs)
    if active is None or not active.policy:
        return
    active.substep += 1
    if active.substep % 4:
        return
    active.metrics.update(data, active.command)
    s = active.metrics.samples[-1]
    R = data.xmat[active.base_id].reshape(3,3)
    roll = np.arctan2(R[2,1], R[2,2])
    pitch = np.arctan2(-R[2,0], np.hypot(R[2,1],R[2,2]))
    row = dict(time=float(data.time), root_pos=data.xpos[active.base_id].copy(),
        root_quat=data.xquat[active.base_id].copy(), joint_pos=data.qpos[active.qpos_ids].copy(),
        joint_vel=data.qvel[active.qvel_ids].copy(), command=active.command.copy(), action=active.action.copy(),
        wheel_pos=data.xpos[active.wheel_body_ids].copy(), hip_pos=data.xpos[active.metrics.hips].copy(),
        wheel_ground=s["wheel_ground"], contact_force=s["forces"], extension=s["extension"],
        knee=s["knee"], clearance=s["clearance"], support=s["support"],
        contact_time=active.metrics.contact_time.copy(), roll=roll, pitch=pitch,
        torque=data.actuator_force[active.actuator_ids].copy())
    records.append(row)
    for i in range(data.ncon):
        contact = data.contact[i]
        bodies = model.geom_bodyid[[contact.geom1, contact.geom2]]
        if 0 not in bodies or bodies[0] == bodies[1]:
            continue
        body = int(bodies[1] if bodies[0] == 0 else bodies[0])
        if body in active.wheel_body_ids:
            continue
        f = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, f)
        magnitude = float(np.linalg.norm(f[:3]))
        if magnitude <= 1:
            continue
        name = model.body(body).name
        entry = collisions.setdefault(name,dict(samples=0,first_time=float(data.time),max_force_n=0.))
        entry["samples"] += 1
        entry["max_force_n"] = max(entry["max_force_n"], magnitude)
    if len(records) % 5 == 0:
        if writer is None:
            writer = imageio.get_writer(OUT / "closed_loop.mp4", fps=10, codec="libx264", quality=7)
        writer.append_data(render_frame(data))
    # Snapshot extrema only while climbing, excluding initial standing and platform.
    levels = np.rint(np.asarray(s["wheel_ground"]) / .15).astype(int)
    climbing = active.command[0] > .1 and levels.max() > 0 and levels.min() < 20
    if not climbing:
        return
    ext = np.asarray(s["extension"])
    support = np.asarray(s["support"])
    for tag, inds in (("front_support_min", np.flatnonzero(support[:2] & (active.metrics.contact_time[:2] >= .2))),
                      ("rear_support_min", np.flatnonzero(support[2:] & (active.metrics.contact_time[2:] >= .12)) + 2),
                      ("front_swing_min", np.flatnonzero(~support[:2]))):
        if not len(inds):
            continue
        leg = int(inds[np.argmin(ext[inds])])
        value = float(ext[leg])
        if value < best[tag]:
            best[tag] = value
            imageio.imwrite(OUT / f"{tag}.png", render_frame(data, 100 if leg % 2 == 0 else -100))
            snapshots[tag] = dict(time=float(data.time),leg=("fl","fr","hl","hr")[leg],extension_m=value,
                                  knee_rad=float(s["knee"][leg]),contact_time_s=float(active.metrics.contact_time[leg]))
    if abs(pitch) > best["pitch_abs_max"]:
        best["pitch_abs_max"] = abs(float(pitch))
        imageio.imwrite(OUT / "pitch_abs_max.png", render_frame(data))
        snapshots["pitch_abs_max"] = dict(time=float(data.time),pitch_deg=float(np.degrees(pitch)))

deploy.M20Contract = MeasuredContract
deploy.TerminalInput = ScriptedTerminal
mujoco.mj_step = measured_step
args = argparse.Namespace(model=Path('/home/ubuntu/桌面/m20_1/M20_perception_Lidar_rl_2/M20_perception_Lidar_rl/deep_robotics_model/M20/mjcf/M20.xml'),
    checkpoint=ROOT/'logs/rsl_rl/deeprobotics_m20_stair_teacher/2026-10-03_15-12-45_gait_resume/model_149999.pt',
    terrain_xml=ROOT/'deploy/deploy_mujoco/terrains/stairs_ascent.xml', terrain=None,
    stair_height=None, tread_depth=selection.tread_depth, stair_count=None, stair_start=None,
    viewer=True, autoplay=False, steps=1600, output=OUT/'basic.npz', print_every=250, stop_on_fall=True)
try:
    result = deploy.interactive_rollout(args)
finally:
    if writer is not None:
        writer.close()
    if renderer is not None:
        renderer.close()
    if records:
        np.savez_compressed(OUT/'telemetry.npz', **{k:np.asarray([row[k] for row in records]) for k in records[0]})
if active is None or not records:
    raise RuntimeError('No policy telemetry collected')
result.update(gait=active.metrics.summary(),landings=active.metrics.landings,
              nonwheel_terrain_contacts=collisions,snapshots=snapshots,input_events=input_events,
              control_source='scripted terminal Z/C and repeated W through original interactive FSM',
              forward_command_m_s=selection.speed,
              checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              terrain_sha256=hashlib.sha256(args.terrain_xml.read_bytes()).hexdigest())
(OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
print('[GAIT]',json.dumps(result['gait']),flush=True)
