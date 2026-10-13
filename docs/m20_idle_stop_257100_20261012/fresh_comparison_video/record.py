"""Record frames during actual original MuJoCo interactive closed-loop physics."""
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import subprocess
import sys

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
sys.path.insert(0, str(ROOT / "deploy/deploy_mujoco"))
import deploy_mujoco as dm
from quiet_landing_metrics import QuietMujocoMetrics

label, case = sys.argv[1:3]
assert label in {"baseline", "fixed"}
assert case in {"零命令", "平地启停", "仅w"}
contract = json.loads((BASE / "criteria.json").read_text())
delivery = json.loads((BASE.parent / "validation.json").read_text())
checkpoint = Path(delivery["baseline_checkpoint"] if label == "baseline" else ROOT / delivery["checkpoint"])
expected_sha = delivery["baseline_sha256"] if label == "baseline" else delivery["sha256"]
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == expected_sha
sources = {**contract["source_sha256"], **json.loads((BASE / "instrument_sources.json").read_text())["source_sha256"]}
for name, expected in sources.items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
OUT = BASE / "evaluations" / label
OUT.mkdir(parents=True, exist_ok=True)
video = OUT / (case + ".mp4")
assert not video.exists(), video
full, poses = [], []
state = {"step1_calls": 0, "step2_calls": 0, "last_command": np.zeros(3), "frames": 0}
font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 18)
small = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 15)
title = "原版：model_257100" if label == "baseline" else "修复版：idle_stop_distilled_257100"
clip_start, clip_end = {"零命令": (10., 36.), "平地启停": (6., 36.), "仅w": (8., 22.)}[case]
reference_time = 10. if case == "零命令" else 14.02
original_step1, original_step2 = mujoco.mj_step1, mujoco.mj_step2


def step1(model, data):
    state["step1_calls"] += 1
    return original_step1(model, data)


def step2(model, data):
    state["step2_calls"] += 1
    return original_step2(model, data)


mujoco.mj_step1, mujoco.mj_step2 = step1, step2


class Viewer:
    def __init__(self, model, data):
        self.cam = SimpleNamespace()
        self.model, self.data = model, data
        self.visual_data = mujoco.MjData(model)
        self.renderer = mujoco.Renderer(model, height=360, width=640)
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.distance = 3.7 if case != "仅w" else 2.8
        self.camera.azimuth = 125 if case != "仅w" else 110
        self.camera.elevation = -26
        self.frame_number = 0
        self.reference = None
        self.reference_yaw = None
        self.anchor = None
        self.max_displacement = 0.
        self.pipe = subprocess.Popen([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-n", "-f", "rawvideo",
            "-vcodec", "rawvideo", "-pix_fmt", "rgb24", "-s", "640x456", "-r", "25",
            "-i", "-", "-an", "-c:v", "libx264", "-threads", "1", "-crf", "20",
            "-pix_fmt", "yuv420p", str(video)], stdin=subprocess.PIPE)

    def lock(self): return nullcontext()
    def is_running(self): return True

    def sync(self):
        data = self.data
        full.append((data.time, data.qpos.copy(), data.qvel.copy()))
        poses.append((data.time, data.xquat[state["base"]].copy(), data.xpos[state["base"]].copy()))
        t = float(data.time)
        w, x, y, z = data.qpos[3:7]
        yaw = np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))
        if case != "仅w" and self.reference is None and t >= reference_time:
            self.reference = data.qpos[:2].copy()
            self.reference_yaw = yaw
            self.anchor = data.qpos[:3].copy()
        if self.reference is not None:
            self.max_displacement = max(self.max_displacement, float(np.linalg.norm(data.qpos[:2]-self.reference)))
        self.frame_number += 1
        # Original control=50Hz. Capture every second post-physics control state.
        if self.frame_number % 2 or t < clip_start-1e-7 or t >= clip_end-1e-7:
            return
        self.visual_data.qpos[:] = data.qpos
        self.visual_data.qvel[:] = data.qvel
        # Only a separate rendering data object is forwarded. Physics data and
        # all original control/PD/observation/step calls are untouched.
        mujoco.mj_forward(self.model, self.visual_data)
        self.camera.lookat[:] = self.anchor if self.anchor is not None else data.qpos[:3]
        self.renderer.update_scene(self.visual_data, camera=self.camera)
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
        # Ground grid exists only in the rendering scene; it has no collision,
        # height-ray or dynamic state and gives the viewer a position reference.
        scene = self.renderer.scene
        for axis in range(2):
            for value in np.arange(-4, 8.01, .5):
                if scene.ngeom >= scene.maxgeom: break
                geom = scene.geoms[scene.ngeom]
                start, end = np.array([-4., value, .002]), np.array([8., value, .002])
                if axis: start, end = start[[1, 0, 2]], end[[1, 0, 2]]
                mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LINE, np.ones(3), np.zeros(3),
                                  np.eye(3).ravel(), np.array([.65, .69, .73, 1], np.float32))
                mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, 1., start, end)
                scene.ngeom += 1
        canvas = Image.new("RGB", (640, 456), "#f8fafc")
        canvas.paste(Image.fromarray(self.renderer.render()), (0, 40))
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 8), title, font=font, fill="#334155" if label == "baseline" else "#047857")
        command = state["last_command"]
        phase = "零命令驻足" if case == "零命令" else ("楼梯连续前进" if case == "仅w" else
                ("前进" if np.linalg.norm(command) else ("已松W：停车" if t >= 14 else "准备前进")))
        draw.text((10, 402), f"t={t:.2f}s  {phase}   命令=({command[0]:.2f}, {command[1]:.2f}, {command[2]:.2f})", font=small, fill="#0f172a")
        speed = float(np.linalg.norm(data.qvel[:2]))*1000
        if self.reference is None:
            footer = f"x={data.qpos[0]:.2f}m  y={data.qpos[1]:.2f}m  平移速度={speed:.1f}mm/s"
        else:
            delta = np.rad2deg(np.arctan2(np.sin(yaw-self.reference_yaw), np.cos(yaw-self.reference_yaw)))
            footer = f"最大位移={self.max_displacement*100:.1f}cm   速度={speed:.2f}mm/s   偏航变化={delta:.2f}°"
        draw.text((10, 427), footer, font=small, fill="#0f172a")
        self.pipe.stdin.write(np.asarray(canvas).tobytes())
        state["frames"] += 1
        if state["frames"] in {1, 300, 650}:
            canvas.save(OUT / f"{case}_frame_{state['frames']}.jpg")

    def close(self):
        self.pipe.stdin.close()
        assert self.pipe.wait() == 0
        self.renderer.close()


def launch(model, data, key_callback=None):
    state.update(data=data, base=model.body("base_link").id)
    return Viewer(model, data), []


class Keys:
    _window_keys = None
    def __init__(self): self.z = self.c = False
    def on_keycode(self, *args): pass
    def take_events(self):
        t = state["data"].time
        if not self.z: self.z = True; return ["z"]
        if t >= 6 and not self.c: self.c = True; return ["c"]
        return []
    def velocity_command(self):
        t = state["data"].time
        value = [.7, 0, 0] if (case == "仅w" and t >= 8) or (case == "平地启停" and 8 <= t < 14) else [0, 0, 0]
        state["last_command"] = np.array(value, np.float32)
        return state["last_command"].copy()
    def clear_motion(self): pass
    def close(self): pass


class Terminal:
    def __init__(self, *args): pass
    def start(self): return False
    def poll(self): pass
    def close(self): pass


QuietMujocoMetrics.draw = lambda *args: None
dm.launch_viewer = launch
dm.draw_height_scan = lambda *args: None
dm.TeleopKeyboard, dm.TerminalInput = Keys, Terminal
dm.time.sleep = lambda *args: None
args = SimpleNamespace(checkpoint=checkpoint, model=Path(contract["model"]), terrain=None,
    terrain_xml=Path(contract["flat_terrain"] if case == "平地启停" else contract["terrain"]),
    quiet_landing=True, live_plot=False, output=OUT/(label+"_"+case+"_原部署回放.npz"),
    steps=1800, print_every=600, stop_on_fall=True)
result = dm.interactive_rollout(args)
assert result["steps_executed"] == 1800 and not result["fell_in_policy"]
assert state["step1_calls"] == state["step2_calls"] == 7200
assert state["frames"] == round((clip_end-clip_start)*25), state["frames"]
np.savez_compressed(OUT/(case+"_full.npz"), time=np.array([r[0] for r in full]),
    qpos=np.array([r[1] for r in full]), qvel=np.array([r[2] for r in full]))
np.savez_compressed(OUT/(label+"_"+case+"_姿态.npz"), time=np.array([r[0] for r in poses]),
    quat=np.array([r[1] for r in poses]), pos=np.array([r[2] for r in poses]))
receipt = {"label":label, "case":case, "checkpoint":str(checkpoint.resolve()),
    "checkpoint_sha256":expected_sha, "source_sha256":sources, "terrain_xml":str(args.terrain_xml.resolve()),
    "Z_s":0, "C_s":6, "W_s":8 if case != "零命令" else None,
    "W_release_s":14 if case == "平地启停" else None,
    "command_schedule":[[0,[0,0,0]]]+([[8,[.7,0,0]]]+([[14,[0,0,0]]] if case == "平地启停" else []) if case != "零命令" else []),
    "course_correction":False, "physics":True, "entry":"deploy_mujoco.interactive_rollout",
    "headless_substitutions":["window draw replaced by live video renderer", "keyboard source", "wall time pacing"],
    "video":str(video.resolve()), "video_sha256":hashlib.sha256(video.read_bytes()).hexdigest(),
    "capture":"live post-physics states during closed-loop rollout; no saved-state-only replay",
    "visual_only_changes":["camera", "ground grid in rendering scene", "shadows/reflections off", "text overlays"],
    "clip_start_s":clip_start, "clip_end_s":clip_end, "fps":25, "frames":state["frames"],
    "mj_step1_calls":state["step1_calls"], "mj_step2_calls":state["step2_calls"]}
(OUT/(label+"_"+case+"_操作条件.json")).write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
print(json.dumps({"label":label,"case":case,"frames":state["frames"],"step1":state["step1_calls"],"step2":state["step2_calls"]},ensure_ascii=False))
