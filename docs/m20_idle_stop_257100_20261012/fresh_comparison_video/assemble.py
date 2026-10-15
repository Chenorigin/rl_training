"""Verify fresh physics recordings and assemble unaccelerated comparison clips."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import subprocess
import sys

import numpy as np

B = Path(__file__).resolve().parent
ROOT = B.parents[2]
CASES = [("零命令", "stationary_compare", 650),
         ("平地启停", "move_stop_compare", 750),
         ("仅w", "stairs_compare", 350)]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def probe(path):
    data = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=width,height,avg_frame_rate,nb_frames,duration:format=duration,size",
        "-of", "json", str(path)]))
    stream = data["streams"][0]
    return dict(width=int(stream["width"]), height=int(stream["height"]),
                fps=stream["avg_frame_rate"], frames=int(stream["nb_frames"]),
                duration_s=float(data["format"]["duration"]),
                bytes=int(data["format"]["size"]), sha256=digest(path))


recordings = []
for label, old_label in [("baseline", "baseline"), ("fixed", "distilled_round5")]:
    for case, _, frames in CASES:
        folder = B / "evaluations" / label
        receipt = json.loads((folder / f"{label}_{case}_操作条件.json").read_text())
        assert receipt["mj_step1_calls"] == receipt["mj_step2_calls"] == 7200
        assert receipt["physics"] and receipt["frames"] == frames
        assert receipt["capture"].startswith("live post-physics")
        assert digest(Path(receipt["checkpoint"])) == receipt["checkpoint_sha256"]
        metadata = probe(folder / (case + ".mp4"))
        assert metadata["frames"] == frames and metadata["fps"] == "25/1"
        assert metadata["width"] == 640 and metadata["height"] == 456
        assert abs(metadata["duration_s"] - frames / 25) < 1e-6
        assert metadata["sha256"] == receipt["video_sha256"]
        with np.load(folder / (case + "_full.npz")) as new, np.load(
                B.parent / "evaluations" / old_label / (case + "_full.npz")) as old:
            equality = {key: bool(np.array_equal(new[key], old[key])) for key in ("time", "qpos", "qvel")}
            assert all(equality.values()), (label, case, equality)
        recordings.append(dict(label=label, case=case, **metadata,
                               physics_state_bitwise_matches_previous=equality,
                               step1_calls=7200, step2_calls=7200))

for label in ("baseline", "fixed"):
    result = subprocess.run([sys.executable, str(B / "judge.py"), label],
                            capture_output=True, text=True)
    (B / (label + "_judge.log")).write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stderr
verdict = json.loads((B / "evaluations/fixed/verdict.json").read_text())
assert verdict["passed"] and len(verdict["checks"]) == 11

outputs = {}
for case, name, frames in CASES:
    output = B / (name + ".mp4")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-n",
                    "-i", str(B / "evaluations/baseline" / (case + ".mp4")),
                    "-i", str(B / "evaluations/fixed" / (case + ".mp4")),
                    "-filter_complex", "[0:v][1:v]hstack=inputs=2[v]", "-map", "[v]",
                    "-an", "-c:v", "libx264", "-threads", "1", "-filter_complex_threads", "1",
                    "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)], check=True)
    outputs[output.name] = probe(output)
    assert outputs[output.name]["frames"] == frames
    assert outputs[output.name]["width"] == 1280
    print(f"Assembled {output.name}", flush=True)

montages = [("comparison_concat.txt", "m20_257100_vs_idle_stop_mujoco.mp4",
             [B / (name + ".mp4") for _, name, _ in CASES], 1280),
            ("original_concat.txt", "original_257100_mujoco.mp4",
             [B / "evaluations/baseline" / (case + ".mp4") for case, _, _ in CASES], 640),
            ("fixed_concat.txt", "fixed_idle_stop_mujoco.mp4",
             [B / "evaluations/fixed" / (case + ".mp4") for case, _, _ in CASES], 640)]
for list_name, name, inputs, width in montages:
    playlist = B / list_name
    playlist.write_text("".join("file '" + str(path) + "'\n" for path in inputs))
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-n", "-f", "concat",
                    "-safe", "0", "-i", str(playlist), "-map", "0:v:0", "-c", "copy",
                    "-an", "-movflags", "+faststart", str(B / name)], check=True)
    outputs[name] = probe(B / name)
    assert outputs[name]["frames"] == 1750 and outputs[name]["duration_s"] == 70
    assert outputs[name]["width"] == width and outputs[name]["height"] == 456
    print(f"Assembled {name}", flush=True)

# Inspect actual encoded frames at the end of stationary, after stop, and stairs.
for name, moment in [("stationary_compare", 25), ("move_stop_compare", 28), ("stairs_compare", 8)]:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-n", "-ss", str(moment),
                    "-i", str(B / (name + ".mp4")), "-frames:v", "1", "-threads", "1",
                    str(B / (name + "_preview.jpg"))], check=True)

logs = {}
for log in B.glob("*.log"):
    content = log.read_text()
    rss = re.search(r"Maximum resident set size \(kbytes\): (\d+)", content)
    wall = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([^\n]+)", content)
    if rss:
        assert "Exit status: 0" in content
        assert int(rss[1]) < 2 * 1024 * 1024
        logs[log.name] = dict(max_rss_KiB=int(rss[1]), wall_time=wall[1])
assert len(logs) == 6

# Preserve the two separate training paths for a directly reviewable provenance.
scratch = ROOT / ".scratch/idle_stop_20261012"
shutil.copy2(scratch / "candidate_v2.log", B / "isaac_ppo_not_selected.log")
shutil.copy2(scratch / "fit_policy_round5.log", B / "selected_supervised_fit.log")
validation = dict(recordings=recordings, outputs=outputs, primary_verdict=verdict,
                  resource_measurements=logs,
                  scene_only_changes=["camera", "render grid", "shadow flags", "text overlays"],
                  original_control_physics_observations_unchanged=True,
                  speed="1x, 25fps; no time remapping", left="original 257100", right="fixed idle_stop",
                  montage_chapters_s=[dict(start=0, end=26, case="零命令"),
                                     dict(start=26, end=56, case="平地启停", release_W_at_video_s=34),
                                     dict(start=56, end=70, case="仅w")])
(B / "recording_validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2))
print(json.dumps(dict(videos=list(outputs), passed=True,
                      six_physics_trajectories_match_previous_bitwise=True,
                      primary_checks=verdict["checks"]), ensure_ascii=False, indent=2))
