# M20 Reinforcement Learning and Student Distillation

[M20 真机策略部署操作指南](docs/m20_real_robot_deployment.md)（中文：导出、两板部署、手柄接管、停止与回退）

[CTS v15 软件限位与续训入口](docs/m20_cts.md#软件关节限位)：训练和 MuJoCo 检查目标角、修正目标与反馈角；从 v14/1500 迁移使用 `--init-cts`。

[CTS v15 地图过期与稀疏恢复训练](docs/m20_cts.md#地图过期与稀疏恢复训练)：新 run 默认加入采集断流和恢复扰动，保持原观测维度；参数、对照开关和验证边界见指南。

[CTS v15 后腿上阶余量奖励](docs/m20_cts.md#后腿上阶余量奖励)：提前抬后轮、减少顶住阶沿和双后腿同时失去支撑；使用新 run 训练生效。

[CTS v15 复合指令与逐轴跟踪](docs/m20_cts.md#复合指令与逐轴跟踪)：补齐两轴组合、随机加减轴，并逐轴检查跟踪；采用新训练目标使用 `--init-cts`。

[CTS v15 墙旁通行与门框转角](docs/m20_cts.md#墙旁通行与门框转角)：加入单墙、窄通道、门框、前墙停车及左右转角，覆盖横偏与全身碰撞包络；新 run 训练生效。

English | [简体中文](README_CN.md)

Isaac Lab training, teacher-to-student distillation, and MuJoCo playback for the Deep Robotics M20 wheeled-legged robot. This repository contains PPO teachers, sensor-limited student policies, and selected checkpoints with their provenance.

Release: **`v0`** · Python package: **`0.1.0`** · Repository: [Chenorigin/rl_training](https://github.com/Chenorigin/rl_training).

## Installation

Use an existing, working Isaac Lab environment. The validation environment used Python 3.11, Isaac Sim 5.1.0, Isaac Lab 2.3.2, and `rsl-rl-lib==5.0.1`; installing this package alone does not install Isaac Sim or Isaac Lab.

```bash
git clone --branch v0 --depth 1 --recurse-submodules https://github.com/Chenorigin/rl_training.git
cd rl_training

conda activate <your-isaaclab-environment>
python -m pip install -e source/rl_training
```

The `deep_robotics_model/` submodule provides robot assets. For stair-based tasks, set the USD path on your machine; the existing default can refer to a local development asset directory. MuJoCo requires an explicit M20 MJCF/XML path, including its referenced meshes.

```bash
export RL_TRAINING_M20_USD_PATH=/absolute/path/to/M20/usd/M20.usd
```

MuJoCo playback also requires `mujoco`, PyTorch, NumPy, and TensorDict in the active environment.

All commands below run from the repository root. Before training, check available GPU memory, host memory, and output-disk capacity, then choose an appropriate environment count. If the Conda C++ runtime is incompatible, see `python scripts/tools/setup_conda_runtime.py` and reactivate the environment.

## Project layout

| Path | Contents |
| --- | --- |
| `source/rl_training/rl_training/` | Robot assets, task registration, and shared environment code |
| `source/rl_training/task_support/<task>/` | Task-specific perception, rewards, algorithms, runners, feedback, and model helpers |
| `source/rl_training/m20_student_v0/`, `m20_student_v1/` | Compatibility imports required by saved student checkpoints |
| `configs/distillation/` | Student JSON configuration |
| `scripts/reinforcement_learning/rsl_rl/` | PPO training/playback and the unified student distillation entry point |
| `scripts/tools/` | Reusable inspection, export, evaluation, and recording tools |
| `deploy/deploy_mujoco/` | MuJoCo playback, student adapters, controls, and terrain XMLs |
| `deploy/models/` | Selected full checkpoints with hashes and provenance |
| `tests/` | Observation, model, resume, deployment-interface, and task regression checks |
| `docs/` | Operating guides and required replay inputs |
| `logs/` | Local training runs and evaluation output; excluded from the release |

For the adjacent perception and robot deployment workspaces, see the [m20_heightmap_ws and m20_student_deploy handover context](docs/m20_heightmap_student_deploy_context.md) (Chinese).

Each task has three consistently named entry files under `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/`:

```text
mdp/<task>_env.py
config/wheeled/deeprobotics_m20/<task>_env_cfg.py
config/wheeled/deeprobotics_m20/agents/<task>_ppo_cfg.py
```

Additional programs belong in `task_support/<task>/`. Shared implementations use inheritance or explicit imports. Student `_ppo_cfg.py` files are uniformly named training configuration entries; the student algorithms still perform distillation.

## Registered tasks

The [task registry](source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/__init__.py) defines the Gym IDs. `python scripts/tools/list_envs.py` lists available environments in the Isaac Lab installation.

| Gym task ID | File prefix | Purpose |
| --- | --- | --- |
| `Flat-Deeprobotics-M20-v0` | `flat` | Flat-ground velocity tracking |
| `Rough-Deeprobotics-M20-v0` | `rough` | Rough-terrain velocity tracking |
| `Rough-Deeprobotics-M20-PreTeacher-v0` | `pre_teacher` | Preliminary teacher training |
| `Rough-Deeprobotics-M20-StairTeacher-v0` | `stair_teacher` | Stair teacher |
| `Rough-Deeprobotics-M20-IdleStopMotion-v1` | `idle_stop_motion_v1` | Motion residual over a frozen idle-stop baseline |
| `Rough-Deeprobotics-M20-StairQuiet-v1` | `stair_quiet_v1` | Impact-aware refinement; acceptance requires real sound-level measurements |
| `Rough-Deeprobotics-M20-StairQuiet-v1-Width` | `stair_quiet_v1_width` | [Stair width curriculum](docs/m20_stair_quiet_v1_width.md): physical width rows, per-environment success promotion and failure demotion, final-width contact gates |
| `Rough-Deeprobotics-M20-StairQuiet-v1-Safe` | `stair_quiet_v1_safe` | [Safe teacher refinement](docs/m20_stair_quiet_v1_safe.md): separate ascent/safety inputs, predictive footholds and independent material practice; continue from width weights |
| `Rough-Deeprobotics-M20-StairQuiet-v1-Transition` | `stair_quiet_v1_transition` | [v1 Transition follow-up](docs/m20_stair_quiet_v1_transition.md): continuous stop scan, varied entries, 17/30 closed stairs and per-protocol evidence; continue from complete Safe actor/critic |
| `Rough-Deeprobotics-M20-StairTrot-v1` | `stair_trot_v1` | Ascent-only diagonal trot residual over a frozen complete motion teacher |
| `Rough-Deeprobotics-M20-StairStudent-v0` | `stair_student` | Stair student environment |
| `Rough-Deeprobotics-M20-QuietLanding-v0` | `quiet_landing` | Quiet-landing teacher |
| `Rough-Deeprobotics-M20-GaitRefinement-v2` | `gait_refinement_v2` | Idle/heading refinement with replay preservation |
| `Rough-Deeprobotics-M20-Student-v0` | `student_v0` | Sensor-limited v0 student |
| `Rough-Deeprobotics-M20-Student-v1` | `student_v1` | Sensor-limited v1 student with optional history |
| `Rough-Deeprobotics-M20-Student-v3` | `student_v3` | Front-sensor student with four-frame history |
| `Rough-Deeprobotics-M20-Student-v3.5` | `student_v3_5` | Student distilled from the complete motion-residual teacher |
| `Rough-Deeprobotics-M20-Student-v3-Safe` | `student_v3_safe` | [Complete Safe teacher distillation](docs/m20_student_v3_safe.md), sensor-only student and mandatory landing controller |
| `Rough-Deeprobotics-M20-CTS` | `cts` | [Concurrent teacher/student learning](docs/m20_cts.md): shared PPO policy and critic, continued Safe teacher refinement and sensor-only student distillation |

## PPO training and Isaac playback

Train a teacher:

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Rough-Deeprobotics-M20-StairTeacher-v0 \
  --num_envs 64 --headless
```

Resume a local run with its critic and optimizer state:

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Rough-Deeprobotics-M20-StairTeacher-v0 --resume \
  --load_run '<run-directory-name>' --checkpoint model_N.pt --headless
```

`train.py` also accepts an explicit checkpoint path across experiment directories. For a fresh optimizer with actor initialization only, use `--init_actor_from <checkpoint>` instead of `--resume`; `--init_critic_from <checkpoint>` optionally initializes the critic too. `--max_iterations` controls the number of additional training iterations.

Play a retained teacher through Isaac:

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task Rough-Deeprobotics-M20-StairTeacher-v0 --num_envs 1 \
  --checkpoint deploy/models/stair_teacher/model_214700.pt
```

For `play.py`, an explicit `--checkpoint` is a file path. Omit it to use run-directory/checkpoint-pattern lookup.

`GaitRefinement-v2` reads `docs/m20_idle_stop_257100_20261012/reproduction_manifest.json` and its **12 NPZ input files**. Preserve this directory and its paths: these files are runtime replay data.

## IdleStopMotion-v1

This independent teacher integrates [frider635-beep/Deeprobotics_M20_stair at commit `8fb5d9a`](https://github.com/frider635-beep/Deeprobotics_M20_stair/tree/8fb5d9aa5809363dd9a2cccacf0b24ee6f542d81). The relevant upstream implementation hashes match the manifest embedded in checkpoint 46400.

The actor keeps a **frozen 244-dimensional baseline** and adds a bounded motion residual using the policy input plus a separate **3-dimensional raw `motion_control` command**. A raw zero command disables the residual exactly. The critic receives **285 dimensions plus the same 3-dimensional raw command**; the final action has 16 dimensions. The task preserves the upstream `legacy_v1` heading feedback, active reward weights, and descent-guide schedule. Its tuning and gait-curriculum state are isolated from the other tasks.

Fresh training initializes the baseline from `deploy/models/idle_stop_v1/model_257100_idle_stop.pt` and trains the motion residual and critic:

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Rough-Deeprobotics-M20-IdleStopMotion-v1 --num_envs 64 --headless
```

Resume the included full residual-teacher checkpoint:

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Rough-Deeprobotics-M20-IdleStopMotion-v1 --resume \
  --checkpoint deploy/models/idle_stop_motion_v1/idle_stop_motion_v1_46400.pt \
  --headless
```

Full resume restores actor, critic, optimizer, and guide counters without reinitializing the baseline. The standalone configuration supplies the profile and guide settings; the old upstream `--m20_profile` argument is unnecessary.

Isaac playback uses the same task and explicit checkpoint path. This actor requires both observation groups, so ordinary single-input JIT/ONNX export is unsupported. `play.py` reports that it skips those exports and uses the complete actor. Use normal task-generated commands for this Isaac path; the generic `--keyboard` observation override does not preserve the effective-command/raw-command pairing. The MuJoCo teacher viewer provides the existing interactive playback path described below.

## Student distillation

For reducing real stair-impact noise from the pinned 46400 teacher and 21000
student, see [StairQuiet-v1](docs/m20_stair_quiet_v1.md). It provides an isolated
training task, raw-contact MuJoCo comparison, an explicitly uncalibrated
structural-acoustic predictor, and calibrated WAV analysis. Long training is
started by the user; predicted sound is reported separately from actual acoustic
acceptance, and no 50% reduction has yet been established.

For the independent ascent-trot teacher, see [StairTrot-v1](docs/m20_stair_trot_v1.md).
It retains the complete motion teacher and learns a terrain-gated residual. Training
must be started by the user; the provided checks perform inference only.

Use `distill_student.py` for v0, v1, v3, and v3.5. The old `distill_student_v1.py` entry remains a compatibility wrapper. v3.5 learns the complete residual teacher while retaining the v3 student layout and four-frame history.

For `v1_safe` teachers, use the independent `Rough-Deeprobotics-M20-Student-v3-Safe` task. It learns the complete Safe actor, keeps privileged safety inputs teacher-only, and retains Safe physics/course and landing execution. See [student_v3_safe commands](docs/m20_student_v3_safe.md).

Check weights, observation contracts, and the resource budget before starting:

```bash
python scripts/reinforcement_learning/rsl_rl/distill_student.py \
  --task Rough-Deeprobotics-M20-Student-v3.5 \
  --teacher-checkpoint deploy/models/idle_stop_motion_v1/idle_stop_motion_v1_46400.pt \
  --num-envs 64 --device cuda:0 --preflight-only
```

Preflight does not launch Isaac or update an optimizer. To start this new training run, replace `--preflight-only` with `--headless`. Resume a retained student checkpoint with its saved task, configuration, environment count, and schedules:

```bash
python scripts/reinforcement_learning/rsl_rl/distill_student.py \
  --resume deploy/models/student_v3_5/model_21000.pt \
  --device cuda:0 --headless
```

For a fresh run with different student initialization, use `--teacher-checkpoint ... --init-student ...`. Resume rejects configuration overrides that would change the saved run. Details: [v0](docs/m20_student_v0.md), [v1 perception/history/resume](docs/m20_student_v1.md), and [v3.5 teacher/feedback contract](docs/m20_student_v3_5.md).

**CLI spelling differs:** PPO entry points use underscore-style options: `--num_envs` and `--load_run`, plus `--max_iterations` for training. Student distillation uses `--num-envs`, `--max-iterations`, and `--teacher-checkpoint`. MuJoCo flags also use hyphens, such as `--terrain-xml` and `--noise-profile`.

## MuJoCo playback and student export

StairQuiet-v1, Width, Safe and Transition checkpoints use `deploy/deploy_mujoco/deploy_quiet_landing.py` with the same `--checkpoint`, `--model`, `--terrain-xml` and `--viewer` flags. It automatically loads the matching complete actor, observations and scan routing; see [Safe manual deployment](docs/m20_stair_quiet_v1_safe.md#mujoco-手动部署) and [Transition continuation](docs/m20_stair_quiet_v1_transition.md). The safety48 group remains privileged simulator state under every noise profile.

Play the complete motion-residual teacher:

StairTrot-v1 checkpoints use this same entry point with their full actor and sensor-derived trot control. See the [StairTrot-v1 MuJoCo commands](docs/m20_stair_trot_v1.md#mujoco-部署).

```bash
python deploy/deploy_mujoco/deploy_mujoco.py \
  --checkpoint deploy/models/idle_stop_motion_v1/idle_stop_motion_v1_46400.pt \
  --model /absolute/path/to/M20.xml --terrain ascent --viewer
```

v1, v3, and v3.5 students share the v1 adapter. The saved v3.5 provenance enables its matching sensor-based heading feedback:

```bash
python deploy/deploy_mujoco/deploy_student_v1.py \
  --checkpoint deploy/models/student_v3_5/model_21000.pt \
  --model /absolute/path/to/M20.xml \
  --terrain-xml deploy/deploy_mujoco/terrains/all_terrain.xml \
  --noise-profile student-v1 --viewer
```

Use `deploy_student_v0.py` for v0 checkpoints. Viewer controls: `Z` stand, `C` enter policy control, `W/S/A/D/Q/E` move, `R` damping, and `H` height-map display. See the [MuJoCo guide](docs/m20_student_v0_mujoco.md) and [terrain guide](deploy/deploy_mujoco/terrains/README_CN.md).

All six MuJoCo deployment entry points share terrain-aware reset and gradual keyboard control. Startup and `Backspace`/viewer Reset place the robot just above its actual collision support; interactive reset returns to prone and clears the previous actions and sensor history (`Z`, then `C` to resume). On Ubuntu X11, hold combinations such as `W+A` or `W+Q` in the viewer or a verified local terminal. Each axis ramps up while held and brakes on release; focus loss and `R` clear the command immediately. The speed limits and rise/braking times are centralized in `deploy_mujoco.py` (`TELEOP_*`). SSH, unverified terminals and desktops without X11 key-state access use a limited timeout fallback; use the local X11 viewer for reliable held combinations. Headless/autoplay command timing remains fixed.

All six entries add a blue physical mat with **independent** placement. Use `--spawn-mat-xy X Y` to move only the mat, `--spawn-mat-yaw-deg DEG` to rotate only the mat, and `--spawn-mat-size LENGTH WIDTH THICKNESS` to set its full dimensions in metres. Robot reset X/Y and heading stay at `spawn_xy` / `spawn_yaw_deg`; use these mat options for partial-foot support and height-difference tests. Persistent XML settings are `spawn_mat_xy`, `spawn_mat_yaw_deg`, and `spawn_mat_size`. When omitted, the mat initially uses the robot's spawn location/heading. Restart to apply changes. `--no-spawn-mat` removes the mat. See the [independent mat settings](deploy/deploy_mujoco/terrains/README_CN.md#出生垫位置).

All four MuJoCo entry points accept `--noise-profile`. The teacher (`deploy_mujoco.py`) and quiet teacher (`deploy_quiet_landing.py`) default to `teacher` (clean); student entry points keep their existing `student-v1` default. Common names are `teacher`, `student-v0`, `student-v1`, `lidar_fov`, and `stress`; aliases `clean`, `realistic`, and `train` remain accepted. The teacher, quiet teacher, and v1 student also accept `front_only` / `front_only_30cm` and the existing single-factor v1 presets (`height`, `pose`, `dropout`, `latency`, `proprio`, `action_delay`, `quality_reference`, `confidence`).

Append `--noise-profile front_only` to the manual teacher or quiet command to use the existing front LiDAR profile. Teacher noise settings inherit the pinned v3.5 student's saved perception configuration; `--noise-reference /path/to/student.pt` or `/path/to/deployment.json` selects another v1 configuration source. `--profile-file /path/to/profiles.json` imports preset overrides, and `--noise-level` controls severity. These three entry points use the existing v1 JSON format (preset name → perception fields); v0 retains its `student_v0_profiles.json` format. The startup `[NOISE]` line and final `noise` summary identify the source, effective parameters, seed, and action delay.

Teacher actors still receive 244 values. Missing map cells stay missing; validity, confidence, and original capture age are supplied separately to quiet landing control. Nominal action history is retained while the selected actuator delay is applied. In a noisy viewer, height markers show the delivered map: red means missing and green/yellow indicates confidence. Keyboard control and standing/policy handover remain unchanged.

Teacher and quiet teacher additionally support `--noise-profile front_only_clean`. It uses the same front sector, mounted sensor position, and near-range rule as `front_only`, but directly masks the current clean teacher height scan. Proprioception and retained height values are unchanged; there is no synthetic noise, dropout, capture/reprojection delay, or action delay. Cells outside the sector have zero height/validity/confidence; current valid cells have confidence one and capture age zero. It is available in `deploy_mujoco.py` and `deploy_quiet_landing.py`; student entry points retain their existing profile sets. Optional JSON overrides for this preset accept only `front_sensor_offset_m` and `lidar_horizontal_fov_deg`; `--noise-level` must be omitted or zero.

| Profile | Main effect |
| --- | --- |
| `teacher` | Clean ideal observations |
| `student-v0` | Synthetic deployment noise preset |
| `student-v1` | Reference checkpoint's saved perception settings |
| `front_only` / `front_only_30cm` | Front sector with saved sensor noise and delays; no added 30 cm blind zone |
| `front_only_clean` | Same front sector, instantaneous clean teacher observations |
| `lidar_fov` | LiDAR visibility restrictions |
| `height` | Height noise, bias and artifacts |
| `pose` | Position, heading and tilt estimation errors |
| `dropout` | Point/packet loss, bursts and occlusion |
| `latency` | Map update timing, jitter and delivery delay |
| `proprio` | Gyroscope and joint measurement errors |
| `action_delay` | Action execution delay |
| `quality_reference` | Height/pose errors with synthetic confidence errors disabled |
| `confidence` | Height/pose errors plus confidence errors |
| `stress` | Stronger combined sensor disturbances |

Aliases: `clean` → `teacher`, `realistic` → `student-v0`, `train` → `student-v1`. The teacher/quiet entries accept all rows. V1 students accept all except `front_only_clean`; V0 students accept only `teacher`, `student-v0`, `student-v1`, `lidar_fov`, and `stress` (plus their aliases).

Export a student without starting a rollout:

```bash
python deploy/deploy_mujoco/deploy_student_v1.py \
  --checkpoint deploy/models/student_v3_5/model_21000.pt \
  --export-only --export-dir logs/evaluation/student_v3_5_export
```

Keep the exported policy together with its observation contract and deployment/provenance JSON files. Ordinary teacher MLPs can use `scripts/tools/export_onnx_fast.py`; the complete residual teacher requires its dual-input interface.

## Models, output, and validation scope

The release includes **14 selected full training checkpoints** under `deploy/models/`. Selection covers source baselines, final saved states, and intermediate points with fixed evaluation or existing export provenance. It does not establish that the largest iteration number or a selected model is best. See the [model notes](deploy/models/README.md) and machine-readable [catalog](deploy/models/catalog.json) for exact paths, hashes, origins, and selection reasons.

Training runs belong in `logs/rsl_rl/`; evaluations, plots, videos, and analysis belong in `logs/evaluation/<experiment>/`. Training logs and accumulated run outputs are excluded from publication. Keep temporary worktrees, source copies, and one-off diagnostic scripts outside the project.

Validation for this integration recorded **96 passing CPU tests and one pre-existing confidence/pose-geometry assertion failure** (`tests/test_student_v1.py::SensorTests::test_pose_quality_depends_on_measured_geometry`). A real Isaac smoke test loaded checkpoint 46400 into four environments and executed eight policy steps with zero optimizer updates, confirming the observation groups, 70 reward terms, finite outputs, unchanged actor weights, and unchanged older-task tuning. These are interface and short-run checks; they do not certify full-terrain performance or real-robot operation.

For a focused CPU interface check:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 \
python -m pytest -q -p no:cacheprovider \
  tests/test_distill_student_entry.py \
  tests/test_student_profiles.py \
  tests/test_student_v3_5.py
```

## Attribution and licenses

This project derives from [DeepRoboticsLab/rl_training](https://github.com/DeepRoboticsLab/rl_training), with the BSD 3-Clause terms in [LICENSE](LICENSE), and incorporates robot_lab work by Ziqi Fan under [LICENSE-robot_lab](LICENSE-robot_lab). The IdleStopMotion integration credits the pinned frider635-beep source above. Retain the existing per-file copyright and license notices. Isaac Lab, Isaac Sim, and robot assets remain subject to their respective upstream licenses.
