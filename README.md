# M20 Reinforcement Learning and Student Distillation

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

Each task has three consistently named entry files under `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/`:

```text
mdp/<task>_env.py
config/wheeled/deeprobotics_m20/<task>_env_cfg.py
config/wheeled/deeprobotics_m20/agents/<task>_ppo_cfg.py
```

Additional programs belong in `task_support/<task>/`. Shared implementations use inheritance or explicit imports. Student `_ppo_cfg.py` files are uniformly named training configuration entries; the student algorithms still perform distillation.

## Registered tasks

The [task registry](source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/__init__.py) defines these 12 Gym IDs. `python scripts/tools/list_envs.py` lists available environments in the Isaac Lab installation.

| Gym task ID | File prefix | Purpose |
| --- | --- | --- |
| `Flat-Deeprobotics-M20-v0` | `flat` | Flat-ground velocity tracking |
| `Rough-Deeprobotics-M20-v0` | `rough` | Rough-terrain velocity tracking |
| `Rough-Deeprobotics-M20-PreTeacher-v0` | `pre_teacher` | Preliminary teacher training |
| `Rough-Deeprobotics-M20-StairTeacher-v0` | `stair_teacher` | Stair teacher |
| `Rough-Deeprobotics-M20-IdleStopMotion-v1` | `idle_stop_motion_v1` | Motion residual over a frozen idle-stop baseline |
| `Rough-Deeprobotics-M20-StairStudent-v0` | `stair_student` | Stair student environment |
| `Rough-Deeprobotics-M20-QuietLanding-v0` | `quiet_landing` | Quiet-landing teacher |
| `Rough-Deeprobotics-M20-GaitRefinement-v2` | `gait_refinement_v2` | Idle/heading refinement with replay preservation |
| `Rough-Deeprobotics-M20-Student-v0` | `student_v0` | Sensor-limited v0 student |
| `Rough-Deeprobotics-M20-Student-v1` | `student_v1` | Sensor-limited v1 student with optional history |
| `Rough-Deeprobotics-M20-Student-v3` | `student_v3` | Front-sensor student with four-frame history |
| `Rough-Deeprobotics-M20-Student-v3.5` | `student_v3_5` | Student distilled from the complete motion-residual teacher |

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

Use `distill_student.py` for v0, v1, v3, and v3.5. The old `distill_student_v1.py` entry remains a compatibility wrapper. v3.5 learns the complete residual teacher while retaining the v3 student layout and four-frame history.

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

Play the complete motion-residual teacher:

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
