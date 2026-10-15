# M20 locomotion training

This workspace is a M20-only adaptation of [DeepRoboticsLab/rl_training](https://github.com/DeepRoboticsLab/rl_training), based on Isaac Lab. It registers two M20 environments:

| Task ID | Purpose |
| --- | --- |
| `Flat-Deeprobotics-M20-v0` | Flat-ground velocity tracking |
| `Rough-Deeprobotics-M20-v0` | Rough-terrain velocity tracking |

The M20 asset is loaded from the `deep_robotics_model` submodule. The submodule is kept intact because it contains the required M20 USD; its other robot assets are not registered by this workspace.

## Install

Use an Isaac Sim 5.1.0 / Isaac Lab 2.3.2 environment with Python 3.11, then run from this repository:

```bash
git submodule update --init --recursive
python -m pip install -e source/rl_training
python scripts/tools/list_envs.py
```

If your Conda environment has an incompatible C++ runtime, run `python scripts/tools/setup_conda_runtime.py`, then deactivate and reactivate the environment. The script verifies the required `CXXABI` symbol before installing environment activation hooks.

## Train and play

```bash
python scripts/reinforcement_learning/rsl_rl/train.py --task=Rough-Deeprobotics-M20-v0 --headless
python scripts/reinforcement_learning/rsl_rl/play.py --task=Rough-Deeprobotics-M20-v0 --num_envs=10
```

For a short training check, append `--num_envs=16 --max_iterations=1`. Check GPU and system memory availability before starting a full training run. The flat-ground task can be selected with `--task=Flat-Deeprobotics-M20-v0`.

Training configuration: [`rough_env_cfg.py`](source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/rough_env_cfg.py), [`flat_env_cfg.py`](source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/flat_env_cfg.py), and [`rsl_rl_ppo_cfg.py`](source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/agents/rsl_rl_ppo_cfg.py). Common terrain, curriculum, and reward terms live under [`velocity`](source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/).

## Export and compare

```bash
python scripts/tools/export_onnx_fast.py \
    --checkpoint_path logs/rsl_rl/deeprobotics_m20_rough/<run>/model_5000.pt \
    --robot m20 \
    --output_path exported/m20_policy.onnx

python scripts/tools/compare_runs.py \
    logs/rsl_rl/deeprobotics_m20_rough/<run1> \
    logs/rsl_rl/deeprobotics_m20_rough/<run2>
```

The ONNX export tool records M20 joint and action metadata by default. Check observation and action contracts against the deployment project before using a policy outside simulation.
