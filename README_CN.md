# M20 强化学习与学生蒸馏

[English](README.md) | 简体中文

基于 Isaac Lab 的 M20 轮足机器人训练工程，包含 PPO 教师训练、带噪感知的学生蒸馏，以及 MuJoCo 回放。

本次发布仓库为 [Chenorigin/rl_training](https://github.com/Chenorigin/rl_training)，Git 标签为 **`v0`**，Python 包版本为 **`0.1.0`**。项目继承自 [DeepRoboticsLab/rl_training](https://github.com/DeepRoboticsLab/rl_training)。以下命令均在仓库根目录执行。

## 安装与机器人资产

先准备可用的 Isaac Lab 环境；本仓库采用 editable 安装，不会自动安装 Isaac Sim。已使用的环境组合为 Python 3.11、Isaac Sim 5.1.0、Isaac Lab 2.3.2，训练依赖 `rsl-rl-lib==5.0.1`。

```bash
git clone --branch v0 --depth 1 --recurse-submodules https://github.com/Chenorigin/rl_training.git
cd rl_training

# 激活你已经安装好 Isaac Lab 的 Python/Conda 环境。
python -m pip install -e source/rl_training
```

已有 checkout 可执行 `git submodule update --init --recursive` 补齐 `deep_robotics_model/`。机器人资产由该上游子模块管理，不能只复制 Python 源码后直接启动训练。

楼梯教师及其衍生任务会检查 M20 USD 是否存在。在新机器上显式指定实际资产路径：

```bash
export RL_TRAINING_M20_USD_PATH=/absolute/path/to/M20.usd
```

USD 引用的网格、材质等资产也必须完整。MuJoCo 使用独立的 M20 MJCF，通过 `--model /absolute/path/to/M20.xml` 指定，并保留其引用的网格文件。MuJoCo 回放需要环境中可导入 `mujoco`、PyTorch、NumPy 和 TensorDict。

如遇 Conda C++ 运行库不兼容，可在目标环境内执行 `python scripts/tools/setup_conda_runtime.py`，然后重新激活该环境。开始训练前检查整机内存、整卡显存及输出盘余量，并据此调整环境数量。

## 目录与任务入口

| 路径 | 用途 |
| --- | --- |
| `source/rl_training/rl_training/` | M20 资产、Gym 注册、环境入口及配置 |
| `source/rl_training/task_support/<slug>/` | 任务专用的奖励、感知、PPO/蒸馏算法、runner 和模型辅助程序 |
| `source/rl_training/m20_student_v0/`、`m20_student_v1/` | 保持历史 checkpoint 类路径可加载的兼容导入入口 |
| `configs/distillation/` | 学生蒸馏 JSON 配置 |
| `scripts/reinforcement_learning/rsl_rl/` | 通用 `train.py`、`play.py` 和统一蒸馏入口 `distill_student.py` |
| `scripts/tools/` | 环境列表、配置比较、模型导出、评测和录屏工具 |
| `deploy/deploy_mujoco/` | 教师/学生回放入口、适配器与地形 XML |
| `deploy/models/` | 发布时保留的精选模型、契约和来源清单 |
| `tests/` | 接口、感知、初始化、恢复及模型回归测试 |
| `docs/` | 操作指南及 GaitRefinement-v2 必要回放输入 |
| `logs/` | 本地训练、TensorBoard、评测与视频输出，不上传发布仓库 |
| `deep_robotics_model/` | 上游机器人资产子模块 |

每个任务的 `slug` 对应三个固定入口：

```text
source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/
├── mdp/<slug>_env.py
└── config/wheeled/deeprobotics_m20/
    ├── <slug>_env_cfg.py
    └── agents/<slug>_ppo_cfg.py
```

`_env.py` 指定环境实现，`_env_cfg.py` 定义环境配置，`_ppo_cfg.py` 是统一命名的训练配置入口。学生任务在该入口中仍使用蒸馏算法。专用实现集中到 `task_support/<slug>/`，继承和共享代码按原有接口复用。

当前注册 **13 个 Gym 任务**，以 [任务注册表](source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/__init__.py) 为准：

| Gym ID | slug | 用途 |
| --- | --- | --- |
| `Flat-Deeprobotics-M20-v0` | `flat` | 平地速度跟踪 |
| `Rough-Deeprobotics-M20-v0` | `rough` | 粗糙地形速度跟踪 |
| `Rough-Deeprobotics-M20-PreTeacher-v0` | `pre_teacher` | 前置教师训练 |
| `Rough-Deeprobotics-M20-StairTeacher-v0` | `stair_teacher` | 楼梯教师 |
| `Rough-Deeprobotics-M20-IdleStopMotion-v1` | `idle_stop_motion_v1` | 冻结驻足基座上的运动残差教师 |
| `Rough-Deeprobotics-M20-StairTrot-v1` | `stair_trot_v1` | 上楼对角同步 trot，冻结完整教师以保护平地轮行 |
| `Rough-Deeprobotics-M20-StairStudent-v0` | `stair_student` | 扩展感知输入的 PPO 微调任务 |
| `Rough-Deeprobotics-M20-QuietLanding-v0` | `quiet_landing` | 轻盈落地教师 |
| `Rough-Deeprobotics-M20-GaitRefinement-v2` | `gait_refinement_v2` | 驻足、航向与回放保留训练 |
| `Rough-Deeprobotics-M20-Student-v0` | `student_v0` | v0 学生蒸馏 |
| `Rough-Deeprobotics-M20-Student-v1` | `student_v1` | v1 感知与历史观测蒸馏 |
| `Rough-Deeprobotics-M20-Student-v3` | `student_v3` | v3 学生蒸馏 |
| `Rough-Deeprobotics-M20-Student-v3.5` | `student_v3_5` | 完整运动残差教师的学生蒸馏 |

在 Isaac Lab 环境中运行 `python scripts/tools/list_envs.py` 可查看实际注册的环境。

新建上楼 trot 任务的训练、续训和回放命令见 [StairTrot-v1 操作指南](docs/m20_stair_trot_v1.md)。
训练由用户手动启动；代码交付时只执行无优化器更新的检查，尚无新训练权重或步态达标结论。

## 教师训练与 Isaac 回放

`train.py` 和 `play.py` 的参数使用下划线，例如 `--num_envs`、`--max_iterations`、`--load_run`。

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Rough-Deeprobotics-M20-StairTeacher-v0 \
  --num_envs 64 --device cuda:0 --headless

python scripts/reinforcement_learning/rsl_rl/play.py \
  --task Rough-Deeprobotics-M20-StairTeacher-v0 --num_envs 1 \
  --checkpoint deploy/models/stair_teacher/model_214700.pt
```

完整恢复包含权重、优化器及迭代状态，可传显式 checkpoint 路径，也可用 `--load_run <run> --checkpoint model_N.pt` 从对应任务日志目录选择：

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Rough-Deeprobotics-M20-StairTeacher-v0 --resume \
  --checkpoint deploy/models/stair_teacher/model_214700.pt \
  --num_envs 64 --max_iterations 1000 --headless
```

`--max_iterations` 是本次追加的训练迭代数。通用 PPO 仅初始化权重时使用 `--init_actor_from <checkpoint>`，需要时同时指定形状兼容的 `--init_critic_from <checkpoint>`；这与完整 `--resume` 是不同的操作。上述环境数是命令示例，实际配置应按运行资源确定。

## IdleStopMotion 独立教师

`Rough-Deeprobotics-M20-IdleStopMotion-v1` 集成自 [frider635-beep/Deeprobotics_M20_stair 的固定提交 `8fb5d9aa5809363dd9a2cccacf0b24ee6f542d81`](https://github.com/frider635-beep/Deeprobotics_M20_stair/tree/8fb5d9aa5809363dd9a2cccacf0b24ee6f542d81)。该版本训练源码的指纹与发布的 46400 checkpoint 内嵌记录一致，任务实现与参数位于 `task_support/idle_stop_motion_v1/`。

模型保留冻结的 `244→512→256→128→16` 基座，在其上训练 `247→128→64→16` 有界运动残差。残差输入为 244 维 policy 和 3 维原始 `motion_control`；critic 使用 285 维特权观测和相同的 3 维原始命令。policy 中的有效控制命令与原始命令分别处理，原始零命令严格关闭残差。

新训练默认加载 `deploy/models/idle_stop_v1/model_257100_idle_stop.pt`，校验上游记录的完整 checkpoint SHA256，冻结基座及采样分布，仅更新残差和 critic。critic 默认随机初始化，并保留上游 20 次更新的 warmup。下楼引导默认保持 20000 次迭代，再经过 10000 次迭代退火；这些值由任务 PPO 配置管理，无需旧项目的 profile CLI 参数。

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Rough-Deeprobotics-M20-IdleStopMotion-v1 \
  --num_envs 64 --device cuda:0 --headless

python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Rough-Deeprobotics-M20-IdleStopMotion-v1 --resume \
  --checkpoint deploy/models/idle_stop_motion_v1/idle_stop_motion_v1_46400.pt \
  --num_envs 64 --device cuda:0 --headless
```

完整恢复严格加载 actor、critic、优化器、保存的引导课程及计数，不会重新初始化基座。Isaac 回放使用相同任务的 `play.py --checkpoint ...`，并使用任务自身生成的命令。通用 `--keyboard` 观测覆盖不能保持有效命令与原始 `motion_control` 的配对；需要交互控制时使用下文的 MuJoCo 教师回放入口。

该双输入残差教师不支持普通单输入 JIT/ONNX 导出。`play.py` 会明确报告跳过该导出，并用完整 actor 回放；不能只导出冻结的基座来代替完整残差模型。

## 学生蒸馏

`distill_student.py` 使用连字符参数，例如 `--num-envs`、`--max-iterations`、`--teacher-checkpoint`；它的 `--resume` 直接接 checkpoint 路径。v0、v1、v3、v3.5 共用该入口，`distill_student_v1.py` 仅保留旧命令兼容。

先执行 CPU 预检，检查模型、契约、初始化一致性与资源预算，不启动 Isaac 或执行优化器更新：

```bash
python scripts/reinforcement_learning/rsl_rl/distill_student.py \
  --task Rough-Deeprobotics-M20-Student-v3.5 \
  --teacher-checkpoint deploy/models/idle_stop_motion_v1/idle_stop_motion_v1_46400.pt \
  --num-envs 64 --device cpu --preflight-only
```

开始训练时使用训练设备，去掉 `--preflight-only` 并加上 `--headless`：

```bash
python scripts/reinforcement_learning/rsl_rl/distill_student.py \
  --task Rough-Deeprobotics-M20-Student-v3.5 \
  --teacher-checkpoint deploy/models/idle_stop_motion_v1/idle_stop_motion_v1_46400.pt \
  --num-envs 64 --device cuda:0 --headless

python scripts/reinforcement_learning/rsl_rl/distill_student.py \
  --resume deploy/models/student_v3_5/model_21000.pt \
  --device cuda:0 --headless
```

恢复训练保持 checkpoint 的任务、环境数、观测、数据与课程配置；改变配置或仅初始化学生权重时建立新 run，并使用 `--init-student`。v3.5 学习完整残差教师，保留 v3 的学生输入和网络布局。

详细说明见 [v0 训练与恢复](docs/m20_student_v0.md)、[v1 感知与评测](docs/m20_student_v1.md)、[v3.5 教师与反馈](docs/m20_student_v3_5.md)。

## MuJoCo 回放与学生导出

教师、v0 学生和 v1/v3/v3.5 学生分别使用对应部署入口。MuJoCo 参数采用连字符：

StairTrot-v1 权重也使用 `deploy_mujoco.py`，自动加载完整残差和六维控制状态；具体命令见 [StairTrot-v1 MuJoCo 部署](docs/m20_stair_trot_v1.md#mujoco-部署)。

```bash
python deploy/deploy_mujoco/deploy_mujoco.py \
  --checkpoint deploy/models/idle_stop_motion_v1/idle_stop_motion_v1_46400.pt \
  --model /absolute/path/to/M20.xml --terrain ascent --viewer

python deploy/deploy_mujoco/deploy_student_v0.py \
  --checkpoint deploy/models/student_v0/model_21000.pt \
  --model /absolute/path/to/M20.xml --terrain ascent \
  --noise-profile student-v1 --viewer

python deploy/deploy_mujoco/deploy_student_v1.py \
  --checkpoint deploy/models/student_v3_5/model_21000.pt \
  --model /absolute/path/to/M20.xml \
  --terrain-xml deploy/deploy_mujoco/terrains/all_terrain.xml \
  --noise-profile student-v1 --viewer
```

交互键：`Z` 起立、`C` 接入策略、`W/S/A/D/Q/E` 控制、`R` 阻尼、`H` 显示高程、`Esc` 退出。`student-v1` profile 使用模型保存的训练感知参数，`teacher` 是理想完整观测参考；它们不是实测传感器精度保证。具体操作见 [MuJoCo 指南](docs/m20_student_v0_mujoco.md) 和 [地形指南](deploy/deploy_mujoco/terrains/README_CN.md)。

学生可单独导出，无需加载 MJCF 或启动物理。输出目录须为新目录：

```bash
python deploy/deploy_mujoco/deploy_student_v1.py \
  --checkpoint deploy/models/student_v3_5/model_21000.pt \
  --export-only --export-dir logs/evaluation/student_v3_5_export
```

导出后的 TorchScript、`observation_contract.json` 和 `deployment.json` 应一起保存。普通 244D 教师 MLP 可用 `scripts/tools/export_onnx_fast.py`；上述残差教师不适用这个单输入导出器。

## 发布模型与运行数据

`deploy/models/` 保留 **14 个完整 checkpoint**。完整文件名、来源、SHA256、选择理由及验证结果见 [模型清单](deploy/models/README.md) 和 [catalog.json](deploy/models/catalog.json)。V3 保留 11500、20000，V3.5 保留 20000、21000；入选或保存编号较大不代表性能最佳。

完整学生 checkpoint 内嵌观测契约、感知配置和 provenance，可独立于原 `logs/` 目录加载。训练日志、TensorBoard、视频与其它实验输出不属于发布内容，也不上传仓库；新的输出仍写入本地 `logs/rsl_rl/` 或 `logs/evaluation/<实验名>/`。

`docs/m20_idle_stop_257100_20261012/reproduction_manifest.json` 及其引用的 **12 个 NPZ** 是 GaitRefinement-v2 默认回放训练输入。它们必须随项目保留，不能当作历史报告删除。

## 验证状态

最近记录的 CPU 回归结果为 **96 项通过、1 项已知失败**。失败位于 `tests/test_student_v1.py::SensorTests::test_pose_quality_depends_on_measured_geometry`：当地图 age 为 1 时，置信度实现仍返回非零值，而测试期望零。该失败尚未修复，不应把测试集描述为全部通过。

可在具备依赖的环境中运行 CPU 回归：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 \
python -m pytest -q -p no:cacheprovider tests
```

测试包含小型 CPU 优化和 MuJoCo 物理用例，并非全部都是静态检查。IdleStopMotion 另完成过真实 Isaac **4 个环境、8 步**的推理冒烟及完整 checkpoint 恢复检查，未执行优化器更新；这些结果验证接线与有限输出，不能证明楼梯通行性能、长期稳定性或真机可用性。

## 来源与许可证

- 项目基础来自 [DeepRoboticsLab/rl_training](https://github.com/DeepRoboticsLab/rl_training)，保留其版权声明与 [BSD-3-Clause 许可证](LICENSE)。
- robot_lab / Isaac Lab 衍生代码保留原作者及文件内声明，另见 [Apache-2.0 许可证](LICENSE-robot_lab)。
- IdleStopMotion 的模型、PPO、奖励、命令与日志实现来自前述 frider635-beep 仓库的固定提交，保留对应文件版权及来源指纹。PPO 实现包含 ETH Zurich / NVIDIA 的 RSL-RL 来源声明。
- `deep_robotics_model`、Isaac Sim、Isaac Lab、RSL-RL 等依赖与资产遵循其各自上游许可证；具体文件以保留的版权头和 SPDX 声明为准。
