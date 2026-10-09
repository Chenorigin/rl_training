# M20 Pro 楼梯教师策略：训练、回放和 MuJoCo 部署

以下命令从**仓库根目录**执行，使用 Conda 环境 `m20_wzh`。楼梯任务 ID 为 `Rough-Deeprobotics-M20-StairTeacher-v0`，轮式预训练任务 ID 为 `Rough-Deeprobotics-M20-PreTeacher-v0`。Isaac Lab 的训练与回放需要 M20 USD；MuJoCo 部署需要 M20 XML 与教师 checkpoint。`train.py` 和 `play.py` 会优先导入此仓库的 `source/rl_training`；这台机器的 Conda 环境里还有另一份旧版同名包，直接使用它会导致教师任务未注册。

## 环境与模型路径

```bash
cd /home/ubuntu/cyq_shixi_projects/rl_training
```

楼梯教师任务默认使用本机的 `/home/ubuntu/桌面/m20_1/M20_perception_Lidar_rl_2/M20_perception_Lidar_rl/deep_robotics_model/M20/usd/M20.usd`，无需每次 `export RL_TRAINING_M20_USD_PATH`。模型搬家时才设置该环境变量覆盖默认路径。

工作区本机的 `.vscode/settings.json` 已指定 `m20_wzh` 解释器。VS Code 选中该解释器后，**新建的集成终端**通常会自动激活；旧终端或 VS Code 外的 shell 不受影响。可用 `which python` 确认，预期为 `/home/ubuntu/miniconda3/envs/m20_wzh/bin/python`；若不是，再运行 `conda activate m20_wzh`。MuJoCo 的 `--model` 使用 XML：

```text
/home/ubuntu/桌面/m20_1/M20_perception_Lidar_rl_2/M20_perception_Lidar_rl/deep_robotics_model/M20/mjcf/M20.xml
```

## 轮式预训练 pre_teacher

先用平地与少量缓坡、云深处 M20 的基础速度跟踪和稳定性奖励训练轮式运动。该任务关闭原始配置里会鼓励平地抬轮的转向腾空、对角接触步态奖励及纯转向滑移惩罚。actor、critic、动作维度与楼梯教师一致；预训练日志单独写入 `logs/rsl_rl/deeprobotics_m20_pre_teacher/`。

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Rough-Deeprobotics-M20-PreTeacher-v0 \
  --headless \
  --max_iterations 20000
```

使用预训练权重初始化楼梯教师时，只加载 actor，critic、优化器和迭代计数重新开始。将下方路径换成预训练后实际生成的 checkpoint；`--init_actor_from` 不能与 `--resume` 同用。

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Rough-Deeprobotics-M20-StairTeacher-v0 \
  --headless \
  --init_actor_from logs/rsl_rl/deeprobotics_m20_pre_teacher/<预训练运行目录>/model_<迭代数>.pt \
  --max_iterations 150000
```

预训练先看平地、缓坡上的速度跟踪、跌倒情况和轮子接触，再选择迁移用的 checkpoint；达到迭代数不等于已经学会稳定移动。

`2026-10-01_12-22-19` 的后退失败分析、奖励修复及检查命令见
[训练分析报告](docs/training_analysis/2026-10-01_stair_teacher/REPORT_CN.md)。
修复后的 stair_teacher 增加前进命令下楼梯速度不足的成本，抬腿采用每级有上限的增量信用；
原有“左右交替迈到相邻台阶”的成功奖励仍然保留。
TensorBoard 的 `Curriculum/step_completion/up_targets_started` 表示建立目标次数，
`up_lift_credit` 表示准备抬腿信用，`up_retreat_fraction` 表示有效前进楼梯目标期间实际后退的比例。
必须结合 `up_successes`、`up_rear_successes` 和上楼地形 level 判断；总奖励或抬腿信用单独上涨不代表完成爬楼。

## 训练 train

**2026-10-08 gait_v3 退化修复后的续训方式**见
[诊断、验证和推荐命令](docs/gait_v3_regression/README_CN.md)。建议从已验证能爬楼的
`model_199998.pt` 初始化 actor 与 critic，重置优化器，并启用已接入的参考策略约束；
先做1000轮观察，不再覆盖为旧的 `learning_rate=1e-4` / `schedule=adaptive`。

从头训练 150000 次 PPO 迭代：

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Rough-Deeprobotics-M20-StairTeacher-v0 \
  --headless \
  --max_iterations 150000
```

从已有 checkpoint 继续训练，例如再训练 150000 次迭代：

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Rough-Deeprobotics-M20-StairTeacher-v0 \
  --headless \
  --resume \
  --load_run 2026-09-29_19-34-19 \
  --checkpoint model_39300.pt \
  --max_iterations 150000
```

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Rough-Deeprobotics-M20-StairTeacher-v0 \
  --headless \
  --init_actor_from logs/rsl_rl/deeprobotics_m20_pre_teacher/2026-09-30_18-24-38/model_19999.pt \
  --max_iterations 150000
```

训练输出位于 `logs/rsl_rl/deeprobotics_m20_stair_teacher/<时间戳>/`。当前奖励配置已调整；旧 checkpoint 使用的是旧奖励训练结果，继续训练属于从旧权重微调。

| 常用参数 | 作用 |
| --- | --- |
| `--task` | Gym 任务 ID，使用本项目楼梯教师任务时必填。 |
| `--headless` | 不打开 Isaac Sim 图形窗口。 |
| `--max_iterations N` | 本次训练的 PPO 迭代数；当前 RSL-RL 恢复训练时也会额外执行 N 次。 |
| `--num_envs N` | 覆盖并行环境数。 |
| `--seed N` | 随机种子；`-1` 随机选择。 |
| `--device cuda:0` | 指定仿真设备。 |
| `--run_name NAME` | 给新日志目录加后缀。 |
| `--resume --load_run RUN --checkpoint FILE` | 从指定运行目录中的权重恢复。 |
| `--init_actor_from PATH` | 从 checkpoint 仅初始化 actor；新建 critic、优化器和训练迭代计数。 |
| `--init_critic_from PATH` | 配合 actor 初始化继承 critic 权重，优化器仍重置。 |
| `--preserve_actor_from PATH` | 指定冻结的能力参考 actor，仅用于 StairGuardedPPO；actor 初始化默认自动设置参考。 |
| `--logger tensorboard` | 选择 TensorBoard 日志。 |
| `--video --video_length N --video_interval N` | 训练期间定期录制视频，需启用相机。 |
| `--distributed` | 多 GPU／多节点训练。 |

## 回放 play（Isaac Lab）

回放指定的教师 checkpoint：

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task Rough-Deeprobotics-M20-StairTeacher-v0 \
  --num_envs 1 \
  --checkpoint logs/rsl_rl/deeprobotics_m20_stair_teacher/2026-09-29_19-34-19/model_39300.pt
```

键盘控制回放：在上面命令末尾追加 `--keyboard`。回放脚本会自动把环境数设为 1，并把速度命令观测接到 Isaac Lab 键盘控制器。无窗口回放可加 `--headless`。

| 常用参数 | 作用 |
| --- | --- |
| `--checkpoint PATH` | 直接指定 checkpoint 文件，支持相对仓库根目录的路径。 |
| `--load_run RUN` | 未给 `--checkpoint` 时，从该训练运行目录选取权重。 |
| `--num_envs N` | 回放环境数；脚本默认 50，`--keyboard` 时强制 1。 |
| `--keyboard` | 用 Isaac Lab 键盘控制速度命令；需要图形窗口。 |
| `--real-time` | 尽可能按实际时间运行。 |
| `--headless` | 无图形窗口回放；不适合键盘控制。 |
| `--video --video_length N` | 录制指定步数的视频。 |
| `--seed N`、`--device cuda:0` | 覆盖随机种子与仿真设备。 |

## MuJoCo 部署 deploy_mujoco

交互式楼梯仿真：

```bash
python deploy/deploy_mujoco/deploy_mujoco.py \
  --model "/home/ubuntu/桌面/m20_1/M20_perception_Lidar_rl_2/M20_perception_Lidar_rl/deep_robotics_model/M20/mjcf/M20.xml" \
  --checkpoint logs/rsl_rl/deeprobotics_m20_stair_teacher/2026-10-02_00-12-17/model_30700.pt \
  --terrain-xml deploy/deploy_mujoco/terrains/stairs_ascent.xml --viewer
```

在 MuJoCo 窗口中：`Z` 起立，`C` 切换到教师策略，`W/S` 前后、`A/D` 左右、`Q/E` 转向；`H` 切换高程扫描显示，`X` 趴下，`R` 阻尼模式，`Esc` 退出。非交互模式不加 `--viewer`，默认执行 1000 个 50 Hz 控制步；可用 `--steps` 覆盖。`--viewer --autoplay` 则在窗口中执行固定速度命令回放。

Ubuntu X11 下，仿真窗口中的移动键按住持续生效，松开或窗口失去焦点后命令归零；支持组合按键。MuJoCo 的窗口回调只提供首次按下，程序通过 `keyboard_input.py` 查询按住状态，不依赖键盘连发。终端输入仍依赖系统连发并在超时后归零；其他桌面会话若不支持按住状态查询，可使用聚焦终端输入。修改部署程序后需重新启动仿真。

| 参数 | 作用；默认值 |
| --- | --- |
| `--model PATH` | **必填**，M20 MuJoCo XML。 |
| `--checkpoint PATH` | 教师 checkpoint；默认使用上方示例的 `model_39300.pt`。 |
| `--terrain-xml PATH` | 选择独立地形 XML；与 `--terrain` 互斥。地形参数及单位写在各 XML 的注释中。 |
| `--terrain NAME` | 内置地形快捷名：`flat/ascent/descent/slope_up/slope_down/rough/obstacles/hurdles/mixed`；均对应独立 XML，省略地形选项时使用 `ascent`。 |
| `--stair-height M` | 覆盖楼梯 XML 的 `step_height`；省略则读取 XML。仅用于上/下楼梯。 |
| `--tread-depth M` | 覆盖楼梯 XML 的 `tread_depth`；省略则读取 XML。 |
| `--stair-count N` | 覆盖楼梯 XML 的 `step_count`；省略则读取 XML。 |
| `--stair-start M` | 覆盖楼梯 XML 的 `start_x`；省略则读取 XML。 |
| `--viewer` | 打开交互式窗口，默认从趴下姿态启动。 |
| `--autoplay` | 与 `--viewer` 同用时改成固定命令自动回放。 |
| `--command-x/y/yaw VALUE` | 自动回放时的前后、横向、转向速度命令；默认 `0.5/0/0`。交互模式使用键盘命令。 |
| `--steps N` | 限制控制步数；交互模式默认一直运行到退出，自动模式默认 1000 步。 |
| `--print-every N` | 每隔 N 步输出状态；默认 `10`，对应0.2秒仿真时间。策略控制周期为0.02秒。 |
| `--stop-on-fall` | 跌倒后停止自动回放。 |
| `--output PATH.npz` | 保存压缩轨迹与运行摘要。 |

### 地形切换与调参

地形文件在 [deploy/deploy_mujoco/terrains](deploy/deploy_mujoco/terrains/README_CN.md)。切换时替换上面命令的 `--terrain-xml` 路径即可，例如 `slope_up.xml` 或 `mixed.xml`；机器人仍由 `--model` 指定。

每个 XML 的 `<custom><numeric ... data="..."/></custom>` 保存该地形参数；改 `data` 并重新启动，碰撞体会重新生成。长度单位为米，`*_deg` 为角度；坐标系为世界 `+X` 前、`+Y` 左、`+Z` 上。启动输出和 `.npz` 摘要记录实际 XML 路径及生效参数，显式命令行覆盖优先于 XML。

普通模板通过部署程序生成几何；`custom.xml` 则展示直接编写原生 MJCF `worldbody/geom` 的方式。出生高度、跌倒判定和 187 点高程观测均查询实际碰撞地形，支持楼梯、坡面和障碍。Actor 输入仍为 57+187 维。

`cleared_terrain/cleared_stairs` 仅表示末帧四轮越过路线终点、位于路线宽度内且本轮未跌倒；未检查绕行，不能用它独立判断策略是否完成各障碍。

地形预览与数值检查记录：[docs/mujoco_terrains](docs/mujoco_terrains/README_CN.md)。

`deploy/deploy_real/` 目前是预留目录，没有真机部署程序或命令。

## 楼梯步态微调与量化评测

当前楼梯教师已修正逐阶登记与前后腿落脚历史、前腿承载/摆动夹角和下楼后腿前倾门控，
并加入独立后腿准备奖励及专门TensorBoard指标；下一轮从
`2026-10-06_22-06-56_stair_resume/model_199998.pt` 初始化actor。
根因、奖励逻辑、指标、实际检查和训练建议见 [10月8日步态修正](docs/gait_v3_20261008/README_CN.md)。主要上楼完成奖励现按安全相邻换侧转换支付，并分别监控前后腿的跳阶比例、同阶比例和实际转换次数。
上楼方向偏移惩罚继续保留；之前的回退操作记录见 [方向偏移修正](docs/ascent_direction_20261008/README_CN.md)。

Student框架新增任务`Rough-Deeprobotics-M20-StairStudent-v0`，actor/critic输入均为618维，另有244维teacher标签查询组。它目前提供观测和PPO配置，尚未包含蒸馏训练入口；接口、验证和待讨论流程见 [Student框架](docs/stair_student_20261008/README_CN.md)。

楼梯 teacher 的扫描门控、逐阶事件账本、奖励函数、步态约束和课程指标统一维护在
`source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/stair_teacher.py`。
步态包络参数的唯一来源为该文件的 `TUNING`；奖励启用、权重和实体绑定由
`config/wheeled/deeprobotics_m20/stair_teacher_env_cfg.py` 配置。旧的 `stair_ascent.py`
和 `gait_refinement.py` 仅兼容历史保存配置的导入路径，不再包含实现。

最终训练流程以 `pre_teacher` 的 actor 为初始化，在同一次 StairTeacher 训练中学习
地形任务、交替迈步、姿态约束和多地形泛化，训练过程中使用地形课程和域随机化。
开发期间的 resume 用于验证和调整候选配置；最终质量需要以固定测试集上的任务完成、
落脚顺序、折叠程度和速度跟踪验证，文件合并本身不保证训练收敛。
合并与检查记录见 [统一模块记录](docs/stair_teacher_merge_20261006/README_CN.md)。

上楼前腿的支撑/摆动过度折叠奖励与测试记录：[前腿修正](docs/front_fold_20261003/README_CN.md)。

上楼逐阶交替、下楼承重后腿折叠、Q/E紧凑转向的奖励修正、微调命令和验证范围见 [步态修正记录](docs/gait_refinement_20261002/README_CN.md)。新配置需重新启动训练才能生效；pre_teacher配置保持原样。

`scripts/tools/evaluate_stair_gait.py` 用固定 checkpoint 在上楼、下楼、左右转向和平地采集落脚事件、伸展、摆腿幅度、速度误差及能耗，不训练策略。必填 `--checkpoint`、`--model`、`--output-dir`；可选 `--cases`、`--steps`、`--stair-height`、`--tread-depth`、`--stair-count`、`--min-extension`。评测轨迹为未审核留样，不能直接当作合格AMP专家数据。
