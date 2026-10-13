# M20 学生蒸馏 v1

v1 使用独立环境和蒸馏算法，与 v0 共用 `distill_student.py` 入口，由 `--task` 区分。由用户启动训练。教师沿用 `quiet_landing_v1/model_257100.pt`；可直接从教师扩展初始化，也可用 `--init-student` 加载学生权重，优化器和回放数据重新建立。

默认 actor 输入仍为 **57 + 187 × 3 = 618 维**。教师使用干净的 244 维观测。环境中的 `critic` 是与学生同源的观测占位组；本版本是行为蒸馏，没有训练价值网络，也没有 PPO 更新。

## 逐条实现

| 改进 | 已接入的行为 | 实现位置 |
|---|---|---|
| 观测对齐 | Isaac 与 MuJoCo 共用 `StudentObservationState`；固定网格顺序、高度基准、坐标系、时间和上一步下发动作的定义；checkpoint/JIT 带版本化契约 | `m20_student_v1/contract.py`、`perception.py`、`environment.py`、`student_v1_adapter.py` |
| 真实视场与历史地图 | 前、后雷达分别判断视场、距离、垂直范围及高程地形遮挡；每格选最新已送达有效观测，当前盲区可保留过去观测，按各格原始时间过期 | `perception.py` |
| 时序与噪声 | 噪声等级改变误差强度，地图周期、延迟和固定视场独立；空间相关高度、整图偏置、相对位置/航向偏置及连续漂移、成块和连续缺测、本体误差、动作延迟 | `configuration.py`、`perception.py`、`student_v1_env_cfg.py` |
| 一致的姿态误差 | 同一 roll/pitch 估计误差同时影响投影重力和地图坡度；地图与动作队列在单环境 reset 时一起清空，其他环境历史保持 | `perception.py`、`environment.py` |
| 有效性与置信度 | `m` 表示当前有未过期的有效来源；`c` 根据估计高程方差、地形梯度传播的相对位姿不确定性、边缘歧义和年龄计算；加入方差估计偏差及偶发过度自信 | `perception.py` |
| 教师到学生的执行课程 | 按 episode 选执行者；满足学生完成 episode 数和失败率门槛后，逐步降低教师比例；末段可全部由学生执行，教师继续提供标签 | `distillation.py`、`runner.py` |
| 难状态和方向覆盖 | 按行走阶段、局部地形、实际命令方向和地形等级联合分层；普通回放之外保留落地/恢复状态；命令显式覆盖前、后、横移、转向、混合和站立 | `distillation.py`、`student_v1_env_cfg.py` |
| 保留轻盈落地行为 | 腿与轮分别计算均值损失并加权，避免仅由维数决定权重；提高落地前、触地、恢复状态权重；继承原轻盈落地物理任务和诊断 | `distillation.py`、`student_v1_env_cfg.py` |
| 可选记忆 | 显式追加过去的带噪本体观测，reset 零填充；默认单帧 MLP | `contract.py`、`perception.py` |
| 独立验收 | 分因素、置信度成对对照及组合扰动；冻结模型/地形/已登记代码哈希；记录命令跟踪、通行、触地速度、每米触地能量、力峰值/冲量及原有滑移诊断 | `profiles.py`、`evaluate_student_v1.py` |

训练入口的数据流为：`gym.make(v1 task)` → 新环境的共享感知状态 → RSL-RL runner → `StudentV1Distillation`。MuJoCo 入口复用已有手动控制和 PD 基础设施，在独立模块中接入 v1 适配器。v0 与 v1 的算法、配置及 checkpoint 格式分别保留；`distill_student_v1.py` 只作为统一入口的兼容转发。

## 训练命令

统一入口支持 `Rough-Deeprobotics-M20-Student-v0/v1` 及对应的 `StudentV15-v0/v1` task。新训练省略 task 时保留 v0 默认；续训省略 task 时读取 checkpoint 的 task，显式指定不同 task 会报错。v0 不接受 v1 专用的配置、历史、学生初始化及教师比例结束值参数，并保留原有的 512 环境上限。

在项目根目录和原来的训练环境中执行：

```bash
conda activate m20_wzh
python scripts/reinforcement_learning/rsl_rl/distill_student.py \
  --task Rough-Deeprobotics-M20-Student-v1 \
  --teacher-checkpoint logs/rsl_rl/deeprobotics_m20_quiet_landing_v1/2026-10-10_00-48-58_quiet_landing_v1/model_257100.pt \
  --history-steps 4 \
  --num-envs 4096 \
  --max-iterations 10000 \
  --device cuda:0 \
  --headless \
  --run-name student_v1_history4
```

`--init-student` 接受具体 checkpoint 或 run 目录。目录会在启动时固定为最新完整的 `model_N.pt`，记录路径和 SHA256。只加载模型权重，重新建立优化器、回放和课程状态。省略该选项时从教师 actor 权重扩展初始化。当前实际检查使用了 v0 的 `model_29999.pt`。

输出目录为 `logs/rsl_rl/deeprobotics_m20_student_v1/<时间>_<run-name>/`，包含资源预算、观测契约、配置、来源记录、源代码快照、TensorBoard 和 v1 checkpoint。正常结束后导出 `exported/student_v1.pt` 及配套契约。

环境数可显式设置为 `--num-envs 4096`；入口不再设固定的 512 环境上限。实际启动仍检查整机内存、GPU 余量和输出盘空间。新训练默认 `algorithm.save_replay=false`：checkpoint 保存模型、教师、优化器、课程计数与来源，不重复落盘普通/关键回放；训练内存中的回放容量和采样方法保持原配置。预算由 `output_budget()` 根据实际保存策略计算。省略 `--init-student` 会从教师 actor 扩展初始化，新增有效性、置信度及历史输入列零初始化，并非随机初始化。

改变环境数会增加每轮采样数和优化 minibatch 数；教师与噪声课程仍按迭代计数，不能把不同环境数下的相同 iteration 当作相同采样预算。此修改未启动训练或测量 4096 环境的实际仿真峰值。

入口先检查实际模型、资源和磁盘，再启动 Isaac。磁盘预留不足会停止并报告，不会自动删除旧文件。使用 `--log-root` 可指定另一个日志盘。资源预算是工程预留，扩大环境数前应以实际峰值校准。

只想先做 CPU 权重/契约检查，可在上述命令中将 device 改为 `cpu` 并加 `--preflight-only`。它不启动仿真、不收集 rollout、不更新优化器。

首次启动前也可由你添加 `--validate-only --num-envs 4` 做 Isaac 推理接线检查；此模式会物理步进和写验证记录，但不训练。此次交付未运行这个模式。

## 配置、历史和续训

所有噪声与课程数值的权威默认台账为 `source/rl_training/m20_student_v1/configuration.py`。`--config <JSON>` 支持 `perception`、`training`、`history_steps` 三个顶层字段；只填写需要覆盖的字段。未知字段会报错。传感器位置、垂直视场、延迟等默认值是待实测校准的工程值。

配置结构示例，不改变默认值：

```json
{
  "perception": {},
  "training": {"algorithm": {}},
  "history_steps": 1
}
```

本体历史可用 `--history-steps 4` 开启。此时输入为 `618 + 57 × 3 = 789` 维：当前 618 维后追加三个过去本体帧，按旧到新排列；新增输入权重零初始化。部署必须使用同一个 checkpoint/契约，不能向 618 维模型直接提供 789 维输入。

继续一个已有 v1 run：

```bash
python scripts/reinforcement_learning/rsl_rl/distill_student.py \
  --resume /实际路径/v1/model_N.pt \
  --max-iterations 10000 --device cuda:0 --headless
```

这里 iterations 表示追加迭代数。续训恢复教师、学生、优化器及课程状态；必须保持环境数、观测与配置。默认轻量 checkpoint 从空回放重新收集，不能声称恢复原训练轨迹。需要保留回放时，在新 run 的 JSON 配置中设置 `training.algorithm.save_replay=true`，checkpoint 会包含普通/关键回放。旧 run 配置缺少此字段时仍沿用完整回放保存方式。需要改变感知配置或历史时，用 `--teacher-checkpoint` 加 `--init-student` 开新 run。v0 checkpoint 不能作为 v1 完整续训输入。

默认采用 MLP 与可选显式历史。GRU、地形重建辅助网络、PPO 微调尚未实现；这些是闭环证据表明当前架构不足时的后续升级，不属于已验证功能。

## TensorBoard

```bash
tensorboard --logdir logs/rsl_rl/deeprobotics_m20_student_v1
```

| 指标 | 用途 |
|---|---|
| `Loss/behavior`、`Loss/leg_mse`、`Loss/wheel_mse` | 分组加权拟合及腿、轮误差，比较时考虑噪声/阶段分布变化 |
| `Loss/heldout_action_mse`、`Loss/heldout_phase_*_mse` | 在线回放随机留出的拟合诊断，观察落地/恢复误差 |
| `Loss/heldout_leg_position_rmse_rad`、`Loss/heldout_wheel_velocity_rmse_rad_s` | 将动作残差换算为对应关节位置/轮速单位 |
| `Loss/teacher_ratio_target`、`Loss/teacher_executed_fraction`、`Loss/teacher_decay_gate_open` | 区分目标比例与 episode 内实际执行比例，确认课程是否进入全学生阶段 |
| `Loss/student_failure_ema`、`Loss/student_completed_episodes` | 只统计学生执行 episode 的终止情况和调度依据 |
| `Loss/coverage_phase_*`、`Loss/coverage_terrain_*`、`Loss/coverage_direction_*`、`Loss/coverage_terrain_level_*` | 排查关键状态、方向、难地形覆盖不足 |
| `StudentSensor/*`、`Loss/noise_level` | 检查有效比例、置信度、年龄、视场、混合分布及噪声课程 |

在线随机留出与训练样本可能是相邻帧，不是独立验证轨迹。均值动作误差降低不能证明落地更轻，也不能证明 real gap 缩小；应结合原有任务日志和下面的闭环评测。

## MuJoCo 手动部署

训练后使用 v1 checkpoint：

```bash
python deploy/deploy_mujoco/deploy_student_v1.py \
  --checkpoint /实际路径/v1/model_N.pt \
  --model "/home/ubuntu/桌面/m20_1/M20_perception_Lidar_rl_2/M20_perception_Lidar_rl/deep_robotics_model/M20/mjcf/M20.xml" \
  --terrain descent --noise-profile realistic --viewer
```

保持原来的手动操作：Z 站立、C 接入学生、W/S/A/D/Q/E 控制、R 阻尼。默认不自动进入策略模式。也可用 `--policy-jit /实际路径/exported/student_v1.pt`，需保留相邻的契约和来源 JSON。

`train` 使用 checkpoint 当时的感知参数与混合分布；`realistic` 使用完整名义扰动；`clean` 是理想完整观测参考，不代表真实雷达。`quality_reference` 与 `confidence` 具有相同的高度/位姿扰动，只在置信度估计误差上不同，用于成对归因。

## 同条件对照评测

默认只生成计划，不运行物理仿真：

```bash
python scripts/tools/evaluate_student_v1.py \
  --v0 logs/rsl_rl/deeprobotics_m20_student_v0/2026-10-12_23-24-52_380795_student_v0_stage_a/model_29999.pt \
  --v1 /实际路径/v1/model_N.pt \
  --teacher logs/rsl_rl/deeprobotics_m20_quiet_landing_v1/2026-10-10_00-48-58_quiet_landing_v1/model_257100.pt \
  --model "/home/ubuntu/桌面/m20_1/M20_perception_Lidar_rl_2/M20_perception_Lidar_rl/deep_robotics_model/M20/mjcf/M20.xml" \
  --terrain descent \
  --profiles clean lidar_fov height pose latency realistic quality_reference confidence \
  --commands forward \
  --output-dir /实际路径/新评测目录
```

看完 `plan.json` 后由你执行同一计划：

```bash
python scripts/tools/evaluate_student_v1.py --run-plan /实际路径/新评测目录/plan.json
```

学生均使用 v1 公共传感器管线和相同 profile，适合比较策略在相同扰动定义下的表现；它不复现旧 v0 部署报告的感知实现。教师只作为 clean 理想观测参考，不把教师 clean 与学生 noisy 当同条件对照。每个地形使用独立输出目录。

评测会写逐 case 原始 telemetry、触地/接触力记录和汇总。力峰值与冲量按完整的触地时间窗口统计，末尾截断窗口单列；全程接触力分位数另有清晰定义。位移使用连续策略边界位置，观测使用同步后的 scratch kinematics，避免 MuJoCo 积分后字段滞后一拍影响指标。

计划检查主模型 XML、地形 XML、checkpoint 和已登记代码的哈希，模型引用的外部 mesh/include 资产未递归冻结。默认初始位姿样本是种子索引的敏感性偏移，不是统计显著性的独立随机实验。原始记录应与完成率、命令跟踪及每米指标一起分析。

## 已验证与尚未验证

25 项 v1 离线检查及 8 项既有 v0 视场回归通过。检查包括共享训练/部署观测对拍、视场/坐标变换、时序解耦、固定采集坐标重投影、逐格历史/过期、单环境 reset、本体历史、动作延迟、相对位姿质量、倾斜姿态的误差传播、置信度成对对照、分层抽样、分组/阶段损失、冻结教师、标签所有权、续训拒绝错误状态、实际 checkpoint 初始化与 JIT 对拍。验证记录见 `m20_student_v1_validation.json`。

**此次没有启动训练，没有进行 Isaac 或 MuJoCo 闭环物理测试。** 因此不宣称 v1 的通行或轻盈落地性能已经改善。共享感知是高程图层面的 2.5D 近似，没有模拟逐束扫描、反射率、悬空结构或机身自身遮挡；真实传感器参数和质量标定仍待实测。
