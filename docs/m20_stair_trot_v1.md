# StairTrot-v1：上楼对角同步、平地保留完整教师

任务：`Rough-Deeprobotics-M20-StairTrot-v1`。本次直接修改 v1，没有新建 v2。默认以发布目录内完整
`idle_stop_motion_v1_46400.pt` 为起点。保留其基础网络、运动残差、航向反馈和动作分布，
只训练新的上楼残差。初始化时新残差为零。

## 三个入口

路径均相对 `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/`：

| 职责 | 文件 |
| --- | --- |
| env | `mdp/stair_trot_v1_env.py` |
| env_cfg | `config/wheeled/deeprobotics_m20/stair_trot_v1_env_cfg.py` |
| ppo_cfg | `config/wheeled/deeprobotics_m20/agents/stair_trot_v1_ppo_cfg.py` |

专用纯张量逻辑、模型和 runner 位于 `source/rl_training/task_support/stair_trot_v1/`。
相位、残差及稳定尺度在 `signals.py::TrotSettings`、`stability.py`；路线和成功阈值在
`course.py::StairCourseSettings`；奖励、地形与摩擦范围在 `env_cfg`，优化参数在 `ppo_cfg`。

## 目标与保护范围

- 左前＋右后、右前＋左后相差半个周期，同对角的脚共享摆动目标；两次摆动之间有短暂四脚支撑。
- 门控使用局部高度扫描中的向上台阶及当前前进指令。平地、缓坡、下楼、停止、后退、
  明显横移/转弯或无效扫描关闭新残差。后轮尚在最后一级台阶时仍可启用。
- 用实际接触力、离地间隙和离地时间判摆腿。同步事件要求另一对角承重；事件只消费一次，
  支撑丢失或相位超时即撤销落地资格。四脚跳跃、四脚静站和单脚反复抖动不能冒充同步事件。
- 正向步态奖励乘以新增前进距离；原地踏步或前后摇摆不能反复领取同一段路程的奖励。
  世界坐标路线与最高进度在 episode 内锁存，门控闪断不会清空。
- 移除旧前后落脚顺序、同级停留、原始关节镜像及抬脚奖励，保留速度、越阶、机腹净空、
  非轮接触、刮擦及力矩约束。
- 防滑使用接触点速度代理 `v_link + ω × r_contact`，根据实际默认 USD 的 capsule 碰撞体
  求支撑点；前后轮分别计分，不误罚正常滚动的轮心运动。这不是实测接触点滑速。
- 姿态允许前后踏面高度差所需的俯仰，约束横滚及横滚速度。冲击使用上一控制步接近速度
  与接触力历史峰值；边缘余量按高度扫描估计且只在落地时惩罚，两者均为采样精度有限的代理。
- 二阶动作差分把零动作正常计入，重置时屏蔽无效历史；功率使用 `sum(abs(τ*q̇))`。
  新稳定性项不会因初始地形等级为零而失效。
- 新门控为零时，对**相同输入**的动作输出严格等于冻结的完整教师；这项保证不等于
  已证明上楼后回到平地的整个轨迹无退化，后者仍需闭环对照。

原始 `policy(244)` 和 `motion_control(3)` 均不改变，另加
`trot_control(6) = [gate, sin(phase), cos(phase), cross_track, sin(heading_error), cos(heading_error)]`。
路线偏移成本对应的状态对残差可见。critic 使用原 `critic(285)`、原始指令及新控制状态。
新权重需要完整多输入 actor，不能直接交给旧单输入导出。项目 MuJoCo 入口已适配
完整教师、新上楼残差及六维控制状态，可直接读取训练保存的 `.pt`。

## 地形与课程

有踢面、无踢面各有短/中/长踏深列，阶高随课程变化，踏深独立覆盖，具体范围见 `env_cfg`。
同时保留平地、原上下楼、坡地和粗糙地面，并随机化接触摩擦。无踢面碰撞网格只有朝上的
水平三角面，没有薄板侧面或隐藏竖直支撑面。它是理想零厚度踏板；PhysX 边缘仍可能产生
局部接触作用，不能代表所有现实开放楼梯。

专用上楼列从低平台面向 +X 起步；所有轮进入出口平台后停止指令。必须四轮越过终阶、
有向上承载、航向/横移/姿态/速度合格并保持稳定，才记录完成并给予一次奖励。
完成且 episode 未失败才升阶；防坠底板、绕路线或跌倒不能晋级。普通地形保留原课程。
六类上楼分别统计实际 episode 数和成功率，零样本表示未测。

实际位置进度和整段完成目标参考了
[Chamorro 等的楼梯任务研究](https://arxiv.org/abs/2402.06143)；
出口和行走/滚动切换验收参考了
[Lee 等的轮足运动研究](https://arxiv.org/abs/2405.01792)。
这里只采用任务设计思路，不是复现其算法或宣称同等性能。

## 用户启动训练

以下命令在本机工程根目录执行。该环境及 USD 默认路径已用于接口检查。
新机器须按根 README 安装 Isaac Lab，并设置 `RL_TRAINING_M20_USD_PATH`。

```bash
cd /home/ubuntu/cyq_shixi_projects/rl_training
conda activate m20_wzh
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
nvidia-smi
df -h .
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Rough-Deeprobotics-M20-StairTrot-v1 \
  --num_envs 1024 --device cuda:0 --headless \
  --max_iterations 20000
```

上面的运行库设置与本轮实际 Isaac 检查相同，仅影响当前终端及其子进程。
首次训练不加 `--resume`。runner 自动加载上述完整教师，冻结它，使用新 critic 和新优化器。
环境数量是保守起步值，训练吞吐和资源占用尚未实测。输出在
`logs/rsl_rl/deeprobotics_m20_stair_trot_v1/`，不会覆盖旧教师权重。
可用 TensorBoard 查看该目录；除了总奖励，重点看 `trot` 的
`active_samples`、`diagonal_observed`、`pair_mismatch`、`support_deficit` 与原有越阶、速度指标。
`active_samples=0` 表示没有测到有效上楼样本，不能把其他零值理解为步态已学好。
同时看 `stair_course` 的六类完成率及样本数、`trot` 的滑移/冲击/航向/横移指标。
滑移按有效承载样本、冲击按落地样本统计；对应 `*_samples=0` 时不能据零成本下结论。

完整续训：

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Rough-Deeprobotics-M20-StairTrot-v1 \
  --resume --checkpoint /absolute/path/to/new_run/model_N.pt \
  --num_envs 1024 --device cuda:0 --headless --max_iterations 10000
```

`--max_iterations` 是本次额外迭代数。续训检查任务、观测、地形/课程/摩擦、奖励和 PPO 契约，
恢复 actor、critic、优化器及 warmup 计数；旧教师 checkpoint 不能当新任务完整续训。
本次补强前的三维控制输入 v1 checkpoint 也不能直接完整续训；默认从完整教师重新初始化。

## 检查与回放

CPU 反例测试无物理仿真、无优化器更新：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -p no:cacheprovider \
  tests/test_stair_trot_v1.py tests/test_stair_trot_v1_terrain.py \
  tests/test_stair_trot_v1_stability.py tests/test_stair_trot_v1_course.py
```

本次有限 Isaac 推理检查真实注册、环境、观测维度、有限动作/奖励、权重不变和门控关闭时
动作一致性，不等于训练效果验收。试运行脚本及输出用完清理，永久保留上述回归测试。

训练后可视化回放：

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task Rough-Deeprobotics-M20-StairTrot-v1 --num_envs 16 \
  --checkpoint /absolute/path/to/new_run/model_N.pt
```

回放保留确定性地形列映射，不使用通用随机 5×5 地形替换。
不要加 `--keyboard`，它会破坏路线指令与原始指令门控的配对，脚本对此显式报错。

## MuJoCo 部署

在 `m20_wzh` 环境中直接执行，不需要先导出 ONNX，也不需要启动 Isaac Sim。
把 `--checkpoint` 替换成想查看的本任务 checkpoint；以下使用当前已经保存的训练快照。

```bash
cd /home/ubuntu/cyq_shixi_projects/rl_training
conda activate m20_wzh
python deploy/deploy_mujoco/deploy_mujoco.py \
  --checkpoint logs/rsl_rl/deeprobotics_m20_stair_trot_v1/2026-10-20_15-10-21_stair_trot_v1/model_500.pt \
  --model /home/ubuntu/桌面/m20_1/M20_perception_Lidar_rl_2/M20_perception_Lidar_rl/deep_robotics_model/M20/mjcf/M20.xml \
  --terrain ascent --stair-height 0.10 --tread-depth 0.35 --stair-count 6 --stair-start 0 \
  --command-x 0.4 --viewer --autoplay --steps 1000 --stop-on-fall
```

`--viewer --autoplay` 打开窗口并自动行走；只保留 `--viewer` 则从趴下开始，
`Z` 起立、等待 ready 后 `C` 进入策略，`W/S/A/D/Q/E` 前后/横移/转向，
`R` 阻尼、`X` 趴下、`Esc` 退出。去掉 `--viewer` 可无窗口运行；
增加 `--output logs/evaluation/stair_trot_v1_mujoco/ascent.npz` 保存轨迹与控制状态。
手动控制沿用原按键速度；要固定在训练速度范围内，优先使用上述 autoplay 命令。

看平地时改用 `--terrain flat` 并删除所有楼梯覆盖参数。
MuJoCo 的 `ascent` 预设为实心楼梯；使用 `--terrain ascent_open`，或
`--terrain-xml deploy/deploy_mujoco/terrains/stairs_ascent_open.xml` 可切换无竖直面的楼梯。
沿用 `--stair-height`、`--tread-depth`、`--stair-count` 自定义参数。该预设使用零半径水平
三角 flex，起步和出口平台也没有竖直碰撞面；默认查看器通过同位置的纯显示 site 呈现踏板。
参数及几何说明见 [无踢面地形说明](../deploy/deploy_mujoco/terrains/README_CN.md#无竖直面的上楼梯)。
MuJoCo 和 Isaac 的边缘接触及求解器仍可能不同，需分别验收。

运行开头会打印 `[POLICY] StairTrot-v1`。部署直接加载保存的全部 actor 权重，
按 checkpoint 的 `trot_settings` 重建门控和相位；高度射线来自实际碰撞地形，
轮子法向力和离地时间来自物理子步。直线上楼路线固定为世界 +X，初始偏航/横移
不会改写目标路线。停止、后退、大幅横移/转向或平地关闭新残差。

专用适配器为 `deploy/deploy_mujoco/stair_trot_adapter.py`，新控制轨迹记录为 `trot_control`。
报告将完整 actor 输入记为 253 维，并分组列出；critic 不参与推理。
MuJoCo 使用指定 MJCF，Isaac 使用 USD，两个物理模型及求解器仍有差异。
无窗口短时物理检查只证明接口和执行链，不能替代完整楼梯通过率、步态及平地能力验收；
通用 `cleared_stairs` 也只是几何终线判据，不等同于训练中的终台稳定成功判据。

部署 CPU 回归测试（无优化器更新）：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python -m pytest -q -p no:cacheprovider tests/test_stair_trot_v1_deployment.py
```

## 训练后验收

同一固定地形、随机种子、指令和时长，对照完整源教师与新模型：

1. 上楼：阶高 × 踏深 × 有/无踢面 × 摩擦分别统计，并改变接近偏航与随机种子。
   看四脚接触时间线与视频中的 FL＋HR / FR＋HL 真正离地交替、抬落同步、整段通过率、
   跌倒和停滞；同时看后轮滑移、航向/横移误差及尾部、横滚/俯仰、机腹碰撞、踏边落脚、
   冲击、力矩与功率，不能只看总奖励或均值。
2. 平地：前进、后退、横移、转向、停止以及楼梯出口切回平地；
   对照速度跟踪、漂移和跌倒。独立平地同输入动作相等只是其中一项。
3. 下楼与粗糙地面：检查原能力没有意外损伤，尤其门控在坡顶和平台处的切换。

只有同步性改善且通过率、稳定性和平地能力均满足目标，才可称步态达标。
本轮不启动训练，因此交付的是已接线的 v1 训练任务，并非已训练成功的策略。
