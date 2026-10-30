# M20 StairQuiet-v1：上楼砸地声音减小

**当前交付包含可训练候选、电脑声压预测和校准录音计算；尚未得到降噪后的模型，也没有“分贝下降 50%”的实测结果。**
长时间教师训练、学生蒸馏均由用户启动。代码检查、仿真接触指标、训练 reward 不能代替声音验收。

研究融合版的实际执行记录见
[改造与验证报告](../logs/evaluation/stair_quiet_v1/fusion_v3/REPORT.md)。
当前组合控制在标准楼梯中能够完成任务，但声音模型尚无稳定双指标收益；
执行器退化压力下原基线和组合控制均摔倒。因此下面命令用于生成并评估实验候选，
不表示当前参数已通过降噪或实机验收。

研究融合后的实现增加了共享落地控制层，详见
[控制设计与部署边界](../source/rl_training/task_support/stair_quiet_v1/CONTROL.md)。
教师训练、专用学生蒸馏与 MuJoCo 使用同一 `landing.py`。默认 `combined`，
支持 `off/precontact/phase/combined` 消融；新 checkpoint 必须携带并执行控制契约。
旧部署器会拒绝这类 nominal actor，不能只导出 16 维网络就直接上机。

## 固定基线与实现

- 教师：`deploy/models/idle_stop_motion_v1/idle_stop_motion_v1_46400.pt`。
- 学生：`deploy/models/student_v3_5/model_21000.pt`。
- 文件 SHA256 在 `task_support/stair_quiet_v1/baselines.py` 固定；原模型不被覆盖。
- 新 Gym ID：`Rough-Deeprobotics-M20-StairQuiet-v1`。
- 三个入口：标准 mdp/config/agents 目录内的
  `stair_quiet_v1_env.py`、`stair_quiet_v1_env_cfg.py`、`stair_quiet_v1_ppo_cfg.py`。
- 辅助代码和声学工具统一位于 `source/rl_training/task_support/stair_quiet_v1/`。

教师严格加载完整 46400 actor（包括旧运动残差）与 critic，然后使用新优化器。
只继续训练已有 residual，冻结旧基座与动作分布，原始零命令的策略输出保持基线。
平地运动采用完整教师锚定损失；这是软约束，不能据此宣称平地闭环不退化。
旧下楼引导已在 46400 时结束，新训练不重新开启它。

原爬楼进度、碰撞、姿态、功率、终止与混合地形奖励全部保留。
每个物理子步计算落地逼近速度、加载率、滚动窗口冲击和悬空下降，
另增加接触输入功代理、短窗反弹再撞击、不同轮冲击重叠三个成本。
接触速度使用 `v_com + ω × r`，按实际 USD 的 Capsule 轮碰撞体、轮轴姿态与
COM 偏移估计支持点；运行时检查形状与尺度，不套用 URDF 的 Cylinder 假设。
只取法向逼近，不把正常滚动切向速度当砸地。多接触合力方向仍是近似。
这些成本寻找更轻落地动作，不能单独作为声音指标。
新增成本按本次 quiet 训练迭代平滑加权，从较小权重过渡到全权重；
resume 恢复本任务时钟，缺课程接线会报错。数值唯一来源是 `StairQuietSettings`。
原材料、碰撞接触刚度和控制周期保持不变。新增控制层会在接触前有限地修正腿目标，
按接触阶段调整显式 PD 增益，并让增益与目标经过同样的执行器延迟。
不修改策略 nominal action/history；实际发出的目标和增益另行记录。
TensorBoard 中 `mean_reset_batch_*_max_*` 明确表示各重置批次峰值的均值，
不能读成整轮全局最大值；速度均值/RMS 按触地数合并。

训练完成后，从指定 `model_21000.pt` 完整初始化学生，拟合新教师。
教师保持 `244+3 → 16`，学生保持 `789 → 16`。新学生任务是
`Rough-Deeprobotics-M20-StairQuiet-v1-Student`；两种策略的 nominal 动作均经过共享控制层。
学生控制器只使用学生实际收到的有延迟、噪声和缺失的地图。
当前接触反馈仍来自仿真，实机接触观测与执行器响应尚待验证。
学生蒸馏后仍须单独测试真实声音，教师通过不代表学生通过。

## 由用户启动训练

以下命令从 `rl_training` 根目录执行，使用本机已工作的 Isaac 环境。
训练前重新检查资源。机器上的其它任务和余量会变化，不能靠旧记录决定大规模并发。

```bash
cd /home/ubuntu/cyq_shixi_projects/rl_training
conda activate m20_wzh
nvidia-smi
free -h
df -h logs
```

资源预估沿用现有学生 preflight 的工程预留：
`8 GiB 仿真基础 + 4 MiB × 环境数 + 实际模型/rollout/replay 张量`。
该式不是显存实测上界。先从 64 环境启动，观察整卡与主机峰值，再决定另开新 run 扩大；
不要直接照搬正在运行的另一任务的环境数。此任务的附加 tracker 为
`ceil(window_seconds / physics_dt) × num_envs × 4 × 4 bytes` 的力历史，
再加每环境四轮的标量统计。保留日志与 checkpoint 的磁盘余量。
新控制还有有限容量的地形缓存及邻域查询张量。当前四环境实测缓存常驻约
0.35 MiB，不包括查询临时张量，不能将该数当作训练整体内存上界。

教师训练命令（执行后会真正开始长训练）：

```bash
systemd-run --user --scope -p MemoryMax=20G \
  python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Rough-Deeprobotics-M20-StairQuiet-v1 \
  --init_actor_from deploy/models/idle_stop_motion_v1/idle_stop_motion_v1_46400.pt \
  --num_envs 64 --device cuda:0 --headless --max_iterations 10000
```

输出在 `logs/rsl_rl/deeprobotics_m20_stair_quiet_v1/`。
10000 是首轮候选训练预算，不是承诺达标的轮数。
不要按最大 checkpoint 编号认定最佳；先用下述相同楼梯电脑对照检查爬楼与声音假设，
再用校准声学数据验证实际降幅。

同一任务续训：

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Rough-Deeprobotics-M20-StairQuiet-v1 --resume \
  --checkpoint /absolute/path/to/stair_quiet_run/model_N.pt \
  --num_envs 64 --device cuda:0 --headless --max_iterations 5000
```

续训核验任务/奖励/观测/传感器/物理/算法/源码契约、完整 reference 与冻结基座。
修改这些内容时应建立明确的新实验，不能通过去掉检查静默续训。
原 46400 只能作为新实验初始化，不能当 StairQuiet 续训 checkpoint。

选择候选教师后，先验证学生加载与资源（不启动仿真、不更新优化器）：

```bash
PYTHONPATH=source/rl_training python -m task_support.stair_quiet_v1.distill \
  --teacher-checkpoint /absolute/path/to/stair_quiet_run/model_N.pt \
  --init-student deploy/models/student_v3_5/model_21000.pt \
  --num-envs 64 --device cuda:0 --preflight-only
```

由用户启动学生蒸馏：

```bash
systemd-run --user --scope -p MemoryMax=20G \
  env PYTHONPATH=source/rl_training python -m task_support.stair_quiet_v1.distill \
  --teacher-checkpoint /absolute/path/to/stair_quiet_run/model_N.pt \
  --init-student deploy/models/student_v3_5/model_21000.pt \
  --log-root logs/rsl_rl/deeprobotics_m20_stair_quiet_v1_student \
  --run-name stair_quiet_v1_student \
  --num-envs 64 --max-iterations 10000 --device cuda:0 --headless
```

此处必须用 `--init-student`。`--resume model_21000.pt` 会恢复原教师与旧训练状态，
不能用来换成新教师。

## 只做有限接口检查

```bash
PYTHONPATH=source/rl_training python -m task_support.stair_quiet_v1.validate \
  --num-envs 4 --steps 32 --device cuda:0 --headless \
  --output logs/evaluation/stair_quiet_v1/isaac_smoke.json
```

该命令不训练，默认在新环境中运行完整基线教师；可用 `--checkpoint` 检查新候选。
结果必须读取 JSON 内的状态和字段，不能只看进程退出码。
普通 `play.py` 会修改地形与噪声配置，与本任务的严格恢复契约不一致。
旧 MuJoCo CLI 不含新控制层，会拒绝带控制契约的 checkpoint；使用专用入口：

```bash
PYTHONPATH=source/rl_training python -m task_support.stair_quiet_v1.benchmark \
  --role candidate-teacher --landing-control combined \
  --checkpoint /absolute/path/to/stair_quiet_run/model_N.pt \
  --model /absolute/path/to/M20.xml \
  --output logs/evaluation/stair_quiet_v1/my_candidate
```

上述 Isaac 检查本身不求声压。下面另接独立的声学预测工具，
通过冲击激励、结构模态和辐射声压计算假设场景下的 dB；默认参数尚未实测标定。

## 电脑端完整对照

从实际 MuJoCo 闭环导出每个接触点的力、位置、表面与时刻，保存原始数据和源码/模型哈希。
沿用两份基线自己的教师/学生部署适配器、观测、命令反馈、动作延迟和 PD。
默认是相同的八级上楼场景，可通过参数明确改阶数/阶高/踏面，不能分别给候选调简单地形。
默认环境不训练，CPU 推理；实际内存约 1 GiB，运行时可用 6 GiB 上限保护。

```bash
export PYTHONPATH="$PWD/source/rl_training"
MODEL=/home/ubuntu/xyproject/m20_wzh/m20_rl_training/deep_robotics_model/M20/mjcf/M20.xml
python -m task_support.stair_quiet_v1.benchmark \
  --role baseline-teacher --model "$MODEL" \
  --output logs/evaluation/stair_quiet_v1/my_comparison/teacher
python -m task_support.stair_quiet_v1.benchmark \
  --role baseline-student --model "$MODEL" \
  --output logs/evaluation/stair_quiet_v1/my_comparison/student
```

输出目录必须新建，保留失败与不完整试次。`contacts.npz` 是逐步数据，
`contact_points.npz` 保留各接触的分力与接触点，`events.json` 保留全部声学激励前史。
结束判据要求每个轮足真实接触顶平台，然后继续同一命令捕获固定的末次冲击窗口；
腾空越过终点、从侧面绕过、跌倒均不能算通过。
冲量提取使用固定触地/重新加载门槛与窗口，不声称恢复了真实高频接触波形。

新教师训练后执行（学生用 `--role candidate-student`）：

```bash
python -m task_support.stair_quiet_v1.benchmark \
  --role candidate-teacher --landing-control combined \
  --checkpoint /absolute/path/to/quiet/model_N.pt \
  --model "$MODEL" --output logs/evaluation/stair_quiet_v1/my_comparison/candidate
python -m task_support.stair_quiet_v1.compare \
  --baseline-teacher logs/evaluation/stair_quiet_v1/my_comparison/teacher \
  --baseline-student logs/evaluation/stair_quiet_v1/my_comparison/student \
  --candidate logs/evaluation/stair_quiet_v1/my_comparison/candidate \
  --output logs/evaluation/stair_quiet_v1/my_comparison/sound
```

没有新候选时可省略 `--candidate`，只建立两基线的声学预测基准。
同一份原基线不能换个名字当候选。比较检查完整上楼、停顿、完成时长、命令、
编译后物理模型、源码和事件提取条件；同类学生还核对感知与延迟。
教师与学生原有观测差异会保留在报告，不暗中抹平。

不训练即可比较“同一基线网络 + 新控制层”，使用显式 `control-teacher` 或
`control-student` 角色，不能写成新训练策略。例：

```bash
python -m task_support.stair_quiet_v1.benchmark \
  --role control-student --landing-control combined --model "$MODEL" \
  --output logs/evaluation/stair_quiet_v1/my_comparison/control_student
```

`--stress-profile` 可选 `nominal/terrain_high/terrain_low/contact_delay_loss/actuator_chain/combined`。
压力参数是假设实验，尚未由 M20 数据辨识；必须在相同 profile/seed 下重跑两个基线。
比较器拒绝把正常基线与不同延迟、限矩或地图误差的候选混比。
`contacts.npz` 增加逐子步目标、增益、阶段、修改与失效计数。

声学求解为“归一化接触脉冲 → 有阻尼结构模态 → 体积加速度 → 自由场声压 Pa”，
再计算 LAeq/LAFmax；保留滤波和结构振动前史，不用 `log(force)` 代替声压。
同一组接触时长与阻尼假设成对用于基线、候选，输出全部敏感性结果，
不能按策略选择最有利参数。源码/参数与限制见
[声学模型](../source/rl_training/task_support/stair_quiet_v1/SOUND_MODEL.md)。

`comparison.json` 与 `summary.md` 会给出**未标定模型预测**的 dBA、分贝差和读数降幅，
原始 Pa 数据/浮点 WAV 同时保存。WAV 数值单位为 Pa，播放设备并非校准声场，
不要通过调播放器音量“验证”降噪。默认模型不包含实际楼梯/轮胎/房间标定，
也不包含连续滚动、滑动及电机全部声音，不能宣称真机达标。
所有模型报告固定 `actual_acoustic_pass=false`。

## 原始声音在电脑上计算

`audio.py` 可以读取未压缩 WAV，用同增益采集的声校准器 WAV 标定到 Pa，
计算 A 计权等效声级、Fast 最大声级和采样峰值声压；检查校准漂移、
数字削波、背景、测量窗、身份和采集时间。保留录音前史，不能先裁剪再计算 Fast。
来源不全返回 `inconclusive`，计算完成也不等于机器人达标或 IEC 仪器认证。

```bash
python -m task_support.stair_quiet_v1.audio --help
```

完整命令、录音清单与格式见
[电脑声级计算](../source/rl_training/task_support/stair_quiet_v1/AUDIO.md)。
该模块不代填证书或真实试次，现有严格的声级验收仍独立执行。

## 唯一降噪达标判据

对同一台机器人、楼梯材料与几何、轮面、负载、速度命令、麦克风位置，
分别测两份原始基线与最终候选。完整走完同一段楼梯；不能通过停住、变慢或少走台阶降低声音。
记录声级计原始导出和同步视频。轮足碰撞以 LAFmax 限制，整段声响以 LAeq 限制。

按用户要求的字面门槛，**两项 dB 读数均须不超过各基线读数的 50%**，
并保留仪器不确定度余量。每次候选都单独冻结测试协议，交错采集重复试次。
最终学生必须相对两份指定基线均通过；若教师也要投入使用，教师另做同一套验收。

分贝是对数单位，声能减半约为 −3 dB，不能拿它冒充本任务的“分贝读数减半”。
测量与校准依据可参阅 [NIOSH 声级测量说明](https://www.cdc.gov/niosh/noise/prevent/understand.html)；
本任务工程门槛的唯一数值源为 `acoustic_protocol.json`。

执行步骤见 [声级验收工具说明](../source/rl_training/task_support/stair_quiet_v1/ACOUSTICS.md)。
工具只接受实际声级计测量；缺测、校准不合格、背景掩盖、试次不完整等返回
`inconclusive`，绝不以训练指标补齐。

本机没有此次候选的真实声音数据，因此本次不能宣布降噪达标。
训练后若声音未达门槛，该候选仍然不合格，应依据实际声音记录继续改进。
