# M20 stair teacher 步态修正记录

## 本轮范围与状态

修正 stair_teacher 的上楼同阶停留、下楼承重后腿折叠和原地转向摆腿幅度；新增评测、轨迹留样及 TensorBoard 指标。采用现有 PPO 微调路线，未接入 AMP，未启动正式训练。新奖励需要重新启动训练才能生效，正在运行的训练仍使用启动时的配置。

actor 的 57+187 维输入、16 维输出、扫描数值定义及 PD 均保持原样；critic 仍有训练特权信息。`pre_teacher_env_cfg.py` 与部署控制程序的 SHA256 和修改前一致，未调整 pre_teacher 奖励或配置。

修改前源文件及冻结 checkpoint 在 `.scratch/gait_refinement_20261002/baseline/`；文件哈希见 [baseline_manifest.json](baseline_manifest.json)。该目录用于精确复现本轮基线，不是新训练产生的权重。

## 奖励修改与预期动作顺序

所有新增约束阈值和权重的唯一代码来源为 `mdp/gait_refinement.py::TUNING`。它们是可微调的软约束，不代表硬件安全限值。

| 修改 | 激活条件和实现 | 避免的错误 |
| --- | --- | --- |
| `stair_up_same_tread_cost` | 同轴两轮在中间阶有向上承重接触、轮心高度匹配踏面；每阶一次扣分，适度提高权重 | 一腿已获通过奖励后，另一腿再落回同阶 |
| `stair_up_same_tread_dwell_cost` | 同一踏面累计同阶时间超过短暂支持转移预算后，以有界速率扣分；离开同阶停止当步成本 | 一次性扣分之后长时间停留；反复抬脚、暂时转头或停下续期免费时间 |
| 后续踏面识别 | 检查完整扫描里的上升边缘，以记录的楼梯方向、相邻间距及高度筛选；后轴还能使用前轴记录的物理台阶 | 最近边缘仍是第一阶造成漏检；前轴已经到顶后漏管后轴；末端平台误罚 |
| `stair_descent_rear_fold_cost` | 下行命令和下降地形/尚未结束的下楼事件；只检查稳定承重、有效地面且轮心匹配接触面的后轮；髋至轮心距离不足或膝角超过宽松包络才扣分 | 对正常屈膝、上楼抬腿或悬空摆动腿一律惩罚 |
| `turn_swing_size_cost` | 严格原地转向、实际悬空且局部地面有效；超出地面相对高度或髋轮水平距离包络才扣分；复杂地形包络更宽 | 世界坐标高度差被误当成抬脚；前进混合转向受到原地转向步态约束 |
| `feet_air_time_ang_z_M20` | 起落事件上的有界三角得分，超过合理空中时间不再增加；实际同方向 yaw 进展和速度误差参与门控 | 长时间腾空、原地抬脚跳舞或反向转动刷奖励 |
| `rotation_gait_status` | 接触和局部高度共同确认一组对角轮承重、另一组确实悬空，且真实转向 | 两组都接触、只是站在不同台阶上也能刷步态奖励 |
| 原地转向姿态及对称项 | 恢复弱 hipy/knee 姿态参考；保留侧向放松；缩小 symmetry 的命令门控并降低其权重 | 纯转向时关节约束完全关闭；原转向步态项侵入低速登楼 |

预期运动顺序：

1. 平地主要使用轮子平移；扫描看到楼梯后，保留原来的前进和台阶完成反馈。
2. 第一阶允许任一前腿先越过边缘、落在踏面并稳定支撑。
3. 下一条前腿越过下一阶，例如左前阶1 → 右前阶2 → 左前阶3；原来的顺序与相邻阶完成账本继续工作。
4. 后腿独立交替，跟随前腿记录的台阶；前腿到顶不会提前关闭后腿约束。允许短暂支持转移，不要求全程只有一个轮子支撑，也不奖励两腿长时间停在同一中间踏面。
5. 最后进入平台后，允许两前轮/两后轮同高支撑，关闭同阶样式成本。
6. 下楼允许必要俯仰和屈膝；只限制已承重后腿的过度折叠，不锁定膝角或摆动腿轨迹。
7. Q/E 转向时鼓励真实跟踪转向命令及紧凑抬落脚，前进混合转向不会收到纯转向步态奖励。

这是奖励偏好的方向，不能保证 PPO 一定学出唯一的左右交替序列；须以新 checkpoint 的落脚事件验收。

## 固定权重的实测基线

冻结 `2026-10-02_00-12-17/model_31800.pt`，每个场景 750 个 50 Hz 控制步。上下楼为 5 阶、阶高 15 cm、踏面 30 cm，命令前进 0.5 m/s；转向分别为 ±0.6 rad/s。原始探索结果在 `baseline/`，接触判据补上轮心与实际地面匹配后的权威结果在 [baseline_v2/summary.json](baseline_v2/summary.json)。这次采样并非用户指定 checkpoint 的确认，用户尚未提供此前手动测试的精确权重。

| 场景 | 测量结果 |
| --- | --- |
| 上楼 | 四轮通过末端、无跌倒；前/后相邻阶交替率均 75%，各有4次判定；后轴中间同阶3次，前轴0次 |
| 下楼 | 四轮通过末端、无跌倒；后腿髋轮距离最小0.306 m（含摆动）；有效承重距离5分位0.423 m，过度缩短承重样本约1.30% |
| 左转 | 无跌倒；离地高度95分位0.113 m，水平伸距95分位0.237 m；yaw误差均值0.595 rad/s |
| 右转 | 无跌倒；离地高度95分位0.166 m，水平伸距95分位0.270 m；yaw误差均值0.609 rad/s |
| 平地 | 无跌倒；平移速度误差均值0.085 m/s |

当前基线转向实际平均速度约为命令的两倍，因此后续评测同时看 yaw 跟踪，不能把“脚步小了但不转了”判为通过。通过末端只检查末帧位置和跌倒状态，未证明没有绕行；平地场景未越过终点不表示平地运动失败。

## 新增 TensorBoard 指标

沿用现有 `Episode_Reward/<奖励名>`，新增 `Curriculum/gait_quality/*`：

- `turn_samples`、`descent_samples`、`rear_support_samples`：先确认对应场景确实有数据。
- `turn_clearance_mean_m`、`turn_reach_mean_m`、`turn_yaw_error_mean`：转向摆腿幅度和转向跟踪同时评价。
- `rear_extension_mean_m`、`rear_fold_support_fraction`：下楼后腿稳定支撑时的伸展及过度折叠比例。

`Curriculum/step_completion/*` 另增加 `up_front/rear_same_tread_events`、`up_front/rear_same_tread_dwell_s`。现有的通过率、违规、后退、停滞和分地形 level 继续保留。样本计数为零时，成本或比例为零不表示步态已学好。

## 检查结果与限制

1. `check_stair_rewards.py --assert-fixed`：既有抬腿反馈、重试、三阶交替、后轴交替、防重复信用以及楼梯姿态权重检查通过。
2. `check_gait_refinement.py`：新奖励反例通过。真实187点双阶扫描能检测同阶；单级末端平台不罚；前轴上平台后的后轴约束有效；抬脚/转头/暂停不能重获同阶免费时间；静止或反方向转身不获转向信用。
3. Isaac Lab 最终源码：16环境、60步，actor 244维、critic 285维，动作/观测/奖励有限；实体解析成功，转向幅度成本实际触发。详见 [isaac_smoke_final.json](isaac_smoke_final.json)。本次回放未触发楼梯事件，下降折叠及同阶的非零成本由数值轨迹验证；不把它们报成物理闭环已验证。
4. 独立 Codex reviewer 的复跑找到了近场下一阶漏检和grace朝向续期，均已修复并加入反例。未取得异厂模型 reviewer，不能称为异构模型复审。
5. 系统 Python 的 pyflakes：无新增绑定错误；既有 stair_teacher 未使用 import 与 smoke 注册任务的副作用 import 告警保留。`m20_wzh`未安装 pyflakes，未修改其依赖。

最初250步检查缺少进度输出且超过限定检查时间，停止的是本轮自己的进程；随后工具补上分阶段进度和超时堆栈，并用最终源码60步成功复验。未停止用户的正式训练。另一次检查工具缩进错误已修复，最终真实执行完成。

没有新 PPO 训练、新权重改善对比、AMP 或真机验证；不能宣称已经解决可见步态问题。固定评测判据在 [criteria.json](criteria.json)，后续用同场景比新旧 checkpoint，同时要求任务通过能力、速度跟踪不恶化。下楼包络只修承重折叠；悬空腿的过度弯曲仍需结合轨迹单独确认。

## 微调与复评命令

在项目根目录、`m20_wzh` 环境运行。下面只初始化现有 stair_teacher 的 actor，重新适应新的奖励价值函数与优化器；使用固定较小学习率。它不是从 pre_teacher 重新开始，也不是恢复旧优化器的 `--resume`。第一轮有限微调后先复评，不能只看训练总奖励。

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Rough-Deeprobotics-M20-StairTeacher-v0 \
  --headless \
  --init_actor_from logs/rsl_rl/deeprobotics_m20_stair_teacher/2026-10-02_00-12-17/model_31800.pt \
  --run_name gait_refine \
  --max_iterations 5000 \
  agent.algorithm.learning_rate=0.0003 \
  agent.algorithm.schedule=fixed
```

复评时将 `--checkpoint` 换成该轮实际生成的 checkpoint；输出目录必须与基线分开。

```bash
python scripts/tools/evaluate_stair_gait.py \
  --checkpoint logs/rsl_rl/deeprobotics_m20_stair_teacher/<新运行目录>/model_<迭代数>.pt \
  --model "/home/ubuntu/桌面/m20_1/M20_perception_Lidar_rl_2/M20_perception_Lidar_rl/deep_robotics_model/M20/mjcf/M20.xml" \
  --output-dir docs/gait_refinement_20261002/candidate \
  --steps 750 --stair-height 0.15 --tread-depth 0.30 --stair-count 5 --min-extension 0.35
```

评测器可选 `--cases ascent descent turn_left turn_right flat`；每场景 `.npz` 包含本体位姿、关节、动作、命令、轮位置和接触力，方便筛选未来参考轨迹。当前这些文件标为 **unreviewed**，不是已经合格的 AMP 专家数据。

## 经验记录：现象 / 真因 / 判据 / 修法

- **现象**：同阶约束既会罚最后的平台，又会漏罚机身未跨过第一阶时的两前轮同阶；抬脚或改变朝向可能续期免费时间。
- **真因**：旧事件成本未检查后续上升边缘；只用最近边缘的高度无法判断下一阶；过渡预算按每次接触/门控启闭重新计算。
- **判据**：生产187点扫描的双阶与终端平台对照；固定同一阶重复抬脚、转头和命令暂停；前轴扫描为平地但后轴仍有历史阶的对照。
- **修法**：整幅扫描相邻边缘和后轴历史联合门控；过渡预算按物理阶累计，仅新阶或episode清零。

Hub职责/在线知识检索受网络权限限制，未同步公司知识库或推送远端。经验保存在本项目本文。`company-rules check` 本项目通过；全局扫描因 `~/.gnupg` 权限被拒而未全绿，未修改其它仓库。
