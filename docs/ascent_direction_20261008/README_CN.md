# 10月8日：回退奖励并增加上楼方向偏移惩罚

## 本次采用的基线

按用户确认，放弃10月7日 `gait_safety_v2` 模型作为后续起点，奖励也回退到该轮改动之前。
后续权重起点为：

```text
logs/rsl_rl/deeprobotics_m20_stair_teacher/2026-10-06_22-06-56_stair_resume/model_199998.pt
```

checkpoint SHA256：`b500859bedc871db1a860298944e699d1aed7481142b35400267c12ce1f53f76`。

恢复依据为 `.scratch/gait_safety_20261007/before/` 的代码备份。
原有55个函数及类的 AST 均与该备份一致；既有奖励数值检查的26项输出也一致。
10月7日新增的前身间隙、后腿前倾约束以及同阶事件历史判定随此次回退移除。
日志和权重未删除；本次没有启动、停止或更新任何训练。

## 新增方向惩罚

用户描述的现象是“走成斜线或自行偏航”。新增
`stair_ascent_direction_cost` 惩罚以下三项：

| 分量 | 计算方式 | 避免的误判 |
| --- | --- | --- |
| 横向偏移 | 每步实际XY位移在命令路径法向上的分量，逐步累计 | 上楼速度减慢不会产生纵向位置误差惩罚 |
| 横向速度 | 世界坐标速度在命令路径法向上的分量 | 允许命令要求的侧移/斜向前进 |
| 偏航误差 | 实际yaw与命令积分得到的期望yaw之差，处理角度环绕 | 允许命令要求的转弯 |

参考方向在检测到上行台阶并开始前进时初始化，之后按照命令角速度积分。
参考方向不会随着机器人自行偏航重新初始化，从而避免“转歪后继续按本体系向前运动”逃避惩罚。
命令平移方向有明显改变时，横向偏移重新计零；命令要求的原地转向结束后建立新的行进参考。

每个分量都有容差、平滑饱和，参数统一放在
[stair_teacher.py 的 TUNING](../../source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/stair_teacher.py#L49)。
本次权重为负，原始总cost小于2；按正常RewardManager时间步积分。
不新增roll/pitch约束。

### 激活与退出

- 前方发现上行台阶且命令要求向前运动时，记录已看到的最高台阶并激活状态。
- 短暂失去扫描或机器人自行转开，不会直接清除该状态。
- 四轮稳定承重、越过已记录台阶并位于其上平台，且当前没有新上行台阶时退出。
- 转向误差已在容差内、前方变为下行地形时也退出。
- 停止、纯转向、反向运动时本项不计罚；从未激活上楼状态的平地和下楼不计罚。
- 环境reset清除累计量；同一步多次调用不会重复积分。

这是针对命令方向的约束。最高台阶只指扫描已看到的台阶，无法证明它是整段楼梯的最后一级；
本项不提供绕障路径规划，也不能单独保证交替落脚或优美姿态。

## TensorBoard

奖励项：`Episode_Reward/stair_ascent_direction_cost`。

指标前缀：`Curriculum/ascent_direction/`。

| 指标后缀 | 含义 |
| --- | --- |
| `samples` | 本次汇总环境中激活状态下的控制步数 |
| `lateral_error_mean_m` | 激活期间累计横向偏移绝对值的均值，米 |
| `lateral_error_max_m` | 本次汇总环境中最大的横向偏移绝对值，米 |
| `lateral_speed_mean_m_s` | 横向速度绝对值的均值，米/秒 |
| `heading_error_mean_deg` | 偏航误差绝对值的均值，度 |
| `outside_tolerance_fraction` | 横向偏移超过容差的激活步占比 |
| `cost_mean` | 激活期间未乘权重的平均cost |

均值按激活控制步汇总，来源为环境reset时的课程日志。`samples=0`时零误差不代表爬楼良好。
应同时检查上楼成功率、上楼地形level和固定MuJoCo场景；只看负奖励变小不能判定改善。

## 验证与限制

| 检查 | 实际结果 |
| --- | --- |
| 新方向项数值检查 | 通过；正确直行、侧移、命令转弯及减速无误罚；左右偏移对称；自行偏航仍被惩罚；reset、扫描丢失和边界状态通过 |
| 独立数值复审 | 通过；8个环境、120个随机步，向量执行与逐环境执行相同；复审为同模型家族，不声称异构模型复审 |
| 旧步态包络检查 | 通过；恢复的AST测试夹具补充numpy注入，生产配置未因此改变 |
| 旧奖励基线归因 | 原55个函数/类与回退前基线一致；26项既有检查结果一致 |
| `check_stair_rewards.py --assert-fixed` | **未通过**：新版测试要求的同阶事件判定已随回退移除；回退前旧基线也在同一项失败，未放宽断言 |
| Isaac运行 | 通过；严格加载199998 actor，16个环境、100步；actor244维、critic285维、16维动作，观测/动作/奖励有限；实际方向奖励与课程指标绑定有效 |
| 新策略改善程度 | **未验证**；未进行优化训练，100步冒烟不能证明长楼梯通过率或步态改善 |

证据：[方向检查](cpu_checks.json)、[旧步态检查](gait_regression.json)、
[回退一致性](rollback_verification.json)、[Isaac结果](isaac_smoke.json)、[文件验收清单](verification.json)。
独立复审记录位于 `.scratch/ascent_direction_20261008/reviewer_report.md`。

旧版同阶事件判定的已知局限与基线一起恢复，仍可能漏掉部分双腿延后落到同一台阶的行为。
此次修改没有解决这个旧问题；不能把整体奖励验收表述为全绿。

收尾pyflakes通过。`company-rules check`本项目可机检条款未发现问题；全机分发扫描仍因
`~/.gnupg`权限而未完成，另外10个他人维护检出被工具跳过，未改这些无关目录。
共享知识库查询受网络限制，知识条目先保存为
`.scratch/ascent_direction_20261008/kb_note.md`，未声称已写入或推送共享库。

## 下一轮训练起点

奖励目标已改变，建议先继承199998的actor，新建critic和优化器，避免旧价值估计直接沿用。
这是已有CLI支持的actor初始化方式，迭代计数从零开始。

```bash
conda activate m20_wzh
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Rough-Deeprobotics-M20-StairTeacher-v0 \
  --headless \
  --init_actor_from logs/rsl_rl/deeprobotics_m20_stair_teacher/2026-10-06_22-06-56_stair_resume/model_199998.pt \
  --run_name ascent_direction \
  --max_iterations 5000
```

上述命令为建议的首段验证训练，未执行；本次Isaac实测验证的是加载和运行，不是PPO训练启动。
`--init_actor_from`与`--resume`不能同时使用。PPO、速度采样、地形、pre_teacher、部署和网络输入输出保持当前配置。
训练配置只有在重新启动的新进程中生效，已有训练进程不会自动接收此次修改。

如必须继承critic和优化器，则改用已有`--resume --load_run 2026-10-06_22-06-56_stair_resume --checkpoint model_199998.pt`，
删除`--init_actor_from`。这种方式也会继承原模型训练状态，需要分别评估新奖励下的价值损失。
