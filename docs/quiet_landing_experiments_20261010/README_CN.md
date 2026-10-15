> 历史实验记录：V1.1/V1.2 独立任务与配置已移除，旧命令不再可用。当前融合版本见 [V1.5](../quiet_landing_v1_5_20261010/README_CN.md)。checkpoint、日志及本记录保留用于复核。

# Quiet Landing V1.1 / V1.2：实验架构与训练准备

状态：实验配置已实现；不代表降噪有效，不晋级为 V1.5。本目录记录训练前检查，正式实验尚未启动。

## 边界与入口

| 版本 | Gym task | 唯一奖励差异 | 输出目录 |
|---|---|---|---|
| V1 对照 | Rough-Deeprobotics-M20-QuietLanding-v0 | 原方案 | deeprobotics_m20_quiet_landing_v1 |
| V1.1 | Rough-Deeprobotics-M20-QuietLanding-V1_1-v0 | impact_peak 从截断平方改为平方后连续线性增长 | deeprobotics_m20_quiet_landing_v1_1 |
| V1.2 | Rough-Deeprobotics-M20-QuietLanding-V1_2-v0 | 保持原力峰惩罚，增加 quiet_load_balance | deeprobotics_m20_quiet_landing_v1_2 |

共用现有 `M20QuietLandingEnv`、env_config 文件、ppo_config 文件和 `train.py`，没有新训练入口。新配置继承 V1；不修改 pre_teacher/stair_teacher。Actor244、Critic285、Action16、固定PD及 PPO 超参数保持；新增信号只用于奖励/诊断，不拼进 Actor 或 Critic。V1.5、V2、V3 没有实现或注册。

两实验必须采用同一 V1 checkpoint、随机种子、环境数量与追加训练预算。默认起点是本次对照使用的 model_230300.pt。新实验初始化 Actor 与 Critic 权重，但优化器和 iteration 重新开始；保存的参考 Actor 也明确设为该 V1 checkpoint，以重新执行 critic 预热，而不是继承旧训练已结束的预热计数。

## V1.1 力峰尾部

设 x=max(Fn_peak−allowed_normal_force,0)/force_reference，k=sqrt(cost_cap)：

```
V1:   C(x)=min(x², k²)
V1.1: C(x)=x²                 x≤k
           k²+2k(x−k)        x>k
```

切换点的值和斜率连续，大冲击仍有区分，尾部线性增长以减轻异常值影响。权重、允许力、归一化尺度、窗口和其他奖励保持。仍通过 C(新峰值)−C(旧峰值)逐子步付费，跨策略步不重复收费。冲量、加载率等其他惩罚保留原截断。

## V1.2 稳定支撑的对角负载

FL/FR/HL/HR 顺序由生产环境运行时核验。用各轮法向力的指数滑动平均计算：

```
D=(Fbar_FL+Fbar_HR−Fbar_FR−Fbar_HL)/sum(Fbar)
C_balance=gate * min((max(|D|−deadband,0)/reference)², cost_cap)
```

惩罚是物理时间积分，reward adapter 除策略dt，与 RewardManager 乘dt抵消。数值唯一权威是 `m20_quiet/extended.py::ExtendedQuietSettings`；启用权重唯一权威是 `DeeproboticsM20QuietLandingV12EnvCfg`。运行时参数另存各 run 的 params/env.yaml。

gate 要求：原高程图全部有效且高度范围满足平坦阈值、四轮法向拟合有效、四轮持续接触且有有效载荷、无新触地、过出生保护、无明显转向指令、机身基本直立、垂直速度和角速度满足稳定阈值；连续保持到 settle 时间后才收费。断开任何条件即清平滑历史与保持计时，重入重新等待；episode reset 清全部累计量。

它约束稳定四轮支撑的持续对角偏载，不要求四轮等力，也不约束楼梯换相。平坦平台必须落在原高程图范围内才能启用；接触不稳定或扫描跨台阶边缘时不启用。若长期没有合格样本，应先核对 gate 覆盖率，不能把零奖励当成负载已平衡。

## TensorBoard

保留既有 QuietReward、QuietLanding、任务速度误差与终止信息，不引入旧步态调试card。新增：

- `QuietLoad/{fl,fr,hl,hr}/mean_normal_force_N`：出生保护之后的全时域平均力，包含离地的零载荷；不是只接触时段的平均值。
- `QuietLoad/{fl,fr,hl,hr}/contact_fraction`：同一时间范围的接触占比。
- `QuietLoad/stable_support_fraction`、`stable_has_samples`：稳定支撑门控的覆盖比例和是否有样本。
- `QuietLoad/stable_diagonal_signed_mean`、`stable_diagonal_abs_mean`：仅有合格样本时输出；无样本不填零。
- V1.2 独有 `QuietReward/quiet_load_balance`，其值是加权回合贡献。

四轮力和接触占比是按当前 reset 环境组的物理时长汇总；TB迭代均值仍是 logger 对环境组的均值，不是全训练的精确按时长加权均值。MuJoCo原速度/动能/力可视化和200Hz数据继续可用；MuJoCo诊断默认仍用V1配置，未接入实验奖励gate，不能把其中的stable_has_samples=0解释成实际负载平衡。

## 正式实验启动命令（本轮未执行）

从仓库根目录运行，先复查资源余量。两项串行或分别运行，不直接同时启动。本次既有 V1 的 cli.yaml 使用4096环境；以下沿用这一规模，不把8环境准备检查当成正式资源峰值证明。

```bash
PYTHON=/home/ubuntu/miniconda3/envs/m20_wzh/bin/python
V1_CHECKPOINT=logs/rsl_rl/deeprobotics_m20_quiet_landing_v1/2026-10-10_00-48-58_quiet_landing_v1/model_230300.pt

# V1.1：仅力峰尾部实验
"$PYTHON" scripts/reinforcement_learning/rsl_rl/train.py \
  --task Rough-Deeprobotics-M20-QuietLanding-V1_1-v0 \
  --headless --device cuda:0 --num_envs 4096 --seed 42 --max_iterations 10000 \
  --init_actor_from "$V1_CHECKPOINT" --init_critic_from "$V1_CHECKPOINT" \
  --preserve_actor_from "$V1_CHECKPOINT"

# V1.2：仅稳定支撑负载平衡实验
"$PYTHON" scripts/reinforcement_learning/rsl_rl/train.py \
  --task Rough-Deeprobotics-M20-QuietLanding-V1_2-v0 \
  --headless --device cuda:0 --num_envs 4096 --seed 42 --max_iterations 10000 \
  --init_actor_from "$V1_CHECKPOINT" --init_critic_from "$V1_CHECKPOINT" \
  --preserve_actor_from "$V1_CHECKPOINT"
```

公平效果实验还应有一份 V1 同预算续训对照：使用上面的同一组参数，仅 task 改为原 V1，添加 `--run_name experiment_control`。否则无法区分奖励改动收益与继续训练收益。4096/5000/seed42 是此命令的共同实验预算，不是算法最优结论。

中断续训使用原 `--resume --checkpoint /path/to/experiment/model_x.pt`，保持对应实验 task，去掉三个初始化/参考参数，让 optimizer、iteration、参考Actor和预热计数从实验checkpoint恢复。`--max_iterations` 表示追加更新数；不要误当成 checkpoint 的绝对迭代终点。

## 后续验收

效果评测前另行冻结测试集、阈值、训练预算和容许退化范围。本轮只完成准备，不声明奖励有效。

- V1.1：下楼入口、左前轮、P95及最大力峰、法向比动能、单位路程累计能量。
- V1.2：平地/平台稳定支撑覆盖率、对角持续偏置、各轮冲量/接触占比、姿态振动与侧向漂移。
- 两者都检查实际速度、触地次数、上下楼/平台通过能力；按共同覆盖路段比较，不能混入摔倒后数据。
- 两项分别有效且无明显退化，再创建 V1.5 联合实验。V1.5仍需独立验证，不自动晋级。

训练前检查与实际缺口见 `verification.json`、`extended_regression.log`、`base_regression.log` 和两份 `*_smoke.log`。

## 已执行的训练前检查

- 原 V1 CPU 回归、实验正反例和 MuJoCo POST 缓存反例均通过。
- 两任务分别通过原 train.py 的8环境、2次PPO更新：先critic预热，再Actor更新。checkpoint核验Actor244/输出16、Critic285；预热后Actor与起点权重差为零，参考Actor与起点差为零；预热计数各为2。
- 两份运行时 env.yaml 的差异恰好是峰值模式、负载权重、新增负载reward term；其他配置相同。TB指标有限、两个目录隔离、无旧Episode_Reward/Curriculum调试card。
- 0.6s短回合没有合格稳定支撑样本，两实验的稳定覆盖率为零；这只验证集成，不能证明负载已平衡。
- 补充V1.2平面/零速度指令检查，4次critic更新、Actor冻结：生产门控成功激活，负载reward非零，稳定偏置仅在有样本时输出。按TB环境组统计，稳定覆盖率约1.2%～3.9%，仍很低；正式训练必须看覆盖率，检查是否因减少合格支撑而逃避惩罚。
- 补充检查前两次启动失败已保留日志：一次Hydra浮点参数误写为整数；一次平面地形与继承的楼梯课程不兼容。最终仅在该诊断命令中关闭terrain_levels课程完成检查，没有修改正式任务的课程。
- 独立Codex同family复审通过；异构family复审未执行。其发现的无稳态样本填零问题已修正，并补了正反例。
- 未执行正式训练、效果消融、全地形新版本评测或实机声音测试；整卡/主机精确峰值未测，本目录有资源预算和开跑前余量，正式规模仍需重新核算。

`preparation_contract.json`保留第一次检查前的只读契约；复审后仅修正稀疏指标输出并补测试，最终源码哈希记录在 `source_manifest_final.json`。不覆盖已有V1评测产物和其冻结契约。
