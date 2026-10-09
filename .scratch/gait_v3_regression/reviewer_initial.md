# gait_v3 回退独立初审（2026-10-08）

## 范围与证据

本审计由独立 Codex/GPT reviewer 执行，与主实现者同 family；没有异构审计能力证明。只读取本项目，在 `m20_wzh` 下以 OMP/OpenBLAS/MKL/Torch 各 1 线程读取 TensorBoard 和 CPU checkpoint，不启动 GPU、Isaac、ROS、优化器或新策略 rollout，不修改生产代码。

- 直接重读事件文件脚本：`.scratch/gait_v3_regression/reviewer_tb.py`，真实执行 exit 0，结果 `reviewer_tb.json`。
- 基线：`2026-10-06_22-06-56_stair_resume/model_199998.pt`，SHA256 `b500859bedc871db1a860298944e699d1aed7481142b35400267c12ce1f53f76`。
- 失败模型：`2026-10-08_16-51-12_gait_v3/model_1700.pt`，SHA256 `0fe328764899ad7fc0609f45a704afa4e14791c9d0c242e2f0aee4dce9b92da1`。
- 失败奖励版本固定为 `.scratch/gait_v3_regression/stair_teacher.before.py`，SHA256 `cb4465a2f342c3154725a20a838de35eb12fd5c20f577bf344b40916e2256785`。当前生产候选正由主实现者修改，不能将其新奖励归因于失败训练。
- MuJoCo 是主实现者已实际运行的五工况对照，本审计独立读取 `docs/gait_v3_regression/{baseline,latest}/summary.json` 与保存的 landing 列表；没有重做仿真，也不冒充独立 rollout。
- 曲线的 early/late 是各自首/末 **50** 条，不是父报告中的首 200/末 100。基线最后 step 199998；失败运行事件在读取时已到 1866。因此曲线末窗口描述运行尾部，MuJoCo 描述固定的 1700 模型。

## 回退已经成立，并不限于楼梯风格

| 同一工况 | 基线 199998 | gait_v3 1700 |
|---|---:|---:|
| 上楼通过 | 是 | 否，最高有效落阶为 0 |
| 上楼速度误差 m/s | 0.09068 | 0.46235 |
| 下楼通过 | 是 | 否，偏出测试走廊 |
| 下楼速度误差 m/s | 0.09541 | 0.34201 |
| 平地速度误差 m/s | 0.05706 | 0.35976 |
| 平地机械功率 W | 9.05 | 52.70 |
| 左转平移漂移误差 m/s | 0.03963 | 0.29180 |
| 右转平移漂移误差 m/s | 0.04820 | 0.44680 |

两模型五个工况均未摔倒。纯转向不应要求向前穿越终点，不能因其 `cleared_terrain=false` 判转向失败；应看偏移和角速度误差。新模型上楼功率从 67.30W 降到 28.73W 是停在楼梯前的结果，不能证明节能奖励成功。上楼 same-tread 为 0、alternation=null 是没有有效落阶样本，不能证明同阶改善。基线能完成任务但前、后轴交替率都只有 0.25，仍不是用户需要的 FL1→FR2→FL3 / 后轴对应逐阶交替。

## 主要根因与证伪边界

### 1. 风格与任务信用绑定，导致成功但不完美的上楼失去主要正奖励

失败版本首次落阶只给 bounded preparation；主 front/rear completion 要求当前相邻、换侧、安全落脚，且前一个 tread 历史仅落过一侧。前一个台阶一旦两脚都落过，下一阶的首脚也没有主 completion 信用。之后仍可通过新的安全单侧→对侧相邻 transition 恢复，状态机没有永久卡死，但对基线习惯路径几乎不给主要上楼信用。

保存的基线实测首次落脚顺序为前轴 `FL1, FR1, FL2, FR2, FL3, FR3, FR4, FL4, FR5, FL5`，后轴 `HR1, HL1, HR2, HL2, HR3, HL3, HR4, HL4, HL5, HR5`。将这个真实 landing 时间线代入失败版本的明确规则，前一中间阶在进入下一阶前已成为历史双侧，所有主 transition 可以为 0；四个中间阶 × 两轴的历史 duplicate 总事件罚为 **-36**。这是对保存的事件序列的数学推导，不是生产状态机全量轨迹重放；扫描/clearance 条件还可能进一步降低正信用，不能把 -36 当作该次 rollout 已测总 reward。

TB 首 50 条里 order+same-tread 为 **-0.38950/名义秒**，front+rear completion 仅 **+0.03473/名义秒**，比例约 **11.2 倍**。失败 reward 将“会爬但风格不对”的基线动作变成大量负样本，容易奖励停住、绕开或姿态退化。该证据支持严重目标冲突，不证明每一个坏动作由某一个奖励独立造成。

### 2. 事件单位不是 50 倍 bug，时间位置及量级仍不友好

控制周期为 .005×4=.02 秒。事件函数除以 dt，RewardManager 又乘 dt，故主 completion +3、单轴单阶 duplicate -4.5，确实是事件总额，不是意外放大 50 倍。一般速度奖励最高为线速度 5×.02=.10、角速度 3×.02=.06 每帧；一个 duplicate 的瞬时 -4.5 相当于约 28.1 帧最大速度正奖励，两轴 -9 相当于约 1.125 秒最大速度正奖励。gamma=.99 的时间折扣使立即受罚、迟后完成更难成为正回报。

front fold -3 是 bounded 速率成本，最多 -.06/帧；它与 duplicate 的事件成本不同。现有“严格交替为正、故意补脚为负”的 CPU 判据只验反作弊，没有同时验“有缺陷但真实完成任务”的策略是否保留任务回报。这是前次验证缺口。

### 3. 续训不是普通 resume：随机 critic 与自适应 LR 同时放开 actor

params `resume:false`、迭代从 0 开始。model_0（首个更新后）actor 相对基线 MLP 权重漂移只有 **0.000819**，critic 漂移 **0.999387**，确认 actor-only 初始化、critic 新建。actor 漂移在 100/500/1700 分别为 **3.432% / 4.435% / 4.660%**。新奖励确实需要适应 critic，但没有先冻结 actor 的适应段，也没有对任务能力的参考约束。

配置 LR=1e-4 并不构成实际上限：官方 adaptive PPO 在低 KL 时乘 1.5，日志峰值 **3.844e-4**，step100 **3.375e-4**，500 后降到 floor **1e-5**。nominal 小 LR、实际初期放大并快速漂移，后期降 LR不能恢复丢失能力。建议将奖励冲突与优化状态一起修；只追加姿态成本不足以解决平地、转向同步回退。

随机 critic + 未约束 actor 更新是具有数据支持的机制推断，未做单因素消融，不能声称已证明唯一因果。官方 PPO 分别裁剪 actor/critic 梯度，不能错误归因“critic 抢占全局梯度裁剪”。

### 4. 历史信用部分不可观测，指标下降可能来自避开任务

actor244 / critic285 没有 map ledger、历史 duplicate、expected side 输入；当前相同姿态/scan 因过去某 tread 是否两侧落过，后续 transition 信用可不同。清掉 ambiguous expected 避免了固定隐藏 leader，但未消除所有历史信用差异。单步 MLP 的价值估计因此仍受额外部分可观测性影响。反作弊历史仍有必要；更稳妥的是独立任务信用 + bounded 风格塑形，不先扩展 ABI。

课程均级提高不能替代通过率；same-tread 下降必须结合已访问中间阶、穿越数和样本数看。训练后平均 episode 长度仍约 980/1000 帧，符合长时间存活的差动作，不是大量摔倒截短。

## 曲线量化

| 末 50 条均值 | 基线 | 失败运行 |
|---|---:|---:|
| Train/mean_reward | 112.188 | 18.964 |
| mean_episode_length | 997.585 | 980.105 |
| 正 reward rate 总和 | 7.49160 | 5.48939 |
| 负 reward rate 总和 | -1.88347 | -4.55011 |
| 净 reward rate | 5.60813 | 0.93928 |
| 速度跟踪 error_xy | .38691 | 1.02883 |
| base_z | .54170 | .42948 |
| knee_pos | .75970 | 1.46002 |
| value loss | .18819 | 1.57507 |

Episode_Reward 是累积加权 reward 除名义 episode_length_s=20，不是按每个实际截短 episode 的秒数重新归一。mean_reward 与其净 rate×20数量相符。失败末期最大的新增损失不仅风格事件，还包括 hipx、smooth、knee、standstill 等一般项，符合全局动作失控/能力漂移。

std 从 .37220 到 .44307（1700 checkpoint，约 +19%），value loss 峰值约 7.17，没有 std 爆炸、value 无穷或 NaN 的证据。不能把普通增幅夸成数值发散。

## 恢复建议及交付限制

从 task-capable **199998** 建新分支，保留 1700 作为失败样本，不作为 champion。修法至少包含：独立全身物理穿越正信用、降低初始风格事件成本、恢复完整速度信用；critic 用旧权重 warm start 并在 actor 冻结时适应新奖励；固定小 LR、冻结基线 actor 的保留约束、std 上限与可恢复的辅助状态。

后续门控必须同时观察 ascent/descent/flat/left/right：已有任务通过率、走廊偏移、速度误差、真实落阶样本数、逐阶交替/跳阶、姿态角、机械功率。风格指标变好但任务失去、能量下降但停住，均不能晋级。短训练冒烟和 CPU reward checks 只证明接线/数学，不能证明步态已学会。
