# gait_v3 回退修复二审（2026-10-08）

## 结论

**当前候选的奖励与 PPO 接线审查通过；发现的旧 checkpoint resume 漏洞已修复并真实 CPU 复跑通过。无剩余具体阻断。** 这表示代码和短程任务保留的证据齐备，不表示严格逐阶交替、前腿夹角、下楼后腿姿态已经学会，也不保证长期训练不回退。

独立 reviewer 为 Codex/GPT，与主实现者同 family；异构 family 不可用，如实记录，不声称异构验收。只修改本目录 scratch 证据文件；生产代码由主实现者单独修改。本 reviewer 没有启动 GPU、Isaac、ROS 或新的 MuJoCo rollout。

失败原因和直接 TensorBoard 重读详见 `reviewer_initial.md` / `reviewer_tb.json`。其中失败奖励 source 固定为 cb4465a，不与本轮候选混用。

## 验证版本

| 文件 | SHA256 |
|---|---|
| mdp/stair_teacher.py | `c47c5e0823962f8343749967df6803ce262ee277e253d4dfff79a6e6e111381b` |
| stair_teacher_env_cfg.py | `aed4e702642395ee1598524685a3fff61244643a86a75f734c0ac4a7ceff570e` |
| rl_training/stair_guarded_ppo.py | `630870284cb7f6a5a5787f14116e39b8666886e9e29bcbf0503ff70f23c63e9e` |
| agents/stair_teacher_ppo_cfg.py | `45eb4cbb0a240d6d9f256f5e2a60cba7e25dfab75c892d46bd79262e6af281b4` |
| scripts/reinforcement_learning/rsl_rl/train.py | `6b7fe859b90a5a4898f4d5a9948f92d9862869424875b59ab498fe71c4ed6719` |

## 已实际复跑的检查

全部在 m20_wzh Python 下，OMP/OpenBLAS/MKL/NumExpr/Torch 各 1 线程。

1. `scripts/tools/check_stair_preservation.py --output .scratch/gait_v3_regression/reviewer_preservation_cpu.json`：真实 RSL-RL MLP/PPO/storage/update；actor warmup 精确不变、critic 改变、anchor 非零、reference 无梯度且参数不变、辅助 checkpoint 状态恢复、std 增长上限、task crossing 反例、legacy resume 修复，exit 0。
2. `scripts/tools/check_gait_v3.py --output .scratch/gait_v3_regression/reviewer_gait_cpu.json`：左右首迈各 8 阶、前后轴独立、1→3 跳阶不支付风格 transition、三阶后轴每阶双脚落地、同脚连续、错阶恢复、迟到同阶、批量/reset、射线区间早结算、坏 map、平地退出、姿态单调、速度不折扣，exit 0。
3. `.scratch/gait_v3_regression/reviewer_candidate.py`：独立构造旧学习率 resume 反例，真实 CPU RSL-RL；同时使用实际 IsaacLab 纯 Python configclass 与官方 RSL 配置类加载生产 config 文件，避免启动 simulator initializer，exit 0。结果 `reviewer_candidate.json`。
4. `_guarded_update` 与本环境官方 PPO.update 的 AST diff：`reviewer_vendor_diff.txt`。唯一新增逻辑是将 anchor loss 加进 **同一次 backward、梯度裁剪、optimizer.step**，没有另起隐藏 post-PPO 优化器，也没有遗漏官方 surrogate/value/storage 逻辑。

这些是有限反例与代码契约检查，不是模型性能认证。

## 找到并关闭的 P1

第一次独立 CPU 反例：fixed 配置 `self.learning_rate=1e-5`，加载旧 optimizer 后真实 param_groups LR 仍是 **.000170859375**；旧模型无辅助 reference，`reference_actor=None`，warmup/anchor 全跳过。普通 `--resume` 因此可能继续旧的大 LR 且缺少任务保留约束。

修复后独立重放：

- fixed schedule 的真实 optimizer LR 为 **1e-5**。
- 旧 checkpoint 完整 load 自动以加载的 actor 建立冻结 reference，warmup counter=0。
- 新带辅助 checkpoint 恢复原 reference 与 counter（反例设 63，load 后实际 63）。
- 显式更换 reference 会重新置 warmup counter=0。
- partial actor-only load 不偷偷恢复辅助状态，train.py 的 actor init 自动以 `init_actor_from` 建 reference。

full resume 自动冻结的是**被加载的** actor，不能自动识别其质量；从失败 1700 加载并冻结会保住错误行为。恢复仍应从已通过任务的 199998 开始。未提供参考且从零训练的 GuardedPPO 没有 anchor/warmup，因此该安全机制的结论只覆盖任务基线初始化/恢复路径。

## 奖励复核

候选保留已有严格相邻换侧 transition 与两轴独立 ledger；same-tread 事件成本从 -4.5 降到 -1.5、dwell 从 -1 降到 -.2、front-fold 从 -3 降到 -1；姿态不再扣掉原速度跟踪信用。

新增全身上楼物理 crossing 与风格分开：

- 同一 riser 前、后轴各曾有一只轮足在上台面稳定支持；稳定支持由实际轮足位置、接触和时间累计确认。
- 当前四轮全在射线 riser 区间之外、四轮均在局部走廊内、轮底高于该台面容差、本体高于台面 .22m、Rzz>.35 且至少两轮 loaded。
- 只对 known/正向允许的真实 riser、每 riser 一次支付；retreat、旁绕、仅前轴历史、低身/翻倒不给信用。
- 不要求 `map_safe`（空中跨过沿口的风格条件），所以从脸面接触恢复后真实爬上也能支付任务奖励；impact 单独受罚。
- helper CPU 检查中 map_safe 全零仍能得到物理任务信用，证明任务奖励没有再次被空中风格绑死。

map_landed 是历史稳定支持，不等于当前所有轮足必须站在该一级；当前四轮全身过线/高度/走廊/直立与接触条件补足了“只有历史、现在未通过”的漏洞。重复支付由 map_task_paid 拦住。该奖励不支付严格跳过且从未落过的台阶；风格 skip 指标与处罚另行保留。

仍有训练目标的边界：任务 +2/阶并不单独数学压过两轴同阶合计 -3/阶；其余速度/前进/准备信用共同决定完整回报，较旧 -9/阶更温和但不是“必然选择爬”的证明。hidden map/history 对 MLP 的部分可观测性也仍存在。不能用这次小反例或短程保留结果宣称设计已经完美；长期必须继续以物理任务完成作为风格优化的前置保留门。

## PPO 与配置复核

- reference 为 deepcopy actor、strict load、eval、requires_grad(False)，不在 PPO optimizer 内；实际 reference 无 grad，actor anchor 梯度非零。
- anchor 是按参考 std 归一的均方动作均值误差；flat/uneven 权重 1/.1，在 CPU fixture 中实测比例 9.999998。它是软惩罚，不是硬 KL 上限；固定 LR 和 std 上限也不是闭环性能保证。
- 最初 50 次 update 禁用 actor 参数梯度，保留 critic 优化；随后恢复原 requires_grad。reference std×1.05 上限真实接线。
- exact StairTeacher runner 构造出的 algorithm 是 GuardedPPO、LR1e-5/fixed、entropy0、clip.1；PreTeacher subclass 仍是原 PPO、LR1e-3/adaptive、entropy.003、clip.2，实例修改不串。官方 configclass 的真实 CPU 构造通过。
- train.py 支持 actor+critic 权重初始化并用 fresh optimizer；`init_critic_from` 必须配套 actor init，禁止与 resume 混用；init_actor_from 自动参考，preserve_actor_from 可指定；full resume 会保存/恢复 reference、counter。teacher 244/285 ABI 未变。

## 已有实际运行证据与限制

本 reviewer 独立读取主实现者运行的 `final_reward_smoke.json`：16 env×200 step、finite=true、244/285、上述最终 reward SHA 一致。**真实 crossing 事件为 0**，因此该次 Isaac 接线冒烟不能证明 crossing 支付已在动态物理中发生；其反作弊和允许条件的证据来自 CPU 夹具。

主实现者真实 64 env×55 update 后 model54 的相同五项 MuJoCo summaries：上/下楼和平地均通过、无摔倒；上楼速度误差 .09558（基线 .09068），下楼 .09576（.09541），平地 .06213（.05706），左右转平移误差 .04175/.04862（.03963/.04820）。平地末 y=-.18194（基线 -.16972），下楼末 y=-.68639（基线 -.13971），仍需长期偏移门控。机械功率也与基线相近，没有回到失败1700的大幅漂移。

**55 update 中 50 次 actor 冻结，仅约 5 次 actor 真正学习。** 该结果支持参考接线和短期能力保留，不支持风格改善、已解决三项步态问题、长期稳定或全地形泛化。没有重做多 seed、多速度、多楼梯高度的性能实验，不静默省略，也不因这轮系统修复扩大 GPU 测试范围。
