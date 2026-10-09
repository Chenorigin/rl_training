# 跳阶与同阶指标独立 CPU 复审（2026-10-08）

## 结论与范围

本次独立复跑通过，未发现新增指标的分母或批量汇总问题。只读取生产代码、执行 CPU 数值脚本，新增文件全部位于本 scratch 目录；未改生产奖励、配置或主测试断言，未运行 GPU、Isaac、优化器或新的策略动态评估。此前审计报告保留在 `.scratch/gait_v3_20261008/reviewer_design.md`。

审计者为 Codex / GPT，与实现者同 family；独立执行成立，异构性未取得。

## 真实执行

使用 `/home/ubuntu/miniconda3/envs/m20_wzh/bin/python`，限制 OMP/OpenBLAS 线程为 1。运行前内存可用约 100 GiB，单进程 CPU 张量夹具规模小于 7×64 个地图槽；此次没有 GPU 占用。

1. `scripts/tools/check_gait_v3.py --output .scratch/step_skip_followup_20261008/reviewer_exact_checks.json`：exit 0，日志 `reviewer_exact_checks.log`。
2. `.scratch/step_skip_followup_20261008/reviewer_metric_aggregation.py`：exit 0，结果 `reviewer_metric_aggregation.json`，日志同名 `.log`。

| 精确运动反例 | 实测 | 判断 |
| --- | --- | --- |
| FL1→FR3 | 首落信用 1；跳阶事件 1；第 3 阶此次完成信用 false；跳阶比例 0.5 | 换侧不能掩盖跳阶 |
| HL1→HR3 | 首落信用 1；跳阶事件 1；第 3 阶此次完成信用 false；跳阶比例 0.5 | 后轴同样拒绝跨过第 2 阶 |
| HL1 HR1 HL2 HR2 HL3 HR3 | 同阶事件 3；已访问中间踏面 3；同阶比例 1；严格覆盖 0；严格连续长度 0 | 后脚双脚逐级同阶不会被严格交替指标认作成功 |
| HL1→HL2 | 首落信用 1；违规 1；第二次后轴完成信用 false | 连续同腿前进不能冒充交替 |
| 跳阶后相邻下一阶 | 两轴下一正确相邻阶均可恢复信用 | 指标保持局部恢复能力，不暗示已完成严格整段 |

后轴双脚逐级同阶的完成与同阶事件总回报为 -4.5；即使暂时有首落信用，事后同阶确认仍取消该踏面的严格资格并产生处罚。

## 分母、无样本与批量核对

独立夹具直接调用当前生产 `stair_step_completion_metrics`，并非重新实现该函数。

- 三环境批量：环境 0 有 2 次首落、1 次跳阶、2 个同阶踏面；环境 1 有 8 次首落、0 次跳阶、0 个同阶踏面；环境 2 无样本。实测两轴跳阶比例均 `1/10=0.1`，同阶比例均 `2/10=0.2`。没有错误地平均各环境比值，也没有因加入空环境稀释事件比例。
- 已知地图共 17 个立面，其中中间立面 15 个、已访问中间踏面 10 个。未来尚未落脚的已知阶不进入同阶比例分母；实测 `visited_intermediate_risers=10`。
- 单独选择环境 `[0,2]`：跳阶比例 `1/2=0.5`，同阶比例 `2/2=1`，已访问中间踏面 2，证明索引与空环境处理一致。
- 最终平台首次落脚加入后：物理首落总数变成 11，跳阶比例 `1/11≈0.090909`；平台不属于中间踏面，同阶比例仍为 0.2、已访问中间踏面仍为 10。
- 四环境均无样本且状态已初始化：新比例均有限且为 0，已访问中间踏面均为 0。**无样本 0 不是成功**，应同时查看物理首落数和已访问中间踏面数。
- 100 组 7 环境随机合成有效分母账本：所有新比例有限、落在 `[0,1]`，与独立总计分母计算精确相等。这只核对统计聚合，不将任意合成账本称为真实运动轨迹。

## 防止指标误读

- `up_*_skip_fraction` = 跳阶首落事件数 / 物理首落事件数。一次跨过多个阶仍是一次跳阶事件；它不是“遗漏阶数/总阶数”，也不是整段楼梯失败率。第二只脚晚落到同阶会被同阶账本识别，不增加此次首落分母。
- `up_*_same_tread_fraction` = 已访问中间踏面中，曾被该轴双脚稳定落脚的踏面比例。它包含延迟同阶落脚；不要求双脚同一时刻在该阶，也不是双脚同阶驻留时间占比。
- `visited_intermediate_risers` 是所选环境合计样本数，`physical_landings` / `skipped_riser_events` 是各环境计数均值；比例使用各环境事件总和，不能直接用样本合计除以计数均值。
- 仍需结合 `strict_coverage`、`longest_strict_run`、各腿稳定踏面数和真实 MuJoCo 序列。首落信用、局部恢复信用、比例零值都不能单独证明整段交替或训练改善。

## 第一轮源一致性（仅统计修改时）

与本目录 `before_stair_teacher.py` 比较：AST 只允许 `stair_step_completion_metrics` 变化，去掉此函数后的整个模块 AST 完全一致，`TUNING` 完全一致。因此新增统计没有修改奖励计算或权重。

- 本次源 SHA256：`e6118080109f365305822c56c0c673cbb449fc5548fb623e75715309d5b7192c`。
- 配置 SHA256：`a16d22f2bc2ad466aa51f555d5381ec40bcdd1a27268d640162c29ff1e71fd33`。

此次跳过新的 MuJoCo 策略 rollout、GPU / Isaac 冒烟和训练效果复测，符合派单范围；本报告只确认该源版本的 CPU 事件判定反例及指标统计正确，不宣称最新模型已修正步态。

## 第二轮：折扣漏洞与相邻换侧支付复审

第一轮使用未折扣总和检验同阶净回报，不能充分排除较晚补另一脚的时序获利。实现者补充了这个具体 P1，本审计独立重放同一轨迹确认，不沿用第一轮的未折扣结论宣称反作弊完整。

### 修复前后真实 CPU 对照

`reviewer_transition_payment.py` 使用实际生产 AST 的 `ascent_state`，记录每个 20 ms 帧的主完成事件、真实 preparation gain 和 duplicate 事件，按 PPO `gamma=0.99` 逐帧折扣。比较源分别为本目录 `before_stair_teacher.py` 与最终生产源。

| 后轴三阶序列 HL1 HR1 HL2 HR2 HL3 HR3 | 修前 | 修后 |
| --- | --- | --- |
| 各阶首次落脚与对侧补脚之间额外停留 0.6 s | 主完成事件 3；折扣风格回报 +0.35455 | 主完成事件 0；折扣风格回报 -5.68375 |
| 额外停留 1.0 s | 主完成事件 3；折扣风格回报 +1.34329 | 主完成事件 0；折扣风格回报 -3.92483 |

这项“风格回报”严格只包含 rear completion（每事件 3）、rear preparation（gain×0.5）、duplicate（每事件 -4.5）；不包含泛用速度/姿态/能耗或同阶 dwell 项。修前与修后轨迹、准备 gain 和 duplicate 时序完全相同，三阶 preparation units 合计 1.883333，只改变主完成事件支付条件。

### 支付逻辑与反例

- 主事件在更新 `axis_last_slot`、`map_paid`、`map_correct` 之前判定。当前落脚必须是单侧、安全、顺序合法、相邻且未跳阶；前一物理踏面必须同 flight、单侧曾稳定落脚、安全、未 duplicate，且上一侧与当前侧相反。新 flight 先将 last 视作 -1，不跨 flight 支付。
- **两轴两种首迈**：独立夹具分别重放前轴与后轴四个连续正确落脚，均得到几何合法计数 4、主 transition 3；初次落脚不支付主完成。主检查进一步验证每轴两种首迈的八阶序列，每轴 transition 7。
- **首迈准备信号存在**：独立 rear 左首迈与右首迈均可获得初始 preparation units 0.933333（权重后约 0.466667），主完成 0。此项只证明奖励可达，不证明 PPO 探索一定足够。
- **后轴独立**：前轴完成计数始终 0，后轴四个正确落脚仍支付 transition 3。实现没有使用 front_successes 解锁后轴支付。
- **同阶后恢复**：HL1 HR1 后，HL2 单侧首落不支付；随后 HR3 的安全相邻换侧支付 1。不会把双侧站姿中不可见的历史首侧永远设为硬目标。
- **跳阶后恢复**：HL1→HR3 的跳阶不支付；安全单侧 HR3→HL4 相邻换侧可支付 1。前阶只要求物理安全，不错误地要求前阶曾获得 correct / 主完成信用。
- **同帧 previous duplicate**：先 HL1，再零命令期间 HR1 补脚及 HR2 前进。零命令期间地图接触历史保留而支付暂停；恢复正速度的同一帧里，前阶 duplicate event=1、当前几何合法累计=2，但主 transition 累计仍为 0。这验证读取的是本帧已更新 `map_landed`，即使旧 duplicate_paid 尚未置位也拒付。

对旧源执行 `--assert-new` 明确 exit 1（首次落脚仍错误地产生主事件），日志 `reviewer_transition_expected_red.log`；对新源相同检查 exit 0，保留红绿判别证据。结果为 `reviewer_transition_before.json` 和 `reviewer_transition_after.json`。

### 隐藏历史与监控解释

Actor 观测不新增 phase，仍可从当前本体姿态和地形看出两脚位于相邻不同高度；历史账本用于防重复支付和迟到同阶检测。双脚同阶后的 expected=-1 允许任选一侧重新起步。CPU 不能证明无历史 actor 在所有局部观测混叠状态下都能唯一判断奖励目标，也不能保证 preparation 与泛用跟踪足以从预训练权重探索新步态；需使用实际闭环训练和 sim2sim 对照验证。

`transition_successes` 是实际主事件累计支付数；`*_successes` 仍记录局部几何合法的首落，`strict_coverage/longest_strict_run` 还会被迟到同阶落脚取消。不得把任何一种计数单独称为整段严格交替完成。

此前已经得到合法 transition 后又很晚补脚的任意轨迹，仍通过未来 duplicate 处罚纠错；这次修复针对用户“每阶先一脚、再另一脚同阶”的反复追赶获利，**不证明任意延期、任意泛用奖励组合都绝无 reward hacking**，也不宣称奖励可保证唯一或“完美”步态。

### 最终统计与源版本

最终源重新运行独立批量 fixture（100×7 环境）及完整 `scripts/tools/check_gait_v3.py`，均 exit 0。原比例、无样本、已知/已访问分母、最终平台处理结论不变。证据为 `reviewer_metric_aggregation_final.json` / `reviewer_final_exact_checks.json` 及各自日志。

与第一轮之前的源比，最终变化仅在 `_new_state`、`ascent_state`、`stair_step_completion_metrics` 三个函数：加入实际 transition 支付计数与门控及统计。去掉这三函数后的整个模块 AST 一致；TUNING 一致，配置哈希不变。

- 最终审计源 SHA256：`cb4465a2f342c3154725a20a838de35eb12fd5c20f577bf344b40916e2256785`。
- 配置 SHA256：`a16d22f2bc2ad466aa51f555d5381ec40bcdd1a27268d640162c29ff1e71fd33`。

本轮无剩余已复现但未修复的范围内 P1。未运行 GPU、Isaac、优化器、新策略 rollout，未改生产文件；最终判断是 CPU 支付条件与指标统计通过，实际训练效果尚待验证。
