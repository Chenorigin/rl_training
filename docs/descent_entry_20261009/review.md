# 下楼入口后轮后收引导独立复审（2026-10-09）

独立 reviewer：Codex/GPT family；与主代理同 family，不是异构复审。只读检查生产代码，未启动 GPU、训练或真机。

## 已真实执行

- 查看 `.scratch/descent_entry_20261009/stair_teacher.before.py`、`env.before.py` 对当前生产源 diff。
- 查看 `position_analysis.json`，后轮 hip 相对x良好样本主要0–0.15m、明显前倾样本约0.21–0.33m；只属于所采固定轨迹证据，不能概括所有台阶。
- 单线程执行 `scripts/tools/check_descent_entry.py --output docs/descent_entry_20261009/reviewer_checks.json`，初版全部通过。使用实际生产函数 AST 与合成接口，非物理步态结果。
- 编写并真实执行 `.scratch/descent_entry_20261009/reviewer_entry_probe.py`，初版发现下列重复领奖反例，输出 `reviewer_entry_probe.json`。

## 初版发现 P1：无实时下沿扫描时物理入口身份使用移动机体位置

当 `down_gate=0`、`edge_x=0`，但 retained descent phase 保持 `down=true` 且前脚已在低层 (`front_lower=true`) 时，入口允许启动；历史key却仍由 `scanner_pos + heading*edge_x` 构成，成为机体扫描器当前位置而非真实台阶沿。

实际反例结果：第一次后轮从x0.3收至0.1信用0.69643；退回原top、零命令26帧后ready=true；同物理入口再次进入时扫描器沿x移动0.3m，信用再次0.69643，started=2。此为合成接口反例，不声明机器人实际已采用刷奖动作。

建议修复：有有效实时下沿则使用扫描出的物理下沿；无实时下沿则使用已锁定 `_m20_descent_pose_phase.xy/heading`；两者均无则关闭新入口启动，不以当前位置替代。历史沿方向投影判定可容许横向漂移，但需要真实沿坐标。

## 初版其余检查

- 初始姿态不领奖；往返重复姿态无新增信用；每脚score新增最高值的总预算≤1。
- 停止、反向躯干、明显世界后轮反滚、支持切换不会补领此前积存分。
- 后脚离开原top即将该脚最高分封为1；不跟随之后每一级楼梯。
- x下界−0.05m兼顾过后收处罚；上界受伸展深度与20°姿态限制，35cm深度时0.1274m、45cm深度时0.15m。具有高度放宽而非固定硬落点。
- `lower_z` / `upper_z` 是沿前后扫描列高度；改成min作为低层高度在下降情况下正确。
- 新RewTerm均使用 `_gait_entities()` 并调用canonical stair_teacher；一项event gain除dt，一项连续cost保持每秒权重，量纲正确。
- 单环境reset对应张量行归零；同一 `_context` step缓存避免两项RewTerm重复结算。

## P2 / 待验证

- 引导允许世界接地点保持而躯干向前推进使相对x减少，符合用户所述后收，不强制轮子世界倒走；但它是位置包络，不保证动作一定呈现为腿主动后摆。
- band是基于少量固定轨迹的软范围，须续训后验证下降通过率、关节/力/角度，不据CPU正确宣称更安全。
- `stair_descent_entry_position_cost` 相比其他任务项权重较小，保持停滞时也处罚，不能用增加信用代替任务完成验收。

初版结论：P1身份漏洞需修复后复跑，其他CPU检查通过。

## 第二轮：P1修复后真实重跑

主代理已将入口身份改成实时扫描物理下沿，或有效 retained descent phase 的物理xy/heading；两者都没有则关闭启动。helper的scanner_pos可以显式传入，生产_context使用sensor_cfg实际绑定位置，消除硬编码传感器选择问题。

reviewer修改自己的反例接口以提供固定真实锁定phase（不修改生产）。真实重跑结果：首次正credit0.69643；退回暂停可ready；scanner沿x移动0.3m，同物理入口第二次credit0、started仍1。随后关闭锁定phase并reset，无实时沿条件credit0、started0。该反例同时证明“允许正确首次引导”与“阻止重复/未知身份”，并非把所有入口禁用伪装通过。

再次真实运行 `check_descent_entry.py` 全部通过，输出 `reviewer_checks.json`。P1已关闭，剩余P2为续训姿态与任务能力的闭环验收；当前CPU代码契约复审通过。尚未在本reviewer启动Isaac/GPU，不能声称真实训练接线已测。

## 最终版本后收展开约束复审

最终生产源SHA256：`d9f9ef07d53693cfa72b8ae736e292cef372a0de4bb2155ec1cbf20db7c751dd`。

主代理补上healthy_reach：后收正奖励需要已有rear_min_extension腿长0.35m与垂直展开0.30m，防止仅折腿缩短使x进入区间。reviewer真实再次执行 `check_descent_entry.py`，新增 `collapsed_leg_unpaid=true` 和 `extension_cycle_unpaid=true`，其余反例全部通过。

reviewer同时复跑物理phase身份反例：首次有效credit0.69643、同入口重新进入credit0、started1；关闭phase后未知沿credit0/started0。无新增P1。

本阶段CPU最终复审通过；最终版Isaac接线100步与train.py resume两迭代，由主代理独立核验，不在本报告冒充已完成。
