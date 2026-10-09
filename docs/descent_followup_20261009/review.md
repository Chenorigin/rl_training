# 下楼后腿前倾跟进修复独立复审

Reviewer为独立Codex/GPT family，与主代理同family，未宣称异构复审。生产源只读，未启动GPU、训练或真机。

最终核验源码SHA256：`d7c118093bdb8dbcc3a8710cebd2a0c3e5223291d100b01dfb8b4febdabf51da`。

## 根因证据的独立读取

实际读取回放，旧实现3.22秒后腿b约46°，入口处罚0.8399；3.32秒后轮仍无力，ray ground从0.75降到0.6，entry_active立即false；3.42秒b42.5/44.2°却global和entry处罚均0。文件scope明确为真实MuJoCo轨迹上的生产奖励反事实回放、接触时间适配不同，不等于真实Isaac训练中的逐帧奖励。

## 真实执行

- `scripts/tools/check_descent_entry.py --output docs/descent_followup_20261009/reviewer_entry_checks.json`：退出0。
- `scripts/tools/check_gait_v3.py --output docs/descent_followup_20261009/reviewer_gait_checks.json`：退出0。
- 对修改前备份做AST函数比较，ascent_state、rear_preparation_gain、rear_target_score、rear completion、recovery、refund及front fold函数体全部相同，确认没有改动刚修好的前后交替逻辑。

## 新反例结果

1. 初始入口姿态良好、score已满后变坏：gain仍0，连续成本1.34260，说明不依赖正准备分才约束。
2. 只有ray降低但后腿无力：入口保留且成本为正。
3. 单后轮真正在低层稳定承重：入口保留；另一脚射线在低层但脚底仍悬高也保留。
4. 两脚先后真正稳定落在低层、脚底高度和有效ground吻合：settled标志全真，入口关闭。
5. 零/关闭奖励命令不销毁物理状态，恢复后连续成本再次为正；无重新预算或支持切换补领。
6. 无效地图、原地反滚、过后收、折腿缩短、展开往复、入口重复历史与reset回归通过。
7. 双倍Huber软界有限且单调，u=.5/1/2/3成本约.223/.686/1.528/2.078，而非早期平方项快速饱和。

## 同轨迹新旧成本

独立读取before/after反事实回放：

|时刻|旧全局|新全局|旧入口|新入口|新入口状态|
|---|---:|---:|---:|---:|---|
|3.22s|.5228|.7324|.8399|1.7139|active|
|3.32s|.0262|.3422|0|.9450|active|
|3.42s|0|.1210|0|.6391|active|

因此确实补上先前实际轨迹中最关键的无载转移处罚空档；这是对奖励函数的验证，未重新训练轨迹不能称姿态已经改善。

## P1与限制

未发现新增P1实现/接线错误。低层结束判定使用有效ground、真实加载/接触时长、脚底-ground误差；物理phase与命令门控分离正确。

P2：全局与入口成本同时提高，有可能使下楼动作保守，需监控下楼通过率、速度、停滞、后退与loaded/swing P95；不能只观察Episode_Reward或正retract信用。入口正分仍有有限最高值预算，初始score1导致整个实际回放gain_total=0属于设计结果，并不代表连续guard失效。

CPU代码契约复审通过；未在本reviewer执行最终Isaac接线或续训闭环评估。

## 最终补充：摆动折腿逃避位置处罚

真实读取并单独执行新compact guard。最短展开由单台账定义为0.30m、25cm高放宽到0.26m；连续成本取position与compact中较大值，compact只针对非承重后腿。

独立实际函数体两环境反例：x0.05m/depth0.15m摆动虽然处于位置band，cost0.806895；depth0.35m正常摆动位置处于band，cost0；相同短腿改为承重且位置处于band，cost0，未重复施加loaded折腿处罚。输出 `reviewer_compact_checks.json`。

这轮核验源SHA：`8df8ef14d2929a9d471cff86beec1cfc5d53a364d2ede1265433c8fee85f8afb`。无新增P1。

P2监控备注：初版transfer_excess仅统计guard_error，新增compact处罚可能cost>0却excess=0；已通知主代理用完整cost触发或独立compact指标反映该项。此备注不等于奖励未生效。
