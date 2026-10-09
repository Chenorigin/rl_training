# 后腿交替瓶颈：诊断与修正

## 冻结参照

- 检查开始时最新完整权重：`2026-10-09_01-55-35_gait_fix_descent_entry/model_214000.pt`，不跟随仍在训练的后续检查点滚动。
- TensorBoard读取快照至214075；前500与后500记录比较。所有地形/命令混合统计，terrain_level总体2.447→5.899，不能作为固定难度因果实验。
- Actor244 / critic285 / action16不变。MuJoCo使用实际M20.xml、200Hz物理/50Hz策略、无键盘输入、固定起立姿态，保留原跌倒判据；不是实时viewer截图实验。
- GPU实时49GB/已用6.8GB、RAM可用约101GB；MuJoCo与数值探针CPU单线程；Isaac接线检查16环境100步、不训练、预算3GB。未停止用户训练。
- 仓库有大量已有修改及暂存删除；origin的额外提交为M20S改动，未拉取覆盖当前M20工作树。

## TensorBoard

| 指标 | 前500 | 后500 |
|---|---:|---:|
| 前腿 transition_per_landing | .03782 | .06989 |
| 后腿 transition_per_landing | .00920 | .01030 |
| 前腿 longest_strict_run | .4264 | .5270 |
| 后腿 longest_strict_run | .3437 | .3790 |
| 后腿 six_step_strict_fraction | .0002 | .0003 |
| 后腿准备累计信用 | 2.2344 | 2.2347 |
| 后腿准备加权奖励 | .05586 | .05587 |
| 后腿转换加权奖励 | .01182 | .01277 |

这些曲线支持后腿交替信号稀少、前腿相对更有进展，不支持持续交替已经成熟。准备信用增加不能等同于落脚完成。完整233项见tensorboard/tensorboard_summary.json和曲线png。

## 实际MuJoCo结果

- 15cm高、30cm深、20阶、vx=.7：前腿 FL1@1.16s→FR2@1.50s→FL3@1.86s→FR3@2.16s；随后两脚同阶。后腿 HL1@2.10s→HR1@2.32s→HL2@2.44s→HR2@2.64s，持续同阶。
- 4–8.5s各轴首次到达新高度的推进率：前2.564阶/s、后2.538阶/s；前后轮中心平均轴间距中位0.734m。不能据此断言后腿速度只有前腿一半。严格交替、每只脚摆动次数、整轴新阶推进是不同指标。
- 20cm高、25cm深、20阶、vx=.5：前后推进率1.872/1.860阶/s，轴间距中位0.632m，前腿开始也两脚同阶。
- 两组均出现明显偏航，横向偏移最大均约1.97m。15cm组已越过楼梯终点，随后在顶部区域10.98s跌倒；20cm组未越过终点、13.22s跌倒。因此“后腿同阶未改善”复现，但失败并非只由后腿拖累；不声称这两个固定样本证明全地形通过率或唯一因果。
- first landing历史去重统计，不使用原GaitMetrics的first-highest换脚率冒充“另一脚从未补踩同阶”。详见chronology.json、各case npz/summary.json。

## 已定位的奖励漏洞

独立生产函数反例：承重轮脚底保持0m，滚近台阶也得.19085/.21318信用；不动脚、只改变前轮支持也得.40404。历史lead在同阶后可能仍锁定下一目标一侧。修复前同阶后的首次安全恢复只作seed，无直接主转换奖，探索困难。

实际轨迹反事实奖励回放：旧准备信用9.472、其中承重信用.579；新7.622、承重信用0。适配器接触计时和扫描观测与Isaac不同，此结果只检验生产函数在该轨迹上的响应，不是精确重建训练回报；地图可见性/偏航也影响台阶归属。

## 修正

1. 后腿准备奖励只在真实离载、脚底相对源台面净抬升>4cm、其他至少2个轮支持、靠近相邻目标且不越过下一沿时可支付。几何分数乘归一化净抬升；承重滚近不支付。
2. 每目标每后腿分别维护未乘门控的几何最高分；初始化、支持切换、换腿、退回旧姿态不补领；所有可观测姿态都更新最高值。每物理目标双方合计准备信用≤1，权重保持.5。
3. unknown expected按当前合法新增进展选择后腿，不沿用隐藏历史lead。已确定交替侧仍必须使用正确侧。
4. 同阶后，单后脚安全进入紧邻上一阶、伙伴仍稳定支持原阶，给小额once恢复奖1；它不计严格转换，随后伙伴应该迈到再上一阶才能得到原严格转换3。
5. 若目标后来被另一后腿补踩成同阶，一次退回该阶此前实际发出的后腿严格奖3或恢复奖1，并保留原同阶惩罚。全身穿越任务奖励保持独立。
6. 新曲线前缀 `Curriculum/step_completion/`：up_rear_prep_blocked_gain、up_rear_recovery_events、up_rear_recovery_per_landing、up_rear_style_refund。主验收仍看up_rear_transition_per_landing、longest_strict_run、six_step_strict_fraction、same_tread_fraction以及任务通过，不追求准备分升高。

## 自检与限制

- check_gait_v3：严格八级左右起步、前后独立、跨级、历史同阶、reset、raw maxima支持切换、恢复、重复退款通过。
- check_stair_gait_fix与check_descent_entry回归通过；未调整前腿、下降、观测、速度命令、PPO或地形配置。
- 独立复审实际生产ascent_state：承重滚近0、真实摆动.31690、支持切换0、重复0；同阶恢复后补踩名义style净值−1.5。review.md记录同family复审，不宣称异构。
- 真实Isaac16环境100步、加载214000，244/285观测有限，新RewTerm实际注册权重1/−1。短程中后腿相关新正事件为0，不能据此宣称已在Isaac学会交替。
- 时间折扣下延迟退款仍可能留下正现值；未观察到邻沿时intermediate可能欠计；新训练后的步态效果尚未验收。有限高水位信用也不是完整势函数，不能保证PPO最优策略不变。
- company-rules check可机检条款未发现问题，但全机目录分发扫描有权限跳过；不能报全机检查通过。
- 当前训练进程已导入旧代码，不会自动采用修改；新run需重新启动。未自动开始长期训练。

## 建议续训

保留已得到的前腿初始交替，从本次实际测过的214000续训；沿用199998任务能力参照，先短程观察，避免直接无人值守跑数万轮。max_iterations为新增迭代数。这里的正式续训命令未实际启动；已验证checkpoint actor与新环境接线。

```bash
conda activate m20_wzh
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Rough-Deeprobotics-M20-StairTeacher-v0 \
  --headless --resume \
  --load_run 2026-10-09_01-55-35_gait_fix_descent_entry \
  --checkpoint model_214000.pt \
  --preserve_actor_from logs/rsl_rl/deeprobotics_m20_stair_teacher/2026-10-06_22-06-56_stair_resume/model_199998.pt \
  --run_name rear_alternation_fix \
  --max_iterations 5000
```

后续固定验收必须使用相同命令/地形，看HL1→HR2→HL3及相反先手是否能连续≥6阶，且通过率/方向误差不恶化。恢复事件多但严格转换不上升仍是不合格；退款高表示补踩反复发生。不得以全局reward上涨代替。
