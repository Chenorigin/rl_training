# 三项楼梯步态奖励修正与续训

## 改动范围

唯一生产改动是 `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/stair_teacher.py`。原有env配置已经通过TUNING读取权重并调用这些函数，真实Isaac RewardManager接线再次确认。观测244/285、动作16、地形、速度采样、课程、pre_teacher、student、部署均未修改。CPU检查更新 `scripts/tools/check_ascent_direction.py`，新增 `scripts/tools/check_stair_gait_fix.py`。

修改前备份：`.scratch/gait_fix_20261009/stair_teacher.before.py`、`stair_teacher_env_cfg.before.py`。最终source SHA见isaac_smoke.json；与实际测试加载版本完全相同。

## 1. 上楼方向

原有累计横向偏移、命令积分参考航向、速度误差三种定义及合法侧向/转向命令处理保留。成本改为软Huber尾部，增大速度/偏航在成本中的比例，同时下调总权重，以免简单放大方向项损害任务。

实际生产函数横移1→2→4m时未加权成本2.311→3.077→3.519；加权系数-1，仍能区分已经较大的偏移。原实现对应约0.928→0.983→0.996。成本仍有软上界，超过容差仍有连续变化。合成极端例总成本上界12，对应最大理论加权-12/秒；正常任务奖励是否被压制必须看续训数据，代码检查不能证明学习结果。

新增 `Curriculum/ascent_direction/activation_fraction`、`large_error_fraction`，同时保留偏移、偏航、横向速度原指标。large_error_fraction统计进入软尾部的横移样本，不是硬截断占比。

## 2. 下楼后腿前倾

b/g定义保持髋→轮心相对机身/重力向下的前倾角。承重和摆动采用不同软阈值；高台阶仍适当放宽。不对后伸角取绝对值，避免把正常后伸也惩罚。

承重成本有非零载荷增益底值，轻载不再接近零；摆动成本单独计算并给较轻权重，正常摆动保留空间。下降边缘相位持续到两后轮在下层越过边缘并支撑，防止机身前方扫描不再有下沿后，后腿转移失去约束；离开边缘范围或转离原方向退出，episode reset清除相位。

实测合成b=g=49°：旧成本0N=0、6N=0.059；新成本0N=0.132、6N=0.373。承重垂直、后伸、普通平地、纯转向、反向命令均不误罚。

新增 `Curriculum/gait_quality/` 下：
- `rear_swing_samples`、`rear_swing_excess_fraction`。
- `hl_swing_excess_fraction`、`hr_swing_excess_fraction`。
- `hl_loaded_b_p95_upper_deg`、`hr_loaded_b_p95_upper_deg`；g同理。
- `hl_swing_b_p95_upper_deg`、`hr_swing_b_p95_upper_deg`；g同理。

P95为正向角的5°分箱上沿估计，负角归入零箱，不能当作精确带符号分位数。空样本输出0，所以必须同时看样本数。全程统计与原 `rear_entry_*` 入口统计一起观察。

## 3. 后腿交替

保留原严格转换奖励、跳阶/同阶处罚和整机跨阶奖励，不通过硬改动作指定步态。原rear lift奖励现在比较目标下一阶落脚区与轮底的水平/垂直距离：接近目标提高分数，抬得过高或越过目标会降低分数；有明确期待侧时仅该腿能得引导。每个物理目标初始化只记基线，仅分数的新最高值获得差额credit，首次静止姿态不支付；历史目标重访不重新初始化。单目标最大引导总预算有限，错误腿或旧姿态重复不支付。

准备引导本身不是交替成功：支撑资格由其他腿变化变为满足时可能产生一次有限credit；同阶步态也可能获得有限准备分，不能因此宣称交替已学会。主要交替奖励仍以相邻异侧、安全真实落脚结算。

新增 `Curriculum/step_completion/up_rear_transition_per_landing`（前腿同理），应与以下原指标联合判断：
- `up_rear_longest_strict_run`、`up_rear_six_step_strict_fraction`。
- `up_rear_same_tread_fraction`、`up_rear_skip_fraction`。
- `up_whole_body_crossings`、`up_whole_body_crossing_coverage`。
- `up_retreat_fraction` / `up_stationary_fraction`（实际键名以TensorBoard记录为准）。

## 实际验证与限制

1. CPU实际函数体的反例：承重/摆动、左右偏移、合法曲线命令、缓存、批处理、局部reset、下沿暂失、平台退出、引导单调性、反复摆腿、暂停恢复、错误腿、旧目标重访。输出 fix_checks.json、direction.json、gait.json、reviewer_probe.json、reviewer_final.json。
2. 独立同family Codex代理复跑，报告review.md。没有可用异构family，未将同family宣称为异构。
3. 实际Isaac 16环境100步：actor244、critic285、奖励/观测/动作有限；有效绑定的方向权重-1、后腿前倾-2。源码SHA一致，8个分腿P95指标接入。该短测下楼样本数0，不能据此证明下楼改善。
4. 正式train.py实际`--resume`16环境2迭代：从199998加载，完成199998、199999更新，保存独立smoke目录model_199999.pt。模型全部有限，固定LR1e-5、冻结参考确为基准199998；前两次是critic预热，actor按设计未改变。报告resume_smoke.json和resume_smoke.log。未开始正式长训。
5. 首次8环境短测执行到末段但未生成结果，退出卡在Isaac关闭路径，已核验PID仅清理本轮进程；加异常可见性的临时测量入口后，16环境重跑退出0。不将首次未完整验收计作通过。
6. pyflakes在m20_wzh未安装，跳过该工具，不为检查安装依赖；实际函数反例与实际环境执行用于验证。`git diff --check`通过。

## 推荐续训命令

从基准199998开新日志分支；不使用能力已退化的9999或本轮只有两次critic更新的smoke权重：

```bash
conda activate m20_wzh
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Rough-Deeprobotics-M20-StairTeacher-v0 \
  --headless \
  --resume \
  --load_run 2026-10-06_22-06-56_stair_resume \
  --checkpoint model_199998.pt \
  --preserve_actor_from logs/rsl_rl/deeprobotics_m20_stair_teacher/2026-10-06_22-06-56_stair_resume/model_199998.pt \
  --run_name gait_fix_oct09 \
  --max_iterations 20000
```

`max_iterations`是追加迭代数；此命令新增20000轮，最终编号约219997。`preserve_actor_from`在恢复模型后重设能力参考并重新进行50轮critic预热，然后解冻actor；固定低学习率和既有flat能力保护保持。默认并行环境数量继承当前配置，本轮仅验证16环境启动与更新，不宣称已测试默认全规模资源占用。

建议新增2000–5000轮就做相同地形、相同速度的固定对照，联合看严格连续交替、方向误差、承重/摆动倾角和实际通过率。代码修改完成不等于新模型步态已改善。
