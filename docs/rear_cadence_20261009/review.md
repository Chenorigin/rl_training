# 后腿交替瓶颈修复独立复审

Reviewer：独立 Codex/GPT family，与主代理同family，未作异构复审声明。生产文件只读；未启动GPU、训练或真机。CPU测试编译实际函数体，使用合成传感器接口，不证明新训练步态已经改善。

## 修改前真实复现

- 后轮脚底维持0m并持续100N承重，滚近下一阶分别获得0.19085、0.21318准备信用。
- 轮位置完全不变，仅一只前轮从不承重变为承重，准备信用0.40404。
- 左后1→右后1同阶后，左后2的安全恢复无严格转换奖励；继续右后3才得一次。

## 修改后真实复跑

源SHA256：`cbb95a341bd945999176b7aa608e62703f0ced46fe94141115df8dd2d5e1ff80`。

1. `scripts/tools/check_gait_v3.py --output docs/rear_cadence_20261009/reviewer_gait_checks.json`：通过。目标每脚raw high-water mark、初始化/门控/选腿/重试/reset、严格转换、恢复事件、退款一次性、同阶循环名义净值−1.5均通过。
2. 独立实际ascent_state轨迹：loaded接近0、loaded滚动0、真实离载抬升0.31690、同姿态重复0、仅支持mask切换0。见 `reviewer_motion_checks.json`。
3. 配置接线：主rear completion与账本都用同一 `TUNING[rear_completion_weight]`；recovery使用同一recovery weight；refund term权重−1且按记录实际style credit归还，三者量纲匹配。

## 复审中真正逼红并纠正的测试

首跑新gait测试失败于recovery事件断言。独立定位：测试将前轮整体放在x1.2m，扫描器移到x0.85m，riser0.3m在scan后向范围−0.45m之外，map列表变为 `[0,.6,.9,1.2,1.5]`，漏掉下一阶；首阶不再被证明是intermediate，因而正确地没有duplicate/recovery。测试前轮位置改为0.6或0.9m后，map连续且生产recovery真实触发。没有为让测试绿而放宽生产安全门控。

这也说明真实运行中若相邻台阶边界未曾扫描到，intermediate/落脚/同阶判断会延迟或欠计；它属于现有高度图可见性限制，应与步态效果分开报告。

## P1

未发现新增P1接线或记账错误。原loaded滚近/资格切换奖励缺口已关闭；unknown expected不再靠历史map_lead选择领奖腿，解除隐藏历史侧偏置。

## P2：退款不等于消除所有时间折扣收益

退款消除未折扣名义style收益，但gamma=.99时，先得3、100个control steps后同阶返3并罚1.5，style折扣现值为 `3-(3+1.5)*.99^100 ≈ 1.353`，可能仍正。不能声明数学上排除了所有延迟策略。一次性历史记账、严格转换判定、准备预算、同阶dwell与任务约束共同作用；应看实测落脚间隔及same_tread/transition指标，不采用无界gamma逆退款。

## 验收限度

代码契约通过，不是全地形任务或交替成功率验收。新模型须检查 rear transition_per_landing、strict run/coverage、recovery events、same-tread、refund、退缩与任务完成；不能把准备信用上涨当作后腿交替已学会。
