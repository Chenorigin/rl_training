# 下楼入口后轮后收引导

## 定义与测量

机身坐标系 X 向前，x 为 `wheel_center - hip_center` 经过完整机身姿态逆旋转后的X分量。正值表示轮心在髋前，负值表示在髋后；不以世界坐标后退代替相对后收。

分析已有真实MuJoCo物理轨迹：基准model199998、旧最新model9999，5级15cm高、30cm踏深下楼，vx=0.5。入口测量只选前轮已有下层承重、后轮仍在原顶层承重的样本。原始数据来自 docs/gait_audit_20261009，统计 position_analysis.json。

| 入口样本 | 基准199998 | 旧最新9999 |
|---|---:|---:|
| 承重后轮x中位数 | +0.142m | +0.168m |
| 承重后轮x P95 | +0.318m | +0.334m |
| b处于−10°到25°的子集x P5/P95 | −0.042/+0.184m | −0.042/+0.181m |
| b>25°子集x中位数 | +0.270m | +0.276m |
| 温和角度子集垂直展开中位数 | 0.439m | 0.443m |

角度子集是按已有姿态定义挑选，不是独立“成功/安全”标签；样本数少，来自单一固定楼梯。目标区间是据此制定的初始软包络，不是已经优化或安全认证的参数。

## 软目标区间

基础下界−0.05m，上界+0.15m。上界随腿垂直展开d和台阶高度调整：

`upper = min(0.15 + 0.02 * relax, d * tan(20° + 5° * relax))`

relax沿用当前15–25cm台阶适应量。普通15cm台阶，垂直展开35cm时上界约12.7cm，45cm时为15cm；高台阶最大位置上界17cm。目标允许轮心适度在髋前，不强制后腿后伸。x超过上界的前伸样本向后收有正向学习信号；x小于下界的过度后伸也有位置处罚。

区间与所有奖励参数只在 stair_teacher.py 的TUNING作为可执行台账。

## 入口相位与两项奖励

- 看到近处下降边缘，或者已有前轮下层支撑、两后轮仍在同一原顶层稳定支撑时，锁定本段楼梯入口。
- 记住顶层高度和物理边缘；后续不随扫描改成每一级台阶。
- 每只后脚离开原顶层即永久关闭该脚本次入口信用；两后脚离开或转离/远离入口时结束。
- 平地稳定后允许检测新的楼梯段；本episode物理入口历史阻止退回再进入刷奖。无实时/锁定沿信息则关闭新启动。

### stair_descent_entry_retract

score随到区间的距离衰减，首次只记录基线，仅新的最好score得到差额。左右分别累计，总加权奖金每次入口不超过0.75。初始就在区间内不会白领奖。

正奖励还要求：后轮仍在顶层稳定承重，另外至少两轮支撑，机身实际向命令前方移动，世界系后轮沿入口前向速度不低于−0.05m/s，机身未严重倾倒，后腿长度≥既有0.35m下限且垂直展开≥0.30m。不能用停止、倒退、明显反滚或折腿缩短来获取后收奖金。被阻止时的几何进度也记录，不会在开闸后补领。

### stair_descent_entry_position_cost

只在入口有效后轮支撑阶段计算到区间的双侧越界成本，权重−0.25。防止原先位置合适、之后又向前伸，或过度后伸；不在摆动下阶阶段强迫后轮保留坡顶姿态。现有b/g承重和摆动前倾约束继续生效。

## 附带修正

原扫描 `lower_z/upper_z` 是沿前向的前侧/后侧列，并不保证高度大小顺序。此前下降相位把 `lower_z` 当作物理低层，有提前结束约束的风险；现改成 `minimum(lower_z, upper_z)`。这项修正不改变actor高程图数值定义。

## TensorBoard

在 `Curriculum/gait_quality/` 下新增：

- `descent_entry_started`。
- `hl_entry_samples`、`hr_entry_samples`。
- `hl_entry_x_mean_m`、`hr_entry_x_mean_m`。
- `hl_entry_in_band_fraction`、`hr_entry_in_band_fraction`。
- 各腿 `entry_too_forward_fraction`、`entry_too_rearward_fraction`。
- 各腿 `entry_retract_credit`、`entry_blocked_fraction`。

同时看已有后腿承重/摆动b/g P95、下降入口角度、速度误差、通过率。信用大或x小不能单独证明步态更好。

## 验证与诚实边界

- CPU真实函数体反例：初始不领奖、往复、停止、机身倒退、后轮反滚、支持切换、过度后收、折腿、单行reset、无效地图、仅首阶、同入口侧移/回退重入。脚本 scripts/tools/check_descent_entry.py；checks.json。
- 独立同family代理发现无实时边缘时的入口身份P1；锁定物理边缘修复后，再测首次信用0.69643、同入口重入0、未知入口0。见review.md和reviewer_entry_probe.json。未宣称异构复审。
- 实际MuJoCo700控制步、原actor无改动，使用实际射线和关节/接触测量适配生产奖励函数：入口激活1次约0.98秒，左右支撑样本36/33。基准初始已在目标区间，所以奖金0；越界位置处罚累计约−0.0264。该探针是MuJoCo数据上的反事实生产奖励，不是Isaac原生接触时间，也不是新模型效果。
- 实际Isaac16环境100步验证新奖励绑定、指标、有限观测/奖励；输出 isaac_smoke.json，并核对与最终源SHA一致。短测不能证明训练后改善。
- 正式train.py16环境2迭代resume检验输出独立descent_entry_smoke目录；两轮属于critic预热，不证明actor步态改善。最终加载与保存检查见resume_smoke.json/log。
- 未启动正式长训、未测试默认全规模显存预算、未做真机验证。资源检查GPU总49GB/已用约0.85GB、可用内存约106GB；测试采用与此前成功小环境短测相同的16环境预算，顺序运行GPU测试，输出为少量JSON及约8MB测试checkpoint。

生产改动：mdp/stair_teacher.py、stair_teacher_env_cfg.py（新增两项RewTerm）；测试新增check_descent_entry.py。观测、网络、动作、上楼交替、方向项、地形、命令、pre_teacher和部署源码没有改动。修改前备份在 .scratch/descent_entry_20261009。

## Resume

建议从已验证任务能力的199998开新分支。不要把只有两轮critic更新的smoke权重当作训练成果。

```bash
conda activate m20_wzh
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Rough-Deeprobotics-M20-StairTeacher-v0 \
  --headless \
  --resume \
  --load_run 2026-10-06_22-06-56_stair_resume \
  --checkpoint model_199998.pt \
  --preserve_actor_from logs/rsl_rl/deeprobotics_m20_stair_teacher/2026-10-06_22-06-56_stair_resume/model_199998.pt \
  --run_name gait_fix_descent_entry \
  --max_iterations 20000
```

追加20000轮，先50轮critic预热，再更新actor；2000–5000轮先做固定楼梯对照，不凭总reward上涨判断姿态改善。
