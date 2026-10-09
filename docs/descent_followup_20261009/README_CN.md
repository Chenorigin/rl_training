# 下楼入口后腿前倾：复测与修正

## 当前模型与参照

冻结最新完整model_214400.pt（2026-10-09_01-55-35_gait_fix_descent_entry）；对照既有baseline model_199998.pt固定MuJoCo数据。实际M20.xml，15cm高/30cm深/5阶，固定vx=.5，200Hz物理/50Hz策略，无键盘FSM。最新实际模拟20s未跌倒。入口统计区间root_x∈(0,1.1)m，中段[1.1,2.4)m；这是固定XML的分段，不是普遍地形分类。

| 入口后腿角度 | baseline | 最新214400 |
|---|---:|---:|
| 承重body b P95 | 39.43° | 43.49° |
| 摆动body b P95 | 50.53° | 51.26° |
| 承重body hip-relative X P95 | .302m | .343m |

中段最新承重b P95只有2.06°，因此本次问题主要发生于从顶端跨入下降楼梯的转移阶段。这里只比较一种固定地形与命令；不能代表全部下楼场景。测量使用四元数逆变换得到相对本体角b，分别统计承重与非承重，不用均值掩盖瞬态峰值。

## 曲线与运行事实

同一训练run已导出的TensorBoard快照至214075（见../rear_cadence_20261009/tensorboard）：
- rear_forward_cost平均加权贡献约−.0028→−.0033，entry_retract平均接近0，entry_position_cost平均仅约−.0001。
- 后腿loaded b P95分箱上沿约24°→27°，没有改善。混合地形曲线平均与固定入口P95不能直接作同一分布比较。
- 实际旧轨迹反事实生产reward回放，入口started=1但正信用0：初始rear X已在目标带内，best=1，之后前倾再恢复也不产生新增最高值。该机制原本用于防反复刷奖；它不能担当维持姿态的奖励。
- t3.22：loaded b约46°，原入口position成本.8399。
- t3.32：后轮射线ground降低至.6m，但后轮力为0。原入口active因gone.all立即结束；此时b约51°。
- t3.42：b42.5°/44.2°低于旧swing45°，原持续前倾成本和入口成本均为0。入口约束空档复现。

回放为实际MuJoCo动力学轨迹+生产奖励函数CPU接口适配；接触时间和nearest height适配与Isaac不完全一致，不宣称精确重建训练奖励。

## 修改（统一在stair_teacher.py）

1. 入口物理状态不随零/反向命令销毁。奖励门控仍要求前行下降；原最高值和入口身份账本保留，恢复命令不能重新发奖。
2. 后脚完成入口要求实际在低层稳定承重、脚底与地面误差<4cm。射线降高、无载跨沿、仅一脚落地均不能关闭整体入口；空间/朝向退出和reset仍可结束。
3. 连续入口位置成本覆盖承重及摆动，直至两后脚都已稳定下层。承重带保持原−5cm至≤15cm；摆动适度放宽−10cm至≤22cm，并受腿深度与35°夹角限制，台阶高度可放宽。具体数值唯一台账为TUNING。
4. 持续摆动前倾阈值body35°/gravity25°（旧45°/35°），mix .75（旧.4）；承重25°/20°阈值及高台阶放宽规则保持。
5. 后腿前倾及入口位置cost采用2*(sqrt(1+excess²)−1)并软上限4，增强严重偏差的区分；入口position RewTerm权重−1（旧−.25），持续rear_forward权重仍−2。
6. 入口摆动成本取位置与过短展开成本中较大项。低台阶摆动展开软下限.30m，高台阶放宽.26m，防仅折腿缩小前伸量；不额外叠加承重折腿项。
7. 正入口奖励仍有原一次最高值预算，并只在原台面支持阶段支付。没有新增常驻正奖励鼓励停车，也没有跟随每阶重置后收预算。

前腿、上楼后腿交替/恢复/退款、观测、命令、地形和PPO保持。已有训练进程不会热加载这些修改；后续新进程续训才采用。

## 同一条轨迹的奖励对照（不是新模型改善）

两次回放的root_x逐帧完全相同，actor/物理控制未变。

| 原轨迹帧 | 旧持续前倾成本 | 新持续前倾成本 | 旧入口成本 | 新入口成本 |
|---|---:|---:|---:|---:|
| 3.22s | .523 | .732 | .840 | 1.714 |
| 3.32s | .026 | .342 | 0 | .945 |
| 3.42s | 0 | .121 | 0 | .639 |

14s回放累计加权前倾罚绝对值.1465→.2879；累计加权入口罚.0281→.4727。仅处罚真正转移阶段，非全程乘大系数。增强处罚可能使下楼更谨慎或慢，需任务通过率和速度同时验收。

## TensorBoard看什么

前缀 `Curriculum/gait_quality/`，hl/hr分别表示左/右后腿：
- hl_loaded_b_p95_upper_deg / hr_loaded_b_p95_upper_deg：承重前倾分箱P95上沿，越低越好，但必须看样本量。
- hl_swing_b_p95_upper_deg / hr_swing_b_p95_upper_deg：摆动前倾。
- hl_entry_transfer_samples / hr_entry_transfer_samples：入口到真实落地阶段样本数。
- hl_entry_transfer_excess_fraction / hr_entry_transfer_excess_fraction：该阶段位置越带比例。
- hl_entry_transfer_swing_fraction / hr_entry_transfer_swing_fraction：摆动覆盖。
- hl_entry_transfer_compact_fraction / hr_entry_transfer_compact_fraction：摆动腿展开过短比例（分母为摆动样本，零样本不解释为成功）。
- hl_entry_transfer_cost_mean / hr_entry_transfer_cost_mean：覆盖整个转移的综合成本。

原entry_retract_credit接近0未必异常：初始已在目标带就不该额外领一次奖励。改善判据是同地形入口角度峰值/P95下降、轮位越带比例下降，同时下楼通过率、动作平滑和速度没有明显退化；不能用total_reward或罚项更负单独判成功。

## 验证状态

- check_descent_entry：initial good→bad持续处罚、lower ray+airborne、单后脚落地、另一脚悬高、双脚真实稳定低层、暂停恢复、折腿逃避、正常摆动、原入口防刷/reset均通过。
- check_stair_gait_fix、check_gait_v3上楼回归通过；独立review比较上楼/前腿函数AST与备份保持相同。
- 真实Isaac16环境100步，加载214400，新TUNING与RewTerm接线、244/285观测有限；短程未产生下降正事件，不能把接线检查当下楼效果评估。
- CPU实际MuJoCo单线程，RAM可用约102GB；Isaac运行前GPU空余约42GB，预算3GB；未启动长期训练，未停止用户进程。
- 新策略姿态改善尚未验证，须启动新run后固定MuJoCo复测。使用前文rear_alternation_fix续训命令即可同时应用此修正，亦可把加载checkpoint换为本次测过的214400。

最终生产源SHA256：`8ce61820368355126ce33ee9ac6b29f4b3cc80c8bd3eca0c73a78f72ef1ae05c`；与真实Isaac接线测试一致。
