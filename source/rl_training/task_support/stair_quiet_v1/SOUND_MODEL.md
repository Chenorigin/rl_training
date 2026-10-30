# 轮足冲击的离线声压预测：未校准的等效声源模型

本模块将显式假设的撞击脉冲传入有阻尼结构模态，再由等效体积加速度计算自由场声压，输出单位 **Pa**。它不是把接触力取对数。输出固定为 `uncalibrated_simulation`、`calibrated: false`、`actual_pass: false`；没有实际机器狗声学验收能力，不能据此宣称真机砸地声分贝已经下降 50%。

本模型的用途是：在相同、公开的声学假设下，对两份独立仿真事件进行电脑预测，检查结论是否依赖接触时长或阻尼选择。默认参数是便于验证的假设等效声源，没有测量过 M20 轮胎、轮毂、楼梯或房间，也没有为了某个降幅拟合参数。

## 方程与一手依据

对一个从 `t_i` 开始、法向动态冲量为 `J_i` 的撞击，假设接触时长 `T` 内的力为：

```text
F_i(t) = π J_i / (2T) · sin(π(t-t_i)/T),  t_i <= t <= t_i+T
∫F_i(t)dt = J_i
```

`T` 是 profile 中公开的假设。200 Hz 刚体仿真只能提供低时间分辨率接触证据，无法恢复真实接触脉冲或可听高频。这里不会声称将这些未知高频“重建”出来。

每个模态的广义位移 `q_j` 满足：

```text
m_j q̈_j + 2 ζ_j ω_j m_j q̇_j + m_j ω_j² q_j = φ_j F_i
ω_j = 2πf_j
```

`m_j` 是模态质量 kg，`ζ_j` 是阻尼比，`φ_j` 是无量纲力参与因子，`q_j` 单位 m。外力驱动独立模态并叠加属于常见模态合成形式，见 Julius O. Smith III 的作者教材 [Modal Expansion](https://dsprelated.com/freebooks/pasp/Modal_Expansion.html)。本程序并未从真实结构辨识其模态。

将各模态的有效体积变化定义为 `V_j=A_j q_j`，`A_j` 是有效辐射面积 m²。对接收点到第 `i` 次撞击的距离 `r_i`：

```text
p_i(t) = ρ_air / (4πr_i) · Σ_j A_j q̈_ij(t-r_i/c)
p(t)   = Σ_i p_i(t)
```

单位为 `kg/m³ × m² × m/s² ÷ m = Pa`。这是无限自由场中的紧致等效单极子近似。Gade、Møller、Hald、Alkestrup 的 Inter-noise 2004 论文 [The use of Volume Velocity Source in Transfer Measurements](https://www.bksv.com/media/doc/InterNoise2004TheuseofVolumeVelocitySourceinTransferMeasurements.pdf)，式 (4)，给出频域 `p/Q = jρck exp(-jkr)/(4πr)`；此处以 `Q=ΣA_j q̇_j` 写成时域形式。

每个采样区间的半正弦力采用解析积分，离散冲量严格等于输入 `J_i`。结构状态空间用 [SciPy cont2discrete 的 ZOH 方法](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.cont2discrete.html) 精确离散，再以稳定二阶滤波器递推；不会用显式 Euler 积分高频振子。声程使用分数采样延迟插值。采样率收敛仍需检验，因为接触连续形状被分段保持、声程有插值误差。

## 文件与 API

数值参数统一位于 `sound_profiles.json`。`load_profile()` 返回独立副本。输入例子：

```json
{
  "schema": "stair_quiet_v1.impact_events.v1",
  "duration_s": 3.0,
  "measurement_window_s": [0.1, 2.5],
  "provenance": {
    "kind": "unvalidated_simulation_event_extraction",
    "checkpoint_sha256": "填写实际模型哈希",
    "extraction_rule": "记录实际加载周期、阈值、支持力扣除和接触身份规则"
  },
  "events": [
    {
      "time_s": 0.1,
      "normal_impulse_N_s": 0.8,
      "position_m": [0.0, 0.0, 0.17],
      "wheel": "fl",
      "contact_id": "fl:tread_1:loading_cycle_1",
      "surface": "tread_1"
    }
  ]
}
```

事件必须来自明确的动态加载周期。**不能把站立支撑力乘站立时间作为冲击冲量**。事件提取本身是否正确，不由声学模型认证。`contact_id` 应标识一次独立加载周期，重复标识拒绝；同一轮同时接触踏面与立面是合法的不同事件，允许叠加。没有事件时输出零声压，不产生“静态力噪声”。已有静态位移平衡中的恒定力也不产生加速度；突然从零施加恒力会产生暂态，两者不能混淆。

```python
from task_support.stair_quiet_v1.sound_model import (
    load_profile, render_sound, write_sound, paired_profile_sweep,
)
from task_support.stair_quiet_v1.audio import measure_pressure

pressure_pa, metadata = render_sound(events_document, load_profile())
metrics = measure_pressure(
    pressure_pa, metadata["sample_rate_hz"],
    start_s=metadata["measurement_start_s"],
    end_s=metadata["measurement_end_s"],
)
# metrics 是假设模型产生的声压的计量结果，不是实际录音。
write_sound("logs/evaluation/stair_quiet_v1/sound/prediction", pressure_pa, metadata)
sensitivity = paired_profile_sweep(baseline_events, candidate_events, load_profile())
```

`duration_s` 是完整 rollout 的源观察时长；可选 `measurement_window_s:[start,end]` 是相对此源时间零点的上楼子区间（缺省整个观察窗）。不能通过加入静止时间稀释 LAeq，也不能为了裁剪声级窗口删掉前面的事件：此前撞击的结构振动、传播和声级滤波器历史仍会影响窗口。程序始终渲染完整事件记录，输出包含前置零数据和额外的模态衰减尾声。`measurement_start_s=pre_roll_s+start`；`measurement_end_s=pre_roll_s+end+propagation_margin_s`。固定声程余量等于 profile 的 `max_source_receiver_distance_m/sound_speed_m_s`，所有策略共享，不能由当前事件的最远位置决定。这样新增零冲量或窗外远事件不会增加 LAeq 的平均时长；任何事件超出该距离域，即使冲量为零，也拒绝。默认距离域只是公开的建模边界，不是已验证的场地测量值。

元数据中的 `max_propagation_delay_s` 与 `propagation_margin_s` 都指此固定余量；`observed_max_propagation_delay_s` 单独记录实际事件中的最远传播时间，只用于诊断，不参与计量窗长度。额外保存的模态尾声不计入平均时长。此保守的统一延长是明确的测量约定，不会逐事件移动真实时刻；基线/候选须采用相同 profile。采样时刻以输出数组开头为零，源事件时刻另加 `pre_roll_s`。

从仓库根目录运行：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
PYTHONPATH=source/rl_training \
python -m task_support.stair_quiet_v1.sound_model impact_events.json \
  --profile illustrative_compact_panel \
  --output-prefix logs/evaluation/stair_quiet_v1/sound/prediction
```

输出 `.npz` 保存 float64 原始 Pa、采样率、时间轴与 JSON 元数据；`.wav` 是 IEEE float32，每个数值单位仍为 1 Pa，**完全没有幅值归一化**。WAV 必须与 JSON sidecar 一起保留。通用播放器不是校准声压输出设备，超出其期望 `[-1,1]` 的值可能削波，不能用播放响度评价结果。已有输出拒绝覆盖。

## 参数敏感性与不能省略的边界

`paired_profile_sweep()` 按 profile 中的接触时长倍数和阻尼倍数运行完整交叉网格。每个网格点对基线和候选使用同一份参数，并记录相同的 profile SHA256；不存在“给每个策略选最有利 profile”的入口。应报告所有网格点，不只挑最好一个。还应检查模态质量、频率、参与因子、有效面积、接收点和事件提取阈值；当前网格不代表覆盖了全部物理不确定性。

- 接触脉冲宽度是成对共享的假设，不能表达某策略实际通过延长接触时间来降噪；同冲量、同位置、同时刻的事件在这里可能得到完全相同预测。不能据此排除真实软着陆收益。
- 200 Hz 事件时刻无法可靠解析高频模态相位，例如默认最高模态；将事件时间补成高采样率不会解决此问题。当前九组参数只覆盖脉冲时长和阻尼，不覆盖事件亚步时刻或模态频率的不确定性，不能视为完整的稳健性证明。
- 模型只有有限模态与公开的名义低通频带，不是完整可听频段的砸地声模型。低通是因果 Butterworth，频带以上有滚降尾部，不能当成理想砖墙截止。
- 单极子假定具有净体积辐射。自由薄板两侧可能表现为偶极子并发生抵消；用几何面积代替有效辐射面积没有保证。本程序没有真实模式形状、方向性、机械声辐射效率或接触点参与因子的标定。
- 按名义频带检查 `ka` 和接收距离，只能排除明显违背紧致源近似的参数，不能证明真实楼梯是单极子。每个撞击的位置在其衰减期间保持固定。
- 不包含地面/楼梯遮挡反射、房间混响、结构之间传声、非线性轮胎接触、摩擦尖叫、多个接触的耦合、电机/齿轮声。独立声压相加保留相位，但这些相位也依赖未经验证的事件时间与相同局部等效模态假设。
- 自由场中压力按 `1/r` 变化，冲量按线性模型比例改变压力。该比例性质是模型数学性质，不是实机实验结论。
- 未校准有效面积、模态质量与耦合会改变绝对 dB。即使两个策略的声压比例在某个模型下相近，“dB 数字减半”的结果仍会随绝对增益变化，不能从这里晋级为实际验收。

本模块不能直接当作完整的声学训练奖励，也不能作为候选真机晋级或实际降噪达标证据。实际校准至少需要同步接触/振动和校准麦克风的测量来辨识接触波形、模态、幅值和声传播，随后用**未参与拟合**的落地事件验证误差范围。届时应建立新的经验证模型及证据链；不能把本文件的 `calibration_status` 改成 `calibrated` 来绕过边界。

短 CPU 测试覆盖冲量积分、线性压力缩放、距离与传播延迟、阻尼衰减、静态平衡、采样收敛、事件重放、同轮多接触、参数敏感性成对共享、原始 Pa 文件回读和禁止 `actual_pass`。这些是计算器的物理/数值反例，不是机器狗的实际降噪证据。
