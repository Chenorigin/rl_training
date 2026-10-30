# WAV 与声压数组的电脑计算入口

`audio.py` 实际计算声压指标，不生成“机器狗降噪达标”的结论。合成声波、声学假设模型和真实麦克风记录即使经过同一数值函数，证据等级也不同。当前测试使用已知数学信号与合成 WAV；不是机器狗的实测声音。

## 两层接口

```python
measure_pressure(
    pressure_pa, sample_rate_hz,
    start_s=0.0, end_s=None,
) -> dict
```

低层输入为一维声压 **Pa** 序列，不能传入牛顿、任意接触代理量或未经标定的 PCM 幅值。返回：

- `laeq_db`：测量窗内 A 计权均方声压对应的声级；
- `lafmax_db`：A 计权声压平方经过 Fast 指数平均后的窗口最大声级；
- `lzpeak_db`、`peak_pressure_pa`：窗口内原始、不加权声压采样点的绝对峰值；
- `rms_pressure_pa`、`a_weighted_rms_pressure_pa`、实际取整后的时间窗及方法说明。

峰值是采样声压的瞬时峰值，**不是 LAFmax**，也不声称恢复了采样点之间或传感器带宽之外的模拟峰值。低层不去除 DC，不判定输入是真实录音，不返回 `actual_pass`。全零信号的对数声级为 `null`，不能当作零分贝或完美降噪。

从文件起点以零滤波状态处理整段声压，然后只汇总 `start_s:end_s`。低层允许起点没有历史，但会报告启动瞬态警告。上层录音要求测量窗前保留至少一秒实录历史；不要先裁掉首阶声音来满足历史要求。模型调用方可提供前置声压历史，并单独标注 `uncalibrated_simulation`。

```python
analyze_recording(
    recording_path, calibration_before_path,
    reference_level_db=..., start_s=..., end_s=...,
    calibration_after_path=...,
    background_before_path=..., background_after_path=...,
    calibration_frequency_hz=1000.0, channel=0,
    manifest=...,
) -> dict
```

高层读取未压缩整数或浮点 PCM WAV。其 `evidence_kind` 是 `calibrated_wav_numerical_metrics`，`physical_measurement_status` 始终为 `not_independently_verified`。即使所有输入检查通过，也只返回 `computed`，不是声学目标的 `pass`。缺少身份、同增益、校准后测、背景、时间或量程质量证据时返回 `inconclusive`；已经算出的原始指标可能仍保留，但带 `raw_metrics_are_provisional=true`。

## 标定与公式

参考声压为 `p₀=20 μPa`。已知校准声级 `Lcal` 对应的 RMS 声压为
`pcal = p₀ × 10^(Lcal/20)`，不是把名义 94 dB 强制取成精确 1 Pa。
先量校准 WAV 的归一化 RMS，得到 `Pa/normalized_sample = pcal / rms(cal)`，
然后以同一因子换算记录和背景。不能对每段 WAV 分别做峰值归一化。

整数 WAV 的比例来自数字容器位宽。特别是 SciPy 将 packed 24-bit PCM 左对齐读入 int32，所以应除以 `2^31`，不是 `2^23`；8-bit unsigned PCM 先减中心值。正削波门槛使用 WAV 的有效位数，能够发现低于 int32 最大值的 24-bit 正数字轨。浮点 PCM 不采用整数的固定量化步长。高层去除整段 ADC DC 均值并报告去掉的量；明显 DC 偏置、数字轨命中或低于可用整数分辨率会阻止 `computed`。

同一前置校准因子用于整个试验。后置校准用于检查漂移，不通过两次各自归一化掩盖增益改变。程序还核对校准音主频、音调纯度、分块幅值稳定性、校准与记录采样率。阈值的代码台账在 `audio.py`。

A 计权采用模拟 A 曲线的四个 DC 零点及六个实极点，转换成数字 SOS。为了减小高频双线性频率扭曲，先以多相 FIR 插值到四倍采样率；数字滤波器在 1 kHz 归一化。LAeq 是 A 计权压力平方在窗口内的平均，然后取对数，不是逐帧 dB 的算术平均。

Fast 对 A 计权压力平方做指数平均，时间常数为 0.125 s；使用
`E[n] = α E[n−1] + (1−α) pA[n]^2`，其中 `α=exp(−1/(fs×τ))`。
LAFmax 再取测量窗中 `10 log10(E/p₀²)` 的最大值。

上述为工程数值实现，**没有 IEC 61672 认证**。当前测试涵盖正弦频响和解析 Fast 响应，不等于完整频响、脉冲、线性、环境、电磁兼容和整条传感器链的计量认证。双线性法在高频仍存在偏差；四倍插值并不能增加原始录音的带宽。单频声校准也不能证明麦克风在整个音频频段响应平直，必须保留真实麦克风频响/适用带宽和不确定度记录。

公式和实现依据均为原始资料：

- [NI：声压参考值、麦克风与标定](https://www.ni.com/en/shop/data-acquisition/sensor-fundamentals/measuring-sound-with-microphones.html)
- [MathWorks：A 计权模拟极点/零点及双线性设计](https://www.mathworks.com/help/audio/ref/weightingfilter-system-object.html)
- [SciPy：bilinear_zpk 不自动预畸变](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.bilinear_zpk.html)
- [SciPy：WAV 位宽、左对齐与浮点格式](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.read.html)
- [NTi Audio：Fast / Slow 时间计权](https://www.nti-audio.com/en/support/know-how/fast-slow-impulse-time-weighting-what-do-they-mean)

## CLI 与采集记录

使用已有 NumPy/SciPy 环境，不需要训练、GPU 或安装新依赖：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/ubuntu/miniconda3/envs/m20_wzh/bin/python \
  source/rl_training/task_support/stair_quiet_v1/audio.py \
  --recording /path/to/ascent.wav \
  --cal-before /path/to/calibration_before.wav \
  --cal-after /path/to/calibration_after.wav \
  --background-before /path/to/background_before.wav \
  --background-after /path/to/background_after.wav \
  --reference-db 94 --calibration-frequency-hz 1000 --channel 0 \
  --start-s 1 --end-s 21 \
  --manifest /path/to/acquisition.json \
  --output /path/to/new_raw_metrics.json
```

`94`、时间窗等是命令示例，必须替换成该次实际校准设定和完整上楼窗口。输出拒绝覆盖已有文件。退出码 `0` 只表示 `computed`，`2` 表示证据不完整或无法可靠计算。

采集清单由现场记录提供，不能从文件名猜测。需要这些字段：

```json
{
  "recorder_id": null,
  "microphone_id": null,
  "gain_setting": null,
  "calibrator_id": null,
  "calibrator_certificate_record": null,
  "calibrator_certificate_valid_until": null,
  "microphone_response_record": null,
  "measurement_window_record": null,
  "expanded_uncertainty_db": null,
  "same_gain_all_files": null,
  "agc_disabled": null,
  "processing_disabled": null,
  "background_source_inactive": null,
  "files": {
    "calibration_before": {"sha256": null, "started_at": null, "gain_setting": null},
    "background_before": {"sha256": null, "started_at": null, "gain_setting": null},
    "recording": {"sha256": null, "started_at": null, "gain_setting": null},
    "background_after": {"sha256": null, "started_at": null, "gain_setting": null},
    "calibration_after": {"sha256": null, "started_at": null, "gain_setting": null}
  }
}
```

`started_at` 和证书有效截止时间采用含时区的 ISO8601 格式。程序校验完整文件时段的顺序和不重叠：校准前测 → 背景前测 → 上楼记录 → 背景后测 → 校准后测。录音中的前置历史也算完整文件的一部分。每个背景文件至少包含一秒滤波历史和一个与上楼测量窗等长的背景窗；使用背景文件的一秒之后这一窗口，不自动减去背景能量。

背景 LAeq、LAFmax 都必须低于记录，且留出规定裕量和两倍不确定度；背景掩盖、静音或未确认源不活动不放行。声卡/麦克风的模拟削波、AGC 或降噪有时不能仅凭 PCM 数值识别，清单声明必须由原始采集配置及记录支持。

指标可作为现有 `acoustics.py` 验收流程的输入材料，但它们不会自动填充真实机器、楼梯、模型身份、合格仪器等级或校准证书；没有完整有效测量链时不能冒充 IEC 声级计读数。仍须检查两基线、成功完成楼梯、匹配速度/时长及全部重复。

## 资源与边界

当前仅接受 48 kHz、96 kHz；每个 WAV 不超过 64 MiB、全部通道合计不超过 3,000,000 个样本，最多八通道。低层 Pa 数组也有同一样本数上限。上限控制原始数据、四倍插值、A 滤波和 Fast 状态数组的内存；超限返回错误，不悄悄降采样或截短。多通道显式选择一个通道，不能把左右麦克风直接平均抵消声音。当前不支持压缩音频、RF64、多 data chunk、流式超长记录或自动频响补偿。

测试命令：

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/ubuntu/miniconda3/envs/m20_wzh/bin/python \
  -m unittest discover -s tests -p test_stair_quiet_v1_audio.py -v
```

测试有已知 Pa 正弦、半幅、增益共同缩放、解析 Fast 响应、不同 PCM 位宽、空/NaN/截断/削波、漂移、背景和错误身份反例。全部合成输入只验计算量具，不是机器人降噪证据。
