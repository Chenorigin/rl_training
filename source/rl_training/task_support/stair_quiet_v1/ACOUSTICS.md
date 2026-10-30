# stair_quiet_v1 声音验收

此工具验收声级计的真实读数，不能把接触力、冲量、仿真奖励或手机未标定录音变成分贝。当前没有实测记录，因此不能宣称机器狗已经降噪。

目标严格采用用户字面要求：相对两个指定 checkpoint，各自满足 **LAeq 和 LAFmax 的 dB 读数都至少下降 50%**。例如基线读数为 80 dB 时，读数目标为 40 dB；还必须留出仪器不确定度余量。声能减半只是约 −3.0103 dB，不符合这里的目标。dB 是对数刻度，读数百分比不等于响度或声能百分比。定义依据：[OSHA 声学技术手册](https://www.osha.gov/otm/section-3-health-hazards/chapter-5)、[声级计制造商对 LAeq / LAFmax 的说明](https://www.nti-audio.com/en/support/know-how/what-are-laeq-and-lafmax)。

`acoustic_protocol.json` 是数值与基线 SHA256 的唯一台账。协议采用 A 计权 LAeq 和 Fast 时间计权 LAFmax；LAeq 按各次有效测量时长做能量平均，LAFmax 取全部重复中的最大值。保守验收为 `候选读数 + U <= 0.5 * (基线读数 - U)`，`U` 必须来自仪器及测量布置的不确定度记录。这里的重复次数、背景裕量和时长容差是本项目的预设工程判据，不声称是某个声学标准的完整合规认证，也不声称提供总体噪声分布的统计置信界。

从仓库根目录运行（普通 CPU，标准库，无 Isaac/torch 依赖）：

```bash
mkdir -p logs/evaluation/stair_quiet_v1/acoustics
python3 source/rl_training/task_support/stair_quiet_v1/acoustics.py setup-template \
  --output logs/evaluation/stair_quiet_v1/acoustics/setup.json
```

现场操作者先填完 setup：同一机器、同一负载与轮胎状态、同一楼梯及材质、同一麦克风位置/距离/高度、同一前进与转向指令、同一完整路线。记录声级计与麦克风序列号、有效校准证书、声校准器及证书、可用量程和不确定度。证书有效期采用含时区的 ISO8601 时间。位置记录应链接现场照片/示意图；楼梯记录应含阶高、阶深、总阶数和材质。背景低于仪器可校准量程也不能当作有效低噪声数据。

候选训练完成后，**采样前**冻结协议与候选文件：

```bash
python3 source/rl_training/task_support/stair_quiet_v1/acoustics.py freeze \
  logs/evaluation/stair_quiet_v1/acoustics/setup.json \
  --candidate-checkpoint /absolute/path/to/trained_candidate.pt \
  --output logs/evaluation/stair_quiet_v1/acoustics/protocol.json
python3 source/rl_training/task_support/stair_quiet_v1/acoustics.py measurement-template \
  --protocol logs/evaluation/stair_quiet_v1/acoustics/protocol.json \
  --output logs/evaluation/stair_quiet_v1/acoustics/measurements.json
```

冻结文件不覆盖已有文件。采集前留存文件及其 SHA256，例如进入本次试验记录；禁止看过结果后更改阈值、条件或删失败尝试。模板给出两个基线与候选轮换交错的采样顺序，按顺序逐次测量，不把基线和候选隔天分组采样。所有尝试均保留；一次跌倒、急停、未完成全阶或停住都会使这次完整验收为 `inconclusive`，应先解决性能问题再重新完整采集。

现场试验须由有急停能力的操作者看护。每次记录实际加载模型 SHA256、完整上楼时长/距离/阶数、最大静止区间及视频或观测记录。测量窗统一从第一次接触首阶，到全部轮足上顶层平台，不能加入前后静止时间稀释 LAeq。每条 `conditions` 必须记录该次实际条件，与冻结条件相符；不能无核验地复制模板。

采集前后做声校准器检查并留存导出，时间包住所有上楼和背景采样窗口。每个试次前后测源不活动时的背景 LAeq/LAFmax，持续时长不少于该次上楼时间；分别填写含时区的 `started_at` 和完整 `duration_s`，录入保留在声级计导出文件中的原始读数。时序必须是“背景前测结束 ≤ 上楼开始 < 上楼结束 ≤ 背景后测开始”，本次背景后测结束后才可开始下次背景前测；窗口允许首尾紧接，禁止重叠或复用背景片段。仅上楼开始时间递增不能证明测量独立。背景裕量不足、削波/超量程、外部噪声事件或校准漂移超限均 `inconclusive`，不靠背景相减或丢掉吵闹试次制造通过。候选每次上楼时长还需接近两个基线各自的中位时长。

```bash
python3 source/rl_training/task_support/stair_quiet_v1/acoustics.py evaluate \
  logs/evaluation/stair_quiet_v1/acoustics/measurements.json \
  --protocol logs/evaluation/stair_quiet_v1/acoustics/protocol.json \
  --output logs/evaluation/stair_quiet_v1/acoustics/report.json
```

退出码：`0=pass`，`1=fail`（证据完整但未达到两项降幅），`2=inconclusive`（缺测、不可信或不可比）。报告分别展示两基线的比较、dB 差值、读数降幅与声能比，避免混用。工具只能检查输入的一致性，不能认证操作者填写的数据真实性；验收必须同时保留仪器原始导出、校准记录、加载模型记录和完整观测视频。`tests/test_stair_quiet_v1_acoustics.py` 中全为合成仪器数据，只证明量具接受/拒绝路径正确，不是机器狗的降噪证据。
