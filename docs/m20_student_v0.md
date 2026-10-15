# M20 v0 学生环境与蒸馏入口

本版本已接好学生环境、配置、在线蒸馏、续训和导出。正式训练由用户执行下面的命令启动。推理验证不会更新权重，也不会生成训练 checkpoint。

## 已实现的接口

- 默认任务 `Rough-Deeprobotics-M20-Student-v0` 继承现有 Quiet Landing V1 的机器人、动作、地形、终止条件和物理步进。
- `Rough-Deeprobotics-M20-StudentV15-v0` 使用现有 V1.5 的物理任务配置。选择与教师训练任务匹配的版本；两者具有相同的观测和动作接口。
- 教师输入为原有本体状态与干净高程扫描；学生输入为本体状态、高程、有效性、置信度。维度、切片、网格顺序、高度参考和未知值定义统一记录在 `source/rl_training/m20_student_v0/contract.py`，每次运行都会写出 `observation_contract.json`。
- 学生是前馈 MLP，保持教师网络布局。第一层原有输入列复制教师权重，新增质量通道权重初始化为零；干净地图下可复现教师动作。
- 教师冻结、使用确定性动作作为标签。每个 episode 选择教师或学生控制，整组动作只由一个控制器执行；教师始终在当前真实执行状态上生成标签。标签与执行前的学生观测成对保存。
- 原生 RSL-RL 5 蒸馏 runner 使用动作 MSE 和有界 replay；一半采样来自当前 rollout，一半来自整个缓存。记录整体、腿部和轮部误差，以及教师实际执行比例。不使用 PPO 或价值网络。

## 感知退化

参数的唯一配置文件是 `source/rl_training/m20_student_v0/perception.py` 中的 `StudentV0PerceptionCfg`。训练参数在 `source/rl_training/m20_student_v0/configuration.py`；任务注册通过 `agents/student_v0_distillation_cfg.py` 引用同一份配置。

当前在仿真高程扫描上模拟高度噪声、每个 episode 的持续高度/平面/偏航偏置、连续区域遮挡、点缺测、地图更新周期、传输延迟和丢包。旧地图保留采集时的世界高度及坐标，按当前机器人位置和偏航重投影；地图外和过期区域变为未知。reset 会清空旧 episode 的地图。高程、有效性和置信度共享同一张快照。

`--perception-level 0` 用于干净基准，`1` 表示配置中的完整退化强度。置信度依据来源高度方差和采集年龄计算，未知区域为零。该方差包含所配置的随机高度噪声和持续高度偏置方差，不读取实际噪声误差或教师真值误差。该分数不是经过标定的正确概率；参数是工程起始值，需要用真实雷达数据校准。

这是合成的局部地图观测模型，尚未模拟实际雷达束、精确的机器人自遮挡或多帧地图融合，也没有接入 ROS elevation_mapping 消息。真机适配器需要按导出的契约采样同一张地图，并把 mapper 的高度方差、有效性和采集年龄转换为质量通道。传感器外参、机身高度参考、本体状态顺序与动作缩放必须和教师保持一致。

## 启动前推理检查

以下示例使用现有 Quiet Landing V1 checkpoint，脚本会在日志中记录其路径和 SHA256。替换 checkpoint 时，要确认其教师架构及任务语义一致。

```bash
cd /home/ubuntu/cyq_shixi_projects/rl_training
conda activate m20_wzh

python scripts/reinforcement_learning/rsl_rl/distill_student.py \
  --task Rough-Deeprobotics-M20-Student-v0 \
  --teacher-checkpoint logs/rsl_rl/deeprobotics_m20_quiet_landing_v1/2026-10-10_00-48-58_quiet_landing_v1/model_257100.pt \
  --num-envs 4 --perception-level 0 \
  --validate-only --validation-steps 8 \
  --headless --device cuda:0
```

此模式只做模型加载和有限步推理，检查观测、动作、质量通道和初始化一致性。另一次检查可将感知强度设为完整退化，并增加验证步数。`--validate-only` 始终执行学生动作；教师比例参数只用于正式训练的采样。

## 正式训练命令（由用户启动）

```bash
cd /home/ubuntu/cyq_shixi_projects/rl_training
conda activate m20_wzh

python scripts/reinforcement_learning/rsl_rl/distill_student.py \
  --task Rough-Deeprobotics-M20-Student-v0 \
  --teacher-checkpoint logs/rsl_rl/deeprobotics_m20_quiet_landing_v1/2026-10-10_00-48-58_quiet_landing_v1/model_257100.pt \
  --num-envs 64 --max-iterations 5000 \
  --teacher-ratio 0.5 --perception-level 0.25 \
  --run-name student_v0_stage_a \
  --headless --device cuda:0
```

这是温和退化下的首轮蒸馏起点，迭代次数不是性能承诺。脚本读取整卡、整机的当前内存余量和日志目标盘，按 rollout/replay 张量大小加仿真工程预留计算预算，超过占用上限则拒绝启动。日志盘还会核算本次完整训练的 checkpoint、TensorBoard 和代码快照预算，空间不足时在 CUDA/Isaac 启动前拒绝运行。工程预留没有替代实测：扩大并行环境数前应观察整个进程的显存和内存峰值。v0 仅支持单进程单卡。

训练输出默认写入 `logs/rsl_rl/deeprobotics_m20_student_v0/<时间>_<run-name>/`，包括 TensorBoard、checkpoint、环境/训练配置、观测契约、教师来源和代码快照。正常训练结束会导出 `exported/student_v0.pt` 和契约，并用最后一帧观测检查导出动作一致性。`--no-export` 可关闭结束时导出。

## 续训与增加退化

```bash
python scripts/reinforcement_learning/rsl_rl/distill_student.py \
  --resume logs/rsl_rl/deeprobotics_m20_student_v0/<运行目录>/model_<迭代>.pt \
  --max-iterations 3000 \
  --teacher-ratio 0.2 --perception-level 0.5 \
  --run-name student_v0_stage_b \
  --headless --device cuda:0
```

`--max-iterations` 表示本次追加的迭代次数。续训默认恢复任务、并行环境数、网络/优化器配置、感知配置、教师、学生、优化器和 replay；显式的种子、教师比例、感知强度参数覆盖保存值。并行环境数和 replay 容量必须保持一致。仿真从新 episode 开始，不恢复上一进程的物理状态或随机数状态。

增加退化应依据学生自身闭环的表现推进：先检查动作误差和脚轮行为，再逐步降低教师控制比例、提高感知强度。最终候选必须在教师比例为零的闭环评估中通过既有上下楼梯和平地基准。仅蒸馏 loss 下降、或有教师控制的回报良好，均不能作为上机判据。

训练后的无梯度推理检查可使用 `--resume <checkpoint> --validate-only`；默认保留 checkpoint 的并行环境数。该有限步检查用于接口验证，不包含完整地形性能评估。

## 磁盘空间不足时的恢复

`OSError: [Errno 28] No space left on device` 表示输出文件系统无法继续写入。TensorBoard 报错的位置不说明模型或蒸馏 loss 有问题。释放空间后，可以从最后一个完整 checkpoint 继续；不必重新从教师初始化。

现已增加运行期间的磁盘保护：每次迭代写日志前、每次保存 checkpoint 前都检查目标盘余量。保留量通过 `--min-free-disk-gb` 设置，其默认值及预算估计统一定义在 `source/rl_training/m20_student_v0/output_storage.py`。低于保留量时停止训练，并尝试在剩余空间中原子写入 `model_disk_stop_<已完成更新数>.pt`；若成功，报错消息会给出恢复路径。空间已被其它进程耗尽时，恢复保存仍可能失败，应使用之前完整的 checkpoint。

Checkpoint 先写入同目录临时文件并 flush/fsync，成功后原子替换目标；中途失败不会覆盖旧 checkpoint。未增加自动删除旧运行、旧模型或其它项目文件的行为。TensorBoard 在正常结束和异常退出时均会关闭，确保后台写入线程结束。

已检查本次用户失败运行留下的 `model_400.pt`，其学生、教师和 replay 数据均可读取，记录已完成 401 次更新。若要完成原定总共 5000 次更新，可执行：

```bash
cd /home/ubuntu/cyq_shixi_projects/rl_training
/home/ubuntu/miniconda3/envs/m20_wzh/bin/python \
  scripts/reinforcement_learning/rsl_rl/distill_student.py \
  --resume logs/rsl_rl/deeprobotics_m20_student_v0/2026-10-12_22-02-56_675354_student_v0_stage_a/model_400.pt \
  --max-iterations 4599 --run-name student_v0_stage_a_resume \
  --headless --device cuda:0
```

续训保留该 checkpoint 的环境数量、教师比例和感知配置，并写入一个新运行目录。本次修复使用真实 TensorBoard 的微型 CPU 单元 fixture、低空间故障注入和原 checkpoint 的 Isaac 无梯度推理验证；没有代用户重新启动正式训练。修复记录见 `m20_student_v0_disk_fix_validation.json`。

## 验证记录与范围

本次实施执行了 CPU 行为测试及真实 Isaac 小规模推理检查，具体结果见同目录 `m20_student_v0_validation.json`。CPU 测试包含一个微型合成数据优化 fixture，用于验证更新、冻结教师和 checkpoint 恢复；未执行机器人策略的正式训练。

本次没有执行真实雷达接入、完整地形闭环评测、长时间训练或真机部署。训练后导出的模型仍需完成这些环节，才能判定可上机。
