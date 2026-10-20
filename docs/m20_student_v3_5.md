# M20 student v3.5：蒸馏带运动残差的教师

任务：`Rough-Deeprobotics-M20-Student-v3.5`。仍使用统一入口
`scripts/reinforcement_learning/rsl_rl/distill_student.py`，没有另建训练工程。

学生保持 v3 的 `789 → 512 → 256 → 128 → 16` ELU MLP、4 帧本体历史、前视传感器、延迟、丢包和噪音配置。`student_v3_5.json` 的 perception 与 `student_v3_front_only.json` 相同。不会引入 GRU 或复制教师残差分支。算法、物理任务及损失配置沿用基础 v3；镜像损失默认仍为 v3 的 0。

教师是 `idle_stop_motion_v1`：冻结的 244D 基座加 247D 输入的限幅残差。额外 `motion_control` 组只提供给教师，保持修正前的原始命令；监督目标是完整教师的最终 16D 动作。零原始命令关闭残差。教师不参与优化，不建立价值网络，也不执行 PPO。

## 代码位置

- JSON：`configs/distillation/student_v3_5.json`
- 配置/教师/命令反馈/环境胶水：`source/rl_training/task_support/student_v3_5/v3_5_*.py`
- 环境：现有 `deeprobotics_m20/student_v3_5_env_cfg.py`
- agent：现有 `deeprobotics_m20/agents/student_v3_5_ppo_cfg.py`
- 注册：现有 `deeprobotics_m20/__init__.py`
- 学生部署：继续使用 `deploy/deploy_mujoco/deploy_student_v1.py`

保留 `student_v1` checkpoint/导出格式以复用原部署链路；provenance 的 task/task_version 标识 v3.5，并完整保存教师 manifest 和反馈约定，断点恢复不依赖原教师文件仍位于旧路径。

## 初始化和命令约定

默认仅把教师**冻结基座**扩展初始化到学生：新增地图质量/历史列置零。该初始化不是“学生一开始就等价于完整残差教师”；残差效果通过后续完整教师标签学习。若有可用的旧 v3 学生，可加 `--init-student /path/to/student/model_N.pt`，只加载学生权重，重新开始优化器和采样。不要把残差教师填进 `--init-student`。

v3.5 新增教师对应的航向 PD 命令反馈，但反馈仅使用学生实际得到的带噪 yaw/gyro、已交付的 height/validity/confidence，不读取清洁全量地形或奖励状态。缺失、遮挡或尚未到达的地图不能通过 command 槽泄漏楼梯信息。当前命令与历史命令均在同一传感器快照中更新。教师和学生都看到反馈后的有效命令，教师额外看到独立的原始命令。

PD 参数来自教师 manifest。楼梯增益选择改为可部署的传感器判据，因此不宣称复现同事教师训练时依赖内部奖励 tracker 的整个闭环。完整 Actor 在相同 244D/3D 输入下与原共享源码对拍。MuJoCo 学生适配器按 provenance 自动启用同一反馈；旧 v3 没有这项 metadata 时不启用。实际机器的外部部署程序同样需要接入该传感器反馈约定，不能只替换神经网络文件。

## 用户自行启动

在工程根目录和 m20_wzh 环境下，先只做 CPU/权重/资源检查（不启动仿真、不更新权重）：

```bash
python scripts/reinforcement_learning/rsl_rl/distill_student.py \
  --task Rough-Deeprobotics-M20-Student-v3.5 \
  --teacher-checkpoint deploy/models/idle_stop_motion_v1/idle_stop_motion_v1_46400.pt \
  --num-envs 64 --device cuda:0 --preflight-only
```

训练命令：

```bash
python scripts/reinforcement_learning/rsl_rl/distill_student.py \
  --task Rough-Deeprobotics-M20-Student-v3.5 \
  --teacher-checkpoint deploy/models/idle_stop_motion_v1/idle_stop_motion_v1_46400.pt \
  --num-envs 64 --device cuda:0 --headless
```

默认自动读取 `student_v3_5.json`，训练 30000 次更新，每 500 次保存。
默认输出：`logs/rsl_rl/deeprobotics_m20_student_v3_5/<时间>_student_v3_5/`，包含权重、TensorBoard、provenance、配置和源码快照。先按实时整机/整卡资源检查，再自行调整环境数。

恢复同一任务：

```bash
python scripts/reinforcement_learning/rsl_rl/distill_student.py \
  --resume logs/rsl_rl/deeprobotics_m20_student_v3_5/<run>/model_N.pt \
  --device cuda:0 --headless
```

恢复时不重新提供 teacher、config 或 init-student，环境数、数据与调度保持一致。只有新建训练才能用 `--init-student` 更换初始化和环境数。

## 检查范围

接口回归见 `tests/test_student_v3_5.py`：完整教师标签、基座初始化、源哈希拒绝、原始零指令门控、保存恢复、学生部署加载，以及带噪观测和 reset 下的训练/部署一致性。

接口测试和有限步推理检查不能代替正式训练后的闭环评测；步态、防滑和停车效果需单独评估。
