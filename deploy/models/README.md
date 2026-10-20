# 保留模型

本目录是 v0 发布中唯一收录 `.pt` 权重的目录。`logs/` 与其它训练输出不上传；下列完整模型按原字节保存，未剥离恢复训练所需的状态。

按任务保存完整训练 checkpoint，共 14 份、135.47 MiB。选择覆盖关键基线、阶段末次保存点，以及有固定评测或已有导出来源的中间点。文件与原 checkpoint 的 SHA256 一致，独立存放，不链接到训练日志。

| 任务目录 | checkpoint | 保留理由 |
| --- | --- | --- |
| `pre_teacher/` | [19999](pre_teacher/model_19999.pt) | 前置训练基线 |
| `stair_teacher/` | [214700](stair_teacher/model_214700.pt) | 后续教师共同来源、固定评测基线 |
| `quiet_landing_v1/` | [257100](quiet_landing_v1/model_257100.pt) | 多个学生及驻足修复分支的父模型 |
| `idle_stop_v1/` | [257100_idle_stop](idle_stop_v1/model_257100_idle_stop.pt) | CPU 驻足修复产物、v3 教师 |
| `idle_stop_motion_v1/` | [46400](idle_stop_motion_v1/idle_stop_motion_v1_46400.pt) | 独立运动残差与楼梯航向修正教师，上游完整训练 checkpoint |
| `gait_refinement_v2/` | [2400](gait_refinement_v2/model_2400.pt)、[4999](gait_refinement_v2/model_4999.pt) | 固定三模型对照点；末次保存状态 |
| `student_v0/` | [21000](student_v0/model_21000.pt)、[29999](student_v0/model_29999.pt) | 固定 sim2sim 原 checkpoint；后续楼梯对照点 |
| `student_v1/` | [25000](student_v1/model_25000.pt) | 新版地图有效期的末次保存点、固定楼梯对照点 |
| `student_v3/` | [11500](student_v3/model_11500.pt)、[20000](student_v3/model_20000.pt) | 固定楼梯对照点；末次保存状态 |
| `student_v3_5/` | [20000](student_v3_5/model_20000.pt)、[21000](student_v3_5/model_21000.pt) | 已有导出及 SDK 初始化来源；终止训练前的末次保存状态 |

[idle_stop_motion_v1 完整残差教师](idle_stop_motion_v1/idle_stop_motion_v1_46400.pt) 保留。学生需要部署导出时，从对应完整 checkpoint 生成，并将导出文件、观测契约和 deployment.json 一起保存到本地评测输出目录。

完整文件路径、原训练来源、字节数、SHA256、选择理由及校验结果见 [catalog.json](catalog.json)。学生完整 checkpoint 内嵌观测契约、感知配置与来源记录，无需额外 sidecar；与原 logs 路径的关系仅用于追溯。

这些 checkpoint 可直接传给现有部署入口的 `--checkpoint`：教师用 `deploy/deploy_mujoco/deploy_mujoco.py`，v0 用 `deploy_student_v0.py`，v1/v3/v3.5 用 `deploy_student_v1.py`。

最初复制的 13 份均通过 CPU 严格加载及每份 32 个固定输入的前向检查，输出有限且与源文件加载结果逐位一致。运动残差教师另通过完整状态恢复及真实 Isaac 4 环境、8 步推理检查，未进行优化器更新。未做完整闭环性能评测或真机验收；历史评测有各自条件，末次保存编号及入选不表示性能最佳，v3.5 的 SDK 初始化也不代表行走或楼梯验收。

`idle_stop_motion_v1/idle_stop_motion_v1_46400.pt` 对应独立任务 `Rough-Deeprobotics-M20-IdleStopMotion-v1`，从旧 `model_46400.pt` 目录原字节重命名；网络、奖励和训练入口见仓库 README 的 IdleStopMotion 说明。
