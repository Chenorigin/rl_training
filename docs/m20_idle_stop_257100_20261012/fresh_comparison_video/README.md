# 原257100与驻足/停车修复模型：新录MuJoCo对比

[左右并排完整视频](m20_257100_vs_idle_stop_mujoco.mp4)，左侧原版，右侧修复版；原速、25fps、70秒。另有[原版独立视频](original_257100_mujoco.mp4)和[修复版独立视频](fixed_idle_stop_mujoco.mp4)。

| 完整视频时间 | 内容 | 单独对比片段 |
|---|---|---|
| 00:00–00:26 | 零命令驻足，物理时间10–36s | [驻足](stationary_compare.mp4) |
| 00:26–00:56 | 平地前进并松W停车，物理时间6–36s；完整视频00:34松W | [前进与停车](move_stop_compare.mp4) |
| 00:56–01:10 | W持续前进通过楼梯，物理时间8–22s | [楼梯](stairs_compare.mp4) |

每例完整运行36s，Z=0s，C=6s；前进命令为0.7m/s，W=8s，平地停车例于14s释放W。驻足与停车以各自测试窗口起点为位置、偏航参考；进入停车窗口后摄像机固定，地面网格提供位置参照。视频标注速度是当前底盘平移速度，验收的末段速度是最后10s实际路径长/时间，两者含义不同。

## 录制方式与验证

本次重新运行 `deploy_mujoco.interactive_rollout`，在闭环物理运行中捕获画面。键盘输入改为相同定时操作、窗口改为无窗口录制、取消墙钟等待；实际原始观测、actor推理、PD、50Hz控制和200Hz物理均执行。每个模型每例调用真实 `mj_step1` 与 `mj_step2` 各7200次。

渲染使用单独 `MjData`，从当前物理状态复制位姿；新增网格只在渲染场景中，不参与接触、扫描或控制。摄像机、字幕、阴影与反射设置仅用于画面。六例录制的时间、qpos、qvel均与此前验收数据逐位一致，完整源文件和checkpoint身份以SHA256核验。新录制固定三例再次通过原11项判据，见[验证收据](recording_validation.json)与[本次判据结果](evaluations/fixed/verdict.json)。

本次使用CPU、单线程OSMesa、单个仿真进程顺序录制，无新增GPU训练。资源依据与预留见[运行前资源账](resource_plan.json)，实际RSS和时间记录在验证收据和六份录制日志中。

## 修复模型究竟如何得到

确实修改了隔离副本中的Isaac Lab训练代码，并实际运行过PPO；但视频中的最终修复模型来自后续CPU监督拟合。两条路径必须分开说明。

1. **Isaac Lab PPO实验，未采用。** 从原 `model_257100.pt` 恢复，新增驻足运动/姿态惩罚、启停命令采样、平地暴露与原actor保留约束。GPU上512个并行环境，种子42，运行200次PPO迭代，最后产物 `model_257299.pt`，进程耗时6分22秒，退出0。该末次候选在MuJoCo中驻足漂移35.7cm、停车偏移34.8cm，最后10s路径速度约13.7mm/s，仍未达标，因此没有交付。真实命令、任务接线、迭代输出与耗时见[Isaac Lab训练日志](isaac_ppo_not_selected.log)；实验配置在[隔离训练配置](../../../.scratch/idle_stop_20261012/code/source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/idle_stop_env_cfg.py)。

2. **CPU监督拟合，最终采用。** 从原257100 actor重新开始，使用12份实际MuJoCo观测文件，共17,988条记录；补充轮速度、上一动作收敛等监督输入后共78,465个样本。平地零命令样本的四轮输出目标设为零，腿部与移动样本以原actor为保留目标；只训练原MLP末两层，前两层冻结。种子42，Adam执行8000个优化步，拟合约19秒。最终仍是原244→512→256→128→16 ELU actor，没有新增部署分支或运行时动作钳制。代码、拟合日志、样本与逐位复现证据均保留：[`finetune_idle_stop_wheels.py`](../../../scripts/tools/finetune_idle_stop_wheels.py)、[实际拟合日志](selected_supervised_fit.log)、[拟合参数与统计](../fit.json)、[逐位复现](../reproduction.json)。监督扩增数据并非新增物理实验。

视频实际加载的[修复模型](../../../logs/rsl_rl/deeprobotics_m20_idle_stop_v1/2026-10-12_idle_stop_distilled_257100/model_257100_idle_stop.pt)SHA256为 `0e6d0c93560b5a3f7ce9219c7b6a0cea84bd4a2d3f5dba33c02d79f7d4eb5f34`；[原模型](../../../logs/rsl_rl/deeprobotics_m20_quiet_landing_v1/2026-10-10_00-48-58_quiet_landing_v1/model_257100.pt)SHA256为 `368fa6287d3fc5e999b4ad596ad9f84be23f25d72bbd861236ec535a66b51c1c`。

本轮视频展示的是固定三例的比较。额外横移停车仍存在约3°瞬态偏航，未通过额外1°判据；坡上停车和真机未验证，详细限制见[原验收报告](../README.md)。

## 重录

在项目根目录执行以下单例命令。输出目录内对应MP4须不存在，脚本拒绝覆盖；原始模型、物理与部署源码哈希须符合固定证据。

```bash
MUJOCO_GL=osmesa LP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
/home/ubuntu/miniconda3/envs/m20_wzh/bin/python \
  docs/m20_idle_stop_257100_20261012/fresh_comparison_video/record.py baseline 平地启停
```

模型标签可用 `baseline` / `fixed`；案例可用 `零命令` / `平地启停` / `仅w`。每例保留实际物理NPZ、动作/命令记录、checkpoint与源码身份收据及分帧视频。`assemble.py`核验六例并合成，拒绝覆盖已有视频。
