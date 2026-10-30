# 楼梯宽度课程

任务入口为 `Rough-Deeprobotics-M20-StairQuiet-v1-Width`。任务独立拥有宽度课程、有效接触材质、四轮安全落脚、滑移恢复和冲击尾部目标；参数以 [`WidthSettings`](../source/rl_training/task_support/stair_quiet_v1_width/settings.py) 为准。环境、配置和 PPO 直接继承基础 v1，全部任务专用实现位于 `task_support/stair_quiet_v1_width/`。

楼梯总宽按用户指定从 4 m 到 1 m，等间隔分成 10 级，内部编号 0–9。宽度序列由 `4-i/3` 生成。每个环境从最宽一级开始，连续成功达到台账阈值后收窄一级，失败退回一级并清空连续成功计数；在两端截断。成功沿用原任务的四轮触顶、最终稳定捕获和失败优先判据，不能用单纯前进距离代替。课程由完成的回合推进，不按 PPO 迭代数推进。

所有宽度的楼梯预先生成真实碰撞网格和射线网格。每次重置才迁移到目标宽度对应的行；进行中的回合不改变楼梯。地形单元在横向为最宽楼梯留出实际侧向落差。每个宽度都覆盖台账中的几何/接触组合。参考组包括空心薄板楼梯，也覆盖其他阶高、踏深和封闭立面。前后腿安全余量保持独立，四轮安全区和机身越界检查均使用所在行的真实宽度。

`front_tread_margin` / `rear_tread_margin` 分别控制前、后腿接触斑到踏面前后沿的余量；`front_lateral_margin` / `rear_lateral_margin` 分别控制完整轮子外廓到楼梯左右边沿的余量。四腿顺序为 FL / FR / HL / HR，机身另用含航向角的包络检查。持续不安全承载、滑移、恢复中的再冲击、力峰和窗口超额冲量分别计入机械目标。四轮触顶并完成稳定捕获才算成功，发生真实失败时失败优先。

`PROFILES` 覆盖摩擦、等效接触刚度和阻尼组合。Isaac 用完整射线网格配合分别绑定实际材质的碰撞网格；轮子材质归一后做 PhysX 读回。材质编号不进入 actor，critic 保持原有有效摩擦观测维度。薄板几何与等效接触没有模拟木板振动、空腔共鸣或声压，参数尚未用实测木楼梯标定。

材质阶段只计入最终目标宽度的实测回合。较宽楼梯上的成功不解锁更难接触组合；材料阶段仍要求每个当前 case 达到原有回合数和成功率。`Width/<宽度>m/` 分别记录累计回合、成功、侧向越界、恢复失败和停滞超时。失败原因标签可能重叠，不能相加作为失败回合总数；准确失败回合数为 `episodes-successes`。TensorBoard 显示会聚合采样，完整原始计数保存在 checkpoint 的 `width_curriculum.statistics` 中。

上阶速度沿用来源教师的直线上阶练习范围，以 `WidthSettings.speed_range` 为准。滑移仍计入机械代价；恢复失败检查持续无净进度或持续少于两轮支撑，继续前进不会仅因恢复总时长而被重置。净进度按新的前向位置里程碑计算，原地晃动和速度噪声不能续期；短暂换脚不累计成持续支撑丢失。侧向越界、前后腿安全落脚区、姿态/机身高度保护与四轮触顶判据保持有效。

## 启动与恢复

在仓库根目录运行，启动前先检查同机资源。这里仅提供命令，长训练由用户启动：

```bash
conda activate m20_wzh
nvidia-smi
free -h
df -h .
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Rough-Deeprobotics-M20-StairQuiet-v1-Width \
  --num_envs 4096 --max_iterations 50000 \
  --run_name teacher_width_speed_recovery_fix --device cuda:0 --headless
```

本次速度和恢复语义已改变，必须新开 run，不要用旧 Width checkpoint 执行 `--resume`。默认完整加载固定 v1 的 `model_12300.pt` actor 和 critic，重新建立优化器、critic warmup 和课程计数；冻结基网络、探索方差及原始教师参照，训练有界残差和 critic。日志在 `logs/rsl_rl/deeprobotics_m20_stair_quiet_v1_width/`。

完整恢复使用本任务 checkpoint，加 `--resume --checkpoint /absolute/path/to/model_N.pt`，且保持并行数、课程、物理、目标、PPO 和源码契约一致。恢复 actor、critic、优化器、已完成更新数、材质阶段、每环境宽度目标和连续成功计数，以及逐宽度/逐 case 累计统计。物理位姿重启于保存的目标宽度，未结束回合不补计成功或失败。

任务实现从旧继承布局迁到独立布局时，旧 checkpoint 的源码与 callable 路径契约会变化。必须先核验可信旧源码、配置与实际奖励/终止项等价，再另存经审计的迁移副本；不能用权重初始化冒充完整恢复，也不能关闭严格检查。正在运行的旧布局进程仍会读取旧路径，切换代码前应先保存 checkpoint 并结束该进程。

固定宽度的独立对照训练可在新 run 追加 `env.stair_tail.width_curriculum_enabled=false env.stair_tail.initial_width_level=9`，直接训练最终宽度；对照应使用相同初始权重、seed、训练预算和其他配置。不能用改变契约的参数执行完整恢复或直接当作已有 checkpoint 的固定宽度评估。

## MuJoCo 查看

Width checkpoint 使用专用入口 `deploy_quiet_landing.py`。该入口自动读取 checkpoint 的落脚控制设置，核验共享控制器源码和设置，并在每个物理子步执行控制；策略历史保留控制前的 nominal action。普通教师入口会拒绝此类 checkpoint，因为它只执行 nominal PD。

```bash
python deploy/deploy_mujoco/deploy_quiet_landing.py \
  --checkpoint logs/rsl_rl/deeprobotics_m20_stair_quiet_v1_width/2026-10-22_00-43-59_teacher_width_speed_recovery_fix/model_8300.pt \
  --model /home/ubuntu/桌面/m20_1/M20_perception_Lidar_rl_2/M20_perception_Lidar_rl/deep_robotics_model/M20/mjcf/M20.xml \
  --terrain-xml deploy/deploy_mujoco/terrains/stairs_ascent.xml --viewer
```

`--viewer` 为手动操控：启动趴卧等待，`Z` 站起，站稳后 `C` 接管策略；`W/S` 前进/后退、`A/D` 左右平移、`Q/E` 左右转，松开移动键回到零指令。`X` 趴下、`R` 阻尼停控、`H` 切换高度点显示，`Esc` 退出。默认持续运行到退出，只有显式传 `--steps` 才限制时长。窗口中的移动指令来自键盘；`--command-x/y/yaw` 仅用于无窗口固定指令检查。

启动会打印 `[CONTROL] StairQuiet landing=...; checkpoint settings verified`，结束摘要的 `controller_execution` 记录实际执行次数。落脚控制仅在策略模式中执行，退出/重新接管时清空接触和地形缓存。`--stop-on-fall` 可在策略模式检测到跌倒后退出。手动模式 `--output` 保存模式、键盘原始指令、有效指令与动作，另存 `_quiet.npz` 信号文件；其中 `landing_*` 控制诊断仅覆盖策略物理子步，使用独立的 `landing_time_s` 时间轴。

MuJoCo 地形来自指定 XML，不自动重放 Isaac 的宽度/材质课程、随机化或回合终止规则。运行入口通过不代表整段楼梯性能或实机安全通过。

在上面的命令末尾加 `--noise-profile front_only` 可导入现有前雷达噪声配置；也支持 `teacher/student-v0/student-v1/stress` 和现有 V1 单因素配置，默认 `teacher`。`--profile-file` 读取 V1 格式的 JSON 覆盖；`--noise-reference` 可改用其他学生 checkpoint 或 deployment.json 中保存的感知参数，默认沿用固定学生基线。完整名称与格式见根目录 README 的 MuJoCo 部署说明。落脚控制使用同一份已交付带噪声地图及原始采集年龄，缺测处不补仿真真值；键盘操作保持不变。

只想限制到相同前方范围、保持教师观测干净时，改用 `--noise-profile front_only_clean`。该配置对当前教师高度图直接施加 `front_only` 的空间遮罩，保留范围内的本体和高度值不变，不经过随机噪声、丢包或延迟队列；落脚控制同步获得当前有效前方地图，范围外仍为未知。

## 验证范围

CPU 回归：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m pytest -q tests/test_stair_quiet_v1_width.py tests/test_stair_quiet_v1_width_mechanics.py
```

临时分析脚本、冒烟入口与其产物用完即清理；生产源码哈希只记录运行依赖。实际闭环验证保持零优化器更新，不能据短检查宣称窄楼梯、多材质、真实降噪或实机安全已通过。长期训练由用户自行启动。
