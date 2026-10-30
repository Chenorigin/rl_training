# M20 student_v3_safe

任务：`Rough-Deeprobotics-M20-Student-v3-Safe`。统一入口仍为
`scripts/reinforcement_learning/rsl_rl/distill_student.py`，默认配置为
`configs/distillation/student_v3_safe.json`。长期训练由用户启动。

## 蒸馏对象

教师必须是 `StairQuiet-v1-Safe` 的完整 checkpoint。适配器严格加载冻结基网络、
原动作残差和安全残差；教师输入为原始 `teacher244`、原始 `motion_control3`、
`locomotion_scan187` 和特权 `safety48`。安全分支与原残差合并后的完整名义动作
作为监督标签。输入分组、教师源码、物理契约和落脚控制契约不匹配时拒绝启动。

学生保持 v3.5 的前视感知、地图质量通道、四帧本体历史和
`789 → 512 → 256 → 128 → 16` MLP。额外特权观测仅进入教师，学生只读 `policy`。
教师与学生使用同一份仅依赖学生传感器的航向反馈；教师另读原始命令用于零命令门控。
默认初始化是已固定哈希的 `deploy/models/student_v3_5/model_21000.pt`；
只加载学生权重，优化器、回放和学生课程证据重新建立。

## 物理任务与执行

学生环境继承 Safe 的真实宽度/材质和预测落脚统计；扩展地形使用学生专属的
几何查询、重置、全程终点停车和终止规则，冻结的教师实现保持原样。
学生专属命令在原上楼命令基础上增加等待与起停练习，参数见默认 JSON 的
`training.idle_practice`；不会修改教师任务命令或放宽原卡住/摔倒阈值。
初始化按教师 checkpoint 的宽度目标
重采样环境，并继承已开放材质阶段；学生的回合与机械统计从零开始。
教师退出门控以真正完成整段课程为成功；未成功的超时和真实失败计为失败，
成功与真实失败同时发生时失败优先。

名义动作执行后仍经过共享物理子步落脚控制；学生控制层使用已交付的带噪地图，
未知地图不补教师真值。控制层的接触/位姿输入仍有仿真假设，尚未完成硬件适配。
导出的 `student_v3_safe.pt` 是带强制执行契约的名义 actor，不是独立机器人控制器。
原普通学生部署入口会拒绝缺失落脚控制的加载。

## 预检与训练

### 混合楼梯课程

原任务已加入两类地形，唯一参数入口是默认 JSON 的 `training.terrain_course`：

- `round_trip`：参考 `sample_stairs.xml` 的上楼—平台—下楼轮廓，使用静态固定结构。
  未模拟原 XML 中模块移动、平台弹性转动的动力学。只有经过顶部、完成下楼并在
  下方出口稳定落地才算成功；停在顶端平台不算完成。
- `open_ascent`：参考 `stairs_ascent_open.xml`，使用仅有水平三角面的楼梯，
  没有垂直踢面、侧壁或底板碰撞面。轮子碰到板边时仍可能产生接触力；不把
  “没有垂直面”解释成“没有任何水平接触力”。

`family_cycle` 控制每次重置的类别分配，默认保留原地形并混入以上两类；
这是重置样本比例，不保证不同回合时长下的物理步数比例相同。每类继续使用
原真实宽度与材质课程。原有地形调用原碰撞网格生成器，所有课程共用加长后的
地块；回合时限按最长楼梯、最慢指令速度、起停练习和终点保持时间校验。

教师安全观测和奖励查询都使用新课程的实际表面。高程187维、学生789维和
动作16维保持原契约；对角交替奖励限于上楼段，已有环境奖励的策略梯度覆盖
整个前进课程。下楼不会被强制套用上楼步态奖励。

新增日志 `StudentTerrain/<family>/episodes`、`successes`、`active_fraction`。
验证模式额外核对真实 PhysX 踏面/踢面碰撞、材质绑定与受载脚接触。
短闭环接入通过不等于教师或学生已掌握下楼、长开放楼梯，训练后仍须分地形
做学生独立闭环评测。

已运行的训练不会热加载本次修订。下一次手动启动继续使用本任务，从已有
checkpoint 的学生权重初始化；旧课程的全状态 `--resume` 不兼容。示例：

```bash
conda activate m20_wzh
python scripts/tools/distill_student_headless.py \
  --task Rough-Deeprobotics-M20-Student-v3-Safe \
  --teacher-checkpoint logs/rsl_rl/deeprobotics_m20_stair_quiet_v1_safe/2026-10-22_13-29-16_teacher_safe_from_width10000/model_18400.pt \
  --init-student logs/rsl_rl/deeprobotics_m20_student_v3_safe/2026-10-26_13-28-33_059999_student_v3_safe/model_500.pt \
  --num-envs 2048 --max-iterations 5000 --device cuda:0
```

先追加 `--preflight-only` 检查当时资源；本轮未自动启动长期训练。该命令用于
当前长训练结束后的下一次启动；同卡再开一份大环境训练的预检已出现余量不足拒绝。
回归入口为 `tests/test_student_terrain_course.py`，验证收据位于
`logs/evaluation/m20_terrain_extension_20261026/`。

从项目根目录执行。以下固定教师 `model_18400.pt`，仅表示来源选择，
不表示它已经通过完整性能验收；换教师时须显式更换路径并重新预检。

```bash
conda activate m20_wzh

python scripts/reinforcement_learning/rsl_rl/distill_student.py \
  --task Rough-Deeprobotics-M20-Student-v3-Safe \
  --teacher-checkpoint logs/rsl_rl/deeprobotics_m20_stair_quiet_v1_safe/2026-10-22_13-29-16_teacher_safe_from_width10000/model_18400.pt \
  --num-envs 4096 --device cuda:0 --preflight-only
```

预检不启动 Isaac、不更新优化器；检查实际权重、完整教师对拍、学生初始化和导出一致性，
并依据实际配置计算 replay、额外教师输入、地图队列、显存/内存工程预留和输出盘预算。
整机/整卡余量不足时拒绝运行。预算包含工程预留，扩大环境数后的峰值仍需实测。

由用户启动正式训练：

```bash
python scripts/reinforcement_learning/rsl_rl/distill_student.py \
  --task Rough-Deeprobotics-M20-Student-v3-Safe \
  --teacher-checkpoint logs/rsl_rl/deeprobotics_m20_stair_quiet_v1_safe/2026-10-22_13-29-16_teacher_safe_from_width10000/model_18400.pt \
  --init-student deploy/models/student_v3_5/model_21000.pt \
  --num-envs 4096 --max-iterations 30000 \
  --run-name student_v3_safe_from_safe18400 \
  --device cuda:0 --headless
```

输出独立保存在 `logs/rsl_rl/deeprobotics_m20_student_v3_safe/`，包含配置、来源、
完整教师权重、学生 checkpoint、TensorBoard、源码快照及最终 `exported/`。
正式指标包括原分组蒸馏 loss、教师执行比例、学生失败率、感知有效率，
以及继承 Safe 的逐宽度/材质/轮子机械统计。

有限推理检查可将环境数改小并追加 `--validate-only --validation-steps 16`；
该模式执行有限物理步，验证保存恢复和导出，全程不更新优化器。

## 恢复与更换来源

```bash
python scripts/reinforcement_learning/rsl_rl/distill_student.py \
  --resume logs/rsl_rl/deeprobotics_m20_student_v3_safe/<run>/model_N.pt \
  --max-iterations 10000 --device cuda:0 --headless
```

追加迭代数恢复模型、教师、优化器、蒸馏调度及已完成的宽度/材质课程证据。
物理状态从新回合开始，未完成回合不补计成功或失败。默认不保存 replay，恢复后重新采样；
不宣称逐轨迹复现。教师权重与 manifest 随学生 checkpoint 保存，恢复不依赖原教师文件路径。
环境数、观测/课程/配置和运行源码必须保持一致；漂移即拒绝恢复。

更换教师、环境数或感知配置时，新建 run，使用 `--teacher-checkpoint` 和 `--init-student`；
不能用旧学生的 `--resume` 来更换教师。

## MuJoCo 与回归

训练后的学生使用专用执行入口，继续支持原手动操作：

```bash
python deploy/deploy_mujoco/deploy_student_v3_safe.py \
  --checkpoint /absolute/path/to/student_v3_safe/model_N.pt \
  --model /absolute/path/to/M20.xml \
  --terrain-xml deploy/deploy_mujoco/terrains/stairs_ascent.xml \
  --noise-profile student-v1 --viewer
```

启动趴卧，`Z` 站起、`C` 接入学生，`W/S/A/D/Q/E` 移动，`R` 阻尼、`X` 趴下、`Esc` 退出。
该入口只建立学生观测，不建立 Safe 教师的 48 维安全真值。有限无窗口检查可省略 viewer
并显式添加 `--steps`，不代表完整楼梯或实机通过。

CPU 回归：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python -m pytest -q tests/test_student_v3_safe.py tests/test_distill_student_entry.py tests/test_student_v3_5.py
```

最终评估必须使教师执行比例为零，并按真实宽度/材质比较通行、越界、落脚、滑移恢复、
冲击尾部和停车表现。动作 MSE、有限步检查及训练混合轨迹均不能替代学生独立闭环评测。

## 接管抖动后的蒸馏修订

Safe 默认配置现已启用两项训练约束，模型结构和输入维度不变：

- `temporal_residual_mse`：使用同环境真实相邻状态，学习学生与教师的动作变化差异。
  排除重置边界、任一端留出样本、落地/恢复阶段和原始请求命令切换；航向反馈的
  小幅变化不视为用户切换命令。教师必要的动作变化不被直接压成零。
- `spatial_action_mse`：对同一状态叠加小幅测量扰动，约束输出变化；扰动幅度从
  实际感知配置换算，保留命令、历史动作、历史状态和未知地图通道的语义。
- 平地等待后起步、短暂停车后继续上楼，轮换覆盖环境；记录
  `StudentPractice/selected_fraction` 与 `StudentPractice/scheduled_pause_fraction`。

损失系数是待训练闭环验证的工程初值，数值统一见默认 JSON。其他学生任务默认
不启用两项稳定性损失。该改动不会让已经保存的旧模型自动变稳。

源码与课程已经改变，旧 checkpoint 的全状态 `--resume` 会按原规则拒绝；
使用 `--init-student` 从其权重开始新实验，重建优化器、回放与课程统计，不能删掉
来源校验来续训。当前已完成权重初始化、导出对拍及有限推理验证的启动示例：

```bash
conda activate m20_wzh
python scripts/tools/distill_student_headless.py \
  --task Rough-Deeprobotics-M20-Student-v3-Safe \
  --teacher-checkpoint logs/rsl_rl/deeprobotics_m20_stair_quiet_v1_safe/2026-10-22_13-29-16_teacher_safe_from_width10000/model_18400.pt \
  --init-student logs/rsl_rl/deeprobotics_m20_student_v3_safe/2026-10-25_16-43-26_585732_student_v3_safe_resume10500_tmux/model_20000.pt \
  --num-envs 64 --max-iterations 5000 \
  --run-name student_v3_safe_stability_from20000 --device cuda:0
```

先追加 `--preflight-only` 可在不启动仿真的情况下检查当前资源。环境数是已预检的
保守示例，扩大之前需重新预检。长期训练由用户启动。新 checkpoint 必须重新比较
零指令接管、长时间停站、起步/停车及爬楼，不能用训练损失下降宣称抖动已解决。

回归入口：`tests/test_student_stability.py`；本次验证收据与限制位于
`logs/evaluation/m20_student_stability_patch_20261026/`。

## 保留后退、侧移与转向能力

最新稳定性训练的在线课程仍只有前进/停站；延长这类数据的蒸馏不能保证保留其它
运动能力。原任务默认新增非前进动作的独立保留损失，不新增 Gym 任务或策略分支：

- 输入为真实 MuJoCo 闭环的789维状态，覆盖后退、左/右移、左/右转。监督由经过
  相同传感器与执行控制筛查的旧 `student_v3_5/model_21000.pt` 提供名义动作。
- 采用旧专家自身已通过有限闭环筛查的状态。另行尝试的500轮学生偏离状态中，
  旧专家也会输出异常动作；该批数据已拒绝用于训练，不能把“首次摔倒前”等同于
  监督可信，也不能把学生退化后的动作当作保留目标。
- 五类动作平衡抽样；完整seed42参与训练，完整seed43留出优化。未来闭环性能必须
  用未参与采集的seed。前进、停站不使用旧动作锚定，保留当前爬楼和降抖优化空间。
- 记录 `retention_mse`、`retention_heldout_mse` 及逐动作误差。在线方向覆盖和保留
  数据覆盖分开报告，不把在线仍为零的方向统计改成假覆盖。

运行输入位于 `deploy/models/student_v3_safe/nonforward_retention_20261026/`。
manifest和NPZ的SHA固定在原默认JSON；文件缺失、变化或训练/留出分组缺类即拒绝
启动/恢复。必须保留这些输入，跨机续训时一起携带；推理部署只需模型与执行契约。
保留数据内存已计入资源预检，数据本身不重复序列化进每个checkpoint。

当前运行进程不会热加载磁盘修订。以下命令由用户在下一次训练启动时使用；从500轮
权重初始化以保留已获得的前进/停站变化，再用旧全向监督纠正其它方向：

```bash
python scripts/tools/distill_student_headless.py \
  --task Rough-Deeprobotics-M20-Student-v3-Safe \
  --teacher-checkpoint logs/rsl_rl/deeprobotics_m20_stair_quiet_v1_safe/2026-10-22_13-29-16_teacher_safe_from_width10000/model_18400.pt \
  --init-student logs/rsl_rl/deeprobotics_m20_student_v3_safe/2026-10-26_13-28-33_059999_student_v3_safe/model_500.pt \
  --num-envs 2048 --max-iterations 5000 --device cuda:0
```

先追加 `--preflight-only` 核对即时资源。不要用旧 `--resume` 绕过来源变化，也不要
同时叠加两轮长期训练。实现和无更新推理验证不代表新模型已恢复运动能力；报告见
`logs/evaluation/m20_motion_retention_20261026/`，回归见 `tests/test_student_retention.py`。

## 楼梯对角交替奖励与强化梯度

原 `student-v3-safe` 已加入 `student_diagonal_alternation`，强化的是左前+右后与
右前+左后交替支撑，不鼓励脚跨过身体中线。187维高程、789维学生观测和16维输出
均不变，接触和课程真值只用于训练奖励，不增加部署输入。

奖励只在现有直线 +X 上楼课程、前进指令、身体姿态及落脚安全门限内生效。必须先
形成持续对角支撑，再切换到另一对；只有本步超过本回合已到达的最远位置才得分。
短暂单脚/三脚/四脚过渡允许但不给分，完全无支撑会清空交替历史。原地换脚、保持同一对支撑、腾空、同侧步态、
前后跳、越过身体中线，以及后退/侧移/转向/停站都不能获得这项正奖励。
持续时间目前按控制帧检查，不能据此证明帧间每个物理子步的接触都连续。
不安全帧不给分，但短暂失效期间保留上一个相位；这不等于整个换脚过程全程安全。

原纯蒸馏损失不使用环境奖励。因此同一任务增加小权重、固定高斯探索的策略梯度
辅助项，合入原来的同一次优化；保留教师监督、降抖和非前进动作保留损失。只使用
学生当轮实际执行的探索样本，按实际动作队列延迟对齐回报，排除重置前未执行的
动作及rollout尾部；教师回合和旧replay不进入该强化项。回报包含原有安全代价，
不因随后离开奖励区域而截掉摔倒等后果。部署仍使用确定性输出。

这是有限窗口、无critic/无bootstrap的裁剪策略梯度，不是完整PPO。短窗口存在
回报截断偏差；比率裁剪只约束辅助项，不能保证教师监督与保留损失引起的总更新
满足同一信赖域。记录联合更新后的KL，并在当轮采样KL超阈值后停止该轮强化项。
系数与探索幅度都是待训练验证的工程初值，唯一数值台账为
`configs/distillation/student_v3_safe.json` 的 `training.diagonal_reward` 和
`training.algorithm.gait_policy_gradient`。

观察 `StudentGait/reward_rate`、`StudentGait/alternation_fraction`、
`gait_pg_samples`、`gait_pg_stopped`、`gait_pg_joint_final_kl`，同时检查原安全、
抖动及逐方向保留指标。奖励权重设0可关闭该奖励，PG权重设0可关闭探索及强化项；
两者同时为0才是原蒸馏目标的对照。配置变化使用权重初始化，不能绕过旧全状态
恢复的来源校验。

当前训练不会热加载修改。下一次由用户启动时，可使用上一节的500轮初始化命令；
仍是原任务，新run自动保存修订来源。长期训练后须做独立学生闭环对照，检查台阶
成功率、对角切换、越界/交叉、冲击、滑移及后退/左右移/左右转，不能只看奖励升高。
本次实现、无更新验证与限制见 `logs/evaluation/m20_diagonal_reward_20261026/`；
正式回归见 `tests/test_student_gait_reward.py`。
