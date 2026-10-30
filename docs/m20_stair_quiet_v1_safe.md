# StairQuiet v1 Safe

任务为 `Rough-Deeprobotics-M20-StairQuiet-v1-Safe`，日志独立放在 `logs/rsl_rl/deeprobotics_m20_stair_quiet_v1_safe/`。它继承 width 的真实楼梯、接触材质、四轮安全边界、滑移判据与落脚控制，新增教师感知分工、预测落脚代价及独立材质练习。参数唯一来源为 `task_support/stair_quiet_v1_safe/settings.py` 和继承的 width 参数台账。

## 程序入口

| 文件 | 作用 |
|---|---|
| `mdp/stair_quiet_v1_safe_env.py` | 物理子步接入预测落脚和滑移统计；重置前记录 Safe 自己的课程证据 |
| `config/wheeled/deeprobotics_m20/stair_quiet_v1_safe_env_cfg.py` | 新增观测组、落脚代价及重置路由 |
| `config/wheeled/deeprobotics_m20/agents/stair_quiet_v1_safe_ppo_cfg.py` | 新任务和日志身份，使用 Safe runner |
| `task_support/stair_quiet_v1_safe/actor.py` | 保留冻结基网络和已有残差，增加零输出安全分支 |
| `task_support/stair_quiet_v1_safe/observations.py` | 原始地形、纵向上阶感知和四轮安全特征分开 |
| `task_support/stair_quiet_v1_safe/signals.py` | 预测落脚风险、初次承载风险、滑移距离和后退距离 |
| `task_support/stair_quiet_v1_safe/course.py` | 真实宽度课程与较宽楼梯材质练习分别推进，保存分宽度/材质/轮子的证据 |
| `task_support/stair_quiet_v1_safe/runner.py` | 从 width 迁移模型与宽度目标；Safe 内部完整保存恢复和契约检查 |

注册由 M20 配置目录下 `stair_quiet_v1_safe/__init__.py` 被已有 Isaac 包扫描器发现，不修改运行中 width 的注册文件或源码哈希。

## 观测与动作

原始 `policy` 保持教师原布局。新增 `locomotion_scan` 只替换真实楼梯左右范围之外的高度点，使用最近的有效纵向采样点；范围内的台阶高度保留。冻结基网络与继承的残差使用该纵向高度。整个扫描都在楼梯之外时不虚构有效点。

新安全分支接收原始 policy、原始指令和 `safety`。四轮顺序固定 FL/FR/HL/HR，每轮包含踏面、左右余量及其短时预测、支撑、接触点切向速度、垂向力和水平轮速；另含机身横向位置、速度和航向。安全分支最后一层从零开始，与旧残差合并后统一限幅，不叠加额外动作上限。零指令时使用原始基网络输出。

critic 保持原有输入与完整权重，继续获得原始地形及有效摩擦信息。安全特征是教师仿真真值，没有伪装成实机可观测输入；MuJoCo 使用下述专用入口的 Safe 适配，后续学生蒸馏仍需对应的观测适配。

物理楼梯左右落差、四轮边界和机身保护保持生效。落脚预测是短时运动外推的奖励塑形，不是安全保证或落点投影控制器。新增代价覆盖接触前下降阶段和初次承载，原有稳定支撑检查仍保留。

## 两条课程

保持用户指定的十级真实宽度。部分环境固定在较宽的真实楼梯上练材质，其余环境继续按完成回合晋级/退级。材质晋级需要每个已开放几何/接触组合在足够宽的真实楼梯上达到台账中的回合数和成功率，不再等待目标窄楼梯成功。晋级只在重置时切换实际碰撞材质所在的地形单元。

接触配置是未经实测标定的等效摩擦、刚度和阻尼候选；静态空心板几何没有结构振动或声压仿真。不能将接触指标改善称作真实分贝下降。

`Width/<宽度>m/` 继续记录回合和终止计数；`Tail/safe/<宽度>m/<轮子>/` 记录滑移/不安全承载时间占比、每回合滑移距离和不安全初次触地次数，`Tail/material/` 记录材质滑移比例。完整分宽度、接触 case、轮子的累积量保存在 checkpoint 的 `safe_mechanics`；分母是实际承载时间或完成回合，不混用不同长度回合的原始时长。失败原因可能重叠。

## 从当前 width 继续训练

首次迁移使用完整 width checkpoint 的 actor、critic 和宽度目标。新增分支零输出，但纵向感知路由会按新任务生效；新分支随机隐层、重置的物理状态和新课程意味着行为不保证逐步相同。新优化器、critic 预热、新课程统计和 Safe 迭代数重新建立，来源迭代及旧统计作为 provenance 单独保留。材质练习环境的宽度目标使用自己的课程设置，不继承窄楼梯目标。不要用 `--resume` 直接读取 width。

长期训练由用户启动，先自行处理仍在运行的旧训练并核对资源余量：

```bash
conda activate m20_wzh
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Rough-Deeprobotics-M20-StairQuiet-v1-Safe \
  --init_actor_from logs/rsl_rl/deeprobotics_m20_stair_quiet_v1_width/2026-10-22_00-43-59_teacher_width_speed_recovery_fix/model_9500.pt \
  --num_envs 4096 --max_iterations 30000 \
  --run_name teacher_safe_from_width9500 --device cuda:0 --headless
```

Safe 自身中断后，保持环境数与训练契约一致，用同一任务和 `--resume --checkpoint /absolute/path/to/safe/model_N.pt`，省略 `--init_actor_from`。这时完整恢复 actor、critic、优化器、新任务更新数、宽度目标和连续成功数、材质阶段以及机械统计；重启物理回合不补计失败或成功。

CPU 回归入口为 `tests/test_stair_quiet_v1_safe.py`。有限步数、零优化器更新的仿真只验证实际接线和保存恢复，不能替代训练后的固定条件性能对照。

## MuJoCo 手动部署

Safe、Width 与原 StairQuiet 共用 `deploy_quiet_landing.py`，自动识别 checkpoint；普通 `deploy_mujoco.py` 仍拒绝带落脚控制契约的模型，防止遗漏物理子步控制。

```bash
python deploy/deploy_mujoco/deploy_quiet_landing.py \
  --checkpoint logs/rsl_rl/deeprobotics_m20_stair_quiet_v1_safe/2026-10-22_13-29-16_teacher_safe_from_width10000/model_3000.pt \
  --model /absolute/path/to/M20.xml \
  --terrain-xml deploy/deploy_mujoco/terrains/stairs_ascent.xml \
  --noise-profile teacher --viewer
```

窗口初始趴下，按 `Z` 站起、`C` 进入策略，`W/S/A/D/Q/E` 自己控制移动，`R` 阻尼、`X` 趴下、`Esc` 退出。默认不设步数或预设运动指令。

适配器完整加载基网络、原动作残差及安全残差，严格核对输入分组、共享源码哈希和权重。规则楼梯的纵向扫描复用训练公式；安全特征使用当前 XML 与命令行覆盖后的台阶高度、深度、数量、总宽和平台终点，四轮保持 FL/FR/HL/HR 顺序。支持参数化 `ascent`、`ascent_open` 和 `flat`，以及碰撞体全部由 box/plane 构成的原生 `custom` 地形；其它几何明确拒绝。平地宽度是行走走廊，不是无限平面的物理边缘。

视频组件地形可直接将上述命令的地形参数替换为：

```bash
--terrain-xml deploy/deploy_mujoco/terrains/sample_stairs.xml
```

这类原生组件由 `safe_native_geometry.py` 读取碰撞体当前世界姿态，计算四轮落脚边距与支撑状态；共面踏板在实际并集为凸面时合并接缝，真实缝隙、高差及组件移动保留。边界按包住轮子支撑区的矩形保守收缩，检查斜边越界；非凸并集保留分片边界，可能给出更保守的余量。滑移速度扣除支撑体的接触点速度，预测边距使用各组件当前刚体速度外推。该预测不包含未来碰撞或弹性力的变化。原生地形没有单一的直楼梯横向边界，因此 locomotion187 保留已交付扫描，不进行规则楼梯的横向射线替换；地形 `width` 只用于根部走廊位置归一化，轮子边距来自实际支撑面。模型尺寸、输入分组及权重契约不变；这些适配不代表已经测得该策略在活动组件上的通过率。

所有现有 `--noise-profile`、`--profile-file`、`--noise-reference` 继续有效。纵向扫描只加工已交付的地图，未知点保持零，不回填干净射线。**安全 48 维仍使用仿真真值**；包括 `front_only_clean` 在内的配置只限制原始观测和纵向地图，不能把整个 Safe 教师称为前方雷达可部署策略。

轮子速度取 link 原点，支撑/滑移力取上一轮已完成的物理求解；刷新位姿不重新求解接触力。特征中的轮子包络保持训练 capsule 定义，MuJoCo 原模型的 cylinder 碰撞体不变，这个差异保存在 `safe_runtime` 元数据中。此处支持教师 sim2sim 验证，不构成实机观测接口或通过率、真实声压验收。

可选 `--output /path/to/trace.npz` 导出输入与控制轨迹。手动模式额外生成 `trace_quiet.npz`；其中 `safe_policy`、`safe_motion_control`、`safe_locomotion_scan`、`safe_safety` 按 `safe_policy_time_s` 记录，`safe_contact_sample_time_s` 标明实际使用的已完成求解时刻，接管后首次输入为 NaN（无历史接触）。部署回归为 `tests/test_safe_mujoco_runtime.py`。
