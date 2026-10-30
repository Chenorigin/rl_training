# StairQuiet v1 Transition

任务：`Rough-Deeprobotics-M20-StairQuiet-v1-Transition`。这是 v1 的后续分支，针对登顶停步时的观测切换和行进中接阶碰棱，保留 Width/Safe 的真实宽度课程、落脚控制、四轮越界及失稳终止条件。日志目录为 `logs/rsl_rl/deeprobotics_m20_stair_quiet_v1_transition/`。

## 修改内容

- 冻结基网络和动作残差在运动、零指令时连续使用纵向楼梯扫描，消除登顶归零时从纵向扫描切回原始侧边落差扫描的跳变。安全分支继续接收原始地形；零指令残差门仍关闭。
- 回合交替覆盖直接接阶、阶前短停后加速、较低速度接近后加速；直接接阶占一半。接近距离、横向位置与初始朝向随机化，速度由策略在真实物理中建立。部署没有强制停车辅助。
- 增加几何条件约束的逆向水平力做功代理、下降轮提前碰下一阶棱的软代价。前者使用上个物理子步的速度，排除正常承载制动及开式踏板下方无立面的情形；后者排除先落回当前踏面的正常落地。没有新增硬终止。
- 新增与 `stairs_ascent.xml` 对应的阶高/踏深闭式几何族，合计 35 个几何与接触组合；每个入口协议分别累计成功、越界及碰棱指标。只扩大几何覆盖，训练台阶数量沿用原六级课程。
- 计划内、有稳定支撑的短暂停顿不累计无进展超时；越界、失去支撑及已有失败照常生效。登顶停止锁存到回合结束。

参数唯一来源为 `source/rl_training/task_support/stair_quiet_v1_transition/settings.py` 及其继承的 Safe/Width 台账。碰棱指标是净接触力构造的训练代理，并非接触点冲量或实测结构参数。初始朝向扰动用于覆盖失稳场景；没有把暂未复现的完整侧翻归因写成事实。

## 从 Safe 续训

使用本轮已核对的 Safe `model_18400.pt`，完整加载 actor（含已训练的安全残差）、critic 和原保真参考网络，并继承宽度目标、已解锁材质阶段。不同环境数时，宽度学习环境单独从源宽度学习环境重采样，材质练习环境保持其专用宽度。建立新优化器、新任务更新数、新课程统计及连续成功计数；新扫描与奖励语义下的成绩单独计算。

```bash
conda activate m20_wzh
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Rough-Deeprobotics-M20-StairQuiet-v1-Transition \
  --init_actor_from logs/rsl_rl/deeprobotics_m20_stair_quiet_v1_safe/2026-10-22_13-29-16_teacher_safe_from_width10000/model_18400.pt \
  --num_envs 4096 --max_iterations 30000 \
  --run_name teacher_transition_from_safe18400 \
  --device cuda:0 --headless
```

首次从 Safe 迁移使用 `--init_actor_from`。初始化入口明确要求完整 Safe checkpoint，并拒绝 Width 或缺少安全分支的权重。Transition 自身中断后，保持环境数及契约一致，使用同一任务和 `--resume --checkpoint /absolute/path/to/transition/model_N.pt`，省略 `--init_actor_from`。完整恢复优化器、更新数、宽度/材质课程和分协议统计；Safe checkpoint 用于跨任务初始化，Transition checkpoint 用于自身完整恢复。

## MuJoCo 部署

`deploy_quiet_landing.py` 自动识别新版本并加载对应扫描路由。把以下 checkpoint 路径换为本次训练产物：

```bash
python deploy/deploy_mujoco/deploy_quiet_landing.py \
  --checkpoint /absolute/path/to/transition/model_N.pt \
  --model /absolute/path/to/M20.xml \
  --terrain-xml deploy/deploy_mujoco/terrains/stairs_ascent.xml \
  --noise-profile teacher --viewer
```

零指令平地继续使用已交付的原扫描。规则楼梯使用新纵向路由；`sample_stairs.xml` 等活动组件仍走现有原生碰撞几何适配。安全 48 维使用仿真真值，此版本仍为教师 sim2sim 策略。

## 验证与评价

回归入口：`tests/test_stair_quiet_v1_transition.py`。可复用、无优化器更新的有限 Isaac 验证入口：`scripts/tools/validate_stair_quiet_transition.py`。从 Safe 迁移的实际执行记录位于 `logs/evaluation/stair_quiet_v1_transition_from_safe_20261023/`，其中 `safe_initialized_only.pt` 仅用于加载链路验证，不是训练结果。

训练后的验收应固定相同几何、宽度、速度和初始朝向，对照直接接阶与阶前停顿，并分别报告首次碰棱、朝向偏转、四轮越界、登顶停止及成功率；不能用整体奖励或仅加载成功代替闭环性能。
