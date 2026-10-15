# 独立 M20 Quiet Landing V1

2026-10-10：用户要求补齐缺失奖励，当前代码已扩展，最新清单/公式/验证见 [expanded 说明](../quiet_landing_expanded_20261010/README_CN.md)。本文其余内容保留为扩展前版本的交付记录。

已实现并通过短时集成验证。未启动正式长训，未证明策略已降低实际声音或提高楼梯/平台通行成功率。

## 独立任务入口

Gym task：`Rough-Deeprobotics-M20-QuietLanding-v0`。

- env：`source/rl_training/rl_training/quiet_landing_env.py`。
- env_config：`source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/quiet_landing_env_cfg.py`。
- ppo_config：同目录 `agents/quiet_landing_ppo_cfg.py`。
- 独立 runner/logger：`source/rl_training/rl_training/quiet_landing_runner.py`。
- 独立通行奖励快照：`.../velocity/mdp/quiet_landing_traversal.py`。新环境配置不继承/导入 pre_teacher 或 stair_teacher 配置，也不直接调用其可变奖励模块；保留共同 rough 场景/资产/通用奖励与已有保策略 PPO 算法。
- 训练和 MuJoCo 共用事件逻辑：`source/rl_training/m20_quiet/signals.py`，不依赖 Isaac。

原 pre_teacher/stair_teacher 环境、配置、奖励文件和原 MuJoCo 入口均未修改。本轮在 M20 registry 增加新 task，并在 setup.py 注册新的纯信号包；已有工作树修改保持原样。

Actor/critic/action 实测为 244/285/16；固定 PD、原动作及观测语义保持兼容。所有静音参数以 `QuietLandingSettings` 为唯一台账，run 另保存当时的配置与源码快照。初始奖励设置是待标定候选，不是已确认最优参数。

## 落地奖励与日志

每个物理子步在仿真推进前缓存轮心速度，在推进/scene 更新后读真实接触力，用滞回与最小离地时间识别新接触。初始 settle 有保护窗口，持续滚动/支撑不重复罚；已分离的反弹仍计数。主项用水平踏面接触前的向下速度平方；侧接触单列，不将轮心水平滚动速度当作砸地速度。

`quiet_touchdown` 替换旧 impact 项。RewardManager 乘 policy_dt，离散事件项在函数内抵消该时间因子，防止奖励随策略频率改变。原加速度/一阶/二阶平滑惩罚复用；本次未同时加大权重。

高程图仅使用原扫描点估计间隙与接近窗口覆盖率，无效数据保留无效。当前未启用新的高程接近窗口奖励；未增加 Actor 接触观测。主奖励仍靠仿真真值，故不要求实机具有足端接触传感器。

TensorBoard 只保留：

- 标准 `Loss/*`、`Policy/*`、`Perf/*`、`Train/*`。
- `QuietLanding/*`：四轮及总体的事件数、接触前速度、比法向动能 J/kg、有效地图比例、窗口覆盖率、每米指标。
- `QuietReward/*`：落地和原有平滑/加速度项的加权贡献。
- 两个速度跟踪误差、终止原因、总回报率。

旧 `Curriculum/gait_quality/*`、`ascent_direction`、`step_completion`、腿姿态/入口/角度卡片，以及 `base_z`、`knee_pos` 和旧逐项 `Episode_Reward/*` 不输出。日志器合并有样本的条件键，解决首条空日志或无落地样本导致整轮静音卡片被遗漏的问题。速度分位数为有界直方图近似；超出范围另有 tail count 和真实 max。训练卡片按结束回合分组统计，logger 对这些组求均值，不能当作全训练数据的精确 pooled 分位数。

## 使用

在仓库根目录、使用现有 `m20_wzh` Python。启动前按项目准则确认整卡/整机资源和磁盘余量，再选择环境数。下面是命令模板，不代表已启动正式训练：

```bash
/home/ubuntu/miniconda3/envs/m20_wzh/bin/python \
  scripts/reinforcement_learning/rsl_rl/train.py \
  --headless --task Rough-Deeprobotics-M20-QuietLanding-v0 \
  --num_envs 4096 --max_iterations 5000 --resume \
  --checkpoint logs/rsl_rl/deeprobotics_m20_stair_teacher/2026-10-09_01-55-35_gait_fix_descent_entry/model_214700.pt \
  --preserve_actor_from logs/rsl_rl/deeprobotics_m20_stair_teacher/2026-10-09_01-55-35_gait_fix_descent_entry/model_214700.pt
```

当前统一使用train.py，专用训练脚本已删除。上面命令完整恢复指定checkpoint，并将其Actor显式设为冻结参考，替换baseline中继承的旧参考。输出只进入 `logs/rsl_rl/deeprobotics_m20_quiet_landing_v1/`。

续训 quiet run：

```bash
/home/ubuntu/miniconda3/envs/m20_wzh/bin/python \
  scripts/reinforcement_learning/rsl_rl/train.py \
  --headless --task Rough-Deeprobotics-M20-QuietLanding-v0 \
  --resume --checkpoint /path/to/quiet_run/model_x.pt \
  --num_envs 4096 --max_iterations 2000
```

当前续训由checkpoint恢复已有anchor和warmup进度，不依赖历史launch.json；显式 `--preserve_actor_from` 会重新设参考/预热。旧专用脚本的集成测试参数已随脚本删除，统一train.py的短回合、地形规模和预热覆盖使用现有Hydra命令行参数，不能用于正式效果比较。

MuJoCo 独立静音回放（固定命令；原交互 teacher 部署入口不变）：

```bash
/home/ubuntu/miniconda3/envs/m20_wzh/bin/python \
  deploy/deploy_mujoco/deploy_quiet_landing.py \
  --model /path/to/M20.xml --checkpoint /path/to/model.pt \
  --terrain descent --command-x 0.5 --steps 1000 \
  --viewer --live-plot --output .scratch/quiet_descent.npz
```

画面显示四轮状态颜色、落地闪烁/速度箭头、最近触地速度和动能表。`--live-plot` 开独立四轮实时曲线窗，需要桌面。`--output` 保存物理子步 NPZ、事件 JSON 和曲线 PNG。动能按单位质量记录，并非声能/dB。上楼使用 `--terrain ascent`；平台使用独立 `--terrain-xml`，后续正式评测需提供并冻结独立上下平台场景。

## 已验证的证据

1. `scripts/tools/check_quiet_landing.py`：生产函数 CPU 正反例通过；覆盖出生/滚动/触地前速度/重复支撑/reset/反弹/侧撞/无地图/非有限值/时间尺度，以及日志器空首条/条件键/NaN反例。
2. `validation/2026-10-09_23-45-03_integration_probe/probe.json`：真实 Isaac 8 环境、160 策略步、Actor 244D/Critic 285D；425 个落地事件；无 PPO 更新。该短探针未结束 episode，所以没有 reset 日志，不能单独用它证明 TB 接线。
3. `validation/2026-10-09_23-47-37_logger_ppo_smoke/verification.json`：真实 8 环境、两轮 PPO、测试专用短回合。第一轮 Critic 预热、第二轮 Actor 更新；Actor 最大权重变化 0.0005173646，anchor 与用户 baseline 差值为零；实际 TensorBoard 写出 QuietLanding/QuietReward。
4. `validation/2026-10-09_23-49-39_resume_isolation_smoke/verification.json`：quiet checkpoint 恢复后又一轮真实更新；preservation_updates 从 2 延续到 3，anchor 仍与用户 baseline 完全一致；88 个实际 scalar 标签，无旧 Curriculum/Episode_Reward/base_z/knee_pos 卡片。
5. `mujoco_descent.npz`：指定 baseline 6 秒真实物理回放，1200 个子步、74 个水平踏面事件；有四轮原始接触力与触地前速度，正常支撑力向上。这里只验证量具，不是完整下楼成功率。
6. `quiet_view.png`、`mujoco_descent.png`：已生成并实际查看标记/数值表及四轮曲线。离屏画面与静态导出已验证；桌面 live-plot 窗口尚未做人工交互验收。
7. 独立同 family 只读复审发现并修复 anchor 重置、稀疏日志漏卡、MuJoCo 跌倒判据三项问题，复审生产 CPU 和日志反例通过；异构 family 复审未执行。

启动资源读取：整卡约 42GiB 显存余量、整机约 100GiB MemAvailable、磁盘约 37GiB 余量。小规模验证缩至既有 16 环境 smoke 依据的一半，未停止其它训练。正式规模需重新算资源账。

末次代码还保留了 reset 当帧事件快照，防止 inference probe 的 events_seen 在返回前因 tracker reset 漏计；此前已完成的 PPO 奖励和 episode 指标不受该诊断修正影响。尚未运行完整高度/踏深/速度/平台矩阵、奖励消融、长训或实机录音，不能据本报告宣称降噪成功。

## 坑的记录

现象：新静音日志虽然在 extras 中出现，训练仍可能不出卡片。真因：安装的 RSL Logger 只遍历 ep_extras 第一条字典的 keys；空第一条/条件键会丢指标。判据：首条空、后续有落地数据的实际 logger 反例。修法：不发送空日志，并由独立任务 logger 合并条件键。

现象：加载 quiet checkpoint 后再次进入预热，参考策略变成当前候选。真因：load 后无条件 set_reference。判据：恢复前后 preservation_updates 以及参考 Actor 与原 baseline 的逐权重比较。修法：首次跨任务迁移才换参考，quiet resume 保留 checkpoint 内状态。

现象：Isaac 运行报错却退出 0。真因：Kit fast shutdown 在 Python 异常展开前关闭进程。判据：完整 PASS 标记/报告产物与 Traceback 同时核对。修法：关闭前输出异常，失败分支明确非零退出；不得仅凭 shell rc 判集成通过。
