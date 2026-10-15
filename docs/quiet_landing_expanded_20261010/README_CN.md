# M20 Quiet V1 — 2026-10-10

用户要求把缺失的奖励同时补齐。当前全部固定权重静音奖励统一属于V1，包含新接触窗口峰值/超额冲量及加载率；V2留作奖励课程学习的补充扩展，尚未实现。全部新增项默认非零；固定PD、Actor244D/Critic285D/Action16D、原任务保持奖励和Teacher baseline不变。尚未启动正式长训，没有降噪收益结论。

## 入口与隔离

任务仍为 `Rough-Deeprobotics-M20-QuietLanding-v0`，使用独立env/env_config/ppo_config。默认run名为 `quiet_landing_v1`。当前统一使用 `train.py`；原专用训练脚本已删除，历史验证中的launch.json/source_snapshot只作留档，不是启动依赖。本目录名称保留以追溯已完成验证，不代表另一个任务版本。原论文式可变PD和新增Actor接触观测不在本次范围内。

`train.py` 按agent_cfg.class_name加载QuietLandingRunner，按 `--task` 加载独立quiet任务。`--resume --checkpoint` 现在支持完整文件路径，跨experiment目录恢复；原 `--load_run` 配合checkpoint文件名/正则的行为保留。`params/env.yaml`、`params/agent.yaml`、`params/cli.yaml` 和原git diff记录配置、启动参数及源码变更。旧步态卡片过滤仍在独立env/logger中实现。

生产实现：`source/rl_training/m20_quiet/extended.py`；接线：`source/rl_training/rl_training/quiet_landing_env.py` 和独立 `quiet_landing_env_cfg.py`。只有quiet任务调用这些新增项。原stair_teacher/pre_teacher文件未修改。本目录保存扩展前signals/env/env_cfg副本，供恢复原精简V1；旧交付说明中的奖励清单对应扩展前版本。

启动模板（尚未执行正式训练，环境数需重新核资源）：

```bash
/home/ubuntu/miniconda3/envs/m20_wzh/bin/python \
  scripts/reinforcement_learning/rsl_rl/train.py \
  --headless --task Rough-Deeprobotics-M20-QuietLanding-v0 \
  --num_envs 4096 --max_iterations 5000 --run_name quiet_landing_v1 \
  --resume \
  --checkpoint logs/rsl_rl/deeprobotics_m20_stair_teacher/2026-10-09_01-55-35_gait_fix_descent_entry/model_214700.pt \
  --preserve_actor_from logs/rsl_rl/deeprobotics_m20_stair_teacher/2026-10-09_01-55-35_gait_fix_descent_entry/model_214700.pt
```

上面命令完整恢复用户的 `model_214700.pt`，并显式设为冻结参考，覆盖baseline中更旧的参考Actor。PPO架构及保策略算法不变。两次小规模PPO验证中的1轮critic预热是测试覆盖设置；正式配置仍预热50轮。输出仍为 `logs/rsl_rl/deeprobotics_m20_quiet_landing_v1/`。

恢复quiet checkpoint（不再传 `--preserve_actor_from`，保留checkpoint已有参考和预热计数）：

```bash
/home/ubuntu/miniconda3/envs/m20_wzh/bin/python \
  scripts/reinforcement_learning/rsl_rl/train.py \
  --headless --task Rough-Deeprobotics-M20-QuietLanding-v0 \
  --num_envs 4096 --max_iterations 2000 --resume \
  --checkpoint /path/to/quiet_run/model_x.pt
```

如果恢复时显式传 `--preserve_actor_from`，会重新设参考并重置预热计数。不要把首次迁移命令原样用于quiet续训。只传 `--checkpoint` 而不传 `--resume` 不会恢复模型。

## 唯一参数台账与公式

`QuietLandingSettings` 管触地速度/接触判定；`ExtendedQuietSettings` 管本次新增项。参数权重集中在两个dataclass。设置新增项weight=0可禁用，保留全部键；每个run另保存完整配置。以下为候选值，不是经过声学/通过率标定的最优值。

记 `S(x)=min(x²,4)`；物理步长h=0.005s；控制步长dt=0.02s。新接触窗口40ms，由原滞回、离地时间、踏面方向和重置保护事件触发。Fn=max(F·n,0)，X=max(Fn-100N,0)。静态地形近似，n来自原187扫描点的局部平面拟合：仅选取距离0.18m以内、距最近扫描点高度不超过0.025m的同表面点，防止跨台阶拟合假坡；不足3点/退化时明确退回世界竖直法向并记录有效率。不是实际碰撞几何真值法向。

| 新增/改进项 | 权重 | 实际计算 |
|---|---:|---|
| quiet_touchdown | -0.02（原值） | s=max(-v_pre·n,0)，每事件(s/0.5)²；不再仅限竖直方向 |
| quiet_impact_peak | -0.005 | 窗口内最大X为P，逐物理步支付S(P_new/200)-S(P_old/200)，四轮求和 |
| quiet_impact_impulse | -0.005 | J=ΣXh，支付S(J_new/5)-S(J_old/5)，四轮求和 |
| quiet_loading_rate | -0.02 | 窗口内Σ_i S(max((Fn_t-Fn_prev)/h,0)/20000)h |
| quiet_wheel_jerk | -0.02 | mean_4 S((qddot_t-qddot_prev)/(h·10000))h |
| quiet_base_angular_acceleration | -0.02 | mean_xy S((omega_world_t-omega_world_prev)/(h·50))h，约束世界水平轴，不约束yaw |
| quiet_base_vertical_acceleration | -0.02 | S((vz_world_t-vz_world_prev)/(h·50))h |
| quiet_torque_rate | -0.01 | mean_16 S((tau_t-tau_prev)/(h·2000))h，实际施加力矩 |
| quiet_wheel_slip | -0.02 | Σ_i contact_i S(norm(v_contact_tangent)/0.5)h |
| quiet_swing_clearance | -0.1 | Σ_i swing_gate_i S(max(0.04-clearance_i,0)/0.04)h |
| quiet_action_leg_rate | -0.01 | mean_12 (a_leg_t-a_leg_prev)²，替换旧统一Action Rate |
| quiet_action_wheel_rate | -0.01 | mean_4 (a_wheel_t-a_wheel_prev)²，替换旧统一Action Rate |

前10项以物理步累计，其reward adapter除dt抵消RewardManager乘dt；峰值/冲量是离散增量，其余是时间积分。两组Action Rate仍按通常w·dt·C结算。腿轮两组均方独立于关节数量，不乘旧全局gait_level；这改变了原sum×g的强度，不能称为等价保留。动作使用12腿位置动作+4轮速度动作的明确slice，不使用articulation joint ids索引动作。旧action_smooth和腿轮关节加速度保留。

峰值/冲量增量在新窗口开始时归零，不重复收费；窗口跨policy step连续累计，回合结束前已累计部分不会丢。饱和上限保护奖励量级，但超出饱和区的不同冲击受到同样惩罚，后续需结合原始曲线校准参考尺度。100N允许载荷继承已有force阈值作为初始选择，尚未按不同支撑轮数量标定，不应视为M20的精确静载。

轮接触点近似为轮心-r·n，速度v_center+omega_world×(-r·n)，扣除法向分量后惩罚切向速度，允许纯滚动。该近似未考虑轮倾斜时真实圆柱接触点、多个接触点、移动地面；不是真实轮胎滑移率。正常纯滚动正反例已验证。

Swing Clearance条件：扫描有超过0.08m的高度变化、平移指令>0.1m/s、轮未接触、轮正沿n上升>0.05m/s、地图有效且已过重置保护。仅抬轮阶段罚净空不足，不在落地下降阶段强迫抬轮。机身净空/腿间净空等原约束仍保留。不是通用摆动轨迹或精确前方立面净空规划。

角/竖直加速度由连续world-frame速度差分，不混用旋转body-frame；重置和初始保护期间屏蔽差分尖峰。所有项检查有限值。这里的动力学量是声音代理，不是dB；原论文没有验证我们全部新增扩展。

## 可视化与验证

TensorBoard输出全部 `QuietReward/*` 的加权回合贡献，另有窗口最大法向力、最大超额冲量、最大加载率、接触门控的时间平均滑移和扫描法向有效率。旧步态Curriculum/Episode_Reward卡片不输出。法向有效率明确为最后子步指标；滑移时间均值包含离地的零贡献，不是只接触时段均值。回合组均值不是全训练数据的精确pooled统计。

MuJoCo保存新增原始量到NPZ，另生成 `*_extended.png` 六组曲线；原触地速度/比动能viewer及live plot继续使用。新六组曲线目前为离线导出，未增加新的桌面实时窗口。正常reset保护不会删除原始图上的出生冲击，只有奖励/episode指标屏蔽保护期。

实际验证：

- `scripts/tools/check_quiet_extended.py`：地形法向/台阶不拟合假坡、无效地图、持续支撑/触地、跨policy窗口、窗口结束、纯滚动/滑移、离地、条件净空、差分/reset、dt抵消、动作顺序、非有限值正反例通过。
- `scripts/tools/check_quiet_landing.py`：原触地与稀疏logger回归通过。
- `isaac_final_smoke.log`：真实8环境、两轮PPO，一轮critic预热+一轮Actor更新；244/285/16契约和所有新增非零权重term的运行期检查通过。小atlas/0.6s短回合只验集成，不能用于成功率或声音结论。TensorBoard标签与各项有限贡献另存run的verification.json；新增净空在实际样本中非零。
- `mujoco_descent.npz`：baseline策略4秒回放，800子步、22落地事件，全部新增原始量有限。完成不是训练后效果比较，未评估整段下楼成功率。曲线已实际查看。
- multi-round-review同family独立复审发现MuJoCo `mj_step2` 后objectVelocity仍读PRE缓存。修复为独立scratch MjData刷新POST运动学，不覆盖本步已求解的接触力/qacc/torque；自由落体生产反例验证刷新后的vz等于积分后的qvel，原qacc未改。异构family复审缺档。

资源依据：沿用既有8环境smoke规模；开跑前整卡空闲47243MiB，整机Available106521MiB，磁盘37GiB；只启动本任务8env短时PPO和串行CPU MuJoCo，未停止其它任务。正式规模需重新计算。

未验证：长训收益、奖励消融、全楼梯/平台/速度矩阵、训练后sim2sim、真实声音与真机安全。新增项更多不等于降噪更好，必须保留同场景同指令的baseline对照，比较通过率、速度误差、每米落地指标及原始冲击曲线。

## 统一训练入口验证

专用训练脚本已删除，无新增训练入口。统一 `train.py` 只扩展了现有checkpoint参数的完整路径解析，现有filename/regex查找保持；利用已有preserve参数设参考。另保存cli.yaml并明确报告运行完成/异常非零退出。历史专用入口的源码快照与报告仍保留为验收证据，当前命令均已替换。

- `unified_train_smoke.log` / `unified_train_verification.json`：统一入口真实8环境两轮PPO，首次baseline迁移；参考Actor与baseline误差0，Actor最大权重变化0.0004833341，guard_updates=2，12个quiet贡献写入TB，无旧步态卡片。
- `unified_resume_smoke.log` / `unified_resume_verification.json`：从上个quiet checkpoint再训一轮，不传preserve；参考Actor仍与baseline误差0，guard_updates从2延续至3。
- 运行期由env核验Actor244/Action16，CPU直接执行生产AST路径分支核验绝对/相对/./路径、旧filename/regex和缺失文件。独立同family只读复核通过，无GPU复跑委派，异构复审未执行。
- 这两次均为8env、小地形、0.6s短回合、测试用1轮预热；正常训练无需这些Hydra覆盖。实际开跑前显存空闲47250MiB、系统Available106365MiB、磁盘37GiB；未启动正式长训。

统一入口保留现有Hydra命令行覆盖。例如集成测试可以追加 `env.scene.terrain.terrain_generator.num_rows=2 env.scene.terrain.visual_material=null env.scene.sky_light=null env.commands.base_velocity.debug_vis=false env.episode_length_s=0.6 agent.algorithm.critic_warmup_iterations=1`，无需为这些覆盖新增专用flag。

## 坑：积分后运动学缓存

现象：MuJoCo积分后读取轮/机身速度，看似POST却与Isaac不同。真因：mj_step2更新qpos/qvel但不刷新cvel。判据：自由落体qvel.z=-0.04905而objectVelocity仍0。修法：scratch数据单独执行mj_fwdPosition/mj_fwdVelocity读取POST，原数据保留本步力/加速度。不能用直接mj_forward覆盖原数据把下一步求解样本混入本步。

## 四轮速度 / 动能 / 接触力可视化

直接使用原MuJoCo入口，没有新增应用入口：

```bash
/home/ubuntu/miniconda3/envs/m20_wzh/bin/python \
  deploy/deploy_mujoco/deploy_mujoco.py \
  --model "/home/ubuntu/桌面/m20_1/M20_perception_Lidar_rl_2/M20_perception_Lidar_rl/deep_robotics_model/M20/mjcf/M20.xml" \
  --checkpoint /path/to/quiet_run/model_x.pt \
  --terrain descent --viewer --quiet-landing --live-plot
```

保持原键盘交互：Z起立、C接策略、W/S/A/D平移、Q/E转向、H高程点显示、X趴下、R阻尼、Esc退出。固定指令回放可追加 `--autoplay --command-x 0.5 --steps 1000`。`--live-plot`自动启用quiet诊断；只要数值表/箭头可只传 `--quiet-landing`。去掉两个标志恢复原回放路径。

画面数值表逐轮显示：世界系轮心vx/vy/vz、速度模长m/s、轮刚体平移动能Etr[J]、地形接触合力模长|F|[N]、Fz[N]、启动以来合力峰值Fmax[N]、最近触地前法向接近速度TDvn及其比动能J/kg。蓝箭头表示当前速度，橙箭头表示地形净接触力，方向在世界系；长度分别0.15m/(m/s)、0.001m/N，均封顶0.45m；准确量值读取数值表。绿色标记支撑、蓝色离地、红色新触地闪烁。

Etr=0.5*m_wheel*|v_center_world|²，是轮刚体平移动能代理，不包含轮转动能、不等于整腿/整机接触有效动能或声能。速度模长包含正常滚动平移，静音比较应结合TD法向速度和接触力。接触力直接读mj_contactForce并转换world系，累加各轮与静态地形接触，排除自碰撞，显示本步求解力。RK4不能用split-step保持原积分，启用诊断时明确拒绝；本M20模型已验证原轨迹不变。

实时曲线为4轮×3列，分别速度m/s（含触地前法向接近速度）、平移动能J、接触力N；保留最近1000物理样本。加 `--output /path/to/run.npz` 时，原轨迹保存run.npz，新增200Hz数据保存run_quiet.npz及速度/动能/力PNG、扩展PNG和事件JSON。NPZ含world速度向量、速率、动能、world力向量、合力模长和wheel_masses_kg等口径；无output时历史有界，适合长时间查看。

实际验证：原入口150策略步/600物理步真实回放，新增三个物理量均有限且分别核对速度模长、0.5mv²和力向量模长；开启/关闭诊断时root_pos、root_quat、observation、action最大差均0。`wheel_metrics_hud.png`离屏截图已实际查看，`wheel_metrics_live_layout.png`使用真实样本执行LiveSignals.update并查看4×3布局。交互生产循环用fake viewer执行3帧，物理采样、scene/text接线通过。原CPU回归通过。真实桌面窗口/键盘联动尚未人工操作验收；上述渲染与Agg更新不替代该项。
