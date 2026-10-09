# gait_v3 能力退化：诊断与修复（2026-10-08）

## 本轮实测

检查运行 `2026-10-08_16-51-12_gait_v3`，以诊断开始时最新完整权重 `model_1700.pt` 对比 `2026-10-06_22-06-56_stair_resume/model_199998.pt`。固定模型为用户指定的 M20.xml，MuJoCo 带物理闭环，每项750步（15秒），命令与地形冻结在 evaluation_contract.json。不是摆姿态回放。

| 指标 | gait_v3 早100轮均值 | 最近100轮均值（截至1705） |
|---|---:|---:|
| 平移速度误差 | 0.323 | 1.062 |
| 平均回报 | 88.31 | 17.12 |
| 每回合前腿合法交替转换 | 0.226 | 0.212 |
| 前腿支撑折叠占比 | 0.124 | 0.192 |
| 前腿稳定踩阶数FL/FR | 3.80/3.53 | 2.58/2.73 |

原始曲线摘要见 tensorboard.json，图见 tensorboard_trends.png。难度同时升高，因此不能仅凭首尾训练回报断言因果；独立固定 MuJoCo 对照确认性能退化。timeout比例升高只能说明活得久，不能说明通过楼梯。重复踩阶减少也必须结合实际踩阶总量、任务通过率判断。

| MuJoCo固定测试 | 199998 | gait_v3 1700 | 保护方案短测54 |
|---|---|---|---|
| 上楼平台通过 | 是 | 否，停在首阶前 | 是 |
| 下楼最终在走廊内越过终线 | 是 | 否，最终偏出走廊 | 是 |
| 平地最终横向偏移 | -0.170 m | +6.471 m | -0.182 m |
| 两方向转向是否跌倒 | 否 | 否，但漂移/耗能增加 | 否 |

最终终线与走廊判定不是逐障碍穿越证明。各项完整失败/成功、轨迹、能耗及步态指标分别在 baseline/latest/guarded_smoke/summary.json；不挑选有利案例。短测54未证明严格交替率提高。baseline_ascent_10s.png和latest_ascent_10s.png是重新实际运行物理闭环10秒后渲染的画面，已人工查看；mujoco_trajectories.png展示实际轨迹。

## 原因与证据边界

1. **任务与风格混在一起。** 新主奖励只给相邻、异侧、安全的转换，第一脚不付主奖励，同阶追赶则前后轴各罚一次。独立复核：原基准五级楼梯轨迹按旧v3逻辑可能没有主转换奖励，8次中间阶同阶事件扣36分。早50轮顺序+同阶处罚约为主转换收益的11.2倍。拒绝接近楼梯能规避这些事件；这是奖励结构的危险激励，不能把样本的因果解释当作完整消融实验。
2. **续训缺少长期能力约束。** model0 actor接近199998，但critic与原模型显著不同，确认重建了critic。PPO只限制相邻更新，没有冻结参考策略约束。平地与转向也退化，不能只归咎某个楼梯角度惩罚。
3. **学习率失控风险。** 指定1e-4 adaptive后实际峰值约3.844e-4，后落到1e-5。std均值0.372→0.443，约增加19%，不是“噪声爆炸”。
4. **惩罚与难度一起变化。** gait_level升到1使通用姿态、平滑与耗能成本增大。保留既定命令和课程规则；本轮用critic预热和能力保护应对收益分布变化，不把课程level上升解释为任务完成率提升。

## 修改

权威奖励仍在 stair_teacher.py，观察244/285、动作16不变。

- 加入 `stair_up_task_crossing`：全身穿越物理台阶后付一次任务奖励。前后轴都有真实稳定上阶接触历史；当前四轮越过台阶且在走廊内、轮底不低于台面、至少两轮垂直支撑、本体高度与直立条件满足。已付台阶不重复发钱，绕行、前腿单独上阶、回撤、趴地不付。
- 任务奖励不要求风格性空中越棱证据；恢复一次撞阶后仍可完成任务。碰撞仍由独立负奖励约束。
- 严格交替事件与跳阶/同阶检测逻辑保留。任务成功和步态成功分别监控。
- 降低同阶事件、同阶驻留与前腿折叠成本；恢复原速度追踪收益，姿态单独处罚，避免双重压制。数值台账为生产 TUNING，不在这里维护第二份。
- 新 `rl_training/stair_guarded_ppo.py`：以安装的官方RSL-RL5.0.1 PPO.update为底本，只在原优化步骤内加入冻结参考actor的动作损失。按参考std归一化，在平地较强、非平地较弱；不追加另一次独立actor优化。
- 先冻结actor适应critic，再联合训练；探索std不超过参考的指定倍数；固定小学习率、取消熵推动的探索扩张。参考actor和预热计数保存在checkpoint中，完整resume能恢复。独立复审逼红了legacy resume保护缺失/恢复旧optimizer学习率的问题，已修复：旧checkpoint自动冻结所加载actor并重新预热，fixed schedule强制同步optimizer实际LR；CLI可明确另选能力参考。
- train.py增加 `--init_critic_from`（配合actor初始化，不继承optimizer）、`--preserve_actor_from`（指定冻结能力参考）。stair任务actor初始化会自动设置参考；原始普通PPO不接这项参数。
- 仅stair teacher本身使用保护算法；pre_teacher继承配置仍保持原PPO参数。student代码不改。

TensorBoard新增：`Episode_Reward/stair_up_task_crossing`、`Curriculum/step_completion/up_whole_body_crossings`、`up_whole_body_crossing_coverage`、`Loss/anchor`、`Loss/critic_warmup`。继续联合查看每轴physical_landings、transition_successes、strict_coverage和same_tread_fraction，以及加载checkpoint后的固定MuJoCo任务表现。

## 验证

- preservation_cpu.json：真的构造RSL-RL模型、rollout storage并调用PPO.update；预热actor逐参数完全不变、critic改变；注入动作偏移使anchor变正；冻结参考无梯度且未改变；保存恢复参考和计数；std上限有效。穿越奖励的重复、侧绕、回退、前腿独走、趴地/翻倒反例都拒绝。
- reward_cpu.json：原有左右起步、前后轴独立交替、FL1→FR3→FL3、后腿同阶追赶、遮挡/离开地图、接触力/姿态反例保留通过。修正原检查中“同阶已经拿到主转换奖金”的错误假设；现在同阶不拿风格主奖金。明确允许物理任务成功取得任务收益，不能再宣称“所有同阶轨迹总回报为负”。
- 64环境55轮实际Isaac优化成功，50轮critic预热+5轮actor更新，checkpoint在 `2026-10-08_18-04-44_preservation_smoke/model_54.pt`。model0 actor最大变化为0，model54最大参数变化约0.000786。五项固定MuJoCo测试无跌倒且任务能力保留。**这是短期接线验证，不能外推几千轮不退化或步态已改善。**
- 55轮测试使用取消风格依赖前的task crossing；随后最终版本重新运行16环境200步Isaac，final_reward_smoke.json记录最终源SHA、244/285/16接口、实体绑定与有限值。观测7个上升棱，短探针全身穿越事件为0，不把零值当成功奖励生效证明；正例支付条件由CPU反例检查验证。
- 静态检查：新算法、配置、CPU工具无新增pyflakes告警。train.py既有 `rl_training.tasks` 注册导入被pyflakes记unused，原有情况未改。
- 两次测试初始失败分别为configclass没有类级default属性、MuJoCo离屏缓冲640×480不允许960宽；均修正后实际重跑成功。首次CPU导入包触发Isaac依赖，改为从真实算法文件加载CPU模块。

## 重新训练

本轮没有终止用户的gait_v3训练、没有删除模型、没有自动开始长期训练。其运行进程不会热加载新代码；先在该终端Ctrl+C结束旧训练，再启动下面的新运行。**回到199998；不继承1700及其optimizer。** 短测54仅验证用，暂不作为正式起点。

```bash
conda activate m20_wzh
cd /home/ubuntu/cyq_shixi_projects/rl_training
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Rough-Deeprobotics-M20-StairTeacher-v0 \
  --headless --num_envs 4096 --max_iterations 1000 \
  --init_actor_from logs/rsl_rl/deeprobotics_m20_stair_teacher/2026-10-06_22-06-56_stair_resume/model_199998.pt \
  --init_critic_from logs/rsl_rl/deeprobotics_m20_stair_teacher/2026-10-06_22-06-56_stair_resume/model_199998.pt \
  --run_name gait_preserve
```

不再加旧的learning_rate=1e-4、schedule=adaptive覆盖项。1000是观察窗口，不是收敛承诺。每100轮checkpoint用同一五项评测检查：丢失上下楼梯、平地明显侧偏或转向退化则不继续扩大训练，不靠mean_reward/level单独选最好模型。恢复这个新运行才使用 `--resume --load_run <新目录> --checkpoint model_N.pt`，参考与warmup计数会恢复；不要同时传actor初始化。

## 范围与未处理事项

原模型、MuJoCo控制/地形、命令采样、课程规则、pre_teacher、student、外部雷达仓库保持原样。工作区预先存在大量暂存删除，未批量提交/回滚/拉取。origin领先的M20S提交不包含本轮权威奖励实现，未混入本轮。GPU实时空闲约41GB，55轮验证64环境预算4GB/240秒，最终探针16环境2GB/180秒；未启动长跑占卡。根盘98%但仍约45GB空闲，未清理用户资料或修改系统警告。共享KB本轮实际fetch仍因损坏git对象失败，方法记录保存在本仓库 `.scratch/gait_v3_regression/kb_note.md`，未同步远端；未修理其他仓库。
