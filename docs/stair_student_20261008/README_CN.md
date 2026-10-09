# Stair Student 框架与待定训练流程

## 已接线的框架

- `mdp/stair_student.py`：618维观测契约、同帧三通道缓存、基础噪声/缺点、reset清缓存、244→618 actor第一层扩展和动作蒸馏损失。
- `config/wheeled/deeprobotics_m20/stair_student_env_cfg.py`：继承当前楼梯教师的物理任务、奖励、地形和课程，独立生成student观测，新增非物理缓存reset事件。
- `agents/stair_student_ppo_cfg.py`：后续PPO微调配置，actor/critic均使用student观测，MLP仍为512/256/128，关闭观测归一化，初始学习率1e-4及adaptive日程。
- 注册任务：`Rough-Deeprobotics-M20-StairStudent-v0`。

| 观测组 | 维数 | 用途 |
| --- | --- | --- |
| policy | 618 | student actor：57本体、187高度、187validity、187confidence |
| critic | 618 | student价值函数，当前与actor输入一致，没有teacher critic特权项 |
| teacher | 244 | 蒸馏时查询冻结teacher actor动作标签，不被PPO actor或critic选中 |

57维本体绑定和缩放由教师配置最终解析后复制，保留实际上一执行动作。首版本体噪声关闭；后续增加本体测量退化时也应保证actor/critic共享同一测量。本体角速度、重力、命令各3维，关节位置/速度/上一动作各16维，总计57。

187点排列为11行Y×17列X：每行X从-.8到+.8，Y逐行从-.5到+.5，间距.1m，只随偏航旋转。高度为`clip(base_z-terrain_z-.5,-1,1)`，单位米。未知高度用0占位，validity=0、confidence=0；不是把未知解释为平地。

`student_perception.height_noise_std=0`、`dropout_probability=0`为默认干净基线。配置非零噪声/随机缺点时三通道在同一快照中生成，confidence仅由配置的来源噪声尺度计算、遇到无效点为0，不以真值误差计算。该confidence只是基础合成评分，不是已校准的实机概率。

## 本轮没有实现的部分

这是可运行框架，不是完成的蒸馏系统。尚未实现：真实ROS数据桥、世界地图重投影、逐格年龄/来源、连续遮挡、整帧延迟、空间偏差、多阶段退化课程、DAgger采样器、蒸馏训练入口和模型导出部署。现有感知工程的315/99点XYZ不能直接当作187点三通道输入。

`initialize_student_actor()`已被冒烟脚本实际调用：复制teacher第一层244列，其余374列置零，后续参数严格加载；没有加载critic和优化器。它尚未接入训练入口。普通`train.py --init_actor_from`仍要求同形状严格加载，不能用该参数直接把244维teacher加载到618维student。

`stair_student_ppo_cfg.py`是PPO阶段配置，不自动进行蒸馏。当前新增teacher观测组只提供244维标签查询输入，没有自动加载teacher模型，也不会把teacher动作作为实际执行动作。

NaN/Inf占位保护针对student三通道；teacher组保留原始干净height_scan定义，异常oracle射线仍可能产生NaN。因此后续蒸馏入口必须在查询teacher标签前检查干净扫描及本体有效性，拒收异常标签，不能用零高度伪造teacher真值。

## 建议讨论并落实的训练流程

### A. 选定teacher与干净输入基线

等正在训练的teacher取得可接受的任务和步态表现后，固定一个checkpoint及任务版本。199998只用于本轮ABI验证，不是已经选定的最终蒸馏老师。扩展student第一层，从干净618维输入开始，检查两者确定性动作以及闭环通过率，而不是从随机student重新学任务。teacher在蒸馏阶段不更新。

### B. 模仿预热

初期用teacher执行轨迹，student模仿其确定性16维归一化动作，先用基础MSE损失，分别记录12个腿关节和4个轮子的动作误差；不复制teacher critic。teacher与student读取同一实际执行上一动作。纯蒸馏阶段不需要student critic，不需要PPO。

### C. student轨迹上的在线蒸馏

逐渐由student执行动作，并在其实际访问的仿真状态上查询teacher干净观测的动作标签；teacher执行占比逐渐下降。保留已采集状态的训练样本，避免只学teacher成功轨迹而不会恢复自身偏差。降低teacher接管占比必须以student闭环表现为条件，不能只按迭代固定切换。

感知退化按同一生成过程联合输出height/validity/confidence：点噪声、空间连续空洞、遮挡、地图延迟、位姿偏差、重投影历史值、台阶边缘误差及短时整帧中断。先依据实机回放确定量级；不独立随机生成三个互相矛盾的通道，也不把所有错误都标成低confidence而遗漏“看起来可靠但有偏差”的情况。

首版是无记忆MLP。大面积或持续全缺测时，当前618维不能凭空恢复不可见楼梯；若闭环对照显示记忆确有需要，再加入GRU或历史观测，不用低MSE声称未知区域已经恢复。

### D. PPO微调

蒸馏达到干净和受损观测的闭环门槛后，加载student actor，建立618维student critic和新优化器，以当前楼梯任务奖励进行PPO。teacher可保留为初期动作约束，随后根据闭环表现减弱约束。具体蒸馏/PPO配比尚未设置，先讨论再施工。

student奖励在仿真中仍可使用接触力等训练信号；这不代表把这些量送进actor或critic。降低冲击、能耗与改善步态需要真实闭环指标裁决，不能只看模仿误差。

### 监控与验收

建议新增：总动作误差及腿/轮分组误差、有效点比例、confidence分布、已知/填充/过期比例、地图和观测年龄、teacher执行占比、student独立执行通过率。继续保留当前任务成功、前后交替/跳阶/同阶、前a角/后b-g角和能耗指标。

固定干净基线、带退化输入、屏蔽质量通道三组对照，并按退化种类分别报告成功率和跌倒率。数据缺失或任务样本量下降不能被解释为cost改善。

## 实际验证

- `cpu_checks.json`：干净高度、NaN/Inf、全缺点、同帧缓存、返回值不污染缓存、reset及同step重复reset、非法配置/错误点数拒绝、teacher标签不接收梯度。
- `isaac_smoke.json`：实际4环境8步，policy/critic618、teacher244、动作16，使用199998 actor扩展后与teacher动作最大误差0；噪声/缺点模式观测与奖励有限，actor/critic同一张地图，无效高度和confidence均为0。
- `reviewer.md`：同family独立CPU复审通过，实际RSL MLP扩展等价及缓存reset红绿反例通过；异构family未取得。`verification.json`记录框架文件版本及冻结文件核对。
- 没有启动优化训练、ROS、真机或全楼梯性能评测。Isaac启动日志出现文件监听额度告警，运行仍正常完成；未修改系统额度。磁盘实际剩余约45GB，未录制点云或创建新权重。
