# StairStudent 框架独立复审

## 结论

在本轮“创建 student 框架、随后讨论训练流程”的范围内通过，未发现未修复的阻断性缺陷。当前是基础观测与 PPO 配置框架，不是已经完成的蒸馏训练、雷达建图或部署系统。

审计者 Codex / GPT，与实现者同 family；独立读取与 CPU 执行成立，异构性未取得。只创建本 scratch 下审计文件，未修改生产文件，未启动 Isaac、GPU、ROS、优化器或策略训练；teacher/pre/commands/deploy 和其它感知工程均未改动。

## 实际独立 CPU 执行

运行前可用主机内存约 99 GiB；此夹具只有 4×187 扫描和小批量 MLP 前向，远小于 1 GiB 张量预算，1 个进程、Torch/OMP/OpenBLAS 各 1 线程，不使用 CUDA。

命令：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/ubuntu/miniconda3/envs/m20_wzh/bin/python \
  .scratch/stair_student_20261008/reviewer_cpu.py
```

实际 exit 0，结果 `reviewer_cpu.json`，日志 `reviewer_cpu.log`。

| 项目 | 独立检查结果 |
| --- | --- |
| 干净高度数值 | 与本机 Isaac `height_scan` 相同公式及 clip 完全一致，4×187；validity/confidence 均 1 |
| NaN / Inf 输入 | student 三通道全部有限，无效高度/validity/confidence 均 0；无效 base pose 对该环境全图置无效 |
| 100% dropout | 三通道全部为 0，不把未知高度混同有效零高度 |
| 三通道缓存 | 同一步重复读取保持相同随机结果；返回 clone，外部修改高度张量不污染缓存；新步刷新 |
| 同步 reset | reset EventTerm 清缓存后，同 global step、episode_length=0 也刷新正确 |
| 244→618 扩展 | 使用本机已安装的真实 `rsl_rl.models.MLPModel`，512/256/128，Gaussian log-std，归一化关闭；确定性 16D 动作误差 0，新增 374 列全 0，teacher 源参数不变 |
| 蒸馏损失 | student 收到梯度，teacher 标签无梯度 |
| 注册 | 实际注册 AST 执行指向 `stair_student_env_cfg:DeeproboticsM20StairStudentEnvCfg` 与对应 PPO class |

本次独立 MLP 检查使用同架构随机源参数，没有读取训练 checkpoint；实现者另用 199998 checkpoint 验证的动作误差 0，是实现者的运行证据，不能冒充为本审计独立 checkpoint 测试。

## reset 反例与修复接线

仅用 step 和 episode_length 判断新鲜度不能覆盖启动维度探测后的初次 reset：本机 ObservationManager 初始化会先调用 term 求 shape，而 env reset 可以保持 step=0、episode_length=0。

独立反例：扫描平地、base_z=.57 时缓存高度 .07；同一步重置后 ground_z=.1，正确高度 -.03；未触发 reset callback 时仍返回 .07，误差 .10。这不是用“值没变”判 reset 正确的假绿。

最终 `reset_student_perception(env, env_ids)` 清空缓存，并由 student env 的 `self.events.student_perception_reset = EventTermCfg(..., mode="reset")` 接线。独立调用该 callback 后返回 -.03，与直接计算新扫描完全相等；已静态检查其 EventTerm 函数与 mode。结果保留了不调用事件时的红反例和调用事件后的绿结果。

## 观测与 critic 边界

student env 先解析 teacher 的本体顺序/实体绑定/缩放，再只复制其 policy，禁用本体噪声，将 height_scan 替换为 student 高度并追加 validity/confidence。critic 由 deepcopy(student policy) 构造，没有复制 teacher critic，所以不引入 acceleration、torque、轮足力、摩擦或干净 oracle 地图作为隐藏 critic 输入。

- 57 本体：角速度、投影重力、命令各 3；关节位置、速度、上一执行动作各 16。
- student actor/critic：57 + 187×3 = 618。
- teacher 标签查询：57 + 187 = 244。
- 三组在环境里存在，不意味着 PPO 模型都使用它们；student PPO 的 `obs_groups` 只选择 policy 与 critic。
- 仿真奖励依然可用接触力等训练信号，这不等于把特权量送入 student 网络。

本机 Isaac grid 默认 ordering="xy"，torch meshgrid 展平对应 11 行 Y × 17 列 X，与代码声明的点顺序一致；ray caster 的 20 m offset 作用于 ray starts，height_scan 采用 sensor.data.pos_w 的跟踪 frame 高度，未把 20 m 发射偏移当成身体高度。

实际 618/618/244 的 ObservationManager 环境维度不是本次独立 CPU 夹具自行证明的。已读取实现者 `docs/stair_student_20261008/isaac_smoke.json`：4 env×8 step，actor/critic618、teacher244、动作16，critic 9 个合法 term、观测与奖励有限、无训练。该证据单独归属于实现者；本审计未启动第二个 Isaac。

## 已明确的限制与后续准入

1. **Teacher 标签异常**：student 的 NaN / Inf 占位保护不适用于继承的 clean teacher 组。raw height 的 NaN 经过 clip 仍为 NaN，独立夹具已确认。实现者已在 `docs/stair_student_20261008/README_CN.md` 明确未来查询 teacher 前需检查 oracle 与本体有效性、拒收异常标签，不以 student 零占位伪造真值。这是后续蒸馏入口准入条件，当前没有入口，不将其谎称为已接线的标签过滤。
2. **入口未接入**：`initialize_student_actor` 和 `action_distillation_loss` 已有可执行 helper，但普通 `train.py --init_actor_from` 仍为同形状严格加载；不能直接用该参数把 244D checkpoint 加载进 618D student。PPO 配置不是 distillation runner，不自动载入/冻结 teacher，也不会执行 DAgger。
3. **Confidence 是合成评分**：当前每格仅由配置 std 与有效性生成；没有真实多源估计方差、年龄、遮挡模型、时间延迟、地图重投影或校准概率。默认 clean baseline 的 validity/confidence 常数通道不意味着已经实现实机鲁棒性。
4. **无记忆模型**：大面积持续缺测时，不可见台阶没有足够当前信息；网络和 loss 均未验证能够恢复这种状态。实际样本退化、闭环成功率和是否需要历史/GRU 应在流程讨论后单独验收。
5. **无训练效果声明**：动作扩展等价、有限输入、同帧共享和短步冒烟仅证明 ABI 与接线，不能证明 student 步态、全地形成功率或 sim2real。

## 审计版本

源码与已读取实现者 verification 文件一致：

| 文件 | SHA256 |
| --- | --- |
| mdp/stair_student.py | `d04249808dc91c0657f1540f956f523d6f901fc768cc3efaf5d21c48429a8bf6` |
| stair_student_env_cfg.py | `933cbd480204cab0094b6df8f47526a9d0d325078badfa165d49a2a39e573d83` |
| agents/stair_student_ppo_cfg.py | `e7e5f38b60c34f2a41f8f684cbe152492d7e5304859608b89bd7ad2e6eba1687` |
| deeprobotics_m20/__init__.py | `f85f2bc64fdda387e05bb6046a3c5b2c05a1cfbd9df2ed3985b07058a5e0d793` |

结论：可交付本轮框架并进入流程讨论。ROS/真实地图、训练入口、蒸馏性能、新策略闭环和部署本轮均跳过，符合授权边界。
