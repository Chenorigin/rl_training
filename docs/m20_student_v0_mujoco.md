# M20 v0 学生 MuJoCo sim2sim

入口是 `deploy/deploy_mujoco/deploy_student_v0.py`。使用 CPU 进行确定性学生推理，以 `mj_step` 执行物理闭环；输入 `[1,618]`，输出 `[1,16]`。不需要教师参与控制，也不会更新策略权重。

本轮固定使用蒸馏运行 `2026-10-12_23-24-52_380795_student_v0_stage_a/model_21000.pt`，已将其学生网络导出到 `deploy/models/student_v0_21000/student_v0.pt`。训练继续产生的新 checkpoint 不会自动替换这份评测模型。

## 自己操控户外组合场景

在本机终端运行：

```bash
cd /home/ubuntu/cyq_shixi_projects/rl_training
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/home/ubuntu/miniconda3/envs/m20_wzh/bin/python \
  deploy/deploy_mujoco/deploy_student_v0.py \
  --policy-jit deploy/models/student_v0_21000/student_v0.pt \
  --model '/home/ubuntu/桌面/m20_1/M20_perception_Lidar_rl_2/M20_perception_Lidar_rl/deep_robotics_model/M20/mjcf/M20.xml' \
  --terrain-xml deploy/deploy_mujoco/terrains/realistic_outdoor.xml \
  --noise-profile student-v0 --viewer
```

`--viewer` 默认键盘操控，机器人从趴下姿态开始，等待你的指令。先按 `Z` 起立，终端显示 `ready` 后按 `C` 接入学生策略。起立过程中按 `C` 会等待起立完成再接入。该入口沿用已有部署的键盘读取、姿态过渡和 PD 控制，但执行的是 618 维学生网络。

| 按键 | 操作 |
| --- | --- |
| `Z` | 从趴下或阻尼模式起立。 |
| `C` | 起立就绪后接入学生策略。 |
| `W / S` | 前进 / 后退。 |
| `A / D` | 左移 / 右移。 |
| `Q / E` | 左转 / 右转。 |
| `X` | 退出学生控制并趴下。 |
| `R` | 立即退出学生控制，清空延迟动作队列，进入阻尼模式。 |
| `H` | 显示 / 隐藏学生地图。 |
| `Esc` | 退出，也可关闭窗口。 |

Ubuntu X11 的 MuJoCo 窗口支持按住持续移动、松开或失焦后速度命令归零，可组合移动键。也可以聚焦终端输入；其他桌面会话若无法查询窗口按住状态，建议聚焦终端，通过键盘连发和超时归零操控。松开移动键只将速度指令归零，学生仍控制站立；若要退出学生控制，按 `R`。

地图圆点显示学生实际输入：红色为未知，黄色置信度低，绿色置信度高。模拟观测噪声和地图缓存仍然生效。按 `R/X` 或默认跌倒转阻尼时会清空未执行的学生动作；再次按 `Z`、等待 `ready`、按 `C` 可重新接入。

键盘模式默认持续运行到你退出，可用 `--steps` 限制整个会话长度。不会使用 `--command-x/y/yaw` 自动前进；`--track-course` 在键盘模式下会报错，避免覆盖你的指令。

如果需要恢复此前的固定命令回放，显式加 `--autoplay`。例如在上面命令后加 `--autoplay --track-course --command-x 0.5 --steps 5000`。`--track-course` 是与已有教师部署相同的外部路线指令反馈：根据当前航向和横向偏移生成侧向、转向命令，沿出生航向行走。此时 `--command-x` 表示路线前进速度，命令的 y/yaw 分量由反馈生成。反馈读取仿真真实位姿，未加入定位噪声，因此不能将它当成真实导航系统的验证。

单独看上楼梯可将地形参数替换为 `--terrain ascent`，下楼为 `--terrain descent`，碎石路为 `--terrain rough`；其余参数可保留。楼梯高度、踏深等以对应 XML 为准，也可用 `--stair-height`、`--tread-depth`、`--stair-count` 在单回合中覆盖。

## 观测与噪声

学生输入严格使用 channel-major 排列 `[proprio57, height187, validity187, confidence187]`。网格和高度变换直接使用现有部署契约，不将 h/m/c 按每个栅格交错排列。关节按名称映射，腿位置残差、轮速度尺度和 PD 控制复用现有部署代码。

地图端直接调用训练中的 `m20_student_v0.perception.SyntheticMapState`，保持同一套采集、延迟投递、地图缓存、按世界坐标重投影、过期处理和置信度计算。不会把陈旧的机身坐标向量直接作为当前地图使用。

| 配置 | 含义 |
| --- | --- |
| `teacher` | 无地图误差；每个控制周期获得干净地图，有效点 m=c=1。 |
| `student-v1` | 从 checkpoint 保存的 perception 读取当前训练分布；没有额外本体噪声或动作延迟。 |
| `lidar_fov` | 仅加入前后雷达水平视野限制和地形视线遮挡，无额外地图随机误差、本体噪声或动作延迟。 |
| `student-v0` | 更强地图扰动，增加前后雷达水平视野限制、地形视线遮挡、陀螺仪、重力方向、编码器噪声，以及执行动作延迟。 |
| `stress` | 在更强误差上增加周期性地图传输中断，检查地图陈旧、过期和全未知状态。 |

额外参数的唯一配置入口是 `deploy/deploy_mujoco/student_v0_profiles.json`。默认基础参数来自 checkpoint，配置文件按 profile 覆盖；`--noise-level` 可再覆盖地图扰动倍率，`--profile-file` 可提供另一份 JSON。启动摘要记录完整实际参数、模型来源和哈希。

`lidar_fov/student-v0/stress` 使用前后双雷达可见区，其总水平开角由配置文件中的 `lidar_horizontal_fov_deg` 指定。角度以各自传感器原点为中心，前雷达沿机身 +X、后雷达沿 -X；使用完整机身旋转矩阵，所以俯仰、侧倾和转向也改变视野。前后原点分别由 `sensor_offset_body_m`、`rear_sensor_offset_body_m` 指定，是待真机外参校准的工程近似。当前后视野开角相同时，两侧形成对称盲区；传感器位于不同位置，因此近处盲区不等于简单从机身中心剪掉两个扇区。

每个采样点必须同时满足“在某一雷达的水平视野内”和“从该雷达原点看未被地形挡住”，才允许采集。两雷达结果取并集，未知地形不会以仿真真值填补。矩形地图仍保留原来的全部采样点和 channel-major 排列，因此无需重新导出网络。`teacher/student-v1` 保留此前的完整视野，用作理想观测和训练分布参照；参数默认 `--noise-profile student-v1`，要启用双视野必须选择 `lidar_fov/student-v0/stress`。

视线遮挡是在对应传感器原点与地形顶点之间用 MuJoCo 射线判断前方表面是否阻挡，将不可见采样点标记为未知。射线 API 以最近交点距离为依据，见 [MuJoCo 官方 API](https://mujoco.readthedocs.io/en/3.10.0/APIreference/APIfunctions.html#mj-ray)。它仍是高程图误差模拟：没有垂直视野/量程限制、扫描线束、反射率、扫描运动畸变、机器人自遮挡或多帧地图融合，传感器位置也未用真机标定数据校准。

视野限制发生在地图采集之前。初次观测或持续没有被看见的盲区为 `h=m=c=0`；移动或转弯时，最近一次缓存中已观测过的世界点仍可重投影到当前两侧，置信度按原有地图年龄衰减并最终过期。因此地图有效性不等同于当前瞬时视野；这保留了已有缓存语义，不是使用盲区真值。这一实现仍是单份延迟地图块，未增加多帧融合。

想先看视野变化，把上面的键盘启动命令中 `--noise-profile student-v0` 换成 `--noise-profile lidar_fov`；按 `H` 查看学生实际输入中的盲区。需要同时模拟观测噪声则继续用 `student-v0`。本轮没有修改蒸馏环境或在训模型，已有模型能保持输入兼容，但在新盲区下的行走表现仍需由你启动仿真观察；此前完整视野评测不能代表修改后的效果。

置信度保留训练公式和语义：未知点 h=m=c=0，有效点随估计高度方差和地图年龄衰减；没有把仿真真值误差作为置信度。XY/yaw 偏差会改变取样位置，但其协方差尚未直接进入 c，摘要中明确写 `confidence_pose_covariance_included=false`。这次没有改变在训模型的观测语义。

本体噪声只加在陀螺仪、重力方向、腿关节位置、关节速度测量上；轮位置始终为零，命令和上一条 issued action 保持原契约。动作执行延迟单独建队列，记录 issued/applied 两者。`stress` 的地图中断禁止新采集，已采集的包仍按自身时间戳投递，陈旧地图超过有效年龄后转为未知。

`--friction-scale` 可单独对静态地形的滑动摩擦做控制变量测试，实际摩擦和接触混合优先级会生效。户外 XML 已包含路缘、接缝、楼梯、土路、圆石、坡面和低摩擦路段，仍是刚性地形近似。

## 保存轨迹和批量评测

在上述单回合命令后加 `--output-dir docs/my_student_test_001`，退出时保存会话摘要和压缩轨迹。输出目录必须不存在，避免覆盖前一次结果。键盘会话轨迹另含 `mode` 字段，区分趴下、起立、就绪、学生控制和阻尼阶段；`student_policy_steps` 记录实际学生控制步数。默认不指定输出目录时，不积累整段轨迹。

删除 `--viewer` 即为快速无窗口固定命令闭环；没有显式 `--steps` 时运行 1000 个控制步。窗口中的自动评测需要同时加 `--viewer --autoplay`。

以下命令串行比较不同地形和观测条件。所有回合只执行学生策略：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/home/ubuntu/miniconda3/envs/m20_wzh/bin/python \
  deploy/deploy_mujoco/deploy_student_v0.py \
  --policy-jit deploy/models/student_v0_21000/student_v0.pt \
  --model '/home/ubuntu/桌面/m20_1/M20_perception_Lidar_rl_2/M20_perception_Lidar_rl/deep_robotics_model/M20/mjcf/M20.xml' \
  --suite --track-course --steps 1000 \
  --suite-profiles teacher student-v1 student-v0 stress \
  --suite-terrains flat ascent descent rough \
  --suite-seeds 42 43 44 \
  --output-dir docs/my_student_suite_001
```

种子改变感知噪声，初始姿态和地形保持固定；clean 条件下不同种子并不产生不同的初始状态。进一步验证应覆盖不同速度、转弯、初始姿态、地形参数和物理条件。

自动评测主要看 `fell/first_fall_time_s`、`velocity_xy_rmse_m_s`、`velocity_yaw_rmse_rad_s`、`tilt_p95_deg`、`valid_fraction_mean`、`confidence_valid_mean`、`all_unknown_frames`，以及 `ever_outside_course`。键盘会话不汇总这些固定指令误差指标，可用保存的指令、观测和状态轨迹分析实际控制阶段。`crossed_finish/cleared_terrain` 只是末帧越线判据，不保证没有绕开某个障碍。

轨迹中的 `time_s/qpos/qvel/observation/action_issued/action_applied/command` 对应积分前的同一状态；`post_time_s/fell_post_step` 明确记录积分后的结果。速度误差是本回合实际测得的 RMSE，与训练 TensorBoard 的命令累积误差定义不同。

可用 `--snapshot PATH.png` 保存最后一个实际观测帧的地图渲染。没有桌面时设置 `MUJOCO_GL=osmesa LP_NUM_THREADS=1`，使用 CPU 软件渲染；本机此路径已验证。

## 使用后续 checkpoint

无需结束训练即可直接指定一个已经完整保存的模型：将 `--policy-jit ...` 换成 `--checkpoint logs/.../model_N.pt`。入口会拒绝 244 维教师 checkpoint。

也可以只导出网络后再部署：

```bash
/home/ubuntu/miniconda3/envs/m20_wzh/bin/python \
  deploy/deploy_mujoco/deploy_student_v0.py \
  --checkpoint logs/你的学生运行/model_N.pt \
  --export-only --export-dir deploy/models/student_v0_N
```

导出只读取学生权重并检查重载输出一致性，不创建训练 runner 或优化器。保持 TorchScript、`observation_contract.json`、`deployment.json` 同目录；后者让 `student-v1` profile 能恢复训练时的感知参数。

此前自动回放的实测与限制见 [评测报告](student_v0_sim2sim_20261013/report.md)。本次键盘操控修改按用户要求没有运行仿真、训练或测试，交互行为尚未实际验证。

本次双雷达视野修改仅做离线观测检查：角度边界、坐标变换、传感器原点与视线遮挡配对、实际适配器的输入形状/盲区三通道、地图缓存及过期处理、配置加载和命令行选项。未创建 MuJoCo 模型、未加载策略、未执行物理步进或训练，闭环行走效果和窗口交互跳过，留给你启动。

离线检查入口（不启动仿真）：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/home/ubuntu/miniconda3/envs/m20_wzh/bin/python tests/test_student_v0_fov.py
```

主配置名称映射：`teacher` 对应旧 `clean`，`student-v0` 对应旧 `realistic`，`student-v1` 对应旧 `train`。这些名称只选择感知配置，模型仍由 checkpoint 决定；旧名称和旧自定义 profile 文件保持兼容。
