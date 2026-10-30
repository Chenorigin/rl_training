# M20 真机策略部署操作指南

## 一条命令启动或部署（2026-10-29）

在 `rl_training` 根目录运行新入口。下面地址仅为本次 1515 网络示例；换机器时显式更换 SSH 目标，脚本会读取实际板卡角色、序列号及模型身份，不把地址当作身份。

```bash
# 启动双板当前已部署模型；进入板内 tmux，保留原 START_STUDENT 确认
bash scripts/tools/m20_robot.sh start --aos user@10.21.41.1 --gos user@10.21.31.104

# 指定 checkpoint：CPU 导出 → 双板校验/切换 → 启动到手柄待机
# --python 指向原来能够加载该 checkpoint 的完整导出环境
bash scripts/tools/m20_robot.sh start --aos user@10.21.41.1 --gos user@10.21.31.104 \
  --checkpoint logs/rsl_rl/deeprobotics_m20_cts/2026-10-28_20-30-29_370770_cts_allterrain_v13/model_15000.pt \
  --python /home/ubuntu/miniconda3/envs/m20_wzh/bin/python

# 已有完整导出 bundle 时，用 --bundle /path/to/bundle 替代 --checkpoint
# 只部署、不启动：把 start 改为 deploy

# 无电机检查、查看状态、查看最近一次工具运行的日志
bash scripts/tools/m20_robot.sh check --aos user@10.21.41.1 --gos user@10.21.31.104
bash scripts/tools/m20_robot.sh status --aos user@10.21.41.1 --gos user@10.21.31.104
bash scripts/tools/m20_robot.sh logs --aos user@10.21.41.1

# 正常停止 SDK；保留感知，不需要 GOS 在线、模型完好或现场准入确认
bash scripts/tools/m20_robot.sh stop --aos user@10.21.41.1
```

首次使用需要本机已有 `ssh`、`scp`、`safepush`，并已核验两板 SSH 主机密钥。密码由 SSH/sudo 在终端询问，不写入脚本或配置；不要通过管道自动回答现场确认。缺省安装入口是 `/home/user/m20_heightmap/current`，可用 `--release`、`--gos-release` 指定已有安装，脚本不会装 ROS/SDK、改标定、开 SDK 模式或放宽限位。

部署前输入 `DEPLOY_PRONE`，表示已让狗趴稳、SDK开启、急停就绪且无人操作手柄。随后原启动脚本仍要求 `START_STUDENT`。看到 `[M20_READY]` 才表示 SDK 初始化就绪；**站立和进入 RL 仍由现场手柄操作**。当前门禁在 R1 后需要单独 R2 再 L1；L2 进入策略。`Ctrl-b d` 只断开 tmux 显示，控制器继续在板端运行；要停止请用 `stop`，它也会取消尚未确认的启动。

工具只支持既有部署工程已接受的 CTS 契约，不自动挑“最新”、不自动把新版本加入白名单。当前相邻导出器已包含经过验证的 v13 支持；遇到未知版本会停止。两个板的 ONNX、契约和 checkpoint 身份必须一致。已有活动控制器时，直接 `start` 会拒绝重复启动；更换模型只在趴稳确认后正常停止旧实例。原 GOS 感知若健康则复用。

日志和状态保存到板端数据分区 `m20_control_traces/operator_*`，CPU 导出产物在 `logs/evaluation/m20_operator_*`。收到策略故障后，监督进程请求正常停止且不自动重启；断开 SSH 不会丢掉监督。`check` 只做 ARM/实时输入等无电机检查，完整手柄心跳门禁仍由原启动脚本执行。SDK 初始化成功不代表实际行走稳定，15000 已记录的目标越限问题并未由这个工具修复。

部署采用独立目录及逐板验收，不覆盖原控制器、配置或标定。若跨板激活中断，可能有一板已切换，工具会保留事务标记并阻止启动；`status` 会显示 `deployment_pending`，逐板备份位于 `runtime/activate_*/before.json`。此时先根据备份核对/恢复双板模型和 AOS 离线收据，确认身份一致后再完成原事务；没有 `force`/自动反复重试入口。工具不会把这种情况报为成功，也不会擅自热切或重启。

开发验证：本机隔离回归包含实际子进程的就绪、故障停止、待确认取消和中断清理；板端另做无电机检查。完整新工具驱动的物理启动/行走不作为自动测试执行。

以下为最初整理的手工操作指南（2026-10-29，源码基点 `494cd48`）；当时只核对源码，未连接板端、导出或启动电机。新工具及其验证范围见上方入口。旧记录中的地址和运行状态不能当作当前状态。

## 1. 先看这条部署路线

```text
训练电脑：选 checkpoint → 导出独立候选 → 离线检查 → 打包
两块板：核身份 → 传入新目录 → 本机构建 → 配置核对
AOS：ARM 模型数值验收
GOS：启动纯感知 → AOS 只读检查真实地形
两块板：现场检查并 commissioning
AOS：tmux 内启动 SDK → 真实手柄 R2、L1、L2 → 小范围验证 → 停止
```

**当前 CTS 仍有未解决的真机问题。** 最近已核对的 10 月 28 日记录中，CTS v10 `model_1000` 进入策略后报 `CTS nominal target outside software position limits`，之后用户要求停止 SDK，感知保留。没有持续行走或楼梯通过的结论。换新模型前先解决、离线复核这条关节目标越限；不要靠放宽限位或反复启动试错。详见[接手上下文及原始收据索引](m20_heightmap_student_deploy_context.md#2-最近部署状态与时间顺序)。

| 在哪里操作 | 工作与入口 |
| --- | --- |
| 训练电脑 `rl_training` | 训练 checkpoint、训练配置和仿真验证 |
| 训练电脑相邻 `m20_student_deploy` | 本文的导出、候选打包脚本；板端控制软件来源 |
| GOS 板 | 原生雷达、独立 LIO、CPU 高程；不加载策略，不发电机命令 |
| AOS 板 | 本体与手柄输入、ONNX 推理、SDK/FSM、电机指令 |

初次安装的新板不适合直接照“更换模型”流程运行：必须先具备已验证的原厂 SDK/数据服务、ROS 2 Foxy、ARM CPU ONNX Runtime、FastDDS、GOS 独立 LIO 和安装标定。缺少这些时先完成[原现场接线流程](../../m20_heightmap_ws/docs/REAL_ROBOT_STEPS.md)。

## 2. 确认你要部署的模型

| checkpoint 类型 | 本文使用的导出入口 | 真机适用边界 |
| --- | --- | --- |
| student-v3.5 | `prepare_v35_release.py` | 当前 MLP/ELU、789 维输入、16 维输出；需匹配原训练配置与反馈语义 |
| CTS student | `prepare_cts_release.py` | 完整 student encoder＋共享 decoder；独立 `raw_command` 输入 |
| 教师、Safe/Quiet 教师、其它 student 结构 | 不走本指南直接替换 | 先完成对应观测、网络及执行器适配；普通教师 ONNX 不能代替学生模型 |

`v1` 是部署 profile 和模型目录名，**不代表模型训练版本**。当前部署适配器显式接受 CTS v2/v8/v10；更新训练版本即使维度相同，也要先经过兼容性核验，不能直接加入白名单。

CTS ONNX 接口为 FP32 `obs[1,789]`、`raw_command[1,3]` → `actions[1,16]`。保存的是学生与 decoder，不含教师、critic、优化器。`raw_command` 来自航向反馈前的真实命令，不能从处理后的观测反推。

CTS 完整落脚控制需要经实测验证的触地、运动学、时钟和执行器反馈，目前硬件 binding 未验收。现场曾显式使用 `aos.cts_execution_mode="nominal_pd"`，只执行 nominal 动作和原 PD，跳过落脚修正。它属于执行消融，不能承诺与带完整控制器的仿真表现相同。缺省/`landing` 模式被拦截时不要绕过；本指南不把修改这个字段当作通用放行手段。详见[CTS 部署边界](../../m20_student_deploy/docs/CTS_DEPLOYMENT.md)。

## 3. 训练电脑：准备一个新候选

先选择一个**写入完成**的 checkpoint，记录训练 run、任务、checkpoint SHA256 和配置。不要按迭代数字大小判断哪个策略更适合真机。不要直接把训练 `.pt` 放进板端 ONNX 槽位。

下面路径均为你需要替换的占位符；输出目录和压缩包必须不存在。候选、压缩包和日志放在空间充足的数据盘。激活原来能够加载该 checkpoint 的 Python 环境，先确认 CPU、内存和磁盘余量；本文不要求 GPU，也不启动训练。

```bash
# 训练电脑；以下变量只在当前 shell 有效
export TRAINING=/absolute/path/to/rl_training
export DEPLOY_SOURCE=/absolute/path/to/m20_student_deploy
export CHECKPOINT=/absolute/path/to/model_N.pt
export CANDIDATE=/absolute/data/path/m20_candidate_NEW
export PAYLOAD=/absolute/data/path/m20_candidate_NEW.tar.gz

cd "$DEPLOY_SOURCE"
git rev-parse HEAD
git status --short
sha256sum "$CHECKPOINT"
free -h
df -h "$(dirname "$CANDIDATE")"
```

导出需要 PyTorch、NumPy、ONNX、ONNX Runtime、PyYAML 等部署工程依赖；在原工作环境操作，不在板端安装整套训练环境。下面两个分支**只执行与你模型对应的一个**：

```bash
# CTS
python scripts/prepare_cts_release.py \
  --checkpoint "$CHECKPOINT" --training "$TRAINING" --output "$CANDIDATE"
```

```bash
# student-v3.5；env.yaml 应为该 checkpoint 同轮训练保存的配置
python scripts/prepare_v35_release.py \
  --checkpoint "$CHECKPOINT" --env-yaml /absolute/path/to/env.yaml \
  --output "$CANDIDATE"
```

CTS 候选不继承旧 runtime、commissioning 或启动许可。v3.5 工具可能复制部分 runtime 配置，但同样不继承现场确认；复制来的配置必须重新比对，不能视为当前板端事实。

```bash
cd "$CANDIDATE"

# CTS 的电脑端检查；v3.5 不运行这两条 CTS 专用命令
python scripts/validate_cts_bundle.py --model-dir models/v1
python scripts/offline_cts.py --model-dir models/v1
```

确认 `models/v1/` 中模型、`observation_contract.json`、导出元数据与固定数值测试输入配套，保存整个候选，不只拷贝 `policy.onnx`。v3.5 准备工具已执行 bundle 校验与测试样本生成。任一导出或检查失败就停在电脑端。

```bash
# 两种候选通用：在候选根目录打包
python scripts/package_release.py --output "$PAYLOAD"
sha256sum "$PAYLOAD"
```

保存打包输出中的 SHA256 和 `PAYLOAD_SHA256.json`。不要向候选混入旧 `commissioned.json`、历史启动许可或旧构建产物。

## 4. 两板：核对身份、传输并构建

分别登录你本次确认的 AOS、GOS；地址使用当前设备台账/现场网络，不照抄历史 IP。以下只读命令在两板各执行一次，记录角色、序列号、boot 和原 release：

```bash
cat /opt/robot/os_type
tr -d '\000' < /proc/device-tree/serial-number
cat /etc/machine-id
cat /proc/sys/kernel/random/boot_id
ip -br addr
readlink -f /home/user/m20_heightmap/current
ps -eo pid,ppid,comm,args
free -h
df -h /home/user
```

原 current 路径只是历史常用入口；不存在时先查清本机部署位置。记录已有自研 SDK、感知 supervisor 的 PID、cwd、启动日志和配置 SHA；不要重复启动或停止身份不明的进程。目标文件比本地更新且不同，先保留并回灌现场改动。

在电脑端，用 `safepush` 将压缩包和候选中的 `scripts/stage_board_release.py` 分别传到两板的**新临时路径**。例如将已核实的 `user@本次AOS地址` 填入变量，GOS 重复同样操作：

```bash
# 训练电脑
export BOARD_LOGIN=user@本次AOS地址
export REMOTE_ARCHIVE=/home/user/m20_candidate_NEW.tar.gz
export REMOTE_STAGE=/home/user/m20_stage_NEW.py
safepush "$PAYLOAD" "$BOARD_LOGIN:$REMOTE_ARCHIVE"
safepush "$CANDIDATE/scripts/stage_board_release.py" "$BOARD_LOGIN:$REMOTE_STAGE"
```

在**每块目标板自己的 shell** 重新填写变量；不要假定电脑变量会传到 SSH。两板可以用相同目录名，但它们是两份独立构建：

```bash
export RELEASE=/home/user/m20_student_deploy/releases/candidate_NEW
export REMOTE_ARCHIVE=/home/user/m20_candidate_NEW.tar.gz
export REMOTE_STAGE=/home/user/m20_stage_NEW.py
export PAYLOAD_SHA=填写电脑端计算的压缩包SHA256
export BOARD_SERIAL=填写本块板实际读取的序列号

sha256sum "$REMOTE_ARCHIVE"
python3 "$REMOTE_STAGE" --archive "$REMOTE_ARCHIVE" \
  --sha256 "$PAYLOAD_SHA" --serial "$BOARD_SERIAL" --destination "$RELEASE"
cd "$RELEASE"
python3 scripts/prepare_board.py
```

解包程序核对序列号和压缩包摘要，只允许不存在的新目录，不切换 current，不启动应用。`prepare_board.py` 只生成本 release 的 DDS 配置；默认 AOS `eth2`、GOS `eth0`，不同接线需先核实再用 `--interface`。它默认**不会生成高程 runtime**。

构建前阅读本机资源余量、部署工程 README 及 `scripts/build_board.sh`，为原厂服务和现有任务保留余量。按其预算确定本次地址空间上限，随后两板分别运行：

```bash
export BUILD_LIMIT_MB=填写本次核准的内存预算MB
python3 scripts/run_with_budget.py --limit-mb "$BUILD_LIMIT_MB" \
  --receipt runtime/build_resource.json -- bash scripts/build_board.sh
```

构建入口固定使用单任务顺序编译，AOS 编译 SDK，GOS 编译感知。依赖本机原厂 SDK、ARM ORT、Foxy 和 native FastDDS。`run_with_budget.py` 只检查可用内存并限制地址空间，不会替你预留整机余量或限制总 CPU；失败就检查收据和依赖，不提高并发反复重试。电脑的 x86 `build/install` 不能拷到 ARM 代替编译。

## 5. 对齐配置并完成无电机验收

对比旧 release 与新候选的以下配置，保留已核实的物理标定和当前软件边界；不要整目录覆盖旧部署：

| 对象 | 要核对什么 |
| --- | --- |
| `runtime/heightmap_v1.local.json` | 当前感知、地图支持域、时效/质量门禁、CTS 执行模式 |
| `deploy/robot.json` | 关节顺序/方向/零偏、默认姿态、动作缩放、PD、位置与速度边界 |
| `deploy/site.json`、`deploy/gos_lio.yaml` | 实际传感器安装、坐标和独立 LIO 配置 |
| `models/v1/` | 模型部署族/版本、契约、几何及反馈参数与实物一致 |

参数权威为[部署参数台账](../../m20_student_deploy/docs/DEPLOYMENT_PARAMETERS.md)与实际 release 配置。本文不另抄一套可调数值。GOS 启动还要求已有且可校验的 `/home/user/staircase_gos_ws/lio_release`；缺少时先解决 LIO 依赖。

首次缺少 runtime 时，在两板各自新 release 内运行以下预览准备，再核对候选配置与实物。该命令不生成运控 commissioning，不构成标定通过：

```bash
python3 scripts/prepare_board.py --prepare-preview
```

GOS 先做 CTS 元数据校验（v3.5 改为 `validate_v35_bundle.py`）：

```bash
source scripts/board_env.sh
python3 scripts/validate_cts_bundle.py --metadata-only
```

AOS 做真实 ARM C++ 推理验收：

```bash
source scripts/board_env.sh

# CTS
python3 scripts/offline_cts.py --model-dir models/v1 \
  --probe install/m20_student_sdk/lib/m20_student_sdk/student_onnx_probe

# v3.5 使用这一条，替代上面的 CTS 命令
# python3 scripts/offline_board_models.py

# 模型与执行模式均符合后，自动复验并记录软件验收
python3 scripts/accept_offline_release.py
```

成功后应有 `runtime/offline_acceptance.json`，AOS 的 `STAGED_NOT_ACCEPTED.txt` 由程序移除。**不要手工删除 marker。** 这是离线软件验收，不是站立/行走许可。CTS 完整 landing binding 未通过时应保留阻断；不能为了验收把模式偷偷改成 nominal_pd。

## 6. 先启动 GOS 感知，检查 AOS 实际收到的地形

确认旧感知进程已由原操作者停止，再启动新副本；如果旧感知仍承担现场任务，不要先起第二套。建议 GOS 也在板内 tmux 运行：

```bash
# GOS，新 release 内；只启动感知
sudo -E bash scripts/operator_start_gos.sh v1
```

另开 AOS 终端，在相同候选 release 内：

```bash
source scripts/board_env.sh
python3 scripts/live_preflight.py --profile v1
python3 scripts/terrain_snapshot.py --profile v1
```

首次尚未 commissioning 时，部分检查可能仍被拦截：只做感知预览、完成实际几何检查，再执行下一节现场确认并重新预检；不能用忽略错误代替通过。

逐项观察：真实点云/IMU/位姿持续更新，跨板地形来源与时效正确；11 行 Y、17 列 X 的方向正确；未知格显示 `??`；台阶升高时 `h=base_z-ground_z-0.5` 应下降。`h=0` 不能单独证明平地，遮挡与过期格不能伪装成可信平地。GOS 日志在 `runtime/native_cloud.log`、`native_inputs.log`、`lio.log`、`pose.log`、`heightmap.log`。ROS 话题存在或进程 active 不等于真实输入通过。

如果新目录启动日志仍引用旧路径，核对 `install`、Python import 和 LIO 链接，必要时在正确目录重新构建；不要判成缓存问题。配置变更后重新加载感知并核对实际运行版本。

## 7. 两板现场确认，再启动 AOS SDK

到这里仍不应由代理自动启动 SDK。现场操作者先完成：

1. 场地清空，机器人有可靠支撑，实体急停可立即操作；阻尼会失去支撑力，不能靠人徒手接住机器人。
2. 按厂家流程切入 SDK 模式，确认唯一控制权，原厂运动命令静默；保留提供本体数据的原厂进程和 `basic_server`。不要杀掉整个原厂软件栈。
3. 确认雷达/IMU 安装、坐标、高程符号与覆盖、关节和速度保护范围。仅在新配置与实际检查一致时签收。
4. 两板在各自候选内分别执行：

```bash
source scripts/board_env.sh
python3 scripts/operator_commission.py --profile v1
```

实际检查完成后才输入 `CONFIRM_PHYSICAL_CHECKS`。生成的 `runtime/commissioned.json` 绑定本板 boot 与配置、robot、site、LIO 内容；重启或配置变化后须重新检查，不能拷贝/改写旧收据。签收只记录你的确认，不会代替实际测量。

重新运行 AOS 的只读预检。默认 strict 通过后，AOS 在**板内交互式 tmux** 运行：

```bash
# AOS：会话名已存在时使用 tmux attach -t m20_policy_NEW
tmux new -s m20_policy_NEW

# 进入 tmux 后重新设置本板真实候选路径
cd /home/user/m20_student_deploy/releases/candidate_NEW
sudo -E bash scripts/operator_start_aos.sh v1
```

入口会检查执行模式、ARM 模型、真实手柄、原厂数据 owner、现场确认与 live preflight，然后要求输入 `START_STUDENT`。**输入之后即启动 SDK，包含 offset 初始化命令，可能产生物理动作。** 不要把这一步当作探活，不直接运行 `student_control`，不手工伪造放行环境变量。

`supported-init` 是原受限初始化路径，会处理特定初始化告警；它不限制后续 L2，也不解决越限故障。strict 失败时先查原因，不把它当通用重试开关。

## 8. 手柄接管、现场验证和退出

真机使用已验证的原厂真实手柄：左摇杆平移，右摇杆转向。不要照搬 MuJoCo 的 `Z/C/R` 键，也不默认手机 App 已经过验证。

| 顺序 | 操作 | 必须看见的结果 |
| --- | --- | --- |
| 1 | 手柄心跳稳定、摇杆回中，单独按下并释放 R2 | 阻尼/待站立状态，站立重新使能；R2 不是物理断电 |
| 2 | 单独按 L1 | 日志出现站立切换，实物平稳站起；站立未稳不继续 |
| 3 | 单独按 L2 | 日志进入 `rl_control`，零命令下实物稳定；不能只看推理次数 |
| 4 | 小幅给定，及时回中 | 低速平地启停和方向正确，无明显失稳、越限、过期或控制权异常 |
| 5 | 结束或异常，先使用 R2；紧急情况用实体急停 | 实际停止且机身安全承托；再结束控制进程 |

当前手柄实现要求稳定心跳后用独立 R2 重新解锁站立；L1/L2 都受状态约束。看到 `Ignoring L1 ... settled heartbeat ... R2` 时，先回中、等心跳、单独 R2 再 L1，不连续乱按。进入 `rl_control` 也不代表楼梯能力通过。

正常结束：现场先回中、使用 R2 并确认承托安全，再在 AOS 控制终端 Ctrl-C，检查日志退出与本次 `student_control` PID 消失。SDK 的停止路径会发送阻尼，Ctrl-C 或 SSH 断开都不能代替实体急停。tmux detach 只断开显示，**控制器仍在运行**。

若正常退出失效，先保证物理安全；只针对已核对 exe/cwd/启动时间的本次 PID 处理，不使用模糊 `pkill -f`，不杀原厂数据进程。停止 GOS 时，在其 supervisor 终端 Ctrl-C；它清理本次感知子进程。仅停止策略时可以保留感知。

## 9. 出错时查哪一层

| 现象 | 下一步与停止位置 |
| --- | --- |
| 导出拒绝版本、结构、契约或源码哈希 | 留在电脑端，对齐训练与部署版本；不能只改模型名或白名单 |
| ARM 固定样本数值超差 | 保存差异与后端版本，检查模型配套和 ORT；不放宽容差后上机 |
| `STAGED_NOT_ACCEPTED.txt` 阻断 | 核对 AOS 离线验收及执行模式；不手删 marker |
| 无地形、时间/几何/commissioning 失败 | SDK 保持停止，查两板配置、boot、原始来源、LIO 和真实收件 |
| `Another GOS mapper is running` | 核旧 supervisor 身份与现场用途，由原操作者退出后再启动 |
| 手柄/owner/唯一发布者检查失败 | 停在启动确认前；确认厂家 SDK 模式与真实手柄，不关闭检查 |
| `CTS nominal target outside software position limits` | 停止策略，保存 fault 前后 raw/nominal action、映射后目标、关节名与边界；对齐动作映射/缩放，离线复核后再决定现场试验 |
| 推理有输出但机器人不动 | 核实际 FSM、SDK 模式、控制权及反馈；不把模型输出当作电机执行 |

## 10. 保存证据与回退

每轮在训练工程 `logs/evaluation/<本次实验名>/` 保存：训练来源和 checkpoint 摘要、候选与压缩包摘要、两板身份和构建收据、离线验收、配置摘要、commissioning、启动终端输出、AOS `runtime/control_trace_*.jsonl`、GOS 感知日志及现场结果。板端原始日志先保留，再按本轮目录归集；秘密不进入报告或仓库。

回退按“策略停止并确认物理安全 → 保留故障日志 → 确认原 release 配置/模型/二进制 → 必要时退出新 GOS、启动原感知 → 原副本重新预检与现场确认”的顺序。本文使用新目录绝对路径，不要求切换 current；不要为了恢复而覆盖原 release、复制旧 commissioning 或自动重启 SDK。

验收分开记录：文件到货、ARM 数值通过、真实感知通过、SDK 初始化、站立、零速策略、平地启停、楼梯。每项写通过/失败/未运行及证据；前一项通过不能替代后一项。

## 11. 依据与本次核验范围

- [部署工程 README](../../m20_student_deploy/README.md)、[CTS 部署说明](../../m20_student_deploy/docs/CTS_DEPLOYMENT.md)、[现场原流程](../../m20_heightmap_ws/docs/REAL_ROBOT_STEPS.md)。相邻目录需实际存在。
- 本次逐项读取 `prepare_*_release.py`、`package_release.py`、`stage_board_release.py`、`prepare_board.py`、`build_board.sh`、`accept_offline_release.py`、`operator_start_{gos,aos}.sh`、`operator_commission.py`、`live_preflight.py` 与真实手柄/FSM 源码核对命令和边界。
- 本次未执行模型导出、板端构建、离线推理或真机动作；指南属于基于当前源码整理的操作步骤，不是新策略真机验收收据。训练和真机工程的未提交工作树内容需与 HEAD 分别记录。

### 关节限位可复位故障

支持该功能的 SDK 以 `[STUDENT_RECOVERABLE_FAULT]` 请求阻尼并锁定，守护状态显示 `fault_latched`，进程保持接收手柄。原 `[STUDENT_FAULT]` 仍停止控制器。现场支撑、急停与状态检查恢复后，单独按 R1 确认、R2 复位，再用新的 L1、L2 站立和进入策略；没有自动续跑。复位日志为 `[STUDENT_RECOVERY_READY]`。旧版 SDK 或旧版守护程序不具备完整恢复链，须一起更新。地图输入敏感性不会由恢复流程修复。
