# StairTeacher 奖励模块统一

## 最终训练目标

`pre_teacher` 预训练 → 加载其 actor，启动一次 StairTeacher 联合训练 → 固定测试集验收。

在这一次楼梯训练中同时启用任务奖励、上楼交替、上下楼姿态、紧凑转向、能耗和接触约束，
结合多地形课程与域随机化学习泛化。课程可以在同一训练过程中调整难度和随机化范围。
开发期间的 resume 是检验候选设计的手段；正式配置需要经过验证后固化。
模块合并不保证一次训练就达到目标，也不修复先前发现的时间顺序同阶检测盲区。

## 文件组织

唯一实现是 `source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/stair_teacher.py`：

| 分节 | 内容 |
| --- | --- |
| 1 | `TUNING` 步态软包络参数 |
| 2 | 速度命令曝光统计、critic 特权观测 |
| 3 | 高程扫描几何、地形门控 |
| 4 | 前后轴逐阶事件账本、抬腿信用、落脚顺序 |
| 5 | 楼梯任务、姿态、接触及能耗奖励 |
| 6 | 前后腿分阶段折叠约束、紧凑转向、步态指标 |
| 7 | 距离课程和分地形 level 指标 |

`stair_teacher_env_cfg.py` 继续负责奖励启用、权重和实体绑定，所有楼梯和步态函数统一从
`stair_teacher` 引用。基础通用奖励仍使用既有 `rewards.py`。

旧 `stair_ascent.py`、`gait_refinement.py` 只导出统一模块的对象，兼容历史保存的 YAML/对象
函数路径，没有独立实现或参数副本。仓库不存在名为 `gait_reinforcement.py` 的当前实现，
本次合并的步态文件实际为 `gait_refinement.py`。

数值检查脚本和 Isaac 检查脚本已经迁移到统一模块。历史实验报告保留当时的文件路径；
今后修改应以本记录和统一模块为准。

## 自检证据

- 55 个函数和类的 AST 与合并前一致，仅移除了指回统一模块的局部导入；无重名定义。
- `TUNING` 字典、环境配置中的奖励语义和参数保持一致，环境配置只改变模块引用。
- 两组 CPU 检查的 26 + 52 个结果字段逐项完全一致，包含有效/无效扫描、交替、重复信用、
  同阶、支撑/摆动折叠和转向反例。它们验证合并前后的一致性，不证明现有设计没有奖励漏洞。
- Isaac 使用真实 StairTeacher 注册任务和最新 `model_149999.pt`，16 环境执行 60 控制步。
  actor 输入为 244，critic 为 285；动作、观测和奖励均有限。
- 7 个关键奖励的运行时函数来源均为统一模块；两个兼容模块中的全部导出对象与统一模块
  对象相同。相关实体解析正常。
- `pre_teacher_env_cfg.py` 和 `agents/pre_teacher_ppo_cfg.py` 哈希与合并前一致。
- pyflakes 无新增告警。原有未使用导入告警保留；`git diff --check` 通过。

报告见本目录 JSON。原始备份、冻结判据及完整运行日志位于
`.scratch/stair_teacher_merge_20261006/`，备份只含本次涉及的代码和文档，没有复制模型权重。
短回放没有上楼目标样本，不能用它证明登楼动作或奖励非零分支；这些分支由 CPU 数值轨迹检查。

可重跑：

```bash
conda activate m20_wzh
python scripts/tools/check_stair_rewards.py --assert-fixed
python scripts/tools/check_gait_refinement.py
python scripts/tools/smoke_stair_rewards.py \
  --checkpoint logs/rsl_rl/deeprobotics_m20_stair_teacher/2026-10-03_15-12-45_gait_resume/model_149999.pt \
  --num_envs 16 --steps 60 --headless \
  --output .scratch/stair_teacher_merge_20261006/recheck.json
```

## 范围与限制

没有调整奖励权重、门控、落脚规则、地形、域随机化、观测或动作定义；没有启动 PPO 训练，
没有更新模型权重，没有测试新步态质量或真机。仍需后续处理相邻阶交替检测、窄踏面分布和
极端摆动折叠问题，再验证最终从 pre_teacher 初始化的完整训练。

资源依据、整机余量和分配见 `resource_plan.json`。运行结束后的余量已读取，未采集运行期间
资源峰值。Isaac 启动有 CPU 拓扑和驱动警告，完整日志中有实际 runner、checkpoint 加载、
reset、控制步进及结果记录，没有把仅启动返回成功当作运行成功。

远端已 fetch：比当前底本多一个 M20S 提交，没有这三个本地楼梯模块。没有对用户已有
大量未提交改动执行全仓 pull/覆盖。

## 经验记录

**现象**：楼梯任务、逐阶事件及步态约束分散在三个实现文件，并通过局部导入互相访问。

**原因**：开发迭代过程中将新增功能分开，参数和调用入口随之分散。

**判据**：移动前后实现和数值一致；真实环境调用统一模块；历史函数路径仍可解析。

**修法**：集中实现并按职责分节，参数只维护一份，当前配置直接引用统一模块，旧路径只做兼容导出。

项目级 `company-rules check` 通过；全机分发扫描有 `~/.gnupg` 权限不足和其他检出跳过项，未修改其权限或配置。

经验条目已写入共享知识库本地 `memory/-home-ubuntu-cyq-shixi-projects-rl-training/stair-teacher-unified-20261006.md`。
同步前执行 `git fetch hub main` 失败：知识库 Git 松散对象损坏并缺少所需 commit，
因此没有提交或推送，也没有处理该共享仓库中其他窗口的未提交文件。
