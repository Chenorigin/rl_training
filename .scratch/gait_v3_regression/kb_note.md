# M20 gait_v3：任务完成与步态风格分开，续训保护已学能力

## 现象
2026-10-08 gait_v3续训1700轮后，TensorBoard回报下降、平移误差增大；固定15cm五级楼梯MuJoCo闭环从基准可通过变成停在首阶前，平地侧偏从约0.17m增到6.47m。同阶计数减少但稳定踩阶总量也减少。

## 真因与证据边界
主转换奖励要求相邻异侧安全落脚；第一脚不付主奖、前后同阶各罚。旧v3前期顺序/同阶成本约为主转换收益11倍，合法任务完成但风格不佳也可能没有主奖、总罚36。actor-only初始化重建critic，未保护旧策略；adaptive LR可上涨到指定初值以上。固定测试证明退化，奖励结构与策略漂移是有证据的风险机制，未完成大规模因果消融，不把单一因果说成定论。

## 判据
固定物理模型、同一楼梯尺寸/命令/步数，基准和候选逐项评测任务通过/走廊/跌倒/平地轨迹；回报、terrain level、timeout、风格事件计数不能替代任务性能。对事件指标同时看样本暴露总量与归一化比例。恢复优化器时读取param_group实际LR，不仅读配置/self.learning_rate。

## 修法
独立全身物理穿越任务奖，严格交替保留额外风格收益；降低支配性的事件罚与重复任务折扣。继承任务可用的actor/critic，重置optimizer；critic先预热而actor冻结；冻结参考actor动作约束在PPO同一loss内，固定小LR、探索上限。辅助参考/预热计数checkpoint保存恢复；legacy checkpoint无辅助参考时自动冻结所加载actor，fixed schedule恢复后强制同步optimizer实际LR，防止保护静默缺失。短55轮（50critic+5actor）只能证明接线与短期保留能力，不能证明长期收敛或严格步态已改善。

生产与证据：/home/ubuntu/cyq_shixi_projects/rl_training/docs/gait_v3_regression/README_CN.md。新算法rl_training/stair_guarded_ppo.py采用实际安装官方RSL-RL5.0.1底本；其他版本未验证。没有停止原用户训练、没有自动启动长跑；新源码不会热加载进旧进程。
