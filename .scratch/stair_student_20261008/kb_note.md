# Stair student scaffold（共享知识库恢复后入库）

## 现象
用户要求618维student，teacher仍训练中；直接复用teacher critic会漏进特权量，普通train.py init_actor_from不能把244维严格加载到618维。
## 真因
观测继承和网络第一层维度不同；感知三通道如果各自生成噪声/缺点，会产生不一致输入。teacher actor labels不需要teacher critic。
## 判据
实际Isaac4env8step：policy/critic618，teacher244，16动作；199998 actor零扩展374列后干净输出最大误差0；噪声缺点模式actor/critic相同，h/c在v0处为0。CPU reset含同step重复reset以及错尺寸/坏参数拒绝通过。未训练或实机验证。
## 修法
新student模块、环境、PPO配置和独立任务；复制teacher最终57维绑定后换三通道，共享缓存并加reset EventTerm；critic复制student policy而非teacher critic；扩展加载helper留给未来专用蒸馏入口，目前只在真实冒烟调用。clean teacher扫描异常需未来label查询前校验，student占位不能用于伪造teacher标签。仅基础合成感知，未实现ROS/世界地图重投影和DAgger。teacher、pre、commands和部署冻结文件哈希一致。共享KB先前fetch因Git松散对象损坏失败，本轮仅本地归档，没有push或修外部库。
