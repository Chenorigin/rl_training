# MuJoCo 地形 XML

从项目根目录运行 README 中的部署命令，只需改 `--terrain-xml` 后的文件路径。相对路径相对于终端当前目录；内置 `--terrain NAME` 相对于部署程序目录，和终端位置无关。两种选择方式互斥。

## 文件与参数

| 文件 | 地形 | 主要可调参数 |
| --- | --- | --- |
| `flat.xml` | 无限平地 | 摩擦、出生位置、统计终点 |
| `stairs_ascent.xml` | 上楼梯及顶部平台 | `step_height`、`tread_depth`、`step_count`、`platform_length` |
| `stairs_descent.xml` | 高平台下楼梯 | 同上；出生位置须在高平台内部 |
| `slope_up.xml` | 连续上坡及顶部平台 | `slope_deg`、`ramp_length`、`ramp_thickness`、`platform_length` |
| `slope_down.xml` | 高平台连续下坡 | 同上；出生位置须在高平台内部 |
| `rough.xml` | 随机不同高度的方块路面 | `length`、`cell_size`、`gap`、`height_min/max`、`seed` |
| `obstacles.xml` | 离散方块障碍 | `boxes` 每行：中心 X、中心 Y、顶面高度、X 长度、Y 宽度 |
| `hurdles.xml` | 横向障碍条 | 数量、间距、厚度、首条高度、逐条高度增量 |
| `mixed.xml` | 上楼、下楼、方块、上下坡和障碍条组成的连续路线 | 各地形段参数、段间缓冲距离、平台长度 |
| `all_terrain.xml` | 全地形长路线：上下楼、上下坡、粗糙方块、错位障碍、连续障碍条、横坡、低摩擦区 | 原生 `geom` 的位置、半尺寸、旋转、摩擦；各段参数及调参公式直接标在 XML |
| `realistic_outdoor.xml` | 园区表面近似：路缘石、混凝土接缝、建筑楼梯、修补路面、连续起伏土路、固定圆石、缓坡、湿路低抓地 | 原生几何位置/尺寸/旋转/摩擦、内嵌高度场振幅；坐标及公式标在 XML |
| `custom.xml` | 原生 MJCF 障碍示例 | 直接编辑 `worldbody` 中的 `geom`，可复制增加更多几何 |

每个文件均包含公共参数：`width`、`friction`、`spawn_xy`、`spawn_yaw_deg`，并在注释中说明单位与用途。具体数值以 XML 为准。

## 调整方法

例如调整楼梯单级高度，只需修改对应 XML 中 `step_height` 的 `data`；全部台阶顶面、平台总高度和下楼出生高度会同步重算。坡度修改 `slope_deg`，水平长度修改 `ramp_length`；坡顶高度由两者计算。

修改后重新启动部署程序。不要同时保留覆盖同一参数的命令行选项，否则以命令行值为准。下楼、下坡时缩短出生平台后，也要调整 `spawn_xy` 保证出生点在平台内部，建议为整个机器人留出足够余量。

`rough` 固定种子时可复现相同几何；修改种子生成另一组方块。方块之间有平地间隙，不模拟碎石移动。

`mixed` 沿世界 +X 拼接全部地形段；路线终点随各段长度自动更新。`flat/custom` 则由 `goal_x` 指定统计终点。

## 原生 XML 扩展

复制 `custom.xml`，保留公共参数和 `goal_x`，在 `worldbody` 下增加静态 `geom`。允许原生 `asset`（例如 mesh/hfield）和材料，其依赖路径按地形文件所在目录解析。不要在地形文件里重复定义机器人或放入动态 `body`。

- 原生 `geom` 的 `size` 为半尺寸，`pos` 为中心坐标；与参数化模板的全长、顶面高度定义不同。
- 地形碰撞体须 `group="0"` 并启用碰撞；机器人碰撞与视觉组由机器人文件定义。
- 原生几何的摩擦在各 `geom` 中独立设置。公共 `friction` 用于自动生成的地面与参数化地形。
- 未提供平面时自动补充 z=0 无限平地；原机器人 XML 的静态 group-0 地面会被替换，避免重复碰撞。
- 高程图表示从上方探测到的表面，不适用于隧道内部、悬空桥下等多层空间感知。

普通模板的几何由 `terrain_loader.py` 生成，单独在通用 MuJoCo 查看器打开模板不会显示生成的楼梯/斜坡；请通过 `deploy_mujoco.py` 加载。

## 全地形测试路线

`all_terrain.xml` 使用原生静态碰撞几何，既可由当前部署入口加载，也可单独编译查看地形。
它有多段平地缓冲，可测试连续穿越、停车再启动、转向、侧移和反向通过。
横坡的入口/出口还包含高度变化，低摩擦区使用有厚度的薄板覆盖地面。

```bash
conda activate m20_wzh
python deploy/deploy_mujoco/deploy_mujoco.py \
  --model "/home/ubuntu/桌面/m20_1/M20_perception_Lidar_rl_2/M20_perception_Lidar_rl/deep_robotics_model/M20/mjcf/M20.xml" \
  --checkpoint logs/rsl_rl/deeprobotics_m20_stair_teacher/2026-10-06_22-06-56_stair_resume/model_156300.pt \
  --terrain-xml deploy/deploy_mujoco/terrains/all_terrain.xml \
  --viewer
```

出生后按 `Z` 起立、`C` 切换策略。各段长度、阶高、坡角、障碍尺寸和摩擦以 XML 为准。
这里的几何直接由 `geom` 定义，修改公共 `width`、`friction` 元数据不会自动改写已有几何；
实际表面摩擦在该 XML 的命名 `default` 和局部 `geom` 中设置。
单类楼梯的 `--stair-height/--tread-depth/--stair-count/--stair-start` 不能覆盖此综合地形。

几何、高程及部署加载检查见 [全地形检查记录](../../../docs/all_terrain_20261007/README_CN.md)。

## 园区户外测试路线

`realistic_outdoor.xml` 将离散方块换成混凝土板、轻微倾斜的修补板、固定圆石和连续起伏高度场，
并加入路缘石、建筑楼梯和缓坡。所有资产嵌在 XML，参数注释可直接编辑。
土路和潮湿表面分别是刚性起伏与低摩擦近似，不模拟土壤变形、散石移动或液体。
使用上方部署命令，将地形参数改为：

```bash
--terrain-xml deploy/deploy_mujoco/terrains/realistic_outdoor.xml
```

完整命令、预览和本轮加载/高程/碰撞检查见 [步态与新地形评测](../../../docs/gait_quality_20261007/README_CN.md)。
该检查包括部署入口短时运行，尚未验证模型全路线通行能力。

## 控制与输出

键盘流程：趴下 → `Z` 站起 → `C` 策略；`W/S/A/D/Q/E` 控制速度，`H` 切换 187 点高程显示。

终点字段只做沿 +X 的几何越线统计，未检查是否绕过障碍；侧向、倒退测试应查看实际轨迹、姿态和跌倒情况。调整地形不会改变 Actor/控制契约，也不代表已验证策略的全地形能力。
