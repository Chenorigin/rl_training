# MuJoCo 地形 XML

从项目根目录运行 README 中的部署命令，只需改 `--terrain-xml` 后的文件路径。相对路径相对于终端当前目录；内置 `--terrain NAME` 相对于部署程序目录，和终端位置无关。两种选择方式互斥。

## 文件与参数

| 文件 | 地形 | 主要可调参数 |
| --- | --- | --- |
| `flat.xml` | 无限平地 | 摩擦、出生位置、统计终点 |
| `stairs_ascent.xml` | 上楼梯及顶部平台 | `step_height`、`tread_depth`、`step_count`、`platform_length` |
| `stairs_ascent_open.xml` | 无踢面上楼梯，仅水平碰撞面 | 同上，另有低平台长度 `approach_length`、防坠底面 `floor_height` |
| `stairs_ascent_walls.xml` | 小区安全通道式双墙直跑楼梯及顶部平台 | `open_risers`、`walls_enabled`、阶高/踏深/级数、墙高/墙厚/间距 |
| `platform.xml` | 平地上的独立实心矩形平台 | `platform_height`、`platform_length`、`width`、`start_x` |
| `stairs_descent.xml` | 高平台下楼梯 | 同上；出生位置须在高平台内部 |
| `slope_up.xml` | 连续上坡及顶部平台 | `slope_deg`、`ramp_length`、`ramp_thickness`、`platform_length` |
| `slope_down.xml` | 高平台连续下坡 | 同上；出生位置须在高平台内部 |
| `rough.xml` | 随机不同高度的方块路面 | `length`、`cell_size`、`gap`、`height_min/max`、`seed` |
| `obstacles.xml` | 离散方块障碍 | `boxes` 每行：中心 X、中心 Y、顶面高度、X 长度、Y 宽度 |
| `hurdles.xml` | 横向障碍条 | 数量、间距、厚度、首条高度、逐条高度增量 |
| `mixed.xml` | 上楼、下楼、方块、上下坡和障碍条组成的连续路线 | 各地形段参数、段间缓冲距离、平台长度 |
| `all_terrain.xml` | 全地形长路线：上下楼、上下坡、粗糙方块、错位障碍、连续障碍条、横坡、低摩擦区 | 原生 `geom` 的位置、半尺寸、旋转、摩擦；各段参数及调参公式直接标在 XML |
| `realistic_outdoor.xml` | 园区表面近似：路缘石、混凝土接缝、建筑楼梯、修补路面、连续起伏土路、固定圆石、缓坡、湿路低抓地 | 原生几何位置/尺寸/旋转/摩擦、内嵌高度场振幅；坐标及公式标在 XML |
| `real_world_final.xml` | 生活场景与异形综合关卡：干湿瓷砖、门槛、盲道、格栅坑洼、树根木桥，以及变阶高楼梯、半圆扇形楼梯、错位踏步、屋脊/波浪坡、V 沟、多边形踏石、S 形窄桥 | 独立原生 XML，内嵌纹理、高度场和凸网格；几何和接触参数见 XML |
| `maze.xml` | 平地围墙迷宫：双侧墙直道、连续折返和逐渐收窄的开口 | 各墙 `geom` 的 `pos` / `size`；XML 注释说明净宽和参考路线 |
| `custom.xml` | 原生 MJCF 障碍示例 | 直接编辑 `worldbody` 中的 `geom`，可复制增加更多几何 |
| `sample_stairs.xml` | 按 `sample.mp4` 和实测外尺寸重建的五件活动楼梯组件 | 两件并排小梯、一件整梯、白平台、一个深色木台；原生几何、密度及地垫接触参数 |

每个文件均包含公共参数：`width`、`friction`、`spawn_xy`、`spawn_yaw_deg`，并在注释中说明单位与用途。具体数值以 XML 为准。

## 调整方法

例如调整楼梯单级高度，只需修改对应 XML 中 `step_height` 的 `data`；全部台阶顶面、平台总高度和下楼出生高度会同步重算。坡度修改 `slope_deg`，水平长度修改 `ramp_length`；坡顶高度由两者计算。

修改后重新启动部署程序。不要同时保留覆盖同一参数的命令行选项，否则以命令行值为准。下楼、下坡时缩短出生平台后，也要调整 `spawn_xy` 保证出生点在平台内部，建议为整个机器人留出足够余量。

`rough` 固定种子时可复现相同几何；修改种子生成另一组方块。方块之间有平地间隙，不模拟碎石移动。

`mixed` 沿世界 +X 拼接全部地形段；路线终点随各段长度自动更新。`flat/custom` 则由 `goal_x` 指定统计终点。

## 上楼梯：开放踏板与双侧墙独立切换

三个上楼模板共用同一个生成器，都可以直接改 XML 中以下两项：

```xml
<numeric name="open_risers" data="0"/>
<numeric name="walls_enabled" data="1"/>
```

| `open_risers` | `walls_enabled` | 生成结果 |
| --- | --- | --- |
| 0 | 0 | 普通实心楼梯，无墙 |
| 1 | 0 | 开放踏板楼梯，无墙 |
| 0 | 1 | 普通实心楼梯，左右实体墙 |
| 1 | 1 | 开放踏板楼梯，左右实体墙 |

`open` 指没有竖直踢面，不是“没有围墙”。`stairs_ascent_walls.xml` 默认是双墙实心楼梯，
模拟安全通道中的一段直跑楼梯，包含起步走廊和顶部平台，不包含折返段或天花板。
原 `stairs_ascent.xml` / `stairs_ascent_open.xml` 保留各自尺寸、摩擦和无墙默认值。
它们是不同预设；若要仅对比开关效果，请在同一 XML 中切换，避免混入原预设的尺寸/摩擦差异。

- `width`：踏面全宽；`wall_gap`：踏面每侧到墙内侧的净距。两墙内侧净宽 = `width + 2 * wall_gap`。
- `wall_height`：每段墙高出相邻踏面的高度；`wall_thickness`：墙的完整厚度。墙不会侵占踏面宽度。
- `approach_length`：首阶前的起步区长度；墙从此处连续延伸至顶部平台末端，出入口保持开放。
- `step_height` / `tread_depth` / `step_count`：阶高、踏深、正整数级数；顶部高度自动联动。
- `platform_length`：顶部平台长度。墙的长度和顶部高度随上述参数一起重算。
- `floor_height`：仅开放踏板模式使用的下方防坠平面。实心模式使用 z=0 地面。

墙是参与碰撞和扫描的真实 box，墙顶随台阶逐段升高。开放踏板本身仍只有水平碰撞面，
新增墙体独立存在；没有用隐藏踢面替代 open 模式。
开放模式出生点应位于起步区内；缩窄通道或更改出生点时为机器人和可选出生垫留足空间。
窄版 `stairs_ascent.xml` 已通过 `spawn_mat_size` 显式配置适合其通道的出生垫，
直接切换两个开关即可保留垫子。修改通道宽度、起步区长度、垫子位置或朝向后，
仍须保证整个垫子处于同一平面内且不碰墙；命令行 `--spawn-mat-size` 会覆盖 XML 尺寸。

在现有部署命令中替换地形选项，二选一：

```bash
--terrain-xml deploy/deploy_mujoco/terrains/stairs_ascent_walls.xml
# 或
--terrain ascent_walls
```

## 可调矩形平台

`platform.xml` 生成从 z=0 地面升起的固定实心平台，默认在前方地面出生。
修改以下 XML 参数后重启即可，不需要改 Python：

| 参数 | 含义 |
| --- | --- |
| `platform_height` | 台面距离地面的高度 |
| `platform_length` | 沿 X 的完整长度 |
| `width` | 沿 Y 的完整宽度，横向中心为 y=0 |
| `start_x` | 平台前缘的世界 X；后缘自动取前缘加长度 |
| `spawn_xy` / `spawn_yaw_deg` | 机器人出生位置、朝向；竖直高度自动按实际支撑面计算 |
| `friction` | 平台与地面的滑动、扭转、滚动摩擦 |

高度、长度、宽度必须为正数，尺寸单位均为米，填写全尺寸而非半尺寸。
平台前后都是竖直边缘，没有自动生成斜坡或楼梯。可将出生点改到台面内测试下台，
但必须为整个机体留足边缘余量；沿 +X 的越线统计终点自动设在平台后缘。

```bash
--terrain-xml deploy/deploy_mujoco/terrains/platform.xml
# 或
--terrain platform
```

建议纯地形对比时加 `--no-spawn-mat`。这些参数化 XML 需要通过现有部署加载器生成场景，
不能直接用通用 MuJoCo 查看器打开模板来查看生成结果。

## 围墙迷宫与敏感性对比

`maze.xml` 只增加静态实体围墙，地面保持平整，出生坐标/朝向与 `flat.xml` 相同。
先经过双侧墙直道，再按北、南、北、南的顺序绕过四道隔墙，最后从右下侧出口离开。
场地已压缩为紧凑布局，隔墙间距和开口宽度均已收窄，适合观察贴墙减速、反复摆头、动作抖动或无碰撞却停滞等行为。应连接的墙端有实体搭接，只有出入口和折返通道留空；入口横墙与外围墙的接头不再留侧缝。墙的具体尺寸和净宽标在 XML 中。

在原老师/学生部署命令中，将原 `--terrain` 或 `--terrain-xml` 选项替换为：

```bash
--terrain-xml deploy/deploy_mujoco/terrains/maze.xml --no-spawn-mat --viewer
```

人工键控通过迷宫；XML 的 `maze_route_xy` 仅记录参考路线，不会自动导航。
窄口前先在宽区域调整朝向，避免把真实撞墙或转弯空间不足误判成策略过敏。
本地 X11 支持组合方向键，侧移和转向也可配合使用。

对照时用同一模型、checkpoint、噪声配置和速度指令，把地形改成 `--terrain flat`，同样保留 `--no-spawn-mat`。
分开观察“尚未接触墙就出现的动作变化”和“实际撞墙后的反应”；若采用会产生噪声/遮挡的传感配置，两次测试保持一致。
该 XML 可以单独用 `python -m mujoco.viewer --mjcf=deploy/deploy_mujoco/terrains/maze.xml` 查看。
墙为 group 0 的真实碰撞 box，起终点色块仅为无碰撞 site。调整宽度或墙高时修改实际墙的 `pos` / `size` 后重启；不会改狗的出生位置。

## 出生垫位置

垫子和机器狗的位置独立设置。**只移动垫子用 `--spawn-mat-xy`，不要用 `--spawn-xy`（那是狗的出生坐标）。**
六个部署入口都支持以下选项；单位米，世界 +X 向前、+Y 向左，旋转角度单位为度：

```bash
--spawn-mat-xy -1.35 -0.35 --spawn-mat-yaw-deg 45 --spawn-mat-size 1.4 1.1 0.08
```

上述参数只改变垫子的中心、朝向及完整长宽厚；狗保留原出生 XY、朝向和关节姿态。
旋转垫子再偏移，可以只让一个角的脚位于垫外；厚度就是相对垫下地面的高差。
具体留哪只脚取决于模型、姿态和垫子边界。程序不要求四只脚都在垫上，也不会把狗平移回垫子中央或自动调整各脚使其同时接触。
Reset 仍按实际碰撞支撑确定竖直高度、防止穿透；垫外脚允许悬空，后续接触、俯仰/侧倾与稳定性由物理和当前控制器决定。

想永久保存，只修改当前地形 XML 的 `<custom>` 中这些垫子参数，保留狗的 `spawn_xy`、`spawn_yaw_deg`：

```xml
<numeric name="spawn_mat_xy" data="-1.35 -0.35"/>
<numeric name="spawn_mat_yaw_deg" data="45"/>
<!-- 完整长、宽、厚；不是 MuJoCo geom 的半尺寸 -->
<numeric name="spawn_mat_size" data="1.4 1.1 0.08"/>
```

同名项直接修改，不要重复添加。垫子命令行参数优先于对应 XML 参数。
省略垫子坐标/朝向时，默认先放在狗的出生点、与狗同向；这只是默认值，显式设置垫子参数不会修改狗的出生设置。
默认尺寸唯一来源是 `spawn_mat.py` 的 `DEFAULT_SIZE`。底面高度自动读取垫子所在位置的真实地面。
垫子自身需放在静止、近水平且连续的支撑面上，支持 box、plane 或固定三角 flex；深坑、台阶或活动组件上会明确报错。此检查针对垫子底面，不要求狗的四脚同时位于垫内。
追加 `--no-spawn-mat` 可关闭垫子。参数在启动时读取，修改后重启部署程序生效。

## 无竖直面的上楼梯

`stairs_ascent_open.xml` 与原上楼模板使用相同阶高、踏深、梯数参数。
可以直接修改 XML 的 `step_height`、`tread_depth`、`step_count`，也可以在启动时覆盖：

```bash
python deploy/deploy_mujoco/deploy_mujoco.py \
  --checkpoint /path/to/model_N.pt --model /path/to/M20.xml \
  --terrain-xml deploy/deploy_mujoco/terrains/stairs_ascent_open.xml \
  --stair-count 10 --stair-height 0.15 --tread-depth 0.35 \
  --command-x 0.4 --viewer --autoplay
```

也可用 `--terrain ascent_open` 代替 `--terrain-xml ...`，两者不要同时填写。
梯数必须为正整数，阶高和踏深为正数，长度单位为米。`--stair-start` 可覆盖首阶 X；
修改起步位置时，应使 `spawn_xy` 位于 `start_x-approach_length` 与 `start_x` 之间、宽度以内。
出口平台高度和统计终点会随梯数、阶高及踏深自动重算。

碰撞使用固定在 world 的原生二维 `flex`，每个踏面两个水平三角形，半径为零。
起步和出口平台也只有水平面，不存在踢面、侧面、底面或斜撑；
`walls_enabled=0` 时不生成墙，改成 1 时单独添加上节描述的两侧实体墙。
下方独立防坠平面高度由 `floor_height` 指定；零厚度边缘仍可能发生正常边缘接触。
采用的是 [MuJoCo 原生 flex](https://mujoco.readthedocs.io/en/3.10.0/XMLreference.html#deformable-flex)，
本机 MuJoCo 3.10 已实测 rigid 二维 flex 可用零半径编译和承重；不自动回退为薄盒子。

显示采用纯可视化的极薄 `site`，其顶面与碰撞面同位置；site 永不参与碰撞和射线检测。
碰撞和扫描仍使用实际零厚度 flex 三角面，显示对象不会提供侧面或底面支撑。
这样默认查看器能看见踏板，也不会多出无限碰撞平面。
射线、轮子法向力、落地指标和学生可见性均已适配 flex，避免把接触中的 `geom=-1`
误当成最后一个机器人碰撞体。原实心楼梯的生成方式保持原样。

几何回归验证了可调参数、水平面法向、踏面承重、原踢面位置可通过，以及旧地形加载；
这些检查不代表任意策略都能通过开放楼梯。

## 原生 XML 扩展

复制 `custom.xml`，保留公共参数和 `goal_x`，在 `worldbody` 下增加静态 `geom`。允许原生 `asset`（例如 mesh/hfield）和材料，其依赖路径按地形文件所在目录解析。不要在地形文件里重复定义机器人；独立可移动组件按下节的显式约定配置。

- 原生 `geom` 的 `size` 为半尺寸，`pos` 为中心坐标；与参数化模板的全长、顶面高度定义不同。
- 地形碰撞体须 `group="0"` 并启用碰撞；机器人碰撞与视觉组由机器人文件定义。
- 原生几何的摩擦在各 `geom` 中独立设置。公共 `friction` 用于自动生成的地面与参数化地形。
- 未提供平面时自动补充 z=0 无限平地；原机器人 XML 的静态 group-0 地面会被替换，避免重复碰撞。
- 高程图表示从上方探测到的表面，不适用于隧道内部、悬空桥下等多层空间感知。

普通模板的几何由 `terrain_loader.py` 生成，单独在通用 MuJoCo 查看器打开模板不会显示生成的楼梯/斜坡；请通过 `deploy_mujoco.py` 加载。

## 视频中的五件活动组件

`sample_stairs.xml` 可直接在 MuJoCo 打开，也可以作为现有老师/学生部署入口的 `--terrain-xml`。
尺寸以 XML 中的实际 `geom size/pos` 及相邻中文注释为准；`size` 是半尺寸。
沿 +X 先经过两件并排的浅踏深小梯，再经过白色平台，最后从整件深踏深楼梯下楼。
唯一深色木台在白平台的 -Y 侧。白平台横向居中及组件间隙是摆放近似。

```bash
python -m mujoco.viewer --mjcf=deploy/deploy_mujoco/terrains/sample_stairs.xml
```

在已有机器人测试命令中使用：

```bash
python deploy/deploy_mujoco/deploy_mujoco.py \
  --model /path/to/M20.xml --checkpoint /path/to/model_N.pt \
  --terrain-xml deploy/deploy_mujoco/terrains/sample_stairs.xml --viewer
```

五件各有一个根 `freejoint`，依靠地垫与相邻组件接触支撑，均可平移、转动、倾晃。
白平台的四根细腿各用两个正交被动弹性铰链等效支腿/连接处柔度，台面会相对脚底摆动。
`white_leg_flex` 默认类中的 `stiffness`、`damping`、`range` 分别控制刚度、耗散和内部角度范围；
这些弹簧只连接腿与台面，不把平台拴在世界坐标。脚底仍会接触、离地或滑移。
木件按空心板材构成整体；没有周期晃动动画。地垫摩擦控制滑移，惯量由板材及质量参数计算。
外尺寸来自用户；支腿截面、质量、刚度、阻尼、摩擦和接触柔度为估算，未按实物踩踏标定。
白平台与深色木台的净缝及高差在 XML 的白平台注释中说明；开口内没有隐藏支撑或过渡斜坡，
轮子经过时会遇到实际台面边缘，且平台摆动会动态改变净缝。
合成载荷测试包含同几何/质量的锁腿对照、载荷下摆动及恢复、净缝射线和轮子撞台面边缘。
这些验证不代表真实机器人的通过率或实物晃动幅度已经匹配。
使用 StairQuiet/Width/Safe checkpoint 时，请走 `deploy_quiet_landing.py` 并传入同一个
`--terrain-xml`。Safe 的原生组件安全观测读取当前 box/plane 碰撞面及其运动，保留实际缝隙；
地图保留已交付扫描及未知点。详见[Safe 手动部署](../../../docs/m20_stair_quiet_v1_safe.md#mujoco-手动部署)。
参考 [MuJoCo 自由关节](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-freejoint)
和 [接触参数](https://mujoco.readthedocs.io/en/stable/modeling.html#contact-parameters)。

扩展同类活动地形时：保持 `terrain_kind=custom`，在 `custom/numeric` 中声明
`dynamic_components` 为顶层组件数量；每件必须有一个自由根关节，其刚度、阻尼和干摩擦为零，无驱动器或对世界固定。
需要内部柔度时，用 `passive_internal_joints` 显式声明内部关节总数；内部只接受有界、
零弹簧平衡角、正刚度及正阻尼的被动 hinge。不允许 mocap、重力补偿、驱动器、equality 或 tendon。
碰撞体必须命名、使用 group 0；纯视觉几何用无碰撞的 group 2，或使用 site。
`terrain_` 是加载器保留的命名空间，机器人自身不能使用此 body/geom 前缀。
高度扫描查询组件当前姿态，轮地接触统计也包含活动组件；重新 reset 会恢复所有组件初始位置。
原 XML 的几何和接触属性直接生效，公共 `width/friction` 不会反向改写原生几何。

## 全地形测试路线

`all_terrain.xml` 使用原生静态碰撞几何，既可由当前部署入口加载，也可单独编译查看地形。
它有多段平地缓冲，可测试连续穿越、停车再启动、转向、侧移和反向通过。
横坡的入口/出口还包含高度变化，低摩擦区使用有厚度的薄板覆盖地面。

```bash
conda activate m20_wzh
python deploy/deploy_mujoco/deploy_mujoco.py \
  --model /path/to/M20.xml \
  --checkpoint /path/to/teacher/model_N.pt \
  --terrain-xml deploy/deploy_mujoco/terrains/all_terrain.xml \
  --viewer
```

出生后按 `Z` 起立、`C` 切换策略。各段长度、阶高、坡角、障碍尺寸和摩擦以 XML 为准。
这里的几何直接由 `geom` 定义，修改公共 `width`、`friction` 元数据不会自动改写已有几何；
实际表面摩擦在该 XML 的命名 `default` 和局部 `geom` 中设置。
单类楼梯的 `--stair-height/--tread-depth/--stair-count/--stair-start` 不能覆盖此综合地形。

使用上方部署命令和实际机器人 XML、checkpoint 检查地形加载。

## 园区户外测试路线

`realistic_outdoor.xml` 将离散方块换成混凝土板、轻微倾斜的修补板、固定圆石和连续起伏高度场，
并加入路缘石、建筑楼梯和缓坡。所有资产嵌在 XML，参数注释可直接编辑。
土路和潮湿表面分别是刚性起伏与低摩擦近似，不模拟土壤变形、散石移动或液体。
使用上方部署命令，将地形参数改为：

```bash
--terrain-xml deploy/deploy_mujoco/terrains/realistic_outdoor.xml
```

策略能否通过完整路线，需要对所选 checkpoint 单独评测。

## 控制与输出

键盘流程：趴下 → `Z` 站起 → `C` 策略；`W/S/A/D/Q/E` 控制速度，`H` 切换 187 点高程显示。

终点字段只做沿 +X 的几何越线统计，未检查是否绕过障碍；侧向、倒退测试应查看实际轨迹、姿态和跌倒情况。调整地形不会改变 Actor/控制契约，也不代表已验证策略的全地形能力。
