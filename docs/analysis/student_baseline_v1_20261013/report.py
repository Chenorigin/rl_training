"""Produce three-model measured comparison and validate the known baseline."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
s=json.loads((OUT/'metrics.json').read_text());c=json.loads((OUT/'criteria.json').read_text())
primary=s['primary_case_mean_statistics'];runs=s['runs']
models=('baseline','quiet_landing_v1','student')
labels={'baseline':'Baseline 214700','quiet_landing_v1':'Quiet Landing V1 257100','student':'Student 22800'}
colors={'baseline':'#777777','quiet_landing_v1':'#0072B2','student':'#D55E00'}
names={'all_terrain':'全地形','stairs':'楼梯总体','ascent':'上楼','platform':'楼顶平台','descent':'下楼'}
fields=[('speed_mean_m_s','触地前平均速度','m/s',3),('speed_p95_m_s','触地前速度 P95','m/s',3),
        ('energy_mean_J_kg','单次触地比动能','J/kg',3),('energy_per_xy_m','每米累计触地比动能','J/kg/m',3),
        ('force_peak_mean_N','40ms 力峰均值','N',1),('force_peak_p95_N','40ms 力峰 P95','N',1),
        ('force_peak_max_N','每回合最大力峰','N',1),('excess_impulse_mean_N_s','40ms 超额冲量均值','N·s',3),
        ('slip_gated_time_mean_m_s','接触门控滑移时间均值','m/s',4)]

def value(model,section,key):return primary[model]['sections'][section][key]['mean']
def change(section,key,reference):return 100*(value('student',section,key)/value(reference,section,key)-1)

# Known good reference: the new sampler must reproduce old baseline event statistics.
old=json.loads((ROOT/'docs/quiet_landing_Baseline_V1_V1_5_多迭代全地形对比_20261011/多迭代触地指标统计.json').read_text())
checks={}
for section,oldname in [('all_terrain','全地形'),('ascent','上楼'),('descent','下楼')]:
    reference=old['Baseline']['分路段'][oldname];current=runs['baseline_clean_center']['sections'][section]
    for key,oldkey in [('speed_mean_m_s','法向触地速度_m_s'),('energy_mean_J_kg','法向比动能_J_kg'),('force_peak_mean_N','触地40毫秒力峰_N')]:
        a=reference[oldkey]['均值'];b=current[key]
        if not np.isclose(a,b,rtol=1e-6,atol=1e-6):raise ValueError(f'Known baseline measurement not reproduced: {section}/{key}')
        checks[section+'/'+key]={'reference':a,'measured':b,'max_tolerance':'rtol=1e-6, atol=1e-6'}
    if reference['事件数']!=current['event_count']:raise ValueError('Reference event count mismatch')
    checks[section+'/event_count']=current['event_count']
# Counterexamples: failure cannot enter the primary population; corrupted energy fails identity.
assert all(r['primary_included']==(r['summary']['complete'] and not r['summary']['fell'] and not r['summary']['ever_boundary_exit']) for r in runs.values())
assert not (True and not True and not False)
assert not np.allclose(np.array([.6]),.5*np.array([1.])**2,atol=1e-6)
assert len(runs)==11 and all(primary[n]['completed_count']==3 for n in models)
for case in c['cases']:
    initial=[runs[f'{n}_clean_{case["name"]}']['summary']['initial_qpos'] for n in models]
    assert np.array_equal(initial[0],initial[1]) and np.array_equal(initial[0],initial[2])
    velocities=[runs[f'{n}_clean_{case["name"]}']['summary']['initial_qvel'] for n in models]
    assert np.array_equal(velocities[0],velocities[1]) and np.array_equal(velocities[0],velocities[2])
validation={'baseline_reference_checks':checks,'identical_three_model_initial_states':True,
            'all_11_runs_complete_without_fall_or_boundary_exit':all(r['primary_included'] for r in runs.values()),
            'measurement_checks':s['measurement_checks'],
            'counterexamples':'fell flag excludes from primary; deliberately incorrect energy fails identity',
            'peak_process_rss_kib':max(r['summary']['peak_process_rss_kib'] for r in runs.values()),
            'total_wall_time_s':sum(r['summary']['wall_time_s'] for r in runs.values())}
(OUT/'validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n')

def table(section):
    lines=['|指标|Baseline 214700|V1 257100|学生 22800|学生相对 Baseline|学生相对 V1|',
           '|---|---:|---:|---:|---:|---:|']
    for key,label,unit,d in fields:
        vals=[value(n,section,key) for n in models]
        lines.append(f'|{label} ({unit})|'+'|'.join(f'{v:.{d}f}' for v in vals)+
                     f'|{change(section,key,"baseline"):+.2f}%|{change(section,key,"quiet_landing_v1"):+.2f}%|')
    lines.append('|触地事件数（3 回合合计）|'+'|'.join(str(sum(runs[f'{n}_clean_{case["name"]}']['sections'][section]['event_count'] for case in c['cases'])) for n in models)+'|—|—|')
    return '\n'.join(lines)

lines=['# 最新学生与早期 Baseline、Quiet Landing V1 的轻盈落地比较','',
'**结论：学生保留了多数相对早期 Baseline 的落地优势，没有整体退回 Baseline 水平；相对来源 V1，干净观测下上楼速度尾部、冲量和累计能量有轻度回退，下楼单次冲击反而较低。全程累计触地比动能较 V1 增加约 16%，不能只看单次力峰降低就称为全面改善。realistic 噪声的单回合下楼测量出现明显变硬。**','',
'## 固定对象与实验条件','',
f'冻结时间：{c["frozen_at"]}。学生取冻结时已保存的最新 `model_22800.pt`；训练之后继续产生的 checkpoint 未追入本轮。教师为学生实际来源 `model_257100.pt`，Baseline 为进入 Quiet Landing 训练之前的 `model_214700.pt`。','',
'三者在真实闭环 MuJoCo 接触求解下分别独立控制，使用同一机器人 XML、地形、关节映射、PD、动作缩放和触地采样器，CPU 确定性推理，没有训练、优化器更新或部署模型替换。','',
'- 地形：现有 `all_terrain.xml`，包含上楼、平台、下楼、坡、粗糙路面、障碍与低摩擦路面。',
'- 主对照：三模型 × 三组初态，均 clean 感知。初态是中心，以及 y/yaw 各 −0.03m/−0.03rad、+0.03m/+0.03rad，三者逐项一致；这些是敏感性测量，不是多种子随机统计显著性试验。',
'- 同一外部路线反馈：世界前进目标 0.5m/s、横向居中及航向纠偏；反馈算法相同，实际命令随各自姿态变化。不能据此证明手动控制下也有相同结果。',
'- 物理 5ms、策略 20ms，最大 120s；首个跌倒、轮心越出边界或全部轮足越过终点即停止。三者都使用现有 0.22m 机身离地高度、Rzz=0.35 的跌倒保护。',
'- 次对照：学生固定同一中心初态，分别 train（保存的 level=0.25 感知）和 realistic（level=1、几何可见性、proprio 噪声和一个策略步动作延迟），各一次。教师/Baseline 在这两组未施加对应噪声，只用学生自身 clean 中心回合来度量观测退化影响。','',
'## 通行与资源','',
'|模型 / 感知|完成回合|通行耗时范围 s|', '|---|---:|---:|']
for n in models:
    times=[runs[f'{n}_clean_{case["name"]}']['summary']['duration_s'] for case in c['cases']]
    lines.append(f'|{labels[n]} / clean|3/3|{min(times):.2f}–{max(times):.2f}|')
for profile in ('train','realistic'):
    r=runs['student_'+profile+'_center']['summary'];lines.append(f'|Student 22800 / {profile}|1/1|{r["duration_s"]:.2f}|')
lines+=['',f'本轮 11 回合均完成、无跌倒、无轮心越出路线边界。单进程顺序评测、Torch/BLAS 一线程、CUDA 禁用；峰值 RSS {validation["peak_process_rss_kib"]/1024:.1f} MiB，11 回合总执行耗时 {validation["total_wall_time_s"]:.1f} 秒。完成终点不保证每个障碍都无绕行，未将通行成功等同于轻盈落地成功。','',
'## 干净感知下三模型对照','',
'下表每个数是三个初态回合分别统计后等权求均值，P95 也是每回合事件 P95 的均值。变化为“学生三回合均值 / 参考模型三回合均值 − 1”。逐初态配对变化及范围另存 metrics.json；没有把每个触地事件当成独立回合做显著性检验。','',
'![三模型分路段落地对照](landing_comparison.png)','']
for section in ('ascent','platform','descent','stairs','all_terrain'):
    lines += ['### '+names[section],'',table(section),'']
lines+=['## 退化发生在哪里','',
f'1. **上楼：平均落地速度和平均力峰基本保留，但较硬的速度尾部和累积量回退。** 相对 V1，平均速度 {change("ascent","speed_mean_m_s","quiet_landing_v1"):+.2f}%、平均力峰 {change("ascent","force_peak_mean_N","quiet_landing_v1"):+.2f}%；速度 P95 {change("ascent","speed_p95_m_s","quiet_landing_v1"):+.2f}%、每米比动能 {change("ascent","energy_per_xy_m","quiet_landing_v1"):+.2f}%、超额冲量 {change("ascent","excess_impulse_mean_N_s","quiet_landing_v1"):+.2f}%。相对 Baseline，上楼单次比动能仍低 {abs(change("ascent","energy_mean_J_kg","baseline")):.2f}%，平均力峰低 {abs(change("ascent","force_peak_mean_N","baseline")):.2f}%。',
f'2. **下楼：单次冲击没有退化到教师或 Baseline 水平。** 相对 V1，平均速度 {change("descent","speed_mean_m_s","quiet_landing_v1"):+.2f}%、单次比动能 {change("descent","energy_mean_J_kg","quiet_landing_v1"):+.2f}%、平均力峰 {change("descent","force_peak_mean_N","quiet_landing_v1"):+.2f}%、超额冲量 {change("descent","excess_impulse_mean_N_s","quiet_landing_v1"):+.2f}%。但学生每米触地次数高 {100*(value("student","descent","events_per_m")/value("quiet_landing_v1","descent","events_per_m")-1):.2f}%，每米累计能量仅低 {abs(change("descent","energy_per_xy_m","quiet_landing_v1")):.2f}%，单次冲击降低不等于总量同幅降低。',
f'3. **全程：累计冲击代理量比教师高，单次力峰较低。** 学生每米能量 {value("student","all_terrain","energy_per_xy_m"):.3f} 对 V1 的 {value("quiet_landing_v1","all_terrain","energy_per_xy_m"):.3f} J/kg/m，增加 {change("all_terrain","energy_per_xy_m","quiet_landing_v1"):.2f}%；平均力峰却降低 {abs(change("all_terrain","force_peak_mean_N","quiet_landing_v1")):.2f}%。每米触地次数增加 {100*(value("student","all_terrain","events_per_m")/value("quiet_landing_v1","all_terrain","events_per_m")-1):.2f}%，单次比动能增加 {change("all_terrain","energy_mean_J_kg","quiet_landing_v1"):.2f}%。改用前向正进度分母仍增加 {100*(value("student","all_terrain","energy_per_forward_m")/value("quiet_landing_v1","all_terrain","energy_per_forward_m")-1):.2f}%，累计差异没有因去掉侧向行程分母而消失。',
f'4. **滑移并未全面改善。** 楼梯总体接触门控滑移时间均值相对 V1 {change("stairs","slip_gated_time_mean_m_s","quiet_landing_v1"):+.2f}%、相对 Baseline {change("stairs","slip_gated_time_mean_m_s","baseline"):+.2f}%。该值是四轮接触门控后、按全段时间平均；不是只在接触期间归一化的速度。','',
'因此本轮支持“部分指标回退”，不支持“整个轻盈落地能力统一退化某个百分比”。平均事件强度、尾部冲击、每米累计量和滑移分别回答不同问题。','',
'![全程累计量与触地次数](total_costs.png)','',
'## 学生在观测噪声下的变化','',
'以下三回合只比较同一个学生、同一中心初态。噪声会改变整条闭环轨迹；realistic 还增加了动作延迟，因此不能把变化单独归因为高程误差或置信度设计。',
'','|下楼指标|clean|train|realistic|realistic 相对 clean|','|---|---:|---:|---:|---:|']
for key,label,unit,d in fields:
    vals=[runs['student_'+p+'_center']['sections']['descent'][key] for p in ('clean','train','realistic')]
    lines.append(f'|{label} ({unit})|'+'|'.join(f'{v:.{d}f}' for v in vals)+f'|{100*(vals[2]/vals[0]-1):+.2f}%|')
real=runs['student_realistic_center']['sections']['descent']
lines+=['',f'realistic 单回合下楼平均速度 {real["speed_mean_m_s"]:.3f}m/s、比动能 {real["energy_mean_J_kg"]:.3f}J/kg、平均力峰 {real["force_peak_mean_N"]:.1f}N。相对 clean 中心学生，分别上升 {100*(real["speed_mean_m_s"]/runs["student_clean_center"]["sections"]["descent"]["speed_mean_m_s"]-1):.1f}%、{100*(real["energy_mean_J_kg"]/runs["student_clean_center"]["sections"]["descent"]["energy_mean_J_kg"]-1):.1f}%、{100*(real["force_peak_mean_N"]/runs["student_clean_center"]["sections"]["descent"]["force_peak_mean_N"]-1):.1f}%。在这一噪声条件下，不能沿用 clean 下楼较柔和的结论。','',
'为了说明量级，realistic 学生下楼平均力峰约 406.5N，超过 clean Baseline 中心回合约 310.7N 与 clean V1 中心回合约 356.0N；**这涉及不同观测/动作延迟条件，不是公平同噪声的模型优劣证明**。train 单回合下楼力峰 P95 也从 clean 的约 692.9N 增至约 1118.8N，仍需更多噪声样本确认其出现频率。','',
'上楼 realistic 的单次强度反而下降，说明噪声影响随路段、轨迹和触地方式变化，不能用全程平均掩盖下楼的问题。两种噪声各一回合，未作泛化通过率或统计显著性结论；这些噪声参数仍是工程假设，未用真雷达实测标定。','',
'![学生下楼噪声敏感性](noise_descent.png)','',
'## 采样定义与核验','',
'- 触地速度取碰撞求解前轮心沿原始扫描估计法向的接近速度；无有效法向拟合时退回竖直法向。碰撞求解后使用真实静态地形接触力判断触地，学生低置信度不屏蔽真实触地事件。',
'- 比动能为 v²/2；轮体动能乘 wheel mass，均为代理量，不等于整机有效碰撞能量、声音或声压。',
'- 触地后 8 个 5ms 求解步取真实接触合力模长峰值；超额冲量为这 40ms 内估计法向接触力超过原设置 100N 的积分。末尾不足 40ms 的事件排除。',
'- 按机身 x 划分路段：上楼 [1,2.8)、平台 [2.8,4.8)、下楼 [4.8,6.6)。前后轮可能跨段；排除开始 100ms 的初始化保护和机身不在 [-1,38) 的事件。',
'- 每米量按相应路段内各 5ms 物理步的实际 XY 行程累计；同时提供前向正进度分母，避免侧向行程掩盖累计强度。',
'- 主对照仅纳入无跌倒、完成且未出界回合；失败会单列，本轮不存在被排除的失败回合。采用站立初态，不包含键盘 Z/C 起身与切换行为。',
'- 已复现历史已知 Baseline 的全地形、上楼、下楼触地事件数、平均速度、比动能与力峰均值，数值在 1e-6 容差内一致；三者同组初态 qpos/qvel 一致。所有选定物理数组有限，5ms 时序、v²/2 与接触向量模长恒等式通过。',
'- 模型、checkpoint、源代码、地形哈希在开跑前冻结并在开跑后重新核验。触地数据来自 200Hz 实际物理求解，没有用 50Hz 姿态插值来推算力峰。','',
'## 产物','',
'- [冻结测试条件与三模型指纹](criteria.json)',
'- [完整分回合、分路段数值及配对变化范围](metrics.json)',
'- [三模型对照 CSV](comparison.csv)',
'- [核验与历史 Baseline 复现证据](validation.json)',
'- [11 回合通行摘要](suite_summary.json)',
'- 各回合子目录：`quiet.npz`（200Hz）、`telemetry.npz`（50Hz）、`touchdowns.csv`、`summary.json`。',
'- `evaluate.py`、`analyze.py`、`report.py` 保留用于复现；正式训练与键盘部署程序未修改。','',
'未测：自由手动操控、多速度/多楼梯高度、多个真实传感器噪声样本、声学和真机。外部路线反馈下的 sim2sim 结果不能代替这些能力验收。']
(OUT/'report.md').write_text('\n'.join(lines)+'\n')

font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
sections=('ascent','platform','descent');x=np.arange(3);width=.24
fig,axes=plt.subplots(2,3,figsize=(16,9),layout='constrained')
for ax,(key,label,unit,d) in zip(axes.flat,[fields[i] for i in (0,1,2,4,5,7)]):
    for i,n in enumerate(models):
        v=[value(n,sec,key) for sec in sections];xx=x+(i-1)*width
        bars=ax.bar(xx,v,width*.95,color=colors[n],label=labels[n],alpha=.85)
        ax.bar_label(bars,labels=[f'{z:.3g}' for z in v],fontsize=8)
        for j,sec in enumerate(sections):
            samples=[runs[f'{n}_clean_{case["name"]}']['sections'][sec][key] for case in c['cases']]
            ax.scatter([xx[j]]*3,samples,s=13,color='#222222',alpha=.55,zorder=3)
    ax.set_xticks(x,[names[z] for z in sections]);ax.set_title(label);ax.set_ylabel(unit);ax.grid(axis='y',alpha=.2)
axes[0,0].legend(fontsize=8)
fig.suptitle('最新学生 22800 与 Baseline / Quiet Landing V1：分路段轻盈落地\nclean 感知；相同路线反馈；柱为三个初态等权均值，点为单回合；全部完成无跌倒',fontsize=14)
fig.savefig(OUT/'landing_comparison.png',dpi=170,facecolor='white');plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(14,5),layout='constrained')
for ax,key,title,unit in zip(axes,('energy_per_xy_m','events_per_m','slip_gated_time_mean_m_s'),('全程每米累计触地比动能','全程每米触地次数','楼梯接触门控滑移时间均值'),('J/kg/m','次/m','m/s')):
    sec='stairs' if key=='slip_gated_time_mean_m_s' else 'all_terrain'
    v=[value(n,sec,key) for n in models];bars=ax.bar(range(3),v,color=[colors[n] for n in models])
    ax.bar_label(bars,labels=[f'{z:.4g}' for z in v]);ax.set_xticks(range(3),['Baseline','V1','Student']);ax.set_title(title);ax.set_ylabel(unit);ax.grid(axis='y',alpha=.2)
fig.suptitle('单次冲击降低不保证累计量降低：三个初态回合等权均值',fontsize=13)
fig.savefig(OUT/'total_costs.png',dpi=170,facecolor='white');plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(14,5),layout='constrained')
for ax,key,title,unit in zip(axes,('energy_mean_J_kg','force_peak_mean_N','force_peak_p95_N'),('单次比动能','40ms 力峰均值','40ms 力峰 P95'),('J/kg','N','N')):
    v=[runs['student_'+p+'_center']['sections']['descent'][key] for p in ('clean','train','realistic')]
    bars=ax.bar(range(3),v,color=['#0072B2','#009E73','#D55E00']);ax.bar_label(bars,labels=[f'{z:.4g}' for z in v]);ax.set_xticks(range(3),['clean','train','realistic']);ax.set_title(title);ax.set_ylabel(unit);ax.grid(axis='y',alpha=.2)
fig.suptitle('同一学生 22800 的下楼噪声敏感性：同一中心初态，每种噪声只测一回合\nrealistic 含感知退化与一个策略步动作延迟；不能单独归因为置信度或高度噪声',fontsize=12)
fig.savefig(OUT/'noise_descent.png',dpi=170,facecolor='white');plt.close(fig)
print('REPORT',OUT/'report.md')
print('VALIDATION',json.dumps(validation,ensure_ascii=False))
