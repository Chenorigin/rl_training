from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

OUT=Path(__file__).resolve().parent
OLD=Path('docs/quiet_landing_零命令倾斜与直线上下楼修复_20261011')
R=json.loads((OUT/'原部署分路段与驻足统计.json').read_text())
TB=json.loads((OUT/'TensorBoard分窗口统计.json').read_text())
TAGS=list(R);COLORS=['#5377a6','#469a83','#9865ab']
font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'font.size':10,
                     'axes.spines.top':False,'axes.spines.right':False})

def plot_bars(ax,key,title,unit,measure=None):
    names=['楼梯总体','上楼','楼顶平台','下楼'];x=np.arange(len(names))
    for i,(tag,color) in enumerate(zip(TAGS,COLORS)):
        values=[R[tag]['仅w']['sections'][name][key] for name in names]
        if measure:values=[v[measure] for v in values]
        bar=ax.bar(x+(i-1)*.25,values,.24,label=tag,color=color)
        labels=[f'{v:.0f}' if key=='force_peak_40ms_N' else f'{v:.3g}' for v in values]
        ax.bar_label(bar,labels=labels,fontsize=8)
    ax.set(xticks=x,xticklabels=names,title=title,ylabel=unit);ax.grid(alpha=.2,axis='y')

def yaw(q):
    w,x,y,z=q.T
    return np.rad2deg(np.unwrap(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))))

fig,axes=plt.subplots(2,4,figsize=(20,9),constrained_layout=True)
configs=[('normal_speed_m_s','触地前法向速度均值','米/秒','mean'),
         ('wheel_normal_energy_J','轮体法向动能代理均值','焦耳','mean'),
         ('force_peak_40ms_N','触地后40毫秒合力峰均值','牛','mean'),
         ('specific_energy_per_m_J_kg_m','每米累计触地比动能','焦耳/千克/米',None),
         ('events','有效触地事件数','次',None),
         ('force_peak_40ms_N','40毫秒合力峰P95','牛','p95'),
         ('force_peak_40ms_N','40毫秒合力峰最大值','牛','max'),
         ('duration_s','路段停留时间','秒',None)]
for ax,cfg in zip(axes.flat,configs):plot_bars(ax,*cfg)
axes[0,0].legend(fontsize=9)
fig.suptitle('V1.5 230300、233500与最新236200：原部署只按w的楼梯冲击对比\n六次回放均无摔倒；没有外部纠偏；均值同时结合触地次数与每米累计量解读',fontsize=15)
quiet_plot='V1_5最新236200与230300及233500_上下楼和平台触地速度轮体动能力峰及每米累计能量对比.png'
fig.savefig(OUT/quiet_plot,dpi=160);plt.close(fig)

fig,axes=plt.subplots(2,2,figsize=(14,10),constrained_layout=True)
for i,(tag,color) in enumerate(zip(TAGS,COLORS)):
    folder=OUT if tag==TAGS[-1] else OLD
    pose=np.load(folder/f'{tag}_仅w_姿态.npz');mask=(pose['time']>=8)&(pose['pos'][:,0]<=8)
    p=pose['pos'][mask];angle=yaw(pose['quat'])[mask]
    axes[0,0].plot(p[:,0],p[:,1],color=color,label=tag)
    axes[0,1].plot(p[:,0],angle,color=color,label=tag)
    zero=np.load(folder/f'{tag}_零命令_姿态.npz');mask=zero['time']>=6;start=np.flatnonzero(mask)[0]
    axes[1,0].plot(zero['time'][mask]-6,zero['pos'][mask,0]-zero['pos'][start,0],label=tag,color=color)
    force=R[tag]['零命令']['four_wheel_mean_normal_force_N']
    axes[1,1].bar(np.arange(4)+(i-1)*.25,force,.24,label=tag,color=color)
for ax in axes[0]:
    for lo,hi,name,c in [(1,2.8,'上楼','#d4e7f6'),(2.8,4.8,'平台','#e7e7e7'),(4.8,6.6,'下楼','#f5e4d4')]:
        ax.axvspan(lo,hi,color=c,alpha=.5);ax.text((lo+hi)/2,.98,name,transform=ax.get_xaxis_transform(),va='top',ha='center')
    ax.axhline(0,color='black',linestyle='--',linewidth=1);ax.set_xlabel('世界前向位置 x（米）')
axes[0,0].set(title='仅按w时的实际世界轨迹',ylabel='侧向偏移 y（米）')
axes[0,1].set(title='仅按w时的世界航向',ylabel='航向（度）')
axes[1,0].set(title='零命令按c后前向漂移',xlabel='策略运行时间（秒）',ylabel='前向位移（米）')
axes[1,1].set(title='零命令稳态四轮法向力（第10–36秒）',xticks=np.arange(4),xticklabels=['左前FL','右前FR','左后HL','右后HR'],ylabel='平均法向力（牛）')
for ax in axes.flat:ax.grid(alpha=.2,axis='y');ax.legend(fontsize=9)
fig.suptitle('V1.5最新236200与历史迭代：无外部纠偏的路径、驻足和四轮受力\n世界侧偏更大；没有把最大航向角单独当作直行能力',fontsize=15)
motion_plot='V1_5最新236200与230300及233500_仅按w世界直线路径偏差零命令漂移与四轮受力对比.png'
fig.savefig(OUT/motion_plot,dpi=160);plt.close(fig)

data=np.load(OUT/'V1_5当前全部标量.npz')
fig,axes=plt.subplots(2,3,figsize=(17,9),constrained_layout=True)
configs=[('QuietLanding/all/pre_speed_mean_m_s','训练触地速度均值','米/秒',1),
         ('QuietLanding/specific_energy_per_m_J_kg_m','训练每米累计触地比动能','焦耳/千克/米',1),
         ('QuietLanding/impact_window_excess_impulse_max_N_s','重置批次超额冲量最大值的均值','牛·秒',1),
         ('QuietLanding/impact_window_force_max_N','重置批次力峰最大值的均值','牛',1),
         ('QuietLoad/stable_support_fraction','稳定支撑有效时间覆盖率','%',100),
         ('QuietLoad/stable_diagonal_abs_mean','稳定支撑对角受力差绝对值','比例',1)]
for ax,(key,title,unit,scale) in zip(axes.flat,configs):
    a=data[key];x=[];y=[]
    for lo in range(214800,236200,500):
        v=a[(a[:,0]>=lo)&(a[:,0]<min(lo+500,236200)),2]
        if len(v):x.append((lo+250)/1000);y.append(float(v.mean())*scale)
    ax.plot(x,y,color=COLORS[-1]);ax.set(title=title,xlabel='训练迭代（千次）',ylabel=unit);ax.grid(alpha=.2)
    for step in [230300,233500,236200]:ax.axvline(step/1000,color='gray',linestyle=':',linewidth=.8)
fig.suptitle('V1.5截至236200的TensorBoard趋势（每500次更新块均值）\n训练包含失败与动态课程；力峰口径是批次最大值，不等于MuJoCo单次触地峰均值',fontsize=15)
tb_plot='V1_5截至236200_TensorBoard触地速度累计能量力峰超额冲量与受力平衡门控趋势.png'
fig.savefig(OUT/tb_plot,dpi=160);plt.close(fig)

def delta(a,b):return 100*(b/a-1)
a=R[TAGS[1]];b=R[TAGS[2]]
changes={}
for name in ['楼梯总体','上楼','楼顶平台','下楼']:
    sa=a['仅w']['sections'][name];sb=b['仅w']['sections'][name]
    changes[name]={key:delta(sa[key]['mean'],sb[key]['mean']) for key in ['normal_speed_m_s','specific_energy_J_kg','force_peak_40ms_N']}
    changes[name]['events_percent']=delta(sa['events'],sb['events'])
    changes[name]['energy_per_m_percent']=delta(sa['specific_energy_per_m_J_kg_m'],sb['specific_energy_per_m_J_kg_m'])
(OUT/'236200相对233500变化.json').write_text(json.dumps(changes,ensure_ascii=False,indent=2)+'\n')
s=b['仅w']['sections']['楼梯总体'];old=a['仅w']['sections']['楼梯总体'];down=changes['下楼']
lines=['# V1.5最新236200与230300、233500实际部署对比','',
       '本次最新保存checkpoint固定为236200，来自修改前的V1.5训练运行，不含新驻足及世界路线奖励。',
       '三个checkpoint均使用原interactive_rollout操作：z站起，第6秒c，第8秒开始w=机身[0.7,0,0]，至第36秒。另各有一条全程零速度命令回放。关闭窗口绘图和墙钟等待，使用真实MuJoCo物理、原PD、原Actor输入；没有外部路线反馈。',
       '两条最新回放本轮真实执行，四条历史回放在源码、checkpoint哈希及操作条件核验后复用。六条回放均无摔倒；三个移动回放都越过楼梯段终点，但36秒内未完成38米全地形。',
       '', '## 结论','',
       f'- 最新没有表现出一致优于历史模型的效果。相对233500，楼梯触地均值更硬，世界直行更差；同时触地次数减少，每米累计触地能量略降，不能用某一项单独概括整体噪声。',
       f'- 下楼触地速度均值变化{down["normal_speed_m_s"]:+.1f}%，比动能均值{down["specific_energy_J_kg"]:+.1f}%，40ms合力峰均值{down["force_peak_40ms_N"]:+.1f}%；每米累计比动能仅变化{down["energy_per_m_percent"]:+.1f}%。',
       f'- 楼梯总体触地次数{old["events"]}→{s["events"]}，每米累计比动能{old["specific_energy_per_m_J_kg_m"]:.3f}→{s["specific_energy_per_m_J_kg_m"]:.3f} J/kg/m。均值变大部分受到低强度小触地减少影响，不应忽略事件数。',
       f'- 最大世界侧偏{old["max_abs_world_y_m"]:.3f}→{s["max_abs_world_y_m"]:.3f}米，直行偏差明显变大。偏移方向改变，比较使用绝对值；航向最大值略低也不能抵消累计路径偏移。',
       '- 零命令漂移和对角偏载依旧存在。相对230300有改善，但相对233500未继续改善。',
       '- TensorBoard显示训练冲击代理量缓慢改善，不能直接证明这次0.7m/s原部署上下楼也改善。没有新奖励的运行不应期待自行解决刚定位的驻足和直行问题。',
       '- 对本次原部署直行与下楼单次冲击目标，233500比236200更值得保留作参考。230300仍是先前“带路线反馈、较低速度、完整全地形”评测的累计能量候选；本次条件不同，不能据此直接改写那个结果。',
       '',f'![冲击对比]({quiet_plot})','',f'![路径与驻足]({motion_plot})','',
       '## 零命令与直行','',
       '|模型|零命令30秒前向位移_m|航向变化_deg|稳态对角组力差占总力|楼梯最大侧偏_m|楼梯最大航向绝对值_deg|楼梯耗时_s|',
       '|---|---:|---:|---:|---:|---:|---:|']
for tag,r in R.items():
    z=r['零命令'];w=r['仅w']['sections']['楼梯总体']
    lines.append(f'|{tag}|{z["policy_forward_displacement_m"]:.3f}|{z["policy_yaw_change_deg"]:.2f}|{z["diagonal_ratio"]:.2%}|{w["max_abs_world_y_m"]:.3f}|{w["max_abs_world_yaw_deg"]:.2f}|{w["duration_s"]:.3f}|')
lines+=['','## 触地指标','',
        '|模型|路段|事件数|法向速度均值_m/s|比动能均值_J/kg|轮体法向动能均值_J|40ms合力峰均值_N|P95_N|最大值_N|每米累计比动能_J/kg/m|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for tag,r in R.items():
    for name,v in r['仅w']['sections'].items():
        values=[v['normal_speed_m_s']['mean'],v['specific_energy_J_kg']['mean'],v['wheel_normal_energy_J']['mean'],v['force_peak_40ms_N']['mean'],v['force_peak_40ms_N']['p95'],v['force_peak_40ms_N']['max'],v['specific_energy_per_m_J_kg_m']]
        lines.append(f'|{tag}|{name}|{v["events"]}|'+ '|'.join(f'{x:.4f}' for x in values)+'|')
lines+=['','## TensorBoard等长窗口','',
        f'日志共{TB["scalar_tags"]}个标量指标，冻结至{TB["max_log_iteration"]}；比较每个checkpoint前1000次更新[k−1000,k)，晚于236200的日志不用于它的窗口统计。训练包含失败与动态课程，未分上/下楼。',
        '', '|指标|230300前1000更新|233500前1000更新|236200前1000更新|最新相对233500|', '|---|---:|---:|---:|---:|']
for key in TB['statistics']['236200']:
    values=[TB['statistics'][str(step)][key]['mean'] for step in [230300,233500,236200]]
    lines.append('|'+key+'|'+'|'.join(f'{x:.6g}' for x in values)+f'|{delta(values[1],values[2]):+.2f}%|')
q=TB['statistics']['236200'];coverage=q['QuietLoad/stable_support_fraction']['mean'];D=q['QuietLoad/stable_diagonal_abs_mean']['mean'];ratio=abs(q['QuietReward/quiet_load_balance']['mean']/q['QuietReward/quiet_impact_peak']['mean'])
lines+=['',f'最新窗口稳定支撑有效时间覆盖率为{coverage:.4%}；该门控内对角偏差为{D:.3f}。均衡惩罚的日志绝对值仅约为冲击峰惩罚的{ratio:.4%}，覆盖和贡献仍很低。',
        '',f'![训练趋势]({tb_plot})','', '## 测量与限制','',
        '- 动态触地汇总从w开始后计算，排除站起和c切换瞬态；若回放摔倒，整条回放不纳入冲击及驻足平均。本轮无需排除，全部无摔倒。',
        '- 速度取真实碰撞求解前轮心沿地图估计法向的接近速度，specific_energy=speed²/2。轮体质量来自物理模型，乘比动能得到轮体法向动能代理，不是整机有效碰撞能量。',
        '- 接触力为静态地形实际合力模长，取触地起8个5ms物理步的峰值；末尾不足完整40ms窗口的事件排除。法向力用于受力平均，合力峰用于冲击，二者不是同一个量。',
        '- 路段由机身x切分，前后轮可能跨段。只在同一楼梯空间范围比较累计量，没有把三条不同长度的整段回放称为全地形比较。',
        '- 零命令受力取第10–36秒时间平均，包含零力；对角比值由各轮时间平均力计算，不等于训练稳定门控内的对角差。MuJoCo采样器默认门控不作为训练V1.5门控验收。',
        '- TensorBoard为每次日志记录值的等权均值，不是汇总所有触地后按事件数加权的物理均值。批次最大力峰均值与单次MuJoCo触地力峰均值不能直接横比。',
        '- 每个条件只一条确定性回放；没有多速度、多种初始化或扰动成功率统计，未量真实声音/分贝。后续不应把这次单回合变化当作确定的长期噪声改善或退化。',
        '- 本轮只分析，不修改奖励、模型、部署接口、训练配置或运行中的训练进程。']
(OUT/'V1_5最新236200相对230300和233500_原部署冲击直行驻足与训练趋势详细分析.md').write_text('\n'.join(lines)+'\n')
files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.suffix in ['.npz','.json','.png','.md','.csv','.py'] and p.name!='产物哈希清单.json'}
(OUT/'产物哈希清单.json').write_text(json.dumps(files,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(changes,ensure_ascii=False,indent=2))
