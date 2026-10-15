from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

OUT=Path(__file__).resolve().parent
C=json.loads((OUT/'本轮对照冻结条件.json').read_text())
R=json.loads((OUT/'步态与触地综合统计.json').read_text())
VERIFY=json.loads((OUT/'独立重算与资源核验.json').read_text())
TAGS=list(R);LABELS=['V1 257100','旧版V1.5 236200','新版V1.5 230900']
COLORS=['#5377a6','#9865ab','#d37b33']
font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'font.size':10,
                     'axes.spines.top':False,'axes.spines.right':False})

def yaw(q):
    w,x,y,z=q.T
    return np.unwrap(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)))

def bars(ax,key,title,unit,measure=None):
    names=['楼梯总体','上楼','楼顶平台','下楼'];x=np.arange(4)
    for i,(tag,color,label) in enumerate(zip(TAGS,COLORS,LABELS)):
        vals=[R[tag]['仅w']['sections'][n][key] for n in names]
        if measure:vals=[v[measure] for v in vals]
        b=ax.bar(x+(i-1)*.25,vals,.24,color=color,label=label)
        ax.bar_label(b,labels=[f'{v:.0f}' if key=='force_peak_40ms_N' else f'{v:.3g}' for v in vals],fontsize=8)
    ax.set(xticks=x,xticklabels=names,title=title,ylabel=unit);ax.grid(alpha=.2,axis='y')

assert VERIFY['six_fresh_physical_replays'] and VERIFY['all_no_fall']
assert all(R[tag]['仅w']['stairs_exited'] for tag in TAGS)
fig,axes=plt.subplots(2,4,figsize=(21,10),constrained_layout=True)
cfg=[('normal_speed_m_s','触地前法向速度均值','米/秒','mean'),
     ('wheel_normal_energy_J','轮体法向动能代理均值','焦耳','mean'),
     ('force_peak_40ms_N','触地后40毫秒合力峰均值','牛','mean'),
     ('specific_energy_per_m_J_kg_m','每米XY行程累计触地比动能','焦耳/千克/米',None),
     ('events','有效触地事件数','次',None),
     ('force_peak_40ms_N','40毫秒合力峰P95','牛','p95'),
     ('specific_energy_per_forward_m_J_kg_m','每米前向进度累计比动能','焦耳/千克/米',None),
     ('duration_s','路段停留时间','秒',None)]
for ax,v in zip(axes.flat,cfg):bars(ax,*v)
axes[0,0].legend(fontsize=9)
fig.suptitle('三个运行最新checkpoint：V1 257100、旧V1.5 236200、新V1.5 230900原部署触地对比\n六条新执行回放均未摔倒；w=0.7米/秒，无外部纠偏；不混入站起和切策略瞬态',fontsize=15)
quiet_plot='三个运行最新模型_上下楼平台触地法向速度轮体动能40毫秒力峰与累计能量详细对比.png'
fig.savefig(OUT/quiet_plot,dpi=150);plt.close(fig)

fig,axes=plt.subplots(2,3,figsize=(19,10),constrained_layout=True)
for i,(tag,color,label) in enumerate(zip(TAGS,COLORS,LABELS)):
    a=np.load(OUT/f'{tag}_仅w_姿态.npz');mask=(a['time']>=8)&(a['pos'][:,0]<=8)
    xy=a['pos'];q=yaw(a['quat']);j=np.flatnonzero(a['time']>=8)[0]
    axes[0,0].plot(xy[mask,0],xy[mask,1],color=color,label=label)
    axes[0,1].plot(xy[mask,0],np.rad2deg(q[mask]),color=color,label=label)
    cross=(xy[:,:2]-xy[j,:2])@np.array([-np.sin(q[j]),np.cos(q[j])])
    axes[0,2].plot(xy[mask,0],cross[mask],color=color,label=label)
    z=np.load(OUT/f'{tag}_零命令_姿态.npz');zm=z['time']>=6;k=np.flatnonzero(zm)[0]
    axes[1,0].plot(z['time'][zm]-6,z['pos'][zm,0]-z['pos'][k,0],color=color,label=label)
    axes[1,1].bar(np.arange(4)+(i-1)*.25,R[tag]['零命令']['four_wheel_mean_normal_force_N'],.24,color=color,label=label)
    axes[1,2].plot(z['time'][zm]-6,np.rad2deg(yaw(z['quat'])[zm]-yaw(z['quat'])[k]),color=color,label=label)
for ax in axes[0]:
    for lo,hi,name,c in [(1,2.8,'上楼','#d4e7f6'),(2.8,4.8,'平台','#e7e7e7'),(4.8,6.6,'下楼','#f5e4d4')]:
        ax.axvspan(lo,hi,color=c,alpha=.5);ax.text((lo+hi)/2,.98,name,transform=ax.get_xaxis_transform(),va='top',ha='center')
    ax.axhline(0,color='black',linestyle='--',linewidth=1);ax.set_xlabel('世界前向位置 x（米）')
axes[0,0].set(title='仅按w的实际世界路径',ylabel='世界侧偏 y（米）')
axes[0,1].set(title='仅按w的世界航向',ylabel='世界航向（度）')
axes[0,2].set(title='相对w开始位置/航向的路线偏差',ylabel='初始路线法向偏移（米）')
axes[1,0].set(title='零命令按c后前向漂移',xlabel='策略运行时间（秒）',ylabel='前向位移（米）')
axes[1,1].set(title='零命令稳态四轮法向力（10–36秒）',xticks=np.arange(4),xticklabels=['左前FL','右前FR','左后HL','右后HR'],ylabel='平均法向力（牛）')
axes[1,2].set(title='零命令按c后航向变化',xlabel='策略运行时间（秒）',ylabel='航向变化（度）')
for ax in axes.flat:ax.grid(alpha=.2,axis='y')
axes[1,0].legend(fontsize=9)
fig.suptitle('三个运行最新模型的步态起点比较：V1 257100对世界楼梯轴的路径偏差最小\n同时保留初始路线参考的结果；三个模型零命令漂移都未解决',fontsize=15)
motion_plot='三个运行最新模型_仅按w上下楼世界侧偏航向与初始路线误差零命令漂移及四轮受力对比.png'
fig.savefig(OUT/motion_plot,dpi=150);plt.close(fig)

fig,axes=plt.subplots(2,3,figsize=(18,9),constrained_layout=True)
wheel_colors=['#427bb4','#dd8b3d','#61a16d','#a36caa']
for col,(tag,label) in enumerate(zip(TAGS,LABELS)):
    d=np.load(OUT/f'{tag}_仅w_原部署回放_quiet.npz')
    for row,(lo,hi,name) in enumerate([(1,2.8,'上楼'),(4.8,6.6,'下楼')]):
        mask=(d['time']>=8)&(d['root_pos_w'][:,0]>=lo)&(d['root_pos_w'][:,0]<hi)
        t=d['time'][mask];fn=d['normal_force_N'][mask]
        for w,(leg,c) in enumerate(zip(['左前FL','右前FR','左后HL','右后HR'],wheel_colors)):
            axes[row,col].plot(t-t[0],fn[:,w],color=c,label=leg,linewidth=.8,alpha=.8)
        axes[row,col].set(title=f'{label}：{name}真实轮足法向力',xlabel='进入路段后时间（秒）',ylabel='法向力（牛）')
        axes[row,col].grid(alpha=.2)
        if row==0 and col==0:axes[row,col].legend(fontsize=9)
for row in range(2):
    ymax=max(axes[row,col].get_ylim()[1] for col in range(3))
    for col in range(3):axes[row,col].set_ylim(0,ymax)
fig.suptitle('三个运行最新模型上下楼四轮受力时序\n每行统一力轴；动态支撑不能要求四轮瞬时等力，平均对角偏载与时序共同查看',fontsize=15)
force_plot='三个运行最新模型_上楼下楼四轮真实法向接触力时序与动态支撑协调性对比.png'
fig.savefig(OUT/force_plot,dpi=150);plt.close(fig)

v1=R[TAGS[0]];old=R[TAGS[1]];new=R[TAGS[2]]
def pct(a,b):return 100*(b/a-1)
changes={}
for tag in TAGS[1:]:
    changes[tag]={}
    for name,s in R[tag]['仅w']['sections'].items():
        ref=v1['仅w']['sections'][name]
        changes[tag][name]={k:pct(ref[k]['mean'],s[k]['mean']) for k in ['normal_speed_m_s','specific_energy_J_kg','force_peak_40ms_N']}
        changes[tag][name].update({k:pct(ref[k],s[k]) for k in ['specific_energy_per_m_J_kg_m','specific_energy_per_forward_m_J_kg_m','max_abs_world_y_m']})
(OUT/'两个V1_5最新模型相对V1最新模型变化.json').write_text(json.dumps(changes,ensure_ascii=False,indent=2)+'\n')

lines=['# V1与V1.5两个运行的最新checkpoint：步态调整起点选择','',
       '## 建议','',
       '**这三个最新checkpoint中，建议选择V1 model_257100.pt作为步态调整起点。** 它在本次仅w、无外部纠偏的原部署上下楼中，世界楼梯轴侧偏、世界航向和楼梯动态平均对角偏载较小，同时保留了比较好的触地冲击水平。旧V1.5 236200的驻足漂移较小，但路线偏差和单次冲击较大；新V1.5 230900累计触地动能最低、静态偏载较小，但路线偏航与驻足漂移仍然较大。',
       '上一轮推荐旧233500是在历史模型集合中选择。本轮按用户要求，只比较三个运行各自最新模型，233500没有参加本轮新回放；不能把本次推荐解释为证明257100优于233500所有方面。',
       '', '推荐checkpoint：','',f'```text\n{C["checkpoints"][TAGS[0]]}\n```','',
       '若后续从V1恢复进入步态实验，保留当前满意的quiet_landing奖励约束，逐项加入/调整驻足和直行奖励；保持244D Actor观测、16D动作及固定PD。模型起点和继续训练时的奖励配置是两个选择，选V1模型不意味着回退柔和落地目标。此次只评测，未启动或改动训练。',
       '', '## 固定比较对象和操作条件','',
       f'冻结时刻：{C["freeze_time_CST"]}。训练继续产生的后续checkpoint不追入这次比较。',
       '', '|模型|运行开始时间|checkpoint|从baseline 214700新增更新|', '|---|---|---:|---:|']
for tag,label in zip(TAGS,LABELS):
    cp=Path(C['checkpoints'][tag]);iteration=int(cp.stem.split('_')[-1]);lines.append(f'|{label}|{cp.parent.name[:19]}|{iteration}|{iteration-214700}|')
lines+=['',
        '新版230900属于包含驻足与世界直线奖励的10月12日运行；旧236200属于修改前的10月10日运行。三者训练时长不相等，本轮用于选择已有模型起点，不用于断言哪套奖励算法更优。',
        '每个模型全新执行两条真实MuJoCo物理回放：all_terrain、z站起、第6秒c；移动条件第8秒开始仅w（机身[0.7,0,0]），驻足条件全程零命令，均到第36秒。使用生产interactive_rollout，只替换窗口绘图、键盘事件源和墙钟等待，不加外部路径/航向反馈。',
        '六条回放均未跌倒，三个移动回放均越过机身x=6.6米楼梯段。未完成38米all_terrain，因此比较共同楼梯空间段和驻足，不报告完整全地形成功率。',
        '', '## 驻足与直行','',
        '|模型|零命令30秒前移_m|最后10秒平移速度_m/s|稳态对角组力差|楼梯最大世界侧偏_m|楼梯最大世界航向_deg|楼梯耗时_s|',
        '|---|---:|---:|---:|---:|---:|---:|']
for tag,label in zip(TAGS,LABELS):
    z=R[tag]['零命令'];s=R[tag]['仅w']['sections']['楼梯总体']
    lines.append(f'|{label}|{z["policy_forward_displacement_m"]:.3f}|{z["last10_mean_translation_m_s"]:.4f}|{abs(z["diagonal_ratio"]):.2%}|{s["max_abs_world_y_m"]:.3f}|{s["max_abs_world_yaw_deg"]:.2f}|{s["duration_s"]:.3f}|')
lines+=['',
        '- V1最大世界侧偏0.248米，旧/新版V1.5为1.251/1.200米；V1最大世界航向13.82°，旧/新版为17.76°/26.71°。对于直线进入楼梯并沿世界楼梯轴登顶的目标，V1明显更接近。V1也还存在偏航，并未达到直行问题已经解决。',
        '- 零命令漂移三者均未解决；旧V1.5最小，新版V1.5最大。全部四轮在稳态都保持加载，漂移不是某轮长期悬空导致的简单现象。',
        '- 对角力差|FL+HR−FR−HL|/ΣF使用第10–36秒各轮法向力平均（含零力）。新版9.14%优于V1的10.97%和旧V1.5的11.22%，但受力偏置改善没有保证驻足变好。不能由对角力差单独推出漂移的原因。',
        '',f'![路径、驻足和四轮力]({motion_plot})','',
        '### 区分世界楼梯轴与W开始时的路线','',
        '|模型|W开始世界航向_deg|相对W开始位置/航向最大侧偏_m|相对W开始航向最大角度变化_deg|楼梯时间平均对角组力差|',
        '|---|---:|---:|---:|---:|']
for tag,label in zip(TAGS,LABELS):
    w=R[tag]['仅w'];s=w['sections']['楼梯总体']
    lines.append(f'|{label}|{w["W_start_heading_deg"]:.2f}|{s["max_cross_track_from_W_heading_m"]:.3f}|{s["max_heading_change_from_W_deg"]:.2f}|{s["time_mean_diagonal_ratio"]:.2%}|')
lines+=['',
        '旧V1.5相对W开始航向定义的参考线偏差略小于V1，不能说V1在每个直线指标上都最好。但用户要求沿楼梯轴直线进入/登顶时，旧版的世界侧偏和起点/终点路线更不合适，故仍以V1作为调整起点。未来评测也应保留两种参考线口径，不按其中较好的一项选择性报告。',
        '楼梯动态平均对角组力差：V1约1.17%，新版V1.5约2.21%，旧版约14.35%。这是整段时间平均，不能解释为每一瞬间四轮等力，也不能要求摆动轮与支撑轮等力。',
        '', '|模型|稳态平均横滚_deg|稳态平均俯仰_deg|稳态最大横滚绝对值_deg|稳态最大俯仰绝对值_deg|', '|---|---:|---:|---:|---:|']
for tag,label in zip(TAGS,LABELS):
    z=R[tag]['零命令'];lines.append('|'+label+'|'+'|'.join(f'{z[k]:.4f}' for k in ['steady_roll_mean_deg','steady_pitch_mean_deg','steady_roll_abs_max_deg','steady_pitch_abs_max_deg'])+'|')
lines+=['','本次稳态横滚/俯仰量都很小，不能把持续前移和偏航描述成大幅身体倾倒。按c后的瞬态可从完整姿态原始数组复查；上述稳态统计从第10秒起。',
        '',f'![上下楼四轮真实力时序]({force_plot})','',
        '## 柔和落地是否保留','',
        '|模型|路段|事件数|法向速度均值_m/s|比动能均值_J/kg|轮体法向动能均值_J|40ms合力峰均值_N|P95_N|每米XY比动能_J/kg/m|每米前向比动能_J/kg/m|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for tag,label in zip(TAGS,LABELS):
    for name,s in R[tag]['仅w']['sections'].items():
        vals=[s[k]['mean'] for k in ['normal_speed_m_s','specific_energy_J_kg','wheel_normal_energy_J','force_peak_40ms_N']]+[s['force_peak_40ms_N']['p95'],s['specific_energy_per_m_J_kg_m'],s['specific_energy_per_forward_m_J_kg_m']]
        lines.append(f'|{label}|{name}|{s["events"]}|'+ '|'.join(f'{v:.4f}' for v in vals)+'|')
lines+=['',
        '- V1上楼速度、单次力峰均值均低于两个V1.5最新模型。上楼比动能均值略高于新版V1.5，新版累计量也更低；V1并非每项冲击指标最优。',
        '- V1下楼速度0.610m/s、比动能0.362J/kg、合力峰358N；新版分别0.595、0.307、371，旧版0.728、0.432、429。V1下楼力峰均值/P95较低，新版下楼能量较低。',
        '- 楼梯总体每米XY累计比动能V1为9.020，旧版9.126，新版7.642J/kg/m。新版比V1低15.3%；改用前向进度分母后，新版8.032仍低于V1的9.171，说明其累计能量优势并非完全由绕行使分母变大造成。',
        '- 新版平台仅4次有效触地，单次平均速度、动能和力峰都高于V1，但累计量略低；少量事件的P95/均值不应过度解读。完整逐次CSV与事件数一并保留。',
        '',f'![触地冲击对比]({quiet_plot})','',
        '## 起点选择的取舍','',
        '用户已经满意当前柔和落地，下一阶段的主要问题是零命令驻足、直线进入/上下楼和支撑协调。在保留合理触地冲击的前提下，V1 257100的世界路线与动态对角负载更接近目标，因此本次三个最新模型中优先选它。新版V1.5 230900可保留作累计动能和静态受力平衡对照；旧236200可保留作零命令漂移较小的对照。',
        '这不是新增控制策略、不是删除V1.5已有奖励、也不是宣告V1整体优于V1.5。一次固定速度回放无法覆盖速度、地形初始化、扰动和不同PD下的表现，推荐是现有证据支持的训练起点。',
        '', '## 测量与核验','',
        '- 接触速度为碰撞求解前轮心沿地图估计表面法向的接近速度；比动能v²/2，轮体动能代理为轮体质量0.638kg乘比动能，不是整机有效碰撞能量或真实声音/分贝。',
        '- 接触力为真实静态地形接触合力向量；新触地起8个5ms物理步取合力模长峰值。末尾不足完整40ms的事件不纳入。力峰与法向平均支撑力不是同一量。',
        '- quiet.npz旧元数据energy_definition提到post_step平动速度；报告采用采样器实际的speed²/2法向触地前速度数组，已独立重算核验。',
        '- 只统计W开始后动态触地，排除站起及c切换瞬态。若跌倒，整条排除，失败单列；本次六条全部未摔倒，不存在跌倒瞬间的大力污染。',
        '- 路段按机身x划分，前后轮可能跨段。XY行程分母保留，同时新增前向进度分母，避免侧向绕行稀释每米累计指标。',
        '- 最新checkpoint/源码/物理模型/地形哈希、操作条件、保存的实际命令值、比动能与40ms力峰均值均核验通过。每条回放资源峰值实测见资源日志，原部署源码和训练保持不变。',
        '- 仅一个速度、每条件一条确定性轨迹；没有完整38米全地形成功率或真机声音测试，不据此作泛化或实机结论。',
        '', '|回放|峰值RSS_MiB|摔倒|', '|---|---:|---|']
for c in VERIFY['checks']:lines.append(f'|{c["model"]} {c["case"]}|{c["peak_RSS_KiB"]/1024:.1f}|{"是" if c["fell"] else "否"}|')
(OUT/'三个运行最新checkpoint_柔和落地驻足直行与步态调整起点选择报告.md').write_text('\n'.join(lines)+'\n')
files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.suffix in ['.npz','.json','.png','.md','.csv','.py'] and p.name!='产物哈希清单.json'}
(OUT/'产物哈希清单.json').write_text(json.dumps(files,ensure_ascii=False,indent=2)+'\n')
print('Three Chinese figures and report saved. Recommended start: '+C['checkpoints'][TAGS[0]])
