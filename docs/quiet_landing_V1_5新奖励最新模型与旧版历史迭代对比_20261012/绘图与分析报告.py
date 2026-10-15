"""Frozen CPU MuJoCo replay comparison; no changes to deployment or training."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

OUT=Path(__file__).resolve().parent
OLD=Path('docs/quiet_landing_V1_5最新236200与历史迭代原部署对比_20261012')
C=json.loads((OUT/'本轮对照冻结条件.json').read_text())
R=json.loads((OUT/'原部署分路段与驻足统计.json').read_text())
TB=json.loads((OUT/'TensorBoard分窗口统计.json').read_text())
OTB=json.loads((OLD/'TensorBoard分窗口统计.json').read_text())
TAGS=list(R);NEW=TAGS[-1];COLORS=['#5377a6','#469a83','#9865ab','#d37b33']
LABELS=['旧版230300','旧版233500','旧版236200','新版230200（新奖励）']
font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'font.size':10,
                     'axes.spines.top':False,'axes.spines.right':False})

def yaw(q):
    w,x,y,z=q.T
    return np.unwrap(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)))

checks=[];extra={}
for source,expected in C['source_sha256'].items():
    assert hashlib.sha256(Path(source).read_bytes()).hexdigest()==expected,source
for tag in TAGS:
    folder=Path(C['replay_folders'][tag]);extra[tag]={}
    cp=Path(C['checkpoints'][tag])
    digest=hashlib.sha256(cp.read_bytes()).hexdigest()
    assert digest==C['checkpoint_sha256'][tag]
    for case in ['零命令','仅w']:
        a=np.load(folder/f'{tag}_{case}_姿态.npz')
        d=np.load(folder/f'{tag}_{case}_原部署回放_quiet.npz')
        op=json.loads((folder/f'{tag}_{case}_操作条件.json').read_text())
        meta=json.loads(str(d['metadata']))
        assert op['checkpoint_sha256']==digest
        assert op['command_body']==([.7,0,0] if case=='仅w' else [0,0,0])
        assert op['course_correction'] is False and op['physics'] is True
        assert np.allclose(d['specific_energy'],.5*d['speed']**2,atol=1e-6)
        assert np.allclose(d['force_norm_N'],np.linalg.norm(d['force'],axis=-1),atol=1e-5,rtol=1e-5)
        checks.append({'model':tag,'case':case,'checkpoint_sha256':digest,
                       'physics':True,'external_correction':False,'fell':bool(meta['fell_in_policy']),
                       'event_count':int(d['event'].sum()),'samples':len(d['time'])})
        if meta['fell_in_policy']:continue
        if case=='零命令':
            i=np.flatnonzero(a['time']>=26)[0]
            extra[tag]['zero_last10_mean_translation_m_s']=float(np.linalg.norm(a['pos'][-1,:2]-a['pos'][i,:2])/(a['time'][-1]-a['time'][i]))
            continue
        i=np.flatnonzero(a['time']>=8)[0];q=yaw(a['quat']);xy=a['pos'][:,:2]
        e=(xy-xy[i])@np.array([-np.sin(q[i]),np.cos(q[i])])
        mask=(a['time']>=8)&(xy[:,0]>=1)&(xy[:,0]<6.6)
        extra[tag].update({'W_start_heading_deg':float(np.rad2deg(q[i])),
                          'max_cross_track_from_W_heading_m':float(np.abs(e[mask]).max()),
                          'max_heading_change_from_W_deg':float(np.rad2deg(np.abs(q[mask]-q[i]).max()))})
        # Recompute contact statistics independently from saved vectors and event masks.
        for lo,hi,name in [(1,6.6,'楼梯总体'),(1,2.8,'上楼'),(2.8,4.8,'楼顶平台'),(4.8,6.6,'下楼')]:
            events=np.argwhere(d['event']&(d['time']>=8)[:,None] & ((d['root_pos_w'][:,0]>=lo)&(d['root_pos_w'][:,0]<hi))[:,None])
            events=[(int(i),int(w)) for i,w in events if i+8<=len(d['time'])]
            stats=R[tag][case]['sections'][name]
            assert len(events)==stats['events']
            if not events:continue
            ee=[.5*float(d['speed'][i,w])**2 for i,w in events]
            ff=[float(np.linalg.norm(d['force'][i:i+8,w],axis=-1).max()) for i,w in events]
            assert np.isclose(np.mean(ee),stats['specific_energy_J_kg']['mean'])
            assert np.isclose(np.mean(ff),stats['force_peak_40ms_N']['mean'],rtol=1e-5)
(OUT/'独立重算与部署条件核验.json').write_text(json.dumps({'checks':checks,'extra_metrics':extra,'source_sha256_checked':C['source_sha256'],'result':'PASS','all_eight_replays_no_fall':all(not x['fell'] for x in checks),'full_all_terrain_completed':False},ensure_ascii=False,indent=2)+'\n')

def bars(ax,key,title,unit,measure=None):
    names=['楼梯总体','上楼','楼顶平台','下楼'];x=np.arange(4)
    for i,(tag,color,label) in enumerate(zip(TAGS,COLORS,LABELS)):
        vals=[R[tag]['仅w']['sections'][n][key] for n in names]
        if measure:vals=[v[measure] for v in vals]
        b=ax.bar(x+(i-1.5)*.2,vals,.19,color=color,label=label)
        ax.bar_label(b,labels=[f'{v:.0f}' if key=='force_peak_40ms_N' else f'{v:.3g}' for v in vals],fontsize=7)
    ax.set(xticks=x,xticklabels=names,title=title,ylabel=unit);ax.grid(alpha=.2,axis='y')

fig,axes=plt.subplots(2,4,figsize=(22,10),constrained_layout=True)
cfg=[('normal_speed_m_s','触地前法向速度均值','米/秒','mean'),
     ('wheel_normal_energy_J','轮体法向动能代理均值','焦耳','mean'),
     ('force_peak_40ms_N','触地后40毫秒合力峰均值','牛','mean'),
     ('specific_energy_per_m_J_kg_m','每米累计触地比动能','焦耳/千克/米',None),
     ('events','有效触地事件数','次',None),
     ('force_peak_40ms_N','40毫秒合力峰P95','牛','p95'),
     ('force_peak_40ms_N','force max','牛','max'),
     ('duration_s','路段停留时间','秒',None)]
cfg[6]=('force_peak_40ms_N','40毫秒合力峰最大值','牛','max')
for ax,v in zip(axes.flat,cfg):bars(ax,*v)
axes[0,0].legend(fontsize=9)
fig.suptitle('新版V1.5 230200与旧版230300、233500、236200：同条件原部署上下楼冲击\n均无摔倒，无外部纠偏，w=机身0.7米/秒；每次触地强度与每米累计量共同解读',fontsize=16)
quiet_plot='新版230200与旧版230300及233500及236200_上下楼平台触地速度轮体动能40毫秒力峰与每米累计能量对比.png'
fig.savefig(OUT/quiet_plot,dpi=150);plt.close(fig)

fig,axes=plt.subplots(2,3,figsize=(19,10),constrained_layout=True)
for i,(tag,color,label) in enumerate(zip(TAGS,COLORS,LABELS)):
    folder=Path(C['replay_folders'][tag]);a=np.load(folder/f'{tag}_仅w_姿态.npz')
    mask=(a['time']>=8)&(a['pos'][:,0]<=8);xy=a['pos'];q=yaw(a['quat']);j=np.flatnonzero(a['time']>=8)[0]
    axes[0,0].plot(xy[mask,0],xy[mask,1],color=color,label=label)
    axes[0,1].plot(xy[mask,0],np.rad2deg(q[mask]),color=color,label=label)
    e=(xy[:,:2]-xy[j,:2])@np.array([-np.sin(q[j]),np.cos(q[j])])
    axes[0,2].plot(xy[mask,0],e[mask],color=color,label=label)
    z=np.load(folder/f'{tag}_零命令_姿态.npz');zm=z['time']>=6;k=np.flatnonzero(zm)[0]
    axes[1,0].plot(z['time'][zm]-6,z['pos'][zm,0]-z['pos'][k,0],color=color,label=label)
    axes[1,1].bar(np.arange(4)+(i-1.5)*.2,R[tag]['零命令']['four_wheel_mean_normal_force_N'],.19,color=color,label=label)
    axes[1,2].plot(z['time'][zm]-6,np.rad2deg(yaw(z['quat'])[zm]-yaw(z['quat'])[k]),color=color,label=label)
for ax in axes[0]:
    for lo,hi,name,c in [(1,2.8,'上楼','#d4e7f6'),(2.8,4.8,'平台','#e7e7e7'),(4.8,6.6,'下楼','#f5e4d4')]:
        ax.axvspan(lo,hi,color=c,alpha=.5);ax.text((lo+hi)/2,.98,name,transform=ax.get_xaxis_transform(),va='top',ha='center')
    ax.axhline(0,color='black',linestyle='--',linewidth=1);ax.set_xlabel('世界前向位置 x（米）')
axes[0,0].set(title='仅按w的世界轨迹',ylabel='世界侧移 y（米）')
axes[0,1].set(title='仅按w的世界航向',ylabel='航向（度）')
axes[0,2].set(title='相对w开始时位置/航向的直线偏差',ylabel='沿初始路线法向的偏移（米）')
axes[1,0].set(title='零命令按c后前向漂移',xlabel='策略运行时间（秒）',ylabel='前向位移（米）')
axes[1,1].set(title='零命令稳态四轮法向力（第10–36秒）',xticks=np.arange(4),xticklabels=['左前FL','右前FR','左后HL','右后HR'],ylabel='平均法向力（牛）')
axes[1,2].set(title='零命令按c后航向变化',xlabel='策略运行时间（秒）',ylabel='航向变化（度）')
for ax in axes.flat:ax.grid(alpha=.2,axis='y')
axes[1,0].legend(fontsize=9)
fig.suptitle('新版V1.5 230200与旧版历史模型：世界路径、初始路线偏差、驻足漂移和对角受力\n新版受力偏置减小，但驻足和直行尚未同时改善',fontsize=16)
motion_plot='新版230200与旧版历史模型_世界直行路径和初始航向路线偏差零命令漂移及四轮受力详细对比.png'
fig.savefig(OUT/motion_plot,dpi=150);plt.close(fig)

data=np.load(OUT/'V1_5当前全部标量.npz');odata=np.load(OLD/'V1_5当前全部标量.npz')
fig,axes=plt.subplots(2,3,figsize=(18,9),constrained_layout=True)
cfg=[('QuietLanding/specific_energy_per_m_J_kg_m','每米累计触地比动能','焦耳/千克/米'),
     ('QuietLanding/impact_window_force_max_N','重置批次力峰最大值均值','牛'),
     ('QuietStand/diagonal_abs_mean','平地零命令对角受力差','比例'),
     ('QuietStand/mean_xy_speed_m_s','平地零命令实际平移速度','米/秒'),
     ('QuietRoute/cross_track_abs_mean_m','直行段累计侧偏','米'),
     ('QuietRoute/heading_error_abs_mean_deg','直行段航向误差','度')]
for ax,(key,title,unit) in zip(axes.flat,cfg):
    sources=[(data,230200,COLORS[-1],'新版')]
    if key in odata:sources.insert(0,(odata,236200,COLORS[2],'旧版'))
    for ds,end,color,label in sources:
        a=ds[key];x=[];y=[]
        for lo in range(214800,end,500):
            v=a[(a[:,0]>=lo)&(a[:,0]<min(lo+500,end)),2]
            if len(v):x.append((lo+min(lo+500,end))/2000);y.append(float(v.mean()))
        ax.plot(x,y,color=color,label=label)
    ax.set(title=title,xlabel='训练迭代（千次）',ylabel=unit);ax.grid(alpha=.2);ax.legend()
fig.suptitle('新版V1.5截至230200：TensorBoard每500次更新均值\n训练奖励与命令采样已改变，跨运行趋势仅供参考；训练含失败与动态课程',fontsize=15)
tb_plot='新版V1_5截至230200_TensorBoard累计触地动能力峰驻足受力平衡速度与世界路径误差趋势.png'
fig.savefig(OUT/tb_plot,dpi=160);plt.close(fig)

def pct(old,new):return (new/old-1)*100
changes={}
for tag in TAGS[:-1]:
    changes[tag]={}
    for name in R[NEW]['仅w']['sections']:
        a=R[tag]['仅w']['sections'][name];b=R[NEW]['仅w']['sections'][name]
        v={k:pct(a[k]['mean'],b[k]['mean']) for k in ['normal_speed_m_s','specific_energy_J_kg','force_peak_40ms_N']}
        v.update({k:pct(a[k],b[k]) for k in ['events','specific_energy_per_m_J_kg_m','max_abs_world_y_m']})
        changes[tag][name]=v
(OUT/'新版230200相对各旧版变化百分比.json').write_text(json.dumps(changes,ensure_ascii=False,indent=2)+'\n')

lines=['# 新版V1.5最新230200与旧版历史迭代模型效果分析','',
       '## 当前最新模型的身份','',
       '本轮固定分析10月12日00:52:08新启动训练的model_230200.pt，含驻足运动/姿态/载荷和世界直线路径奖励，以及世界Z轴角速度跟踪。旧运行236200虽然编号更大，但属于修改前的奖励版本。两条运行都从baseline 214700恢复；新版230200新增15500次更新，旧230300新增15600次，训练预算最接近。评测过程中训练继续，本文不会追逐后续新增checkpoint。',
       f'新运行目录：`{Path(C["checkpoints"][NEW]).parent}`。实际保存的env配置及TensorBoard的新QuietStand/QuietRoute标签均确认新奖励接线。',
       '', '## 结论','',
       '- 新版230200没有同时解决驻足、直行和柔和下楼三个目标，不建议仅凭编号或训练总奖励将它判为优于旧版。',
       '- 平地零命令下对角组平均力差改善，但前移没有改善：旧230300/233500/236200为1.069/0.747/0.781米，新版1.097米。新版最后10秒仍有约0.035米/秒平均平移，漂移并非仅发生在按c的切换瞬间。',
       '- 世界楼梯轴侧偏：新版上楼最大0.233米，较旧233500的0.326米减小；但楼顶0.543米、下楼0.951米，较旧233500的0.299/0.669米增加。新版比旧236200的1.251米总体侧偏小，但没有超过旧230300和233500。',
       '- 下楼每次触地合力峰均值降低，触地速度/动能和每米累计动能没有同步降低。不能据此认定下楼噪声已经变小。楼顶平台的速度、动能和力峰也有退步。',
       '- 上楼每米累计比动能为14.700 J/kg/m，优于三个旧模型；单次力峰均值605N，仍高于旧230300/233500的579/565N。降低累计量和降低每次冲击是不同的结果。',
       '- 新版训练中对角受力差下降、累计触地动能下降；驻足速度和世界路线误差尚无持续改善证据。当前更适合作为继续调试的候选，而非已验证的替代模型。',
       '',f'![冲击对比]({quiet_plot})','',f'![路径与驻足]({motion_plot})','',
       '## 同条件MuJoCo实测','',
       '真实all_terrain地形、原interactive_rollout、固定PD和原244D Actor输入/16D动作；z站起，第6秒c，第8秒开始机身命令[0.7,0,0]，36秒结束。另各有全程零速度回放。窗口绘图、键盘事件源和墙钟等待被无界面替代，物理、策略、控制器与观测计算保持原实现；没有外部航向或侧偏反馈。',
       '共8条有效确定性回放均无摔倒，4条移动回放均越过楼梯段机身x=6.6米；36秒内未完成38米全地形，因此本文只比较共同楼梯空间路段和零命令驻足，不报告全地形成功率。旧230300/233500回放按checkpoint/源码/条件哈希核验后复用；旧236200与新版230200在本轮实际回放。',
       '', '|模型|零命令30秒前移_m|零命令航向变化_deg|稳态对角力差|楼梯最大世界侧偏_m|最大世界航向_deg|相对W开始路线侧偏_m|楼梯耗时_s|',
       '|---|---:|---:|---:|---:|---:|---:|---:|']
for tag,label in zip(TAGS,LABELS):
    z=R[tag]['零命令'];s=R[tag]['仅w']['sections']['楼梯总体'];ex=extra[tag]
    lines.append(f'|{label}|{z["policy_forward_displacement_m"]:.3f}|{z["policy_yaw_change_deg"]:.2f}|{abs(z["diagonal_ratio"]):.2%}|{s["max_abs_world_y_m"]:.3f}|{s["max_abs_world_yaw_deg"]:.2f}|{ex["max_cross_track_from_W_heading_m"]:.3f}|{s["duration_s"]:.3f}|')
lines+=['','对角力差定义为|FL+HR−FR−HL|/ΣF，F取第10–36秒各轮法向力时间平均，包含零力。这仅反映两组对角轮的偏载，不能单独证明四轮均载、俯仰或横滚已改善。',
        '上表同时给世界楼梯轴和W开始位置/航向所定义直线的偏差，避免把最大世界航向略小误解为走得更直。各模型W开始前已偏航约4.9–5.9°；新版相对初始路线侧偏1.720米，也没有体现直行改善。初始路线侧偏与训练中的长期段参考线不完全同口径。',
        '', '|模型|路段|事件数|法向速度均值_m/s|比动能均值_J/kg|轮体动能均值_J|40ms合力峰均值_N|P95_N|最大值_N|每米累计比动能_J/kg/m|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for tag,label in zip(TAGS,LABELS):
    for name,s in R[tag]['仅w']['sections'].items():
        values=[s[k]['mean'] for k in ['normal_speed_m_s','specific_energy_J_kg','wheel_normal_energy_J','force_peak_40ms_N']]+[s['force_peak_40ms_N']['p95'],s['force_peak_40ms_N']['max'],s['specific_energy_per_m_J_kg_m']]
        lines.append(f'|{label}|{name}|{s["events"]}|'+ '|'.join(f'{v:.4f}' for v in values)+'|')
lines+=['','### 新版相对旧233500','', '|路段|触地速度均值|比动能均值|力峰均值|每米累计比动能|事件数|', '|---|---:|---:|---:|---:|---:|']
for name,s in changes[TAGS[1]].items():
    lines.append('|'+name+'|'+'|'.join(f'{s[k]:+.1f}%' for k in ['normal_speed_m_s','specific_energy_J_kg','force_peak_40ms_N','specific_energy_per_m_J_kg_m','events'])+'|')
lines+=['','下楼新版76次触地，旧230300/233500/236200分别71/73/47次。新版相对旧233500：速度+10.3%，比动能+33.9%，力峰均值−4.3%，每米累计比动能+40.8%。相对旧230300这个更接近训练预算的参考：速度+1.9%，比动能+4.3%，力峰均值−11.1%，每米累计比动能+11.2%。力峰P95相对旧233500也增加约3.6%，均值降低没有保证尾部冲击降低。',
        '', '## TensorBoard：趋势与运行间比较','',
        f'新日志冻结前缀截至迭代{TB["max_log_iteration"]}，共{TB["scalar_tags"]}个标量标签。新checkpoint窗口[229200,230200)，早期窗口[214800,215800)，各1000次更新。旧窗口也取各checkpoint前1000次更新。训练包含失败、动态地形课程及不同命令采样，跨运行数值不可视为固定测试集。',
        '', '|指标|旧230300|旧233500|旧236200|新版230200|新版早期215800|新版早期→当前|', '|---|---:|---:|---:|---:|---:|---:|']
for key,cur in TB['statistics']['230200'].items():
    vals=[OTB['statistics'][str(k)].get(key,{}).get('mean') for k in [230300,233500,236200]]
    early=TB['statistics']['215800'][key]['mean'];current=cur['mean']
    lines.append('|'+key+'|'+'|'.join('—' if v is None else f'{v:.6g}' for v in vals)+f'|{current:.6g}|{early:.6g}|{pct(early,current):+.2f}%|')
lines+=['',
        '- 新版早期→当前：触地速度0.447→0.410m/s（−8.4%），触地比动能0.340→0.277J/kg（−18.4%），每米累计比动能6.116→4.342（−29.0%）。训练有学习效果，但不能替代本次原部署下楼结果。',
        '- 平地零命令门控内对角差0.221→0.151（−31.5%），平移速度0.171→0.165m/s（仅−3.1%）。零命令平地覆盖率约6.7%，该门控内四轮加载覆盖率约90.8%；不能用旧稳定支撑门控的低覆盖率代替新增驻足奖励的覆盖率。',
        '- 路线侧偏0.325→0.344m（+6.0%），航向误差5.00→5.25°（+5.2%）。直行门控覆盖率约33.6%。在训练地形/命令分布变化下，这不能独立证明奖励无效，但也没有改善证据。',
        '- 最新1000次更新里，QuietReward/quiet_idle_motion、quiet_idle_pose、quiet_idle_load、quiet_straight_route均值分别−0.04850、−0.07101、−0.05181、−0.33875，均非零。新增奖励确实在训练中被计算，驻足/路线效果不足不能归因于仍在跑旧代码。',
        '- 新版重置批次超额冲量最大值均值35.50→38.50N·s（+8.5%），未随触地速度一同降低；速度跟踪误差0.354→0.381（+7.6%）。任务性能和冲击尾部仍需一起观察。',
        '- 新版每米累计比动能4.342低于旧230300/233500/236200训练窗口，但触地速度和单次比动能略高；不同命令采样、触地事件数与课程会改变均值，不能用累计量单独宣告全面优于旧版。',
        '',f'![训练趋势]({tb_plot})','',
        '## 下一步判断','',
        '保留新版230200作为新奖励阶段的参考，旧233500作为原部署下楼单次冲击/驻足参考，旧230300作为近似相同训练预算参考。当前新版尚未形成一致改进，不应删除旧候选或直接将它作为默认部署。建议后续首先验证零命令实际速度为何几乎不降，以及平地门控、奖励归一化和命令采样是否足以压制持续轮滚；直行需要按上楼/平台/下楼拆分训练诊断，避免总航向误差掩盖平台和下楼的累计侧偏。',
        '奖励已开始改善受力偏置和训练累计触地能量，可以继续做有界训练与同条件检查，但不能预期仅增加迭代就必然解决驻足和下楼。若多个后续检查点仍重复上述退步，再调整奖励，而不是扩大步态修改范围。本文只分析，未改配置或中断训练。',
        '', '## 测量定义与限制','',
        '- 速度取轮心在接触求解前沿地图估计表面法向的接近速度；比动能e=v²/2，轮体法向动能代理E=0.638e焦耳（轮体质量来自MuJoCo）。不是整机有效碰撞能量，也不是实测声音/分贝。速度含躯干与腿运动，不等于轮胎表面旋转速度。',
        '- 合力来自MuJoCo真实地形接触力向量；每次有效新触地取接下来8个5ms物理步的合力模长最大值。不是把整段平均支撑力称为冲击力。接触末尾不足40ms的事件不纳入。力峰和能量已从原始数组独立重算通过。',
        '- quiet.npz元数据中的energy_definition沿用了旧post_step_center_speed_squared描述；本次实际数组specific_energy已核验等于speed²/2，报告采用采样器真实计算的触地前法向速度定义，不按旧描述解释。',
        '- 动态触地从W开始后计算，排除站起和切策略瞬态；若回放跌倒，整条不纳入触地/驻足统计。本轮有效8条均无跌倒，不存在跌倒尖峰污染平均。',
        '- 上楼/平台/下楼按机身世界x范围1–2.8、2.8–4.8、4.8–6.6米归类，前后轮可能跨段。每米累计量分母是该空间范围内的XY实际行程；侧偏会使分母变大，不是楼梯设计长度。比较时同时呈现路径、耗时及事件数。',
        '- TensorBoard统计按日志记录等权平均，不是全触地事件加权；力峰/超额冲量指标是重置批次最大值的均值，不能与MuJoCo单次触地力峰均值直接横比。',
        '- 每个模型/条件一条确定性轨迹，没有不同种子、速度、载荷或扰动统计。全部无摔倒只能说明本次8条回放，不能当作泛化成功率。',
        '- 以前带外部路线纠偏、0.5米/秒的38米全地形结果不与本次仅W=0.7米/秒的原部署数据混用。']
(OUT/'新版V1_5最新230200与旧版230300及233500及236200_模型效果详细分析.md').write_text('\n'.join(lines)+'\n')
files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.suffix in ['.npz','.json','.png','.md','.csv','.py'] and p.name!='产物哈希清单.json' and '_W_' not in p.name}
(OUT/'产物哈希清单.json').write_text(json.dumps(files,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(changes,ensure_ascii=False,indent=2))
