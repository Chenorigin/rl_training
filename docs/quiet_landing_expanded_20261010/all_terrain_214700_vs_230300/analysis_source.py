import json, hashlib, csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

OUT=Path('/home/ubuntu/cyq_shixi_projects/rl_training/docs/quiet_landing_expanded_20261010/all_terrain_214700_vs_230300')
contract=json.loads((OUT/'evaluation_contract.json').read_text())
for k,p in contract['paths'].items():
    assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==contract['sha256'][k],k
legs=['fl','fr','hl','hr']
data={}; events={}; summaries={}
def stats(rows):
    r={'count':len(rows)}
    for key in ['speed','specific_energy','wheel_normal_energy','pre_center_speed','pre_center_energy','force_at_event','force_peak_40ms']:
        v=np.array([e[key] for e in rows])
        r[key]={s:float(f(v)) if len(v) else None for s,f in [('mean',np.mean),('p50',np.median),('p95',lambda v:np.quantile(v,.95)),('max',np.max)]}
    r['normal_fit_valid_fraction']=float(np.mean([e['normal_fit_valid'] for e in rows])) if rows else None
    return r
for tag in ['baseline','quiet_v1']:
    d=np.load(OUT/(tag+'_quiet.npz')); data[tag]=d
    m=json.loads(str(d['metadata']));t=d['time'];pos=d['root_pos_w'];mass=np.array(m['wheel_masses_kg'])
    for k in ['time','root_pos_w','speed','specific_energy','pre_velocity','force','force_norm_N']:
        assert np.isfinite(d[k]).all(),(tag,k)
    assert np.allclose(np.diff(t),.005,atol=1e-8)
    assert np.allclose(d['specific_energy'],.5*d['speed']**2,atol=1e-6)
    assert np.allclose(d['force_norm_N'],np.linalg.norm(d['force'],axis=-1),rtol=1e-5,atol=1e-5)
    end=t[-1]-.02 if m['fell'] else t[-1]
    domain=(t>=.1)&(t<end)&(pos[:,0]>=-1)&(pos[:,0]<38)
    rows=[]; discarded=0
    for i,w in np.argwhere(d['event'] & domain[:,None]):
        # Eight solved 5ms intervals beginning with the event: [i,i+8).
        if i+8>len(t) or t[i+7]>=end:
            discarded+=1;continue
        x=float(pos[i,0]);section=next(name for a,b,name in contract['sections'] if a<=x<b)
        v=float(d['speed'][i,w]);vs=float(np.linalg.norm(d['pre_velocity'][i,w]));force=d['force_norm_N'][i:i+8,w]
        rows.append(dict(time=float(t[i]),x=x,y=float(pos[i,1]),wheel=legs[w],section=section,speed=v,specific_energy=.5*v*v,wheel_normal_energy=.5*mass[w]*v*v,pre_center_speed=vs,pre_center_energy=.5*mass[w]*vs*vs,force_at_event=float(force[0]),force_peak_40ms=float(force.max()),normal_fit_valid=bool(d['normal_fit_valid'][i,w])))
    events[tag]=rows
    with (OUT/(tag+'_touchdowns.csv')).open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    sections={name:stats([r for r in rows if r['section']==name]) for a,b,name in contract['sections']}
    summaries[tag]={'run':{k:m[k] for k in ['steps_executed','max_x','final_x','fell','crossed_finish','inside_course','cleared_terrain']},'duration_s':float(t[-1]),'max_abs_root_y_m':float(abs(pos[:,1]).max()),'goal_crossing_time_s':float(t[np.flatnonzero(pos[:,0]>=38)[0]]) if np.any(pos[:,0]>=38) else None,'discarded_incomplete_windows':discarded,'whole_covered_domain':stats(rows),'sections':sections,'stairs':stats([r for r in rows if 1<=r['x']<6.6]),'stairs_wheels':{w:stats([r for r in rows if 1<=r['x']<6.6 and r['wheel']==w]) for w in legs},'common_complete_domain':stats([r for r in rows if r['x']<11])}
    for label,a,b in [('stairs',1,6.6),('common_complete_domain',-1,11)]:
        idx=np.flatnonzero(domain&(pos[:,0]>=a)&(pos[:,0]<b))
        dist=float(np.linalg.norm(np.diff(pos[idx,:2],axis=0),axis=1).sum())
        q=summaries[tag][label];q['xy_distance_m']=dist;q['events_per_m']=q['count']/dist;q['specific_energy_sum_per_m_J_kg_m']=sum(r['specific_energy'] for r in rows if a<=r['x']<b)/dist
result={'contract':'evaluation_contract.json','method':'200Hz physics; estimated height-map surface normal; wheel COM pre-solver velocity; new tread contact events; force vector resultant; peak of 8 solved 5ms samples from event (40ms); final 20ms of failed run excluded; segments assigned by root x; primary complete common extent x<11; no external sound measurement','summaries':summaries}
(OUT/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family']=font.get_name()
colors={'baseline':'#2e6fbb','quiet_v1':'#e67e22'}
fig,ax=plt.subplots(2,3,figsize=(16,9),constrained_layout=True)
for tag,d in data.items():
    mask=d['root_pos_w'][:,0]<38.5
    ax[0,0].plot(d['time'][mask],d['root_pos_w'][mask,0],label=tag,color=colors[tag])
    ax[0,1].plot(d['root_pos_w'][mask,0],d['root_pos_w'][mask,1],label=tag,color=colors[tag])
    rows=[r for r in events[tag] if 1<=r['x']<6.6]
    ax[0,2].scatter([r['x'] for r in rows],[r['speed'] for r in rows],s=10,alpha=.55,label=tag,color=colors[tag])
ax[0,0].set(xlabel='Time (s)',ylabel='Root x (m)',title='Task progress (cropped at finish)');ax[0,0].axhline(38,color='gray',ls=':')
ax[0,1].set(xlabel='Root x (m)',ylabel='Root y (m)',title='Lateral drift');ax[0,1].axhline(-2,color='red',ls=':');ax[0,1].axhline(2,color='red',ls=':')
ax[0,2].set(xlabel='Root x at touchdown (m)',ylabel='Pre normal closing speed (m/s)',title='Stair touchdown events')
names=['上楼','楼顶平台','下楼'];xs=np.arange(3)
for axis,key,title,unit in [(ax[1,0],'speed','Mean touchdown closing speed','m/s'),(ax[1,1],'specific_energy','Mean normal specific energy','J/kg'),(ax[1,2],'force_peak_40ms','Mean 40ms force peak','N')]:
    for offset,tag in [(-.18,'baseline'),(.18,'quiet_v1')]:
        vals=[summaries[tag]['sections'][n][key]['mean'] for n in names]
        bars=axis.bar(xs+offset,vals,.36,label=tag,color=colors[tag])
        axis.bar_label(bars,fmt='%.3g',fontsize=9)
    axis.set_xticks(xs,names);axis.set(title=title,ylabel=unit)
for a in ax.flat:a.grid(alpha=.2);a.legend(fontsize=8)
fig.savefig(OUT/'comparison.png',dpi=160);fig.savefig(OUT/'comparison.pdf');plt.close(fig)
lines=['# Baseline 214700 与 Quiet V1 230300：all_terrain 实测','', '## 运行与口径','', '- 同一 M20 MJCF、all_terrain.xml、出生点 x=-1/y=0/yaw=0、前进指令 0.5 m/s；50Hz 策略、200Hz 物理，最多120秒，首次跌倒即停止。','- checkpoint/模型/地形/采样代码 SHA256 及判据在只读 evaluation_contract.json；本次两份运行均已完成。','- 法向触地速度 v=max(0,-v_pre·n)，n 来自高程图拟合；轮体中心碰撞前速度，不包含轮胎旋转接触点速度。','- 法向比动能 e=v²/2（J/kg）；轮体质量乘以 e 仅是轮体动能代理，不是整条腿有效碰撞动能。另有碰撞前轮心总平动速度/动能，保存于 CSV/JSON。','- 力：轮足与静态地形接触合力模长，逐次事件取开始后8个5ms物理步（40ms）的峰值；也保留首步力。','- 初始100ms不计；失败运行剔除末20ms及不完整力窗口；位置按机身x归段，两者共同完整路段截至 x=11。滑滚持续支撑不重复算新触地。','', '## 任务覆盖','']
for tag,s in summaries.items():
    r=s['run'];lines.append(f"- {tag}：运行 {s['duration_s']:.2f}s，最大机身x={r['max_x']:.3f}m，跌倒={r['fell']}，终点线越过={r['crossed_finish']}，末帧所有轮心均在赛道内={r['inside_course']}，原入口 cleared_terrain={r['cleared_terrain']}。")
lines+=['','Baseline在坡顶附近偏出路线后触发跌倒；V1在约84秒越过x=38且120秒未跌倒，但末帧轮心越界，不能称为严格全地形成功通过；也未逐障碍验证绕行。后半程只有V1数据，不作跨模型比较。两者楼梯段均完整覆盖，但轨迹和触地次数不同。','', '## 按机身位置划分的共同路段','', '| 路段 | 模型 | 新触地数 | 法向速度均值/P95 (m/s) | 比动能均值 (J/kg) | 40ms力峰均值/P95 (N) |','|---|---|---:|---:|---:|---:|']
for label in ['上楼','楼顶平台','下楼','上坡']:
    for tag,s in summaries.items():
        q=s['sections'][label];v=q['speed'];f=q['force_peak_40ms']
        lines.append(f"| {label} | {tag} | {q['count']} | {v['mean']:.3f} / {v['p95']:.3f} | {q['specific_energy']['mean']:.4f} | {f['mean']:.1f} / {f['p95']:.1f} |")
lines+=['','## 整个楼梯段 x∈[1,6.6) 与共同完整域 x∈[-1,11)','']
for label in ['stairs','common_complete_domain']:
    lines+=['',f'### {label}','', '| 模型 | 事件数 | 平均法向速度 | 平均比动能 | 平均40ms力峰 | P95力峰 | 比动能累计/实际XY路程 |','|---|---:|---:|---:|---:|---:|---:|']
    for tag,s in summaries.items():
        q=s[label];lines.append(f"| {tag} | {q['count']} | {q['speed']['mean']:.4f} m/s | {q['specific_energy']['mean']:.4f} J/kg | {q['force_peak_40ms']['mean']:.1f} N | {q['force_peak_40ms']['p95']:.1f} N | {q['specific_energy_sum_per_m_J_kg_m']:.4f} J/kg/m |")
lines+=['','## 限制与产物','', '- 每模型一次确定性固定指令试验，未进行多种初态/多指令统计；不是显著性结论，也不是实测声音或分贝结论。','- 法向是高程图估计；台阶边缘拟合可能退化，JSON包含每段 normal_fit_valid_fraction。台阶归段以机身位置，前后轮跨界时不是精确轮下地形标签。','- 楼梯末端接触更少/更多会改变按事件均值，因此同时给事件数和单位路程累计比动能。','- comparison.json 包含各段/各轮/整域均值、中位数、P95、最大值，以及碰撞前总轮心速度和轮质量动能。','- *_touchdowns.csv 是逐事件原始分析表；*_quiet.npz 是200Hz原始信号；*.log 是运行日志；comparison.png/pdf 是比较图。','- MuJoCo 无声、无GUI批量执行，未观看实时窗口；物理与原部署入口相同，图由真实采样生成。']
(OUT/'REPORT_CN.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({tag:{key:s[key] for key in ['run','duration_s','goal_crossing_time_s','stairs','common_complete_domain']} for tag,s in summaries.items()},ensure_ascii=False,indent=2))
