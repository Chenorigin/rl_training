"""Compare fixed V1 checkpoints using actual MuJoCo rollout samples."""
from pathlib import Path
import csv, hashlib, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

OUT=Path(__file__).resolve().parent
c=json.loads((OUT/'正式对比冻结条件.json').read_text())
for tag,path in c['checkpoints'].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==c['sha256'][tag]
for path,sha in c['source_sha256'].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha
keys=['法向触地速度_m_s','法向比动能_J_kg','轮体法向动能_J','触地40毫秒力峰_N']
legs=['左前','右前','左后','右后']
results={};data={};all_rows={}
def stats(rows,dist):
    s={'事件数':len(rows),'实际XY路程_m':dist,'事件数每米':len(rows)/dist if dist else None,
       '每米累计比动能_J_kg_m':sum(r['法向比动能_J_kg'] for r in rows)/dist if dist else None}
    for key in keys:
        v=np.array([r[key] for r in rows])
        s[key]={label:float(fn(v)) if len(v) else None for label,fn in [('均值',np.mean),('P95',lambda x:np.quantile(x,.95)),('最大值',np.max)]}
    return s
for tag in c['checkpoints']:
    d=np.load(OUT/(tag+'_正式回放_quiet.npz'));data[tag]=d
    m=json.loads(str(d['metadata']));t=d['time'];p=d['root_pos_w'];mass=np.asarray(m['wheel_masses_kg'])
    result={'运行':{k:m[k] for k in ['fell','cleared_terrain','inside_course','max_x','steps_executed']},'运行秒数':float(t[-1])}
    results[tag]=result
    if m['fell'] or not m['cleared_terrain']:
        result['统计排除原因']='跌倒或未完成全地形；不混入正常触地指标对比';continue
    assert np.allclose(np.diff(t),.005,atol=1e-8)
    for key in ['pre_velocity','speed','specific_energy','force_norm_N','root_pos_w']:
        assert np.isfinite(d[key]).all(),(tag,key)
    assert np.allclose(d['specific_energy'],.5*d['speed']**2,atol=1e-6)
    assert np.allclose(d['force_norm_N'],np.linalg.norm(d['force'],axis=-1),atol=1e-5,rtol=1e-5)
    domain=(t>=.1)&(p[:,0]>=-1)&(p[:,0]<38)
    rows=[]
    for i,w in np.argwhere(d['event']&domain[:,None]):
        if i+8>len(t):continue
        x=float(p[i,0]);section=next(n for a,b,n in c['sections'] if a<=x<b)
        v=float(d['speed'][i,w]);force=d['force_norm_N'][i:i+8,w]
        rows.append({'时间_s':float(t[i]),'机身位置_x_m':x,'机身位置_y_m':float(p[i,1]),'轮足':legs[w],'地形路段':section,
                     '法向触地速度_m_s':v,'法向比动能_J_kg':.5*v*v,'轮体法向动能_J':.5*mass[w]*v*v,
                     '触地首步接触力_N':float(force[0]),'触地40毫秒力峰_N':float(force.max())})
    all_rows[tag]=rows
    if rows:
        with (OUT/(tag+'_逐次触地速度动能接触力.csv')).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    sections=[(-1,38,'全地形'),(1,6.6,'楼梯总体')]+[tuple(x) for x in c['sections']]
    result['分路段']={}
    for a,b,name in sections:
        idx=np.flatnonzero(domain&(p[:,0]>=a)&(p[:,0]<b));dist=float(np.linalg.norm(np.diff(p[idx,:2],axis=0),axis=1).sum())
        subset=[r for r in rows if a<=r['机身位置_x_m']<b]
        s=stats(subset,dist)
        s['分段平均前进速度_m_s']=(b-a)/(t[idx[-1]]-t[idx[0]]+.005) if len(idx) else None
        # Time averaged forces include zero during swing, preventing contact-only bias.
        s['四轮平均接触合力_N']={leg:float(d['force_norm_N'][idx,w].mean()) if len(idx) else None for w,leg in enumerate(legs)}
        result['分路段'][name]=s
    result['四轮楼梯触地']={leg:stats([r for r in rows if 1<=r['机身位置_x_m']<6.6 and r['轮足']==leg],0) for leg in legs}
(OUT/'两次迭代触地指标统计.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
assert len(all_rows)==2,'存在失败回放，已保存状态，停止生成无跌倒对比图'
# A known prior rollout independently checks that reused sampling conditions are identical.
prior=Path('docs/quiet_landing_三版本全地形对比_20261010/V1_正式回放_quiet.npz')
old=np.load(prior)
for key in ['time','root_pos_w','event','speed','force']:
    assert data['V1_230300'][key].shape==old[key].shape
    assert np.allclose(data['V1_230300'][key],old[key],atol=1e-7),key
(OUT/'核验记录.json').write_text(json.dumps({'physics_sampling_s':.005,'energy_identity':'PASS','force_identity':'PASS','all_samples_finite':'PASS','both_complete_no_fall':'PASS','230300_prior_rollout_reproduction':'PASS','new_checkpoint_frozen':237200},ensure_ascii=False,indent=2))
font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc');plt.rcParams['font.family']=font.get_name();plt.rcParams['axes.unicode_minus']=False
colors={'V1_230300':'#e67e22','V1_237200':'#2376b9'}
labels={tag:'V1（'+tag.split('_')[1]+'）' for tag in colors}
names=['上楼','楼顶平台','下楼']
def bars(ax,names,key,measure='均值'):
    xs=np.arange(len(names))
    for offset,(tag,color) in zip([-.18,.18],colors.items()):
        vals=[results[tag]['分路段'][name][key][measure] for name in names]
        b=ax.bar(xs+offset,[v if v is not None else np.nan for v in vals],.34,label=labels[tag],color=color)
        ax.bar_label(b,labels=[f'{v:.3g}' if v is not None else '无事件' for v in vals],fontsize=8)
    ax.set_xticks(xs,names);ax.grid(axis='y',alpha=.2);ax.legend(fontsize=9)
fig,axs=plt.subplots(2,3,figsize=(17,10),constrained_layout=True)
for tag,d in data.items():
    p=d['root_pos_w'];axs[0,0].plot(d['time'],p[:,0],label=labels[tag],color=colors[tag]);axs[0,1].plot(p[:,0],p[:,1],label=labels[tag],color=colors[tag])
    rows=[r for r in all_rows[tag] if 1<=r['机身位置_x_m']<6.6];axs[0,2].scatter([r['机身位置_x_m'] for r in rows],[r['法向触地速度_m_s'] for r in rows],s=12,alpha=.5,label=labels[tag],color=colors[tag])
for ax,title,xlabel,ylabel in zip(axs[0],['全地形前进轨迹：均无跌倒并通过终点','统一路线指令下的侧向漂移','上下楼梯逐次触地前法向速度'],['时间（秒）','机身前进位置（米）','触地时机身位置（米）'],['前进位置（米）','侧向位置（米）','法向速度（米/秒）']):
    ax.set(title=title,xlabel=xlabel,ylabel=ylabel);ax.legend(fontsize=9);ax.grid(alpha=.2)
for ax,key,title,unit in zip(axs[1],[keys[0],keys[1],keys[3]],['触地前法向速度均值','触地前法向比动能均值','触地后40毫秒合力峰的均值'],['米/秒','焦耳/千克','牛顿']):
    bars(ax,names,key);ax.set(title=title,ylabel=unit)
fig.suptitle('V1 的230300与237200：全地形无跌倒通行和上下楼梯触地指标对比\n统一路线反馈；比动能=法向触地速度²/2，接触力为实际地形接触合力',fontsize=15)
fig.savefig(OUT/'V1的230300与237200_全地形无跌倒通行轨迹及上楼平台下楼触地速度比动能与40毫秒接触力峰均值对比.png',dpi=150);plt.close(fig)
fig,axs=plt.subplots(1,3,figsize=(17,5),constrained_layout=True)
for ax,key,measure,title,unit in zip(axs,[keys[0],keys[3],keys[3]],['P95','P95','最大值'],['触地法向速度P95','40毫秒接触力峰P95','单次最大接触力峰'],['米/秒','牛顿','牛顿']):
    bars(ax,names,key,measure);ax.set(title=title,ylabel=unit)
fig.suptitle('V1的230300与237200_上下楼梯高冲击事件对比',fontsize=15)
fig.savefig(OUT/'V1的230300与237200_上楼平台下楼高冲击事件触地速度P95接触力峰P95及最大力峰对比.png',dpi=150);plt.close(fig)
lines=['# V1：230300 与 237200 的全地形闭环对比','','同一次V1训练，追加6900次迭代。两次回放无跌倒且完成全地形；固定checkpoint，不因新文件产生而更换。不是V1.5评测。','','|路段|指标|230300|237200|变化|','|---|---|---:|---:|---:|']
for name in ['全地形','上楼','楼顶平台','下楼']:
    for key,measure in [(keys[0],'均值'),(keys[1],'均值'),(keys[3],'均值'),(keys[0],'P95'),(keys[3],'P95'),(keys[3],'最大值')]:
        a=results['V1_230300']['分路段'][name][key][measure];b=results['V1_237200']['分路段'][name][key][measure]
        lines.append(f'|{name}|{key} {measure}|{a:.4f}|{b:.4f}|{(b/a-1)*100:+.1f}%|' if a and b is not None else f'|{name}|{key} {measure}|{a}|{b}|—|')
for tag,r in results.items():
    lines+=['',f"{labels[tag]}：通过时间{r['运行秒数']:.2f}s；全地形触地事件{r['分路段']['全地形']['事件数']}次；每米累计比动能{r['分路段']['全地形']['每米累计比动能_J_kg_m']:.4f} J/kg/m。"]
lines+=['','口径与限制：','- 使用带物理的MuJoCo闭环，统一路线反馈（世界前进0.5m/s、居中与朝向纠偏），非人工挑选不同指令。','- 触地速度取碰撞求解前轮心沿高程图估计法向的接近速度；比动能=v²/2，轮体动能乘轮体质量，不等于整机有效碰撞能量。','- 接触力取实际地形接触合力，力峰为新触地起40ms内最大值；不包含跌倒回合，不完整窗口排除。','- 路段按机身x划分，前后轮跨段位置可能不一致；姿态与朝向变化可影响结果。','- 每个模型仅一次确定性回放，不能推断统计显著性或真实声音分贝；未做真机或实测声音。','- 230300重跑与历史回放样本匹配；模型、地形、采样源码和checkpoint哈希记录于冻结条件。']
(OUT/'V1两次迭代全地形触地对比报告.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({tag:{name:r['分路段'][name] for name in ['全地形','上楼','下楼']} for tag,r in results.items()},ensure_ascii=False,indent=2))
