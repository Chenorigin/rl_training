"""Compare baseline and fixed V1 checkpoints using actual MuJoCo rollout samples."""
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
(OUT/'多迭代触地指标统计.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family']=font.get_name();plt.rcParams['axes.unicode_minus']=False
order=['Baseline','V1_230300','V1_233500','V1_237200','V1_255900','V1.5_230300','V1.5_233500']
labels={tag:('Baseline 214700' if tag=='Baseline' else tag.replace('_',' ')) for tag in order}
colors=dict(zip(order,['#777777','#f5bb52','#e68c24','#b85b15','#744312','#60b6df','#176ca6']))
valid=[tag for tag in order if tag in all_rows]
names=['全地形','上楼','楼顶平台','下楼']
# Independent measured diagnostics: swing zeros remain in time-average force.
for tag in valid:
 d=data[tag];p=d['root_pos_w'];t=d['time'];fn=d['normal_force_N'];domain=(t>=.1)&(p[:,0]>=-1)&(p[:,0]<38)
 results[tag]['受力诊断']={}
 for a,b,name in [(-1,38,'全地形'),(-1,1,'起步平地'),(2.8,4.8,'楼顶平台'),(36,38,'终点平地')]:
  mask=domain&(p[:,0]>=a)&(p[:,0]<b);avg=fn[mask].mean(axis=0)
  diag=(avg[0]+avg[3]-avg[1]-avg[2])/max(float(avg.sum()),1e-9)
  results[tag]['受力诊断'][name]={'四轮时间平均法向力_N':dict(zip(legs,map(float,avg))), '时间平均力的对角差占比':float(diag),'样本秒数':float(mask.sum()*.005)}
(OUT/'多迭代触地指标统计.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')

def bars(ax,tags,names,key,measure='均值'):
 tags=[t for t in tags if t in valid];xs=np.arange(len(names));width=.8/max(len(tags),1)
 for i,tag in enumerate(tags):
  vals=[results[tag]['分路段'][n][key][measure] for n in names]
  b=ax.bar(xs+(i-(len(tags)-1)/2)*width,vals,width*.95,label=labels[tag],color=colors[tag])
  ax.bar_label(b,labels=[f'{v:.3g}' for v in vals],fontsize=8)
 ax.set_xticks(xs,names);ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)

groups=[(['Baseline','V1_230300','V1_237200','V1_255900','V1.5_233500'],'Baseline与V1历史及V1和V1.5最新模型'),(['Baseline','V1_230300','V1.5_230300','V1_233500','V1.5_233500'],'Baseline与V1和V1.5同迭代230300及233500模型')]
for tags,title in groups:
 fig,axs=plt.subplots(2,3,figsize=(20,10),constrained_layout=True)
 for ax,key,unit in zip(axs[0],[keys[0],keys[1],keys[3]],['米/秒','焦耳/千克','牛顿']):
  bars(ax,tags,['上楼','楼顶平台','下楼'],key);ax.set(title=key+'均值',ylabel=unit)
 for ax,key,measure,unit in zip(axs[1],[keys[0],keys[3],keys[3]],['P95','P95','最大值'],['米/秒','牛顿','牛顿']):
  bars(ax,tags,['上楼','楼顶平台','下楼'],key,measure);ax.set(title=key+measure,ylabel=unit)
 fig.suptitle(title+'：无跌倒完成回合的上下楼梯触地速度、比动能及40毫秒接触合力峰\n仅一次确定性闭环；速度和动能取触地前，力峰取触地后；失败回合不纳入柱状图',fontsize=14)
 fig.savefig(OUT/(title+'_上楼楼顶平台下楼触地速度比动能均值与接触合力峰均值P95及最大值对比.png'),dpi=150);plt.close(fig)
fig,axs=plt.subplots(2,3,figsize=(20,10),constrained_layout=True)
for tag in order:
 d=data[tag];p=d['root_pos_w'];style='-' if tag in valid else '--'
 axs[0,0].plot(d['time'],p[:,0],style,label=labels[tag],color=colors[tag]);axs[0,1].plot(p[:,0],p[:,1],style,label=labels[tag],color=colors[tag])
for ax,title,x,y in [(axs[0,0],'全地形通行轨迹（失败回合虚线）','时间（秒）','机身前进位置（米）'),(axs[0,1],'侧向位置','机身前进位置（米）','侧向位置（米）')]:
 ax.set(title=title,xlabel=x,ylabel=y);ax.legend(fontsize=8);ax.grid(alpha=.2)
for ax,metric,title in [(axs[0,2],'运行秒数','通过耗时（秒）'),(axs[1,0],'事件数','全地形触地次数'),(axs[1,1],'每米累计比动能_J_kg_m','每米累计触地比动能（J/kg/m）'),(axs[1,2],'触地40毫秒力峰_N','全地形触地后40毫秒最大力峰（N）')]:
 vals=[]
 for tag in valid:
  r=results[tag];s=r['分路段']['全地形'];vals.append(r[metric] if metric=='运行秒数' else s[metric]['最大值'] if metric==keys[3] else s[metric])
 b=ax.bar(range(len(valid)),vals,color=[colors[t] for t in valid]);ax.bar_label(b,fmt='%.3g',fontsize=9);ax.set_xticks(range(len(valid)),[labels[t] for t in valid],rotation=30,ha='right',fontsize=9);ax.set_title(title);ax.grid(axis='y',alpha=.2)
fig.suptitle('Baseline、V1与V1.5七个checkpoint：全地形通过情况、耗时、触地次数与累计能量\n累计能量补充事件均值，避免以更多小触地掩盖总量增加',fontsize=15)
fig.savefig(OUT/'Baseline与V1及V1.5七模型_全地形通行轨迹侧向漂移通过耗时触地次数每米累计比动能和最大接触力峰对比.png',dpi=150);plt.close(fig)
fig,axs=plt.subplots(1,4,figsize=(20,5),constrained_layout=True)
for ax,name in zip(axs,['全地形','起步平地','楼顶平台','终点平地']):
 xs=np.arange(4);width=.8/max(len(valid),1)
 for i,tag in enumerate(valid):
  q=results[tag]['受力诊断'][name];v=list(q['四轮时间平均法向力_N'].values())
  ax.bar(xs+(i-(len(valid)-1)/2)*width,v,width,label=labels[tag],color=colors[tag])
 ax.set_xticks(xs,legs);ax.set(title=name,ylabel='平均法向力（N）');ax.legend(fontsize=7);ax.grid(axis='y',alpha=.2)
fig.suptitle('四轮时间平均法向受力（包含摆动零力）；不同运动姿态和支撑时序影响均值\n此诊断未复现训练稳定支撑门控，不等于V1.5受力均衡奖励验收',fontsize=14)
fig.savefig(OUT/'Baseline与V1及V1.5七模型_全地形起步平地楼顶平台终点平地四轮平均法向受力诊断.png',dpi=150);plt.close(fig)

overview=['## 本次结论','','- 七个固定checkpoint均无跌倒且完整通过。最新不代表最优：V1的255900下楼相比237200速度均值+31.4%、比动能均值+62.1%、力峰均值+29.1%；相对baseline下楼这三项也分别+4.5%、+5.5%、+13.8%。','- V1.5最新233500相对baseline，上楼速度/比动能/力峰均值分别降低46.2%/62.2%/38.3%，下楼分别降低19.1%/30.6%/17.7%；下楼最大力峰降低33.0%。','- V1.5最新233500相对自身230300，全地形每米累计比动能增加26.7%，耗时82.60→89.14秒；下楼最大力峰仍改善4.9%，但平均力峰增加7.3%。','- 同233500迭代，V1.5比V1下楼平均力峰降低12.3%、P95降低28.6%、最大力峰降低25.3%；全地形每米累计比动能反而增加8.5%，耗时增加7.5%。','- 本次全地形每米累计比动能最低为V1.5的230300（2.746J/kg/m）。它可保留为后续复测候选，不根据单回合宣称最终最优。','- 四轮平均受力均衡未见一致改善：全地形平均力对角差占比，Baseline为7.14%，V1_230300为2.72%，V1_255900为8.32%，V1.5_230300为16.35%，V1.5_233500为12.99%。须进一步复现稳定支撑门控，不能据此直接认定均衡奖励有效或无效。','']
lines=['# Baseline、V1、V1.5多迭代MuJoCo all_terrain对比','',c['comparison'],'']+overview+['## 回合通过情况','', '|模型|相对baseline新增更新|通过且无跌倒|耗时_s|最大x_m|','|---|---:|---|---:|---:|']
for tag in order:
 r=results[tag];iteration=214700 if tag=='Baseline' else int(tag.split('_')[-1]);lines.append(f"|{labels[tag]}|{iteration-214700}|{'是' if tag in valid else '否：'+r.get('统计排除原因','')}|{r['运行秒数']:.2f}|{r['运行']['max_x']:.3f}|")
for name in names:
 lines+=['','## '+name,'','|模型|事件数|触地速度均值_m/s|比动能均值_J/kg|40ms力峰均值_N|力峰P95_N|最大力峰_N|累计比动能每米_J/kg/m|','|---|---:|---:|---:|---:|---:|---:|---:|']
 for tag in valid:
  s=results[tag]['分路段'][name]
  v=[s[keys[0]]['均值'],s[keys[1]]['均值'],s[keys[3]]['均值'],s[keys[3]]['P95'],s[keys[3]]['最大值'],s['每米累计比动能_J_kg_m']]
  lines.append('|'+labels[tag]+f"|{s['事件数']}|"+'|'.join(f'{x:.4f}' for x in v)+'|')
lines+=['','## 同迭代与历史变化','', '|对照（后者相对前者）|路段|速度均值变化|比动能均值变化|力峰均值变化|力峰P95变化|最大力峰变化|每米累计比动能变化|','|---|---|---:|---:|---:|---:|---:|---:|']
pairs=[('Baseline','V1_255900'),('Baseline','V1.5_233500'),('V1_230300','V1_255900'),('V1_237200','V1_255900'),('V1.5_230300','V1.5_233500'),('V1_230300','V1.5_230300'),('V1_233500','V1.5_233500')]
changes={}
for a,b in pairs:
 if a not in valid or b not in valid:continue
 changes[a+' → '+b]={}
 for name in ['全地形','上楼','下楼']:
  sa=results[a]['分路段'][name];sb=results[b]['分路段'][name]
  va=[sa[keys[0]]['均值'],sa[keys[1]]['均值'],sa[keys[3]]['均值'],sa[keys[3]]['P95'],sa[keys[3]]['最大值'],sa['每米累计比动能_J_kg_m']]
  vb=[sb[keys[0]]['均值'],sb[keys[1]]['均值'],sb[keys[3]]['均值'],sb[keys[3]]['P95'],sb[keys[3]]['最大值'],sb['每米累计比动能_J_kg_m']]
  delta=[100*(y/x-1) for x,y in zip(va,vb)];changes[a+' → '+b][name]=delta
  lines.append('|'+labels[a]+' → '+labels[b]+'|'+name+'|'+'|'.join(f'{x:+.1f}%' for x in delta)+'|')
lines+=['','## 四轮受力诊断','', '|模型|路段|左前_N|右前_N|左后_N|右后_N|平均力对角差占比|','|---|---|---:|---:|---:|---:|---:|']
for tag in valid:
 for name,q in results[tag]['受力诊断'].items():
  lines.append('|'+labels[tag]+'|'+name+'|'+'|'.join(f'{x:.2f}' for x in q['四轮时间平均法向力_N'].values())+f"|{q['时间平均力的对角差占比']:+.2%}|")
lines+=['','## 口径与限制','',
'- 每个模型一次确定性真实物理闭环；3份历史回放经checkpoint与源码哈希核验后复用，4份新回放本轮执行。失败回合整体排除触地汇总，保留状态与轨迹。',
'- 使用同一外部路线反馈：世界前进0.5m/s、横向居中和朝向纠偏；反馈随姿态变化，各模型实际命令序列不会完全相同。',
'- 触地速度为碰撞求解前轮心沿高程图估计法向的接近速度；比动能=v²/2，轮体动能=轮体质量×比动能（详见CSV），不等于整机有效碰撞能量。NPZ旧energy_definition字段描述的是另一个平移动能量，本文实际按speed数组重算并核验specific_energy。',
'- 力取静态地形实际接触合力模长，新触地起8个5ms求解步内峰值；使用5/2N滞回、至少10ms离地及100ms初始化保护、踏面法向力比例至少0.5；末尾不完整40ms窗口排除。',
'- 路段按机身x切分；前后轮可能跨段，完成终点不能单独证明每个障碍都没有绕行。',
'- 最新V1与V1.5训练时长不同；同230300、233500对照控制累计更新次数，但单条训练和单回合仍不能证明奖励变化的统计因果。两训练都从baseline214700继续，seed42、固定学习率1e-5、4096环境。',
'- 四轮受力平均包含摆动零力；全程平均或平地平均不等于训练的稳定四足支撑门控。MuJoCo采样器未复现V1.5稳定门控，本报告不使用其stable_has_samples=0判定学习失败，也不把采样器的默认V1奖励值当作V1.5奖励。',
'- 未测真实声音、声压或分贝；未进行真机和多种速度、多次随机回合鲁棒性测试。']
(OUT/'Baseline与V1及V1.5最新与历史迭代全地形触地及四轮受力对比报告.md').write_text('\n'.join(lines)+'\n')
(OUT/'相对变化统计.json').write_text(json.dumps(changes,ensure_ascii=False,indent=2))
(OUT/'核验记录.json').write_text(json.dumps({'source_and_checkpoint_hashes':'PASS','physics_dt':.005,'finite_samples':'PASS for completed runs','energy_identity':'PASS','force_identity':'PASS','completed_runs':valid,'excluded_runs':[t for t in order if t not in valid],'new_rollouts':['V1_233500','V1_255900','V1.5_230300','V1.5_233500'],'reused_rollouts':list(c['reuse_sources'])},ensure_ascii=False,indent=2))
print(json.dumps({'status':{t:results[t]['运行'] for t in order},'changes':changes},ensure_ascii=False,indent=2))
