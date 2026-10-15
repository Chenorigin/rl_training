"""Reproducible analysis artifact; reads existing MuJoCo records, never runs policies."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

OUT = Path(__file__).resolve().parent
contract = json.loads((OUT/'正式对比冻结条件.json').read_text())
for k,p in contract['checkpoints'].items():
    assert hashlib.sha256(Path(p).read_bytes()).hexdigest() == contract['sha256'][k]
for p,sha in contract['source_sha256'].items():
    assert hashlib.sha256(Path(p).read_bytes()).hexdigest() == sha
colors = {'Baseline':'#2e6fbb','V1':'#e67e22','V1.1':'#239b56'}
labels = {'Baseline':'Baseline（214700）','V1':'V1（230300）','V1.1':'V1.1（1800次更新）'}
legs = ['左前','右前','左后','右后']
datasets = {}; events = {}; results = {}

def stats(rows):
    out = {'事件数':len(rows)}
    for key in ['法向触地速度_m_s','法向比动能_J_kg','轮体法向动能_J','触地首步接触力_N','触地40毫秒力峰_N']:
        values = np.array([r[key] for r in rows])
        out[key] = {name:float(fn(values)) if len(values) else None for name,fn in
                    [('均值',np.mean),('中位数',np.median),('P95',lambda x:np.quantile(x,.95)),('最大值',np.max)]}
    return out

for tag in colors:
    d = np.load(OUT/(tag+'_正式回放_quiet.npz')); datasets[tag] = d
    meta = json.loads(str(d['metadata']))
    results[tag] = {'运行':{k:meta[k] for k in ['fell','cleared_terrain','inside_course','max_x','steps_executed']},'运行秒数':float(d['time'][-1])}
    # A failed traversal never contributes to the normal-contact comparison.
    assert not meta['fell'] and meta['cleared_terrain'], f'{tag}: failed traversal; do not pool it'
    t = d['time']; pos = d['root_pos_w']; mass = np.array(meta['wheel_masses_kg'])
    for key in ['pre_velocity','speed','specific_energy','force_norm_N','root_pos_w']:
        assert np.isfinite(d[key]).all(), (tag,key)
    assert np.allclose(np.diff(t),.005,atol=1e-8)
    assert np.allclose(d['specific_energy'],.5*d['speed']**2,atol=1e-6)
    assert np.allclose(d['force_norm_N'],np.linalg.norm(d['force'],axis=-1),rtol=1e-5,atol=1e-5)
    domain = (t>=.1)&(pos[:,0]>=-1)&(pos[:,0]<38)
    rows = []
    for i,w in np.argwhere(d['event'] & domain[:,None]):
        if i+8 > len(t): continue
        x = float(pos[i,0]); name = next(n for a,b,n in contract['sections'] if a<=x<b)
        v = float(d['speed'][i,w]); f = d['force_norm_N'][i:i+8,w]
        rows.append({'时间_s':float(t[i]),'机身位置_x_m':x,'机身位置_y_m':float(pos[i,1]),'轮足':legs[w],'地形路段':name,
                     '法向触地速度_m_s':v,'法向比动能_J_kg':.5*v*v,'轮体法向动能_J':.5*mass[w]*v*v,
                     '触地首步接触力_N':float(f[0]),'触地40毫秒力峰_N':float(f.max())})
    events[tag] = rows
    with (OUT/(tag+'_逐次触地速度动能接触力.csv')).open('w') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    results[tag]['分路段'] = {n:stats([r for r in rows if r['地形路段']==n]) for a,b,n in contract['sections']}
    for a,b,n in contract['sections']:
        idx = np.flatnonzero(domain & (pos[:,0]>=a)&(pos[:,0]<b))
        distance = float(np.linalg.norm(np.diff(pos[idx,:2],axis=0),axis=1).sum())
        q = results[tag]['分路段'][n]
        q['实际XY路程_m'] = distance
        q['单位路程累计比动能_J_kg_m'] = sum(r['法向比动能_J_kg'] for r in rows if r['地形路段']==n)/distance
        q['分段平均前进速度_m_s'] = (b-a)/(t[idx[-1]]-t[idx[0]]+.005)
    results[tag]['四轮楼梯段'] = {w:stats([r for r in rows if 1<=r['机身位置_x_m']<6.6 and r['轮足']==w]) for w in legs}
    for name,a,b in [('楼梯总体',1,6.6),('全地形总体',-1,38)]:
        subset = [r for r in rows if a<=r['机身位置_x_m']<b]; q = stats(subset)
        idx = np.flatnonzero(domain & (pos[:,0]>=a)&(pos[:,0]<b))
        dist = float(np.linalg.norm(np.diff(pos[idx,:2],axis=0),axis=1).sum())
        q['实际XY路程_m'] = dist;q['单位路程累计比动能_J_kg_m'] = sum(r['法向比动能_J_kg'] for r in subset)/dist
        q['事件次数每米'] = len(subset)/dist
        q['分段平均前进速度_m_s'] = (b-a)/(t[idx[-1]]-t[idx[0]]+.005)
        results[tag][name] = q

(OUT/'三模型触地指标统计.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
font = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family'] = font.get_name();plt.rcParams['axes.unicode_minus'] = False
def annotate_bars(ax,names,key,measure='均值'):
    xs = np.arange(len(names))
    for off,tag in zip([-.25,0,.25],colors):
        values = [results[tag]['分路段'][n][key][measure] for n in names]
        bars = ax.bar(xs+off,[v if v is not None else np.nan for v in values],.24,label=labels[tag],color=colors[tag])
        ax.bar_label(bars,labels=[f'{v:.3g}' if v is not None else '无事件' for v in values],fontsize=8)
        for i,v in enumerate(values):
            if v is None: ax.text(i+off,0,'无事件',rotation=90,fontsize=7,ha='center',va='bottom')
    ax.set_xticks(xs,names);ax.grid(axis='y',alpha=.2)

fig,axes = plt.subplots(2,3,figsize=(17,10),constrained_layout=True)
for tag,d in datasets.items():
    pos = d['root_pos_w'];axes[0,0].plot(d['time'],pos[:,0],label=labels[tag],color=colors[tag])
    axes[0,1].plot(pos[:,0],pos[:,1],label=labels[tag],color=colors[tag])
    rows = [r for r in events[tag] if 1<=r['机身位置_x_m']<6.6]
    axes[0,2].scatter([r['机身位置_x_m'] for r in rows],[r['法向触地速度_m_s'] for r in rows],s=12,alpha=.5,label=labels[tag],color=colors[tag])
axes[0,0].set(title='三策略全地形前进轨迹：均无跌倒并越过终点',xlabel='时间（秒）',ylabel='机身前进位置（米）');axes[0,0].axhline(38,color='gray',ls=':')
axes[0,1].set(title='共同居中指令下的侧向漂移',xlabel='机身前进位置（米）',ylabel='机身侧向位置（米）');axes[0,1].axhline(0,color='gray',ls=':')
axes[0,2].set(title='上下楼梯逐次触地前法向接近速度',xlabel='触地时机身位置（米）',ylabel='法向速度（米/秒）')
names = ['上楼','楼顶平台','下楼']
for ax,key,title,unit in zip(axes[1],['法向触地速度_m_s','法向比动能_J_kg','触地40毫秒力峰_N'],['轮足触地前法向速度均值','轮足触地前法向比动能均值','触地后40毫秒合力峰值的均值'],['米/秒','焦耳/千克','牛顿']):
    annotate_bars(ax,names,key);ax.set(title=title,ylabel=unit)
for ax in axes.flat:ax.legend(fontsize=9);ax.grid(alpha=.2)
fig.suptitle('Baseline、V1、V1.1：all_terrain 无跌倒回放的轮足触地指标对比\n统一路线指令反馈；楼梯按机身位置分段；比动能=法向速度²/2',fontsize=16)
main_name='三策略全地形无跌倒通行轨迹及上楼_楼顶平台_下楼轮足触地速度_比动能_40毫秒接触力峰值对比'
fig.savefig(OUT/(main_name+'.png'),dpi=160);fig.savefig(OUT/(main_name+'.pdf'));plt.close(fig)

names = [n for a,b,n in contract['sections'] if not n.startswith('缓冲')]
fig,axes = plt.subplots(3,1,figsize=(18,13),constrained_layout=True)
for ax,key,title,unit in zip(axes,['法向触地速度_m_s','法向比动能_J_kg','触地40毫秒力峰_N'],['各地形路段法向触地速度均值','各地形路段法向比动能均值','各地形路段40毫秒接触力峰均值'],['米/秒','焦耳/千克','牛顿']):
    annotate_bars(ax,names,key);ax.set(title=title,ylabel=unit);ax.legend(fontsize=9)
fig.suptitle('三策略全地形主要路段：速度、比动能与接触力对比（无跌倒数据）\n无新触地事件的路段不填写零值；后轮/前轮跨段位置以机身x归类',fontsize=15)
fig.savefig(OUT/'三策略全地形各主要路段轮足触地法向速度_比动能_40毫秒接触力峰均值对比.png',dpi=160);plt.close(fig)

fig,axes = plt.subplots(1,3,figsize=(17,5),constrained_layout=True)
for ax,key,measure,title,unit in zip(axes,['法向触地速度_m_s','触地40毫秒力峰_N','触地40毫秒力峰_N'],['P95','P95','最大值'],['楼梯触地速度P95','楼梯接触力峰P95','楼梯单次最大接触力峰'],['米/秒','牛顿','牛顿']):
    annotate_bars(ax,['上楼','楼顶平台','下楼'],key,measure);ax.set(title=title,ylabel=unit);ax.legend(fontsize=9)
fig.suptitle('三策略上下楼梯高冲击事件对比：速度P95、力峰P95与最大力峰',fontsize=15)
fig.savefig(OUT/'三策略上楼_楼顶平台_下楼高冲击事件触地速度P95_接触力峰P95与最大值对比.png',dpi=160);plt.close(fig)

def fmt(x): return '—（无新触地）' if x is None else f'{x:.3f}'
lines=['# Baseline、V1、V1.1 全地形无跌倒回放对比','', '三次正式回放均完整通过终点条件且没有跌倒。发生跌倒的运行不得进入这些统计；本次正式运行没有此类样本。','', '## 固定模型与运行条件','']
for tag,checkpoint in contract['checkpoints'].items():lines.append(f'- {labels[tag]}：`{checkpoint}`；运行{results[tag]["运行秒数"]:.2f}秒。')
lines+=['','V1.1从V1的230300初始化，checkpoint编号1800是实验更新次数，不能与Teacher累计编号直接相减。V1.1仍在训练，本次固定1800，不随新的checkpoint出现而改变。','', 'all_terrain.xml与M20 MJCF保持；200Hz物理、50Hz策略。目标世界前进速度0.5m/s，统一航向/居中指令反馈，不修改PD或策略。原比较固定零yaw rate并不保持航向，Baseline曾侧漂出界；本次Baseline在相同反馈下正常通过。所有模型采用同一反馈，不混用旧comparison.png里的统计。实际逐步指令保存在正式回放.npz的command。','', '出生前100ms排除；法向来自原高程图估计。速度为碰撞前轮心法向接近速度，比动能=v²/2；乘轮质量为轮体法向平动动能代理，不是整腿有效碰撞能量或声能。接触力是轮足-静态地形净合力模长，逐事件取开始后8个5ms物理步的峰值，剔除不完整窗口。统计机身x∈[-1,38)，路段归类按机身位置。','', '## 共同上楼、平台和下楼','', '|路段|模型|新触地数|平均法向速度(m/s)|平均比动能(J/kg)|平均轮体法向动能(J)|平均40ms力峰(N)|P95力峰(N)|最大力峰(N)|','|---|---|---:|---:|---:|---:|---:|---:|---:|']
for n in ['上楼','楼顶平台','下楼']:
    for tag in colors:
        q=results[tag]['分路段'][n]
        lines.append('|'+ '|'.join([n,labels[tag],str(q['事件数']),fmt(q['法向触地速度_m_s']['均值']),fmt(q['法向比动能_J_kg']['均值']),fmt(q['轮体法向动能_J']['均值']),fmt(q['触地40毫秒力峰_N']['均值']),fmt(q['触地40毫秒力峰_N']['P95']),fmt(q['触地40毫秒力峰_N']['最大值'])])+'|')
lines+=['','## 总体统计','', '|范围|模型|事件数|平均法向速度(m/s)|平均比动能(J/kg)|平均40ms力峰(N)|累计比动能/实际路程(J/kg/m)|分段平均前进速度(m/s)|','|---|---|---:|---:|---:|---:|---:|---:|']
for scope in ['楼梯总体','全地形总体']:
    for tag in colors:
        q=results[tag][scope];lines.append('|'+ '|'.join([scope,labels[tag],str(q['事件数']),fmt(q['法向触地速度_m_s']['均值']),fmt(q['法向比动能_J_kg']['均值']),fmt(q['触地40毫秒力峰_N']['均值']),fmt(q['单位路程累计比动能_J_kg_m']),fmt(q['分段平均前进速度_m_s'])])+'|')
lines+=['','## 限制与复现','', '- 每模型一次确定性回放；不是多初态/多速度统计，不代表实测声音改善。V1.1训练预算与V1不同，不能将全部差异单独归因于奖励，需要同预算V1续训对照。','- 三者完成的是几何终点/赛道宽度/无跌倒判据；没有逐障碍接触审计，不能证明未绕过每块错位障碍。','- 没有新触地事件不等于没有接触力：持续滚动支撑不计新触地，图中不将无事件填成零冲击。','- 参考图布局保留，图片文件名、标题、轴名和单位均用中文明确指标。原逐轮200Hz信号在 *_正式回放_quiet.npz；中文逐事件CSV、全部路段/四轮JSON及分析脚本一起保存。','- 冻结条件文件包含checkpoint/模型/地形/采样源码SHA256；主分析前均核验匹配，Baseline可行性检查不混入正式数据。','- 两个新增命令参数仅用于原部署入口的autoplay回放，默认关闭；交互键盘回放不使用。']
(OUT/'三策略全地形无跌倒轮足触地对比报告.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({tag:{k:results[tag][k] for k in ['运行','运行秒数','楼梯总体','全地形总体']} for tag in colors},ensure_ascii=False,indent=2))
