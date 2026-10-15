from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
OUT=Path(__file__).resolve().parent
D={name:np.load(OUT/(name+'_全部标量.npz')) for name in ['V1','V1.5']}
manifest=json.loads((OUT/'日志冻结来源.json').read_text())
windows={};summaries={}
for name,d in D.items():
 end=manifest[name]['step_max']+1
 windows[name]={'早期_跳过前100更新':[214800,215800],'230300前1000更新':[229300,230300],'233500前1000更新':[232500,233500],'最近1000更新':[end-1000,end],'前一段5000更新':[end-10000,end-5000],'最近5000更新':[end-5000,end]}
 if name=='V1':windows[name].update({'237200前1000更新':[236200,237200],'255900前1000更新':[254900,255900]})
 summaries[name]={}
 for window,(lo,hi) in windows[name].items():
  summaries[name][window]={}
  for k in d.files:
   if k.endswith('/time'):continue
   a=d[k];a=a[(a[:,0]>=lo)&(a[:,0]<hi)];v=a[:,2]
   assert np.isfinite(v).all(),(name,k)
   if not len(v):continue
   summaries[name][window][k]={'n':len(v),'mean':float(v.mean()),'median':float(np.median(v)),'sd':float(v.std()),'min':float(v.min()),'max':float(v.max())}
(OUT/'分窗口标量统计.json').write_text(json.dumps({'windows':windows,'statistics':summaries},ensure_ascii=False,indent=2))
with (OUT/'分窗口标量统计.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['版本','窗口','起始迭代_含','结束迭代_不含','指标','样本数','均值','中位数','标准差'])
 for name,rows in summaries.items():
  for win,metrics in rows.items():
   for k,v in metrics.items():w.writerow([name,win,*windows[name][win],k,v['n'],v['mean'],v['median'],v['sd']])
font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc');plt.rcParams['font.family']=font.get_name();plt.rcParams['axes.unicode_minus']=False
colors={'V1':'#d87923','V1.5':'#187fb4'}
def blocks(a,width=500):
 x=[];y=[]
 for lo in range(214800,int(a[-1,0])+1,width):
  v=a[(a[:,0]>=lo)&(a[:,0]<lo+width)]
  if len(v):x.append(float(v[:,0].mean())-214700);y.append(float(v[:,2].mean()))
 return np.array(x),np.array(y)
def plot(ax,k,title,unit,scale=1,names=('V1','V1.5')):
 for name in names:
  if k not in D[name]:continue
  x,y=blocks(D[name][k]);ax.plot(x,y*scale,label=name,color=colors[name])
 ax.set(title=title,xlabel='从baseline214700继续的更新次数',ylabel=unit);ax.legend(fontsize=8);ax.grid(alpha=.2)
fig,axs=plt.subplots(3,3,figsize=(18,13),constrained_layout=True)
configs=[('QuietLanding/all/pre_speed_mean_m_s','触地前法向速度均值','m/s'),('QuietLanding/all/specific_energy_mean_J_kg','每次触地比动能均值','J/kg'),('QuietLanding/specific_energy_per_m_J_kg_m','每米累计触地比动能','J/kg/m'),('QuietLanding/all/pre_speed_p95_hist_m_s','触地速度直方图P95','m/s'),('QuietLanding/impact_window_force_max_N','重置批次力峰最大值的日志均值','N'),('QuietLanding/impact_window_excess_impulse_max_N_s','重置批次超额冲量最大值的日志均值','N·s'),('QuietLanding/events_per_m','每米触地事件数','次/m'),('Metrics/base_velocity/error_vel_xy','平移速度跟踪误差','日志原始单位'),('Train/mean_episode_length','平均回合步数（上限1000）','policy steps')]
for ax,cfg in zip(axs.flat,configs):plot(ax,*cfg)
fig.suptitle('V1与V1.5 TensorBoard物理指标与任务表现：每500次更新取块均值\n训练含失败回合与动态地形课程，未按上楼/下楼分段；曲线长度反映训练预算不同',fontsize=15)
fig.savefig(OUT/'V1与V1.5_TensorBoard全训练触地速度比动能每米累计能量冲击力峰冲量触地频率速度跟踪与回合长度趋势.png',dpi=150);plt.close(fig)
fig,axs=plt.subplots(2,3,figsize=(18,9),constrained_layout=True)
for ax,cfg in zip(axs.flat,[('QuietLoad/stable_support_fraction','稳定支撑有效时间比例','%',100),('QuietLoad/stable_diagonal_abs_mean','稳定支撑对角受力差绝对值','比例',1),('QuietReward/quiet_load_balance','均衡奖励的回合归一化贡献','reward/s',1),('QuietLoad/stable_has_samples','重置批次有稳定样本的日志比例','%',100)]):plot(ax,*cfg,names=('V1.5',))
for leg in ['fl','fr','hl','hr']:
 x,y=blocks(D['V1.5']['QuietLoad/'+leg+'/mean_normal_force_N']);axs[1,1].plot(x,y,label=leg.upper())
axs[1,1].set(title='全回合四轮时间平均法向力',xlabel='继续更新次数',ylabel='N');axs[1,1].legend();axs[1,1].grid(alpha=.2)
for k,title in [('QuietReward/quiet_impact_peak','冲击力峰'),('QuietReward/quiet_touchdown','触地速度'),('QuietReward/quiet_load_balance','受力均衡')]:
 x,y=blocks(D['V1.5'][k]);axs[1,2].semilogy(x,np.maximum(-y,1e-12),label=title)
axs[1,2].set(title='V1.5三项惩罚贡献量级（对数轴）',xlabel='继续更新次数',ylabel='|reward/s|');axs[1,2].legend();axs[1,2].grid(alpha=.2)
axs[0,1].axhline(.05,linestyle='--',color='gray',label='奖励死区0.05');axs[0,1].legend()
fig.suptitle('V1.5受力均衡奖励触发覆盖、对角差及贡献量级\nV1日志没有QuietLoad指标；has_samples是批次有样本标记，不是稳定时间占比',fontsize=15)
fig.savefig(OUT/'V1.5_TensorBoard稳定支撑门控有效时间覆盖对角受力差均衡奖励量级及四轮平均力趋势.png',dpi=150);plt.close(fig)
fig,axs=plt.subplots(2,3,figsize=(18,9),constrained_layout=True)
for ax,cfg in zip(axs.flat,[('Task/episode_reward_rate','总回合归一化奖励（目标不同）','reward/s'),('Loss/value','价值函数损失','loss'),('Loss/anchor','相对baseline的锚定损失','loss'),('Policy/mean_std','动作分布标准差','std'),('QuietReward/quiet_touchdown','触地速度奖励贡献','reward/s'),('QuietReward/quiet_impact_peak','冲击峰值奖励贡献（公式不同）','reward/s')]):plot(ax,*cfg)
fig.suptitle('V1与V1.5 TensorBoard优化状态与奖励贡献\n总奖励和峰值惩罚公式不同，不能按跨版本数值大小直接排名',fontsize=15)
fig.savefig(OUT/'V1与V1.5_TensorBoard总奖励价值损失baseline锚定损失探索标准差与触地冲击奖励趋势.png',dpi=150);plt.close(fig)

# Matched-update comparisons and late trends.
selected=[('QuietLanding/all/pre_speed_mean_m_s','触地速度均值 m/s'),('QuietLanding/all/specific_energy_mean_J_kg','触地比动能均值 J/kg'),('QuietLanding/specific_energy_per_m_J_kg_m','每米累计比动能 J/kg/m'),('QuietLanding/impact_window_force_max_N','批次力峰最大值均值 N'),('QuietLanding/impact_window_excess_impulse_max_N_s','批次超额冲量最大值均值 N·s'),('QuietLanding/events_per_m','每米触地事件数'),('Metrics/base_velocity/error_vel_xy','平移速度误差'),('Train/mean_episode_length','平均回合步数'),('Policy/mean_std','动作标准差'),('Loss/anchor','锚定损失')]
lines=['# V1与V1.5 TensorBoard续训必要性分析','','## 读取范围','']
for name,m in manifest.items():lines.append(f"- {name}：迭代 {m['step_min']}—{m['step_max']}，{m['scalar_tags']} 个标量指标；文件冻结读取字节数与SHA见日志冻结来源.json。")
lines+=['','## 结论','','V1后期收益很小，继续原样长训优先级低。V1.5仍有缓慢改善，不能判定已经收敛；但改善没有稳定迁移到此前MuJoCo全地形回放，且受力均衡项虽然触发，覆盖与贡献很低。应优先验证/改善有效支撑样本覆盖和评测对齐，再决定是否续训。','','## 关键指标：1000次更新窗口均值','','|指标|V1早期|V1_230300前|V1_237200前|V1最近|V1.5早期|V1.5_230300前|V1.5最近|','|---|---:|---:|---:|---:|---:|---:|---:|']
for k,label in selected:
 vals=[]
 for name,win in [('V1','早期_跳过前100更新'),('V1','230300前1000更新'),('V1','237200前1000更新'),('V1','最近1000更新'),('V1.5','早期_跳过前100更新'),('V1.5','230300前1000更新'),('V1.5','最近1000更新')]:vals.append(summaries[name][win][k]['mean'])
 lines.append('|'+label+'|'+'|'.join(f'{x:.5g}' for x in vals)+'|')
lines+=['','## 后期趋势：最近5000更新与前一段5000更新','', '|版本|指标|前5000均值|后5000均值|变化|','|---|---|---:|---:|---:|']
late={}
for name in D:
 late[name]={}
 for k,label in selected:
  a=summaries[name]['前一段5000更新'][k]['mean'];b=summaries[name]['最近5000更新'][k]['mean'];delta=100*(b/a-1) if a else None
  late[name][k]=delta;lines.append(f'|{name}|{label}|{a:.6g}|{b:.6g}|{delta:+.2f}%|')
lines+=['','## 同迭代训练预算比较','','|窗口|指标|V1|V1.5|V1.5相对变化|','|---|---|---:|---:|---:|']
for win in ['230300前1000更新','233500前1000更新']:
 for k,label in selected:
  a=summaries['V1'][win][k]['mean'];b=summaries['V1.5'][win][k]['mean'];lines.append(f'|{win}|{label}|{a:.6g}|{b:.6g}|{100*(b/a-1):+.2f}%|')
q=summaries['V1.5']['最近1000更新'];coverage=q['QuietLoad/stable_support_fraction']['mean'];balance=abs(q['QuietReward/quiet_load_balance']['mean']);impact=abs(q['QuietReward/quiet_impact_peak']['mean']);touch=abs(q['QuietReward/quiet_touchdown']['mean'])
lines+=['','## 受力均衡项：已触发但有效样本少','',f"- 最近1000更新稳定支撑时间比例日志均值 {100*coverage:.5f}%（约每1000秒采样时间0.860秒满足完整稳定门控；这是日志批次比例的均值，不是重新汇总的全局时长比）。",f"- stable_has_samples均值为{100*q['QuietLoad/stable_has_samples']['mean']:.2f}%，它描述重置批次是否有稳定样本，与有效时间比例不是一个量。",f"- 稳定支撑对角差绝对值均值 {q['QuietLoad/stable_diagonal_abs_mean']['mean']:.5f}，奖励死区0.05；早期窗口为{summaries['V1.5']['早期_跳过前100更新']['QuietLoad/stable_diagonal_abs_mean']['mean']:.5f}，近期两个5000更新窗口只变化约-0.56%。",f"- 均衡惩罚贡献约-{balance:.7f} reward/s，冲击峰值约-{impact:.6f}、触地约-{touch:.6f}；均衡项分别约为它们的{100*balance/impact:.5f}%和{100*balance/touch:.5f}%。平均贡献不能直接等同于策略梯度，但说明信号量级很小。",'- 门控要求全扫描高差≤0.025m、四轮连续支撑≥0.15s、四轮法向力>5N、总力≥100N、低垂向速度和低机身角速度、朝向命令较小及机身接近水平。代码上这些条件可能减少迈步时覆盖；现有日志没有逐条件拒绝率，不能判断哪个条件主导。','- V1没有QuietLoad日志，缺失不能解释为零力或零稳定样本；MuJoCo采样器未传balance_gate，与训练门控不同。','', '## 优化与任务保留','', '- 学习率固定1e-5、critic warmup已结束，loss/value有限；日志未显示数值发散，但未记录KL和clip fraction，不能据此证明PPO更新始终理想。','- 动作标准差持续下降（entropy_coef=0），探索正在减弱；不能凭标准差下降单独判定收敛。锚定损失较早期增大，表示相对baseline发生漂移，不能直接判断漂移好坏。','- 回合长度接近1000步上限，表明训练中多数回合持续较久；不等于楼梯通过率。平移跟踪误差较早期有所增加，需关注任务与quiet取舍。','- Episode_Termination来自本机TerminationManager中各环境最近一次终止原因的存储状态；约98.7%的time_out不是严格的本窗口新回合成功率，更不是上/下楼成功率。','', '## 为什么与MuJoCo有不同结果','', '- TensorBoard统计随机命令、多地形课程及训练动作采样；此前MuJoCo使用确定性Actor和单一all_terrain路线反馈。训练物理指标改善不能直接推导固定下楼场景改善。','- 所有训练quiet统计都包含失败回合；冲击峰指标是重置批次内回合最大值，再经过日志平均，不是无跌倒单次触地力峰均值。二者不能直接比较绝对值。','- TensorBoard未保留楼梯方向、地形等级/类型、命令暴露、逐门控拒绝率的分组日志；课程可能改变采样难度。现有日志无法将quiet变化与采样难度变化完全分离。','- V1.5峰值惩罚使用不饱和尾部公式，V1使用饱和公式；V1.5峰值reward更负不等于力更大，总奖励跨版本也不可直接排名。','- 日志以重置批次汇总，再对每更新中的batch取均值；本报告均值是TensorBoard标量的窗口均值，未伪装成逐事件加权总体统计。','', '## 建议','', '- V1：优先暂停原样长训并保留230300/237200/255900候选；收益已很小且下楼闭环有退步。这里是建议，本轮没有停止训练。','- V1.5：不应仅因最新一次MuJoCo退步就认定无训练价值；训练累计动能仍缓慢降低。应先增加门控条件覆盖诊断，确认受力均衡项能得到足够有效样本，再做有限续训及固定场景复测。','- 继续训练的验收应同时看下楼力峰P95/最大值、每米累计比动能、任务完成与速度跟踪、稳定支撑对角差；不以总奖励或迭代编号单独选模型。','- 本轮只读取日志并生成分析，未修改奖励、观测、配置或运行中的训练。']
(OUT/'V1与V1.5_TensorBoard趋势受力均衡门控及继续训练必要性详细报告.md').write_text('\n'.join(lines)+'\n')
(OUT/'近期趋势变化.json').write_text(json.dumps(late,ensure_ascii=False,indent=2))
print(json.dumps({'last5k_vs_prior5k':late,'balance_coverage_pct':coverage*100,'balance_relative_impact_pct':balance/impact*100},ensure_ascii=False,indent=2))
