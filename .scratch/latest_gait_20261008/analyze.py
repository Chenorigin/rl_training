import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path.cwd(); OUT=ROOT/'docs/latest_gait_20261008'
records={}
for label in ('baseline','latest'):
 records[label]={}
 for case in ('xml20','up30','down30','turn_q','turn_e'):
  directory=OUT/label/case
  r=json.loads((directory/'summary.json').read_text()); S=np.load(directory/'telemetry.npz'); P=np.load(directory/'physics.npz'); phase=S['phase']
  r['final_root_pos_measured']=S['root_pos'][-1].tolist(); r['final_wheel_pos_measured']=S['wheel_pos'][-1].tolist()
  r['task_completed']=bool('all_wheels_crossed' in r['stop_reasons'] and not r['fell_in_policy']) if not case.startswith('turn') else None
  for k in ('front_loaded','front_support','front_swing'):
   r[k]['a_min_deg']=None if r[k]['knee_abs_max_deg'] is None else 180-r[k]['knee_abs_max_deg']
   r[k]['a_p05_deg']=None if r[k]['knee_abs_p95_deg'] is None else 180-r[k]['knee_abs_p95_deg']
  if case=='down30':
   v=r['descent_entry'];entry=(S['time']>=v['first_front_down_s'])&(S['time']<=v['first_rear_down_s']+1)
   support=S['support'][:,2:]&(S['contact_time'][:,2:]>=.12)&entry[:,None]
   v['rear_angle_excess_fraction']=float(((S['tilt_body_deg'][:,2:]>35)|(S['tilt_yaw_deg'][:,2:]>25))[support].mean())
   v['rear_body_b_max_deg']=float(S['tilt_body_deg'][:,2:][support].max()); v['rear_gravity_g_max_deg']=float(S['tilt_yaw_deg'][:,2:][support].max())
  if case.startswith('turn'):
   r['turn']['wheel_reach_max_m']=float(np.linalg.norm((S['wheel_pos']-S['hip_pos'])[phase,:,:2],axis=-1).max())
   r['mechanical_energy_j_per_m']=None # yaw-only drift is not forward efficiency
  for axle in ('front_sequence','rear_sequence'):
   if case in ('xml20','up30'):
    levels={e['level'] for e in r[axle]['events']}; expected=set(range(1,20 if case=='xml20' else 8))
    r[axle]['missing_intermediate_treads']=sorted(expected-levels)
    r[axle]['strict_whole_flight']=bool(r['task_completed'] and r[axle]['checks'] and r[axle]['correct']==r[axle]['checks'] and not(expected-levels) and len(r[axle]['events'])==len(expected))
  # Do not rewrite raw summary: derived fields are a separate record.
  records[label][case]=r
(OUT/'comparison.json').write_text(json.dumps(records,indent=2)+'\n')
colors={'baseline':'#2166ac','latest':'#d95f02'}
fig,axs=plt.subplots(2,3,figsize=(15,8),layout='constrained')
for label in records:
 color=colors[label]; prefix=OUT/label
 s=np.load(prefix/'xml20/telemetry.npz'); p=s['phase']; pos=s['root_pos'][p]
 axs[0,0].plot(pos[:,0],pos[:,1],label=label+' root',color=color)
 axs[0,0].plot(pos[:,0],s['wheel_pos'][p,:,1].max(1),label=label+' outer wheel',color=color,ls=':',alpha=.7)
 # knee P05, same completed30cm case.
 r=records[label]['up30']; vals=[r[k]['a_p05_deg'] for k in ('front_loaded','front_swing')]
 x=np.arange(2)+(-.17 if label=='baseline' else .17)
 axs[0,1].bar(x,vals,width=.32,color=color,label=label)
 r=records[label]['down30']['descent_entry']; axs[0,2].bar(x,[r['rear_tilt_body_p95_deg'],r['rear_tilt_yaw_p95_deg']],width=.32,color=color,label=label)
 s=np.load(prefix/'up30/telemetry.npz');t=s['time'];p=s['phase'];
 for axle,ax in [('front_sequence',axs[1,0]),('rear_sequence',axs[1,1])]:
  ev=records[label]['up30'][axle]['events']; lo=min(e['time'] for e in ev)
  for side in (0,1):
   e=[e for e in ev if (e['leg'].endswith('r'))==bool(side)]
   ax.scatter([v['time']-lo for v in e],[v['level'] for v in e],color=color,marker='o' if side==0 else 'x',s=55,label=label+(' left' if side==0 else ' right'))
 vals=[records[label][c]['mechanical_energy_j_per_m'] for c in ('up30','down30')]
 axs[1,2].bar(x,vals,width=.32,color=color,label=label)
axs[0,0].axhline(2,color='gray',ls='--');axs[0,0].axhline(-2,color='gray',ls='--');axs[0,0].set(xlabel='Root X (m)',ylabel='Root Y (m)',title='20 cm tread: latest exits staircase')
axs[0,1].set(xticks=np.arange(2),xticklabels=['Loaded','Swing'],ylabel='Internal knee a, P05 (deg)',title='30 cm ascent: less front folding')
axs[0,2].set(xticks=np.arange(2),xticklabels=['Body b','Gravity g'],ylabel='Rear angle P95 (deg)',title='30 cm descent entry: more forward tilt')
for ax,title in zip(axs[1,:2],['Front landing ledger','Rear landing ledger']): ax.set(xlabel='Time since first landing (s)',ylabel='Riser index',title=title);ax.set_yticks(range(1,8));ax.legend(fontsize=8,ncols=2)
axs[1,2].set(xticks=np.arange(2),xticklabels=['Ascent','Descent'],ylabel='Abs. mechanical work / distance (J/m)',title='Same completed 30 cm courses')
for ax in axs.flat: ax.grid(alpha=.2);ax.set_axisbelow(True)
for ax in axs[0,:]:ax.legend(fontsize=9)
axs[1,2].legend(fontsize=9)
fig.savefig(OUT/'comparison.png',dpi=150);plt.close(fig)
# Full available latest run, bin averages, no mixing resumed iteration axes.
tb=json.loads((OUT/'tensorboard.json').read_text());tags=tb['latest']['tags']
fig,axs=plt.subplots(2,3,figsize=(15,8),layout='constrained')
metrics=[('Train/mean_reward','Mean reward'),('Policy/mean_std','Mean action std'),('Loss/value','Value loss'),('Curriculum/terrain_levels','Mean terrain level'),('Curriculum/step_completion/up_front_strict_riser_fraction','Strict local risers'),('Curriculum/gait_quality/front_loaded_a_mean_deg','Mean front knee a (deg)')]
for ax,(tag,title) in zip(axs.flat,metrics):
 points=np.asarray(tags[tag]['bins_1000']);ax.plot(points[:,0],points[:,1],color=colors['latest'],label=tag.rsplit('/',1)[-1]);ax.set(xlabel='gait_safety_v2 iteration',title=title);ax.grid(alpha=.2);ax.axvspan(16000,17500,color='gray',alpha=.15)
 if tag=='Loss/value':ax.set_yscale('log')
 if 'strict_riser' in tag:
  pts=np.asarray(tags['Curriculum/step_completion/up_rear_strict_riser_fraction']['bins_1000']);ax.plot(pts[:,0],pts[:,1],label='rear_strict_riser',color=colors['baseline']);ax.set_ylim(0,1)
 if 'front_loaded' in tag:
  pts=np.asarray(tags['Curriculum/gait_quality/front_swing_a_mean_deg']['bins_1000']);ax.plot(pts[:,0],pts[:,1],label='swing',color=colors['baseline'])
 ax.legend(fontsize=8)
fig.savefig(OUT/'training_curves.png',dpi=150);plt.close(fig)
print('Written comparison.json / comparison.png / training_curves.png')
