import json, hashlib, math, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco
from PIL import Image, ImageDraw
root=Path.cwd(); out=root/'docs/gait_quality_20261007'
summaries={label:{case:json.loads((out/label/case/'summary.json').read_text()) for case in ('up30','up20','down30')} for label in ('initial','intermediate','latest')}
impulses={}
for label in summaries:
    impulses[label]={}
    for case in summaries[label]:
        p=np.load(out/label/case/'physics.npz'); force=np.linalg.norm(p['forces'],axis=-1); values=[]
        for leg in range(4):
            free=0
            for i in range(len(force)):
                if force[i,leg]<=5: free+=1; continue
                if free>=4 and p['phase'][i] and i+10<=len(force) and p['phase'][i:i+10].all():
                    values.append(float(force[i:i+10,leg].sum()*.005))
                free=0
        impulses[label][case]={'count':len(values),'mean_ns':float(np.mean(values)) if values else None,
                              'p95_ns':float(np.quantile(values,.95)) if values else None,
                              'max_ns':max(values) if values else None}
comparisons={}
for case in ('up30','up20','down30'):
    comparisons[case]={}
    for baseline in ('initial','intermediate'):
        a=summaries[baseline][case]; b=summaries['latest'][case]
        comparisons[case][baseline]={'comparable_completion':a['stop_reasons']==['all_wheels_crossed'] and b['stop_reasons']==['all_wheels_crossed'],
            **{key+'_change_pct':100*(b['force'][key]/a['force'][key]-1) for key in ('mean_n','p95_n','p99_n','max_n')},
            'energy_per_m_change_pct':100*(b['mechanical_energy_j_per_m']/a['mechanical_energy_j_per_m']-1),
            'power_change_pct':100*(b['mechanical_power_mean_w']/a['mechanical_power_mean_w']-1)}
down=np.load(out/'latest/down30/telemetry.npz'); entry=summaries['latest']['down30']['descent_entry']
mask=(down['time']>=entry['first_front_down_s'])&(down['time']<=entry['first_rear_down_s']+1)
rear=down['support'][:,2:]&(down['contact_time'][:,2:]>=.12)&mask[:,None]
fz=np.maximum(down['forces'][:,:,2],0); load=fz[:,2:].sum(1)/np.maximum(fz.sum(1),1)
entry.update(pitch_max_deg=float(down['pitch_deg'][mask].max()),rear_load_fraction_mean=float(load[mask].mean()),rear_load_fraction_max=float(load[mask].max()))
hashes=json.loads((root/'.scratch/gait_quality_20261007/frozen_code.json').read_text())
unchanged={p:hashlib.sha256((root/p).read_bytes()).hexdigest()==h for p,h in hashes.items()}; assert all(unchanged.values())
result={'summaries':summaries,'latest_vs_baselines':comparisons,'touchdown_50ms_force_norm_integrals':impulses,
        'impulse_definition':'Wheel force norm >5 N after >=20ms below threshold, integrate next50ms; full window in phase. Includes support load and riser contacts; not pure collision impulse.',
        'descent_entry_additional':entry,'frozen_code_unchanged':unchanged,
        'scope':'one deterministic CPU rollout per model/geometry; not a success-rate study or energy-reward ablation'}
(out/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
# Independent plots: full pooled wheel-contact quantiles (200 Hz), plus joint/tilt traces (50 Hz).
fig,ax=plt.subplots(1,3,figsize=(13,4),layout='constrained')
labels=['80000','149999','196600']; colors=['#697787','#598dc4','#c26736']
for i,case in enumerate(('up30','up20','down30')):
    for j,label in enumerate(summaries):
        s=summaries[label][case]
        ax[i].bar(j,s['mechanical_energy_j_per_m'],color=colors[j]); ax[i].text(j,s['mechanical_energy_j_per_m']+20,f"{s['mechanical_energy_j_per_m']:.0f}",ha='center')
    ax[i].set_xticks(range(3),labels); ax[i].set_ylabel('Absolute joint mechanical work / distance (J/m)'); ax[i].set_title({'up30':'Ascent: tread 0.30 m','up20':'Ascent: tread 0.20 m','down30':'Descent: tread 0.30 m'}[case])
ax[2].text(.02,.98,'80000 leaves course: unmatched completion',transform=ax[2].transAxes,va='top',fontsize=9,color='#ad2828')
fig.savefig(out/'energy_comparison.png',dpi=150); plt.close(fig)
fig,ax=plt.subplots(2,1,figsize=(10,6),layout='constrained')
up=np.load(out/'latest/up20/telemetry.npz'); selected=up['phase']
for leg,label in enumerate(('FL','FR')): ax[0].plot(up['time'][selected],np.degrees(abs(up['knee'][selected,leg])),label=label)
ax[0].axhline(math.degrees(2.5),ls='--',color='gray',label='current swing envelope'); ax[0].set_ylabel('Absolute front knee angle (deg)'); ax[0].set_title('196600: narrow tread ascent'); ax[0].legend()
for leg,label in ((2,'HL'),(3,'HR')): ax[1].plot(down['time'][mask],down['tilt_body_deg'][mask,leg],label=label+' hip-wheel forward tilt')
ax[1].plot(down['time'][mask],down['pitch_deg'][mask],ls='--',label='body pitch (positive nose down here)'); ax[1].set_ylabel('Angle (deg)'); ax[1].set_xlabel('Simulation time (s)'); ax[1].set_title('196600: descent entry'); ax[1].legend()
fig.savefig(out/'gait_traces.png',dpi=150); plt.close(fig)
# Reconstruct recorded physical states, without changing a trajectory or inventing poses.
sys.path.insert(0,str(root/'deploy/deploy_mujoco')); import deploy_mujoco as deploy
import argparse
robot=Path(summaries['latest']['up20']['model'])
panels=[]
for case,title,signal in [('up20','Front swing: max knee angle','swing'),('up30','Front support: max knee angle','support'),('down30','Descent entry: rear forward reach','rear')]:
    s=np.load(out/'latest'/case/'telemetry.npz'); a=argparse.Namespace(terrain='descent' if case=='down30' else 'ascent',terrain_xml=None,stair_height=.15,tread_depth=.2 if case=='up20' else .3,stair_count=8,stair_start=None)
    m=deploy.load_model(robot,a); c=deploy.M20Contract(m,a.terrain_config); d=mujoco.MjData(m)
    if signal=='rear': eligible=s['support'][:,2:]&(s['contact_time'][:,2:]>=.12)&mask[:,None]; score=np.where(eligible,s['leg_forward_body'][:,2:],-np.inf).max(1)
    else:
        eligible=s['phase'][:,None]&((np.linalg.norm(s['forces'][:,:2],axis=-1)<5) if signal=='swing' else (s['support'][:,:2]&(s['contact_time'][:,:2]>=.20)))
        score=np.where(eligible,np.abs(s['knee'][:,:2]),-np.inf).max(1)
    index=int(score.argmax()); d.qpos[:]=s['qpos'][index]; d.qvel[:]=s['qvel'][index]; mujoco.mj_forward(m,d)
    cam=mujoco.MjvCamera(); cam.lookat[:]=d.xpos[c.base_id]; cam.distance=2.6; cam.azimuth=95; cam.elevation=-15
    with mujoco.Renderer(m,height=480,width=640) as renderer:
        renderer.update_scene(d,camera=cam); renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=0
        im=Image.fromarray(renderer.render()); ImageDraw.Draw(im).text((15,15),f"{title}, t={s['time'][index]:.2f}s",fill='white'); im.save(out/f'{case}_pose.png'); panels.append(im)
print(json.dumps({'comparisons':comparisons,'descent_entry':entry},indent=2))
