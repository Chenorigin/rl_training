"""Compare frozen rollout events, not mixed-controller TensorBoard scalars."""
from pathlib import Path
import csv
import json
import math
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
CRITERIA = json.loads((OUT/'criteria.json').read_text())
SECTIONS = [('all_terrain',-1,38),('stairs',1,6.6),('ascent',1,2.8),('platform',2.8,4.8),('descent',4.8,6.6)]
NAMES={'all_terrain':'全地形','stairs':'楼梯总体','ascent':'上楼','platform':'楼顶平台','descent':'下楼'}
FIELDS=[('speed_mean_m_s','触地前速度均值 m/s'),('speed_p95_m_s','触地前速度 P95 m/s'),
        ('energy_mean_J_kg','单次比动能 J/kg'),('energy_per_xy_m','每米累计比动能 J/kg/m'),
        ('force_peak_mean_N','40ms 力峰均值 N'),('force_peak_p95_N','40ms 力峰 P95 N'),
        ('force_peak_max_N','最大 40ms 力峰 N'),('excess_impulse_mean_N_s','40ms 超额冲量均值 N·s'),
        ('slip_gated_time_mean_m_s','接触门控滑移时间均值 m/s')]

def extract(folder):
    with np.load(folder/'quiet.npz') as saved:
        d={k:saved[k] for k in ('metadata','time','root_pos_w','speed','specific_energy','force','force_norm_N',
                               'normal_force_N','pre_velocity','slip_m_s','contact','event')}
    m=json.loads(str(d['metadata']))
    t,p=d['time'],d['root_pos_w'];dt=CRITERIA['sampling']['step_s']
    if not np.allclose(np.diff(t),dt,atol=1e-8):raise ValueError('Physics sample timing mismatch')
    for k in ('speed','specific_energy','force','force_norm_N','normal_force_N','root_pos_w','pre_velocity','slip_m_s'):
        if not np.isfinite(d[k]).all():raise ValueError('Nonfinite '+k)
    if not np.allclose(d['specific_energy'],.5*d['speed']**2,atol=1e-6):raise ValueError('Energy identity failed')
    if not np.allclose(d['force_norm_N'],np.linalg.norm(d['force'],axis=-1),atol=1e-5,rtol=1e-5):raise ValueError('Force identity failed')
    domain=(t>=.1)&(p[:,0]>=-1)&(p[:,0]<38)
    frames=np.argwhere(d['event']&domain[:,None]);rows=[];truncated=0
    for i,leg in frames:
        if i+8>len(t):truncated+=1;continue
        v=float(d['speed'][i,leg]);normal=d['normal_force_N'][i:i+8,leg]
        fn=d['force_norm_N'][i:i+8,leg]
        rows.append({'time_s':float(t[i]),'x_m':float(p[i,0]),'y_m':float(p[i,1]),'leg':('fl','fr','hl','hr')[leg],
                     'speed_m_s':v,'specific_energy_J_kg':.5*v*v,
                     'wheel_energy_J':.5*m['wheel_masses_kg'][leg]*v*v,
                     'force_peak_N':float(fn.max()),'normal_impulse_N_s':float(normal.sum()*dt),
                     'excess_impulse_N_s':float(np.maximum(normal-100.,0).sum()*dt)})
    sections={}
    for name,a,b in SECTIONS:
        mask=domain&(p[:,0]>=a)&(p[:,0]<b)
        dist=float(np.linalg.norm(np.diff(p[:,:2],axis=0),axis=1)[mask[1:]].sum())
        progress=float(np.maximum(np.diff(p[:,0]),0)[mask[1:]].sum())
        subset=[r for r in rows if a<=r['x_m']<b]
        speed=np.array([r['speed_m_s'] for r in subset]);energy=np.array([r['specific_energy_J_kg'] for r in subset])
        force=np.array([r['force_peak_N'] for r in subset]);impulse=np.array([r['excess_impulse_N_s'] for r in subset])
        values={'event_count':len(subset),'distance_xy_m':dist,'forward_positive_progress_m':progress,
                'sample_duration_s':float(mask.sum()*dt),'events_per_m':len(subset)/dist if dist else None,
                'speed_mean_m_s':float(speed.mean()) if len(speed) else None,
                'speed_p95_m_s':float(np.quantile(speed,.95)) if len(speed) else None,
                'energy_mean_J_kg':float(energy.mean()) if len(energy) else None,
                'energy_per_xy_m':float(energy.sum()/dist) if dist else None,
                'energy_per_forward_m':float(energy.sum()/progress) if progress else None,
                'force_peak_mean_N':float(force.mean()) if len(force) else None,
                'force_peak_p95_N':float(np.quantile(force,.95)) if len(force) else None,
                'force_peak_max_N':float(force.max()) if len(force) else None,
                'excess_impulse_mean_N_s':float(impulse.mean()) if len(impulse) else None,
                'normal_force_time_mean_N':d['normal_force_N'][mask].mean(axis=0).tolist() if mask.any() else None,
                'slip_gated_time_mean_m_s':float((d['slip_m_s'][mask]*d['contact'][mask]).mean()) if mask.any() else None}
        sections[name]=values
    if rows:
        with (folder/'touchdowns.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return {'summary':m,'sections':sections,'incomplete_events_excluded':truncated,
            'primary_included':m['complete'] and not m['fell'] and not m['ever_boundary_exit']}

def main():
    runs={}
    for folder in sorted(OUT.iterdir()):
        if folder.is_dir() and (folder/'quiet.npz').exists():runs[folder.name]=extract(folder)
    paired={};primary={}
    for name in ('baseline','quiet_landing_v1','student'):
        cases=[r for r in runs.values() if r['summary']['model']==name and r['summary']['profile']=='clean' and r['primary_included']]
        primary[name]={'completed_count':len(cases),'attempted_count':3,'sections':{}}
        for section,_,_ in SECTIONS:
            primary[name]['sections'][section]={}
            for key in ['event_count','distance_xy_m','forward_positive_progress_m','sample_duration_s','events_per_m','energy_per_forward_m']+[x[0] for x in FIELDS]:
                vals=[r['sections'][section][key] for r in cases if r['sections'][section][key] is not None]
                primary[name]['sections'][section][key]={'mean':float(np.mean(vals)),'min':float(min(vals)),'max':float(max(vals))} if vals else None
    for reference in ('baseline','quiet_landing_v1'):
        paired[reference]={}
        for section,_,_ in SECTIONS:
            paired[reference][section]={}
            for key,_ in FIELDS:
                values=[]
                for case in CRITERIA['cases']:
                    a=runs.get(f'{reference}_clean_{case["name"]}');b=runs.get(f'student_clean_{case["name"]}')
                    if a and b and a['primary_included'] and b['primary_included']:
                        av,bv=a['sections'][section][key],b['sections'][section][key]
                        if av is not None and bv is not None and av!=0:values.append(100*(bv/av-1))
                if values:
                    paired[reference][section][key]={'mean_percent':float(np.mean(values)), 'min_percent':float(min(values)), 'max_percent':float(max(values)), 'paired_count':len(values)}
    result={'runs':runs,'primary_case_mean_statistics':primary,'paired_student_percent_changes':paired,
            'aggregation':'equal weight per completed initial case; P95 separately computed from events in each case, then averaged; per-case paired percentages averaged',
            'impulse_definition':'sum(max(estimated-normal actual-contact force - 100N,0))*0.005 over 8 solver steps; source ExtendedQuietSettings.allowed_normal_force=100N',
            'measurement_checks':'5ms timing, all selected physics arrays finite, specific_energy=.5*speed^2, force_norm=norm(actual contact force)'}
    (OUT/'metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    with (OUT/'comparison.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['section','metric','baseline','quiet_landing_v1','student','student_vs_baseline_paired_percent','student_vs_v1_paired_percent'])
        for section,_,_ in SECTIONS:
            for key,label in FIELDS:
                vals=[primary[name]['sections'][section][key] for name in ('baseline','quiet_landing_v1','student')]
                changes=[paired[name][section].get(key) for name in ('baseline','quiet_landing_v1')]
                writer.writerow([section,label]+[v['mean'] if v else None for v in vals]+[v['mean_percent'] if v else None for v in changes])
    print('RUNS',len(runs),'PRIMARY', {name:r['completed_count'] for name,r in primary.items()})
    for section in ('ascent','descent','stairs','all_terrain'):
        print('SECTION',section)
        for key,label in FIELDS:
            print(label,[primary[n]['sections'][section][key] for n in ('baseline','quiet_landing_v1','student')],
                  'student_vs_v1',paired['quiet_landing_v1'][section].get(key))

if __name__=='__main__':main()
