from pathlib import Path
import csv,json
import numpy as np

OUT=Path(__file__).resolve().parent
OLD=Path('docs/quiet_landing_零命令倾斜与直线上下楼修复_20261011')
CONTRACT=json.loads((OUT/'本轮对照冻结条件.json').read_text())
TAGS=list(CONTRACT['checkpoints'])

def yaw(q):
    w,x,y,z=q.T
    return np.rad2deg(np.unwrap(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))))

def describe(values):
    return {'mean':float(np.mean(values)),'p95':float(np.quantile(values,.95)),'max':float(np.max(values))} if len(values) else {'mean':None,'p95':None,'max':None}

results={}
for tag in TAGS:
    folder=Path(CONTRACT['replay_folders'][tag])
    cases={};results[tag]=cases
    for case in ['零命令','仅w']:
        prefix=f'{tag}_{case}_'
        pose=np.load(folder/(prefix+'姿态.npz'))
        d=np.load(folder/(prefix+'原部署回放_quiet.npz'))
        meta=json.loads(str(d['metadata']));t=d['time'];p=d['root_pos_w']
        fell=bool(meta['fell_in_policy'])
        result={'fell':fell,'completed_full_all_terrain':bool(meta['cleared_terrain']),
                'inside_course_final':bool(meta['inside_course']),'max_x_m':float(meta['max_x']),
                'total_simulation_s':float(t[-1]),'source':str(folder/(prefix+'原部署回放_quiet.npz'))}
        cases[case]=result
        if fell:
            result['excluded']='Entire falling replay excluded from contact and stationary averages.'
            continue
        assert np.allclose(np.diff(t),.005,atol=1e-8)
        assert np.allclose(d['specific_energy'],.5*d['speed']**2,atol=1e-6)
        assert np.allclose(d['force_norm_N'],np.linalg.norm(d['force'],axis=-1),atol=1e-5,rtol=1e-5)
        for key in ['speed','specific_energy','force_norm_N','root_pos_w','normal_force_N']:
            assert np.isfinite(d[key]).all(),(tag,key)
        pt=pose['time'];xy=pose['pos'];angle=yaw(pose['quat']);start=np.flatnonzero(pt>=6)[0]
        result['policy_forward_displacement_m']=float(xy[-1,0]-xy[start,0])
        result['policy_yaw_change_deg']=float(angle[-1]-angle[start])
        if case=='零命令':
            fn=d['normal_force_N'][t>=10].mean(axis=0)
            result['four_wheel_mean_normal_force_N']=fn.astype(float).tolist()
            result['diagonal_ratio']=float((fn[0]+fn[3]-fn[1]-fn[2])/fn.sum())
            result['policy_xy_displacement_m']=float(np.linalg.norm(xy[-1,:2]-xy[start,:2]))
            continue
        result['sections']={};rows=[]
        for i,wheel in np.argwhere(d['event']&(t>=8)[:,None]):
            if i+8>len(t):continue
            speed=float(d['speed'][i,wheel]);mass=meta['wheel_masses_kg'][wheel]
            rows.append({'time_s':float(t[i]),'root_x_m':float(p[i,0]),'root_y_m':float(p[i,1]),'wheel':int(wheel),
                         'pre_normal_speed_m_s':speed,'specific_energy_J_kg':.5*speed**2,
                         'wheel_normal_energy_J':.5*mass*speed**2,'force_peak_40ms_N':float(d['force_norm_N'][i:i+8,wheel].max())})
        if rows:
            with (OUT/(tag+'_仅w_逐次触地速度动能40毫秒力峰.csv')).open('w') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        # Shared physical route, not the different complete trajectory extents.
        for lo,hi,name in [(1,6.6,'楼梯总体'),(1,2.8,'上楼'),(2.8,4.8,'楼顶平台'),(4.8,6.6,'下楼')]:
            idx=np.flatnonzero((t>=8)&(p[:,0]>=lo)&(p[:,0]<hi))
            mask=(pt>=8)&(xy[:,0]>=lo)&(xy[:,0]<hi)
            subset=[r for r in rows if lo<=r['root_x_m']<hi]
            dist=float(np.linalg.norm(np.diff(p[idx,:2],axis=0),axis=1).sum()) if len(idx)>1 else 0.
            duration=float(len(idx)*.005)
            result['sections'][name]={'events':len(subset),'distance_xy_m':dist,'duration_s':duration,
                'mean_forward_speed_m_s':float((p[idx[-1],0]-p[idx[0],0])/duration) if duration else None,
                'normal_speed_m_s':describe([r['pre_normal_speed_m_s'] for r in subset]),
                'specific_energy_J_kg':describe([r['specific_energy_J_kg'] for r in subset]),
                'wheel_normal_energy_J':describe([r['wheel_normal_energy_J'] for r in subset]),
                'force_peak_40ms_N':describe([r['force_peak_40ms_N'] for r in subset]),
                'events_per_m':len(subset)/dist if dist else None,
                'specific_energy_per_m_J_kg_m':sum(r['specific_energy_J_kg'] for r in subset)/dist if dist else None,
                'max_abs_world_y_m':float(np.abs(xy[mask,1]).max()) if mask.any() else None,
                'max_abs_world_yaw_deg':float(np.abs(angle[mask]).max()) if mask.any() else None,
                'mean_normal_force_N':d['normal_force_N'][idx].mean(axis=0).astype(float).tolist() if len(idx) else None}
        result['stairs_exit_world_y_m']=float(xy[np.flatnonzero(xy[:,0]>=6.6)[0],1]) if (xy[:,0]>=6.6).any() else None
        result['stairs_exited']=bool((xy[:,0]>=6.6).any())
(OUT/'原部署分路段与驻足统计.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
for tag in TAGS:
    print(tag,'ZERO',results[tag]['零命令'])
    for name,stats in results[tag]['仅w'].get('sections',{}).items():print(tag,name,stats)
