from pathlib import Path
import json,hashlib,re
import numpy as np

OUT=Path(__file__).resolve().parent
C=json.loads((OUT/'本轮对照冻结条件.json').read_text())
R=json.loads((OUT/'原部署分路段与驻足统计.json').read_text())

def euler(q):
    w,x,y,z=q.T
    roll=np.arctan2(2*(w*x+y*z),1-2*(x*x+y*y))
    pitch=np.arcsin(np.clip(2*(w*y-z*x),-1,1))
    yaw=np.unwrap(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)))
    return np.rad2deg(np.column_stack([roll,pitch,yaw]))

checks=[]
for path,sha in C['source_sha256'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha,path
for tag,path in C['checkpoints'].items():
    sha=hashlib.sha256(Path(path).read_bytes()).hexdigest();assert sha==C['checkpoint_sha256'][tag]
    for case in ['零命令','仅w']:
        d=np.load(OUT/f'{tag}_{case}_原部署回放_quiet.npz')
        a=np.load(OUT/f'{tag}_{case}_姿态.npz')
        op=json.loads((OUT/f'{tag}_{case}_操作条件.json').read_text())
        meta=json.loads(str(d['metadata']));s=R[tag][case]
        assert op['checkpoint_sha256']==sha and op['physics'] and not op['course_correction']
        assert op['command_body']==([.7,0,0] if case=='仅w' else [0,0,0])
        actual=np.load(OUT/f'{tag}_{case}_原部署回放.npz')
        assert np.allclose(actual['command'][actual['time']>=8.02],op['command_body'])
        assert np.allclose(d['specific_energy'],.5*d['speed']**2,atol=1e-6)
        assert np.allclose(d['force_norm_N'],np.linalg.norm(d['force'],axis=-1),atol=1e-5,rtol=1e-5)
        angles=euler(a['quat']);t=a['time'];xy=a['pos'][:,:2]
        check={'model':tag,'case':case,'sha256':sha,'fell':s['fell'],'samples':len(d['time']),
               'physics':True,'external_correction':False}
        resource=(OUT/f'{tag}_{case}_资源实测.log').read_text()
        match=re.search(r'Maximum resident set size \(kbytes\): (\d+)',resource)
        check['peak_RSS_KiB']=int(match[1]) if match else None
        assert 'Exit status: 0' in resource
        checks.append(check)
        if s['fell']:continue
        if case=='零命令':
            steady=t>=10;last=np.flatnonzero(t>=26)[0]
            s['last10_mean_translation_m_s']=float(np.linalg.norm(xy[-1]-xy[last])/(t[-1]-t[last]))
            s['steady_roll_mean_deg']=float(angles[steady,0].mean())
            s['steady_pitch_mean_deg']=float(angles[steady,1].mean())
            s['steady_roll_abs_max_deg']=float(np.abs(angles[steady,0]).max())
            s['steady_pitch_abs_max_deg']=float(np.abs(angles[steady,1]).max())
            fn=np.array(s['four_wheel_mean_normal_force_N'])
            s['four_wheel_time_mean_force_CV']=float(fn.std()/fn.mean())
            s['four_wheel_loaded_fraction']=float((d['normal_force_N'][d['time']>=10]>5).all(axis=1).mean())
            continue
        start=np.flatnonzero(t>=8)[0];heading=np.deg2rad(angles[start,2])
        cross=(xy-xy[start])@np.array([-np.sin(heading),np.cos(heading)])
        s['W_start_heading_deg']=float(angles[start,2])
        s['W_start_world_xy_m']=xy[start].tolist()
        for lo,hi,name in [(1,6.6,'楼梯总体'),(1,2.8,'上楼'),(2.8,4.8,'楼顶平台'),(4.8,6.6,'下楼')]:
            v=s['sections'][name];mask=(t>=8)&(xy[:,0]>=lo)&(xy[:,0]<hi)
            v['max_cross_track_from_W_heading_m']=float(np.abs(cross[mask]).max()) if mask.any() else None
            v['max_heading_change_from_W_deg']=float(np.abs(angles[mask,2]-angles[start,2]).max()) if mask.any() else None
            fn=np.array(v['mean_normal_force_N'])
            v['time_mean_diagonal_ratio']=float(abs((fn[0]+fn[3]-fn[1]-fn[2])/fn.sum())) if fn.sum() else None
            events=np.argwhere(d['event']&(d['time']>=8)[:,None]&((d['root_pos_w'][:,0]>=lo)&(d['root_pos_w'][:,0]<hi))[:,None])
            events=[(int(i),int(w)) for i,w in events if i+8<=len(d['time'])]
            assert len(events)==v['events']
            if not events:continue
            energy=[.5*float(d['speed'][i,w])**2 for i,w in events]
            force=[float(np.linalg.norm(d['force'][i:i+8,w],axis=-1).max()) for i,w in events]
            assert np.isclose(np.mean(energy),v['specific_energy_J_kg']['mean'])
            assert np.isclose(np.mean(force),v['force_peak_40ms_N']['mean'],rtol=1e-5)
            # Fixed forward span avoids rewarding a longer sideways trajectory in the denominator.
            if mask.any():
                forward=float(xy[mask,0].max()-xy[mask,0].min())
                v['specific_energy_per_forward_m_J_kg_m']=sum(energy)/forward if forward else None

(OUT/'步态与触地综合统计.json').write_text(json.dumps(R,ensure_ascii=False,indent=2)+'\n')
(OUT/'独立重算与资源核验.json').write_text(json.dumps({'result':'PASS','checks':checks,'source_sha256_checked':C['source_sha256'],
        'six_fresh_physical_replays':len(checks)==6,'all_no_fall':all(not c['fell'] for c in checks),
        'fall_exclusion':'Entire falling replay excluded; not selectively cutting around force peaks.'},ensure_ascii=False,indent=2)+'\n')
for tag,r in R.items():
    print(tag,'ZERO',r['零命令'])
    print(tag,'MOVING',r['仅w'].get('stairs_exited'),r['仅w'].get('max_x_m'))
    for name,s in r['仅w'].get('sections',{}).items():
        print(name,'v',s['normal_speed_m_s']['mean'],'E',s['specific_energy_J_kg']['mean'],
              'F',s['force_peak_40ms_N']['mean'],'F95',s['force_peak_40ms_N']['p95'],
              'energy/M',s['specific_energy_per_m_J_kg_m'],'side',s['max_abs_world_y_m'],
              'heading',s['max_abs_world_yaw_deg'],'anchoredSide',s['max_cross_track_from_W_heading_m'])
