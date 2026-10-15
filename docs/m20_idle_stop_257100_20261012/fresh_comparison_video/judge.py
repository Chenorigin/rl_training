from pathlib import Path
import json,sys,hashlib
import numpy as np
B=Path(__file__).resolve().parent;C=json.loads((B/'criteria.json').read_text());A=C['acceptance']
SOURCES={**C['source_sha256'],**json.loads((B/'instrument_sources.json').read_text())['source_sha256']}
def angles(q):
 w,x,y,z=q.T
 return np.rad2deg(np.column_stack([np.arctan2(2*(w*x+y*z),1-2*(x*x+y*y)),np.arcsin(np.clip(2*(w*y-z*x),-1,1)),np.unwrap(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)))]))
def calculate(label):
 out={};folder=B/'evaluations'/label;identity=None
 for case in ['零命令','平地启停','仅w']:
  f=np.load(folder/(case+'_full.npz'));t=f['time'];p=f['qpos'][:,:3];e=angles(f['qpos'][:,3:7]);d=np.load(folder/(label+'_'+case+'_原部署回放_quiet.npz'));a=np.load(folder/(label+'_'+case+'_原部署回放.npz'));m=json.loads(str(d['metadata']))
  receipt=json.loads((folder/(label+'_'+case+'_操作条件.json')).read_text())
  assert receipt['label']==label and receipt['case']==case
  cp=Path(receipt['checkpoint']);actual_sha=hashlib.sha256(cp.read_bytes()).hexdigest()
  assert receipt['checkpoint_sha256']==actual_sha
  assert m['checkpoint']==str(cp.resolve()) and m['model']==C['model']
  assert receipt['source_sha256']==SOURCES
  for name,expected in receipt['source_sha256'].items():
   assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==expected
  expected_terrain=C['flat_terrain'] if case=='平地启停' else C['terrain']
  assert receipt['terrain_xml']==expected_terrain and m['terrain_xml']==expected_terrain
  if identity is None:identity=actual_sha
  else:assert identity==actual_sha
  if label=='baseline':assert actual_sha==C['checkpoint_sha256']['V1_257100']
  assert receipt['course_correction'] is False and receipt['physics'] is True
  assert receipt['Z_s']==0 and receipt['C_s']==6
  assert receipt['W_s']==(None if case=='零命令' else 8)
  assert receipt['W_release_s']==(14 if case=='平地启停' else None)
  expected_schedule=[[0,[0,0,0]]]+([[8,[.7,0,0]]]+([[14,[0,0,0]]] if case=='平地启停' else []) if case!='零命令' else [])
  assert receipt['command_schedule']==expected_schedule
  # Actions/commands use the pre-physics timestamp; saved full states are
  # post-physics. Exclude only the floating-point command switch boundaries.
  command_time=a['time']
  expected_command=np.zeros_like(a['command'])
  moving=(command_time>=8.005)&((command_time<13.995) if case=='平地启停' else True) if case!='零命令' else np.zeros(len(command_time),bool)
  expected_command[moving,0]=.7
  safe=(abs(command_time-8)>.025)&(abs(command_time-14)>.025)
  assert np.allclose(a['command'][safe],expected_command[safe])
  assert (a['mode'][a['time']>=6.06]=='policy').all()
  assert len(t)==1800 and np.allclose(np.diff(t),.02)
  assert np.allclose(np.diff(d['time']),.005)
  assert np.allclose(d['specific_energy'],.5*d['speed']**2,atol=1e-6)
  assert np.allclose(d['force_norm_N'],np.linalg.norm(d['force'],axis=-1))
  assert np.isfinite(f['qpos']).all() and np.isfinite(a['action']).all()
  r={'fell':m['fell_in_policy'],'duration_s':float(t[-1]),'actor_action_dimension':a['action'].shape[1],'no_runtime_stop_clamp':True};out[case]=r
  if case!='仅w':
   start=10 if case=='零命令' else A['stop_time_s'];ids=np.flatnonzero(t>=start);i=ids[0];j=np.flatnonzero(t>=26)[0]
   r.update(drift_m=float(np.linalg.norm(p[i:,:2]-p[i,:2],axis=1).max()),net_displacement_m=float(np.linalg.norm(p[-1,:2]-p[i,:2])),window_path_m=float(np.linalg.norm(np.diff(p[i:,:2],axis=0),axis=1).sum()),residual_last10_m_s=float(np.linalg.norm(np.diff(p[j:,:2],axis=0),axis=1).sum()/(t[-1]-t[j])),yaw_excursion_deg=float(abs(e[i:,2]-e[i,2]).max()),final_wheel_actions=a['action'][-1,12:].tolist())
   if case=='平地启停':
    before=np.flatnonzero(t>=8.02)[0];r['prestop_forward_progress_m']=float(p[i,0]-p[before,0])
   assert np.allclose(a['command'][a['time']>=start+.04],0)
  else:
   assert np.allclose(a['command'][a['time']>=8.02],[.7,0,0])
   exits=np.flatnonzero(p[:,0]>=A['stairs_exit_x_m']);r['stairs_exit_time_s']=float(t[exits[0]]) if len(exits) else None
   mask=(d['time']>=8)&(d['root_pos_w'][:,0]>=1)&(d['root_pos_w'][:,0]<6.6)
   ev=[(int(i),int(w)) for i,w in np.argwhere(d['event']&mask[:,None]) if i+8<=len(d['time']) and mask[i:i+8].all()];r['stairs_events']=len(ev)
   assert len(ev)>0
   r['stairs_40ms_mean_peak_N']=float(np.mean([d['force_norm_N'][i:i+8,w].max() for i,w in ev]))
   span=np.ptp(d['root_pos_w'][mask,0]);r['stairs_specific_energy_per_forward_m']=float(sum(d['specific_energy'][i,w] for i,w in ev)/span)
   r['stairs_max_abs_y_m']=float(abs(d['root_pos_w'][mask,1]).max())
 (folder/'metrics.json').write_text(json.dumps(out,ensure_ascii=False,indent=2));return out
label=sys.argv[1];result=calculate(label)
if label!='baseline':
 base=calculate('baseline');tests={}
 tests['no_fall']=all(not r['fell'] for r in result.values())
 tests['full_duration']=all(r['duration_s']>=35.99 for r in result.values())
 tests['stationary_drift']=result['零命令']['drift_m']<=A['max_stationary_drift_m']
 tests['stationary_yaw']=result['零命令']['yaw_excursion_deg']<=A['stationary_yaw_drift_deg']
 tests['stop_displacement']=result['平地启停']['drift_m']<=A['max_stop_displacement_m']
 tests['stop_yaw']=result['平地启停']['yaw_excursion_deg']<=A['stationary_yaw_drift_deg']
 tests['prestop_moving']=result['平地启停']['prestop_forward_progress_m']>=base['平地启停']['prestop_forward_progress_m']*.8
 tests['residual_speed']=all(result[case]['residual_last10_m_s']<=A['max_last10_residual_speed_m_s'] for case in ['零命令','平地启停'])
 v=result['仅w'];b=base['仅w'];tests['stairs_exit']=v['stairs_exit_time_s'] is not None and v['stairs_exit_time_s']<=b['stairs_exit_time_s']+A['max_stairs_exit_delay_relative_to_baseline_s']
 tests['quiet_peak']=v['stairs_40ms_mean_peak_N']<=b['stairs_40ms_mean_peak_N']*A['stairs_40ms_mean_peak_ratio_max']
 tests['quiet_cumulative_energy']=v['stairs_specific_energy_per_forward_m']<=b['stairs_specific_energy_per_forward_m']*A['stairs_specific_energy_per_forward_m_ratio_max']
 verdict={'candidate':label,'passed':all(tests.values()),'checks':tests,'result':result,'baseline':base,'criteria':str(B/'criteria.json')}
 (B/'evaluations'/label/'verdict.json').write_text(json.dumps(verdict,ensure_ascii=False,indent=2));print(json.dumps(verdict,ensure_ascii=False,indent=2))
else:print(json.dumps(result,ensure_ascii=False,indent=2))
