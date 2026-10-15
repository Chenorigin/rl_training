from pathlib import Path
import numpy as np,json,hashlib
B=Path(__file__).resolve().parent;C=json.loads((B/'criteria.json').read_text());A=C['acceptance'];S={**C['source_sha256'],**json.loads((B/'instrument_sources.json').read_text())['source_sha256']}
commands={'slow':(.35,0,0),'reverse':(-.7,0,0),'turn':(0,0,.6),'lateral':(0,.5,0)}
def calc(label):
 out={};identity=None
 for case,cmd in commands.items():
  p=B/'evaluations'/label;f=np.load(p/(case+'_full.npz'));a=np.load(p/(label+'_'+case+'_原部署回放.npz'));d=np.load(p/(label+'_'+case+'_原部署回放_quiet.npz'));m=json.loads(str(d['metadata']));receipt=json.loads((p/(label+'_'+case+'_操作条件.json')).read_text())
  assert receipt['label']==label and receipt['scenario']==case
  cp=Path(receipt['checkpoint']);sha=hashlib.sha256(cp.read_bytes()).hexdigest();assert sha==receipt['checkpoint_sha256']
  assert receipt['source_sha256']==S
  for name,h in S.items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==h
  assert m['checkpoint']==str(cp.resolve()) and m['terrain_xml']==C['flat_terrain'] and m['model']==C['model']
  assert not receipt['course_correction'] and receipt['physics']
  if identity is None:identity=sha
  else:assert identity==sha
  if label=='extra_baseline':assert sha==C['checkpoint_sha256']['V1_257100']
  at=a['time'];moving=(at>=8.005)&(at<13.995);safe=(abs(at-8)>.025)&(abs(at-14)>.025);expected=np.zeros_like(a['command']);expected[moving]=cmd;assert np.allclose(a['command'][safe],expected[safe])
  assert (a['mode'][at>=6.06]=='policy').all()
  t=f['time'];xy=f['qpos'][:,:2];w,x,y,z=f['qpos'][:,3:7].T;yaw=np.unwrap(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)))
  assert np.isfinite(f['qpos']).all() and np.isfinite(a['action']).all()
  i=np.searchsorted(t,14.02);j=np.searchsorted(t,26);start=np.searchsorted(t,8.02)
  if i>=len(t) or j>=len(t):out[case]={'fell':m['fell_in_policy'],'duration_s':float(t[-1]),'invalid_stop_window':True};continue
  out[case]={'fell':m['fell_in_policy'],'duration_s':float(t[-1]),'stop_max_excursion_m':float(np.linalg.norm(xy[i:]-xy[i],axis=1).max()),'residual_last10_m_s':float(np.linalg.norm(np.diff(xy[j:],axis=0),axis=1).sum()/(t[-1]-t[j])),'stop_yaw_excursion_deg':float(np.rad2deg(abs(yaw[i:]-yaw[i]).max())),'prestop_translation_m':float(np.linalg.norm(xy[i]-xy[start])),'prestop_turn_deg':float(np.rad2deg(abs(yaw[i]-yaw[start])))}
 return out
base=calc('extra_baseline');candidate=calc('extra_round5');checks={}
for case,r in candidate.items():
 b=base[case]
 checks[case]={'no_fall':not r['fell'],'full_duration':r['duration_s']>=35.99}
 if 'invalid_stop_window' not in r:
  checks[case].update(stop_displacement=r['stop_max_excursion_m']<=A['max_stop_displacement_m'],residual_speed=r['residual_last10_m_s']<=A['max_last10_residual_speed_m_s'],stop_yaw=r['stop_yaw_excursion_deg']<=A['stationary_yaw_drift_deg'])
  field='prestop_turn_deg' if case=='turn' else 'prestop_translation_m'
  checks[case]['prestop_motion']=r[field]>=b.get(field,0)*.8 and r[field]>(10 if case=='turn' else .25)
 else:checks[case]['valid_stop_window']=False
result={'baseline':base,'candidate':candidate,'checks':checks,'passed':all(all(v.values()) for v in checks.values()),'note':'Additional flat-terrain stopping cases; same displacement/residual/yaw limits; movement >=80% baseline and nontrivial. No gait/turn-quality optimization claimed.'}
(B/'extra_verdict.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
