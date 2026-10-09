import numpy as np,json
from pathlib import Path
out={}
for tag in ('baseline','latest'):
 d=np.load(f'docs/gait_audit_20261009/{tag}_descent/descent.npz');q=d['root_quat'];v=d['wheel_pos']-d['hip_pos'];qi=np.repeat(q[:,None,:],4,axis=1).copy();qi[:,:,1:]*=-1
 t=2*np.cross(qi[:,:,1:],v);body=v+qi[:,:,:1]*t+np.cross(qi[:,:,1:],t)
 force=d['contact_force'];loaded=(force[:,:,2]>5)&(force[:,:,2]>.5*np.linalg.norm(force,axis=2));ground=d['wheel_ground']
 upper=float(ground[0].max());rearupper=np.abs(ground[:,2:]-upper)<.04
 frontdown=(ground[:,:2]<upper-.05)&loaded[:,:2]
 entry=frontdown.any(-1)[:,None]&rearupper&loaded[:,2:]&(d['root_pos'][:,0]<2.)[:,None]
 x=body[:,2:,0];depth=-body[:,2:,2];b=np.degrees(np.arctan2(x,np.maximum(depth,1e-6)))
 def stats(mask):
  return {'samples':int(mask.sum()),'x_quantiles_m':np.percentile(x[mask],[5,25,50,75,95]).tolist() if mask.any() else [],'depth_quantiles_m':np.percentile(depth[mask],[5,50,95]).tolist() if mask.any() else [],'b_quantiles_deg':np.percentile(b[mask],[5,50,95]).tolist() if mask.any() else []}
 out[tag]={'entry_loaded':stats(entry),'entry_moderate_b':stats(entry&(b<=25)&(b>=-10)), 'entry_forward_b':stats(entry&(b>25))}
 for i,leg in enumerate(('hl','hr')):
  mask=entry[:,i];out[tag][leg]={'samples':int(mask.sum()),'x_p50_m':float(np.median(x[:,i][mask])) if mask.any() else None}
Path('docs/descent_entry_20261009/position_analysis.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
