from pathlib import Path
from array import array
import json,hashlib,datetime
import numpy as np
from tensorboard.backend.event_processing.event_file_loader import LegacyEventFileLoader
OUT=Path(__file__).resolve().parent
runs={'V1':Path('logs/rsl_rl/deeprobotics_m20_quiet_landing_v1/2026-10-10_00-48-58_quiet_landing_v1'),'V1.5':Path('logs/rsl_rl/deeprobotics_m20_quiet_landing_v1_5/2026-10-10_17-55-27_quiet_landing_v1_5_from_baseline')}
manifest={}
for label,run in runs.items():
 p=next(run.glob('events.out*'));limit=p.stat().st_size;snapshot=OUT/(label+'_读取时冻结事件文件.tmp');sha=hashlib.sha256()
 with p.open('rb') as src,snapshot.open('wb') as dst:
  left=limit
  while left:
   b=src.read(min(left,1024*1024))
   if not b:break
   dst.write(b);sha.update(b);left-=len(b)
 values={};events=0
 for e in LegacyEventFileLoader(str(snapshot)).Load():
  events+=1
  for v in e.summary.value:
   if v.WhichOneof('value')=='simple_value':values.setdefault(v.tag,array('d')).extend([e.step,e.wall_time,v.simple_value])
 data={k:np.asarray(v).reshape(-1,3) for k,v in values.items()}
 np.savez_compressed(OUT/(label+'_全部标量.npz'),**data)
 summary={k:{'n':len(v),'first_step':int(v[0,0]),'last_step':int(v[-1,0]),'last_value':float(v[-1,2])} for k,v in data.items()}
 (OUT/(label+'_标量目录.json')).write_text(json.dumps(summary,ensure_ascii=False,indent=2))
 manifest[label]={'path':str(p),'frozen_bytes':limit,'prefix_sha256':sha.hexdigest(),'parsed_events':events,'scalar_tags':len(data),'step_min':int(min(v[0,0] for k,v in data.items() if not k.endswith('/time'))),'step_max':int(max(v[-1,0] for k,v in data.items() if not k.endswith('/time'))),'snapshot_time':datetime.datetime.now().isoformat()}
 snapshot.unlink()
 print(label,manifest[label],flush=True)
 print(json.dumps({k:v for k,v in summary.items() if k.startswith('QuietLoad/stable')},indent=2),flush=True)
(OUT/'日志冻结来源.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
