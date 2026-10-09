import subprocess,os,time,json
from pathlib import Path
root=Path.cwd(); out=root/'.scratch/gait_quality_20261007'; started=time.monotonic(); records=[]
for label in ('latest','initial','intermediate'):
    for case in ('up30','up20','down30'):
        if time.monotonic()-started>600: raise RuntimeError('Global time budget exceeded')
        cmd=['/home/ubuntu/miniconda3/envs/m20_wzh/bin/python',str(out/'measure.py'),'--label',label,'--case',case]
        if label=='latest' and case in ('up30','down30'): cmd.append('--video')
        with (out/f'{label}_{case}.log').open('w') as f:
            r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=120)
        records.append(dict(label=label,case=case,rc=r.returncode));print(records[-1],flush=True)
        (out/'execution.json').write_text(json.dumps(records,indent=2)+'\n')
        if r.returncode: raise RuntimeError('Measurement failed; inspect log')
