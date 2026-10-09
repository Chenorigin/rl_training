import subprocess,os,json,time
from pathlib import Path
root=Path.cwd(); started=time.monotonic(); out=[]
env=os.environ.copy(); env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
for label in ('latest','baseline'):
    for case in ('xml20','up30','down30','turn_q','turn_e'):
        if label=='baseline' and case=='xml20': continue
        assert time.monotonic()-started<600, 'global budget exceeded'
        log=root/'.scratch/latest_gait_20261008'/f'{label}_{case}.log'
        t=time.monotonic()
        with log.open('w') as f:
            try:
                cp=subprocess.run(['/home/ubuntu/miniconda3/envs/m20_wzh/bin/python',str(root/'.scratch/latest_gait_20261008/measure.py'),'--label',label,'--case',case],stdout=f,stderr=subprocess.STDOUT,env=env,timeout=120)
                code=cp.returncode
            except subprocess.TimeoutExpired: code='timeout'
        rec={'label':label,'case':case,'exit':code,'elapsed_s':time.monotonic()-t}; out.append(rec)
        (root/'docs/latest_gait_20261008/execution.json').write_text(json.dumps(out,indent=2))
        print(json.dumps(rec),flush=True)
        if code!=0: print(log.read_text()[-2500:],flush=True)
