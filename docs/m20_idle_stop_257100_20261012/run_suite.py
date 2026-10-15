from pathlib import Path
import os,sys,subprocess
B=Path(__file__).resolve().parent
cp,label=sys.argv[1:3]
entry='collect_inputs.py' if '--capture' in sys.argv[3:] else 'run_case.py'
root=B.parents[1]
env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':''}
for case in ['零命令','平地启停','仅w']:
 out=B/'evaluations'/label;out.mkdir(parents=True,exist_ok=True)
 with (out/(case+'.log')).open('w') as log:
  result=subprocess.run(['/usr/bin/time','-v','timeout','180',sys.executable,str(B/entry),cp,label,case],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
 if result.returncode:raise RuntimeError(f'{label}/{case} failed {result.returncode}; see {out/(case+".log")}')
 print('DONE',label,case,flush=True)
