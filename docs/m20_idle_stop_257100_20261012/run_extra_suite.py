from pathlib import Path
import subprocess,sys,os
B=Path(__file__).resolve().parent;cp,label=sys.argv[1:3]
env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':''}
for case in ['slow','reverse','turn','lateral']:
 out=B/'evaluations'/label;out.mkdir(parents=True,exist_ok=True)
 with (out/(case+'.log')).open('w') as f:
  r=subprocess.run(['/usr/bin/time','-v','timeout','180',sys.executable,str(B/'run_extra.py'),cp,label,case],cwd=B.parents[1],env=env,stdout=f,stderr=subprocess.STDOUT)
 if r.returncode:raise RuntimeError((case,r.returncode))
 print('DONE',label,case,flush=True)
