"""Capture actual actor inputs without changing observation/physics/action values."""
from pathlib import Path
import sys,runpy
import numpy as np
sys.path.insert(0,str(Path('deploy/deploy_mujoco').resolve()))
import deploy_mujoco as dm
rows=[]
original=dm.M20Contract.observe
def observe(self,data,command,previous_action):
 value=original(self,data,command,previous_action)
 rows.append((float(data.time),value.copy()))
 return value
dm.M20Contract.observe=observe
B=Path(__file__).resolve().parent
runpy.run_path(str(B/'run_case.py'),run_name='__main__')
out=B/'evaluations'/sys.argv[2]/(sys.argv[3]+'_inputs.npz')
np.savez_compressed(out,time=np.array([r[0] for r in rows]),policy=np.stack([r[1] for r in rows]))
print('Captured unchanged actual actor inputs',len(rows),out)
