"""Headless physical execution of actual interactive_rollout Z/C/W path.
Only window drawing, keyboard source and wall-clock pacing are substituted.
"""
from pathlib import Path
from types import SimpleNamespace
from contextlib import nullcontext
import sys,json,hashlib
import numpy as np
sys.path.insert(0,str(Path('deploy/deploy_mujoco').resolve()))
import deploy_mujoco as dm
from quiet_landing_metrics import QuietMujocoMetrics
QuietMujocoMetrics.draw=lambda *args:None
OUT=Path(__file__).resolve().parent
contract=json.loads(Path('docs/quiet_landing_Baseline_V1_V1_5_多迭代全地形对比_20261011/正式对比冻结条件.json').read_text())
label,case=sys.argv[1:3]
state={};poses=[]
class Viewer:
 cam=SimpleNamespace()
 def lock(self):return nullcontext()
 def is_running(self):return True
 def close(self):pass
 def sync(self):
  data=state['data'];poses.append((data.time,data.xquat[state['base']].copy(),data.xpos[state['base']].copy()))
def launch(model,data,key_callback=None):
 state.update(data=data,base=model.body('base_link').id);return Viewer(),[]
class Keys:
 _window_keys=None
 def __init__(self):self.z=False;self.c=False
 def on_keycode(self,*args):pass
 def take_events(self):
  t=state['data'].time
  if not self.z:self.z=True;return ['z']
  if t>=6 and not self.c:self.c=True;return ['c']
  return []
 def velocity_command(self):return np.array([.7,0,0] if case=='仅w' and state['data'].time>=8 else [0,0,0],dtype=np.float32)
 def clear_motion(self):pass
 def close(self):pass
class Terminal:
 def __init__(self,*args):pass
 def start(self):return False
 def poll(self):pass
 def close(self):pass
dm.launch_viewer=launch;dm.draw_height_scan=lambda *args:None;dm.TeleopKeyboard=Keys;dm.TerminalInput=Terminal;dm.time.sleep=lambda *args:None
args=SimpleNamespace(checkpoint=Path(contract['checkpoints'][label]),model=Path(contract['model']),terrain=None,terrain_xml=Path(contract['terrain']),quiet_landing=True,live_plot=False,output=OUT/(label+'_'+case+'_原部署回放.npz'),steps=1800,print_every=500,stop_on_fall=True)
res=dm.interactive_rollout(args)
np.savez_compressed(OUT/(label+'_'+case+'_姿态.npz'),time=np.array([p[0] for p in poses]),quat=np.array([p[1] for p in poses]),pos=np.array([p[2] for p in poses]))
(OUT/(label+'_'+case+'_操作条件.json')).write_text(json.dumps({'checkpoint':str(args.checkpoint),'checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),'Z_s':0,'C_s':6,'W_s':8 if case=='仅w' else None,'command_body':[.7,0,0] if case=='仅w' else [0,0,0],'course_correction':False,'physics':True,'entry':'deploy_mujoco.interactive_rollout','headless_substitutions':['window draw','keyboard source','wall time pacing']},ensure_ascii=False,indent=2))
