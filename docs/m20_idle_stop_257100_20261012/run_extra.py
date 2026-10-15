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
BASE=Path(__file__).resolve().parent
checkpoint,label,scenario=sys.argv[1:4]
case=scenario
SCENARIOS={'slow':(.35,0,0),'reverse':(-.7,0,0),'turn':(0,0,.6),'lateral':(0,.5,0)}
assert scenario in SCENARIOS
move_command=SCENARIOS[scenario]
OUT=BASE/'evaluations'/label
OUT.mkdir(parents=True,exist_ok=True)
contract=json.loads((BASE/'criteria.json').read_text())
contract['checkpoints']={label:checkpoint}

source_hashes={}
frozen_sources={**contract['source_sha256'],**json.loads((BASE/'instrument_sources.json').read_text())['source_sha256']}
for name,expected in frozen_sources.items():
 actual=hashlib.sha256(Path(name).read_bytes()).hexdigest()
 assert actual==expected, f'frozen source changed: {name}'
 source_hashes[name]=actual
cp_hash=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()

assert label in contract["checkpoints"], label
state={};poses=[];full=[]
class Viewer:
 cam=SimpleNamespace()
 def lock(self):return nullcontext()
 def is_running(self):return True
 def close(self):pass
 def sync(self):
  data=state['data'];full.append((data.time,data.qpos.copy(),data.qvel.copy()));poses.append((data.time,data.xquat[state['base']].copy(),data.xpos[state['base']].copy()))
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
 def velocity_command(self):return np.array(move_command if 8<=state['data'].time<14 else [0,0,0],dtype=np.float32)
 def clear_motion(self):pass
 def close(self):pass
class Terminal:
 def __init__(self,*args):pass
 def start(self):return False
 def poll(self):pass
 def close(self):pass
dm.launch_viewer=launch;dm.draw_height_scan=lambda *args:None;dm.TeleopKeyboard=Keys;dm.TerminalInput=Terminal;dm.time.sleep=lambda *args:None
args=SimpleNamespace(checkpoint=Path(contract['checkpoints'][label]),model=Path(contract['model']),terrain=None,terrain_xml=Path(contract['flat_terrain']),quiet_landing=True,live_plot=False,output=OUT/(label+'_'+case+'_原部署回放.npz'),steps=1800,print_every=500,stop_on_fall=True)
res=dm.interactive_rollout(args)
np.savez_compressed(OUT/(case+'_full.npz'),time=np.array([p[0] for p in full]),qpos=np.array([p[1] for p in full]),qvel=np.array([p[2] for p in full]))
np.savez_compressed(OUT/(label+'_'+case+'_姿态.npz'),time=np.array([p[0] for p in poses]),quat=np.array([p[1] for p in poses]),pos=np.array([p[2] for p in poses]))
(OUT/(label+'_'+case+'_操作条件.json')).write_text(json.dumps({'label':label,'scenario':scenario,'checkpoint':str(args.checkpoint.resolve()),'checkpoint_sha256':cp_hash,'source_sha256':source_hashes,'terrain_xml':str(args.terrain_xml.resolve()),'command_schedule':[[0,[0,0,0]],[8,move_command],[14,[0,0,0]]],'course_correction':False,'physics':True,'entry':'deploy_mujoco.interactive_rollout'},ensure_ascii=False,indent=2))
