import sys,json
from pathlib import Path
sys.path.insert(0,'scripts/tools')
import torch
from check_gait_safety import load
from check_stair_rewards import MDP,make_env
ns=load(MDP/'stair_teacher.py');torch.set_num_threads(1)
e=make_env();scan={'edge_x':torch.tensor([0.]),'down_gate':torch.tensor([0.])}
w=e.scene['robot'].data.body_pos_w.clone();body=w.clone();body[...,0]=.3;body[...,2]=-.4
ground=torch.tensor([[-.15,-.15,0.,0.]])
valid=torch.ones(1,4,dtype=torch.bool);loaded=valid.clone();ct=torch.full((1,4),.1);velocity=torch.zeros(1,4,3);rootv=torch.tensor([[.5,0,0.]])
cmd=torch.tensor([[.5,0,0.]]);q=torch.tensor([[1.,0,0,0.]])
def tick(down=True):
 e.common_step_counter+=1;e.episode_length_buf+=1
 return ns['_descent_entry_retract_context'](e,scan,torch.tensor([down]),w,body,ground,valid,loaded,ct,velocity,rootv,q,cmd,torch.tensor([0.]))
e._m20_descent_pose_phase={'active':torch.tensor([True]),'xy':torch.tensor([[0.,0.]]),'heading':torch.tensor([[1.,0.]])}
first=tick();body[:,2:,0]=.1;gain1=float(tick()['gain'])
# Retreat and hold zero command on original flat top, which legally rearms.
ground.zero_();cmd.zero_();rootv.zero_()
for _ in range(26):tick(False)
ready=bool(e._m20_descent_entry_state['ready'])
# Same physical first edge, but no current forward-ray edge at rear transfer.
# Existing physical phase is assumed known; current scanner moved .3m.
e.scene['height_scanner'].data.pos_w[:,0]+=.3
body[:,2:,0]=.3;ground[:,:2]=-.15;cmd[:,0]=.5;rootv[:,0]=.5
tick();body[:,2:,0]=.1;gain2=float(tick()['gain'])
out={'first_credit':gain1,'ready_after_retreat_pause':ready,'second_credit_same_physical_entry':gain2,'starts':float(e._m20_descent_entry_state['started'])}
assert gain1>0 and gain2==0 and out['starts']==1,out
print(json.dumps(out,indent=2));Path('docs/descent_entry_20261009/reviewer_entry_probe.json').write_text(json.dumps(out,indent=2)+'\n')
e.episode_length_buf[:]=0;e._m20_descent_pose_phase['active'].zero_();r=tick();out['unknown_edge_gain']=float(r['gain']);out['unknown_edge_starts']=float(e._m20_descent_entry_state['started']);assert out['unknown_edge_gain']==0 and out['unknown_edge_starts']==0
print(json.dumps({'unknown_edge_gain':out['unknown_edge_gain'],'unknown_edge_starts':out['unknown_edge_starts']},indent=2))
Path('docs/descent_entry_20261009/reviewer_entry_probe.json').write_text(json.dumps(out,indent=2)+'\n')
