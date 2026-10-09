from pathlib import Path
import sys,json
sys.path.insert(0,str(Path.cwd()/'scripts/tools'))
import torch,numpy as np
from check_stair_rewards import Entity,make_env,context,MDP
from check_gait_safety import load
ns=load(MDP/'stair_teacher.py');torch.set_num_threads(1)
wheels,scan,contact=Entity('robot'),Entity('height_scanner'),Entity('contact_forces')
r={}

def track():
    env=make_env()
    def step(side=None,x=None,z=None,touch=True,edge=0.,height=.1):
        env.common_step_counter+=1;env.episode_length_buf+=1
        if side is not None:
            env.scene['robot'].data.body_pos_w[0,side,0]=x
            env.scene['robot'].data.body_pos_w[0,side,2]=z
            env.scene['robot'].data.body_lin_vel_w[0,side,2]=0 if touch else .3
            env.scene['contact_forces'].data.current_contact_time[0,side]=float(touch)
            env.scene['contact_forces'].data.net_forces_w[0,side]=torch.tensor([0.,0.,100. if touch else 0.])
        context(env,edge,height)
        return ns['ascent_state'](env,wheels,scan,contact)
    def land(side,edge,height):
        step(side,edge-.12,env.scene['robot'].data.body_pos_w[0,side,2].item(),True,edge,height)
        step(side,edge-.04,height+.15,False,edge,height)
        step(side,edge+.08,height+.15,False,edge,height)
        step(side,edge+.10,height+.09,True,edge,height)
        return step(side,edge+.10,height+.09,True,edge,height)
    step()
    return env,step,land
# Front missed first target landing. Physical motion continues on a higher tread.
e,st,land=track();st(0,-.04,.35,False);st(0,.4,.35,False);st(0,.4,.29,True,edge=.3,height=.2)
for _ in range(100):s=st(0,.4,.29,True,edge=.3,height=.2)
r['front_missed_first_then_step2']={k:float(s[k]) for k in ('front_count','front_successes','active','age')}
# Front gives a real queue; rear skips first tread entirely, then lands step2.
e,st,land=track();land(0,0.,.1);land(1,.3,.2);land(0,.6,.3)
st(2,-.04,.35,False,edge=.9,height=.4);st(2,.4,.35,False,edge=.9,height=.4)
for _ in range(100):s=st(2,.4,.29,True,edge=.9,height=.4)
r['rear_skipped_first_then_step2']={k:float(s[k]) for k in ('front_count','rear_count','rear_successes','rear_target_age')}
# Direct counterexample of posture blind spot independent of leg capability.
params=dict(asset_cfg=wheels,hip_cfg=Entity('robot',body_ids=[4,5,6,7]),knee_cfg=Entity('robot',joint_ids=[0,1,2,3]),sensor_cfg=scan,contact_sensor_cfg=contact)
for low_samples in (False,True):
 e=make_env();w=e.scene['robot'].data.body_pos_w;h=w.clone();h[:,:,2]+=.3
 e.scene['robot'].data.body_pos_w=torch.cat((w,h),1);e.scene['robot'].data.joint_pos=torch.full((1,4),2.2)
 e.scene['robot'].data.root_ang_vel_b=torch.zeros(1,3)
 # Contact and pose identical. Shift sampled height below the physical tread by one step.
 e.scene['height_scanner'].data.ray_hits_w[:,:,2]=-.15 if low_samples else 0
 context(e)
 r[f'front_fold_low_samples_{low_samples}']=float(ns['ascent_front_fold_cost'](e,**params))
# Grace counterexample: positive force, severe fold, 0.02 second contact.
e=make_env();w=e.scene['robot'].data.body_pos_w;h=w.clone();h[:,:,2]+=.3
e.scene['robot'].data.body_pos_w=torch.cat((w,h),1);e.scene['robot'].data.joint_pos=torch.full((1,4),2.2)
e.scene['robot'].data.root_ang_vel_b=torch.zeros(1,3);e.scene['height_scanner'].data.ray_hits_w[:,:,2]=0
e.scene['contact_forces'].data.current_contact_time[:]=.02;context(e)
r['front_severe_new_contact_free']=float(ns['ascent_front_fold_cost'](e,**params))
print(json.dumps(r,indent=2));Path('.scratch/gait_v3_20261008/reviewer_baseline.json').write_text(json.dumps(r,indent=2)+'\n')
