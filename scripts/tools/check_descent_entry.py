#!/usr/bin/env python3
"""Real reward bodies with synthetic entry counterexamples; no gait-success claim."""
from pathlib import Path
from types import SimpleNamespace
import argparse,json
import torch
from check_gait_safety import load
from check_stair_rewards import MDP

def check(ns):
    out={};n=2
    command=torch.tensor([[.5,0,0.]]*n);root_velocity=command.clone()
    wheels=torch.zeros(n,4,3);wheels[:,:,2]=.84
    delta=torch.zeros_like(wheels);delta[:,2:,0]=.30;delta[:,:,2]=-.40
    ground=torch.full((n,4),.75);valid=torch.ones(n,4,dtype=torch.bool)
    loaded=valid.clone();time=torch.full((n,4),.2);velocity=torch.zeros_like(wheels)
    quat=torch.tensor([[1.,0,0,0]]*n);relax=torch.zeros(n)
    scan=dict(down_gate=torch.ones(n),edge_x=torch.full((n,),.4))
    e=SimpleNamespace(num_envs=n,device='cpu',step_dt=.02,episode_length_buf=torch.tensor([2,2]),
        scene={'height_scanner':SimpleNamespace(data=SimpleNamespace(pos_w=torch.zeros(n,3)))})
    def tick(down=True):
        e.episode_length_buf+=1
        return ns['_descent_entry_retract_context'](e,scan,torch.full((n,),down),wheels,delta,ground,
            valid,loaded,time,velocity,root_velocity,quat,command,relax)
    first=tick();assert torch.equal(first['gain'],torch.zeros(n));out['initial_pose_unpaid']=True
    delta[:,2:,0]=.15;r=tick();out['toward_band_gain']=r['gain'].tolist();assert (r['gain']>0).all()
    assert torch.equal(tick()['gain'],torch.zeros(n))
    delta[:,2:,0]=.30;tick();delta[:,2:,0]=.15
    assert torch.equal(tick()['gain'],torch.zeros(n));out['oscillation_unpaid']=True
    delta[:,2:,0]=-.20;r=tick();out['overrear_cost']=r['cost'].tolist();assert (r['cost']>0).all() and not r['gain'].any()
    # Reset only one env: history and first score must restart only there.
    e.episode_length_buf=torch.tensor([0,10]);delta[:,2:,0]=.3;tick()
    assert e._m20_descent_entry_state['count'].tolist()==[1,1]
    # Stopped/backward movement and reverse rolling cannot earn or bank progress.
    delta[:,2:,0]=.10;root_velocity[0,0]=0;r=tick();assert r['gain'][0]==0
    root_velocity[0,0]=.5;assert tick()['gain'][0]==0
    out['stopped_progress_not_banked']=True
    e.episode_length_buf[:]=0;delta[:,2:,0]=.30;tick()
    root_velocity[:,0]=-.1;delta[:,2:,0]=.1;assert not tick()['gain'].any()
    root_velocity[:,0]=.5;assert not tick()['gain'].any();out['reverse_trunk_unpaid']=True
    e.episode_length_buf[:]=0;delta[:,2:,0]=.3;tick()
    velocity[:,2:,0]=-.2;delta[:,2:,0]=.1;assert not tick()['gain'].any()
    velocity.zero_();assert not tick()['gain'].any();out['reverse_wheel_unpaid']=True
    # Descend the original top: bonus ends rather than following every riser.
    ground[:,:]=.6;wheels[:,:,2]=.69;r=tick();assert not e._m20_descent_entry_state['active'].any()
    assert not r['gain'].any();assert not tick()['gain'].any();out['following_stair_not_new_entry']=True
    # A flat landing may rearm; returning to same entry even at a lateral offset cannot farm it.
    scan['down_gate'].zero_()
    for _ in range(30):tick(False)
    assert e._m20_descent_entry_state['ready'].all()
    ground.fill_(.75);scan['down_gate'].fill_(1);e.scene['height_scanner'].data.pos_w[:,1]=.4
    assert not tick()['gain'].any();assert e._m20_descent_entry_state['count'].tolist()==[1,1]
    out['lateral_reentry_unpaid']=True
    # Retargeting bonus cannot be earned at rest even if support switches on.
    e.episode_length_buf[:]=0;e.scene['height_scanner'].data.pos_w.zero_();ground.fill_(.75)
    delta[:,2:,0]=.3;tick();loaded[:,2:]=False;delta[:,2:,0]=.1;tick()
    loaded[:,2:]=True;assert not tick()['gain'].any();out['support_switch_unpaid']=True
    e.episode_length_buf[:]=0;ground.fill_(.75);delta[:,2:,0]=.30;delta[:,2:,2]=-.40;tick()
    delta[:,2:,0]=.10;delta[:,2:,2]=-.20
    assert not tick()['gain'].any();out['collapsed_leg_unpaid']=True
    delta[:,2:,2]=-.40
    assert (tick()['gain']<.05).all()
    delta[:,2:,2]=-.20;tick();delta[:,2:,2]=-.40
    assert not tick()['gain'].any();out['extension_cycle_unpaid']=True
    lo,hi=ns['descent_entry_position_band'](torch.tensor([[.35,.45]]),torch.tensor([0.]))
    out['band_depth35_45cm']={'lower':lo.tolist(),'upper':hi.tolist()}
    assert lo.min()>=-.051 and hi.max()<=.151
    # A good initial entry posture must still be penalized if it later leans
    # forward. Keep the transfer until actual lower support, not a lower ray.
    e.episode_length_buf[:]=0;valid.fill_(True);loaded.fill_(True);time.fill_(.2)
    ground.fill_(.75);wheels[:,:,2]=.84;delta[:,:,2]=-.40;delta[:,2:,0]=.10
    scan['down_gate'].fill_(1);root_velocity[:,0]=.5
    tick();delta[:,2:,0]=.32;r=tick()
    assert not r['gain'].any() and (r['cost']>0).all();out['good_entry_then_bad_is_penalized']=r['cost'].tolist()
    ground[:,2:]=.60;loaded[:,2:]=False;time[:,2:]=0
    r=tick();assert e._m20_descent_entry_state['active'].all() and (r['cost']>0).all()
    out['lower_ray_airborne_guard_retained']=True
    assert not tick(False)['cost'].any() and e._m20_descent_entry_state['active'].all()
    assert (tick(True)['cost']>0).all();out['command_pause_resumes_guard']=True
    # One stable lower rear wheel is insufficient; both must be truly settled.
    loaded[:,2]=True;time[:,2]=.2;wheels[:,2,2]=.69
    tick();assert e._m20_descent_entry_state['active'].all()
    loaded[:,3]=True;time[:,3]=.2;wheels[:,3,2]=.84
    tick();assert e._m20_descent_entry_state['active'].all()  # ray lower but foot above it
    wheels[:,3,2]=.69;r=tick();assert not e._m20_descent_entry_state['active'].any() and not r['cost'].any()
    out['both_true_lower_support_clear_guard']=True
    assert e._m20_descent_entry_state['settled'].all()
    e.episode_length_buf[:]=0;ground.fill_(.75);wheels[:,:,2]=.84;loaded.fill_(True);time.fill_(.2)
    delta[:,:,2]=-.40;delta[:,2:,0]=.10;tick()
    loaded[:,2:]=False;time[:,2:]=0;delta[:,2:,0]=.05;delta[:,2:,2]=-.15
    r=tick();assert (r['cost']>0).all();out['compact_swing_cannot_escape_band']=r['cost'].tolist()
    delta[:,2:,2]=-.35;r=tick();assert not r['cost'].any();out['healthy_swing_in_band_unpenalized']=True
    # New cost stays finite, zero for negative excess, stronger for large errors.
    values=ns['descent_forward_excess'](torch.tensor([-2.,0.,.5,1.,2.,3.,100.]))
    assert values[0]==values[1]==0 and torch.isfinite(values).all()
    assert (values[2:]<ns['TUNING']['descent_cost_soft_cap']).all()
    assert (values[3:]>values[2:-1]).all()
    out['descent_huber_cost']=values.tolist()
    # Nonfinite rays do not start a new detector.
    e.episode_length_buf[:]=0;valid.zero_();r=tick();assert not e._m20_descent_entry_state['active'].any() and not r['gain'].any()
    out['invalid_map_closed']=True
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(1);result=check(load(MDP/'stair_teacher.py'))
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
