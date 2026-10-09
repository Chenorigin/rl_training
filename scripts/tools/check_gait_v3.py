#!/usr/bin/env python3
"""Adversarial production reward checks; these do not certify a learned gait."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from check_stair_rewards import Entity, MDP, context, load_functions, make_env
from check_gait_safety import load


def check(ns):
    ns['track_lin_vel_xy_exp']=lambda *a:torch.ones(a[0].num_envs)
    load_functions(Path(ns['ascent_state'].__code__.co_filename),ns)
    p = dict(asset_cfg=Entity('robot'),sensor_cfg=Entity('height_scanner'),contact_sensor_cfg=Entity('contact_forces'))
    pose_params = dict(p,hip_cfg=Entity('robot',body_ids=[4,5,6,7]),knee_cfg=Entity('robot',joint_ids=[0,1,2,3]))
    out = {}
    def make(steps=8):
        e=make_env(); e.step_dt=.02
        r=e.scene['robot'].data
        r.root_lin_vel_w=torch.zeros(1,3); r.root_ang_vel_b=torch.zeros(1,3)
        r.joint_pos=torch.ones(1,4)
        hips=r.body_pos_w.clone();hips[...,2]+=.45;r.body_pos_w=torch.cat((r.body_pos_w,hips),1)
        def tick(side=None,x=None,z=None,touch=True,force=None,visible=True):
            e.common_step_counter+=1;e.episode_length_buf+=1
            if side is not None:
                r.body_pos_w[0,side,0]=x;r.body_pos_w[0,side,2]=z
                e.scene['contact_forces'].data.current_contact_time[0,side]=.02 if touch else 0
                e.scene['contact_forces'].data.net_forces_w[0,side]=torch.tensor(force or [0,0,100 if touch else 0.])
            r.root_pos_w[0,0]=r.body_pos_w[0,:2,0].mean()-.35
            sc=e.scene['height_scanner'].data;sc.pos_w[0,0]=r.root_pos_w[0,0]
            xs=torch.arange(steps,dtype=torch.float32)*.3-sc.pos_w[0,0]
            valid=(xs>=-.45)&(xs<=.70)&visible
            context(e)
            c=e._m20_stair_scan_context['context'];c.update(ascending_edges_x=xs,
                ascending_edges_valid=valid[None],ascending_edges_upper_z=(torch.arange(steps)+1)[None]*.1,
                ascending_edges_lower_z=torch.arange(steps)[None]*.1)
            has=valid & (xs>=.05)
            idx=int(torch.nonzero(has)[0]) if has.any() else 0
            c['edge_x']=xs[idx:idx+1];c['lower_z']=torch.tensor([.1*idx]);c['upper_z']=torch.tensor([.1*(idx+1)])
            c['up_gate']=torch.tensor([float(has.any())]);c['gate']=c['up_gate'];c['terrain_gate']=torch.tensor([float(valid.any())])
            return ns['ascent_state'](e,**p)
        def land(side,i):
            x=.3*i;z=.1*(i+1)
            tick(side,x-.12,r.body_pos_w[0,side,2].item(),True)
            tick(side,x-.06,z+.16,False);tick(side,x+.08,z+.16,False)
            tick(side,x+.10,z+.09,True);return tick(side,x+.10,z+.09,True)
        tick();return e,tick,land
    # Both lead choices, sustained beyond the first two risers.
    for first in (0,1):
        e,tick,land=make()
        for axle in (0,1):
            name=('front','rear')[axle]
            completion=ns['stair_up_step_completion' if axle==0 else 'stair_up_rear_step_completion']
            actual_paid=0.0
            for i in range(8):
                s=land(2*axle+(first if i%2==0 else 1-first),i)
                units=float(completion(e,**p)*e.step_dt)
                assert units==(0 if i==0 else 1), (axle,i,units)
                actual_paid+=3*units
            assert s[name+'_transition_successes']==7 and actual_paid==21
            out[f'lead{first}_{name}_transition_payments']=float(s[name+'_transition_successes'])
            out[f'lead{first}_{name}_weighted_completion_return']=actual_paid
        s=tick();out[f'lead{first}_front_successes']=float(s['front_successes']);out[f'lead{first}_rear_successes']=float(s['rear_successes'])
        assert s['front_successes']==s['rear_successes']==8
        assert s['same_tread_events_total'].sum()==0
        m=ns['stair_step_completion_metrics'](e,slice(None))
        assert m['up_front_longest_strict_run']==m['up_rear_longest_strict_run']==7
        for name in ('front','rear'):
            assert m[f'up_{name}_skip_fraction']==0 and m[f'up_{name}_same_tread_fraction']==0
        out[f'lead{first}_six_step_fraction']=float(m['up_rear_six_step_strict_fraction'])
    # Rear reward uses discovered geometry even with no front completion.
    e,tick,land=make()
    # The trunk/front pair has moved ahead, exposing the next rear target,
    # without any front stable/completion credit. Geometry must remain usable.
    e.scene['robot'].data.body_pos_w[0,:2,0]=.6
    e.scene['robot'].data.body_pos_w[0,:2,2]=.69
    land(2,0);s=land(3,1)
    out['rear_success_without_front']=float(s['rear_successes']);assert s['front_successes']==0 and s['rear_successes']==2
    assert s['rear_transition_successes']==1 and s['rear_event'].any()
    # User-observed failure: switching sides alone is insufficient. FL1->FR3
    # (and HL1->HR3) skips tread2 even though the leg changes.
    for axle in (0,1):
        e,tick,land=make()
        if axle:
            e.scene['robot'].data.body_pos_w[0,:2,0]=.6
            e.scene['robot'].data.body_pos_w[0,:2,2]=.69
        land(2*axle,0);s=land(2*axle+1,2)
        name=('front','rear')[axle]
        target=int(s['axis_target_slot'][0,axle])
        m=ns['stair_step_completion_metrics'](e,slice(None))
        out[f'{name}_1_to_3']={'first_landing_credits':float(s[name+'_successes']),
            'skips':float(s['skip_total'][0,axle]),
            'skipped_landing_event_credit':bool(s[name+'_event'].any()),
            'skip_fraction':float(m[f'up_{name}_skip_fraction']),
            'next_target_height':float(s['map_z'][0,target]) if target>=0 else None}
        assert s[name+'_successes']==1 and s['skip_total'][0,axle]==1 and not s[name+'_event'].any()
        assert s[name+'_transition_successes']==0
        assert m[f'up_{name}_skip_fraction']==.5
        s=land(2*axle,3)
        assert s[name+'_successes']==2  # Next adjacent step remains learnable.
        assert s[name+'_transition_successes']==1 and s[name+'_event'].any()
    # Full user-observed front failure: FL1 -> FR3 -> FL3. The skip and
    # subsequent same-tread catch-up are two errors in the same trajectory.
    e,tick,land=make()
    # Keep tread2 inside the finite scan after FL1, so its target is observable.
    e.scene['robot'].data.body_pos_w[0,1,0]=-.1
    s=land(0,0)
    target=int(s['axis_target_slot'][0,0])
    assert target>=0 and abs(float(s['map_z'][0,target])-.2)<1e-5
    assert s['front_expected']==1 and s['front_transition_successes']==0
    s=land(1,2)
    assert s['skip_total'][0,0]==1 and not s['front_event'].any()
    s=land(0,2)
    units=float(ns['stair_up_same_tread_cost'](e,**p)*e.step_dt)
    assert s['same_tread_events_total'][0,0]==1 and units==1
    assert not s['front_event'].any() and s['front_transition_successes']==0
    m=ns['stair_step_completion_metrics'](e,slice(None))
    out['front_skip_then_same_tread']={
        'sequence':['FL1','FR3','FL3'],
        'target_after_FL1':{'leg':'FR','tread_height':.2},
        'skip_events':float(s['skip_total'][0,0]),
        'same_tread_events':float(s['same_tread_events_total'][0,0]),
        'paid_transitions':float(s['front_transition_successes']),
        'same_tread_penalty_units':units,
        'same_tread_fraction':float(m['up_front_same_tread_fraction'])}
    # Do not force a retreat to missed tread2 after physical progress to tread3.
    # Tread4 becomes an unpaid seed; tread5 can then prove a new legal transition.
    s=land(1,3);assert not s['front_event'].any()
    s=land(0,4);assert s['front_event'].any() and s['front_transition_successes']==1
    out['front_skip_then_same_tread']['recovered_at_tread5']=True
    # Same-tread chasing must receive no primary transition payments, including
    # before the partner lands and any later penalty is discounted by PPO.
    e,tick,land=make()
    e.scene['robot'].data.body_pos_w[0,:2,0]=.6
    e.scene['robot'].data.body_pos_w[0,:2,2]=.69
    for i in range(3):
        s=land(2,i)
        assert not s['rear_event'].any()
        s=land(3,i)
        assert not s['rear_event'].any()
    m=ns['stair_step_completion_metrics'](e,slice(None))
    out['rear_both_feet_every_tread']={'duplicates':float(s['same_tread_events_total'][0,1]),
        'longest_strict_run':float(m['up_rear_longest_strict_run']),
        'strict_coverage':float(m['up_rear_strict_coverage']),
        'same_tread_fraction':float(m['up_rear_same_tread_fraction']),
        'visited_intermediate_risers':float(m['up_rear_visited_intermediate_risers']),
        'qualified_first_landings':float(s['rear_successes']),
        'paid_transitions':float(s['rear_transition_successes']),
        'completion_plus_duplicate_return':float(3*s['rear_transition_successes']+
            ns['TUNING']['same_tread_event_weight']*s['same_tread_events_total'][0,1])}
    assert s['same_tread_events_total'][0,1]==3 and m['up_rear_longest_strict_run']==0
    assert m['up_rear_strict_coverage']==0 and out['rear_both_feet_every_tread']['completion_plus_duplicate_return']<0
    assert m['up_rear_same_tread_fraction']==1 and m['up_rear_visited_intermediate_risers']==3
    assert s['rear_transition_successes']==0
    # After a duplicate, the next single safe tread is an unpaid seed; its
    # adjacent opposite-side successor can earn again instead of staying stuck.
    # Move the trunk's scan forward so the later risers are actually observed.
    e.scene['robot'].data.body_pos_w[0,:2,0]=1.2
    e.scene['robot'].data.body_pos_w[0,:2,2]=.89
    s=land(2,3);assert not s['rear_event'].any()
    s=land(3,4)
    assert s['rear_event'].any() and s['rear_transition_successes']==1, {
        key:s[key].tolist() for key in ('rear_successes','rear_transition_successes',
            'rear_violations','skip_total','axis_last_slot','map_count')}
    out['rear_recovered_after_duplicate']=True
    # One rear leg repeatedly advances before its partner catches up: changing
    # tread without changing side must not receive the next completion credit.
    e,tick,land=make()
    e.scene['robot'].data.body_pos_w[0,:2,0]=.6
    e.scene['robot'].data.body_pos_w[0,:2,2]=.69
    land(2,0);s=land(2,1)
    out['rear_same_leg_consecutive']={'credits':float(s['rear_successes']),
        'violations':float(s['rear_violations'])}
    assert s['rear_successes']==1 and s['rear_violations']==1 and not s['rear_event'].any()
    assert s['rear_transition_successes']==0
    e,tick,land=make();s=tick(visible=False)
    ns['_reset'](s,torch.tensor([True]))
    m=ns['stair_step_completion_metrics'](e,slice(None))
    for name in ('front','rear'):
        assert m[f'up_{name}_skip_fraction']==m[f'up_{name}_same_tread_fraction']==0
        assert m[f'up_{name}_visited_intermediate_risers']==0
        assert m[f'up_{name}_transition_successes']==0
    out['zero_samples_not_success']=True
    # Miss a riser, charge a violation, then recover next local alternating step.
    for axle in (0,1):
        e,tick,land=make()
        if axle:
            for i in range(5):land(i%2,i)
        s=land(2*axle,1);before=float(s[('front','rear')[axle]+'_successes'])
        assert before==0 and s['skip_total'][0,axle]>0
        s=land(2*axle+1,2)
        out[f'axle{axle}_recovered_after_skip']=float(s[('front','rear')[axle]+'_successes'])
        assert s[('front','rear')[axle]+'_successes']==1
    # Delayed opposite landing after first leg leaves; bounce cannot pay again.
    for axle in (0,1):
        e,tick,land=make()
        for i in range(4):land(i%2,i)
        if axle:
            land(2,0);land(3,1)
        side=2*axle
        tick(side,1.1,.70,False)
        for _ in range(3):s=tick(side+1,.1,.19,True)
        out[f'axle{axle}_delayed_duplicate']=float(s['same_tread_events_total'][0,axle]);assert s['same_tread_events_total'][0,axle]==1
        tick(side+1,.1,.4,False)
        for _ in range(3):s=tick(side+1,.1,.19,True)
        assert s['same_tread_events_total'][0,axle]==1
    # Strong classification test: neither axle first landing need be a success.
    e,tick,land=make();land(0,0);land(0,1);tick(0,.9,.5,False)
    for _ in range(3):s=tick(1,.4,.29,True)
    assert s['front_successes']==1 and s['same_tread_events_total'][0,0]==1
    # A duplicate gets no transition/touchdown/alternation credit. Compare
    # actual preparation plus duplicate cost, not an imaginary paid landing.
    # Whole-body task completion is deliberately a separate positive objective.
    nominal_good=3+ns['TUNING']['prep_weight']+ns['TUNING']['front_touchdown_weight']+.25
    duplicate_net=ns['TUNING']['prep_weight']+ns['TUNING']['same_tread_event_weight']
    out['event_good_vs_duplicate_net']=[nominal_good,duplicate_net];assert nominal_good>0>duplicate_net
    # Both axle events add; actual new ledger counters are the source.
    s['duplicate_event_count'].fill_(1);e._m20_ascent_cache=dict(step=e.common_step_counter,state=s)
    units=float(ns['stair_up_same_tread_cost'](e,**p)*e.step_dt);out['both_axles_units']=units;assert units==2
    # Final platform is exempt, current tread dwell persists across lift/pause.
    e,tick,land=make(steps=1);land(0,0)
    for _ in range(3):s=tick(1,.1,.19,True)
    assert s['same_tread_events_total'].sum()==0
    e,tick,land=make();land(0,0)
    for _ in range(12):s=tick(1,.1,.19,True)
    assert s['same_tread_dwell'][0,0]>0
    tick(1,.1,.4,False)
    for _ in range(3):s=tick(1,.1,.19,True)
    assert s['same_tread_dwell'][0,0]>0
    credit=float(s['front_lift_credit'])
    for z in (.2,.09,.2,.09):tick(1,-.1,z,False)
    assert float(s['front_lift_credit'])-credit<=1+float(s['front_successes'])-credit+1e-5
    before=s['map_stable_time'].clone();assert ns['ascent_state'](e,**p) is s and torch.equal(before,s['map_stable_time'])
    ns['_reset'](s,torch.tensor([True]));assert not s['map_landed'].any() and s['map_count']==0 and torch.isinf(s['map_next_x']).all()
    out['lifecycle']=True
    # Single-frame/face brush never counts a landed wheel.
    e,tick,land=make();tick(0,-.06,.26,False);tick(0,.08,.26,False)
    s=tick(0,.1,.19,True);assert s['front_successes']==0
    s=tick(0,.1,.19,True,force=[100,0,1]);assert s['front_successes']==0
    # Real riser may lie anywhere inside the two-ray bracket. An initially
    # low midpoint followed by a clear far-side lift must still earn credit.
    e,tick,land=make();tick(0,-.07,.16,False)
    tick(0,.02,.23,False);tick(0,.10,.19,True);s=tick(0,.10,.19,True)
    out['quantized_riser_clear_credit']=float(s['front_successes']);assert s['front_successes']==1
    # Stable contact inside the uncertain ray interval must not retire the
    # tread before its completed crossing can be observed.
    e,tick,land=make();tick(0,-.04,.26,False)
    tick(0,.03,.19,True);s=tick(0,.03,.19,True)
    assert s['front_count']==0 and s['front_violations']==0
    tick(0,.10,.19,True);s=tick(0,.10,.19,True)
    out['early_contact_then_cross_credit']=float(s['front_successes']);assert s['front_successes']==1
    # Simply scraping up a riser with no observed clearance earns no style.
    e,tick,land=make();tick(0,-.07,.12,False)
    tick(0,.10,.19,True);s=tick(0,.10,.19,True)
    out['uncleared_riser_style_credit']=float(s['front_successes']);assert s['front_successes']==0
    # Pause, reverse and pure-yaw commands close style gates without erasing
    # a paid tread or renewing its preparation budget.
    e,tick,land=make();land(0,0)
    saved_paid=e._m20_ascent_state['map_paid'].clone()
    credit=float(e._m20_ascent_state['front_lift_credit'])
    for cmd in ((0,0,0),(-.5,0,0),(0,0,.6)):
        e.command_manager.get_command('base_velocity')[:]=torch.tensor([cmd])
        s=tick()
        assert not s['active'].any() and s['front_lift_gain']==0
        assert torch.equal(s['map_paid'],saved_paid) and float(s['front_lift_credit'])==credit
    out['command_pause_preserves_ledger']=bool(saved_paid.any())
    # Posture: maps may be wrong locally, but genuine bearing load still counts.
    def pose(a=110,down=False,swing=False,invalid=False):
        e=make_env();r=e.scene['robot'].data;hips=r.body_pos_w.clone();hips[...,2]+=.45
        r.body_pos_w=torch.cat((r.body_pos_w,hips),1);r.joint_pos=torch.full((1,4),np.deg2rad(180-a),dtype=torch.float32)
        r.root_ang_vel_b=torch.zeros(1,3);r.root_lin_vel_w=torch.zeros(1,3)
        context(e);c=e._m20_stair_scan_context['context'];c['up_gate'].fill_(float(not down));c['down_gate'].fill_(float(down))
        if invalid:e.scene['height_scanner'].data.ray_hits_w.fill_(torch.inf)
        if swing:
            e.scene['contact_forces'].data.net_forces_w[:,:2]=0;e.scene['contact_forces'].data.current_contact_time[:,:2]=0
        else:e.scene['contact_forces'].data.current_contact_time.fill_(.02)
        return e
    for phase in ('loaded','swing'):
        costs=[]
        for a in (110,90,80,60,40,20):
            cost=float(ns['ascent_front_fold_cost'](pose(a,swing=phase=='swing'),**pose_params));costs.append(cost)
        assert all(x<=y for x,y in zip(costs,costs[1:])) and costs[-1]>costs[-2]>costs[0]
        out[f'front_{phase}_monotonic']=costs
    e=pose(40,invalid=True);out['loaded_bad_map_cost']=float(ns['ascent_front_fold_cost'](e,**pose_params));assert out['loaded_bad_map_cost']>0
    e=pose(40);assert ns['ascent_front_fold_cost'](e,**pose_params)>0
    r=e.scene['robot'].data;sc=e.scene['height_scanner'].data
    old_map_count=int(e._m20_ascent_state['map_count'])
    r.body_pos_w[:,:,1]+=3;r.root_pos_w[:,1]+=3
    sc.pos_w[:,1]+=3;sc.ray_hits_w[:,:,1]+=3;sc.ray_hits_w[:,:,2]=0
    e.common_step_counter+=1;e.episode_length_buf+=1
    out['flat_after_lateral_exit_cost']=float(ns['ascent_front_fold_cost'](e,**pose_params))
    assert out['flat_after_lateral_exit_cost']==0 and int(e._m20_ascent_state['map_count'])==old_map_count
    gaps=[]
    for height in (.55,.30,.20):
        e=pose(110);e.scene['robot'].data.root_pos_w[0,0]=-.05
        e.scene['robot'].data.root_pos_w[0,2]=height
        gaps.append(float(ns['ascent_front_body_clearance_cost'](e,**pose_params)))
    out['body_gap_descending_costs']=gaps;assert gaps[0]==0 and gaps[2]>gaps[1]>0
    e=pose(110,invalid=True)
    out['unknown_body_gap_cost']=float(ns['ascent_front_body_clearance_cost'](e,**pose_params));assert out['unknown_body_gap_cost']==0
    e=pose(110,down=True)
    r=e.scene['robot'].data;r.body_pos_w[0,2:,0][:2]=r.body_pos_w[0,6:,0]+.30
    r.body_pos_w[0,2:4,2]=r.body_pos_w[0,6:8,2]-.26
    out['rear_forward_cost']=float(ns['descent_rear_forward_cost'](e,**pose_params));assert out['rear_forward_cost']>0
    out['rear_vertical_free']=float(ns['descent_rear_forward_cost'](pose(110,down=True),**pose_params));assert out['rear_vertical_free']==0
    # A retained descent target also needs a spatially valid phase. Lateral
    # exit onto flat ground cannot keep rear style costs active for its timeout.
    e._m20_stair_step_state=dict(active=torch.tensor([True]),up=torch.tensor([False]),
        edge_xy=torch.zeros(1,2),source_z=torch.tensor([.1]),target_z=torch.zeros(1),
        front_reached=torch.tensor([[True,False]]),rear_support_paid=torch.zeros(1,2,dtype=torch.bool))
    e.common_step_counter+=1;e.episode_length_buf+=1
    assert ns['descent_rear_forward_cost'](e,**pose_params)>0
    r.body_pos_w[:,:,1]+=3;r.root_pos_w[:,1]+=3
    sc=e.scene['height_scanner'].data;sc.pos_w[:,1]+=3;sc.ray_hits_w[:,:,1]+=3;sc.ray_hits_w[:,:,2]=0
    e.common_step_counter+=1;e.episode_length_buf+=1
    out['flat_after_descent_exit_cost']=float(ns['descent_rear_forward_cost'](e,**pose_params))
    assert out['flat_after_descent_exit_cost']==0 and e._m20_stair_step_state['active'].all()
    e=pose(40);ns['ascent_front_fold_cost'](e,**pose_params)
    m=ns['gait_quality_metrics'](e,slice(None));assert m['fl_loaded_a_mean_deg']<41 and m['fl_loaded_a_low_fraction']==1
    out['pose_tags']={k:float(v) for k,v in m.items() if k.startswith(('front_loaded_a','fl_loaded','rear_forward'))}
    ns['track_lin_vel_xy_exp']=lambda *a:torch.ones(a[0].num_envs)
    native=float(ns['stair_route_track_lin_vel_xy_exp'](e,.5,'base_velocity',**pose_params));out['deep_fold_tracking_factor']=native
    assert native == 1.0  # Posture now has its own cost; preserve native task tracking.
    # Preparation tracks raw geometry even when support eligibility is off.
    e=make_env();st=ns['_new_state'](e);row=torch.tensor([0]);target=torch.tensor([0]);has=torch.tensor([True])
    expected=torch.tensor([-1]);eligible=torch.tensor([[False,False]])
    def prep(values,elig=eligible,exp=expected):
        return float(ns['rear_preparation_gain'](st,row,target,has,torch.tensor([values]),elig,exp))
    assert prep([.2,.1])==0
    assert prep([.4,.3])==0  # loaded rolling updates maxima without paying
    assert prep([.4,.3],torch.tensor([[True,True]]))==0  # support flip only
    assert abs(prep([.6,.3],torch.tensor([[True,False]]))-.2)<1e-6
    assert prep([.6,.3],torch.tensor([[True,True]]))==0
    assert prep([.7,.8],torch.tensor([[True,True]]),torch.tensor([0]))>0
    assert prep([.7,.8],torch.tensor([[True,True]]),torch.tensor([1]))==0
    assert prep([.5,.7],torch.tensor([[True,True]]))==0
    assert prep([1.,1.],torch.tensor([[True,True]]))>0
    assert st['rear_prep_paid'].sum()<=1
    ns['_reset'](st,torch.tensor([True]));assert not st['rear_prep_initialized'].any() and not st['rear_prep_paid'].any()
    out['rear_prep_loaded_roll_support_flip_retry_wrong_side_reset']=True
    # Restore from paired stepping, then revoke a later same-tread catch-up.
    e,tick,land=make();e.scene['robot'].data.body_pos_w[0,:2,0]=.6
    e.scene['robot'].data.body_pos_w[0,:2,2]=.69
    land(2,0);land(3,0);s=land(2,1)
    assert s['rear_recovery_event'] and not s['rear_event'].any(), {k:s[k].tolist() for k in ('rear_recovery_event','map_duplicate_paid','map_landed','map_stable_time','rear_successes','rear_violations','skip_total','map_safe','axis_last_slot')}
    recovery=float(ns['stair_rear_transition_recovery'](e,**p)*e.step_dt)
    assert recovery==1
    s=land(3,1)
    refund=float(ns['stair_rear_style_refund'](e,**p)*e.step_dt)
    assert refund==ns['TUNING']['rear_recovery_weight']
    assert not tick()['rear_style_refund'].any()
    out['rear_repair_then_duplicate_net']=recovery*ns['TUNING']['rear_recovery_weight']-refund+ns['TUNING']['same_tread_event_weight']
    assert out['rear_repair_then_duplicate_net']<0
    # A true alternating transition also loses its style credit if its tread
    # later becomes shared; ordinary task crossing credit is untouched.
    e,tick,land=make();e.scene['robot'].data.body_pos_w[0,:2,0]=.6
    e.scene['robot'].data.body_pos_w[0,:2,2]=.69
    land(2,0);s=land(3,1);assert s['rear_event'].any()
    s=land(2,1);assert s['rear_style_refund']==ns['TUNING']['rear_completion_weight']
    out['rear_delayed_duplicate_refund']=float(s['rear_style_refund'])
    return out


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,default=MDP/'stair_teacher.py');parser.add_argument('--output',type=Path);args=parser.parse_args()
    torch.set_num_threads(1);results=check(load(args.source))
    if args.output:args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))
