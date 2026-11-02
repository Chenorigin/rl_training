"""Outcome-based hind credit: adversarial geometry, contact, reset and time tests."""
from copy import deepcopy
from types import SimpleNamespace
import math
import pytest
import torch

from task_support.cts.configuration import default_settings, validate_config
from task_support.cts.hind_ascent import HindAscent, rear_measurement, validate


def fixture(dt=.02, course=0, spawn_z=0.):
    cfg = default_settings()['all_terrain']
    cfg['geometry']['route_start'] = 1.
    rects = torch.tensor([[[1., 1.3, -.6, .6, .2], [1.3, 1.6, -.6, .6, .4],
                           [1.6, 3., -.6, .6, .6]]])
    env = SimpleNamespace(num_envs=2, device='cpu', step_dt=dt, common_step_counter=0,
                          cfg=SimpleNamespace(stair_tail=SimpleNamespace(wheel_radius=.09, wheel_half_length=.027)))
    state = SimpleNamespace(env=env, cfg=cfg, surface=SimpleNamespace(rectangles=rects),
                            course_ids=torch.full((2,), course), tile=torch.zeros(2,dtype=torch.long),
                            origins=torch.tensor([[100.,-100.,spawn_z]]), teacher=torch.tensor([True,False]),
                            failed=torch.zeros(2,dtype=torch.bool), reached=torch.zeros(2,dtype=torch.bool))
    g = dict(position=torch.zeros(2,4,3), axis=torch.zeros(2,4,3), normal=torch.zeros(2,4,3),
             velocity=torch.zeros(2,4,3), force=torch.zeros(2,4,3), force_n=torch.full((2,4),100.),
             height=torch.zeros(2,4), supported=torch.ones(2,4,dtype=torch.bool), command=torch.tensor([[.3,0.,0.]]*2))
    g['axis'][...,1]=1.; g['normal'][...,2]=1.; g['force'][...,2]=100.
    place(g,.6,0.)
    return state, HindAscent(state), g


def place(g, x, z, supported=True):
    g['position'][...,0]=100.+x; g['position'][...,1]=-100.; g['position'][...,2]=z+.09
    g['height'].fill_(z); g['supported'].fill_(supported)


def tick(s,h,g,n=1):
    totals={k:torch.zeros(2) for k in h.values}
    for _ in range(n):
        s.env.common_step_counter+=1; h.advance(g)
        for k,v in h.values.items(): totals[k]+=v*s.env.step_dt
    return totals


def test_confirmed_tread_once_per_wheel_and_no_front_leg_credit():
    s,h,g=fixture();tick(s,h,g)
    # Front wheels alone are irrelevant.
    g['position'][:,:2,0]+=1.;assert not tick(s,h,g,8)['step'].any()
    place(g,1.2,.2);reward=tick(s,h,g,8)
    torch.testing.assert_close(reward['step'],torch.full((2,),s.cfg['hind_ascent']['step_reward_budget']/3))
    assert h.totals['confirmed_steps'].sum()==4
    assert not tick(s,h,g,8)['step'].any()
    place(g,.6,0.);tick(s,h,g);place(g,1.2,.2)
    assert not tick(s,h,g,8)['step'].any()  # Retreat/re-enter cannot farm.
    place(g,1.5,.4);assert (tick(s,h,g,8)['step']>0).all()


@pytest.mark.parametrize('dt',[.01,.02,.04])
def test_event_budget_is_independent_of_control_dt_and_skipped_steps(dt):
    s,h,g=fixture(dt);tick(s,h,g);place(g,1.8,.6)
    credit=tick(s,h,g,math.ceil(.16/dt))['step']
    torch.testing.assert_close(credit,torch.full((2,),s.cfg['hind_ascent']['step_reward_budget']/3))
    assert h.totals['confirmed_steps'].sum()==4  # Actual third tread only.


def test_spawn_baseline_world_z_partial_reset_and_snapshot_validation():
    s,h,g=fixture(spawn_z=4.);place(g,1.2,.2)
    assert not tick(s,h,g,8)['step'].any()  # Already standing at initialization.
    place(g,.6,0.);tick(s,h,g);place(g,1.2,.2)
    assert not tick(s,h,g,8)['step'].any()
    h.reset(torch.tensor([0]));place(g,.6,0.);tick(s,h,g);place(g,1.2,.2)
    r=tick(s,h,g,8)['step'];assert r[0]>0 and r[1]==0
    saved=h.snapshot();h.checked_snapshot(saved)
    bad=deepcopy(saved);bad['step_reward'][0,0]=torch.nan
    with pytest.raises(ValueError):h.checked_snapshot(bad)


def test_landing_requires_own_support_low_slip_and_actual_tread():
    s,h,g=fixture();tick(s,h,g);place(g,1.2,.2,False)
    assert not tick(s,h,g,8)['step'].any()
    # Touchdown cannot cash the displacement previously made in the air.
    g['supported'].fill_(True);g['velocity'][:,2:,0]=.5
    r=tick(s,h,g,8);assert not r['step'].any() and not r['progress'].any() and (r['slip']<0).all()
    g['velocity'].zero_();g['position'][:,2:,1]=-99.1  # Outside the tread corridor.
    assert not tick(s,h,g,8)['step'].any()
    g['position'][:,2:,1]=-100.;g['height'].fill_(1.2)  # A wall top is not the tread.
    assert not tick(s,h,g,8)['step'].any()
    g['height'].fill_(.2);g['supported'][:,2]=False
    assert (tick(s,h,g,8)['step']>0).all()
    assert (h.totals['confirmed_steps'][:,0]==0).all() and (h.totals['confirmed_steps'][:,1]==1).all()


def test_progress_is_same_wheel_supported_and_cannot_be_banked_or_repeated():
    s,h,g=fixture();tick(s,h,g);place(g,1.2,.2);tick(s,h,g)
    g['supported'][:,2]=False;g['position'][:,2,0]+=.04
    assert not tick(s,h,g)['progress'].any()  # Other rear support cannot qualify this foot.
    g['supported'][:,2]=True;assert not tick(s,h,g)['progress'].any()
    g['position'][:,2,0]+=.04;assert (tick(s,h,g)['progress']>0).all()
    g['position'][:,2,0]-=.03;tick(s,h,g);g['position'][:,2,0]+=.03
    assert not tick(s,h,g)['progress'].any()


@pytest.mark.parametrize('kind',['stop','down','flat','failure','nan','inf'])
def test_no_positive_credit_on_invalid_or_inactive_states(kind):
    s,h,g=fixture();tick(s,h,g);place(g,1.2,.2)
    if kind=='stop':g['command'].zero_()
    if kind=='down':s.course_ids.fill_(1)
    if kind=='flat':s.course_ids.fill_(3)
    if kind=='failure':s.failed.fill_(True)
    if kind in ('nan','inf'):g['velocity'][:,2,0]=float(kind)
    r=tick(s,h,g,8)
    assert not r['step'].any() and not r['progress'].any()
    assert all(torch.isfinite(v).all() for v in h.values.values())
    if kind in ('nan','inf'):assert s.failed.all()


def test_pure_rolling_is_not_slip_and_force_magnitude_is_not_rewarded():
    s,h,g=fixture();tail=s.env.cfg.stair_tail
    # Material velocity of a wheel rolling +X at 1m/s on a stationary tread.
    linear=torch.tensor([1.,0.,0.]);omega=torch.tensor([0.,1./.09,0.]);offset=torch.tensor([0.,0.,-.09])
    g['velocity'][:]=linear+torch.linalg.cross(omega,offset)
    assert rear_measurement(g,tail.wheel_radius,tail.wheel_half_length)['slip'].max()<1.e-6
    results=[]
    for force in (100.,500.):
        s,h,g=fixture();g['force'][...,2]=force;g['force_n'].fill_(force)
        tick(s,h,g);place(g,1.2,.2);results.append(tick(s,h,g,8))
    for key in results[0]:torch.testing.assert_close(results[0][key],results[1][key])


def test_tick_idempotent_and_total_positive_budget_bounded():
    s,h,g=fixture();tick(s,h,g)
    total=torch.zeros(2)
    for x,z in ((1.2,.2),(1.5,.4),(1.8,.6)):
        place(g,x,z);r=tick(s,h,g,8);total+=r['step']+r['progress']
        before=h.snapshot();h.advance(g)
        for k,v in before.items():torch.testing.assert_close(v,h.totals[k])
    assert (total<=s.cfg['hind_ascent']['step_reward_budget']+s.cfg['hind_ascent']['progress_reward_budget']+1.e-6).all()


@pytest.mark.parametrize('key',list(default_settings()['all_terrain']['hind_ascent']))
def test_configuration_rejects_nonfinite_and_legacy_absence_is_supported(key):
    cfg=default_settings();cfg['all_terrain']['hind_ascent'][key]=float('nan')
    with pytest.raises(ValueError):validate_config(cfg)
    cfg=default_settings();cfg['all_terrain'].pop('hind_ascent');validate_config(cfg)


@pytest.mark.parametrize('speed',[.12,.3,.65])
@pytest.mark.parametrize('depth',[.15,.2,.35])
def test_continuous_rolling_landing_remains_reachable_on_narrow_treads(speed,depth):
    s,_,g=fixture()
    s.surface.rectangles=torch.tensor([[[1.,1+depth,-.6,.6,.2],
        [1+depth,1+2*depth,-.6,.6,.4],[1+2*depth,3.,-.6,.6,.6]]])
    h=HindAscent(s);g['command'][:,0]=speed;tick(s,h,g)
    for j in range(math.ceil((1+2*depth+.35-.6)/(speed*s.env.step_dt))):
        x=.6+(j+1)*speed*s.env.step_dt
        z=0. if x<1. else .2 if x<1+depth else .4 if x<1+2*depth else .6
        place(g,x,z);tick(s,h,g)
    assert (h.totals['confirmed_steps']==3).all()
