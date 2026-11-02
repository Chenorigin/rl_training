from copy import deepcopy
from types import SimpleNamespace
import pytest
import torch
from test_cts_hind_ascent import fixture as hind_fixture, tick
from task_support.cts.front_descent import FrontDescent, falling_edges, validate


def fixture(dt=.02, course=1):
    s,_,g=hind_fixture(dt,course,spawn_z=4.)
    # Elevated spawn does not offset physical absolute terrain Z.
    s.surface.rectangles=torch.tensor([[[-1.,1.,-.6,.6,.6],[1.,1.3,-.6,.6,.4],[1.3,1.6,-.6,.6,.2]]])
    s.env.scene={'robot':SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.tensor([[101.,-100.,.8]]*2)))}
    g['position'][...,0]=101.15;g['position'][...,2]=.49;g['height'].fill_(.4)
    return s,FrontDescent(s),g


def test_persistent_one_front_leg_absence_and_no_positive_reward():
    s,h,g=fixture();g['supported'][:,0]=False;g['position'][:,0,2]+=.2
    r=tick(s,h,g,70)
    assert (r['air']<0).all() and (r['stall']<=0).all()
    assert h.air_age[:,0].min()>1 and not h.air_age[:,1].any()
    assert h.totals['overdue_s'][:,0].min()>0 and not h.totals['overdue_s'][:,1].any()
    # True support on different treads is fine; same tread is also fine.
    g['supported'].fill_(True);g['position'][:,0,2]=.29;g['height'][:,0]=.2
    tick(s,h,g,2);assert not h.air_age.any() and not h.values['air'].any()


def test_normal_transfer_and_single_sample_touch_do_not_reset_history():
    s,h,g=fixture();g['supported'][:,0]=False
    assert not tick(s,h,g,10)['air'].any()
    tick(s,h,g,40);before=h.air_age.clone()
    g['supported'][:,0]=True;tick(s,h,g);g['supported'][:,0]=False;tick(s,h,g)
    assert (h.air_age[:,0]>before[:,0]).all()
    g['supported'][:,0]=True;tick(s,h,g,2);assert not h.air_age.any()


def test_pause_freezes_absence_and_resume_cannot_launder_it():
    s,h,g=fixture();g['supported'][:,0]=False;tick(s,h,g,70)
    before=h.air_age.clone();g['command'].zero_()
    r=tick(s,h,g,20);assert not r['air'].any() and not r['stall'].any()
    torch.testing.assert_close(h.air_age,before)
    g['command'][:,0]=.3;assert (tick(s,h,g)['air']<0).all()
    g['command'].zero_();g['supported'].fill_(True);tick(s,h,g,2);assert not h.air_age.any()


def test_stall_uses_root_highwater_not_waving_front_wheels():
    s,h,g=fixture();g['position'][:,:2,0]+=.1
    r=tick(s,h,g,170);assert (r['stall']<0).all()
    s.env.scene['robot'].data.root_pos_w[:,0]+=.02;tick(s,h,g)
    assert not h.values['stall'].any()
    s.env.scene['robot'].data.root_pos_w[:,0]-=.02;tick(s,h,g,170)
    s.env.scene['robot'].data.root_pos_w[:,0]+=.02;tick(s,h,g)
    assert (h.values['stall']<0).all()


@pytest.mark.parametrize('course',[0,3,4,5,6,7,8,9])
def test_other_terrains_are_untouched(course):
    s,h,g=fixture(course=course);g['supported'].zero_()
    r=tick(s,h,g,180);assert all(not v.any() for v in r.values())


def test_roundtrip_top_plateau_last_floor_and_partial_reset():
    s,h,g=fixture(course=2);g['supported'][:,0]=False
    g['position'][...,0]=100.5;assert not tick(s,h,g,80)['air'].any()
    assert h.last.item()==pytest.approx(1.6)  # Final drop to implicit z=0 floor.
    g['position'][...,0]=101.8;g['height'].zero_();g['position'][...,2]=.09
    assert (tick(s,h,g,80)['air']<0).all()
    h.reset(torch.tensor([0]));assert not h.air_age[0].any() and h.air_age[1,0]>0
    h.checked_snapshot(h.snapshot());bad=deepcopy(h.snapshot());bad['active_s'][0,0]=torch.nan
    with pytest.raises(ValueError):h.checked_snapshot(bad)


@pytest.mark.parametrize('field',['position','axis','force','normal','velocity','height','force_n'])
@pytest.mark.parametrize('bad',[float('nan'),float('inf')])
def test_nonfinite_measurements_fail_closed_with_finite_costs(field,bad):
    s,h,g=fixture();g[field][:,0]=bad;r=tick(s,h,g)
    assert s.failed.all() and all(torch.isfinite(v).all() for v in r.values())
    assert all(torch.isfinite(v).all() for v in h.totals.values())


@pytest.mark.parametrize('dt',[.01,.02,.04])
def test_cost_integrates_in_time_and_force_size_earns_nothing(dt):
    s,h,g=fixture(dt);g['supported'][:,0]=False;tick(s,h,g,int(1.2/dt))
    r=tick(s,h,g,int(.4/dt));torch.testing.assert_close(r['air'],torch.full((2,),-.1))
    g['force']*=10;g['force_n']*=10
    r2=tick(s,h,g,int(.4/dt));torch.testing.assert_close(r['air'],r2['air'])


def test_disabled_optional_config_and_invalid_values():
    from task_support.cts.configuration import default_settings,validate_config
    c=default_settings();validate_config(c)
    for name in ('front_descent','clutter_consistency'):c['all_terrain'].pop(name)
    validate_config(c)
    c=default_settings()['all_terrain']['front_descent'];c['air_weight']=float('nan')
    with pytest.raises(ValueError):validate(c)
