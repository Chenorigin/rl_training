"""Adversarial cases for ascent shaping, including grounded pushing and landing."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from task_support.cts.configuration import default_settings, validate_config
from task_support.cts.hind_clearance import (rising_edges, clearance_signals,
    hind_step_clearance, hind_riser_contact, hind_support_transfer, validate_hind_clearance)


def cfg():
    return deepcopy(default_settings()['all_terrain']['hind_clearance'])


def sample(x=.94, bottom=0., supported=True, force_x=0., command=True):
    pos = torch.tensor([[[x, .2, bottom+.09], [x, -.2, bottom+.09],
                         [x, .2, bottom+.09], [x, -.2, bottom+.09]]])
    axis = torch.zeros_like(pos); axis[..., 1] = 1.
    force = torch.zeros_like(pos); force[..., 0] = force_x
    support = torch.full((1, 4), supported)
    height = torch.zeros(1, 4)
    rect = torch.tensor([[[1., 1.3, -.6, .6, .2], [1.3, 1.6, -.6, .6, .4],
                          [1.6, 3., -.6, .6, .4]]])
    edges, valid = rising_edges(rect, cfg()['min_rise_m'])
    return dict(position=pos, axis=axis, force=force, supported=support,
        support_height=height, edges=edges, valid_edges=valid, moving=torch.tensor([command]),
        radius=.09, half_length=.027, cfg=cfg())


def test_grounded_push_is_not_exempt_and_early_lift_has_lower_cost():
    a = sample(force_x=-150.)
    low = clearance_signals(**a)
    assert low['clearance'][0, 2] == pytest.approx(1.)
    assert low['riser'][0, 2] == 1.
    a['position'][0, 2, 2] += .25
    a['supported'][0, 2] = False
    high = clearance_signals(**a)
    assert high['clearance'][0, 2] == 0.
    assert high['riser'][0, 2] == 0.
    assert high['clearance'][0, 3] == 1.  # No masking a bad leg with a good one.
    assert high['support'] == 0.  # Other hind supports.
    a['supported'][0, 3] = False
    assert clearance_signals(**a)['support'] == 1.


def test_no_benefit_from_excess_height_and_no_cost_for_settled_landing():
    for bottom in (.25, .4, .8):
        assert clearance_signals(**sample(bottom=bottom, supported=False))['clearance'].sum() == 0
    a = sample(x=1.04, bottom=.2)
    a['support_height'].fill_(.2)
    s = clearance_signals(**a)
    assert not s['active'].any()
    assert s['clearance'].sum() == 0


def test_target_persists_past_centre_crossing_and_uses_world_envelope():
    a = sample(x=1.02, bottom=.1, supported=False)
    s = clearance_signals(**a)
    assert (s['top'] == .2).all()
    assert s['clearance'].min() > 0
    # Roll of wheel capsule increases vertical extent; body frame is unused.
    a['axis'][..., 1] = 0
    a['axis'][..., 2] = 1
    assert (clearance_signals(**a)['deficit'] > s['deficit']).all()


@pytest.mark.parametrize('depth', [.15,.20])
def test_landed_previous_tread_cannot_hide_next_riser_on_short_treads(depth):
    a = sample(x=1+depth-.085, bottom=.2, force_x=-150.)
    a['support_height'].fill_(.2)
    rect = torch.tensor([[[1., 1+depth, -.6, .6, .2],
                          [1+depth, 1+2*depth, -.6, .6, .4]]])
    a['edges'],a['valid_edges'] = rising_edges(rect,cfg()['min_rise_m'])
    s = clearance_signals(**a)
    assert s['active'].all()
    assert (s['top'] == .4).all()
    assert s['clearance'].min() > .99
    assert s['riser'].min() > .99


def test_smooth_approach_and_no_standing_descending_or_far_away_penalty():
    values = [float(clearance_signals(**sample(x=x))['clearance'][0, 2]) for x in (.7, .8, .9, .94)]
    assert values == sorted(values) and values[0] == 0. and values[-1] == 1.
    for a in (sample(x=.5), sample(command=False), sample(x=1.9), sample(x=1.15, bottom=.5)):
        s = clearance_signals(**a)
        assert s['clearance'].sum() == s['riser'].sum() == s['support'].sum() == 0
    a = sample()
    a['valid_edges'].fill_(False)
    assert clearance_signals(**a)['clearance'].sum() == 0


def test_equal_height_seams_padding_and_descents_do_not_create_up_edges():
    r = torch.tensor([[[-2., 1., -.6, .6, .4], [1., 1.3, -.6, .6, .2],
                        [1.3, 1.6, -.6, .6, .2], [0., 0., 0., 0., float('nan')]]])
    _, valid = rising_edges(r, .04)
    # Start boundary of elevated spawn exists but is far behind the +X route.
    assert valid.tolist() == [[True, False, False, False]]


def test_zero_net_riser_force_cost_for_forward_support_force():
    assert clearance_signals(**sample(force_x=150.))['riser'].sum() == 0


def test_environment_entry_uses_actual_geometry_per_wheel_and_full_cohort():
    a = sample(force_x=-150.)
    g = dict(position=a['position'].expand(2,-1,-1), axis=a['axis'].expand(2,-1,-1),
             force=a['force'].expand(2,-1,-1), supported=a['supported'].expand(2,-1),
             height=a['support_height'].expand(2,-1), command=torch.tensor([[.3,0,0],[.3,0,0]]))
    rect = torch.tensor([[[1., 1.3, -.6, .6, .2], [1.3, 3., -.6, .6, .4]]])
    state = SimpleNamespace(cfg={'hind_clearance':cfg()}, surface=SimpleNamespace(rectangles=rect),
        observe=lambda:g, origins=torch.zeros(1,3), tile=torch.zeros(2,dtype=torch.long),
        stairs=torch.tensor([True,False]))
    env = SimpleNamespace(cfg=SimpleNamespace(stair_tail=SimpleNamespace(wheel_radius=.09,wheel_half_length=.027)))
    with patch('task_support.cts.all_terrain_course.course', return_value=state):
        assert hind_step_clearance(env).tolist() == [-1., 0.]
        assert hind_riser_contact(env).tolist() == [-1., 0.]
        assert hind_support_transfer(env).tolist() == [0., 0.]
        # Partial reset / tile relocation changes all queries immediately:
        # no phase/debt history or per-tick cache can leak into another episode.
        state.origins[:,0] = 5.
        assert hind_step_clearance(env).sum() == 0


@pytest.mark.parametrize('key', list(cfg().keys()))
def test_invalid_configuration_fails_closed(key):
    c = cfg(); c[key] = float('nan')
    with pytest.raises(ValueError): validate_hind_clearance(c)


def test_new_defaults_and_old_checkpoint_config_remain_readable():
    c = default_settings()
    assert validate_config(c) is c
    c['all_terrain'].pop('hind_clearance')
    assert validate_config(c) is c
    assert validate_hind_clearance({**cfg(), 'enabled':False})['enabled'] is False
