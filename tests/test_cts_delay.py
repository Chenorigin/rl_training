"""Timing mixture is independent of noise severity and preserves old-task defaults."""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from task_support.cts.configuration import default_settings, validate_config
from task_support.cts.perception import validate_mixture, validate_perception_contract, delay_diagnostics
from task_support.student_v1.configuration import StudentV1PerceptionCfg
from task_support.student_v1.perception import StudentObservationState, grid_points
from task_support.student_v1.profiles import profile_config


def cfg(**overrides):
    values=json.loads(Path('configs/distillation/student_v3_safe.json').read_text())['perception']
    values.update(overrides)
    return StudentV1PerceptionCfg(**values)


def fixture(n):
    grid=grid_points('cpu')
    hits=torch.cat((grid,.1*grid[:,:1]),-1)[None].repeat(n,1,1)
    proprio=torch.zeros(n,57);proprio[:,5]=-1.
    return proprio,hits,torch.tensor([[0.,0.,.5]]).repeat(n,1),torch.zeros(n),torch.eye(3)[None].repeat(n,1,1)


@pytest.mark.parametrize('p',[0.,1.,.2])
def test_actual_packet_queue_and_episode_selection(p):
    torch.manual_seed(42)
    state=StudentObservationState(128,'cpu',.02,cfg(zero_delay_probability=p),4)
    inputs=fixture(128)
    for step in range(50):state.observe(*inputs,step)
    receipt=delay_diagnostics(state,torch.arange(128)%4 != 0)
    assert receipt['packet_queue_checked'] > 0
    if p == 0:
        assert not state.zero_delivery.any() and receipt['zero_delay_queue_packets'] == 0
    elif p == 1:
        assert state.zero_delivery.all() and receipt['delayed_queue_packets'] == 0
    else:
        assert receipt['zero_delay_queue_packets'] > 0 and receipt['delayed_queue_packets'] > 0
    # Keep the same assignment for the complete episode; reset just the requested subset.
    before=state.zero_delivery.clone();counts=state.zero_delivery_resets+state.delayed_delivery_resets
    state.reset(torch.tensor([1,3]));state.observe(*inputs,50)
    keep=torch.ones(128,dtype=torch.bool);keep[[1,3]]=False
    assert torch.equal(before[keep],state.zero_delivery[keep])
    torch.testing.assert_close((state.zero_delivery_resets+state.delayed_delivery_resets-counts)[keep],torch.zeros(126,dtype=torch.long))
    assert ((state.zero_delivery_resets+state.delayed_delivery_resets-counts)[[1,3]] == 1).all()
    # Deliberately corrupt one queue entry: the observer must reject it.
    slot,env=(state.packet_capture >= 0).nonzero()[0]
    state.packet_arrival[slot,env]=state.packet_capture[slot,env]+100
    with pytest.raises(RuntimeError):delay_diagnostics(state,torch.arange(128)%4 != 0)


def test_delay_draw_is_independent_of_clean_regular_stress_selection():
    torch.manual_seed(123)
    state=StudentObservationState(2048,'cpu',.02,cfg(zero_delay_probability=.2,clean_probability=.3,stress_probability=.2))
    state._initialize()
    for level in range(3):
        chosen=state.mixture == level
        assert chosen.sum() > 300
        assert abs(float(state.zero_delivery[chosen].float().mean())-.2) < .055
    assert .16 < float(state.zero_delivery.float().mean()) < .24
    # A clean noise episode can still have delivery delay.
    assert ((state.mixture == 0) & ~state.zero_delivery).any()


def test_noise_scale_zero_does_not_remove_delivery_delay():
    torch.manual_seed(0)
    state=StudentObservationState(8,'cpu',.02,cfg(zero_delay_probability=0.,clean_probability=1.,stress_probability=0.))
    for step in range(30):state.observe(*fixture(8),step)
    assert (state.severity == 0).all()
    actual=(state.packet_arrival-state.packet_capture)[state.packet_capture >= 0]
    assert len(actual) and actual.min() >= 14 and actual.max() <= 21


def test_single_factor_profiles_override_saved_mixture():
    perception=cfg(zero_delay_probability=.2).__dict__
    for name in ('clean','latency','height','pose','proprio','dropout','action_delay'):
        assert profile_config(name,{'perception':perception}).zero_delay_probability == 0
    assert profile_config('student-v1',{'perception':perception}).zero_delay_probability == .2
    assert profile_config('front_only',{'perception':perception}).zero_delay_probability == .2


def test_current_contract_fails_closed_and_historical_default_is_zero():
    settings=default_settings();validate_config(settings)
    assert 'backward_practice' not in settings
    assert settings['motion_practice']['fraction']+settings['platform_practice']['fraction'] == .2
    assert sum(settings['all_terrain']['fractions'][3:]) == pytest.approx(.35)
    values=cfg(zero_delay_probability=.2).__dict__
    validate_perception_contract(settings['map_delay_mixture'],values)
    with pytest.raises(ValueError):validate_perception_contract(settings['map_delay_mixture'],{**values,'zero_delay_probability':0.})
    for p in (-.01,1.01,float('nan'),True,'0.2'):
        with pytest.raises(ValueError):validate_mixture({'zero_delay_probability':p})
        with pytest.raises(ValueError):cfg(zero_delay_probability=p).validate()
    old=deepcopy(values);old.pop('zero_delay_probability')
    assert StudentV1PerceptionCfg(**old).zero_delay_probability == 0.
    wrong=deepcopy(settings);wrong['backward_practice']={}
    with pytest.raises(ValueError):validate_config(wrong)
