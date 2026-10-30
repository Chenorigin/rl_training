"""v15 compound commands: real course/reset math, analytic reward counterexamples."""
from copy import deepcopy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from task_support.cts.compound_commands import (SIGNS, sample_commands, compound_sequence,
    tracking_rewards, validate_compound_commands, reserve_compound_path)
from task_support.cts.all_terrain_config import validate_all_terrain
from task_support.cts.all_terrain_geometry import build_atlas
from test_cts_all_terrain_course import fake, support


@pytest.fixture(scope='module')
def settings():
    return json.loads((Path(__file__).resolve().parents[1]/'configs/distillation/cts.json').read_text())


@pytest.fixture(scope='module')
def source(settings):
    geometry = dict(settings['all_terrain']['geometry'], samples_per_level=3, grid_resolution=.25)
    settings['all_terrain'].pop('wall_practice', None)  # Freeze pre-wall mode/geometry contract.
    return settings, build_atlas(geometry)


def test_all_modes_mirrors_and_full_amplitude(settings):
    cfg = deepcopy(settings['all_terrain']['compound_commands'])
    for k in ('forward_speed', 'backward_speed', 'lateral_speed', 'yaw_rate'):
        cfg[k][0] = cfg[k][1]
    command = sample_commands(torch.arange(19), cfg)
    torch.testing.assert_close(command.sign(), torch.tensor(SIGNS).float())
    assert set(map(tuple, command.sign().tolist())) == set(SIGNS)
    assert ((command != 0).sum(-1) == 2).sum() == 12
    assert command[4, 2] == pytest.approx(.6) and command[6, 0] == pytest.approx(-.7)
    assert command[11, 1] == pytest.approx(.5)
    validate_all_terrain(settings['all_terrain'])


@pytest.mark.parametrize('key,value', [('travel_m',float('nan')),('transition_probability',1.),
    ('enabled',1),('yaw_rate',[0,.6]),('forward_speed',[.7,.1]),('unknown',3)])
def test_invalid_settings_fail_closed(settings,key,value):
    cfg=deepcopy(settings['all_terrain']);cfg['compound_commands'][key]=value
    with pytest.raises(ValueError): validate_compound_commands(cfg)


def test_actual_course_modes_coverage_bounds_and_snapshot(source):
    env,state=fake(source)
    ids=torch.arange(env.num_envs)
    seen=[set() for _ in ids]
    for _ in range(19):
        for i,m in enumerate(state.mode.tolist()): seen[i].add(m)
        pair=(state.command_value.abs()>0).sum(-1)==2
        assert ((state.course_ids[pair]>=3)&(state.course_ids[pair]<=7)).all()
        if pair.any():
            d=state.command_value[pair,:2].norm(dim=-1)*state.move_duration[pair]
            torch.testing.assert_close(d,torch.full_like(d,state.cfg['compound_commands']['travel_m']))
            xy=env.scene.robot.data.root_pos_w[pair,:2];b=state.bounds[state.tile[pair]]
            r=math.hypot(.25,.2)+.09+.027+state.cfg['rewards']['boundary_margin']
            margin=r+state.cfg['compound_commands']['travel_m']
            assert (xy>=b[:,(0,2)]+margin-1e-5).all()
            assert (xy<=b[:,(1,3)]-margin+1e-5).all()
        state.reset(ids)
    for i,modes in enumerate(seen):
        c=int(state.course_ids[i])
        assert modes==({0} if c<3 or c==8 else {0,8} if c==9 else set(range(19)))
    assert state.frames.shape==(2,10,19) and state.num_groups==9
    state.restore(deepcopy(state.snapshot()))
    bad=deepcopy(state.snapshot());bad['frames']=bad['frames'][:,:,:11]
    with pytest.raises(ValueError): state.restore(bad)


def test_disabled_is_legacy_equivalent(source):
    settings,atlas=source
    disabled=deepcopy(settings);disabled['all_terrain']['compound_commands']['enabled']=False
    absent=deepcopy(settings);absent['all_terrain'].pop('compound_commands')
    torch.manual_seed(42);env,a=fake((disabled,atlas))
    torch.manual_seed(42);_,b=fake((absent,atlas))
    for name in ('mode','group','tile','command_value','move_duration','command_ramp','arc_transition'):
        torch.testing.assert_close(getattr(a,name),getattr(b,name),rtol=0,atol=0)
    assert a.num_modes==11
    torch.testing.assert_close(a.raw_command(),b.raw_command(),rtol=0,atol=0)


@pytest.mark.parametrize('secondary_last', [False, True])
@pytest.mark.parametrize('mode',list(range(4,8))+list(range(11,19)))
def test_pair_add_and_withdraw_axis(mode,secondary_last,settings):
    cmd=sample_commands(torch.tensor([mode]),settings['all_terrain']['compound_commands']).repeat(5,1)
    out=compound_sequence(cmd,torch.tensor([.2,1.,2.,3.7,4.1]),torch.full((5,),4.),
        torch.zeros(5),torch.ones(5,dtype=torch.bool),torch.full((5,),secondary_last))
    assert (out[:4].abs()>0).sum(-1).tolist()==[1,2,2,1]
    indices=(cmd[0].abs()>0).nonzero().flatten()
    secondary=indices[-1] if secondary_last else indices[0]
    assert out[0,secondary]==0 and out[1,secondary]!=0
    assert not out[-1].any()
    constant=compound_sequence(cmd,torch.ones(5),torch.full((5,),4.),torch.zeros(5),
        torch.zeros(5,dtype=torch.bool),torch.full((5,),secondary_last))
    torch.testing.assert_close(constant,cmd)


@pytest.mark.parametrize('command', [[.35,.25,0],[-.35,.25,0],[-.35,-.25,0],
    [.35,0,-.3],[0,.25,.3],[0,-.08,-.1]])
def test_reward_requires_every_axis_and_limits_inactive_drift(settings,command):
    cmd=torch.tensor([command]).repeat(5,1);measured=cmd.clone()
    axis=(cmd[0].abs()>0).nonzero().flatten()[-1]
    measured[1,axis]=0;measured[2,axis]*=-1;measured[3,axis]*=2
    inactive=(cmd[0]==0).nonzero().flatten()[0];measured[4,inactive]=1
    state=SimpleNamespace(cfg=settings['all_terrain'],course_ids=torch.full((5,),3))
    g=dict(command=cmd,measured_velocity=measured)
    linear,yaw=tracking_rewards(state,g,torch.ones(5),torch.ones(5))
    assert linear[0]==1
    assert (linear[1:4]<.1).all() and linear[4]<.02
    torch.testing.assert_close(linear,yaw)
    assert linear[2]<linear[1]  # reversed worse than simply missing
    measured.requires_grad_(True);g['measured_velocity']=measured
    tracking_rewards(state,g,torch.ones(5),torch.ones(5))[0].sum().backward()
    assert torch.isfinite(measured.grad).all() and measured.grad[1,axis]!=0


def test_idle_and_special_courses_keep_rewards(settings):
    state=SimpleNamespace(cfg=settings['all_terrain'],course_ids=torch.tensor([0,1,2,8,9,3]))
    g=dict(command=torch.tensor([[.3,0,0]]*5+[[0.,0,0]]),measured_velocity=torch.ones(6,3))
    old=torch.arange(6).float()/10
    a,b=tracking_rewards(state,g,old,old+.1)
    torch.testing.assert_close(a,old);torch.testing.assert_close(b,old+.1)


def test_actual_raw_command_and_special_course_semantics(source):
    env,state=fake(source);ids=torch.arange(env.num_envs)
    for mode in (4,6,11,13,15,18):
        state.resets[:]=mode-state.rank;state.reset(ids)
        env.episode_length_buf.fill_(int((state.cfg['commands']['initial_hold_s']+.01)/env.step_dt))
        early=state.raw_command();moving=(state.course_ids>=3)&(state.course_ids<=7)
        transition=moving&state.arc_transition
        assert ((early[transition].abs()>0).sum(-1)<=1).all()
        # Staircases/straight/platform retain preexisting forward-only/idle command contract.
        special=~moving
        assert not state.command_value[special,1:].any()
        support(env,state)


def test_success_gate_rejects_ignored_small_axis(settings):
    from task_support.cts.compound_commands import axis_tracking_ok
    cfg=settings['all_terrain']['commands']
    command=torch.tensor([[.65,.08,0],[.7,0,.1],[0,.5,.1]])
    measured=command.clone();measured[0,1]=0;measured[1:,2]=0
    frames=torch.full((3,),100.)
    assert not axis_tracking_ok((measured-command).square()*100, command.square()*100, frames,cfg).any()
    assert axis_tracking_ok(torch.zeros_like(command), command.square()*100,frames,cfg).all()


def test_real_course_success_rejects_small_axis_cheat(source):
    env,state=fake(source)
    practice=((state.course_ids>=3)&(state.course_ids<=7)).nonzero().flatten()
    ids=practice[:1];state.resets[ids]=11-state.rank[ids];state.reset(ids)
    state.arc_transition[ids]=False;state.command_ramp[ids]=0
    state.command_value[ids]=torch.tensor([[.65,.08,0.]])
    env.episode_length_buf.fill_(30);env.scene.robot.data.root_lin_vel_b[ids,:2]=torch.tensor([[.65,0.]])
    env.common_step_counter+=1;support(env,state);state.advance()
    assert not state.tracking_ok[ids].any()
    assert state.axis_tracking_error[ids,1]>0
