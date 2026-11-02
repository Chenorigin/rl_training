"""On-stair practice counterexamples; fake physics does not prove learned ability."""
from copy import deepcopy
import pytest
import torch
from task_support.cts.configuration import default_settings
from task_support.cts.stair_hold_practice import StairHoldPractice
from task_support.student_v1.perception import quaternion_rotation
from test_cts_all_terrain_course import source, frame
from test_cts_stair_stop import setup as stop_setup


def setup(source):
    env,s,i=stop_setup(source)
    s.cfg['stair_hold_practice']=deepcopy(default_settings()['all_terrain']['stair_hold_practice'])
    # The reduced legacy atlas can lack feasible poses at high grades. Use
    # level-zero geometry in each grade for logic tests; full atlas tested separately.
    for c in (0,2):
        for level in range(1,7):s.pools[c,level]=s.pools[c,0]
    h=s.stair_hold=StairHoldPractice(s)
    robot=env.scene.robot;d=robot.data
    robot.write_joint_state_to_sim=lambda p,v,env_ids:None
    ids=torch.tensor([i]);h.select=lambda ids:h.active.__setitem__(ids,True)
    s.reset(ids)
    rot=quaternion_rotation(d.root_quat_w[ids])
    d.body_link_pos_w[ids]=(rot@torch.tensor(h.cfg['wheel_centers_body_m']).T).transpose(1,2)+d.root_pos_w[ids,None]
    d.body_com_pos_w[ids]=d.body_link_pos_w[ids]
    d.body_link_quat_w[ids]=d.root_quat_w[ids,None]
    d.projected_gravity_b[ids]=-(rot.transpose(1,2)@torch.tensor([0.,0.,1.]).view(1,3,1)).squeeze(-1)
    env.scene.sensors['contact_forces'].data.net_forces_w[ids]=torch.tensor([0.,0.,100.])
    s.measure_tick=-1
    return env,s,i,h


def tick(env,s,age=.4):frame(env,s,age,'zero')


def test_spawn_zero_and_fixed_anchor_grace(source):
    env,s,i,h=setup(source)
    assert s.raw_command()[i].abs().sum()==0 and not s.recovery.active[i]
    assert h.validate_initial()['fk_max_error_m']<1.e-5
    anchor=h.anchor.clone()
    tick(env,s,.1);assert not h.ready[i] and h.cost[i]==0
    env.scene.robot.data.root_pos_w[i,0]-=.2
    tick(env,s,.4)
    assert not h.ready[i] and h.cost[i]>0
    torch.testing.assert_close(h.anchor,anchor)


def test_supported_hold_succeeds_and_does_not_credit_original_course(source):
    env,s,i,h=setup(source)
    before={k:getattr(s,k).clone() for k in ('episodes','successes')}
    for k in range(65):tick(env,s,.3+k*.02)
    assert h.ready[i] and s.success[i] and s.progress[i]==0
    env.termination_manager.terminated[i]=True
    s.record(torch.tensor([i]));s.record(torch.tensor([i]))
    assert h.totals['episodes'][1,0]==h.totals['successes'][1,0]==1
    for key in ('episodes','successes'):torch.testing.assert_close(getattr(s,key),before[key])
    assert s.curriculum.diagnostics()[int(s.group[i])]['student_episodes']==0
    saved=s.snapshot();s.restore(saved)
    assert h.totals['episodes'][1,0]==1


@pytest.mark.parametrize('case',['root_drift','foot_drift','two_supports','front_only','slip','yaw','nonzero_command','fallen','slow_roll'])
def test_bad_hold_cannot_pass(source,case):
    env,s,i,h=setup(source);d=env.scene.robot.data
    tick(env,s)
    baseline=float(h.cost[i]);assert h.ready[i]
    if case=='root_drift':d.root_pos_w[i,0]-=.3
    elif case=='foot_drift':d.body_link_pos_w[i,:,0]-=.3
    elif case=='two_supports':env.scene.sensors['contact_forces'].data.net_forces_w[i,[1,2]]=0
    elif case=='front_only':env.scene.sensors['contact_forces'].data.net_forces_w[i,2:]=0
    elif case=='slip':d.body_lin_vel_w[i,:,0]=.3
    elif case=='yaw':d.root_ang_vel_b[i,2]=.4
    elif case=='nonzero_command':s.command_value[i,0]=.4
    elif case=='fallen':d.root_pos_w[i,2]=-1
    elif case=='slow_roll':d.root_lin_vel_w[i,0]=.08
    frame(env,s,.42)
    assert not h.ready[i] and not s.success[i]
    if case in ('root_drift','foot_drift','two_supports','front_only','slip'):assert h.cost[i]>baseline


def test_feasible_boundary_separate_and_resume_atomic(source):
    env,s,i,h=setup(source)
    for pool in h.pools.values():assert h.boundary_tiles[pool].any()
    saved=s.snapshot();bad=deepcopy(saved)
    bad['stair_hold_practice']['totals']['episodes'][0,0]=float('nan')
    with pytest.raises(ValueError,match='stair hold'):s.restore(bad)
    torch.testing.assert_close(h.totals['episodes'],saved['stair_hold_practice']['totals']['episodes'])
    h.select=lambda ids:h.active.__setitem__(ids,False)
    s.reset(torch.tensor([i]));assert not h.active[i] and s.recovery.active[i]


def test_zero_command_keeps_internal_heading_correction(source):
    env,s,i,h=setup(source)
    g=s.observe();g['command'][i]=0;g['tracking_command'][i,2]=-.032
    env.episode_length_buf.fill_(20);env.common_step_counter+=1
    h.advance(g)
    assert h.ready[i]


def test_initial_fk_nonfinite_fails_closed(source):
    env,s,i,h=setup(source)
    env.scene.robot.data.body_link_pos_w[i,0,0]=float('nan')
    s.measure_tick=-1
    with pytest.raises(RuntimeError,match='Nonfinite'):h.validate_initial()
