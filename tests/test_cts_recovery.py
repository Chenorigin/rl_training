"""Recovery counterexamples: real course accounting, safe-label gradients, meshes."""
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from task_support.cts.configuration import default_settings, validate_config
from task_support.cts.action_safety import ActionSafety
from task_support.cts.joint_limits import TorchJointLimits
from task_support.cts.all_terrain_geometry import build_atlas, TorchSurface
from task_support.cts.stair_recovery import StairRecovery
from test_cts_all_terrain_course import fake, support, frame, source


def safety():
    cfg = default_settings()
    return ActionSafety(TorchJointLimits(cfg['joint_limits']),cfg['all_terrain']['safety_learning'])


def test_raw_tail_recovers_even_beyond_action_clip_and_excludes_wheels():
    s = safety()
    action = torch.zeros(100,16)
    action[-1,0] = 200.
    action[:,12:] = 10000.
    action.requires_grad_()
    loss = s.tail_cost(action)
    loss.backward()
    assert action.grad[-1,0] > 0
    assert (action.grad[:,:12][:-1] == 0).all()
    assert (action.grad[:,12:] == 0).all()
    assert s.accept(action).sum() == 99
    assert s.tail_cost(torch.zeros(2,16)) == 0
    # A rare boundary violation cannot be hidden by a safe majority.
    assert loss > ((s.raw_targets(action)-s.limits.default).square().mean())


@pytest.mark.parametrize('value',[float('nan'),float('inf')])
def test_nonfinite_labels_rejected_and_predictions_fail_closed(value):
    s = safety(); a = torch.zeros(2,16);a[0,15] = value
    assert s.accept(a).tolist() == [False,True]
    with pytest.raises(RuntimeError):s.tail_cost(a)


def recovering(source):
    cfg, atlas = source
    cfg = deepcopy(cfg)
    cfg['all_terrain']['stair_recovery'] = default_settings()['all_terrain']['stair_recovery']
    c = cfg['all_terrain']['stair_recovery']
    c.update(near_start_probability=1.,pause_probability=1.,perturb_probability=1.)
    return fake((cfg,atlas),n=100)


def test_cold_start_footprint_is_before_riser_zero_velocity_and_speed_range(source):
    env,s = recovering(source);r = s.recovery;ids=r.active.nonzero().flatten()
    p=env.scene.robot.data.root_pos_w
    gap=s.origins[s.tile[ids],0]+s.cfg['geometry']['route_start']-p[ids,0]-r.cfg['front_extent_m']
    assert (gap >= r.cfg['start_gap_m'][0]-1e-4).all()
    assert (gap <= r.cfg['start_gap_m'][1]+1e-4).all()
    assert (env.scene.robot.data.root_lin_vel_b == 0).all()
    assert (s.command_value[ids,0] >= r.cfg['speed_m_s'][0]).all()
    assert (s.command_value[ids,0] <= r.cfg['speed_m_s'][1]).all()
    assert (s.raw_command() == 0).all() # initial hold is preserved
    assert (s.timeout[ids] > 0).all()


def pause_geometry(env,s):
    r=s.recovery;ids=r.active.nonzero().flatten()
    g=s.observe().copy()
    g['position']=g['position'].clone();g['position'][ids,:2,0]=r.trigger_x[ids,None]+.1
    g['supported']=torch.ones(env.num_envs,4,dtype=torch.bool)
    g['command']=s.command_value.clone()
    env.episode_length_buf.fill_(100)
    return ids,g


def test_pause_requires_settling_never_marks_route_complete_and_restarts_once(source):
    env,s=recovering(source);r=s.recovery;ids,g=pause_geometry(env,s)
    r.advance(g,torch.zeros(env.num_envs,dtype=torch.bool))
    assert (r.phase[ids] == 2).all()
    assert not s.reached.any() and not s.success.any()
    assert (r.command(s.command_value)[ids] == 0).all()
    g['command']=r.command(s.command_value)
    for _ in range(3):r.advance(g,torch.zeros(env.num_envs,dtype=torch.bool))
    assert (r.phase[ids] == 2).all()
    for _ in range(round(r.cfg['settle_s']/env.step_dt)+1):
        r.advance(g,torch.ones(env.num_envs,dtype=torch.bool))
    assert (r.phase[ids] == 3).all()
    assert (r.events['restart'][ids] == 1).all()
    assert (r.command(s.command_value)[ids,0] > .1).all()
    assert not s.reached.any() and not s.success.any()


def test_unsettled_pause_fails_and_foreleg_flailing_cannot_reset_stall(source):
    env,s=recovering(source);r=s.recovery;ids,g=pause_geometry(env,s)
    r.advance(g,torch.zeros(env.num_envs,dtype=torch.bool));g['command']=r.command(s.command_value)
    for _ in range(round(r.cfg['pause_timeout_s']/env.step_dt)+2):
        r.advance(g,torch.zeros(env.num_envs,dtype=torch.bool))
    assert s.failed[ids].all() and not s.success.any()
    assert (r.events['pause_failure'][ids] == 1).all()
    # New episode with continuous forward command and zero rear progress.
    s.reset(ids);r.phase[ids]=0;s.failed[ids]=False
    g['command']=s.command_value.clone()
    for _ in range(round(r.cfg['stall_s']/env.step_dt)+2):
        g['position'][ids,:2,0] += .05
        r.advance(g,torch.ones(env.num_envs,dtype=torch.bool))
    assert s.failed[ids].all() and (r.events['stall_failure'][ids] == 1).all()


def test_actual_actor_yaw_used_and_zero_motion_course_still_succeeds(source):
    env,s=recovering(source);r=s.recovery
    # Cache exactly the command each cohort saw, without updating feedback.
    ids=r.active.nonzero().flatten();r.perturb_pending.zero_()
    teacher=torch.zeros(env.num_envs,244);policy=torch.zeros(env.num_envs,789)
    teacher[:,8]=.23;policy[:,8]=-.17
    r.begin_step({'teacher':teacher,'policy':policy})
    actual=torch.where(s.teacher,.23,-.17)
    env.scene.robot.data.root_ang_vel_b[:,2]=actual
    s.measure_tick=-1
    assert (s.observe()['yaw_error'][ids] == 0).all()
    # Legacy non-stair standing mode must still accumulate tracking_frames.
    motion=((s.course_ids==3)).nonzero().flatten()
    s.mode[motion]=8;s.command_value[motion]=0
    env.scene.robot.data.root_ang_vel_b.zero_()
    frame(env,s,1.,'zero')
    assert (s.tracking_frames[motion] > 0).all()
    for _ in range(round(s.cfg['commands']['stop_s']/env.step_dt)+2):
        frame(env,s,s.cfg['commands']['initial_hold_s']+s.move_duration.max().item()+.2,'zero')
    assert s.success[motion].all()


@pytest.mark.parametrize('reference_bias',[200.,float('nan'),float('inf')])
def test_tail_and_rejected_labels_wired_into_both_optimizers(reference_bias):
    from test_cts_teacher_finetuning import setup
    from test_cts_all_terrain_actor import collect
    models,alg,obs,*_=setup()
    alg.joint_limits=safety().limits;alg.action_safety=safety();alg.auxiliary.action_safety=alg.action_safety
    # Make reference labels unsafe without modifying the current teacher.
    with torch.no_grad():
        last=[m for m in models.reference_teacher.mlp.modules() if isinstance(m,torch.nn.Linear)][-1]
        last.bias[0]=reference_bias
    collect(alg,obs)
    metrics=alg.ppo_update()
    assert metrics['reference_rejected_fraction'] > 0
    assert 'teacher_limit_tail_loss' in metrics
    metrics=alg.distill_update()
    assert 'unsafe_teacher_fraction' in metrics and np.isfinite(metrics['total_loss'])


def test_short_wall_and_post_are_real_surface_and_mesh_not_map_dropout():
    cfg=default_settings()['all_terrain'];geom=deepcopy(cfg['geometry'])
    geom.update(geometry_levels=1,samples_per_level=3,grid_resolution=.25)
    atlas=build_atlas(geom,cfg['wall_practice'],cfg['obstacle_response'])
    wall=[r for r in atlas.records if r.get('wall_kind')=='single_wall']
    assert {r['obstacle_variant'] for r in wall} == {'wall','finite_wall','post'}
    surface=TorchSurface(atlas,'cpu',dtype=torch.float64)
    for r in wall:
        rect=r['wall_rectangles'][0];i=r['tile_id'];origin=atlas.origins[i,:2]
        xy=np.array([[(rect[0]+rect[1])/2,(rect[2]+rect[3])/2],[rect[1]+.1,(rect[2]+rect[3])/2]])
        z,_=surface.sample(torch.tensor([i]),torch.tensor((xy+origin)[None]))
        assert z[0,0] == pytest.approx(rect[4]) and z[0,1] == 0
        assert atlas.tile_meshes[i].bounds[1,2] == pytest.approx(rect[4])


@pytest.mark.parametrize('name,key,value',[
    ('stair_recovery','speed_m_s',[0.,.65]),('stair_recovery','pause_step',[0,3]),
    ('stair_recovery','settle_s',4.),('safety_learning','tail_fraction',2.),
    ('obstacle_response','post_half_width_m',float('nan'))])
def test_invalid_recovery_config_rejected(name,key,value):
    cfg=default_settings();cfg['all_terrain'][name][key]=value
    with pytest.raises(ValueError):validate_config(cfg)


def test_airborne_lagging_rear_does_not_reset_stall(source):
    env,s=recovering(source);r=s.recovery;ids,g=pause_geometry(env,s)
    r.phase[ids]=0
    g['position'][ids,2,0]=0.;g['position'][ids,3,0]=1.
    r.advance(g,torch.zeros(env.num_envs,dtype=torch.bool))
    r.stall_age[ids]=1.
    g['supported'][ids,2]=False;g['position'][ids,2,0]=.5
    r.advance(g,torch.zeros(env.num_envs,dtype=torch.bool))
    assert (r.stall_age[ids]>1.).all()
    assert (r.progress_x[ids]==0.).all()
    g['supported'][ids,2]=True
    r.advance(g,torch.zeros(env.num_envs,dtype=torch.bool))
    assert (r.stall_age[ids]==0.).all() and (r.progress_x[ids]==.5).all()
