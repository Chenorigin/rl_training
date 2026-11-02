"""Continuous terrain/motion, rolling constraints and old-checkpoint contracts."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from test_cts_platform import vertical_hit
from task_support.cts.configuration import default_settings, validate_config, VERSION, LEGACY_VERSIONS
from task_support.cts.motion import (TERRAINS, COMMANDS, MotionPractice, validate_motion, variants,
    surface, motion_patch_mesh, wheel_geometry, rolling_costs,
    motion_reward, motion_boundary, motion_timeout, motion_safety_features)
from task_support.cts.practice import cohort_mask, practice_mask, stair_only, FlatCourseState
from task_support.stair_quiet_v1_safe.settings import SafeSettings
from task_support.stair_quiet_v1_width.geometry import face_profile_ids


def fake_motion(n=480):
    cfg = default_settings();s = SafeSettings()
    teacher = cohort_mask(n,.75,42)
    platform = practice_mask(teacher,cfg['platform_practice']['fraction'],113)
    env = SimpleNamespace(cfg=SimpleNamespace(scene=SimpleNamespace(num_envs=n),sim=SimpleNamespace(device='cpu'),
        stair_tail=s),device='cpu',num_envs=n,step_dt=.02,common_step_counter=0,
        episode_length_buf=torch.zeros(n,dtype=torch.long),_tail_course={})
    p = env.cts_motion = MotionPractice(env,cfg['motion_practice'],cfg['algorithm'],platform)
    generator = SimpleNamespace(num_rows=2,num_cols=5,size=(12.,5.))
    _,origins = motion_patch_mesh(generator,p.cfg)
    robot = SimpleNamespace(data=SimpleNamespace(default_root_state=torch.zeros(n,13)),
                            write_root_pose_to_sim=Mock(),write_root_velocity_to_sim=Mock())
    robot.data.default_root_state[:,2] = .5
    class Scene(SimpleNamespace):
        def __getitem__(self,key):return robot
    env.scene = Scene(terrain=SimpleNamespace(terrain_types=torch.arange(n)%5,
        cts_motion_origins=torch.tensor(origins),env_origins=torch.zeros(n,3)),env_origins=torch.zeros(n,3))
    env.termination_manager = SimpleNamespace(terminated=torch.zeros(n,dtype=torch.bool),time_outs=torch.zeros(n,dtype=torch.bool))
    return env,platform


def test_balanced_all_24_cells_in_each_policy_population_and_material_independence():
    env,excluded = fake_motion();p=env.cts_motion
    assert not (p.mask & excluded).any()
    assert int((p.mask & p.teacher).sum()) == 54
    assert int((p.mask & ~p.teacher).sum()) == 18
    for _ in range(120):p.reset(torch.arange(env.num_envs))
    assert (p.cell_resets > 0).all() and (p.cell_profile_resets > 0).all()
    # Exact uniformity over all 24 classes after complete cycles, per cohort.
    assert p.cell_resets[0].unique().numel() == 1
    assert p.cell_resets[1].unique().numel() == 1
    assert p.profile_counts.sum() == p.resets.sum()
    assert TERRAINS == ('flat','slope','wave') and len(COMMANDS) == 8


def test_command_signs_arcs_and_zero_hold_stop_preserve_raw_requested_commands():
    env,_=fake_motion();p=env.cts_motion;p.reset(torch.arange(env.num_envs))
    term=SimpleNamespace(vel_command_b=torch.ones(env.num_envs,3),
                         is_standing_env=torch.zeros(env.num_envs,dtype=torch.bool),
                         is_heading_env=torch.ones(env.num_envs,dtype=torch.bool))
    p.command(term,slice(None));assert (term.vel_command_b[p.mask] == 0).all()
    env.episode_length_buf.fill_(30);p.command(term,slice(None))
    for mode in range(8):
        c=term.vel_command_b[p.mask & (p.mode == mode)]
        assert c.numel() and (c[:,1] == 0).all()
        if mode in (0,4,5): assert (c[:,0] > 0).all()
        elif mode in (1,6,7): assert (c[:,0] < 0).all()
        else: assert (c[:,0] == 0).all()
        if mode >= 2: assert ((c[:,2] > 0) if mode%2 == 0 else (c[:,2] < 0)).all()
    env.episode_length_buf.fill_(180);p.command(term,slice(None))
    assert (term.vel_command_b[p.mask] == 0).all()
    assert not term.is_heading_env[p.mask].any()


def test_mesh_height_normals_and_original_contact_domains_agree_with_true_triangles():
    cfg=default_settings()['motion_practice'];generator=SimpleNamespace(num_rows=2,num_cols=5,size=(12.,5.))
    mesh,origins=motion_patch_mesh(generator,cfg)
    domains=face_profile_ids(mesh,5,5.,5)
    assert set(domains) == set(range(5))
    assert (mesh.face_normals[:,2] > 0).all()
    rng=np.random.default_rng(42)
    # Independent triangle intersections; no mesh.ray optional dependency.
    for variant in range(len(variants(cfg))):
        for col in range(5):
            for x in rng.uniform(-4.5,4.5,3):
                z,normal=surface(torch.tensor([x],dtype=torch.float64),torch.tensor([variant]),cfg)
                hit=vertical_hit(mesh,[origins[variant,col,0]+x,origins[variant,col,1]])
                assert hit == pytest.approx(float(z),abs=1e-7)
                eps=1e-6
                zl,_=surface(torch.tensor([x-eps],dtype=torch.float64),torch.tensor([variant]),cfg)
                zr,_=surface(torch.tensor([x+eps],dtype=torch.float64),torch.tensor([variant]),cfg)
                grad=float((zr-zl)/(2*eps))
                np.testing.assert_allclose(normal.numpy()[0],[-grad/(1+grad**2)**.5,0,1/(1+grad**2)**.5],atol=1e-7)


def test_reset_uses_local_surface_height_tilt_and_cycles_grade_and_waves():
    env,_=fake_motion();p=env.cts_motion;seen=set()
    for _ in range(8):
        p.reset(torch.arange(env.num_envs));ids=p.mask.nonzero().flatten()
        seen.update(p.variant[ids].tolist())
        pose=env.scene['robot'].write_root_pose_to_sim.call_args.args[0]
        local=pose[:,:3]-env.scene.env_origins[ids]
        h,n=surface(local[:,0],p.variant[ids],p.cfg)
        torch.testing.assert_close((local[:,2]-h)*n[:,2],torch.full_like(h,.5),atol=2e-5,rtol=0)
        q=pose[:,3:]
        up=torch.stack((2*(q[:,1]*q[:,3]+q[:,0]*q[:,2]),2*(q[:,2]*q[:,3]-q[:,0]*q[:,1]),1-2*(q[:,1]**2+q[:,2]**2)),-1)
        torch.testing.assert_close(up,n,atol=1e-6,rtol=0)
        torch.testing.assert_close(q.norm(dim=-1),torch.ones(len(ids)))
    assert seen == set(range(len(variants(p.cfg))))


def physics_sample(variant,cfg,s,raise_m=0.):
    n=len(variant);pos=torch.zeros(n,4,3);pos[:,:,0]=torch.tensor([.25,.25,-.25,-.25])
    pos[:,:,1]=torch.tensor([.25,-.25,.25,-.25]);axis=torch.zeros_like(pos);axis[:,:,1]=1
    z,normal=surface(pos[...,0],variant,cfg)
    pos[...,2]=z+(s.wheel_radius+raise_m)/normal[...,2]
    return dict(position=pos,axis=axis,forces=normal*100,velocity=torch.zeros_like(pos),omega=torch.zeros_like(pos),
        root=torch.tensor([[0.,0.,.5]]).expand(n,-1).clone(),root_velocity=torch.zeros(n,3),yaw=torch.zeros(n))


def test_normal_support_and_lift_probe_distinguish_ground_following_from_airborne():
    cfg=default_settings()['motion_practice'];s=SafeSettings();variant=torch.arange(len(variants(cfg)))
    d=physics_sample(variant,cfg,s);g=wheel_geometry(d,variant,cfg,s)
    torch.testing.assert_close(g['clearance'],torch.zeros_like(g['clearance']),atol=1e-6,rtol=0)
    assert g['supported'].all()
    # Axis projected along the slope changes the capsule's normal extent.
    d['axis'][:,:,0]=1.;d['axis'][:,:,1]=0.
    z,n=surface(d['position'][...,0],variant,cfg)
    extent=s.wheel_radius+s.wheel_half_length*(d['axis']*n).sum(-1).abs()
    d['position'][...,2]=z+extent/n[...,2]
    assert wheel_geometry(d,variant,cfg,s)['supported'].all()
    lifted=physics_sample(variant,cfg,s,raise_m=.15);lifted['forces'].zero_()
    air=wheel_geometry(lifted,variant,cfg,s)
    assert not air['supported'].any()
    zero=torch.zeros(len(variant),12)
    good=rolling_costs(g['clearance'],g['supported'],zero,zero,cfg)
    bad=rolling_costs(air['clearance'],air['supported'],zero,zero,cfg)
    assert (bad > good+cfg['linear_reward']).all()


def test_true_no_slip_rolling_on_both_slope_signs_and_waves_has_zero_slip():
    cfg=default_settings()['motion_practice'];s=SafeSettings();variant=torch.arange(len(variants(cfg)))
    for heading in (0,1):
        d=physics_sample(variant,cfg,s)
        d['axis'].zero_();d['axis'][...,heading]=1
        _,normal=surface(d['position'][...,0],variant,cfg)
        offset=-s.wheel_radius*normal-s.wheel_half_length*torch.sign((d['axis']*normal).sum(-1))[...,None]*d['axis']
        d['omega']=d['axis']*5
        d['velocity']=-torch.linalg.cross(d['omega'],offset)
        g=wheel_geometry(d,variant,cfg,s)
        torch.testing.assert_close(g['slip'],torch.zeros_like(g['slip']),atol=1e-7,rtol=0)


def reward_env(n=480):
    env,_=fake_motion(n);p=env.cts_motion;p.reset(torch.arange(env.num_envs))
    n=env.num_envs;robot=env.scene['robot'].data
    robot.root_lin_vel_b=p.command_value.clone();robot.root_lin_vel_b[:,2]=0
    robot.root_ang_vel_b=torch.zeros(n,3);robot.root_ang_vel_b[:,2]=p.command_value[:,2]
    robot.root_quat_w=torch.zeros(n,4);robot.root_quat_w[:,0]=1
    robot.joint_pos=torch.zeros(n,16);robot.default_joint_pos=torch.zeros(n,16);robot.applied_torque=torch.zeros(n,16)
    env._cts_motion_leg_ids=list(range(12))
    env.command_manager=SimpleNamespace(get_command=lambda _:p.command_value)
    env.action_manager=SimpleNamespace(action=torch.zeros(n,16),prev_action=torch.zeros(n,16))
    d=physics_sample(p.variant,p.cfg,env.cfg.stair_tail)
    return env,d


def test_actual_reward_only_preserves_straight_flat_motion_and_counts_real_frames_once():
    # At the v13 15% motion allocation, 640 environments retain one student
    # environment per each of the 24 motion cells in this single-frame probe.
    env,d=reward_env(640);p=env.cts_motion
    with patch('task_support.student_v3_safe.terrain_observations.physical_sample',return_value=d):
        good=motion_reward(env)
        d['position'][...,2] += .15;d['forces'].zero_();env.common_step_counter += 1
        bad=motion_reward(env)
        rolling=p.mask & (p.kind == 0) & (p.mode < 2)
        assert (bad[rolling] < good[rolling]).all()
        torch.testing.assert_close(bad[p.mask & ~rolling],good[p.mask & ~rolling])
        assert (good[~p.mask] == 0).all()
        before=p.cell_frames.clone();motion_reward(env);assert torch.equal(before,p.cell_frames)
        assert (p.cell_frames > 0).all()
        # Wheel positions/commands are not included in the leg posture target.
        robot=env.scene['robot'].data;robot.joint_pos[:,12:]=100
        torch.testing.assert_close(motion_reward(env),bad)
        feature=motion_safety_features(env);assert feature.shape == (env.num_envs,48)
        assert torch.isfinite(feature).all()


def test_boundary_is_relative_to_real_ground_and_timeout_is_distinct():
    env,_=fake_motion();p=env.cts_motion;p.reset(torch.arange(env.num_envs))
    robot=env.scene['robot'].data
    robot.root_pos_w=env.scene.env_origins.clone();robot.root_pos_w[:,2]=.5
    assert not motion_boundary(env).any()
    selected=p.mask.nonzero().flatten();robot.root_pos_w[selected[0],0] += p.cfg['patch_length']
    robot.root_pos_w[selected[1],2]=.05
    assert motion_boundary(env)[selected[:2]].all()
    env.episode_length_buf.fill_(201);assert torch.equal(motion_timeout(env),p.mask)


def test_stair_mask_view_and_resume_reject_unknown_or_negative_motion_accounting():
    env,excluded=fake_motion();p=env.cts_motion
    env.cts_practice_mask=excluded | p.mask
    wrapped=stair_only(lambda _:torch.ones(env.num_envs))
    assert (wrapped(env)[env.cts_practice_mask] == 0).all()
    backing={'height':torch.full((env.num_envs,),.15),'width':torch.ones(env.num_envs)}
    view=FlatCourseState(backing,p)
    assert (view['height'][p.mask] == 0).all()
    assert (view['width'][p.mask] == p.cfg['patch_width']).all()
    state=p.snapshot();p.restore(state)
    for key,value in (('mask',~p.mask),('cell_frames',torch.full_like(p.cell_frames,-1))):
        wrong=deepcopy(state);wrong[key]=value
        with pytest.raises(ValueError):p.restore(wrong)


def test_v8_deploy_schema_remains_supported_and_current_config_requires_real_motion():
    cfg=default_settings();validate_config(cfg)
    assert VERSION.endswith('_v26') and 'm20_cts_action_latent_v8' in LEGACY_VERSIONS
    cfg.pop('motion_practice');cfg.pop('straight_practice');validate_config(cfg,legacy=True)
    with pytest.raises(ValueError):validate_config(cfg)
    for key,value in (('patch_width',1.),('patch_length',1.),('fraction',True),('slope_degrees',[6,12]),
                      ('wave_lengths_m',[0.]),('mesh_resolution',2.),('yaw_rate_range',[1.,0.])):
        wrong=deepcopy(default_settings()['motion_practice']);wrong[key]=value
        with pytest.raises(ValueError):validate_motion(wrong)


def test_wait_and_stop_frames_are_distinct_and_use_zero_commands_on_every_surface():
    env,d=reward_env();p=env.cts_motion
    p.command_value.zero_()
    with patch('task_support.student_v3_safe.terrain_observations.physical_sample',return_value=d):
        motion_reward(env)
        assert (p.wait_frames > 0).all() and not p.stop_frames.any()
        assert not p.cell_frames.any()
        env.episode_length_buf.fill_(180);env.common_step_counter += 1
        motion_reward(env)
        assert (p.stop_frames > 0).all()
        torch.testing.assert_close(p.idle_frames,p.wait_frames+p.stop_frames)
        before=p.stop_frames.clone();motion_reward(env)
        assert torch.equal(before,p.stop_frames)
