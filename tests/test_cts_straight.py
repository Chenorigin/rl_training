"""Real course geometry/control and same-state student-only action gradients."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch
import pytest
import torch

from test_cts_algorithm import fixture
from test_cts_motion import fake_motion, physics_sample
from test_cts_platform import vertical_hit
from task_support.cts.configuration import default_settings, validate_config, VERSION, LEGACY_VERSIONS
from task_support.cts.motion import MotionPractice, motion_timeout, motion_boundary, motion_reward
from task_support.cts.straight import validate_straight, straight_patch_mesh, straight_action_loss, straight_teacher_actions
from task_support.cts.entry import auxiliary_memory_budget
from task_support.stair_quiet_v1_width.geometry import face_profile_ids


def straight_env(n=480):
    env,excluded = fake_motion(n);cfg=default_settings()
    env.cts_motion = MotionPractice(env,cfg['motion_practice'],cfg['algorithm'],excluded,cfg['straight_practice'])
    mesh,origins=straight_patch_mesh(SimpleNamespace(num_rows=2,num_cols=5,size=(12.,5.)),
                                    cfg['motion_practice'],cfg['straight_practice'])
    env.scene.terrain.cts_straight_origins=torch.tensor(origins)
    return env,excluded,mesh


def test_five_percent_is_carved_out_of_motion_both_cohorts_with_new_fraction():
    env,excluded,_=straight_env();p=env.cts_motion
    assert (p.straight_mask & ~p.mask).sum()==0
    assert (p.straight_mask & excluded).sum()==0
    assert int(p.straight_mask.sum())==24
    assert int((p.straight_mask & p.teacher).sum())==18
    assert int((p.straight_mask & ~p.teacher).sum())==6
    assert int((p.mask & ~p.straight_mask).sum())==48
    # The standalone compatibility wrapper leaves 80% outside motion/platform;
    # the v13 scheduler divides that remainder into 70% stairs and 10% new fields.
    assert int((~(p.mask | excluded)).sum())==384
    for _ in range(48):
        p.reset(torch.arange(env.num_envs))
        assert (p.kind[p.straight_mask]==0).all() and (p.mode[p.straight_mask]==0).all()
        assert (p.command_value[p.straight_mask,0]>0).all()
        assert (p.command_value[p.straight_mask,1:]==0).all()
        assert torch.isinf(p.push_at[p.straight_mask]).all()
    # Ordinary motion still cycles all 24 terrain/command cells uniformly.
    resets=p.cell_resets.clone();resets[:,0,0]-=torch.tensor([18,6])*48
    assert (resets>0).all() and resets[0].unique().numel()==1 and resets[1].unique().numel()==1


def test_true_long_mesh_contact_profiles_and_boundary_timing():
    env,_,mesh=straight_env();p=env.cts_motion;p.reset(torch.arange(env.num_envs))
    assert set(face_profile_ids(mesh,5,5.,5))==set(range(5))
    for o in env.scene.terrain.cts_straight_origins.numpy():
        for x in (-7.,0.,7.):assert vertical_hit(mesh,[o[0]+x,o[1]])==pytest.approx(0.)
    assert (mesh.face_normals[:,2]>0).all()
    term=SimpleNamespace(vel_command_b=torch.zeros(env.num_envs,3),is_standing_env=torch.ones(env.num_envs,dtype=torch.bool),is_heading_env=torch.ones(env.num_envs,dtype=torch.bool))
    for age in (.0,.4,4.0,4.02,8.38,8.4,8.42,9.0,9.02):
        env.episode_length_buf.fill_(round(age/.02));p.command(term,slice(None))
        if .4<=age<8.4:assert (term.vel_command_b[p.straight_mask,0]>0).all()
        else:assert (term.vel_command_b[p.straight_mask]==0).all()
        if age>=4.:assert motion_timeout(env)[p.mask & ~p.straight_mask].all()
        assert bool(motion_timeout(env)[p.straight_mask].all())==(age>=9.)
    data=env.scene['robot'].data;data.root_pos_w=env.scene.env_origins.clone();data.root_pos_w[:,2]=.5
    data.root_pos_w[:,0]+=6.
    assert not motion_boundary(env)[p.straight_mask].any()
    assert motion_boundary(env)[p.mask & ~p.straight_mask].all()


def test_course_validation_fails_short_patch_and_rejects_missing_training_contract():
    cfg=default_settings();validate_config(cfg)
    assert VERSION.endswith('_v26') and 'm20_cts_action_latent_v12' in LEGACY_VERSIONS
    old=deepcopy(cfg);old.pop('straight_practice');validate_config(old,legacy=True)
    with pytest.raises(ValueError):validate_config(old)
    for key,value in [('fraction',.25),('patch_length',10.),('move_s',3.),('move_s',100.),('batch_size',True),('leg_action_loss_weight',float('nan'))]:
        bad=deepcopy(cfg['straight_practice']);bad[key]=value
        with pytest.raises(ValueError):validate_straight(bad,cfg['motion_practice'])
    with pytest.raises(ValueError,match='increase num_envs'):straight_env(8)
    no=deepcopy(cfg);no.pop('straight_practice')
    assert auxiliary_memory_budget(cfg)['auxiliary_reserve_bytes']>auxiliary_memory_budget(no)['auxiliary_reserve_bytes']


def action_data():
    models,alg,obs,*_=fixture();cfg=default_settings()['straight_practice']
    obs['motion_control'][:,1:]=0
    with torch.no_grad():
        models.decoder.network[-1].weight.fill_(.025)
        target=straight_teacher_actions(models,obs)
    mask=torch.ones(8,dtype=torch.bool)
    data=dict(obs=obs,teacher=alg.teacher_mask,straight=mask)
    return models,alg,data,target,cfg


def test_decoded_leg_loss_ignores_unrelated_cohort_and_wheels_and_reaches_encoder():
    models,alg,data,target,cfg=action_data()
    for p in models.decoder.parameters():p.requires_grad_(False)
    loss,metrics=straight_action_loss(models,data,target,cfg)
    assert metrics['straight_action_samples']==2
    student=models.student_action(data['obs']['policy'],data['obs']['motion_control'])
    torch.testing.assert_close(loss,((student-target)[~data['teacher'],:12]).square().mean())
    changed=target.clone();changed[data['teacher']]=999;changed[:,12:]=888
    torch.testing.assert_close(straight_action_loss(models,data,changed,cfg)[0],loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in models.student.mlp.parameters())
    assert all(p.grad is None for p in models.decoder.parameters())
    assert all(p.grad is None for p in models.teacher.parameters())
    data['obs']['motion_control'][:,2]=.4
    assert straight_action_loss(models,data,target,cfg)[1]['straight_action_samples']==0


def test_actual_distill_stage_calls_new_loss_current_teacher_and_preserves_ppo_parameters():
    models,alg,obs,*_=fixture();alg.straight_cfg=default_settings()['straight_practice']
    obs['motion_control'][:,1:]=0
    with torch.inference_mode():
        for _ in range(4):
            alg.act(obs)
            alg.process_env_step(obs,torch.ones(8),torch.zeros(8,dtype=torch.bool),{'cts_straight':torch.ones(8,dtype=torch.bool)})
        alg.compute_returns(obs)
    before=deepcopy(models.state_dict())
    with patch('task_support.cts.straight.straight_action_loss',wraps=straight_action_loss) as call:
        result=alg.distill_update()
        assert call.call_count>0 and result['straight_action_samples']>0
    assert any(not torch.equal(v,models.state_dict()[k]) for k,v in before.items() if k.startswith('student.mlp.'))
    assert all(torch.equal(v,models.state_dict()[k]) for k,v in before.items() if not k.startswith('student.'))
    alg.clear_rollout();saved=deepcopy(alg.save());saved['straight_contract']['move_s']+=1
    with pytest.raises(ValueError,match='contract'):alg.load(saved)


def test_fail_closed_on_missing_tags_and_pre_step_course_tensor_is_cloned():
    _,alg,obs,*_=fixture();alg.straight_cfg=default_settings()['straight_practice'];alg.act(obs)
    with pytest.raises(ValueError,match='pre-step'):alg.process_env_step(obs,torch.ones(8),torch.zeros(8,dtype=torch.bool),{})
    tag=torch.ones(8,dtype=torch.bool)
    alg.process_env_step(obs,torch.ones(8),torch.zeros(8,dtype=torch.bool),{'cts_straight':tag});tag.zero_()
    assert alg.rollout[0]['straight'].all()


def test_snapshot_keeps_fixed_membership_and_wheel_metrics_detect_left_front_lift():
    env,_,_=straight_env();p=env.cts_motion;p.reset(torch.arange(env.num_envs))
    n=env.num_envs;robot=env.scene['robot'].data
    robot.root_lin_vel_b=p.command_value.clone();robot.root_lin_vel_b[:,2]=0
    robot.root_ang_vel_b=torch.zeros(n,3);robot.root_ang_vel_b[:,2]=p.command_value[:,2]
    robot.root_quat_w=torch.zeros(n,4);robot.root_quat_w[:,0]=1
    robot.joint_pos=robot.default_joint_pos=torch.zeros(n,16);robot.applied_torque=torch.zeros(n,16)
    env._cts_motion_leg_ids=list(range(12));env.command_manager=SimpleNamespace(get_command=lambda _:p.command_value)
    env.action_manager=SimpleNamespace(action=torch.zeros(n,16),prev_action=torch.zeros(n,16))
    d=physics_sample(p.variant,p.cfg,env.cfg.stair_tail)
    student=p.straight_mask & ~p.teacher;d['position'][student,0,2]+=.15;d['forces'][student,0]=0
    with patch('task_support.student_v3_safe.terrain_observations.physical_sample',return_value=d):
        motion_reward(env);metrics=p.straight_diagnostics()
        assert metrics['student_FL_clearance_peak_m']==pytest.approx(.15)
        assert metrics['student_FL_airborne_fraction']==1.
        assert metrics['teacher_FL_clearance_peak_m']<1e-6
        count=p.straight_frames.clone();motion_reward(env);assert torch.equal(count,p.straight_frames)
    saved=p.snapshot();p.straight_frames.zero_();p.restore(saved);assert torch.equal(count,p.straight_frames)
    wrong=deepcopy(saved);wrong['straight_mask']=~wrong['straight_mask']
    with pytest.raises(ValueError):p.restore(wrong)


def test_clean_teacher_target_is_independent_of_student_sensor_and_proprio_noise():
    models,_,data,_,_=action_data();obs=data['obs']
    before=straight_teacher_actions(models,obs)
    obs['policy']+=torch.randn_like(obs['policy'])*.5
    torch.testing.assert_close(straight_teacher_actions(models,obs),before,rtol=0,atol=0)
    # This oracle rejects the previous shared-decoder common-input path.
    with torch.no_grad():wrong=models.decoder(obs['policy'][:,:57],models.teacher_latent(obs),obs['motion_control'])
    assert (wrong-before).abs().max()>1e-5
    assert not before.requires_grad


def test_previous_v11_deploy_contract_survives_but_cannot_resume_new_curriculum():
    from test_cts_entry import CTSMetadataTests
    from task_support.cts.entry import validate_metadata
    meta=CTSMetadataTests().meta();meta['version']='m20_cts_action_latent_v11';meta['config'].pop('straight_practice')
    meta['config'].pop('all_terrain')
    meta['config'].pop('joint_limits')
    meta['config'].pop('map_recovery')  # v11 predates this v15-only feature.
    meta['perception'].pop('continuous_delay')
    meta['environment']['perception']=deepcopy(meta['perception'])
    validate_metadata(meta,resume=False)
    with pytest.raises(ValueError,match='compatible'):validate_metadata(meta)
    meta['perception']['map_capture_padding_m']=0.
    meta['environment']['perception']=deepcopy(meta['perception'])
    with pytest.raises(ValueError,match='map capture'):validate_metadata(meta,resume=False)


def test_actual_env_step_tags_pre_reset_state_without_importing_isaac():
    # Run the real method body against a controlled reset to prove tags do not
    # come from the post-step episode or alias its mutable course mask.
    import ast
    from pathlib import Path
    from unittest.mock import Mock
    path=Path(__file__).resolve().parents[1]/'source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/cts_env.py'
    tree=ast.parse(path.read_text());cls=next(x for x in tree.body if isinstance(x,ast.ClassDef))
    step=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='step')
    # Omit log/sensor observations after the two real statements of interest.
    boundary=next(i for i,x in enumerate(step.body) if isinstance(x,ast.Assign) and any(isinstance(t,ast.Subscript) and isinstance(t.slice,ast.Constant) and t.slice.value=='cts_straight' for t in x.targets))
    step.body=step.body[:boundary+1]+[ast.Return(value=ast.Name(id='result',ctx=ast.Load()))]
    class Base:
        def step(self,action):
            self.mask.zero_();self.extras={};return (None,None,None,self.extras)
    module=ast.Module(body=[ast.ClassDef(name='Probe',bases=[ast.Name(id='Base',ctx=ast.Load())],keywords=[],body=[step],decorator_list=[])],type_ignores=[])
    ns={'Base':Base};exec(compile(ast.fix_missing_locations(module),str(path),'exec'),ns)
    env=ns['Probe']();env.mask=torch.tensor([False,True]);env.extras={}
    env.cts_motion=SimpleNamespace(straight_active=lambda:env.mask.clone(),push=Mock())
    env.cts_platform=SimpleNamespace(begin_step=Mock(),push=Mock())
    result=env.step(torch.zeros(2,16))
    assert torch.equal(result[3]['cts_straight'],torch.tensor([False,True])) and not env.mask.any()
