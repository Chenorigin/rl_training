"""Physical drop geometry, honest sensing and small-step counterexamples."""
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source/rl_training'))
from task_support.cts.configuration import default_settings, validate_config, algorithm_version, LEGACY_VERSIONS
from task_support.cts.practice import cohort_mask, FlatCourseState
from task_support.cts.platform import (PlatformPractice, validate_platform, platform_patch_mesh,
    physical_heights,height_risk,fall_evidence,platform_geometry,platform_safety_features,
    platform_raw_scan,platform_failure,platform_timeout,platform_reward)
from task_support.stair_quiet_v1_safe.settings import SafeSettings
from task_support.stair_quiet_v1_width.geometry import face_profile_ids


def make_platform(n=80):
    settings = default_settings()
    env = SimpleNamespace(cfg=SimpleNamespace(scene=SimpleNamespace(num_envs=n),sim=SimpleNamespace(device='cpu'),
        stair_tail=SafeSettings()),device='cpu',num_envs=n,step_dt=.02,episode_length_buf=torch.zeros(n,dtype=torch.long))
    teacher = cohort_mask(n,.75,42)
    excluded = torch.zeros_like(teacher)
    env.cts_platform = PlatformPractice(env,settings['platform_practice'],settings['algorithm'],excluded)
    return env,excluded


def sample(n):
    pos = torch.tensor([[.25,.25,.09],[.25,-.25,.09],[-.25,.25,.09],[-.25,-.25,.09]]).expand(n,-1,-1).clone()
    axis = torch.zeros_like(pos);axis[...,1] = 1
    force = torch.zeros_like(pos);force[...,2] = 100
    return dict(position=pos,axis=axis,velocity=torch.zeros_like(pos),omega=torch.zeros_like(pos),forces=force,
        root=torch.tensor([[0.,0.,.5]]).expand(n,-1).clone(),root_velocity=torch.zeros(n,3),
        yaw=torch.zeros(n),upright=torch.ones(n))


def vertical_hit(mesh, xy):
    # Independent vertical triangle intersection, no optional rtree dependency.
    triangles = mesh.triangles
    a = triangles[:,0,:2];u = triangles[:,1,:2]-a;v = triangles[:,2,:2]-a
    det = u[:,0]*v[:,1]-u[:,1]*v[:,0]
    good = np.abs(det) > 1e-10
    point = np.asarray(xy)-a[good]
    alpha = (point[:,0]*v[good,1]-point[:,1]*v[good,0])/det[good]
    beta = (u[good,0]*point[:,1]-u[good,1]*point[:,0])/det[good]
    hit = (alpha >= -1e-7) & (beta >= -1e-7) & (alpha+beta <= 1+1e-7)
    z = triangles[good,0,2]+alpha*(triangles[good,1,2]-triangles[good,0,2])+beta*(triangles[good,2,2]-triangles[good,0,2])
    return z[hit].max() if hit.any() else -np.inf


def test_exclusive_cohorts_and_tiny_batch_fails_before_simulator():
    env,excluded = make_platform()
    p = env.cts_platform
    assert not (p.mask & excluded).any()
    assert int((p.mask & p.teacher).sum()) == 3
    assert int((p.mask & ~p.teacher).sum()) == 1
    with pytest.raises(ValueError,match='disjoint'):
        from task_support.cts.motion import MotionPractice
        tiny,_ = make_platform(4)
        settings = default_settings()
        MotionPractice(tiny,settings['motion_practice'],settings['algorithm'],tiny.cts_platform.mask)


def test_true_heights_are_stratified_reproducible_and_fill_the_old_gap():
    cfg = default_settings()['platform_practice']
    generator = SimpleNamespace(num_rows=2,num_cols=5,size=(22.,5.))
    heights = physical_heights(generator,cfg)
    np.testing.assert_array_equal(heights,physical_heights(generator,cfg))
    changed = deepcopy(cfg);changed['terrain_seed'] += 1
    assert not np.array_equal(heights,physical_heights(generator,changed))
    count = cfg['samples_per_band']
    for band,(lo,hi) in enumerate(cfg['height_bands']):
        for sub in range(count):
            values = heights[band*count+sub]
            assert (values >= lo+(hi-lo)*sub/count).all()
            assert (values < lo+(hi-lo)*(sub+1)/count).all()
    assert ((heights > .12) & (heights < .55)).any()


def test_real_reset_assignment_covers_all_bands_strata_and_approach_modes():
    env,_ = make_platform();p = env.cts_platform
    cfg = p.cfg;generator = SimpleNamespace(num_rows=2,num_cols=5,size=(22.,5.))
    _,origins = platform_patch_mesh(generator,cfg,10.)
    env.scene = SimpleNamespace(terrain=SimpleNamespace(cts_platform_origins=torch.tensor(origins)))
    ids = p.mask.nonzero().flatten();types = ids%5
    route_counts = torch.zeros(16,2,dtype=torch.long)
    for _ in range(64):
        previous = p.variant_resets.clone()
        origin = p.reset_origins(ids,types)
        assert (origin[:,2] == p.height[ids]).all()
        cycle = p.resets[ids]+p.rank[ids]
        variant = p.band[ids]*4+(cycle//4)%4
        for which,route in zip(variant,p.approach[ids]):route_counts[which,int(route)] += 1
        assert int((p.variant_resets-previous).sum()) == len(ids)
        p.resets[ids] += 1
    assert (p.variant_resets == len(ids)*4).all()
    assert (p.band_resets == len(ids)*16).all()
    assert (route_counts[12:]>0).all()  # Every high stratum sees both routes.


def test_risk_and_descent_weights_are_smooth_at_every_band_boundary():
    cfg = default_settings()['platform_practice']
    for boundary in (.12,.25,.55):
        h = torch.tensor([boundary-1e-6,boundary,boundary+1e-6],dtype=torch.float64)
        weights = height_risk(h,cfg)
        assert (weights[1:]-weights[:-1]).abs().max() < 1e-5
        assert ((1-weights)[1:]-(1-weights)[:-1]).abs().max() < 1e-5


def test_fall_is_physical_and_latches_before_landing_erases_speed():
    cfg = default_settings()['platform_practice']
    age = torch.zeros(2);peak=torch.zeros(2)
    # Two real supports while stepping down never count as free fall.
    for _ in range(30):
        age,peak,fell = fall_evidence(age,peak,torch.tensor([2,0]),torch.tensor([-.5,-1.5]),.005,cfg)
    assert not fell[0] and fell[1]
    # A landing cannot erase the previous episode of fast unsupported descent.
    _,_,fell = fall_evidence(age,peak,torch.tensor([2,4]),torch.zeros(2),.005,cfg)
    assert not fell[0] and fell[1]


def test_supported_descent_ends_edge_penalty_and_events_are_paid_only_once():
    env,_ = make_platform();p=env.cts_platform;env.physics_dt=.005
    ids=p.mask.nonzero().flatten()
    p.height.fill_(.25);p.start_root_height.fill_(.5)
    d=sample(80);d['position'][:,:,0] += 6;d['position'][:,:,2] -= .25
    d['root'][:,0]=6;d['root'][:,2]=.25
    with patch('task_support.student_v3_safe.terrain_observations.physical_sample',return_value=d):
        p.sample_physics()
        assert p.descended[ids].all() and p.descent_event[ids].all()
        assert (p.risk[ids] == 0).all() and not p.fell[ids].any()
        assert p.departure_event[ids].all()
        p.begin_step();p.sample_physics()
        assert not p.descent_event.any() and not p.departure_event.any()
        assert (p.risk[ids] == 0).all()


def test_legacy_v5_platform_metadata_remains_deployable_but_not_current_configuration():
    config=default_settings();old=deepcopy(config['platform_practice'])
    for key in ('height_bands','samples_per_band','terrain_seed','risk_height_range','approach_speed_range',
                'fall_speed_m_s','fall_grace_s','impact_cost','departure_cost','stop_speed_m_s'):
        old.pop(key)
    old.update(high_heights=[.55,.8,1.],low_heights=[.04,.08,.12],low_every=4,
               low_speed_range=[.45,.6],fall_clearance=.12)
    config['platform_practice']=old
    validate_config(config,legacy=True)
    with pytest.raises(ValueError):validate_config(config)
    old.pop('low_speed_range')  # Actual early v5 schema, before the speed fix.
    validate_config(config,legacy=True)
    assert 'low_speed_range' not in old  # Validation must not rewrite metadata.


def test_physical_mesh_has_one_meter_top_real_catch_floor_and_all_profiles():
    cfg = default_settings()['platform_practice']
    generator = SimpleNamespace(num_rows=2,num_cols=5,size=(22.,5.))
    mesh,origins = platform_patch_mesh(generator,cfg,10.)
    variants = len(cfg['height_bands'])*cfg['samples_per_band']
    assert origins.shape == (variants,2,5,3)
    assert len(np.unique(origins.reshape(-1,3),axis=0)) == variants*10
    assert mesh.is_watertight
    for hi in range(variants):
        for col in range(5):
            origin = origins[hi,0,col]
            height = origin[2]
            assert vertical_hit(mesh,origin[:2]) == pytest.approx(height)
            assert vertical_hit(mesh,origin[:2]+[0,.49]) == pytest.approx(height)
            assert vertical_hit(mesh,origin[:2]+[0,.51]) == pytest.approx(0.)
            assert vertical_hit(mesh,origin[:2]+[5.5,0]) == pytest.approx(0.)
    profiles = face_profile_ids(mesh,5,5.,5)
    slab = (mesh.triangles_center[:,2]>0) & (mesh.face_normals[:,2]>.9)
    assert np.bincount(profiles[slab],minlength=5).tolist() == [4*variants]*5


def test_height_domains_and_run_config_validation_reject_ambiguous_or_impossible_courses():
    cfg = default_settings()['platform_practice']
    validate_platform(cfg)
    for key,value in (('height_bands',[[.04,.12],[.13,.25],[.25,.55],[.55,1.]]),('samples_per_band',1),('width',0.),
                      ('fraction',.5),('speed_range',[-.5,.5]),('patch_length',1.),('reset_xy_range',[0.,.6])):
        wrong = deepcopy(cfg);wrong[key] = value
        with pytest.raises(ValueError):validate_platform(wrong)
    legacy = default_settings();legacy.pop('platform_practice')
    validate_config(legacy,legacy=True)
    with pytest.raises(ValueError):validate_config(legacy)
    assert algorithm_version(LEGACY_VERSIONS[-1]) == 3


def test_low_descent_minimum_command_can_clear_the_actual_rear_footprint():
    cfg = default_settings()['platform_practice']
    s = SafeSettings()
    start = cfg['patch_length']/2-cfg['low_start_from_end']-cfg['reset_xy_range'][0]
    # Worst rear centre uses the full conservative body length, not base-only.
    rear_end = start+cfg['approach_speed_range'][0]*cfg['move_s']*np.cos(cfg['reset_yaw_range'])-s.body_half_length
    assert rear_end-s.wheel_radius-s.wheel_half_length > cfg['patch_length']/2
    bad = deepcopy(cfg);bad['approach_speed_range'] = [.25,.3]
    with pytest.raises(ValueError,match='rear wheels'):validate_platform(bad)


def test_same_edge_has_continuous_risk_and_height_alone_cannot_terminate():
    cfg = default_settings()['platform_practice'];d = sample(5)
    d['root'][:,1] = .4;d['position'][:,:,1] += .4
    result = platform_geometry(d,cfg,SafeSettings(),torch.tensor([.04,.12,.25,.4,.8]))
    risk = result['risk']
    assert risk[0] == risk[1] == 0
    assert risk[1] < risk[2] < risk[3] < risk[4]
    assert 'fall' not in result
    assert (risk <= 1).all()


def test_teacher_support_reports_real_low_floor_instead_of_invented_platform_height():
    cfg = default_settings()['platform_practice'];d = sample(2);s = SafeSettings()
    height = torch.tensor([.08,.8]);d['position'][:,:,0] += 6
    d['position'][:,:,2] -= height[:,None]
    features = platform_safety_features(d,cfg,s,height,5.)
    feet = features[:,:44].reshape(2,4,11)
    assert feet[:,:,6].all()  # Real floor force/height agrees in BOTH domains.
    assert feet[0,:,0].min() > 0  # Low floor is a permitted continuation.
    assert feet[1,:,0].min() > 0  # Actual support geometry stays true at any height.


def test_course_cache_low_floor_spans_survive_reassignment():
    env,_ = make_platform();p = env.cts_platform
    ids = p.mask.nonzero().flatten();p.height[ids[0]] = .08
    state = dict(width=torch.ones(80),height=torch.ones(80),steps=torch.ones(80,dtype=torch.long),
        spans=torch.zeros(80,4,3),round_trip=torch.zeros(80,dtype=torch.bool),goal=torch.ones(80),
        goal_height=torch.ones(80),ascent_end=torch.ones(80))
    view = FlatCourseState(state,p)
    for _ in range(3):
        view['spans'] = torch.ones(80,4,3)
        assert view['spans'][ids[0],2].tolist() == pytest.approx([5.,9.,-.08])
        assert view['width'][ids[1]] == 1.
        assert view['width'][ids[0]] == 1.
        assert view['round_trip'][p.mask].all()


def test_raw_teacher_scan_retains_exterior_drop_and_preserves_other_envs():
    from task_support.stair_quiet_v1_safe.observations import longitudinal_scan
    # Same transverse ray row: central elevated slab, low exterior.
    y = torch.linspace(-.3,.7,11)[None].expand(2,-1)
    raw = torch.where(y.abs()>.5,.7,-.1)
    filtered = longitudinal_scan(raw,torch.ones_like(raw),y,torch.ones(2))
    assert (filtered[0,y[0]>.5] == -.1).all()  # Explicit false-green control.
    result = platform_raw_scan(raw,filtered,torch.tensor([True,False]))
    torch.testing.assert_close(result[0],raw[0]);torch.testing.assert_close(result[1],filtered[1])


def test_small_drop_is_not_terminal_and_timeout_has_separate_semantics():
    env,_ = make_platform();p = env.cts_platform
    class Scene(SimpleNamespace):
        def __getitem__(self,key):return self.robot
    env.scene = Scene(env_origins=torch.zeros(80,3),robot=SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.zeros(80,3))))
    env.scene.robot.data.root_pos_w[:,2] = .42
    p.height.fill_(.08)
    env.scene.robot.data.root_pos_w[p.mask,0] = 5.8
    assert not platform_failure(env).any()
    ids = p.mask.nonzero().flatten();p.fell[ids[0]] = True
    assert platform_failure(env)[ids[0]]
    env.episode_length_buf.fill_(201)
    assert torch.equal(platform_timeout(env),p.mask)


def test_full_practice_accounting_roundtrip_and_corruption_rejection():
    env,_ = make_platform();p = env.cts_platform
    p.band_descents[:] = 2;p.variant_resets[:] = 3
    saved = p.snapshot();p.band_descents.zero_();p.restore(saved)
    assert (p.band_descents == 2).all()
    bad = deepcopy(saved);bad['band_descents'][0] = -1
    with pytest.raises(ValueError):p.restore(bad)


def test_terminal_penalty_survives_dt_scaling_and_fall_cannot_collect_descent_bonus():
    env,_ = make_platform();p = env.cts_platform
    class Scene(SimpleNamespace):
        def __getitem__(self,key):return self.robot
    robot = SimpleNamespace(root_pos_w=torch.zeros(80,3),root_lin_vel_b=torch.zeros(80,3),
        root_ang_vel_b=torch.zeros(80,3),projected_gravity_b=torch.tensor([[0.,0.,-1.]]).expand(80,-1),
        applied_torque=torch.zeros(80,16))
    env.scene = Scene(env_origins=torch.zeros(80,3),robot=SimpleNamespace(data=robot))
    env.action_manager = SimpleNamespace(action=torch.zeros(80,16),prev_action=torch.zeros(80,16))
    env.command_manager = SimpleNamespace(get_command=lambda name:torch.zeros(80,3))
    failed = p.mask.clone()
    env.termination_manager = SimpleNamespace(active_terms=['cts_platform_failure'],
        get_term_cfg=lambda name:SimpleNamespace(time_out=False),get_term=lambda name:failed)
    p.descent_event[p.mask] = True
    reward = platform_reward(env)*env.step_dt
    assert (reward[p.mask] < -p.cfg['failure_cost']+.1).all()
    failed.zero_()
    reward = platform_reward(env)*env.step_dt
    assert (reward[p.mask] > p.cfg['descent_bonus']).all()
    # Leaving the platform is increasingly costly with true drop height,
    # even if a policy manages to land upright and then exploit the wide floor.
    ids=p.mask.nonzero().flatten()[:3]
    p.height[ids]=torch.tensor([.08,.25,.8]);p.departure_event[ids]=True
    reward=platform_reward(env)*env.step_dt
    assert reward[ids[0]] > reward[ids[1]] > reward[ids[2]]
    assert reward[ids[0]] > 0 and reward[ids[2]] < 0
    # A dangerous-edge stop is permitted without rewriting the raw request.
    command=torch.zeros(80,3);command[:,0]=.6
    env.command_manager.get_command=lambda name:command
    p.risk[ids]=torch.tensor([0.,.5,1.]);p.descent_event.zero_();p.departure_event.zero_()
    before=command.clone();platform_reward(env)
    torch.testing.assert_close(before,command)
    assert p.hazard_stop_samples > 0


def test_real_sensor_surrogate_keeps_unknown_distinct_from_measured_drop():
    from task_support.student_v1.configuration import StudentV1PerceptionCfg
    from task_support.student_v1.perception import StudentObservationState
    cfg = StudentV1PerceptionCfg(level=0.,clean_probability=1.,stress_probability=0.,front_only=True,
        min_delay_s=0.,max_delay_s=0.,update_jitter_s=0.)
    state = StudentObservationState(1,'cpu',.02,cfg,history_steps=4)
    base = torch.tensor([[0.,.2,1.3]])
    hits = torch.cat((state.grid[None]+base[:,None,:2],torch.zeros(1,187,1)),-1)
    hits[:,:,2] = torch.where(hits[:,:,1].abs()<=.5,.8,0.)
    hits[0,-1] = torch.inf
    rotation = torch.eye(3)[None];proprio=torch.zeros(1,57);proprio[:,5] = -1
    for step in range(12):output = state.observe(proprio,hits,base,torch.zeros(1),rotation,step)
    height,valid,confidence = output[:,57:244],output[:,244:431]>.5,output[:,431:618]
    assert valid.any() and (~valid).any()
    assert (height[~valid] == 0).all() and (confidence[~valid] == 0).all()
    assert (confidence[valid] > 0).all()
    # These channels describe height DIFFERENCE from the body. A drop makes
    # the numeric channel larger; it is not a negative-height danger flag.
    expected = (base[:,2,None]-hits[:,:,2]-.5).clamp(-1.,1.)
    assert expected[0,(hits[0,:,1]>.5) & torch.isfinite(hits[0,:,2])].min() > .7


def test_student_clip_preserves_small_drop_contrast_and_saturates_deep_drops():
    from task_support.student_v1.configuration import StudentV1PerceptionCfg
    from task_support.student_v1.perception import StudentObservationState
    cfg = StudentV1PerceptionCfg(level=0.,clean_probability=1.,stress_probability=0.,
        min_delay_s=0.,max_delay_s=0.,update_jitter_s=0.,terrain_visibility=False,lidar_horizontal_fov_deg=360.)
    state = StudentObservationState(4,'cpu',.02,cfg,history_steps=4)
    drop = torch.tensor([.04,.12,.55,1.4]);base = torch.zeros(4,3);base[:,2] = drop+.5
    hits = torch.cat((state.grid[None].expand(4,-1,-1),torch.zeros(4,187,1)),-1)
    rotation = torch.eye(3)[None].expand(4,-1,-1);proprio = torch.zeros(4,57);proprio[:,5] = -1
    for step in range(12):output = state.observe(proprio,hits,base,torch.zeros(4),rotation,step)
    valid = output[:,244:431]>.5
    assert valid.all()
    torch.testing.assert_close(output[:,57:244].mean(-1),torch.tensor([.04,.12,.55,1.]))
