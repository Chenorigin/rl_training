"""Finite-course geometry, completion, assignment and frozen legacy counterexamples."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
import sys
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'source/rl_training'))
from task_support.student_v3_safe.configuration import default_settings
from task_support.student_v3_safe.terrain_geometry import geometries, cases, surfaces, course_mesh, regions, validate_course
from task_support.student_v3_safe.terrain_course import course_state, reset_course
from task_support.student_v3_safe.terrain_signals import StudentTerrainTracker
from task_support.student_v3_safe.environment import course_snapshot, restore_course
from task_support.student_v3_safe.gait_reward import eligible
from task_support.stair_quiet_v1_safe.settings import SafeSettings
from task_support.stair_quiet_v1_safe.signals import SafeTailTracker
from task_support.stair_quiet_v1_width.geometry import stair_mesh
from test_stair_quiet_v1_width_mechanics import sample

CFG = default_settings()['training']['terrain_course']
S = replace(SafeSettings(), tile_length=CFG['tile_length'])
torch.set_num_threads(1)


def data(g, x, z, n=1):
    d = sample(n=n, x=x)
    spans, goal, goal_height = surfaces(g, S)
    d.update(height=torch.full((n,), g.height), depth=torch.full((n,), g.depth),
             steps=torch.full((n,), g.steps), spans=torch.tensor([spans]*n),
             round_trip=torch.full((n,), g.family=='round_trip'),
             goal=torch.full((n,), goal), goal_height=torch.full((n,), goal_height))
    d['position'][..., 2] = z+S.wheel_radius
    d['root'][:, 0], d['root'][:, 2] = x, z+.4
    return d


class Scene(dict):
    pass


def fake_env(n=40):
    robot = NS(data=NS(default_root_state=torch.zeros(n, 13), root_pos_w=torch.zeros(n,3)),
               write_root_pose_to_sim=lambda *a, **kw: None,
               write_root_velocity_to_sim=lambda *a, **kw: None)
    scene = Scene(robot=robot)
    scene.terrain = NS(terrain_types=torch.zeros(n,dtype=torch.long),
        terrain_levels=torch.zeros(n,dtype=torch.long),env_origins=torch.zeros(n,3),
        terrain_origins=torch.zeros(10,40,3))
    scene.env_origins=torch.zeros(n,3)
    env = NS(num_envs=n, device='cpu', scene=scene,
        cfg=NS(stair_tail=S, student_terrain_course=deepcopy(CFG), seed=42,
               student_diagonal_reward=default_settings()['training']['diagonal_reward']))
    env.reset=lambda:reset_course(env,torch.arange(n),{}, {})
    return env


def test_default_geometry_and_invalid_configuration():
    assert len(geometries(CFG))==8 and len(cases(CFG))==40
    for family, key, value in [('round_trip','steps',2.5),('open_ascent','height',float('nan'))]:
        cfg=deepcopy(CFG);cfg[family][key]=value
        with pytest.raises(ValueError):validate_course(cfg)
    cfg=deepcopy(CFG);cfg['family_cycle']=['legacy']
    with pytest.raises(ValueError):validate_course(cfg)
    with pytest.raises(ValueError,match='finishing landing'):
        surfaces(geometries(CFG)[-1],replace(S,tile_length=7.))


def test_open_mesh_has_only_horizontal_faces_and_riser_rays_pass():
    g=geometries(CFG)[-1]
    opened=course_mesh(g,S)
    closed=course_mesh(replace(g,family='legacy',closed_riser=True),S)
    assert np.allclose(opened.face_normals[:,:2],0)
    assert np.allclose(opened.face_normals[:,2],1)
    assert np.any(np.abs(closed.face_normals[:,0])>.9)
    # Probe halfway up the first riser. A closed riser blocks this segment;
    # a horizontal mesh permits it. Test the exact collision source meshes.
    origin=np.array([[.8,0.,g.height/2]])
    direction=np.array([[1.,0.,0.]])
    assert not opened.ray.intersects_any(origin,direction).any()
    assert closed.ray.intersects_any(origin,direction).all()
    points=opened.ray.intersects_location(np.array([[1.15,0.,5.],[1.45,0.,5.]]),
                                         np.array([[0.,0.,-1.]]*2),multiple_hits=False)[0]
    assert np.allclose(sorted(points[:,2]),[g.height,2*g.height])


def test_legacy_collision_mesh_is_exactly_the_original_generator():
    for g in geometries(CFG)[:6]:
        expected, actual = stair_mesh(g,S), course_mesh(g,S)
        assert np.array_equal(expected.vertices,actual.vertices)
        assert np.array_equal(expected.faces,actual.faces)


def test_round_trip_surface_queries_match_mesh_on_ascent_plateau_descent_and_exit():
    g=geometries(CFG)[-2];mesh=course_mesh(g,S)
    spans,goal,goal_z=surfaces(g,S)
    assert goal_z==0 and goal==pytest.approx(3.21)
    for lo,hi,z in spans:
        x=(lo+hi)/2;d=data(g,x,z)
        forward,side,top=regions(d['position'],d['axis'],d,S)
        assert torch.allclose(top,torch.full((1,4),z))
        assert (forward>0).all() and (side>0).all()
        hit=mesh.ray.intersects_location([[x,0.,5.]],[[0.,0.,-1.]],multiple_hits=False)[0]
        assert hit[0,2]==pytest.approx(z)
    # The 45cm top tread and platform merge: their seam is not a false edge.
    d=data(g,1.75,.45)
    assert (regions(d['position'],d['axis'],d,S)[0]>0).all()
    # An actual descending lip must still be unsafe, not smoothed away.
    d=data(g,2.32,.30)
    assert (regions(d['position'],d['axis'],d,S)[0]<0).all()


def test_round_trip_requires_crest_then_final_lower_landing_and_clears_on_partial_reset():
    g=geometries(CFG)[-2]
    t=StudentTerrainTracker(2,'cpu',S)
    t.reset(slice(None),torch.zeros(2))
    for _ in range(220):t.update(**data(g,2.,.45,2))
    assert t.crest_seen.all() and not t.last['crossed'].any() and not t.success.any()
    t.reset(torch.tensor([1]),torch.zeros(1))
    for _ in range(220):t.update(**data(g,3.6,0.,2))
    assert t.success.tolist()==[True,False]
    assert t.last['crossed'].tolist()==[True,False]
    assert not t.crest_seen[1].any()


def test_frozen_legacy_tracker_parity_all_persistent_fields_and_costs():
    g=geometries(CFG)[0]
    old=SafeTailTracker(1,'cpu',S);new=StudentTerrainTracker(1,'cpu',S)
    old.reset(slice(None),torch.zeros(1));new.reset(slice(None),torch.zeros(1))
    for tick in range(80):
        x=1.01 if tick%3==0 else 1+g.depth/2
        d=data(g,x,g.height)
        d['forces'][...,2]=90+tick*8
        d['velocity'][...,0]=.5 if tick%2 else 0.
        a={k:v for k,v in d.items() if k not in ('spans','round_trip','goal','goal_height')}
        old.begin_step();new.begin_step();old.update(**a);new.update(**d)
        for name,value in vars(old).items():
            if isinstance(value,torch.Tensor):
                torch.testing.assert_close(getattr(new,name),value,rtol=0,atol=0)
            elif isinstance(value,dict):
                for key,item in value.items():
                    torch.testing.assert_close(getattr(new,name)[key],item,rtol=0,atol=0)


def test_assignment_mixture_all_cases_partial_reset_and_saved_geometry_guard():
    env=fake_env();state=course_state(env);env.tail_stage=2
    seen=set();counts=torch.zeros(3,dtype=torch.long)
    for _ in range(40):
        env.reset();seen.update(env.scene.terrain.terrain_types.tolist())
        counts+=torch.bincount(state['family'],minlength=3)
    assert seen==set(range(40))
    assert counts.tolist()==[800,400,400]
    oldtype=env.scene.terrain.terrain_types.clone();oldcount=state['assign_count'].clone()
    reset_course(env,torch.tensor([1,3]),{}, {})
    unchanged=torch.ones(40,dtype=torch.bool);unchanged[[1,3]]=False
    assert torch.equal(oldtype[unchanged],env.scene.terrain.terrain_types[unchanged])
    assert torch.equal(oldcount[unchanged],state['assign_count'][unchanged])
    assert state['width_statistics']['episodes'].shape==(10,40)
    saved=course_snapshot(env);restore_course(env,saved)
    assert torch.equal(state['assign_count'],saved['state']['assign_count']+1)
    saved['state']['goal_table'][30]+=.1
    before=state['assign_count'].clone()
    with pytest.raises(ValueError,match='table changed'):restore_course(env,saved)
    assert torch.equal(before,state['assign_count'])


def test_descent_keeps_policy_gradient_but_not_ascent_diagonal_bonus():
    env=fake_env(4);env.scene.terrain.terrain_types[:]=30
    env.scene['robot'].data.root_pos_w[:,0]=torch.tensor([1.2,2.,2.5,4.])
    env.command_manager=NS(get_command=lambda _:torch.tensor([[.5,0.,0.]]*4))
    assert eligible(env).tolist()==[True,False,False,False]
    assert eligible(env,ascent_only=False).tolist()==[True,True,True,False]
