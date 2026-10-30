"""Exact sample geometry and task-reward counterexamples; no optimization."""
from pathlib import Path
import sys
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source/rl_training'))
from task_support.cts.sample_stairs import scene, sample_mesh, support_spans, SampleStairsCourse, SampleTerrainTracker
from task_support.cts.practice import FlatCourseState
from task_support.cts.configuration import default_settings
from task_support.student_v3_safe.configuration import default_settings as student_settings
from task_support.student_v3_safe.terrain_course import course_state
from task_support.student_v3_safe.terrain_signals import StudentTerrainTracker
from task_support.stair_quiet_v1_safe.settings import SafeSettings
from test_cts_platform import vertical_hit

XML = default_settings()['sample_stairs']['terrain_xml']
S = SafeSettings(tile_length=12.)


def fake_env():
    class Scene(SimpleNamespace):
        def __getitem__(self,key):
            return getattr(self,key)
    env = SimpleNamespace(cfg=SimpleNamespace(student_terrain_course=student_settings()['training']['terrain_course'],
        stair_tail=S,cts_legacy_stairs=default_settings()['legacy_stairs']),device='cpu',num_envs=4,cts_practice_mask=torch.tensor([False,False,True,False]),
        cts_motion=SimpleNamespace(teacher=torch.tensor([True,False,True,True])),
        scene=Scene(terrain=SimpleNamespace(terrain_types=torch.tensor([30,30,30,35]),
            terrain_levels=torch.tensor([0,9,7,0]))))
    env.cts_sample = SampleStairsCourse(env,default_settings()['sample_stairs'])
    env._tail_course = None
    state = course_state(env)
    env._tail_course = FlatCourseState(state,env.cts_sample)
    return env


def test_training_initial_collision_mesh_matches_native_xml_rays_including_seams():
    mesh = sample_mesh(S,XML)
    model = mujoco.MjModel.from_xml_path(str(ROOT/XML))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model,data)
    # Points discriminate the formerly missing .30 m top tread and both 5 mm
    # gaps. Lateral points discriminate a .9 m deck from the 1 m stairs.
    points = [(x,y) for x in (.125,.375,.625,.7525,1.03,1.3075,1.45,1.75,2.05,2.25)
              for y in (0.,.44,.47,.52,-.7)]
    group = np.array([1,0,0,0,0,0],dtype=np.uint8)
    for x,y in points:
        geom = np.zeros(1,dtype=np.int32)
        distance = mujoco.mj_ray(model,data,np.array([x,y,2.]),np.array([0.,0.,-1.]),group,1,-1,geom)
        expected = 2.-distance
        np.testing.assert_allclose(vertical_hit(mesh,[x+S.start_length/2,y]),expected,atol=1e-7)


def test_cached_frozen_geometry_reads_receive_sample_profile_and_conservative_width():
    env = fake_env()
    state = course_state(env)
    assert env.cts_sample.mask.tolist() == [True,True,False,False]
    assert torch.allclose(state['width'][:2],torch.tensor([.9,.9]))
    assert state['width'][3] == 4.
    spans,goal = support_spans(S,XML)
    np.testing.assert_allclose(goal,3.21)
    np.testing.assert_allclose(state['spans'][0,:len(spans)],spans,atol=2e-7)
    np.testing.assert_allclose(state['goal'][:2],[3.21,3.21],atol=2e-7)
    assert any(abs(lo-1.75)<1e-8 and abs(hi-1.755)<1e-8 and z==0 for lo,hi,z in spans)
    assert any(abs(lo-2.305)<1e-8 and abs(hi-2.31)<1e-8 and z==0 for lo,hi,z in spans)
    assert any(abs(lo-2.31)<1e-8 and abs(hi-2.61)<1e-8 and abs(z-.45)<1e-8 for lo,hi,z in spans)
    # Repeated source calls rebuild derived arrays; the overlay must stay live.
    for _ in range(3):
        assert abs(float(course_state(env)['width'][0])-.9) < 1e-6
        assert abs(float(course_state(env)['goal'][0])-3.21) < 1e-6


def sample(x,z,y=.2):
    position = torch.tensor([[[x,y,z+.09],[x,-y,z+.09],[x,y,z+.09],[x,-y,z+.09]]])
    axis = torch.zeros_like(position);axis[...,1] = 1
    forces = torch.zeros_like(position);forces[...,2] = 100.
    spans,goal = support_spans(S,XML)
    return dict(position=position,axis=axis,velocity=torch.zeros_like(position),omega=torch.zeros_like(position),
        forces=forces,root=torch.tensor([[x,0.,z+.5]]),root_velocity=torch.zeros(1,3),
        yaw=torch.zeros(1),upright=torch.ones(1),height=torch.tensor([.15]),depth=torch.tensor([.25]),
        width=torch.tensor([.9]),steps=torch.tensor([3]),dt=.005,spans=torch.tensor([spans]),
        round_trip=torch.ones(1,dtype=torch.bool),goal=torch.tensor([goal]),goal_height=torch.zeros(1))


def traverse(path,*,sample_course=False):
    course = SimpleNamespace(mask=torch.ones(1,dtype=torch.bool),cfg=default_settings()['sample_stairs'])
    tracker = SampleTerrainTracker(1,'cpu',S,course) if sample_course else StudentTerrainTracker(1,'cpu',S)
    tracker.reset(slice(None),torch.zeros(1))
    for x,z,y in path:
        for _ in range(60):
            tracker.update(**sample(x,z,y))
    return tracker


def test_inherited_task_reward_accepts_stepwise_descent_but_rejects_bypass_and_side_escape():
    # Four wheel support on the crest followed by ordinary .15 m down steps
    # reaches the inherited success event. No high-platform departure cost.
    normal = traverse([(2.1,.45,.2),(2.75,.30,.2),(3.05,.15,.2),(3.8,0.,.2)])
    assert normal.success.item() and not normal.failed().item()
    assert normal.cost['progress'].item() > 0
    assert normal.cost['edge'].item() == 0
    bypass = traverse([(3.8,0.,.2)])
    assert not bypass.success.item()
    sideways = traverse([(2.1,.45,.43),(3.8,0.,.2)])
    assert sideways.edge_failed.item() and not sideways.success.item()


def test_expected_native_initial_dimensions_are_extracted_from_xml():
    boxes,treads,width = scene(XML)
    assert len(boxes) > 40 and len(treads) == 7
    np.testing.assert_allclose(width,.9)
    np.testing.assert_allclose(treads[-3:],[(1.31,1.61,.45),(1.61,1.91,.30),(1.91,2.21,.15)])


def test_ordered_loaded_route_rewards_once_and_cannot_skip_deck_or_lower_steps():
    ascent = [(1.125,.15,.2),(1.375,.30,.2)]
    normal = traverse([*ascent,(2.1,.45,.2),(2.75,.30,.2),(3.05,.15,.2),(3.8,0.,.2)],sample_course=True)
    assert normal.success.item() and normal.sample_phase.tolist() == [[6,6,6,6]]
    assert normal.cost['sample_route'].item() == 6.
    before = normal.cost['sample_route'].clone()
    for x,z in ((2.1,.45),(2.75,.3),(3.8,0.)):
        for _ in range(60):
            normal.update(**sample(x,z))
    assert torch.equal(normal.cost['sample_route'],before)
    no_deck = traverse([*ascent,(2.46,.45,.2),(2.75,.30,.2),(3.05,.15,.2),(3.8,0.,.2)],sample_course=True)
    assert not no_deck.success.item() and no_deck.sample_phase.max().item() == 2
    skip_down = traverse([*ascent,(2.1,.45,.2),(3.8,0.,.2)],sample_course=True)
    assert not skip_down.success.item() and skip_down.sample_phase.max().item() == 3
    skip_up = traverse([(2.1,.45,.2),(2.75,.30,.2),(3.05,.15,.2),(3.8,0.,.2)],sample_course=True)
    assert not skip_up.success.item() and skip_up.sample_phase.max().item() == 0
    assert not skip_up.last['crossed'].item()
    assert not skip_down.last['crossed'].item()


def test_airborne_or_riser_loads_do_not_advance_route_and_reset_clears_history():
    course = SimpleNamespace(mask=torch.ones(1,dtype=torch.bool),cfg=default_settings()['sample_stairs'])
    tracker = SampleTerrainTracker(1,'cpu',S,course)
    d = sample(2.1,.45)
    d['forces'][...,0] = 200
    for _ in range(100):
        tracker.update(**d)
    assert tracker.sample_phase.max().item() == 0
    d['forces'].zero_()
    for _ in range(100):
        tracker.update(**d)
    assert tracker.sample_phase.max().item() == 0
    tracker.sample_phase.fill_(3)
    tracker.reset(torch.tensor([0]),torch.tensor([0.]))
    assert tracker.sample_phase.max().item() == 0


def test_sample_reset_uses_real_narrow_row_and_all_five_contacts_without_wide_stage_credit():
    env = fake_env()
    terrain,state = env.scene.terrain,env._tail_course
    origins = torch.arange(10*40*3,dtype=torch.float).reshape(10,40,3)
    terrain.terrain_origins = origins
    terrain.env_origins = torch.zeros(4,3)
    env.scene.env_origins = torch.zeros(4,3)
    poses = []
    env.scene.robot = SimpleNamespace(data=SimpleNamespace(default_root_state=torch.zeros(4,13)),
        write_root_pose_to_sim=lambda pose,env_ids:poses.append((pose,env_ids)),
        write_root_velocity_to_sim=lambda velocity,env_ids:None)
    profiles = []
    for i in range(5):
        state['assign_count'][:2] = i
        env.cts_sample.reset(torch.arange(4))
        profiles.append(int(terrain.terrain_types[0]%5))
        assert terrain.terrain_levels[:2].tolist() == [9,9]
        assert state['width_curriculum'].level[:2].tolist() == [9,9]
        assert torch.equal(terrain.env_origins[:2],origins[9,terrain.terrain_types[:2]])
        assert poses[-1][1].tolist() == [0,1]  # Practice and other course excluded.
    assert sorted(profiles) == list(range(5))
    assert (terrain.terrain_levels[:2] > S.material_practice_level).all()
    env.cfg.cts_sample_automatic_stages = True
    env.tail_stage = 0
    active = (state['stage_table'] == 0) & (state['family_table'] != 1)
    state['stage_episodes'][active] = S.stage_min_episodes
    state['stage_successes'][active] = S.stage_min_episodes
    assert state['stage_episodes'][state['family_table'] == 1].sum() == 0
    env.cts_sample.advance_materials()
    assert env.tail_stage == 1
