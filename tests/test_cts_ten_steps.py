"""Physical tenth tread and completion boundaries must agree with CTS metadata."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from test_cts_sample_stairs import fake_env,S
from test_cts_platform import vertical_hit
from task_support.cts.configuration import default_settings,validate_config
from task_support.cts.terrain import geometries
from task_support.cts.sample_stairs import training_stairs
from task_support.student_v3_safe.terrain_geometry import geometries as frozen_geometries
from task_support.student_v3_safe.terrain_course import course_state
from task_support.student_v3_safe.terrain_signals import StudentTerrainTracker


def test_all_six_physical_tenth_treads_and_case_tables_agree_without_changing_frozen_source():
    env = fake_env()
    cfg = default_settings()
    course = env.cfg.student_terrain_course
    actual = geometries(course,cfg['legacy_stairs'])
    assert [g.steps for g in actual] == [10]*6+[3,20]
    assert [g.steps for g in frozen_geometries(course)] == [6]*6+[3,20]
    for i,g in enumerate(actual[:6]):
        terrain = SimpleNamespace(course=course,geometry_index=i,settings=S,
            cts_legacy_stairs=cfg['legacy_stairs'],cts_sample_stairs=cfg['sample_stairs'])
        meshes,origin = training_stairs(.95,terrain)
        # Directly exercise the SAME callback Isaac uses; independently ray-test
        # both the sixth and tenth tread, rejecting a six-step upper landing.
        for step in (6,10):
            x = S.start_length/2+(step-.5)*g.depth
            z = vertical_hit(meshes[0],[x+origin[0],origin[1]])-origin[2]
            assert z == pytest.approx(step*g.height,abs=1e-7)
        assert meshes[0].bounds[1,2] == pytest.approx(g.steps*g.height)
        for _ in range(2):
            env.scene.terrain.terrain_types[0] = i*5
            state = course_state(env)
            assert state['steps'][0] == 10
            assert state['goal'][0] == pytest.approx(1+10*g.depth)
            assert state['goal_height'][0] == pytest.approx(10*g.height)
            assert state['ascent_end'][0] == pytest.approx(1+10*g.depth)


def test_sixth_step_is_not_a_completion_but_tenth_step_finish_is():
    env = fake_env();env.scene.terrain.terrain_types[0] = 0
    state = course_state(env)
    g = geometries(env.cfg.student_terrain_course,default_settings()['legacy_stairs'])[0]
    def tracker_at(x,z):
        tracker = StudentTerrainTracker(1,'cpu',S)
        tracker.reset(slice(None),torch.zeros(1))
        position = torch.tensor([[[x,.2,z+.09],[x,-.2,z+.09],[x,.2,z+.09],[x,-.2,z+.09]]])
        axis = torch.zeros_like(position);axis[...,1] = 1
        force = torch.zeros_like(position);force[...,2] = 100
        for _ in range(100):
            tracker.update(position=position,axis=axis,velocity=torch.zeros_like(position),omega=torch.zeros_like(position),
                forces=force,root=torch.tensor([[x,0.,z+.5]]),root_velocity=torch.zeros(1,3),yaw=torch.zeros(1),
                upright=torch.ones(1),dt=.005,**{k:state[k][:1] for k in (
                    'height','depth','width','steps','spans','round_trip','goal','goal_height')})
        return tracker
    # Just beyond the OLD six-step finish, now on the seventh tread.
    sixth = tracker_at(1+6.5*g.depth,7*g.height)
    assert not sixth.success.item() and not sixth.last['crossed'].item()
    tenth = tracker_at(1+10*g.depth+.5,10*g.height)
    assert tenth.success.item() and tenth.last['crossed'].item()


def test_legacy_checkpoint_schema_accepts_missing_override_but_new_config_requires_it():
    cfg = default_settings();cfg.pop('legacy_stairs')
    validate_config(cfg,legacy=True)
    with pytest.raises(ValueError):validate_config(cfg)
    for steps in (True,0,-1,10.5):
        bad = deepcopy(default_settings());bad['legacy_stairs']['steps'] = steps
        with pytest.raises(ValueError):validate_config(bad)
