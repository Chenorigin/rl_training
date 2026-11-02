"""Geometry counterexamples for support-surface sensing, not policy performance."""
from dataclasses import replace
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'source/rl_training'))
sys.path.insert(0, str(ROOT/'deploy/deploy_mujoco'))
from task_support.student_v1.configuration import StudentV1PerceptionCfg
from task_support.student_v1.perception import grid_points, terrain_visibility, StudentObservationState
from task_support.cts.map_recovery import RecoveryObservationState
from task_support.cts.configuration import default_settings


def scene():
    grid = grid_points('cpu')
    hits = torch.cat((grid, torch.zeros(len(grid), 1)), -1)[None]
    normals = torch.zeros_like(hits); normals[..., 2] = 1.
    return hits, torch.tensor([[0., 0., .4]]), torch.zeros(1), torch.eye(3)[None], normals


def config(**kwargs):
    return StudentV1PerceptionCfg(**({'support_surface_visibility':True} | kwargs))


@pytest.mark.parametrize('angle,distance', [(15,.6), (30,.4), (45,.4)])
def test_old_sensor_leaks_oblique_wall_top_new_sensor_rejects(angle, distance):
    h,b,y,r,n = scene()
    a = math.radians(angle)
    along = h[...,0]*math.cos(a)+h[...,1]*math.sin(a)
    wall = (along >= distance) & (along < distance+.1)
    h[...,2][wall] = 1.5
    old = terrain_visibility(h,b,y,r,config(support_surface_visibility=False))
    new = terrain_visibility(h,b,y,r,config(),surface_normals=n)
    assert (old & wall).any(), 'Negative control must reproduce the original gap'
    assert not (new & wall).any()
    assert torch.equal(new & ~wall, old & ~wall), 'Do not reveal previously occluded ground'


@pytest.mark.parametrize('slope', [.4,.8])
def test_visible_ramp_above_sensor_is_not_clipped_by_height(slope):
    h,b,y,r,n = scene()
    b[:,2]=.1
    h[...,2] = slope*h[...,0]
    n[...,0] = -slope
    cfg = config(terrain_visibility=False)
    seen = terrain_visibility(h,b,y,r,cfg,surface_normals=n)
    assert seen[h[...,2] > .25].all()
    assert (h[...,2] > .25).any()
    assert torch.equal(seen, terrain_visibility(h,b,y,r,replace(cfg,support_surface_visibility=False)))


@pytest.mark.parametrize('kind', ['flat','low_platform','stairs','descent'])
def test_visible_treads_preserve_height_and_visibility(kind):
    h,b,y,r,n = scene()
    if kind == 'low_platform': h[...,2] = (h[...,0]>.3)*.15
    if kind == 'stairs': h[...,2] = torch.floor((h[...,0]+.8)/.3)*.08
    if kind == 'descent': h[...,2] = -torch.floor((h[...,0]+.8)/.3)*.08
    old = terrain_visibility(h,b,y,r,replace(config(),support_surface_visibility=False))
    new = terrain_visibility(h,b,y,r,config(),surface_normals=n)
    assert torch.equal(old,new)


def test_missing_or_invalid_normals_do_not_silently_reenable_top_view():
    h,b,y,r,n = scene()
    with pytest.raises(ValueError,match='normals'):
        terrain_visibility(h,b,y,r,config())
    n[:] = torch.nan
    assert not terrain_visibility(h,b,y,r,config(),surface_normals=n).any()
    with pytest.raises(ValueError): config(support_surface_visibility=1).validate()


def test_world_yaw_translation_and_mesh_winding_do_not_change_visibility():
    h,b,y,r,n = scene();h[...,2][h[...,0]>.4] = 1.2
    expected = terrain_visibility(h,b,y,r,config(),surface_normals=n)
    angle=.71;c,s=math.cos(angle),math.sin(angle)
    rot=torch.tensor([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
    shift=torch.tensor([20.,-40.,3.])
    result=terrain_visibility(h@rot.T+shift,b@rot.T+shift,y+angle,rot[None],config(),surface_normals=-n@rot.T)
    assert torch.equal(expected,result)


def quiet_config(surface=True):
    return config(support_surface_visibility=surface, level=0.,clean_probability=1.,stress_probability=0.,
                  map_update_period_s=.02,update_jitter_s=0.,min_delay_s=0.,max_delay_s=0.,max_map_age_s=.1)


@pytest.mark.parametrize('recovery', [False,True])
def test_actual_packet_pipeline_unknown_is_zero_and_history_expires(recovery):
    cfg=quiet_config();h,b,y,r,n=scene()
    cls=RecoveryObservationState if recovery else StudentObservationState
    extra={'recovery':default_settings()['map_recovery']} if recovery else {}
    state=cls(1,'cpu',.02,cfg,4,**extra)
    state._initialize();state.next_capture.zero_()
    p=torch.zeros(1,57);p[:,5]=-1.
    state.observe(p,h,b,y,r,0,surface_normals=n)
    old=state.last_channels[1].clone()
    h[...,2]=1.5
    state.observe(p,h,b,y,r,1,surface_normals=n)
    assert torch.equal(state.last_channels[1],old), 'Invisible is not a contradictory wall-point fit'
    for step in range(2,8): state.observe(p,h,b,y,r,step,surface_normals=n)
    assert all(not channel.any() for channel in state.last_channels)
    state.reset();state.observe(p,h,b,y,r,8,surface_normals=n)
    assert all(not channel.any() for channel in state.last_channels)


def test_disabled_mode_preserves_rng_and_observations_with_or_without_normals():
    h,b,y,r,n=scene();p=torch.zeros(1,57);p[:,5]=-1
    outputs=[]
    for normals in [None,n]:
        torch.manual_seed(123)
        state=StudentObservationState(1,'cpu',.02,config(support_surface_visibility=False),4)
        obs=[state.observe(p,h,b,y,r,t,surface_normals=normals) for t in range(16)]
        outputs.append((torch.stack(obs),torch.get_rng_state()))
    assert all(torch.equal(a,b) for a,b in zip(*outputs))


def test_mujoco_real_ray_normal_preserves_tilted_surface():
    import mujoco
    from terrain_loader import terrain_ray
    m=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <geom type="plane" size="5 5 .1" euler="0 -30 0"/>
    </worldbody></mujoco>''')
    d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    normal=np.full(3,np.nan)
    distance,geom,flex=terrain_ray(m,d,np.array([.5,0.,2.]),np.array([0.,0.,-1.]),None,normal=normal)
    assert distance>0 and geom==0 and flex==-1
    np.testing.assert_allclose(normal,[-.5,0.,math.sqrt(3)/2],atol=1e-7)


def test_actual_maze_capture_and_student_adapter_reject_hidden_wall_tops():
    import mujoco
    import xml.etree.ElementTree as ET
    from types import SimpleNamespace
    from deploy_mujoco import M20Contract
    from student_v1_adapter import StudentMapAdapter, SensorExtras
    xml=ET.parse(ROOT/'deploy/deploy_mujoco/terrains/maze.xml').getroot()
    ET.SubElement(xml.find('worldbody'),'body',name='sensor_probe',pos='-.2 0 .4')
    m=mujoco.MjModel.from_xml_string(ET.tostring(xml,encoding='unicode'))
    d=mujoco.MjData(m)
    c=M20Contract.__new__(M20Contract)
    c.model=m;c.base_id=m.body('sensor_probe').id;c.terrain=SimpleNamespace(ray_top=2.)
    c.terrain_group=np.array([1,0,0,0,0,0],dtype=np.uint8)
    c.terrain_flex_ids=();c.terrain_geom_ids=set(range(m.ngeom))
    # Proprio is a fixed probe; the measured map goes through the real capture
    # method, MuJoCo intersections, adapter and packet queue. No policy rollout.
    c.observe=lambda data,command,previous: np.zeros(244)
    leaked=0
    for angle in [0,15,30,45,60]:
        a=math.radians(angle);m.body_quat[c.base_id]=[math.cos(a/2),0,0,math.sin(a/2)]
        mujoco.mj_forward(m,d)
        for enabled in [False,True]:
            adapter=StudentMapAdapter(c,quiet_config(enabled),SensorExtras(0,4))
            adapter.map._initialize();adapter.map.next_capture.zero_()
            c.last_scan_hits=c.capture_height_scan(d,adapter.capture_grid)
            observation=adapter.observe(d,np.zeros(3),np.zeros(16),0)
            assert observation.shape==(789,)
            walls=c.last_scan_hits[:,2]>.9
            count=int(adapter.last_channels[1][walls].sum())
            if enabled:
                assert count==0
                assert not adapter.last_channels[0][walls].any()
                assert not adapter.last_channels[2][walls].any()
            else:leaked+=count
    assert leaked>0,'Original consumer must expose the maze counterexample'


@pytest.mark.parametrize('wrapper', ['base','cts'])
def test_observation_entrypoints_forward_measured_normals(monkeypatch, wrapper):
    from types import SimpleNamespace
    from task_support.student_v1 import environment as base
    from task_support.student_v3_5 import v3_5_environment as cts
    module=base if wrapper=='base' else cts.base
    h,b,y,r,n=scene();seen={}
    def observe(*args,**kwargs): seen.update(kwargs);return torch.zeros(1,789)
    monkeypatch.setattr(module,'sensor_state',lambda env:SimpleNamespace(observe=observe))
    monkeypatch.setattr(module,'clean_teacher_proprio',lambda *args:torch.zeros(1,57))
    monkeypatch.setattr(cts,'feedback',lambda env:None)
    class Scene(dict): pass
    scene_object=Scene(robot=SimpleNamespace(data=SimpleNamespace(root_quat_w=torch.tensor([[1.,0.,0.,0.]]),root_pos_w=b)))
    scene_object.sensors={'height_scanner_capture':SimpleNamespace(data=SimpleNamespace(ray_hits_w=h,ray_normals_w=n))}
    env=SimpleNamespace(scene=scene_object,cfg=SimpleNamespace(student_perception=config()),
                        common_step_counter=0,command_manager=SimpleNamespace(get_command=lambda name:torch.zeros(1,3)))
    result=(base if wrapper=='base' else cts).student_observation(env)
    assert result.shape==(1,789)
    assert seen['surface_normals'] is n
