"""Measured extended capture support must preserve the deployed actor contract."""
from copy import deepcopy
from dataclasses import asdict
import json,math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from task_support.student_v1.configuration import StudentV1PerceptionCfg
from task_support.student_v1.perception import StudentObservationState,grid_points,capture_grid_shape
from task_support.student_v1.capture_isaac import configure_capture
from task_support.student_v1.capture_budget import capture_memory_budget
from task_support.cts.configuration import default_settings,validate_config,VERSION,LEGACY_VERSIONS
from task_support.cts.perception import validate_capture


def cfg(padding=0.,**kwargs):
    values=json.loads(Path('configs/distillation/student_v3_safe.json').read_text())['perception']
    values.update(map_capture_padding_m=padding,level=0.,zero_delay_probability=0.,
                  min_delay_s=.4,max_delay_s=.4,map_update_period_s=1.,update_jitter_s=0.,max_map_age_s=.8)
    values.update(kwargs)
    return StudentV1PerceptionCfg(**values)


def inputs(state,x=0.,y=0.,yaw=0.,n=1):
    co,si=math.cos(yaw),math.sin(yaw)
    grid=state.capture_grid
    xy=torch.stack((x+co*grid[:,0]-si*grid[:,1],y+si*grid[:,0]+co*grid[:,1]),-1)
    hits=torch.cat((xy,.1*xy[:,:1]+.2*xy[:,1:2]),-1)[None].repeat(n,1,1)
    p=torch.zeros(n,57);p[:,5]=-1
    rotation=torch.tensor([[co,-si,0.],[si,co,0.],[0.,0.,1.]])[None].repeat(n,1,1)
    return p,hits,torch.tensor([[x,y,.5]]).repeat(n,1),torch.full((n,),yaw),rotation


def single_packet(padding,**kwargs):
    torch.manual_seed(42)
    state=StudentObservationState(1,'cpu',.02,cfg(padding,**kwargs),4)
    state._initialize();state.next_capture.zero_()
    state.observe(*inputs(state),0)
    state.next_capture.fill_(10000)
    return state


def test_geometry_is_aligned_and_output_grid_is_unchanged():
    state=single_packet(.6)
    assert state.capture_shape == (23,29) and len(state.capture_grid) == 667
    torch.testing.assert_close(state.capture_grid[state.output_indices],grid_points('cpu'))
    assert state.last_visibility.shape == (1,187)
    assert state.capture_visibility.shape == (1,667)
    torch.testing.assert_close(state.capture_grid.amin(0),torch.tensor([-1.4,-1.1]))
    torch.testing.assert_close(state.capture_grid.amax(0),torch.tensor([1.4,1.1]))


def test_delayed_capture_has_real_coverage_after_translation_and_no_future_truth():
    outputs=[]
    for padding in (0.,.6):
        state=single_packet(padding)
        current=inputs(state,x=.4)
        # No new acquisition; replace live hits with a deliberately false plane.
        current[1][...,2]=99.
        before=state.observe(*current,19)
        assert not before[:,244:431].any()  # packet has not arrived
        after=state.observe(*current,20)
        assert after.shape == (1,789)
        target=(state.grid[:,0] > .79) & (state.grid[:,1].abs() < 1e-6)
        if padding:
            assert after[0,244:431][target].item() == 1.
            torch.testing.assert_close(after[0,57:244][target],torch.tensor([-.12]),atol=1e-6,rtol=0)
        else:
            assert after[0,244:431][target].item() == 0.
        outputs.append(after)
        expired=state.observe(*current,41)
        assert not expired[:,244:618].any()
    assert outputs[1][:,244:431].sum() > outputs[0][:,244:431].sum()


def test_rotated_query_uses_source_y_extent_and_obeys_capture_time_fov():
    for padding in (0.,.6):
        # Wide FOV for the positive rotation example, then real FOV negative.
        state=single_packet(padding,lidar_horizontal_fov_deg=360.,terrain_visibility=False)
        out=state.observe(*inputs(state,yaw=math.pi/2),20)
        target=(state.grid[:,0] > .79) & (state.grid[:,1].abs() < 1e-6)
        assert out[0,244:431][target].item() == float(bool(padding))
        if padding:torch.testing.assert_close(out[0,57:244][target],torch.tensor([-.16]),atol=1e-6,rtol=0)
    unseen=single_packet(.6)
    out=unseen.observe(*inputs(unseen,yaw=math.pi/2),20)
    target=(unseen.grid[:,0] > .79) & (unseen.grid[:,1].abs() < 1e-6)
    assert out[0,244:431][target].item() == 0.
    outside=single_packet(.6)
    out=outside.observe(*inputs(outside,x=3.),20)
    assert not out[:,244:618].any()


def test_extended_reset_only_clears_requested_environment():
    torch.manual_seed(0)
    state=StudentObservationState(2,'cpu',.02,cfg(.6),4);state._initialize();state.next_capture.zero_()
    state.observe(*inputs(state,n=2),0);state.next_capture.fill_(10000)
    before=state.observe(*inputs(state,n=2),20)
    assert before[0,244:431].sum() > 0
    state.reset(torch.tensor([1]));after=state.observe(*inputs(state,n=2),21)
    assert after[0,244:431].sum() > 0 and not after[1,244:431].any()


def test_missing_extended_rays_stay_unknown_and_short_input_fails_closed():
    state=StudentObservationState(1,'cpu',.02,cfg(.6,min_delay_s=0.,max_delay_s=0.),4)
    state._initialize();state.next_capture.zero_();data=inputs(state);data[1][...,2]=torch.nan
    out=state.observe(*data,0)
    assert torch.isfinite(out).all() and not out[:,244:618].any()
    state.reset();data=list(inputs(state));data[1]=data[1][:,:187]
    with pytest.raises(ValueError,match='capture hits'):state.observe(*data,1)


def test_training_capture_sensor_and_budget_preserve_teacher():
    original=SimpleNamespace(pattern_cfg=SimpleNamespace(size=[1.6,1.],resolution=.1,ordering='xy'),debug_vis=False)
    config=SimpleNamespace(student_perception=cfg(.6),scene=SimpleNamespace(height_scanner=original))
    configure_capture(config)
    assert config.scene.height_scanner is original
    assert original.pattern_cfg.size == [1.6,1.]
    assert config.scene.height_scanner_capture.pattern_cfg.size == pytest.approx([2.8,2.2])
    budget=capture_memory_budget(asdict(cfg(.6)),2048)
    assert budget['capture_points'] == 667 and budget['extra_capture_reserve_bytes'] > 0
    config.student_perception=cfg();configure_capture(config)
    assert config.scene.height_scanner_capture is None
    assert capture_memory_budget(asdict(cfg()),2048)['extra_capture_reserve_bytes'] == 0


def test_current_cts_ledger_and_invalid_capture_options():
    settings=default_settings();validate_config(settings)
    assert VERSION.endswith('_v26') and 'm20_cts_action_latent_v10' in LEGACY_VERSIONS
    for value in (-1.,float('nan'),float('inf'),True,3.,'0.6'):
        with pytest.raises(ValueError):cfg(value).validate()
        with pytest.raises(ValueError):validate_capture({'padding_m':value})
    old=deepcopy(settings);old.pop('map_capture');validate_config(old,legacy=True)
    with pytest.raises(ValueError):validate_config(old)
    assert capture_grid_shape(.61) == (25,31)


def test_mujoco_capture_returns_unknown_for_missing_collision_terrain(monkeypatch):
    import mujoco
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'deploy/deploy_mujoco'))
    from deploy_mujoco import M20Contract
    model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom name="floor" type="box" size=".5 .5 .05" pos="0 0 -.05"/><body name="base" pos="0 0 .5"/></worldbody></mujoco>')
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    c=M20Contract.__new__(M20Contract)
    c.model=model;c.base_id=model.body('base').id;c.terrain=SimpleNamespace(ray_top=20.)
    c.terrain_group=np.array([1,0,0,0,0,0],dtype=np.uint8);c.terrain_flex_ids=();c.terrain_geom_ids={0}
    hits=c.capture_height_scan(data,np.array([[0.,0.],[10.,0.]]))
    assert hits[0,2] == pytest.approx(0.) and np.isnan(hits[1,2])
