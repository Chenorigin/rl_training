"""Actual 2D support queries preserve the frozen Safe teacher's feature units."""
from types import SimpleNamespace
from unittest.mock import patch

import torch

from task_support.cts.all_terrain_terms import _margins, teacher_safety
from task_support.stair_quiet_v1_safe.observations import safety_features
from task_support.stair_quiet_v1_safe.settings import SafeSettings


def fixture():
    cfg = SafeSettings()
    depth,height,width,steps = .15,.15,1.,4
    rects = [[1+i*depth,1+(i+1)*depth,-width/2,width/2,(i+1)*height] for i in range(steps)]
    rects += [[1+steps*depth,7.,-width/2,width/2,steps*height]]
    pos = torch.tensor([[[1.075,.2,.24],[1.225,-.2,.39],[1.375,.2,.54],[1.57,-.2,.69]]])
    axis = torch.zeros_like(pos); axis[...,1] = 1.
    velocity = torch.zeros_like(pos); velocity[...,0] = .3
    omega = torch.zeros_like(pos); omega[...,1] = .3/cfg.wheel_radius
    forces = torch.zeros_like(pos); forces[...,2] = 150.
    root = torch.tensor([[1.3,0.,.8]])
    data = SimpleNamespace(body_lin_vel_w=velocity,body_ang_vel_w=omega,
        body_link_pos_w=pos,body_com_pos_w=pos.clone(),root_pos_w=root,
        root_quat_w=torch.tensor([[1.,0.,0.,0.]]),root_lin_vel_w=torch.tensor([[.3,0.,0.]]))
    geometry = dict(position=pos,axis=axis,supported=torch.ones(1,4,dtype=torch.bool),
                    velocity=torch.zeros_like(pos),slip=torch.zeros(1,4),force=forces,
                    force_n=torch.zeros(1,4))
    state = SimpleNamespace(origins=torch.zeros(1,3),tile=torch.zeros(1,dtype=torch.long),
        surface=SimpleNamespace(rectangles=torch.tensor([rects]),local_bounds=torch.tensor([[-1.,7.,-.5,.5]])),
        stairs=torch.ones(1,dtype=torch.bool),widths=torch.tensor([width]),course_ids=torch.zeros(1,dtype=torch.long),
        body_ids=[0,1,2,3],observe=lambda:geometry)
    env = SimpleNamespace(scene={'robot':SimpleNamespace(data=data)},cfg=SimpleNamespace(stair_tail=cfg))
    source = dict(position=pos,axis=axis,velocity=velocity,omega=omega,forces=forces,
        root=root,root_velocity=data.root_lin_vel_w,yaw=torch.zeros(1),
        height=torch.tensor([height]),depth=torch.tensor([depth]),width=torch.tensor([width]),steps=torch.tensor([steps]))
    return env,state,geometry,source


def test_all_48_features_match_original_teacher_on_actual_narrow_stairs():
    env,state,_,source = fixture()
    with patch('task_support.cts.all_terrain_terms.course',return_value=state):
        actual = teacher_safety(env)
    expected = safety_features(source,env.cfg.stair_tail)
    torch.testing.assert_close(actual,expected,rtol=1.e-5,atol=2.e-6)
    assert actual.shape == (1,48)
    # A 15 cm tread has a positive central contact-patch margin, despite the
    # wheel's 18 cm diameter. Using wheel radius as patch radius fails this.
    assert actual[0,0] > 0


def test_rolling_link_speed_and_force_columns_retain_source_meaning():
    env,state,_,_ = fixture()
    with patch('task_support.cts.all_terrain_terms.course',return_value=state):
        feet = teacher_safety(env)[0,:44].reshape(4,11)
    torch.testing.assert_close(feet[:,7],torch.zeros(4))  # no contact slip
    torch.testing.assert_close(feet[:,8],torch.full((4,),.3))  # world vertical force / 500
    torch.testing.assert_close(feet[:,9],torch.full((4,),.3))  # translating link speed, not contact speed
    torch.testing.assert_close(feet[:,10],torch.zeros(4))


def test_height_aware_lip_support_matches_source_for_wheel_behind_next_tread():
    env,state,g,source = fixture()
    g['position'][0,0,0] = .98
    with patch('task_support.cts.all_terrain_terms.course',return_value=state):
        actual = teacher_safety(env)
    torch.testing.assert_close(actual,safety_features(source,env.cfg.stair_tail),rtol=1.e-5,atol=2.e-6)
    assert actual[0,0] < -.6


def test_adjacent_equal_height_landing_has_no_seam_but_separated_stones_keep_edges():
    _,state,_,_ = fixture()
    extent = torch.tensor([[[.05,.15]]])
    # Last tread and landing join at 1.6; support begins at 1.45 and ends at 7.
    forward,_,_ = _margins(state,torch.tensor([[[1.59,0.]]]),extent)
    torch.testing.assert_close(forward,torch.tensor([[.09]]),atol=1.e-6,rtol=0)
    state.stairs.zero_()
    state.surface.rectangles = torch.tensor([[[0.,.2,-.5,.5,.15],[1.,1.2,-.5,.5,.15]]])
    forward,left,right = _margins(state,torch.tensor([[[.18,0.]]]),extent)
    torch.testing.assert_close(forward,torch.tensor([[-.03]]),atol=1.e-6,rtol=0)
    torch.testing.assert_close(left,torch.tensor([[.35]]))
    torch.testing.assert_close(right,torch.tensor([[.35]]))


def test_rough_ground_uses_finite_tile_boundaries_without_fabricated_stair_edges():
    _,state,_,_ = fixture()
    state.stairs.zero_()
    state.surface.rectangles = torch.tensor([[[0.,0.,0.,0.,-torch.inf]]])
    extent = torch.tensor([[[.05,.15]]])
    forward,left,right = _margins(state,torch.tensor([[[2.,0.]]]),extent)
    torch.testing.assert_close(forward,torch.tensor([[2.95]]))
    torch.testing.assert_close(left,torch.tensor([[.35]]))
    torch.testing.assert_close(right,torch.tensor([[.35]]))
