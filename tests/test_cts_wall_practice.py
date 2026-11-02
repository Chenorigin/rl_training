"""Wall geometry, failure counterexamples, route semantics and legacy isolation."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from task_support.cts.configuration import default_settings, validate_config
from task_support.cts.all_terrain_geometry import build_atlas, TorchSurface
from task_support.cts.wall_practice import (KINDS, MODE_START, WallPractice, validate,
    collision_document, primitive_aabbs, envelope_clearance)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def config():
    c = default_settings()
    c['all_terrain']['geometry'].update(samples_per_level=3,geometry_levels=1,grid_resolution=.25)
    c['all_terrain']['curriculum']['geometry_levels'] = 1
    return c


@pytest.fixture(scope='module')
def atlas(config):
    c = config['all_terrain']
    return build_atlas(c['geometry'],c['wall_practice'])


def test_new_config_preserves_all_existing_v15_features(config):
    validate_config(config)
    for key in ('hind_clearance','compound_commands','wall_practice'):
        assert config['all_terrain'][key]['enabled']
    assert config['map_recovery']['enabled']
    old = deepcopy(config)
    old['all_terrain'].pop('wall_hold_practice', None)
    old['all_terrain'].pop('wall_practice')
    old['all_terrain'].pop('obstacle_response',None)
    old['all_terrain'].pop('stair_walls',None)
    validate_config(old)


@pytest.mark.parametrize('key,value', [('turn_width',.9),('width_range',[.5,1.8]),
    ('episode_fraction',1.),('collision_geometry_sha256','wrong'),('minimum_turn_rate',float('nan'))])
def test_reject_infeasible_or_unpinned_wall_contract(config,key,value):
    cfg = deepcopy(config['all_terrain']['wall_practice']);cfg[key] = value
    with pytest.raises(ValueError):validate(cfg)


def test_real_walls_floor_and_every_curriculum_variant(config,atlas):
    wall = [r for r in atlas.records if r.get('wall_kind')]
    assert {r['wall_kind'] for r in wall} == set(KINDS)
    assert len(wall) == 5*config['all_terrain']['wall_practice']['samples_per_kind_level']
    surface = TorchSurface(atlas,'cpu',dtype=torch.float64)
    for r in wall:
        i = r['tile_id'];origin=atlas.origins[i,:2]
        route = np.array(r['wall_route'])
        for a,b in zip(route[:-1],route[1:]):
            xy = a+(b-a)*np.linspace(0,1,17)[:,None]
            z,_ = surface.sample(torch.tensor([i]),torch.tensor((xy+origin)[None]))
            np.testing.assert_allclose(z.numpy(),0.,atol=1.e-8)
        rect = r['wall_rectangles'][0]
        xy = torch.tensor([[(rect[0]+rect[1])/2+origin[0],(rect[2]+rect[3])/2+origin[1]]])
        height,_ = surface.sample(torch.tensor([i]),xy)
        assert height.item() == pytest.approx(rect[4])
        assert atlas.tile_meshes[i].bounds[1,2] == pytest.approx(rect[4])
        assert r['wall_width'] >= (config['all_terrain']['wall_practice']['turn_width'] if r['wall_kind']=='corner' else config['all_terrain']['wall_practice']['width_range'][0])
    assert {np.sign(r['wall_route'][-1][1]) for r in wall if r['wall_kind']=='corner'} == {-1,1}
    old = build_atlas(config['all_terrain']['geometry'])
    assert not any(r.get('wall_kind') for r in old.records)
    # Adding walls cannot perturb the sampled legacy meshes or their RNG draws.
    for i,(a,b) in enumerate(zip(old.tile_meshes,atlas.tile_meshes)):
        old_origin=np.r_[old.origins[i,:2],0.]
        new_origin=np.r_[atlas.origins[i,:2],0.]
        np.testing.assert_allclose(a.vertices-old_origin,b.vertices-new_origin,atol=1.e-12)
        np.testing.assert_array_equal(a.faces,b.faces)


def test_envelope_includes_body_and_legs_not_only_wheels(config):
    doc = collision_document(config['all_terrain']['wall_practice'])
    assert len(doc['primitives']) == 27
    assert {g['body'] for g in doc['primitives']} >= {'base_link','fl_hipx','hr_knee','hl_wheel'}
    low=torch.tensor([[[-.2,-.2,.2],[2.,2.,.0]]]);high=low+.1
    walls=torch.tensor([[[-.15,-.05,-.3,.3,1.]]])
    assert envelope_clearance(low,high,walls).item() < 0
    assert envelope_clearance(low[:,1:],high[:,1:],walls).item() > 1
    assert envelope_clearance(low+torch.tensor([0.,2.,0.]),high+torch.tensor([0.,2.,0.]),walls).item() > 1


def test_primitive_envelope_matches_actual_mujoco_model(config):
    mujoco=pytest.importorskip('mujoco')
    path=Path('/home/ubuntu/桌面/m20_1/M20_perception_Lidar_rl_2/M20_perception_Lidar_rl/deep_robotics_model/M20/mjcf/M20.xml')
    if not path.is_file():pytest.skip('M20 source MJCF not installed on this host')
    m=mujoco.MjModel.from_xml_path(str(path));d=mujoco.MjData(m)
    doc=collision_document(config['all_terrain']['wall_practice'])
    geoms=[i for i in range(m.ngeom) if m.geom_bodyid[i] and (m.geom_contype[i] or m.geom_conaffinity[i])]
    rng=np.random.default_rng(42)
    for _ in range(8):
        d.qpos[2]=.6;d.qpos[7:]=rng.uniform(-.15,.15,m.nq-7);mujoco.mj_forward(m,d)
        bodies=m.geom_bodyid[geoms]
        lo,hi=primitive_aabbs(torch.tensor(d.xpos[bodies][None]),torch.tensor(d.xquat[bodies][None]),doc,conservative=False)
        safe_lo,safe_hi=primitive_aabbs(torch.tensor(d.xpos[bodies][None]),torch.tensor(d.xquat[bodies][None]),doc)
        assert (safe_lo<=lo+1.e-12).all() and (safe_hi>=hi-1.e-12).all()
        for k,g in enumerate(geoms):
            r=d.geom_xmat[g].reshape(3,3)
            if m.geom_type[g]==6:extent=abs(r)@m.geom_size[g]
            else:
                axis=abs(r[:,2]);extent=m.geom_size[g,0]*np.sqrt(np.maximum(0,1-axis*axis))+m.geom_size[g,1]*axis
            np.testing.assert_allclose(lo[0,k].numpy(),d.geom_xpos[g]-extent,atol=1.e-8)
            np.testing.assert_allclose(hi[0,k].numpy(),d.geom_xpos[g]+extent,atol=1.e-8)


def route_state(config,atlas,kind,sample=0):
    cfg=config['all_terrain'];record=next(r for r in atlas.records if r.get('wall_kind')==kind and r['sample']==sample)
    doc=collision_document(cfg['wall_practice']);names=list(dict.fromkeys(g['body'] for g in doc['primitives']))
    data=SimpleNamespace(root_pos_w=torch.zeros(1,3),root_quat_w=torch.tensor([[1.,0.,0.,0.]]),
        default_root_state=torch.tensor([[0.,0.,.568,1.,0.,0.,0.,0.,0.,0.,0.,0.,0.]]),
        body_link_pos_w=torch.zeros(1,len(names),3),body_link_quat_w=torch.tensor([[[1.,0.,0.,0.]]*len(names)]))
    robot=SimpleNamespace(data=data,find_bodies=lambda requested,preserve_order=True: ([names.index(n) for n in requested],requested))
    env=SimpleNamespace(num_envs=1,device='cpu',common_step_counter=0,episode_length_buf=torch.tensor([25]),step_dt=.02,
        scene={'robot':robot},action_manager=SimpleNamespace(action=torch.zeros(1,16),prev_action=torch.zeros(1,16)))
    state=SimpleNamespace(env=env,cfg=cfg,atlas=atlas,origins=torch.tensor(atlas.origins,dtype=torch.float32),tile=torch.tensor([record['tile_id']]))
    wall=WallPractice(state);p=torch.zeros(1,3);yaw=torch.zeros(1);q=torch.tensor([[1.,0.,0.,0.]])
    wall.reset(torch.tensor([0]),p,yaw,q);data.root_pos_w.copy_(p);data.root_quat_w.copy_(q)
    return env,state,wall


@pytest.mark.parametrize('kind',KINDS)
def test_offset_spawn_and_route_requires_real_progress(config,atlas,kind):
    env,state,wall=route_state(config,atlas,kind)
    record=atlas.records[int(state.tile[0])]
    origin=state.origins[state.tile[0],:2]
    p=env.scene['robot'].data.root_pos_w
    assert p[0,2] == pytest.approx(.568) # Never move the dog onto the wall top.
    assert abs(float(p[0,1]-origin[1])) < record['wall_width']/2-.32
    env.scene['robot'].data.root_quat_w[:] = torch.tensor([[1.,0.,0.,0.]])
    cmd=wall.command().clone();assert cmd[0,0]>0 and cmd[0,1]==0
    assert not wall.done[0]
    # Repeated readers at one physics tick cannot advance the path twice.
    torch.testing.assert_close(wall.command(),cmd)
    if kind=='corner':
        env.scene['robot'].data.root_pos_w[0,:2]=wall.routes[state.tile[0],1]
        env.common_step_counter+=1
        turn=wall.command();assert turn[0,0]==0 and turn[0,2]>.1
        env.scene['robot'].data.root_quat_w[:]=torch.tensor([[2**-.5,0.,0.,2**-.5]])
        env.common_step_counter+=1
        assert wall.command()[0,0]>0
    # Final longitudinal plane accepts a safe off-centre trajectory.
    target=wall.routes[state.tile[0],wall.route_lengths[state.tile[0]]-1]
    env.scene['robot'].data.root_pos_w[0,:2]=target
    env.common_step_counter+=1
    assert wall.command().abs().sum()==0 and wall.done[0]
    # Subset reset clears done/command cache, not the shared policy or weights.
    wall.reset(torch.tensor([0]),p,torch.zeros(1),torch.tensor([[1.,0.,0.,0.]]))
    assert not wall.done[0]


def test_zero_or_reversing_motion_cannot_complete_route(config,atlas):
    env,state,wall=route_state(config,atlas,'front_stop')
    env.scene['robot'].data.root_quat_w[:]=torch.tensor([[1.,0.,0.,0.]])
    for step in range(10):
        env.common_step_counter=step
        wall.command()
    assert not wall.done[0]
    env.scene['robot'].data.root_pos_w[0,0]-=2
    env.common_step_counter+=1
    assert wall.command()[0,0]>0 and not wall.done[0]


def test_actual_course_all_v15_features_reset_advance_and_failure(config,atlas,monkeypatch):
    config=deepcopy(config)
    config['all_terrain'].pop('stair_stop', None)
    config['all_terrain'].pop('wall_hold_practice', None)
    config['all_terrain'].pop('stair_hold_practice', None)
    config['all_terrain'].pop('motion_quality', None)
    config['all_terrain'].pop('objective_consistency',None)  # Pin the legacy course.
    import test_cts_all_terrain_course as helpers
    names=list(dict.fromkeys(g['body'] for g in collision_document(config['all_terrain']['wall_practice'])['primitives']))
    wheel_names=['fl_wheel','fr_wheel','hl_wheel','hr_wheel'];wi=[names.index(n) for n in wheel_names]
    class FullRobot(helpers.Robot):
        def __init__(self,n):
            super().__init__(n)
            for key in ('body_link_pos_w','body_com_pos_w','body_lin_vel_w','body_ang_vel_w'):
                setattr(self.data,key,torch.zeros(n,len(names),3))
            self.data.body_link_quat_w=torch.zeros(n,len(names),4);self.data.body_link_quat_w[...,0]=1
        def find_bodies(self,requested,preserve_order=False):
            return [names.index(n) for n in requested],requested
    def full_support(env,state):
        d=env.scene.robot.data;d.body_link_pos_w[:]=d.root_pos_w[:,None,:]
        offsets=torch.tensor([[.25,.20],[.25,-.20],[-.25,.20],[-.25,-.20]])
        d.body_link_pos_w[:,wi,:2]=d.root_pos_w[:,None,:2]+offsets
        z,normal=state.surface.sample(state.tile,d.body_link_pos_w[:,wi,:2])
        d.body_link_pos_w[:,wi,2]=z+.09
        d.body_com_pos_w.copy_(d.body_link_pos_w)
        env.scene.sensors['contact_forces'].data.net_forces_w.copy_(normal*100)
        state.measure_tick=-1
    monkeypatch.setattr(helpers,'Robot',FullRobot);monkeypatch.setattr(helpers,'support',full_support)
    env,state=helpers.fake((config,atlas),n=400)
    env.action_manager=SimpleNamespace(action=torch.zeros(400,16),prev_action=torch.zeros(400,16))
    all_ids=torch.arange(400);seen=set();offset=[]
    for _ in range(8):
        wall=state.walls.active;seen.update(state.mode[wall].tolist())
        assert state.num_modes==24 and state.num_groups==14
        assert (state.mode[~wall]<19).all()
        assert not state.tags()['cts_anchor'][wall].any()
        for i in wall.nonzero().flatten().tolist():
            r=atlas.records[int(state.tile[i])];assert state.mode[i]==MODE_START+KINDS.index(r['wall_kind'])
            assert state.group[i]==3*14+9+KINDS.index(r['wall_kind'])
            offset.append(abs(float(env.scene.robot.data.root_pos_w[i,1]-state.origins[state.tile[i],1])))
        env.episode_length_buf.fill_(25);env.common_step_counter+=1
        command=state.raw_command();assert command.shape==(400,3) and torch.isfinite(command).all()
        state.advance();assert torch.isfinite(state.walls.cost()).all()
        state.reset(all_ids);full_support(env,state)
    assert seen==set(range(19,24)) and max(offset)>.02
    # Neither standing still at spawn nor a perfect generic stop is wall-route success.
    env.episode_length_buf.fill_(100);env.common_step_counter+=1
    state.advance();assert not state.success[state.walls.active].any()
    ids=state.walls.active.nonzero().flatten();i=int(ids[0]);r=atlas.records[int(state.tile[i])]
    wall_box=r['wall_rectangles'][0];origin=state.origins[state.tile[i],:2]
    env.scene.robot.data.body_link_pos_w[i,:,:]=torch.tensor([(wall_box[0]+wall_box[1])/2+origin[0],
        (wall_box[2]+wall_box[3])/2+origin[1],.5])
    env.common_step_counter+=1;state.measure_tick=-1;state.advance()
    assert state.failed[i] and state.walls.clearance[i]<0
    assert state.walls.cost()[i]>0
    # Partial reset leaves the other episode's geometry/mode ownership intact.
    before=state.tile.clone();state.reset(torch.tensor([i]));assert torch.equal(before[:i],state.tile[:i])
    state.restore(deepcopy(state.snapshot()))


@pytest.mark.parametrize('kind',(*KINDS,'corner_right'))
def test_feasible_reference_commands_finish_without_envelope_collision(config,atlas,kind):
    """Kinematic route/reference check; NOT a learned-policy physics rollout."""
    mujoco=pytest.importorskip('mujoco')
    path=Path('/home/ubuntu/桌面/m20_1/M20_perception_Lidar_rl_2/M20_perception_Lidar_rl/deep_robotics_model/M20/mjcf/M20.xml')
    if not path.is_file():pytest.skip('M20 source MJCF not installed')
    actual_kind = 'corner' if kind=='corner_right' else kind
    env,state,wall=route_state(config,atlas,actual_kind,sample=int(kind=='corner_right'));env.step_dt=.05
    model=mujoco.MjModel.from_xml_path(str(path));data=mujoco.MjData(model)
    default=np.array([0,-.3,.6,0,-.3,.6,0,.3,-.6,0,.3,-.6,0,0,0,0])
    joint_names=[f'{leg}_{part}_joint' for leg in ('fl','fr','hl','hr') for part in ('hipx','hipy','knee')]+[f'{leg}_wheel_joint' for leg in ('fl','fr','hl','hr')]
    qi=[model.joint(n).qposadr[0] for n in joint_names];data.qpos[qi]=default
    d=env.scene['robot'].data;d.root_pos_w[0,2]=.56866824464
    names=list(dict.fromkeys(g['body'] for g in wall.document['primitives']));body_ids=[model.body(n).id for n in names]
    min_margin=100.
    for step in range(1300):
        env.common_step_counter=step;env.episode_length_buf[:]=max(20,step)
        command=wall.command()[0]
        if wall.done[0]:break
        q=d.root_quat_w[0];yaw=float(torch.atan2(2*q[0]*q[3],1-2*q[3]*q[3]))
        vx,wz=float(command[0]),float(command[2]);yaw+=wz*env.step_dt
        d.root_pos_w[0,0]+=vx*np.cos(yaw)*env.step_dt;d.root_pos_w[0,1]+=vx*np.sin(yaw)*env.step_dt
        d.root_quat_w[0]=torch.tensor([np.cos(yaw/2),0,0,np.sin(yaw/2)])
        data.qpos[:3]=d.root_pos_w[0].numpy();data.qpos[3:7]=d.root_quat_w[0].numpy();mujoco.mj_forward(model,data)
        d.body_link_pos_w[0]=torch.tensor(data.xpos[body_ids]);d.body_link_quat_w[0]=torch.tensor(data.xquat[body_ids])
        wall.measure();min_margin=min(min_margin,float(wall.clearance[0]))
    assert wall.done[0],(kind,d.root_pos_w,wall.segment)
    assert min_margin>0,(kind,min_margin)


def test_deployment_contract_does_not_require_training_envelope_asset(config,monkeypatch):
    import task_support.cts.wall_practice as module
    cfg=deepcopy(config['all_terrain']['wall_practice'])
    cfg['collision_geometry_sha256']='0'*64
    validate(cfg)
    with pytest.raises(ValueError,match='provenance'):collision_document(cfg)
    monkeypatch.setattr(module,'ROOT',Path('/nonexistent-training-tree'))
    validate(config['all_terrain']['wall_practice'])
