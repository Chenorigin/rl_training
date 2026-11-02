"""Mixed stair enclosures: real mesh rays, no phantom support and fault counters."""
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import torch
import pytest
from task_support.cts.configuration import default_settings
from task_support.cts.all_terrain_config import validate_all_terrain
from task_support.cts.all_terrain_geometry import build_atlas,TorchSurface
from task_support.cts.stair_walls import StairWalls,LAYOUTS,physical_probes
from task_support.cts.wall_practice import collision_document
from test_cts_all_terrain_geometry import vertical_ray


@pytest.fixture(scope='module')
def settings():
    cfg=default_settings()['all_terrain']
    cfg['geometry'].update(samples_per_level=4,grid_resolution=.25)
    return cfg


@pytest.fixture(scope='module')
def atlas(settings):
    return build_atlas(settings['geometry'],settings['wall_practice'],settings['obstacle_response'],settings['stair_walls'])


def test_each_stair_level_has_all_layouts_without_expanding_atlas(settings,atlas):
    for course in ('up','down','round_trip'):
        for level in range(settings['geometry']['geometry_levels']):
            records=[r for r in atlas.records if r['course']==course and r['geometry_level']==level]
            assert sorted(r['stair_wall_layout'] for r in records)==sorted(LAYOUTS)
    assert all('stair_wall_layout' not in r for r in atlas.records if r['course_id']>=3)
    assert all(not r.get('wall_kind') for r in atlas.records if r['course_id']<3)


def test_legacy_geometry_material_and_goal_draws_unchanged(settings,atlas):
    disabled={**settings['stair_walls'],'enabled':False}
    old=build_atlas(settings['geometry'],settings['wall_practice'],settings['obstacle_response'])
    off=build_atlas(settings['geometry'],settings['wall_practice'],settings['obstacle_response'],disabled)
    np.testing.assert_array_equal(old.mesh.vertices,off.mesh.vertices)
    np.testing.assert_array_equal(old.mesh.faces,off.mesh.faces)
    assert old.records==off.records
    np.testing.assert_array_equal(old.origins,atlas.origins)
    np.testing.assert_array_equal(old.goals,atlas.goals)
    for before,after,rect,newrect in zip(old.records,atlas.records,old.rectangles,atlas.rectangles):
        for key,value in before.items():assert after[key]==value,key
        np.testing.assert_array_equal(rect,newrect)


def test_every_wall_top_is_real_triangle_and_never_used_as_tread(settings,atlas):
    full=TorchSurface(atlas,'cpu',dtype=torch.float64)
    support=TorchSurface(atlas,'cpu',dtype=torch.float64,include_stair_walls=False)
    seen=set()
    # Independently intersect actual mesh triangles, including down-stair spawn/top.
    for i,point,top in physical_probes(atlas):
        r=atlas.records[i]
        key=(r['course'],r['geometry_level'],r['stair_wall_layout'])
        if key in seen:continue
        seen.add(key)
        xy=torch.tensor([point],dtype=torch.float64);ids=torch.tensor([i])
        measured,_=vertical_ray(atlas.tile_meshes[i],point)
        assert measured==pytest.approx(top,abs=1e-8)
        ray,_=full.sample(ids,xy);ground,_=support.sample(ids,xy)
        assert ray.item()==pytest.approx(top) and ground.item()<top-.6
        # The route centre remains exactly the original tread surface.
        center=xy.clone();center[:,1]=atlas.origins[i,1]
        a,_=full.sample(ids,center);b,_=support.sample(ids,center)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        for x0,x1,y0,y1,z in r['stair_wall_rectangles']:
            assert y0>=r['width']/2 or y1<=-r['width']/2
            assert atlas.local_bounds[i,2]<y0<y1<atlas.local_bounds[i,3]
    assert len(seen)==3*7*3


def runtime(settings,atlas):
    doc=collision_document(settings['wall_practice'])
    names=list(dict.fromkeys(g['body'] for g in doc['primitives']))
    tiles=[next(r['tile_id'] for r in atlas.records if r['course']=='up' and r['stair_wall_layout']==layout) for layout in LAYOUTS]
    n=4;pos=torch.zeros(n,len(names),3);q=torch.zeros(n,len(names),4);q[...,0]=1
    for i,tile in enumerate(tiles):pos[i,:,:2]=torch.tensor(atlas.origins[tile,:2]);pos[i,:,2]=.5
    data=SimpleNamespace(body_link_pos_w=pos,body_link_quat_w=q)
    robot=SimpleNamespace(data=data,find_bodies=lambda requested,preserve_order:([names.index(x) for x in requested],requested))
    env=SimpleNamespace(num_envs=n,device='cpu',common_step_counter=0,scene={'robot':robot})
    s=SimpleNamespace(env=env,cfg=settings,atlas=atlas,origins=torch.tensor(atlas.origins,dtype=torch.float32),
        tile=torch.tensor(tiles),stairs=torch.ones(n,dtype=torch.bool),teacher=torch.tensor([True,True,False,False]),
        course_ids=torch.zeros(n,dtype=torch.long),success=torch.ones(n,dtype=torch.bool),failed=torch.zeros(n,dtype=torch.bool))
    return s,StairWalls(s)


def test_body_collision_and_nonfinite_link_pose_fail_closed(settings,atlas):
    s,w=runtime(settings,atlas)
    assert not w.failure().any()
    assert (w.cost()==0).all()
    d=s.env.scene['robot'].data
    r=atlas.records[int(s.tile[1])];wall=r['stair_wall_rectangles'][0]
    # Only the main body moves into the side wall; wheel-only checks would miss it.
    body=0;d.body_link_pos_w[1,body,1]=s.origins[s.tile[1],1]+(wall[2]+wall[3])/2
    d.body_link_pos_w[2,body,0]=torch.nan
    d.body_link_pos_w[3,body,0]=torch.inf
    s.env.common_step_counter+=1
    assert w.failure().tolist()==[False,True,True,True]
    assert torch.isfinite(w.cost()).all() and (w.cost()[1:3]>0).all()
    # Resume validates counters rather than silently accepting impossible success.
    s.failed=w.failure();w.frame();w.record(torch.arange(4));saved=w.snapshot()
    assert saved['episodes'].sum()==4 and saved['wall_failures'].sum()==3
    assert saved['successes'].sum()==1 and saved['frames'].sum()==4
    w.checked_snapshot(saved)
    bad=deepcopy(saved);bad['successes']+=10
    with pytest.raises(ValueError):w.checked_snapshot(bad)
    w.reset(torch.tensor([1]));assert torch.isinf(w.clearance[1])


@pytest.mark.parametrize('key,value',[('thickness_m',0),('height_range_m',[.1,.2]),
    ('side_gap_range_m',[-.1,.2]),('side_gap_range_m',[2.,3.]),('enabled',1)])
def test_invalid_enclosures_rejected(settings,key,value):
    cfg=deepcopy(settings);cfg['stair_walls'][key]=value
    with pytest.raises(ValueError):validate_all_terrain(cfg)
