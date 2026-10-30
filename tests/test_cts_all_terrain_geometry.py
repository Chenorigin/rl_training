"""Real mesh geometry, continuous draws, and independent triangle-ray checks."""
from copy import deepcopy
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from task_support.cts.all_terrain_geometry import COURSES, DEFAULTS, TorchSurface, build_atlas


@pytest.fixture(scope='module')
def atlas():
    # Bounded CPU fixture includes all seven grades and all three motion forms.
    return build_atlas(dict(samples_per_level=3, grid_resolution=.2))


def vertical_ray(mesh, xy):
    """Independent projected-triangle intersection, without trimesh ray extras."""
    triangles = mesh.vertices[mesh.faces]
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    v0, v1 = b[:, :2]-a[:, :2], c[:, :2]-a[:, :2]
    q = np.asarray(xy)-a[:, :2]
    det = v0[:, 0]*v1[:, 1]-v0[:, 1]*v1[:, 0]
    valid = np.abs(det) > 1e-12
    safe = np.where(valid, det, 1.)
    u = (q[:, 0]*v1[:, 1]-q[:, 1]*v1[:, 0])/safe
    v = (v0[:, 0]*q[:, 1]-v0[:, 1]*q[:, 0])/safe
    valid &= (u >= -1e-10) & (v >= -1e-10) & (u+v <= 1+1e-10)
    hits = a[:, 2]+u*(b[:, 2]-a[:, 2])+v*(c[:, 2]-a[:, 2])
    valid &= np.cross(b-a, c-a)[:, 2] > 0
    if not valid.any():
        return np.nan, np.full(3, np.nan)
    face = np.argmax(np.where(valid, hits, -np.inf))
    normal = np.cross(b[face]-a[face], c[face]-a[face])
    return hits[face], normal/np.linalg.norm(normal)


def test_each_course_level_has_continuous_geometry_and_material_parameters(atlas):
    assert len(atlas.records) == 10*7*3
    assert len(atlas.tile_meshes) == len(atlas.records)
    for course in range(10):
        for grade in range(7):
            assert ((atlas.course_ids == course) & (atlas.geometry_levels == grade)).sum() == 3
    stairs = [r for r in atlas.records if r['course_id'] < 3]
    assert len({r['height'] for r in stairs}) == len(stairs)
    assert len({r['depth'] for r in stairs}) == len(stairs)
    assert all(.15 <= r['height'] <= .25 and .15 <= r['depth'] <= .35 for r in stairs)
    for r in stairs:
        level = r['geometry_level']
        assert r['steps'] <= (12 if level < 5 else 20)
        assert r['height'] <= (.20 if level == 0 else .225 if level == 1 else .25)
        assert r['depth'] >= (.25 if level < 3 else .20 if level == 3 else .15)
        assert r['width'] >= (2. if level < 6 else 1.)
    for key, low, high in [('stiffness', 4e4, 4e5), ('damping', 500., 1500.)]:
        values = [r[key] for r in atlas.records]
        assert len(set(values)) == len(values)
        assert min(values) >= low and max(values) <= high
    assert {r['variant'] for r in atlas.records if r['course'] == 'motion'} == {'flat', 'slope', 'wave'}


def assert_actual_boundary_coverage(atlas, cfg):
    """Independently derive difficulty from physical dimensions, not flags."""
    for name in COURSES:
        for level in range(cfg['geometry_levels']):
            records = [r for r in atlas.records if r['course'] == name and r['geometry_level'] == level]
            for r in records:
                if name in ('up', 'down', 'round_trip'):
                    if level <= 2:
                        maximum = cfg['initial_height_max'] if level == 0 else cfg['intermediate_height_max'] if level == 1 else cfg['height_range'][1]
                        boundary = r['height'] >= cfg['height_range'][0]+.8*(maximum-cfg['height_range'][0])
                    elif level <= 4:
                        minimum = cfg['intermediate_depth_min'] if level == 3 else cfg['depth_range'][0]
                        boundary = r['depth'] <= minimum+.2*(cfg['depth_range'][1]-minimum)
                    elif level == 5:
                        boundary = r['steps'] >= cfg['steps_range'][0]+.8*(cfg['steps_range'][1]-cfg['steps_range'][0])
                    else:
                        boundary = r['width'] <= cfg['width_range'][0]+.2*(cfg['width_range'][1]-cfg['width_range'][0])
                elif name == 'straight' or (name == 'motion' and r['variant'] == 'flat'):
                    boundary = True  # No geometric difficulty axis on true flat ground.
                else:
                    if name == 'motion':
                        key = 'slope_angle' if r['variant'] == 'slope' else 'wave_amplitude'
                        value, interval = abs(r[key]), cfg[key+'_range']
                    elif name in ('rough', 'mixed'):
                        value, interval = r['rough_amplitude'], cfg['rough_amplitude_range']
                    elif name == 'blocks':
                        value, interval = max(o['height'] for o in r['obstacles']), cfg['obstacle_height_range']
                    elif name == 'pits':
                        value, interval = max(p['depth'] for p in r['pits']), cfg['pit_depth_range']
                    else:
                        value, interval = r['height'], cfg['platform_height_range']
                    boundary = value >= interval[0]+.8*(interval[1]-interval[0])*(level+1)/7
                assert bool(r['boundary']) == bool(boundary), (name, level, r['sample'])
            assert sum(r['boundary'] for r in records) >= math.ceil(.2*len(records)), (name, level)
            if name == 'motion':
                for variant in {r['variant'] for r in records}:
                    subtype = [r for r in records if r['variant'] == variant]
                    assert sum(r['boundary'] for r in subtype) >= math.ceil(.2*len(subtype)), (name, level, variant)


def test_full_default_frozen_atlas_cannot_strand_a_curriculum_without_boundary():
    # This is the full physical atlas, including its default mesh resolution.
    # Ordinary uniform draws at seed 42 previously left rough/g3 with zero
    # hard samples; its student promotion window could never meet coverage.
    full = build_atlas(DEFAULTS)
    assert len(full.records) == 1120
    assert_actual_boundary_coverage(full, DEFAULTS)


@pytest.mark.parametrize('samples', [1, 2, 3])
@pytest.mark.parametrize('seed', [42, 91])
def test_small_frozen_atlases_retain_real_boundary_samples(samples, seed):
    cfg = dict(DEFAULTS, samples_per_level=samples, seed=seed, grid_resolution=.25)
    assert_actual_boundary_coverage(build_atlas(cfg), cfg)


def test_support_height_and_normal_match_independent_collision_triangles(atlas):
    surface = TorchSurface(atlas, 'cpu', dtype=torch.float64)
    rng = np.random.default_rng(912)
    # Every course and level, with both nonconstant heightfield triangles tested.
    for index in range(0, len(atlas.records), 3):
        bounds = atlas.local_bounds[index]
        xy = np.stack((rng.uniform(bounds[0]+.05, bounds[1]-.05, 4),
                       rng.uniform(bounds[2]+.05, bounds[3]-.05, 4)), -1)
        xy += atlas.origins[index, :2]
        heights, normals = surface.sample(torch.tensor([index]), torch.tensor(xy)[None])
        for point, height, normal in zip(xy, heights[0].numpy(), normals[0].numpy()):
            expected, expected_normal = vertical_ray(atlas.tile_meshes[index], point)
            assert height == pytest.approx(expected, abs=2e-9)
            np.testing.assert_allclose(normal, expected_normal, atol=2e-9)


def test_descending_spawn_is_on_high_landing_and_routes_have_real_finish(atlas):
    surface = TorchSurface(atlas, 'cpu', dtype=torch.float64)
    ids = torch.arange(len(atlas.records))
    spawn_height, spawn_normal = surface.sample(ids, torch.tensor(atlas.spawns[:, :2]))
    np.testing.assert_allclose(spawn_height.numpy(), atlas.spawns[:, 2], atol=1e-12)
    np.testing.assert_allclose(spawn_normal.numpy(), np.tile([0, 0, 1], (len(ids), 1)), atol=1e-12)
    for index, record in enumerate(atlas.records):
        if record['course'] == 'down':
            assert atlas.spawns[index, 2] == pytest.approx(record['steps']*record['height'])
            assert atlas.goals[index, 2] == 0.
        if record['course'] == 'up':
            assert atlas.goals[index, 2] == pytest.approx(record['steps']*record['height'])
        if record['course'] == 'round_trip':
            assert atlas.goals[index, 2] == 0.
            assert record['route_end'] > 2*record['depth']*record['steps']
        assert atlas.bounds[index, 0] < atlas.goals[index, 0] < atlas.bounds[index, 1]-.75
        expected, _ = vertical_ray(atlas.tile_meshes[index], atlas.goals[index, :2])
        assert expected == pytest.approx(atlas.goals[index, 2], abs=2e-9)


def test_pits_remain_below_zero_and_two_dimensional_support_is_not_flat(atlas):
    surface = TorchSurface(atlas, 'cpu', dtype=torch.float64)
    pits = np.flatnonzero(atlas.course_ids == COURSES.index('pits'))
    rough = np.flatnonzero(atlas.course_ids == COURSES.index('rough'))
    for index in pits[::3]:
        x, y, z = atlas.grids[index]
        cell = np.unravel_index(np.argmin(z), z.shape)
        point = atlas.origins[index, :2]+[x[cell[0]], y[cell[1]]]
        height, _ = surface.sample([index], np.array([point]))
        assert float(height) < -.005
        assert vertical_ray(atlas.tile_meshes[index], point)[0] == pytest.approx(float(height), abs=2e-9)
    for index in rough[::3]:
        _, _, z = atlas.grids[index]
        assert np.ptp(z[len(z)//4]) > .001


def test_finite_tile_boundaries_and_world_placement_do_not_overlap(atlas):
    surface = TorchSurface(atlas, 'cpu')
    ids = torch.arange(len(atlas.records))
    outside = torch.tensor(atlas.bounds[:, (1, 3)], dtype=torch.float32)+.1
    heights, normals = surface.sample(ids, outside)
    assert torch.isnan(heights).all() and torch.isnan(normals).all()
    bounds = atlas.bounds
    overlap_x = np.minimum(bounds[:, None, 1], bounds[None, :, 1])-np.maximum(bounds[:, None, 0], bounds[None, :, 0])
    overlap_y = np.minimum(bounds[:, None, 3], bounds[None, :, 3])-np.maximum(bounds[:, None, 2], bounds[None, :, 2])
    overlap = (overlap_x > 0) & (overlap_y > 0)
    np.fill_diagonal(overlap, False)
    assert not overlap.any()
    for mesh, bounds in zip(atlas.tile_meshes, atlas.bounds):
        np.testing.assert_allclose(mesh.bounds[:, 0], bounds[:2])
        np.testing.assert_allclose(mesh.bounds[:, 1], bounds[2:])


def test_rectangle_tops_vertical_stair_risers_and_four_wheel_shape(atlas):
    surface = TorchSurface(atlas, 'cpu', dtype=torch.float64)
    index = next(i for i, r in enumerate(atlas.records) if r['course'] == 'up')
    record = atlas.records[index]
    x = record['route_start']+record['depth']/2
    points = np.array([[[x, -.1], [x, .1], [x+record['depth'], -.1], [x+record['depth'], .1]]])
    points += atlas.origins[index, :2]
    heights, normals = surface.sample([index], points)
    assert heights.shape == (1, 4) and normals.shape == (1, 4, 3)
    np.testing.assert_allclose(heights[0].numpy(), np.array([1, 1, 2, 2])*record['height'])
    assert (np.abs(atlas.tile_meshes[index].face_normals[:, 2]) < 1e-10).any()
    for course in ('blocks', 'platform'):
        index = next(i for i, r in enumerate(atlas.records) if r['course'] == course)
        rect = atlas.rectangles[index][0]
        point = atlas.origins[index, :2]+[(rect[0]+rect[1])/2, (rect[2]+rect[3])/2]
        height, normal = surface.sample([index], np.array([point]))
        assert float(height) >= rect[4]
        np.testing.assert_allclose(normal[0].numpy(), [0, 0, 1])


def test_backwards_courses_really_contain_nonflat_collision_geometry(atlas):
    surface = TorchSurface(atlas, 'cpu', dtype=torch.float64)
    for index, record in enumerate(atlas.records):
        name = record['course']
        if name not in ('rough', 'blocks', 'pits', 'mixed'):
            continue
        x, y, z = atlas.grids[index]
        if name in ('rough', 'mixed'):
            # Check both halves, outside the deliberate stable spawn area.
            for side in (x < -1.7, x > 1.7):
                assert np.ptp(z[side]) > .001
        elif name == 'blocks':
            obstacles = record['obstacles']
            assert any(o['x'] < 0 for o in obstacles) and any(o['x'] > 0 for o in obstacles)
            obstacle = next(o for o in obstacles if o['x'] < 0)
            point = atlas.origins[index, :2]+[obstacle['x'], obstacle['y']]
            actual = vertical_ray(atlas.tile_meshes[index], point)[0]
            assert actual >= obstacle['height']-1e-9
            height, _ = surface.sample([index], np.array([point]))
            assert float(height) == pytest.approx(actual, abs=2e-9)
        else:
            assert any(p['x'] < 0 for p in record['pits']) and any(p['x'] > 0 for p in record['pits'])
            assert z[x < -1.7].min() < -.005
            assert z[x > 1.7].min() < -.005


def test_empty_selection_keeps_four_wheel_shape(atlas):
    heights, normals = TorchSurface(atlas, 'cpu').sample(torch.empty(0, dtype=torch.long), torch.empty(0, 4, 2))
    assert heights.shape == (0, 4) and normals.shape == (0, 4, 3)


def test_platform_preserves_raised_spawn_support_and_lower_catch_floor(atlas):
    surface = TorchSurface(atlas, 'cpu', dtype=torch.float64)
    for index, record in enumerate(atlas.records):
        if record['course'] != 'platform':
            continue
        assert record['route_start'] < -.9 and record['route_end'] > .9
        assert 6. <= record['platform_length'] <= 8.
        assert atlas.origins[index, 2] == pytest.approx(record['height'])
        assert atlas.goals[index, 0] > atlas.origins[index, 0]+record['route_end']
        assert atlas.goals[index, 2] == 0.
        local = np.array([[0., 0.], [-.3, .2], [.3, -.2],
                          [record['route_end']+.5, 0.], [0., record['width']/2+.1]])
        points = local+atlas.origins[index, :2]
        heights, _ = surface.sample([index], points[None])
        np.testing.assert_allclose(heights[0].numpy(), [record['height']]*3+[0., 0.], atol=1e-9)
        for point, height in zip(points, heights[0].numpy()):
            assert vertical_ray(atlas.tile_meshes[index], point)[0] == pytest.approx(height, abs=2e-9)


def test_float32_far_world_step_edges_match_real_mesh_without_tolerance(atlas):
    index = next(i for i, r in enumerate(atlas.records) if r['course'] == 'up' and r['geometry_level'] == 3)
    origin = np.array([10003.2349, -9021.345678, 0.])
    mesh = atlas.tile_meshes[index].copy()
    shift = origin-atlas.origins[index]
    mesh.apply_translation(shift)
    # This is the actual vertex conversion used by USD/Warp/PhysX, not an
    # analytic local-coordinate surface standing in for the collision mesh.
    mesh.vertices = mesh.vertices.astype(np.float32).astype(np.float64)
    bounds = atlas.local_bounds[index]+origin[[0, 0, 1, 1]]
    selected = SimpleNamespace(origins=origin[None], spawns=origin[None],
        goals=atlas.goals[index:index+1]+shift, bounds=bounds[None],
        local_bounds=atlas.local_bounds[index:index+1], grids=[atlas.grids[index]],
        rectangles=[atlas.rectangles[index]])
    surface = TorchSurface(selected, 'cpu')
    rects = atlas.rectangles[index]
    edges = np.unique((rects[:, :2]+origin[0]).astype(np.float32))
    y = np.float32(origin[1])
    points = []
    for edge in edges:
        if bounds[0]+.1 < edge < bounds[1]-.1:
            for x in (np.nextafter(edge, np.float32(-np.inf)), edge,
                      np.nextafter(edge, np.float32(np.inf))):
                points.append([x, y])
    # Check the finite outside boundary with the same world-space rounding.
    side = np.float32(origin[1]+rects[0, 3])
    centre_x = np.float32(origin[0]+(rects[0, 0]+rects[0, 1])/2)
    points.extend([[centre_x, np.nextafter(side, np.float32(-np.inf))],
                   [centre_x, side], [centre_x, np.nextafter(side, np.float32(np.inf))]])
    points = np.array(points, dtype=np.float32)
    heights, _ = surface.sample([0], points[None])
    expected = np.array([vertical_ray(mesh, xy.astype(np.float64))[0] for xy in points])
    np.testing.assert_allclose(heights[0].numpy(), expected, rtol=0, atol=2e-7)
    # A negative oracle proves these points expose the previous whole-riser
    # mismatch, rather than merely testing points well inside a tread.
    local = points-origin[:2].astype(np.float32)
    rounded_rects = rects.astype(np.float32)
    covered = ((local[:, 0, None] >= rounded_rects[None, :, 0])
               & (local[:, 0, None] <= rounded_rects[None, :, 1])
               & (local[:, 1, None] >= rounded_rects[None, :, 2])
               & (local[:, 1, None] <= rounded_rects[None, :, 3]))
    old = np.maximum(0., np.where(covered, rounded_rects[None, :, 4], -np.inf).max(-1))
    assert np.max(np.abs(old-expected)) > .10


def test_configuration_is_applied_to_actual_meshes_and_material_records():
    cfg = dict(samples_per_level=1, geometry_levels=1, grid_resolution=.25,
               height_range=[.16, .26], initial_height_max=.21, intermediate_height_max=.235,
               stiffness_range=[123456., 123457.], damping_range=[876., 877.])
    atlas = build_atlas(cfg)
    twin = build_atlas(cfg)
    assert atlas.records == twin.records
    np.testing.assert_array_equal(atlas.mesh.vertices, twin.mesh.vertices)
    for record in atlas.records:
        assert 123456 <= record['stiffness'] <= 123457
        assert 876 <= record['damping'] <= 877
    record = atlas.records[0]
    assert .16 <= record['height'] <= .21
    xy = atlas.origins[0, :2]+[record['route_start']+record['depth']/2, 0.]
    assert vertical_ray(atlas.tile_meshes[0], xy)[0] == pytest.approx(record['height'])


@pytest.mark.parametrize('key,value', [('samples_per_level', 0), ('geometry_levels', 8),
    ('grid_resolution', 1.), ('height_range', [.25, .15]), ('steps_range', [8., 20.]),
    ('initial_height_max', .30), ('initial_depth_min', .10), ('seed', True)])
def test_invalid_geometry_contract_fails_closed(key, value):
    cfg = deepcopy(DEFAULTS)
    cfg[key] = value
    with pytest.raises(ValueError):
        build_atlas(cfg)
