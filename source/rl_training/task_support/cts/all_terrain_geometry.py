"""Continuous CTS terrain atlas shared by collision meshes and support queries.

An atlas is sampled once from continuous intervals, then frozen for a run. Course
levels select pools of those actual meshes; they never change a reported height
without changing the collision shape. No global z=0 plane covers depressions.
All public positions, heights, bounds and goals use world coordinates.
"""
from dataclasses import dataclass
import math

import numpy as np
import torch
import trimesh


COURSES = ('up', 'down', 'round_trip', 'motion', 'rough', 'blocks', 'pits',
           'mixed', 'straight', 'platform')
COURSE_IDS = {name: index for index, name in enumerate(COURSES)}

# Runtime values come from all_terrain.geometry in the run's configuration.
# Defaults also make the geometry independently inspectable in CPU tools/tests.
DEFAULTS = dict(seed=42, samples_per_level=16, geometry_levels=7,
    tile_length=24., tile_width=6., grid_resolution=.10,
    height_range=[.15, .25], depth_range=[.15, .35], width_range=[1., 4.],
    steps_range=[8, 20], platform_length_range=[.8, 1.5],
    stiffness_range=[4.e4, 4.e5], damping_range=[500., 1500.],
    initial_height_max=.20, intermediate_height_max=.225,
    initial_depth_min=.25, intermediate_depth_min=.20,
    initial_steps_max=12, initial_width_min=2.,
    route_start=1.5, goal_landing=1.2, goal_clearance=.75,
    start_backward=3., field_half_length=8., ground_thickness=.25, tile_gap=2.,
    field_goal=5., spawn_flat_half_length=.9, spawn_transition_length=.8,
    slope_angle_range=[.03, .30], wave_amplitude_range=[.015, .18],
    wave_length_range=[1.2, 3.5], rough_amplitude_range=[.008, .10],
    rough_wavelength_range=[.35, 1.5], rough_components=5,
    obstacle_count_range=[8, 16], obstacle_length_range=[.18, .65],
    obstacle_width_range=[.18, .65], obstacle_height_range=[.025, .26],
    obstacle_x_range=[1.7, 6.5], obstacle_y_half_range=2.25,
    mixed_obstacle_x_range=[4.2, 6.5], mixed_ramp_height_range=[.02, .12],
    mixed_ramp_start=1.2, mixed_ramp_length=2., pit_count_range=[8, 16],
    pit_x_range=[1.6, 6.8], pit_y_half_range=2.2,
    pit_radius_range=[.25, .65], pit_depth_range=[.015, .19],
    platform_height_range=[.04, 1.], platform_width_range=[1.2, 2.6],
    raised_platform_length_range=[6., 8.], raised_platform_back_edge=-3.)


@dataclass
class Atlas:
    mesh: trimesh.Trimesh
    tile_meshes: list
    origins: np.ndarray
    records: list
    course_ids: np.ndarray
    geometry_levels: np.ndarray
    spawns: np.ndarray
    goals: np.ndarray
    bounds: np.ndarray
    local_bounds: np.ndarray
    goal_local: np.ndarray
    grids: list
    rectangles: list


def _number(cfg, key, default, lower, integer=False):
    value = cfg.get(key, default)
    valid = (type(value) is int if integer else type(value) in (int, float))
    if not valid or not math.isfinite(value) or value < lower:
        raise ValueError(f'Invalid atlas {key}: {value!r}')
    return value


def _box(xmin, xmax, ymin, ymax, top, bottom):
    mesh = trimesh.creation.box((xmax-xmin, ymax-ymin, top-bottom))
    mesh.apply_translation(((xmin+xmax)/2, (ymin+ymax)/2, (top+bottom)/2))
    return mesh


def _heightfield_mesh(x, y, z, thickness):
    """Watertight finite field; diagonal is (00, 11) in every grid cell."""
    nx, ny = len(x), len(y)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    vertices = np.stack((xx, yy, z), -1).reshape(-1, 3)
    ii, jj = np.meshgrid(np.arange(nx-1), np.arange(ny-1), indexing='ij')
    v00 = (ii*ny+jj).ravel()
    v10, v01, v11 = v00+ny, v00+1, v00+ny+1
    top = np.concatenate((np.stack((v00, v10, v11), -1),
                          np.stack((v00, v11, v01), -1)))
    rim = np.concatenate((np.arange(nx)*ny,
                          (nx-1)*ny+np.arange(1, ny),
                          np.arange(nx-2, -1, -1)*ny+ny-1,
                          np.arange(ny-2, 0, -1)))
    bottom = vertices[rim].copy()
    bottom[:, 2] = float(z.min())-thickness
    bottom_ids = np.arange(len(rim))+len(vertices)
    centre_id = len(vertices)+len(rim)
    centre = np.array([[(x[0]+x[-1])/2, (y[0]+y[-1])/2, bottom[0, 2]]])
    following = np.roll(np.arange(len(rim)), -1)
    sides = np.concatenate((np.stack((rim, bottom_ids, bottom_ids[following]), -1),
                            np.stack((rim, bottom_ids[following], rim[following]), -1)))
    base = np.stack((np.full(len(rim), centre_id), bottom_ids[following], bottom_ids), -1)
    return trimesh.Trimesh(np.concatenate((vertices, bottom, centre)),
                          np.concatenate((top, sides, base)), process=False)


def _coverage_fraction(rng, sample, count):
    """Continuous strata with a guaranteed hardest-fifth sample in every pool.

    A frozen finite atlas cannot rely on the probability that ordinary uniform
    draws include a curriculum boundary. Reserve ceil(20% * count) strata for
    that interval; even one-sample diagnostic atlases retain a real boundary.
    Fractions within every stratum remain continuously sampled.
    """
    hard = max(1, math.ceil(count*.2))
    regular = count-hard
    if sample < regular:
        return .8*(sample+rng.random())/regular
    return .8+.2*(sample-regular+rng.random())/hard


def _stair_tile(rng, name, level, sample, bounds, cfg):
    fraction = _coverage_fraction(rng, sample, cfg['samples_per_level'])
    height_high = (cfg['initial_height_max'] if level == 0 else
                   cfg['intermediate_height_max'] if level == 1 else cfg['height_range'][1])
    depth_low = (cfg['initial_depth_min'] if level < 3 else cfg['intermediate_depth_min']
                 if level == 3 else cfg['depth_range'][0])
    height = (cfg['height_range'][0]+fraction*(height_high-cfg['height_range'][0])
              if level < 3 else float(rng.uniform(cfg['height_range'][0], height_high)))
    depth = (depth_low+(1.-fraction)*(cfg['depth_range'][1]-depth_low)
             if level in (3, 4) else float(rng.uniform(depth_low, cfg['depth_range'][1])))
    count = (int(cfg['steps_range'][0]+fraction*(cfg['steps_range'][1]-cfg['steps_range'][0]+1))
             if level == 5 else int(rng.integers(cfg['steps_range'][0],
                                     (cfg['initial_steps_max'] if level < 5 else cfg['steps_range'][1])+1)))
    width = (cfg['width_range'][0]+(1.-fraction)*(cfg['width_range'][1]-cfg['width_range'][0])
             if level == 6 else float(rng.uniform(cfg['initial_width_min'], cfg['width_range'][1])))
    platform = float(rng.uniform(*cfg['platform_length_range']))
    start, top = cfg['route_start'], count*height
    rects = []
    if name == 'down':
        rects.append((bounds[0], start, -width/2, width/2, top))
        for index in range(count-1):
            rects.append((start+index*depth, start+(index+1)*depth,
                          -width/2, width/2, (count-index-1)*height))
        end, spawn_z, goal_z = start+count*depth, top, 0.
    else:
        for index in range(count):
            rects.append((start+index*depth, start+(index+1)*depth,
                          -width/2, width/2, (index+1)*height))
        ascent_end = start+count*depth
        if name == 'up':
            rects.append((ascent_end, bounds[1], -width/2, width/2, top))
            end, goal_z = ascent_end, top
        else:
            rects.append((ascent_end, ascent_end+platform, -width/2, width/2, top))
            descent_start = ascent_end+platform
            for index in range(count-1):
                rects.append((descent_start+index*depth, descent_start+(index+1)*depth,
                              -width/2, width/2, (count-index-1)*height))
            end, goal_z = descent_start+count*depth, 0.
        spawn_z = 0.
    goal = end+cfg['goal_landing']
    if goal+cfg['goal_clearance'] >= bounds[1]:
        raise ValueError('Stair route lacks a full landing after its goal')
    record = dict(height=height, depth=depth, steps=count, width=width,
                  platform_length=platform, route_start=start, route_end=end,
                  top_height=top, motion_variant='stairs')
    return rects, record, spawn_z, (goal, 0., goal_z)


def _draw(rng, interval, strength=1.):
    return float(rng.uniform(interval[0], interval[0]+(interval[1]-interval[0])*strength))


def _covered_draw(rng, interval, strength, sample, count):
    fraction = _coverage_fraction(rng, sample, count)
    return float(interval[0]+fraction*(interval[1]-interval[0])*strength)


def _boundary(record, name, level, cfg):
    """Hardest fifth of the one dimension expanded at this geometry level."""
    if name in ('up', 'down', 'round_trip'):
        if level <= 2:
            high = cfg['initial_height_max'] if level == 0 else cfg['intermediate_height_max'] if level == 1 else cfg['height_range'][1]
            return record['height'] >= cfg['height_range'][0]+.8*(high-cfg['height_range'][0])
        if level <= 4:
            low = cfg['intermediate_depth_min'] if level == 3 else cfg['depth_range'][0]
            return record['depth'] <= low+.2*(cfg['depth_range'][1]-low)
        if level == 5:
            return record['steps'] >= cfg['steps_range'][0]+.8*(cfg['steps_range'][1]-cfg['steps_range'][0])
        return record['width'] <= cfg['width_range'][0]+.2*(cfg['width_range'][1]-cfg['width_range'][0])
    key = {'rough': 'rough_amplitude', 'mixed': 'rough_amplitude',
           'platform': 'height'}.get(name)
    interval = {'rough': 'rough_amplitude_range', 'mixed': 'rough_amplitude_range',
                'platform': 'platform_height_range'}.get(name)
    if name == 'motion':
        key = {'slope': 'slope_angle', 'wave': 'wave_amplitude'}.get(record['motion_variant'])
        interval = {'slope': 'slope_angle_range', 'wave': 'wave_amplitude_range'}.get(record['motion_variant'])
    if key is not None:
        low, high = cfg[interval]
        return abs(record[key]) >= low+.8*(high-low)*(level+1)/7
    if name in ('blocks', 'pits'):
        entries, value, interval = ('obstacles', 'height', 'obstacle_height_range') if name == 'blocks' else ('pits', 'depth', 'pit_depth_range')
        low, high = cfg[interval]
        return max(item[value] for item in record[entries]) >= low+.8*(high-low)*(level+1)/7
    return True


def _obstacles(rng, level, cfg, mixed_height=None, sample=None):
    """Continuous box dimensions/locations with a flat spawn corridor."""
    strength = (level+1)/7
    number = int(rng.integers(cfg['obstacle_count_range'][0], cfg['obstacle_count_range'][1]+1))
    maximum = (_covered_draw(rng, cfg['obstacle_height_range'], strength, sample, cfg['samples_per_level'])
               if sample is not None else None)
    rects, records = [], []
    for obstacle_index in range(number):
        # Every tile has obstacles in both commanded travel directions. Random
        # signs alone can accidentally leave the whole backwards route flat.
        sign = -1. if obstacle_index % 2 == 0 else 1.
        centre_x = sign*_draw(rng, cfg['obstacle_x_range'] if mixed_height is None else cfg['mixed_obstacle_x_range'])
        centre_y = float(rng.uniform(-cfg['obstacle_y_half_range'], cfg['obstacle_y_half_range']))
        length, width = _draw(rng, cfg['obstacle_length_range']), _draw(rng, cfg['obstacle_width_range'])
        height = (_draw(rng, cfg['obstacle_height_range'], strength) if maximum is None
                  else maximum if obstacle_index == 0 else float(rng.uniform(cfg['obstacle_height_range'][0], maximum)))
        if mixed_height is not None:
            # Mixed ground reaches this plateau before the obstacle band.
            height += mixed_height
        rects.append((centre_x-length/2, centre_x+length/2,
                      centre_y-width/2, centre_y+width/2, height))
        records.append(dict(x=centre_x, y=centre_y, length=length, width=width, height=height))
    return rects, records


def _field_tile(rng, name, level, sample, bounds, resolution, cfg):
    strength = (level+1)/7
    nx = max(2, int(math.ceil((bounds[1]-bounds[0])/resolution))+1)
    ny = max(2, int(math.ceil((bounds[3]-bounds[2])/resolution))+1)
    if name in ('straight', 'blocks', 'platform'):
        nx, ny = 2, 2
    elif name == 'motion':
        ny = 2
    x, y = np.linspace(bounds[0], bounds[1], nx), np.linspace(bounds[2], bounds[3], ny)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    z = np.zeros_like(xx)
    rects, record = [], dict(motion_variant='flat')
    # Zero exactly throughout the robot's spawn support polygon. Terrain is
    # present in both directions outside the stable reset area.
    gate = np.clip((np.abs(xx)-cfg['spawn_flat_half_length'])/cfg['spawn_transition_length'], 0., 1.)
    gate = gate*gate*(3.-2.*gate)
    if name == 'motion':
        variant = ('flat', 'slope', 'wave')[sample % 3]
        record['motion_variant'] = variant
        # Cover each nonflat variant independently: a boundary slope does not
        # establish coverage of the wave amplitude interval, or vice versa.
        variant_count = (cfg['samples_per_level']+2-sample % 3)//3
        variant_sample = sample//3
        if variant == 'slope':
            angle = _covered_draw(rng, cfg['slope_angle_range'], strength, variant_sample, variant_count)
            angle *= -1 if rng.random() < .5 else 1
            # Rounded ramp entry retains a flat spawn rather than teleporting
            # the robot onto an inclined plane with an upright root pose.
            z = np.tan(angle)*xx*gate
            record['slope_angle'] = angle
        elif variant == 'wave':
            amplitude = _covered_draw(rng, cfg['wave_amplitude_range'], strength, variant_sample, variant_count)
            wavelength = _draw(rng, cfg['wave_length_range'])
            phase = float(rng.uniform(-np.pi, np.pi))
            z = amplitude*np.sin(2*np.pi*xx/wavelength+phase)*gate
            record.update(wave_amplitude=amplitude, wave_length=wavelength, wave_phase=phase)
    elif name in ('rough', 'mixed'):
        amplitude = _covered_draw(rng, cfg['rough_amplitude_range'], strength, sample, cfg['samples_per_level'])
        terms = []
        for _ in range(cfg['rough_components']):
            length_x, length_y = _draw(rng, cfg['rough_wavelength_range']), _draw(rng, cfg['rough_wavelength_range'])
            phase_x, phase_y = float(rng.uniform(-np.pi, np.pi)), float(rng.uniform(-np.pi, np.pi))
            z += amplitude/cfg['rough_components']*np.sin(2*np.pi*xx/length_x+phase_x)*np.sin(2*np.pi*yy/length_y+phase_y)
            terms.append((length_x, length_y, phase_x, phase_y))
        z *= gate
        record.update(rough_amplitude=amplitude, rough_components=terms)
        if name == 'mixed':
            # Flat -> graded rise -> rough plateau -> isolated obstacles.
            ramp_height = _draw(rng, cfg['mixed_ramp_height_range'], strength)
            ramp = np.clip((np.abs(xx)-cfg['mixed_ramp_start'])/cfg['mixed_ramp_length'], 0., 1.)
            z += ramp_height*ramp
            rects, obstacles = _obstacles(rng, level, cfg, mixed_height=ramp_height)
            record.update(obstacles=obstacles, ramp_height=ramp_height)
    elif name == 'pits':
        pits = []
        maximum_depth = _covered_draw(rng, cfg['pit_depth_range'], strength, sample, cfg['samples_per_level'])
        for pit_index in range(int(rng.integers(cfg['pit_count_range'][0], cfg['pit_count_range'][1]+1))):
            centre_x = (-1 if pit_index % 2 == 0 else 1)*_draw(rng, cfg['pit_x_range'])
            centre_y = float(rng.uniform(-cfg['pit_y_half_range'], cfg['pit_y_half_range']))
            radius_x, radius_y = _draw(rng, cfg['pit_radius_range']), _draw(rng, cfg['pit_radius_range'])
            depth = maximum_depth if pit_index == 0 else float(rng.uniform(cfg['pit_depth_range'][0], maximum_depth))
            radius = ((xx-centre_x)/radius_x)**2+((yy-centre_y)/radius_y)**2
            depression = -depth*np.maximum(1-radius, 0.)**2
            z = np.minimum(z, depression)
            pits.append(dict(x=centre_x, y=centre_y, radius_x=radius_x, radius_y=radius_y, depth=depth))
        z *= gate
        record['pits'] = pits
    elif name == 'blocks':
        rects, record['obstacles'] = _obstacles(rng, level, cfg, sample=sample)
    elif name == 'platform':
        height = _covered_draw(rng, cfg['platform_height_range'], strength, sample, cfg['samples_per_level'])
        # Existing platform practice begins on a raised, narrow support slab.
        # A tall box ahead of a ground-level spawn would instead demand a new
        # up-to-one-metre climbing task and discard the fall-prevention lesson.
        length, width = _draw(rng, cfg['raised_platform_length_range']), _draw(rng, cfg['platform_width_range'])
        start = cfg['raised_platform_back_edge']
        rects = [(start, start+length, -width/2, width/2, height)]
        record.update(height=height, platform_length=length, width=width,
                      route_start=start, route_end=start+length)
    goal = (record['route_end']+cfg['goal_landing'] if name == 'platform' else cfg['field_goal'], 0., 0.)
    # Goals on motion fields are sampled from the same mesh by build_atlas.
    return (x, y, z), rects, record, goal


def build_atlas(cfg, wall_practice=None):
    """Build reproducible continuous geometry/material draws without Isaac.

    Required defaults: 10 courses x 7 levels x 16 continuous samples. Integer
    level/count choices are curriculum categories; all physical dimensions and
    contact parameters are sampled inside continuous intervals. ``tile_meshes``
    preserve per-tile contact material binding while ``mesh`` is their exact
    concatenation for rays. Geometry remains immutable during an episode.
    """
    cfg = {**DEFAULTS, **cfg}
    for key, value in cfg.items():
        if key.endswith('_range') and isinstance(value, (tuple, list)):
            if len(value) != 2 or not all(type(v) in (int, float) and math.isfinite(v) for v in value) or not 0 < value[0] < value[1]:
                raise ValueError(f'Invalid continuous atlas interval {key}')
    for key in ('steps_range', 'pit_count_range', 'obstacle_count_range'):
        if not all(type(v) is int for v in cfg[key]):
            raise ValueError(f'Atlas {key} must contain integer counts')
    for key in ('ground_thickness', 'spawn_transition_length', 'tile_gap', 'rough_components'):
        _number(cfg, key, DEFAULTS[key], 1e-6, integer=key == 'rough_components')
    if not cfg['height_range'][0] < cfg['initial_height_max'] <= cfg['intermediate_height_max'] <= cfg['height_range'][1]:
        raise ValueError('Stair height curriculum must expand inside height_range')
    if not cfg['depth_range'][0] <= cfg['intermediate_depth_min'] <= cfg['initial_depth_min'] < cfg['depth_range'][1]:
        raise ValueError('Stair tread curriculum must expand inside depth_range')
    if not cfg['width_range'][0] <= cfg['initial_width_min'] < cfg['width_range'][1]:
        raise ValueError('Stair width curriculum must expand inside width_range')
    if type(cfg['initial_steps_max']) is not int or not cfg['steps_range'][0] <= cfg['initial_steps_max'] <= cfg['steps_range'][1]:
        raise ValueError('Stair length curriculum must expand inside steps_range')
    samples = _number(cfg, 'samples_per_level', 16, 1, integer=True)
    levels = _number(cfg, 'geometry_levels', 7, 1, integer=True)
    if levels > 7:
        raise ValueError('Atlas has seven defined geometry levels')
    seed = _number(cfg, 'seed', 42, 0, integer=True)
    length = _number(cfg, 'tile_length', 24., 24.)
    width = _number(cfg, 'tile_width', 6., 6.)
    if cfg['width_range'][1] > width or cfg['platform_width_range'][1] > width:
        raise ValueError('Route width exceeds the actual tile')
    back = cfg['raised_platform_back_edge']
    if (type(back) not in (int, float) or not math.isfinite(back)
            or not -cfg['field_half_length'] <= back < -cfg['spawn_flat_half_length']
            or back+cfg['raised_platform_length_range'][0] <= cfg['spawn_flat_half_length']
            or back+cfg['raised_platform_length_range'][1]+cfg['goal_landing']+cfg['goal_clearance'] >= cfg['field_half_length']):
        raise ValueError('Raised platform must cover spawn and leave a real landing')
    resolution = _number(cfg, 'grid_resolution', .10, .025)
    if resolution > .25:
        raise ValueError('Atlas resolution above 0.25 m misses small obstacles')
    rng = np.random.default_rng(seed)
    jobs = [(c,name,g,s) for c,name in enumerate(COURSES) for g in range(levels) for s in range(samples)]
    if wall_practice and wall_practice.get("enabled"):
        from .wall_practice import KINDS
        jobs.extend((3,"wall_"+kind,g,s) for kind in KINDS for g in range(levels)
                    for s in range(wall_practice["samples_per_kind_level"]))
    n = len(jobs)
    stride_x = length-cfg['start_backward']+cfg['field_half_length']+cfg['tile_gap']
    stride_y = width+cfg['tile_gap']
    columns = max(1, int(math.ceil(math.sqrt(n*stride_y/stride_x))))
    meshes, origins, records, course_ids, geometry_levels = [], [], [], [], []
    bounds_all, goals, goal_local, grids, rectangles = [], [], [], [], []
    for course, name, level, sample in jobs:
        index = len(records)
        offset = np.array(((index % columns)*stride_x,
                           (index//columns)*stride_y, 0.))
        bounds = (-cfg['start_backward'], length-cfg['start_backward'], -width/2, width/2) if course < 3 else (-cfg['field_half_length'], cfg['field_half_length'], -width/2, width/2)
        if name.startswith("wall_"):
            from .wall_practice import tile
            grid,rects,record,goal = tile(rng,name[5:],level,sample,levels,bounds,wall_practice)
            spawn_z = 0.
        elif course < 3:
            rects, record, spawn_z, goal = _stair_tile(rng, name, level, sample, bounds, cfg)
            grid = (np.array(bounds[:2]), np.array(bounds[2:]), np.zeros((2, 2)))
        else:
            grid, rects, record, goal = _field_tile(rng, name, level, sample, bounds, resolution, cfg)
            spawn_z = record['height'] if name == 'platform' else 0.
        x, y, z = grid
        base = (_box(*bounds, 0., -cfg['ground_thickness']) if z.shape == (2, 2) and not np.any(z)
                else _heightfield_mesh(x, y, z, cfg['ground_thickness']))
        pieces = [base]+[_box(*rect, -cfg['ground_thickness']) for rect in rects]
        mesh = trimesh.util.concatenate(pieces)
        mesh.apply_translation(offset)
        meshes.append(mesh)
        origin = offset.copy()
        origin[2] = spawn_z
        origins.append(origin)
        record.update(course=COURSES[course], course_id=course, geometry_level=level,
                      sample=sample, tile_id=index,
                      contact_stiffness=float(np.exp(rng.uniform(*np.log(cfg['stiffness_range'])))),
                      contact_damping=_draw(rng, cfg['damping_range']),
                      static_friction=1., dynamic_friction=1.)
        record.setdefault('width', width)
        record.setdefault('height', 0.)
        record.setdefault('depth', 0.)
        record.setdefault('steps', 0)
        record['variant'] = record['motion_variant'] if course == 3 else name
        record['stiffness'] = record['contact_stiffness']
        record['damping'] = record['contact_damping']
        record['boundary'] = record.get('wall_boundary', _boundary(record, COURSES[course], level, cfg))
        records.append(record)
        course_ids.append(course)
        geometry_levels.append(level)
        bounds_all.append(bounds)
        goal_local.append(goal)
        goals.append(offset+np.array(goal))
        grids.append(grid)
        rectangles.append(np.asarray(rects, dtype=np.float64).reshape(-1, 5))
    origins = np.asarray(origins)
    local_bounds = np.asarray(bounds_all)
    bounds = local_bounds+origins[:, (0, 0, 1, 1)]
    atlas = Atlas(trimesh.util.concatenate(meshes), meshes, origins, records,
                  np.asarray(course_ids, dtype=np.int64), np.asarray(geometry_levels, dtype=np.int64),
                  origins.copy(), np.asarray(goals), bounds, local_bounds,
                  np.asarray(goal_local), grids, rectangles)
    # Exact mesh heights at goals, including slope/wave and random obstacles.
    surface = TorchSurface(atlas, 'cpu', dtype=torch.float64)
    height, _ = surface.sample(torch.arange(n), torch.as_tensor(atlas.goals[:, :2]))
    atlas.goals[:, 2] = height.numpy()
    atlas.goal_local[:, 2] = atlas.goals[:, 2]
    return atlas


class TorchSurface:
    """GPU-vectorized vertical support on the exact atlas top triangles.

    ``tile_ids`` is [N]; ``xy`` is [N, ..., 2] in world coordinates. Outside a
    finite tile, height/normal are NaN so callers must terminate or mask, rather
    than inventing traversable ground. At an exactly vertical face, the highest
    touching tread is used (the vertical-ray upper envelope).
    """
    def __init__(self, atlas, device, dtype=torch.float32):
        self.device = torch.device(device)
        self.dtype = dtype
        self.origins = torch.as_tensor(atlas.origins[:, :2], device=device, dtype=dtype)
        self.spawns = torch.as_tensor(atlas.spawns, device=device, dtype=dtype)
        self.goals = torch.as_tensor(atlas.goals, device=device, dtype=dtype)
        self.bounds = torch.as_tensor(atlas.bounds, device=device, dtype=dtype)
        self.local_bounds = torch.as_tensor(atlas.local_bounds, device=device, dtype=dtype)
        max_x = max(len(grid[0]) for grid in atlas.grids)
        max_y = max(len(grid[1]) for grid in atlas.grids)
        table = np.zeros((len(atlas.grids), max_x, max_y), dtype=np.float64)
        sizes = []
        for index, (x, y, z) in enumerate(atlas.grids):
            table[index, :len(x), :len(y)] = z
            sizes.append((len(x), len(y)))
        self.height_table = torch.as_tensor(table, device=device, dtype=dtype)
        self.sizes = torch.tensor(sizes, device=device, dtype=torch.long)
        max_rect = max(1, max(len(rects) for rects in atlas.rectangles))
        rect_table = np.zeros((len(atlas.rectangles), max_rect, 5))
        rect_table[..., 4] = -np.inf
        for index, rects in enumerate(atlas.rectangles):
            rect_table[index, :len(rects)] = rects
        self.rectangles = torch.as_tensor(rect_table, device=device, dtype=dtype)
        # USD/Warp convert already-translated mesh vertices to float32. Form
        # the same world edges in double precision before the final cast.
        # Subtracting separately rounded origins from a world ray hit can put
        # an exact step-edge hit on the lower neighbour (one full riser error).
        world_rect_table = rect_table.copy()
        world_rect_table[..., :4] += atlas.origins[:, None, (0, 0, 1, 1)]
        self.world_rectangles = torch.as_tensor(world_rect_table, device=device, dtype=dtype)

    def sample(self, tile_ids, xy):
        ids = torch.as_tensor(tile_ids, device=self.device, dtype=torch.long)
        xy = torch.as_tensor(xy, device=self.device, dtype=self.dtype)
        if ids.ndim != 1 or xy.ndim < 2 or xy.shape[0] != len(ids) or xy.shape[-1] != 2:
            raise ValueError('Surface expects tile_ids[N], xy[N,...,2]')
        shape = xy.shape[:-1]
        if not len(ids):
            return (torch.empty(shape, device=self.device, dtype=self.dtype),
                    torch.empty((*shape, 3), device=self.device, dtype=self.dtype))
        world_points = xy.reshape(len(ids), -1, 2)
        points = world_points-self.origins[ids, None, :]
        bx = self.local_bounds[ids]
        sizes = self.sizes[ids]
        spacing = torch.stack(((bx[:, 1]-bx[:, 0])/(sizes[:, 0]-1),
                               (bx[:, 3]-bx[:, 2])/(sizes[:, 1]-1)), -1)
        uv = (points-bx[:, None, (0, 2)])/spacing[:, None, :]
        cell = torch.floor(uv).long().clamp_min(0)
        cell = torch.minimum(cell, sizes[:, None, :]-2)
        frac = (uv-cell).clamp(0., 1.)
        i, j, tx, ty = cell[..., 0], cell[..., 1], frac[..., 0], frac[..., 1]
        row = ids[:, None]
        p00, p10 = self.height_table[row, i, j], self.height_table[row, i+1, j]
        p01, p11 = self.height_table[row, i, j+1], self.height_table[row, i+1, j+1]
        first = tx >= ty
        dzdx = torch.where(first, p10-p00, p11-p01)/spacing[:, None, 0]
        dzdy = torch.where(first, p11-p10, p01-p00)/spacing[:, None, 1]
        height = torch.where(first, p00+tx*(p10-p00)+ty*(p11-p10),
                             p00+tx*(p11-p01)+ty*(p01-p00))
        normal = torch.stack((-dzdx, -dzdy, torch.ones_like(height)), -1)
        normal = normal/torch.linalg.vector_norm(normal, dim=-1, keepdim=True)
        # Rectangles are actual closed collision boxes, with normals +Z on top.
        rect = self.world_rectangles[ids]
        covered = ((world_points[..., 0, None] >= rect[:, None, :, 0])
                   & (world_points[..., 0, None] <= rect[:, None, :, 1])
                   & (world_points[..., 1, None] >= rect[:, None, :, 2])
                   & (world_points[..., 1, None] <= rect[:, None, :, 3]))
        rect_height = torch.where(covered, rect[:, None, :, 4], -torch.inf).amax(-1)
        raised = rect_height > height
        height = torch.maximum(height, rect_height)
        normal = torch.where(raised[..., None], torch.tensor((0., 0., 1.), device=self.device, dtype=self.dtype), normal)
        world_bounds = self.bounds[ids]
        inside = ((world_points[..., 0] >= world_bounds[:, None, 0]) & (world_points[..., 0] <= world_bounds[:, None, 1])
                  & (world_points[..., 1] >= world_bounds[:, None, 2]) & (world_points[..., 1] <= world_bounds[:, None, 3]))
        height = height.masked_fill(~inside, torch.nan)
        normal = normal.masked_fill(~inside[..., None], torch.nan)
        return height.reshape(shape), normal.reshape(*shape, 3)
