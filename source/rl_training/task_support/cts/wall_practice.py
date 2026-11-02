"""Optional v15 wall courses; privileged geometry stays outside policy inputs.

Whole-robot primitive AABBs are conservative collision envelopes, not contact
force labels. The physical mesh and the unmodified student height map keep walls.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

KINDS = ('single_wall', 'corridor', 'doorway', 'front_stop', 'corner')
MODE_START = 19
ROOT = Path(__file__).resolve().parents[4]


def enabled(cfg):
    return bool(cfg and cfg.get('enabled'))


def collision_document(cfg):
    path = ROOT / cfg['collision_geometry']
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != cfg['collision_geometry_sha256']:
        raise ValueError('Wall collision envelope provenance mismatch')
    document = json.loads(raw)
    for item in document['primitives']:
        if item['type'] not in (5, 6) or not item['body']:
            raise ValueError('Unsupported wall collision envelope primitive')
        values = item['pos']+item['quat']+item['size']
        if not all(math.isfinite(v) for v in values) or not math.isclose(sum(v*v for v in item['quat']),1.,abs_tol=1.e-5):
            raise ValueError('Invalid wall collision envelope transform')
    if not document['primitives']:
        raise ValueError('Empty robot collision envelope')
    return document


def validate(cfg):
    keys = {'enabled','episode_fraction','samples_per_kind_level','width_range','initial_min_width',
        'wall_height_range','wall_thickness','speed_range','yaw_rate','heading_kp','align_angle','minimum_turn_rate',
        'waypoint_tolerance','spawn_yaw','spawn_margin','spawn_half_length','spawn_half_width',
        'offset_fraction','turn_width','stop_margin','timeout_s','safe_margin','penetration_limit',
        'clearance_weight','leg_delta_weight','leg_jerk_weight','target_delta_scale',
        'collision_geometry','collision_geometry_sha256'}
    if isinstance(cfg, dict) and 'corridor_width_m' in cfg:
        keys.add('corridor_width_m')
    if not isinstance(cfg,dict) or set(cfg) != keys or type(cfg['enabled']) is not bool:
        raise ValueError('Unknown or missing wall practice configuration')
    for key in ('width_range','wall_height_range','speed_range'):
        value = cfg[key]
        if (not isinstance(value,list) or len(value) != 2 or
                any(type(v) not in (int,float) or not math.isfinite(v) or v <= 0 for v in value) or value[0] >= value[1]):
            raise ValueError('Invalid wall interval: '+key)
    excluded = {'enabled','collision_geometry','collision_geometry_sha256','width_range','wall_height_range','speed_range'}
    for key in keys-excluded:
        value = cfg[key]
        if type(value) not in (int,float) or not math.isfinite(value) or value <= 0:
            raise ValueError('Invalid wall practice value: '+key)
    if type(cfg['samples_per_kind_level']) is not int or not 0 < cfg['episode_fraction'] < 1:
        raise ValueError('Wall practice must retain non-wall motion episodes')
    if not 0 < cfg['offset_fraction'] <= 1 or not cfg['width_range'][0] <= cfg['initial_min_width'] <= cfg['width_range'][1]:
        raise ValueError('Invalid wall width curriculum or lateral offset')
    if 'corridor_width_m' in cfg and not cfg['width_range'][0] <= cfg['corridor_width_m'] <= cfg['width_range'][1]:
        raise ValueError('Fixed corridor width outside the validated envelope')
    if cfg['width_range'][0] <= 2*(cfg['spawn_half_width']+cfg['spawn_margin']):
        raise ValueError('Narrow wall course cannot contain the supported spawn envelope')
    if cfg['turn_width'] <= 2*(math.hypot(cfg['spawn_half_length'],cfg['spawn_half_width'])+cfg['spawn_margin']):
        raise ValueError('Wall corner requires a feasible rotating envelope')
    if not cfg['width_range'][0] <= cfg['turn_width'] <= cfg['width_range'][1]:
        raise ValueError('Wall turning bay must fit width_range')
    if cfg['wall_height_range'][0] < .7:
        raise ValueError('Wall course must not turn into a climbable step course')
    if not .1 < cfg['minimum_turn_rate'] <= cfg['yaw_rate'] or cfg['spawn_yaw'] >= math.pi/2:
        raise ValueError('Wall turns must exceed the legacy heading deadband and use a forward spawn')
    path = cfg['collision_geometry']
    sha = cfg['collision_geometry_sha256']
    if (not isinstance(path,str) or Path(path).is_absolute() or '..' in Path(path).parts
            or not path.endswith('.json') or not isinstance(sha,str) or len(sha) != 64
            or any(c not in '0123456789abcdef' for c in sha)):
        raise ValueError('Invalid wall envelope reference')
    # Deployment validates the saved numerical contract without requiring the
    # training-only geometry asset. WallPractice verifies its bytes at runtime.
    return cfg


def tile(rng, kind, level, sample, levels, bounds, cfg, response=None):
    """Real solid walls on level ground; route avoids every wall interior."""
    from .all_terrain_geometry import _coverage_fraction
    strength = level/max(levels-1,1)
    low = cfg['initial_min_width']+(cfg['width_range'][0]-cfg['initial_min_width'])*strength
    if kind == 'corner': low = max(low,cfg['turn_width'])
    fraction = _coverage_fraction(rng,sample,cfg['samples_per_kind_level'])
    width = cfg['width_range'][1]-(cfg['width_range'][1]-low)*fraction
    if kind == 'corridor' and 'corridor_width_m' in cfg:
        width = cfg['corridor_width_m']
    height = float(rng.uniform(*cfg['wall_height_range']))
    t = cfg['wall_thickness']
    rects = []
    def box(x0,x1,y0,y1): rects.append((x0,x1,y0,y1,height))
    variant = 'wall'
    if kind == 'single_wall':
        half_length, thickness = 4., t
        if enabled(response):
            variant = ('wall','finite_wall','post','post')[sample % 4]
            if variant != 'wall':
                half_length = response['finite_wall_half_length_m'] if variant == 'finite_wall' else response['post_half_width_m']
                thickness = t if variant == 'finite_wall' else 2*response['post_half_width_m']
        sign = (-1 if rng.random() < .5 else 1) if enabled(response) else (-1 if sample % 2 else 1)
        if sign > 0: box(-half_length,half_length,width/2,width/2+thickness)
        else: box(-half_length,half_length,-width/2-thickness,-width/2)
    elif kind in ('corridor','doorway','front_stop'):
        outer = width if kind != 'doorway' else cfg['width_range'][1]
        box(-4.,4.,outer/2,outer/2+t)
        box(-4.,4.,-outer/2-t,-outer/2)
        if kind == 'doorway':
            box(-t/2,t/2,width/2,outer/2+t)
            box(-t/2,t/2,-outer/2-t,-width/2)
        if kind == 'front_stop': box(2.,2.+t,-outer/2-t,outer/2+t)
    else:
        # Four walls overlap at outer and inner joints; a square turning bay
        # stays wider than the rotating envelope even at the hardest level.
        h = width/2
        box(-4.,h+t,-h-t,-h)
        box(h,h+t,-h-t,3.)
        box(-4.,-h,h,h+t)
        box(-h-t,-h,h,3.)
    end = 2.-cfg['spawn_half_length']-cfg['stop_margin'] if kind == 'front_stop' else 2.
    route = [(-2.,0.),(0.,0.),(0.,2.)] if kind == 'corner' else [(-2.,0.),(end,0.)]
    if kind == 'corner' and sample % 2:
        rects = [(x0,x1,-y1,-y0,h) for x0,x1,y0,y1,h in rects]
        route = [(x,-y) for x,y in route]
    record = dict(wall_kind=kind,wall_width=width,wall_lower_width=low,wall_rectangles=rects,
        wall_route=route,width=width,motion_variant='wall_'+kind,
        wall_boundary=fraction >= .8, obstacle_variant=variant)
    grid = (np.array(bounds[:2]),np.array(bounds[2:]),np.zeros((2,2)))
    return grid,rects,record,(*route[-1],0.)


def rotation(q):
    w,x,y,z = q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),
        2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),
        2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)),-1).reshape(*q.shape[:-1],3,3)


def primitive_aabbs(position, quaternion, document, *, conservative=True):
    """Input body poses already gathered in primitive order, wxyz quaternions."""
    r = rotation(quaternion)
    local = position.new_tensor([g['pos'] for g in document['primitives']])
    center = position+(r@local[...,None]).squeeze(-1)
    geom_r = r@rotation(position.new_tensor([g['quat'] for g in document['primitives']]))
    size = position.new_tensor([g['size'] for g in document['primitives']])
    extent = (geom_r.abs()@size[...,None]).squeeze(-1)
    cylinder = torch.tensor([g['type']==5 for g in document['primitives']],device=position.device)
    axis = geom_r[...,2].abs().clamp(max=1.)
    radial = torch.ones_like(axis) if conservative else (1-axis.square()).clamp_min(0).sqrt()
    cylinder_extent = size[:,0,None]*radial+size[:,1,None]*axis
    extent = torch.where(cylinder[None,:,None],cylinder_extent,extent)
    return center-extent,center+extent


def envelope_clearance(low, high, walls):
    """Signed separation to closed wall boxes, including wall height (3D)."""
    a = torch.stack((walls[...,0],walls[...,2],torch.zeros_like(walls[...,0])),-1)
    b = torch.stack((walls[...,1],walls[...,3],walls[...,4]),-1)
    separation = torch.maximum(a[:,None]-high[:,:,None],low[:,:,None]-b[:,None])
    distance = separation.clamp_min(0).norm(dim=-1)+separation.amax(-1).clamp_max(0)
    present = torch.isfinite(walls[...,4])
    return torch.where(present[:,None],distance,torch.inf).amin(dim=(-1,-2))


class WallPractice:
    def __init__(self,state):
        self.state,self.cfg = state,state.cfg['wall_practice']
        n,device = state.env.num_envs,state.env.device
        self.document = collision_document(self.cfg)
        self.active = torch.zeros(n,dtype=torch.bool,device=device)
        self.done = self.active.clone()
        self.segment = torch.ones(n,dtype=torch.long,device=device)
        self.speed = torch.zeros(n,device=device)
        self.clearance = torch.full((n,),torch.inf,device=device)
        self.delta_cost = torch.zeros(n,device=device)
        self.jerk_cost = self.delta_cost.clone()
        self.last_delta = torch.zeros(n,12,device=device)
        self.history_valid = self.active.clone()
        self.command_cache = torch.zeros(n,3,device=device)
        self.command_tick = -1
        self.measure_tick = -1
        count = max(len(r.get('wall_rectangles',[])) for r in state.atlas.records)
        walls = torch.full((len(state.atlas.records),count,5),torch.nan,device=device)
        route = torch.zeros(len(state.atlas.records),3,2,device=device)
        lengths = torch.zeros(len(state.atlas.records),dtype=torch.long,device=device)
        for i,r in enumerate(state.atlas.records):
            if not r.get('wall_kind'):continue
            walls[i,:len(r['wall_rectangles'])] = walls.new_tensor(r['wall_rectangles'])
            route[i,:len(r['wall_route'])] = route.new_tensor(r['wall_route'])+state.origins[i,:2]
            lengths[i] = len(r['wall_route'])
        self.walls,self.routes,self.route_lengths = walls,route,lengths
        names = list(dict.fromkeys(g['body'] for g in self.document['primitives']))
        ids,actual = state.env.scene['robot'].find_bodies(names,preserve_order=True)
        if len(ids) != len(names) or list(actual) != names: raise ValueError('Wall envelope robot body mapping differs')
        lookup = dict(zip(names,ids))
        self.body_ids = [lookup[g['body']] for g in self.document['primitives']]
        from .configuration import default_settings
        limits = getattr(state.env.cfg,'cts_joint_limits',None) if hasattr(state.env,'cfg') else None
        limits = limits or default_settings()['joint_limits']
        self.action_scale = walls.new_tensor(limits['action_scale'])
        self.last_measure_step = torch.full((n,),-1,dtype=torch.long,device=device)

    def reset(self,ids,position,yaw,quat):
        state,cfg = self.state,self.cfg
        active = torch.tensor([bool(state.atlas.records[int(i)].get('wall_kind')) for i in state.tile[ids]],device=ids.device)
        self.active[ids] = active
        selected = ids[active]
        if selected.numel():
            starts = self.routes[state.tile[selected],0]
            width = position.new_tensor([state.atlas.records[int(i)]['wall_width'] for i in state.tile[selected]])
            angle = torch.empty_like(width).uniform_(-cfg['spawn_yaw'],cfg['spawn_yaw'])
            # Reduce yaw before sampling lateral offsets, so the full rotating
            # spawn reservation fits even the narrowest straight corridor.
            for _ in range(8):
                side = cfg['spawn_half_width']*angle.cos().abs()+cfg['spawn_half_length']*angle.sin().abs()
                angle = torch.where(side+cfg['spawn_margin'] < width/2,angle,angle/2)
            free = (width/2-side-cfg['spawn_margin']).clamp_min(0)*cfg['offset_fraction']
            corners = torch.tensor([state.atlas.records[int(i)]['wall_kind']=='corner' for i in state.tile[selected]],device=ids.device)
            free = torch.where(corners,free.clamp(max=cfg['waypoint_tolerance']/2),free)
            offset = (torch.rand_like(free)*2-1)*free
            position[active,:2] = starts+torch.stack((torch.zeros_like(offset),offset),-1)
            position[active,2] = state.env.scene['robot'].data.default_root_state[selected,2]
            yaw[active] = angle
            quat[active] = 0.
            quat[active,0] = (angle/2).cos()
            quat[active,3] = (angle/2).sin()
            self.speed[selected] = torch.empty_like(width).uniform_(*cfg['speed_range'])
        self.done[ids] = False
        self.segment[ids] = 1
        self.last_delta[ids] = 0.
        self.history_valid[ids] = False
        self.last_measure_step[ids] = -1
        self.command_tick = self.measure_tick = -1

    def command(self):
        state,env,cfg = self.state,self.state.env,self.cfg
        if self.command_tick == env.common_step_counter:return self.command_cache
        d = env.scene['robot'].data
        holding = getattr(state, 'wall_hold', None)
        eligible = self.active if holding is None else self.active & ~holding.active
        ids = eligible.nonzero().flatten()
        result = torch.zeros_like(self.command_cache)
        if ids.numel():
            routes = self.routes[state.tile[ids]]
            end = self.segment[ids]
            target = routes[torch.arange(len(ids),device=ids.device),end]
            error = target-d.root_pos_w[ids,:2]
            previous = routes[torch.arange(len(ids),device=ids.device),end-1]
            tangent = target-previous
            remaining = (error*tangent/tangent.norm(dim=-1)[:,None]).sum(-1)
            arrived = remaining <= cfg['waypoint_tolerance']
            final = end >= self.route_lengths[state.tile[ids]]-1
            self.done[ids] |= arrived & final
            advance = arrived & ~final
            self.segment[ids] += advance.long()
            target = routes[torch.arange(len(ids),device=ids.device),self.segment[ids]]
            previous = routes[torch.arange(len(ids),device=ids.device),self.segment[ids]-1]
            tangent = target-previous
            heading = torch.atan2(tangent[:,1],tangent[:,0])
            q = d.root_quat_w[ids]
            yaw = torch.atan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2].square()+q[:,3].square()))
            angle = torch.atan2(torch.sin(heading-yaw),torch.cos(heading-yaw))
            distance = ((target-d.root_pos_w[ids,:2])*tangent/tangent.norm(dim=-1)[:,None]).sum(-1).clamp_min(0)
            speed = torch.minimum(self.speed[ids],distance*cfg['heading_kp'])
            # Turn at the reserved bay before translating down the next leg.
            result[ids,0] = speed*(angle.abs() <= cfg['align_angle'])
            turn = (cfg['heading_kp']*angle.abs()).clamp(cfg['minimum_turn_rate'],cfg['yaw_rate'])*angle.sign()
            result[ids,2] = torch.where(angle.abs() > cfg['align_angle'],turn,0.)
            age = env.episode_length_buf[ids]*env.step_dt
            active = ~self.done[ids] & (age >= state.cfg['commands']['initial_hold_s'])
            result[ids] *= active[:,None]
        self.command_cache.copy_(result)
        self.command_tick = env.common_step_counter
        return self.command_cache

    def measure(self):
        env,state,cfg = self.state.env,self.state,self.cfg
        if self.measure_tick == env.common_step_counter:return
        fresh = self.last_measure_step != env.common_step_counter
        ids = (self.active & fresh).nonzero().flatten()
        self.clearance[fresh] = torch.inf
        if ids.numel():
            d = env.scene['robot'].data
            low,high = primitive_aabbs(d.body_link_pos_w[ids][:,self.body_ids],
                d.body_link_quat_w[ids][:,self.body_ids],self.document)
            origin = state.origins[state.tile[ids],:2]
            shift = torch.cat((origin,torch.zeros_like(origin[:,:1])),-1)[:,None]
            margin = envelope_clearance(low-shift,high-shift,self.walls[state.tile[ids]])
            self.clearance[ids] = torch.nan_to_num(margin,nan=-torch.inf,neginf=-torch.inf)
        delta = (env.action_manager.action[:,:12]-env.action_manager.prev_action[:,:12])
        delta = delta*self.action_scale
        self.delta_cost[fresh] = (delta[fresh]/cfg['target_delta_scale']).square().mean(-1)
        self.jerk_cost[fresh] = ((delta[fresh]-self.last_delta[fresh])/cfg['target_delta_scale']).square().mean(-1)*self.history_valid[fresh]
        self.last_delta[fresh] = delta[fresh]
        self.history_valid[fresh] = self.active[fresh]
        self.last_measure_step[fresh] = env.common_step_counter
        self.measure_tick = env.common_step_counter

    def cost(self):
        self.measure()
        cfg = self.cfg
        proximity = ((cfg['safe_margin']-self.clearance)/cfg['safe_margin']).clamp(0,4).square()
        response = self.state.cfg.get('obstacle_response',{})
        delta_weight, jerk_weight = cfg['leg_delta_weight'], cfg['leg_jerk_weight']
        if enabled(response):
            delta_weight += response['leg_delta_weight']
            jerk_weight += response['leg_jerk_weight']
        return self.active*(cfg['clearance_weight']*proximity+
            delta_weight*self.delta_cost.clamp(max=16)+jerk_weight*self.jerk_cost.clamp(max=16))
