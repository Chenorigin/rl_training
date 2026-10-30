"""Finite static course surfaces, shared by collision meshes and reward geometry."""
from dataclasses import dataclass
import math
import torch
import trimesh
from task_support.stair_quiet_v1_width.settings import GEOMETRIES, PROFILES
from task_support.stair_quiet_v1_width.geometry import foothold_regions as legacy_regions, wheel_margins, stair_mesh


@dataclass(frozen=True)
class CourseGeometry:
    name: str
    height: float
    depth: float
    steps: int
    family: str
    width: float = 1.
    down_depth: float = 0.
    platform_length: float = 0.
    closed_riser: bool = False


def validate_course(cfg):
    if set(cfg) != {'tile_length','episode_length_s','family_cycle','round_trip','open_ascent'}:
        raise ValueError('Invalid terrain course settings')
    if not isinstance(cfg['family_cycle'],list) or set(cfg['family_cycle']) != {'legacy','round_trip','open_ascent'}:
        raise ValueError('Terrain family cycle must cover all three families')
    for name, keys in (('round_trip',{'height','up_depth','down_depth','steps','platform_length'}),
                       ('open_ascent',{'height','depth','steps'})):
        v=cfg[name]
        if set(v)!=keys or any(isinstance(x,bool) or not math.isfinite(x) or x<=0 for x in v.values()):
            raise ValueError('Invalid stair geometry')
        if type(v['steps']) is not int:
            raise ValueError('Step count must be integer')
    if isinstance(cfg['tile_length'],bool) or not math.isfinite(cfg['tile_length']) or cfg['tile_length'] <= 0:
        raise ValueError('Invalid course length')
    if isinstance(cfg['episode_length_s'],bool) or not math.isfinite(cfg['episode_length_s']) or cfg['episode_length_s'] <= 0:
        raise ValueError('Invalid course duration')


def geometries(cfg):
    validate_course(cfg)
    legacy=[CourseGeometry(g.name,g.height,g.depth,g.steps,'legacy',g.width,closed_riser=g.closed_riser) for g in GEOMETRIES]
    r,o=cfg['round_trip'],cfg['open_ascent']
    return tuple(legacy+[CourseGeometry('round_trip',r['height'],r['up_depth'],r['steps'],'round_trip',
        down_depth=r['down_depth'],platform_length=r['platform_length']),
        CourseGeometry('open_ascent',o['height'],o['depth'],o['steps'],'open_ascent')])


def cases(cfg):
    return tuple((g,p) for g in geometries(cfg) for p in PROFILES)


def surfaces(g,s):
    start=s.start_length/2
    spans=[(-start,start,0.)]
    spans += [(start+i*g.depth,start+(i+1)*g.depth,(i+1)*g.height) for i in range(g.steps)]
    goal=start+g.steps*g.depth
    goal_height=g.steps*g.height
    if g.family=='round_trip':
        spans.append((goal,goal+g.platform_length,goal_height))
        descent=goal+g.platform_length
        spans += [(descent+i*g.down_depth,descent+(i+1)*g.down_depth,
                   (g.steps-1-i)*g.height) for i in range(g.steps)]
        goal=descent+g.steps*g.down_depth
        goal_height=0.
    if goal+s.body_half_length+.5 >= s.tile_length-start:
        raise ValueError('Staircase leaves no safe finishing landing')
    spans.append((goal,s.tile_length-start,goal_height))
    merged=[]
    for lo,hi,z in spans:
        if merged and math.isclose(merged[-1][2],z,abs_tol=1.e-9):
            merged[-1]=(merged[-1][0],hi,z)
        else: merged.append((lo,hi,z))
    return merged,goal,goal_height


def horizontal_rectangle(lo,hi,width,z):
    return trimesh.Trimesh(vertices=[[lo,-width/2,z],[hi,-width/2,z],[hi,width/2,z],[lo,width/2,z]],
                           faces=[[0,1,2],[0,2,3]],process=False)


def course_mesh(g,s):
    if g.family == 'legacy':
        return stair_mesh(g, s)
    spans,_,_=surfaces(g,s)
    parts=[]
    for lo,hi,z in spans:
        if g.family=='open_ascent':
            part=horizontal_rectangle(lo,hi,g.width,z)
        else:
            thickness=z+s.floor_depth if g.closed_riser or g.family=='round_trip' else s.board_thickness
            part=trimesh.creation.box((hi-lo,g.width,thickness))
            part.apply_translation(((lo+hi)/2,0,z-thickness/2))
        parts.append(part)
    # Even the open course's catch floor has no vertical sides.
    parts.append(horizontal_rectangle(-s.start_length/2,s.tile_length-s.start_length/2,s.tile_width,-s.floor_depth))
    return trimesh.util.concatenate(parts)


def regions(position,axis,d,s):
    """Use actual finite surfaces for the up/plateau/down case, old parity elsewhere."""
    old=legacy_regions(position,axis,d['height'],d['depth'],d['width'],d['steps'],s)
    spans=d['spans']
    x=position[...,0,None];bottom=(position[...,2]-s.wheel_radius-s.wheel_half_length*axis[...,2].abs())[...,None]
    lo,hi,top=spans[:,None,:,0],spans[:,None,:,1],spans[:,None,:,2]
    valid=hi>lo
    inside=valid & (x>=lo) & (x<=hi)
    index=torch.where(inside,top, -torch.inf).argmax(-1)
    reach=s.wheel_radius+s.wheel_half_length*axis[...,0,None].abs()
    error=(bottom-top).abs()
    plausible=valid & (error<.05) & (x>=lo-reach) & (x<=hi+reach)
    candidate=torch.where(plausible,error,torch.inf).argmin(-1)
    index=torch.where(plausible.any(-1),candidate,index)
    selected=torch.gather(spans[:,None].expand(-1,4,-1,-1),2,index[...,None,None].expand(-1,-1,1,3)).squeeze(2)
    tread,lateral=wheel_margins(s,position)
    forward=torch.minimum(position[...,0]-selected[...,0],selected[...,1]-position[...,0])-s.contact_patch_radius-s.wheel_half_length*axis[...,0].abs()-tread
    side=d['width'][:,None]/2-position[...,1].abs()-s.wheel_radius-s.wheel_half_length*axis[...,1].abs()-lateral
    mask=d['round_trip'][:,None]
    return tuple(torch.where(mask,a,b) for a,b in zip((forward,side,selected[...,2]),old))
