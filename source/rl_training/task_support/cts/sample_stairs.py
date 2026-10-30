"""CTS static training reconstruction of the native movable sample scene.

The XML owns dimensions and placement. Initial collision boxes are shared by
training contacts and rays; passive module motion remains a MuJoCo evaluation
condition. The frozen student tracker uses a conservative 0.9 m corridor.
"""
from functools import lru_cache
import math
import xml.etree.ElementTree as ET

import numpy as np
import torch
import trimesh

from .configuration import ROOT


def validate_sample(cfg):
    if (not isinstance(cfg, dict) or set(cfg) != {'terrain_xml','support_hold_s','phase_bonus'}
            or cfg['terrain_xml'] != 'deploy/deploy_mujoco/terrains/sample_stairs.xml'):
        raise ValueError('CTS sample course requires the canonical sample_stairs XML')
    if any(type(cfg[k]) not in (int,float) or not math.isfinite(cfg[k]) or cfg[k] <= 0 for k in ('support_hold_s','phase_bonus')):
        raise ValueError('Sample support duration/bonus must be finite and positive')
    return cfg


@lru_cache(maxsize=1)
def scene(relative='deploy/deploy_mujoco/terrains/sample_stairs.xml'):
    """Read initial native collision boxes, including hidden supporting frames.

No MuJoCo import or XML mutation. Reject unsupported transforms/geometries
instead of silently dropping collision parts from the training reconstruction.
"""
    root = ET.parse(ROOT / relative).getroot()
    boxes = []
    for body in root.findall('./worldbody/body'):
        for node in body.iter('body'):
            if node is not body and node.find('body') is not None:
                raise ValueError('Unsupported nested sample body')
        def visit(node, parent):
            if any(key in node.attrib for key in ('euler','axisangle','xyaxes','zaxis')):
                raise ValueError('Unsupported sample rotation')
            transform = trimesh.transformations.quaternion_matrix(
                np.fromstring(node.get('quat','1 0 0 0'),sep=' '))
            transform[:3,3] = np.fromstring(node.get('pos','0 0 0'),sep=' ')
            transform = parent @ transform
            for geom in node.findall('geom'):
                if geom.get('contype','1') == '0' and geom.get('conaffinity','1') == '0':
                    continue
                if geom.get('type','box') != 'box' or any(k in geom.attrib for k in ('quat','euler','fromto')):
                    raise ValueError('Unsupported sample collision geometry')
                size = np.fromstring(geom.get('size',''),sep=' ')
                if size.shape != (3,) or (size <= 0).any():
                    raise ValueError('Invalid sample collision box')
                local = np.eye(4)
                local[:3,3] = np.fromstring(geom.get('pos','0 0 0'),sep=' ')
                boxes.append((geom.get('name'),size,transform @ local))
            for child in node.findall('body'):
                visit(child,transform)
        visit(body,np.eye(4))
    treads = []
    for name,half,transform in boxes:
        if '_tread_' in name or name == 'white_deck':
            extent = abs(transform[:3,:3]) @ half
            centre = transform[:3,3]
            # Two up ladders meet at y=0. Their longitudinal profiles agree.
            if abs(centre[1]) <= extent[1]+1e-9:
                treads.append((centre[0]-extent[0],centre[0]+extent[0],centre[2]+extent[2]))
    treads = sorted(set(tuple(round(v,9) for v in row) for row in treads))
    deck = next((half for name,half,_ in boxes if name == 'white_deck'),None)
    if len(treads) != 7 or deck is None:
        raise ValueError('sample_stairs must contain three up/down treads and a deck')
    return boxes,treads,float(2*deck[1])


def support_spans(settings, relative):
    """Actual centreline, retaining both 5 mm gaps down to the ground."""
    _,treads,_ = scene(relative)
    offset = settings.start_length/2-treads[0][0]
    spans, cursor = [],-settings.start_length/2
    for lo,hi,z in treads:
        lo,hi = lo+offset,hi+offset
        if lo > cursor+1e-9:
            spans.append((cursor,lo,0.))
        spans.append((lo,hi,z))
        cursor = hi
    goal = cursor
    spans.append((cursor,settings.tile_length-settings.start_length/2,0.))
    return spans,goal


def sample_mesh(settings, relative):
    boxes,treads,_ = scene(relative)
    shift = settings.start_length/2-treads[0][0]
    parts = []
    for _,half,transform in boxes:
        part = trimesh.creation.box(2*half,transform=transform)
        part.apply_translation((shift,0,0))
        parts.append(part)
    floor = trimesh.creation.box((settings.tile_length,settings.tile_width,.2))
    floor.apply_translation(((settings.tile_length-settings.start_length)/2,0,-.1))
    return trimesh.util.concatenate([floor,*parts])


def training_stairs(difficulty,cfg):
    """CTS physical geometry: configured legacy count and native sample scene."""
    from dataclasses import replace
    from .terrain import geometries
    from task_support.student_v3_safe.terrain_geometry import course_mesh
    from task_support.stair_quiet_v1_width.settings import width_for_difficulty
    g = replace(geometries(cfg.course,cfg.cts_legacy_stairs)[cfg.geometry_index],
                width=width_for_difficulty(difficulty,cfg.settings))
    mesh = (sample_mesh(cfg.settings,cfg.cts_sample_stairs['terrain_xml']) if g.family == 'round_trip'
            else course_mesh(g,cfg.settings))
    origin = np.array((cfg.settings.start_length/2,cfg.settings.tile_width/2,0.))
    mesh.apply_translation(origin)
    return [mesh],origin


class SampleStairsCourse:
    def __init__(self,env,cfg):
        self.env,self.cfg = env,validate_sample(cfg)
        self._legacy_cache = {}
        scene(cfg['terrain_xml'])

    @property
    def mask(self):
        from task_support.student_v3_safe.terrain_geometry import geometries
        from task_support.stair_quiet_v1_width.settings import PROFILES
        index = next(i for i,g in enumerate(geometries(self.env.cfg.student_terrain_course)) if g.family == 'round_trip')
        return (self.env.scene.terrain.terrain_types//len(PROFILES) == index) & ~self.env.cts_practice_mask

    def override_field(self,key,value):
        if key in ('steps_table','span_table','goal_table','goal_height_table','ascent_end_table'):
            if key not in self._legacy_cache:
                from .terrain import geometries
                from task_support.student_v3_safe.terrain_geometry import surfaces
                from task_support.stair_quiet_v1_width.settings import PROFILES
                indices,rows = [],[]
                for i,g in enumerate(geometries(self.env.cfg.student_terrain_course,self.env.cfg.cts_legacy_stairs)):
                    if g.family != 'legacy':
                        continue
                    spans,goal,top = surfaces(g,self.env.cfg.stair_tail)
                    for profile in range(len(PROFILES)):
                        indices.append(i*len(PROFILES)+profile)
                        if key == 'span_table':
                            if len(spans)>value.shape[1]:
                                raise ValueError('CTS legacy profile exceeds tracker capacity')
                            rows.append(spans+[(0.,0.,0.)]*(value.shape[1]-len(spans)))
                        else:
                            rows.append(g.steps if key == 'steps_table' else top if key == 'goal_height_table' else goal)
                self._legacy_cache[key] = (torch.tensor(indices,device=value.device),
                    torch.tensor(rows,device=value.device,dtype=value.dtype))
            indices,replacement = self._legacy_cache[key]
            value[indices] = replacement
        if key == 'width':
            # Scalar-width source tracker: use the narrowest real route width.
            # Meshes retain 1 m stairs, 0.9 m deck and the lateral dark box.
            value[self.mask] = scene(self.cfg['terrain_xml'])[2]
        elif key in ('span_table','goal_table'):
            from task_support.student_v3_safe.terrain_geometry import cases
            columns = torch.tensor([g.family == 'round_trip' for g,_ in cases(self.env.cfg.student_terrain_course)],device=value.device)
            spans,goal = support_spans(self.env.cfg.stair_tail,self.cfg['terrain_xml'])
            if key == 'goal_table':
                value[columns] = goal
            else:
                if len(spans) > value.shape[1]:
                    raise ValueError('Sample profile exceeds frozen tracker capacity')
                value[columns] = 0
                value[columns,:len(spans)] = torch.tensor(spans,device=value.device,dtype=value.dtype)

    def reset(self,ids):
        """Exact narrow sample from the first reset, with all contact domains.

        Width-row 9 represents the 1 m stair components; the tracker narrows its
        safe corridor to the 0.9 m deck. Sample episodes never count as evidence
        from the wider material-practice rows, or promote width difficulty.
        """
        env = self.env
        ids = ids[self.mask[ids]]
        if not len(ids):
            return
        from task_support.stair_quiet_v1_width.settings import PROFILES
        state,terrain,s = env._tail_course,env.scene.terrain,env.cfg.stair_tail
        profile = (state['assign_count'][ids]+ids) % len(PROFILES)
        types = terrain.terrain_types[ids]//len(PROFILES)*len(PROFILES)+profile
        row = len(s.widths_m)-1
        state['width_curriculum'].level[ids] = row
        state['width_curriculum'].success_streak[ids] = 0
        terrain.terrain_types[ids] = types
        terrain.terrain_levels[ids] = row
        origins = terrain.terrain_origins[row,types]
        terrain.env_origins[ids] = origins
        env.scene.env_origins[ids] = origins
        robot = env.scene['robot']
        position = robot.data.default_root_state[ids,:3].clone()+origins
        position[:,0] += torch.empty(len(ids),device=env.device).uniform_(-.05,.05)
        position[:,1] += torch.empty(len(ids),device=env.device).uniform_(-s.reset_y_range,s.reset_y_range)
        yaw = torch.empty(len(ids),device=env.device).uniform_(-s.reset_yaw_range,s.reset_yaw_range)
        quat = torch.zeros(len(ids),4,device=env.device)
        quat[:,0],quat[:,3] = torch.cos(yaw/2),torch.sin(yaw/2)
        robot.write_root_pose_to_sim(torch.cat((position,quat),-1),env_ids=ids)
        robot.write_root_velocity_to_sim(torch.zeros(len(ids),6,device=env.device),env_ids=ids)

    def advance_materials(self):
        """Original wide-course gate, excluding the permanently narrow sample.

        Sample friction/compliance is independently cycled by reset().
        """
        env,state = self.env,self.env._tail_course
        if not env.cfg.cts_sample_automatic_stages or env.tail_stage >= 2:
            return
        active = (state['stage_table'] <= env.tail_stage) & (state['family_table'] != 1)
        s = env.cfg.stair_tail
        ready = ((state['stage_episodes'][active] >= s.stage_min_episodes)
            & (state['stage_successes'][active]/state['stage_episodes'][active].clamp_min(1) >= s.stage_success_fraction)).all()
        if bool(ready):
            env.tail_stage += 1
            state['stage_episodes'].zero_()
            state['stage_successes'].zero_()

    def diagnostics(self):
        mask = self.mask
        state = self.env._tail_course
        from task_support.student_v3_safe.terrain_geometry import cases
        columns = torch.tensor([g.family == 'round_trip' for g,_ in cases(self.env.cfg.student_terrain_course)],device=mask.device)
        return dict(active=int(mask.sum()),teacher_active=int((mask & self.env.cts_motion.teacher).sum()),
            student_active=int((mask & ~self.env.cts_motion.teacher).sum()),
            episodes=int(state['episodes'][columns].sum()),successes=int(state['successes'][columns].sum()),
            conservative_corridor_width_m=scene(self.cfg['terrain_xml'])[2])


from task_support.student_v3_safe.terrain_signals import StudentTerrainTracker


class SampleTerrainTracker(StudentTerrainTracker):
    """Require wheel support: up .15 → up .30 → deck → down .30 → .15 → ground.

    Each wheel advances independently. Mere x crossing, airborne motion and
    horizontal riser loads cannot earn the once-only phase bonuses or success.
    """
    def __init__(self,n,device,settings,course):
        super().__init__(n,device,settings)
        self.sample_course = course
        self.sample_phase = torch.zeros(n,4,dtype=torch.long,device=device)
        self.sample_hold = torch.zeros(n,4,device=device)
        self.cost['sample_route'] = torch.zeros(n,device=device)

    def update(self,**d):
        super().update(**d)
        mask = self.sample_course.mask
        _,treads,_ = scene(self.sample_course.cfg['terrain_xml'])
        shift = self.s.start_length/2-treads[0][0]
        # Top of the descending stair is a separate .45 surface, not the deck.
        deck = next(row for row in treads if abs(row[1]-row[0]-.55)<1e-8)
        low = [row for row in treads if row[0] > deck[1] and row[2] < deck[2]-1e-8]
        intervals = [*treads[:2],deck,*low,(treads[-1][1],self.s.tile_length-self.s.start_length,0.)]
        table = torch.tensor(intervals,device=mask.device,dtype=d['position'].dtype)
        target = table[self.sample_phase.clamp_max(len(intervals)-1)]
        x = d['position'][...,0]-shift
        bottom = d['position'][...,2]-self.s.wheel_radius-self.s.wheel_half_length*d['axis'][...,2].abs()
        forces = d['forces']
        loaded = (forces[...,2]>self.s.support_force_N) & (forces[...,2]>torch.linalg.vector_norm(forces[...,:2],dim=-1))
        valid = (mask[:,None] & (self.sample_phase<len(intervals)) & loaded
            & (x>=target[...,0]) & (x<=target[...,1]) & ((bottom-target[...,2]).abs()<.05)
            & (d['position'][...,1].abs()+self.s.wheel_radius+self.s.wheel_half_length*d['axis'][...,1].abs() <= scene(self.sample_course.cfg['terrain_xml'])[2]/2)
            & (d['upright'][:,None]>.9))
        self.sample_hold = torch.where(valid,self.sample_hold+d['dt'],0.)
        advance = valid & (self.sample_hold>=self.sample_course.cfg['support_hold_s'])
        self.sample_phase += advance.long()
        self.sample_hold[advance] = 0.
        safe = ~self.edge_failed & ~self.recovery_failed
        self.cost['sample_route'] += advance.float().mean(-1)*safe
        complete = (self.sample_phase == len(intervals)).all(-1)
        self.success &= ~mask | complete
        self.success_age[mask & ~complete] = 0.
        self.last['crossed'] &= ~mask | complete


def sample_route_reward(env):
    tracker = env.tail_tracker
    return tracker.cost['sample_route']*env.cfg.cts_sample_stairs['phase_bonus']/env.step_dt
