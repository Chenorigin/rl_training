"""Stair-side enclosure geometry, separate from traversable support surfaces.

Layouts are balanced within each immutable course/level pool. A private RNG
leaves the original stairs, material draws and all unrelated courses intact.
"""
import math
import numpy as np
import torch

LAYOUTS = ('open','left','right','both')


def enabled(cfg):
    return bool(cfg and cfg.get('enabled'))


def validate(cfg, geometry):
    keys = {'enabled','height_range_m','side_gap_range_m','thickness_m'}
    if not isinstance(cfg,dict) or set(cfg) != keys or type(cfg['enabled']) is not bool:
        raise ValueError('Invalid stair wall settings')
    for key in ('height_range_m','side_gap_range_m'):
        v=cfg[key]
        if (not isinstance(v,list) or len(v)!=2 or
            any(type(x) not in (int,float) or not math.isfinite(x) for x in v) or
            not 0<=v[0]<=v[1]):
            raise ValueError('Invalid stair wall interval: '+key)
    t=cfg['thickness_m']
    if type(t) not in (int,float) or not math.isfinite(t) or t<=0 or cfg['height_range_m'][0]<.7:
        raise ValueError('Stair walls must be solid non-traversable obstacles')
    if geometry['width_range'][1]/2+cfg['side_gap_range_m'][1]+t>=geometry['tile_width']/2:
        raise ValueError('Stair walls must fit inside the physical tile')
    return cfg


def geometry(cfg, terrain, course, level, sample, bounds, treads, record):
    """Return solid wall rectangles (local XY, absolute Z top), never treads."""
    rng=np.random.default_rng(np.random.SeedSequence([terrain['seed'],27191,course,level]))
    layout=int(np.concatenate([rng.permutation(4) for _ in range(math.ceil(terrain['samples_per_level']/4))])[sample])
    rng=np.random.default_rng(np.random.SeedSequence([terrain['seed'],81371,course,level,sample]))
    height=float(rng.uniform(*cfg['height_range_m']))
    gap=float(rng.uniform(*cfg['side_gap_range_m']))
    half=record['width']/2+gap;t=cfg['thickness_m']
    edges=sorted({bounds[0],bounds[1],*[x for r in treads for x in r[:2]]})
    rects=[]
    for x0,x1 in zip(edges[:-1],edges[1:]):
        x=(x0+x1)/2
        floor=max([0.]+[r[4] for r in treads if r[0]<=x<=r[1] and r[2]<=0<=r[3]])
        if layout in (1,3):rects.append((x0,x1,half,half+t,floor+height))
        if layout in (2,3):rects.append((x0,x1,-half-t,-half,floor+height))
    record.update(stair_wall_layout=LAYOUTS[layout],stair_wall_rectangles=rects,
                  stair_wall_height=height,stair_wall_side_gap=gap)
    return rects


class StairWalls:
    """Whole-robot wall clearance; no flat-ground leg/clearance penalties."""
    def __init__(self,state):
        from .wall_practice import collision_document
        self.state=state
        self.cfg=state.cfg['wall_practice']  # shared clearance/fault numerical contract
        self.document=collision_document(self.cfg)
        names=list(dict.fromkeys(g['body'] for g in self.document['primitives']))
        ids,actual=state.env.scene['robot'].find_bodies(names,preserve_order=True)
        if len(ids)!=len(names) or list(actual)!=names:raise ValueError('Stair wall body mapping differs')
        self.body_ids=[dict(zip(names,ids))[g['body']] for g in self.document['primitives']]
        device=state.env.device
        records=state.atlas.records
        self.layout=torch.tensor([LAYOUTS.index(r.get('stair_wall_layout','open')) for r in records],device=device)
        count=max(1,max(len(r.get('stair_wall_rectangles',[])) for r in records))
        self.rectangles=torch.full((len(records),count,5),torch.nan,device=device)
        for i,r in enumerate(records):
            boxes=r.get('stair_wall_rectangles',[])
            if boxes:self.rectangles[i,:len(boxes)]=self.rectangles.new_tensor(boxes)
        self.clearance=torch.full((state.env.num_envs,),torch.inf,device=device)
        self.tick=-1
        # Cohort x stair course x enclosure. Empty cells stay visible in logs.
        self.counts={k:torch.zeros(2,3,4,dtype=torch.long,device=device)
                     for k in ('frames','episodes','successes','wall_failures')}

    def reset(self,ids):
        self.clearance[ids]=torch.inf
        self.tick=-1

    def measure(self):
        from .wall_practice import primitive_aabbs,envelope_clearance
        s=self.state;env=s.env
        if self.tick==env.common_step_counter:return
        self.clearance.fill_(torch.inf)
        ids=(s.stairs & (self.layout[s.tile]>0)).nonzero().flatten()
        if ids.numel():
            d=env.scene['robot'].data
            positions=d.body_link_pos_w[ids][:,self.body_ids]
            orientations=d.body_link_quat_w[ids][:,self.body_ids]
            finite=torch.isfinite(positions).all(dim=(-1,-2)) & torch.isfinite(orientations).all(dim=(-1,-2))
            low,high=primitive_aabbs(positions,orientations,self.document)
            shift=torch.cat((s.origins[s.tile[ids],:2],torch.zeros(len(ids),1,device=env.device)),-1)[:,None]
            self.clearance[ids]=torch.nan_to_num(envelope_clearance(low-shift,high-shift,self.rectangles[s.tile[ids]]),nan=-torch.inf,neginf=-torch.inf,posinf=-torch.inf)
            self.clearance[ids[~finite]]=-torch.inf
        self.tick=env.common_step_counter

    def failure(self):
        self.measure()
        return self.clearance < -self.cfg['penetration_limit']

    def cost(self):
        self.measure()
        return self.cfg['clearance_weight']*((self.cfg['safe_margin']-self.clearance)/self.cfg['safe_margin']).clamp(0,4).square()

    def record(self,ids):
        s=self.state
        ids=ids[s.stairs[ids]]
        flat=((~s.teacher[ids]).long()*3+s.course_ids[ids])*4+self.layout[s.tile[ids]]
        values={
            'episodes':torch.ones_like(ids),'successes':(s.success[ids] & ~s.failed[ids]).long(),
            'wall_failures':self.failure()[ids].long()}
        for key,weight in values.items():self.counts[key].view(-1).index_add_(0,flat,weight)

    def frame(self):
        s=self.state;mask=s.stairs
        if getattr(s,'stair_hold',None) is not None:mask=mask & ~s.stair_hold.active
        ids=mask.nonzero().flatten()
        flat=((~s.teacher[ids]).long()*3+s.course_ids[ids])*4+self.layout[s.tile[ids]]
        self.counts['frames'].view(-1).add_(torch.bincount(flat,minlength=24))

    def diagnostics(self):
        counts={k:v.cpu() for k,v in self.counts.items()}
        out={}
        for t,cohort in enumerate(('teacher','student')):
            for c,course in enumerate(('up','down','round_trip')):
                for k,layout in enumerate(LAYOUTS):
                    for name,v in counts.items():out[f'{cohort}/{course}/{layout}/{name}']=int(v[t,c,k])
        return out

    def snapshot(self):
        return {k:v.cpu().clone() for k,v in self.counts.items()}

    def checked_snapshot(self,saved):
        if not isinstance(saved,dict) or saved.keys()!=self.counts.keys():raise ValueError('Stair wall ledger keys changed')
        for k,v in saved.items():
            target=self.counts[k]
            if not isinstance(v,torch.Tensor) or v.shape!=target.shape or v.dtype!=target.dtype or (v<0).any():
                raise ValueError('Invalid stair wall ledger: '+k)
        if (saved['successes']>saved['episodes']).any() or (saved['wall_failures']>saved['episodes']).any():
            raise ValueError('Stair wall results exceed episodes')
        return {k:v.to(self.state.env.device) for k,v in saved.items()}


def physical_probes(atlas):
    """Every wall segment must be found on the cooked collider, not just the floor."""
    for index,record in enumerate(atlas.records):
        for x0,x1,y0,y1,top in record.get('stair_wall_rectangles',[]):
            yield index, (float(atlas.origins[index,0]+(x0+x1)/2),
                          float(atlas.origins[index,1]+(y0+y1)/2)), top
