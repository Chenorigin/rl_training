#!/usr/bin/env python3
"""Freeze and execute CPU-only descent noise ablations on native v0/v1 pipelines."""
import argparse
from collections import deque
from copy import deepcopy
from dataclasses import asdict
import itertools
import json
import math
from pathlib import Path
import resource
import shutil
import sys
import time
from types import SimpleNamespace
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'deploy/deploy_mujoco'))
sys.path.insert(0, str(ROOT/'source/rl_training'))
from m20_student_v1.model_io import file_hash


def controls(provenance):
    """Single-factor groups; compensate v0's level-dependent timing explicitly."""
    from student_v0_adapter import profile_config
    base, base_extra = profile_config('student-v1', provenance)
    target, target_extra = profile_config('student-v0', provenance)
    if base.level <= 0:
        raise ValueError('Baseline level must be positive to isolate v0 timing')
    baseline = {'perception': asdict(base), 'extras': asdict(base_extra)}
    nominal = {'perception': asdict(target), 'extras': asdict(target_extra)}
    teacher, teacher_extra = profile_config('teacher', provenance)
    result = {'teacher': {'perception': asdict(teacher), 'extras': asdict(teacher_extra)},
              'student-v1': baseline, 'student-v0': nominal}
    action = deepcopy(baseline)
    action['extras']['action_delay_steps'] = target_extra.action_delay_steps
    result['only_action_delay'] = action
    visibility = deepcopy(baseline)
    for key in ('lidar_visibility', 'lidar_horizontal_fov_deg', 'sensor_offset_body_m', 'rear_sensor_offset_body_m'):
        visibility['extras'][key] = nominal['extras'][key]
    result['only_visibility'] = visibility
    proprio = deepcopy(baseline)
    for key in ('gyro_std_rad_s', 'gravity_tilt_std_rad', 'joint_position_std_rad', 'joint_velocity_std_rad_s'):
        proprio['extras'][key] = nominal['extras'][key]
    result['only_proprio'] = proprio
    timing = deepcopy(baseline)
    for key in ('map_update_period_s', 'max_delay_s'):
        timing['perception'][key] = nominal['perception'][key]*target.level/base.level
    result['only_map_timing'] = timing
    map_error = deepcopy(baseline)
    map_error['perception'] = asdict(target)
    # Raising v0's level alone changes several noise rates AND sensor timing.
    # Keep the baseline's effective discrete timing for this error/quality case.
    for key in ('map_update_period_s', 'max_delay_s'):
        steps = max(1 if key == 'map_update_period_s' else 0, math.ceil(getattr(base, key)*base.level/.02))
        map_error['perception'][key] = steps*.02/target.level
    result['only_map_error_quality'] = map_error
    return result


def scene_definitions(descent_s, mixed_s):
    from terrain_loader import read_config
    descent = ROOT/'deploy/deploy_mujoco/terrains/stairs_descent.xml'
    _, p = read_config(descent)
    start, end = p['start_x'][0], p['start_x'][0]+p['tread_depth'][0]*p['step_count'][0]
    mixed = ROOT/'deploy/deploy_mujoco/terrains/real_world_final.xml'
    geoms = {g.get('name'): g for g in ET.parse(mixed).getroot().findall('./worldbody/geom')}
    down = [g for name, g in geoms.items() if name.startswith('stair_down_')]
    bounds = [(float(g.get('pos').split()[0])-float(g.get('size').split()[0]),
               float(g.get('pos').split()[0])+float(g.get('size').split()[0])) for g in down]
    landing, exit_geom = geoms['stair_landing'], geoms['stair_exit']
    center = [float(v) for v in landing.get('pos').split()]
    sizes = [float(v) for v in landing.get('size').split()]
    exit_pos, exit_size = float(exit_geom.get('pos').split()[0]), float(exit_geom.get('size').split()[0])
    return [{'name': 'descent', 'terrain_xml': str(descent), 'spawn_xy': None,
             'stairs_x': [start, end], 'finish_x': end+1., 'duration_s': descent_s},
            {'name': 'real_world_final_descent', 'terrain_xml': str(mixed),
             'spawn_xy': [center[0]-sizes[0]/2, center[1]],
             'stairs_x': [min(a for a,b in bounds), max(b for a,b in bounds)],
             'finish_x': exit_pos+exit_size, 'duration_s': mixed_s}]


def referenced_assets(xml, visited=None):
    """Freeze MJCF includes and referenced mesh/texture/heightfield files."""
    visited = set() if visited is None else visited
    xml = Path(xml).resolve()
    if xml in visited:
        return set()
    visited.add(xml)
    root = ET.parse(xml).getroot()
    settings = {}
    for compiler in root.findall('compiler'):
        settings.update(compiler.attrib)
    result = {xml}
    for node in root.iter():
        filename = node.get('file')
        if not filename:
            continue
        if node.tag == 'include':
            result.update(referenced_assets(xml.parent/filename, visited))
        elif node.tag in ('mesh', 'texture', 'hfield'):
            key = {'mesh': 'meshdir', 'texture': 'texturedir', 'hfield': 'assetdir'}[node.tag]
            directory = settings.get(key, settings.get('assetdir', ''))
            result.add((xml.parent/directory/filename).resolve())
    return result


def freeze(args):
    import psutil
    import torch
    from student_v0_adapter import load_student as load_v0
    actor, meta, identity = load_v0(args.v0)
    del actor
    plan = {'version': 1, 'v0': str(args.v0.resolve()), 'v1': str(args.v1.resolve()) if args.v1 else None,
            'model': str(args.model.resolve()), 'speed_m_s': args.speed, 'warmup_s': 1.,
            'physics_dt_s': .005, 'policy_dt_s': .02, 'seed_list': args.seeds,
            'scenes': scene_definitions(args.descent_duration_s, args.mixed_duration_s),
            'v0_profiles': controls(meta), 'track_course': False,
            'rules': {'collision': 'actual solver limb contacts and signed collision-geometry gaps reported separately; negative gap is penetration, zero is no positive clearance and may not resolve penetration depth',
                      'contact': 'front/rear nonadjacent limb contact with contact.dist <= 0 and active constraint at every physics solve',
                      'zone': 'wheel X footprint intersects native descending stair geometry X interval',
                      'gap_sampling': 'post-integration 5 ms physics boundaries, collision geoms only',
                      'completion': 'all wheels beyond local finish X, no fall and no course exit',
                      'interpretation': 'report all seeds; lower contact count is not a success if passage fails; multi-factor target is not a causal single-factor claim'},
            'resource_budget': {'memory_allowance_bytes': 2*1024**3, 'disk_allowance_bytes': 512*1024**2,
                                'disk_reserve_bytes': 2*1024**3, 'threads': 1, 'gpu_allocation': 0,
                                'memory_available_bytes': psutil.virtual_memory().available,
                                'disk_free_bytes': shutil.disk_usage(args.output_dir.parent).free,
                                'basis': 'existing v0 CPU sim RSS 1.2 GiB + geometry/metadata allowance; one case at a time, bounded 30 s/12 s trajectories'}}
    sources = [Path(__file__), ROOT/'deploy/deploy_mujoco/deploy_mujoco.py', ROOT/'deploy/deploy_mujoco/terrain_loader.py', ROOT/'deploy/deploy_mujoco/student_v0_adapter.py',
               ROOT/'deploy/deploy_mujoco/student_v0_profiles.json', ROOT/'deploy/deploy_mujoco/student_v1_adapter.py',
               *sorted((ROOT/'source/rl_training/m20_student_v0').glob('*.py')),
               *sorted((ROOT/'source/rl_training/m20_student_v1').glob('*.py'))]
    inputs = [args.v0, args.model, *(Path(s['terrain_xml']) for s in plan['scenes']), *sources]
    for xml in (args.model, *(Path(s['terrain_xml']) for s in plan['scenes'])):
        inputs.extend(sorted(referenced_assets(xml)))
    if args.v1: inputs.append(args.v1)
    plan['input_sha256'] = {str(p.resolve()): file_hash(p) for p in inputs}
    plan['cases'] = [{'model': 'v0', 'profile': profile, 'seed': seed, 'scene': scene['name']}
                     for scene in plan['scenes'] for seed in args.seeds for profile in plan['v0_profiles']]
    if args.v1:
        plan['cases'] += [{'model': 'v1', 'profile': profile, 'seed': seed, 'scene': scene['name']}
                          for scene in plan['scenes'] for seed in args.seeds for profile in ('student-v1', 'student-v0')]
    if plan['resource_budget']['disk_free_bytes'] < sum(plan['resource_budget'][k] for k in ('disk_allowance_bytes','disk_reserve_bytes')):
        raise RuntimeError('Output disk does not meet frozen reserve')
    if psutil.virtual_memory().total-psutil.virtual_memory().available+2*1024**3 > psutil.virtual_memory().total*.85:
        raise RuntimeError('Whole-host memory reserve insufficient')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    path=args.output_dir/'plan.json'; path.write_text(json.dumps(plan,indent=2)+'\n'); path.chmod(0o444)
    return plan


class LimbDistance:
    """Signed primitive-geometry distances; enclosing spheres safely prune far pairs."""
    def __init__(self, model):
        import mujoco
        import numpy as np
        self.model=model; self.last_pair={}
        limbs={prefix: [i for i in range(model.ngeom) if (model.geom_contype[i] or model.geom_conaffinity[i])
                        and (mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[i])) or '').startswith(prefix)]
               for prefix in ('fl_', 'fr_', 'hl_', 'hr_')}
        if not all(limbs.values()): raise ValueError('Missing active limb collision geometry')
        self.groups = {name: np.asarray(list(pairs),dtype=np.int32) for name,pairs in (
            ('left', itertools.product(limbs['fl_'],limbs['hl_'])),
            ('right', itertools.product(limbs['fr_'],limbs['hr_'])),
            ('cross', itertools.chain(itertools.product(limbs['fl_'],limbs['hr_']),itertools.product(limbs['fr_'],limbs['hl_']))))}
        self.pair_side = {frozenset((int(a),int(b))): name for name,pairs in self.groups.items() for a,b in pairs}
        self.names={prefix: [int(i) for i in ids] for prefix,ids in limbs.items()}

    def gaps(self, data):
        import mujoco
        import numpy as np
        result={}
        for name,pairs in self.groups.items():
            a,b=pairs[:,0],pairs[:,1]
            lower=np.linalg.norm(data.geom_xpos[a]-data.geom_xpos[b],axis=1)-self.model.geom_rbound[a]-self.model.geom_rbound[b]
            nearest=self.last_pair.get(name,int(np.argmin(lower)))
            ga,gb=pairs[nearest]; best=mujoco.mj_geomDistance(self.model,data,int(ga),int(gb),5.,None)
            for ix in np.argsort(lower):
                if lower[ix]>=best: break
                ga,gb=pairs[ix]; gap=mujoco.mj_geomDistance(self.model,data,int(ga),int(gb),5.,None)
                if gap<best: best,nearest=gap,int(ix)
            self.last_pair[name]=nearest; result[name]=float(best)
        return result

    def contacts(self, data):
        import mujoco
        import numpy as np
        result={name: False for name in self.groups}; peak=0.
        for index, contact in enumerate(data.contact):
            side=self.pair_side.get(frozenset((int(contact.geom1),int(contact.geom2))))
            if side and contact.dist<=0 and contact.efc_address>=0:
                result[side]=True
                force=np.zeros(6); mujoco.mj_contactForce(self.model,data,index,force)
                peak=max(peak,float(np.linalg.norm(force[:3])))
        return result,peak


def run_case(plan,case,output,actors,metas):
    import mujoco
    import numpy as np
    import torch
    import deploy_mujoco as base
    from student_v0_adapter import StudentMapAdapter as Adapter0, SensorExtras, StudentV0PerceptionCfg
    from student_v1_adapter import StudentMapAdapter as Adapter1, profile_config as config1
    scene=next(s for s in plan['scenes'] if s['name']==case['scene'])
    args=SimpleNamespace(terrain=None,terrain_xml=Path(scene['terrain_xml']))
    torch.manual_seed(case['seed'])
    model=base.load_model(Path(plan['model']),args)
    if scene['spawn_xy'] is not None: args.terrain_config.spawn_xy=tuple(scene['spawn_xy'])
    contract=base.M20Contract(model,args.terrain_config); data=mujoco.MjData(model); contract.reset(data)
    if case['model']=='v0':
        values=plan['v0_profiles'][case['profile']]
        cfg=StudentV0PerceptionCfg(**values['perception']); cfg.validate(); extras=SensorExtras(**values['extras']); extras.validate()
        adapter=Adapter0(contract,cfg,extras,case['seed'])
    else:
        cfg,extras=config1(case['profile'],metas['v1']); adapter=Adapter1(contract,cfg,extras,case['seed'])
    distance=LimbDistance(model)
    # Distances on normal kinematics must match an unpruned all-pairs reference.
    exact={name:min(mujoco.mj_geomDistance(model,data,int(a),int(b),5.,None) for a,b in pairs) for name,pairs in distance.groups.items()}
    measured=distance.gaps(data)
    if any(abs(exact[k]-measured[k])>1e-9 for k in exact): raise RuntimeError('Distance pruning calibration failed')
    probe=mujoco.MjData(model); probe.qpos[:]=data.qpos; mujoco.mj_forward(model,probe)
    a=next(i for i in distance.names['fl_'] if model.geom_type[i]==mujoco.mjtGeom.mjGEOM_BOX)
    b=next(i for i in distance.names['hl_'] if model.geom_type[i]==mujoco.mjtGeom.mjGEOM_BOX)
    probe.geom_xpos[b]=probe.geom_xpos[a]; probe.geom_xmat[b]=probe.geom_xmat[a]
    if mujoco.mj_geomDistance(model,probe,a,b,5.,None)>0: raise RuntimeError('Overlap probe was not detected')
    old=np.zeros(16,np.float32); queue=deque([old.copy() for _ in range(extras.action_delay_steps)])
    rows={k:[] for k in ('time_s','qpos','observation','command','velocity_body','base_position','zone','gap_m','contact','force_peak_N')}
    minima={k:float('inf') for k in distance.groups}; all_minima=deepcopy(minima)
    contacts={k:0 for k in distance.groups}; events=deepcopy(contacts); previous_contacts={k:False for k in contacts}
    zone_samples=0; fell=outside=False; peak=0.; start=time.monotonic()
    for step in range(math.ceil(scene['duration_s']/plan['policy_dt_s'])):
        command=np.array([plan['speed_m_s'] if data.time>=plan['warmup_s'] else 0.,0.,0.],np.float32)
        obs=adapter.observe(data,command,old,step)
        with torch.inference_mode(): action=actors[case['model']](torch.from_numpy(obs)[None])[0].numpy()
        if not np.isfinite(action).all(): raise RuntimeError('Nonfinite action')
        issued=np.clip(action,-100,100).astype(np.float32); queue.append(issued.copy()); applied=queue.popleft()
        velocity=np.zeros(6); mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,contract.base_id,velocity,1)
        pre_time=float(data.time); pre_pos=data.xpos[contract.base_id].copy(); pre_q=data.qpos.copy()
        step_gap={k:float('inf') for k in minima}; step_contact={k:False for k in minima}; step_peak=0.; step_zone=False
        for _ in range(round(plan['policy_dt_s']/model.opt.timestep)):
            contract.apply_pd(data,applied); mujoco.mj_step(model,data)
            # Solver forces/contacts correspond to pre-integration kinematics.
            pre_wheels=data.xpos[contract.wheel_body_ids,0]
            contact_zone=pre_wheels.max()>=scene['stairs_x'][0] and pre_wheels.min()<=scene['stairs_x'][1]
            hits,force=distance.contacts(data)
            if contact_zone:
                peak=max(peak,force); step_peak=max(step_peak,force)
                for k,hit in hits.items():
                    contacts[k]+=int(hit); events[k]+=int(hit and not previous_contacts[k]); step_contact[k]|=hit
            previous_contacts={k:bool(hit and contact_zone) for k,hit in hits.items()}
            mujoco.mj_forward(model,data)
            if not np.isfinite(data.qpos).all(): raise RuntimeError('Nonfinite physics state')
            wheels=data.xpos[contract.wheel_body_ids,0]
            zone=wheels.max()>=scene['stairs_x'][0] and wheels.min()<=scene['stairs_x'][1]
            gap=distance.gaps(data)
            for k,v in gap.items():
                all_minima[k]=min(all_minima[k],v)
                if zone: minima[k]=min(minima[k],v); step_gap[k]=min(step_gap[k],v)
            zone_samples+=int(zone); step_zone|=zone
        old=issued
        pos=data.xpos[contract.base_id]
        fell=bool(pos[2]-contract.ground_height(data,*pos[:2])<.22 or data.xmat[contract.base_id].reshape(3,3)[2,2]<.35)
        outside|=bool(np.abs(data.xpos[contract.wheel_body_ids,1]).max()>args.terrain_config.course_width/2)
        values=(pre_time,pre_q,obs,command,np.array([*velocity[3:5],velocity[2]]),pre_pos,step_zone,
                [step_gap[k] for k in minima],[step_contact[k] for k in minima],step_peak)
        for k,v in zip(rows,values): rows[k].append(v)
        if fell or (data.xpos[contract.wheel_body_ids,0]>scene['finish_x']).all(): break
    completed=bool((data.xpos[contract.wheel_body_ids,0]>scene['finish_x']).all() and not fell and not outside)
    moving=np.asarray(rows['time_s'])>=plan['warmup_s']; actual=np.asarray(rows['velocity_body'])
    mean_speed=float(actual[moving,0].mean()) if moving.any() else None
    safe_min={k:(v if math.isfinite(v) else None) for k,v in minima.items()}
    summary={**case,'completed':completed,'fell':fell,'outside_course':outside,'duration_s':float(data.time),'steps':len(rows['time_s']),
             'descent_physics_samples':zone_samples,'minimum_gap_m':safe_min,'all_episode_minimum_gap_m':all_minima,
             'contact_physics_samples':contacts,'contact_events':events,'interleg_force_peak_N':peak,
             'actual_forward_speed_m_s':mean_speed,'end_base_position':pos.tolist(),'wall_time_s':time.monotonic()-start,
             'effective_perception':asdict(cfg),'sensor_extras':asdict(extras),'measured_distance_calibration_m':measured,
             'rss_peak_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'weights_updated':False}
    output.mkdir()
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(output/'trajectory.npz',**{k:np.asarray(v) for k,v in rows.items()})
    print(json.dumps({k:summary[k] for k in ('model','profile','seed','scene','completed','fell','minimum_gap_m','contact_physics_samples','actual_forward_speed_m_s','wall_time_s')}),flush=True)
    return summary


def execute(path):
    import torch
    from student_v0_adapter import load_student as load0
    from student_v1_adapter import load_student as load1
    torch.set_num_threads(1)
    plan=json.loads(path.read_text())
    for filename,digest in plan['input_sha256'].items():
        if file_hash(filename)!=digest: raise ValueError(f'Frozen input changed: {filename}')
    actors={}; metas={}
    actors['v0'],metas['v0'],_=load0(Path(plan['v0']))
    if plan['v1']: actors['v1'],metas['v1'],_=load1(Path(plan['v1']))
    results=[]
    for case in plan['cases']:
        if shutil.disk_usage(path.parent).free<plan['resource_budget']['disk_reserve_bytes']: raise RuntimeError('Output reserve reached')
        target=path.parent/('_'.join(str(case[k]) for k in ('model','scene','profile','seed')))
        results.append(run_case(plan,case,target,actors,metas))
        (path.parent/'summary.json').write_text(json.dumps({'cases':results},indent=2,allow_nan=False)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-plan',type=Path)
    parser.add_argument('--v0',type=Path); parser.add_argument('--v1',type=Path); parser.add_argument('--model',type=Path)
    parser.add_argument('--output-dir',type=Path); parser.add_argument('--speed',type=float,default=.7)
    parser.add_argument('--seeds',type=int,nargs='+',default=[42,43,44])
    parser.add_argument('--descent-duration-s',type=float,default=30.)
    parser.add_argument('--mixed-duration-s',type=float,default=12.)
    args=parser.parse_args()
    if args.run_plan:
        if len(sys.argv)!=3: parser.error('--run-plan accepts no overrides')
        return execute(args.run_plan)
    if not all((args.v0,args.model,args.output_dir)): parser.error('Planning requires v0, model, output-dir')
    if not all(math.isfinite(v) and v>0 for v in (args.speed,args.descent_duration_s,args.mixed_duration_s)): parser.error('Speed/durations must be positive and finite')
    plan=freeze(args)
    print(json.dumps({'plan':str(args.output_dir/'plan.json'),'case_count':len(plan['cases']),'resource_budget':plan['resource_budget']},indent=2))


if __name__=='__main__': main()
