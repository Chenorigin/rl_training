"""Summarize the recorded physical GUI trajectories without rerunning policy."""
import ast
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
source = ROOT/'source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/gait_refinement.py'
tree = ast.parse(source.read_text())
tuning = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
              and any(isinstance(t,ast.Name) and t.id=='TUNING' for t in n.targets))
result = {'threshold_source':str(source), 'cases':{}}
for label in ('interactive_07','interactive_05','interactive_07_tread030'):
    p=OUT/label
    s=json.loads((p/'summary.json').read_text())
    a=np.load(p/'telemetry.npz')
    h=s['terrain_parameters']['step_height'][0]
    count=int(s['terrain_parameters']['step_count'][0])
    half=s['course_width']/2
    t=a['time']; ext=a['extension']; knee=np.abs(a['knee'])
    level=np.rint(a['wheel_ground']/h).astype(int)
    force=np.linalg.norm(a['contact_force'],axis=2)
    outside=np.abs(a['wheel_pos'][:,:,1]).max(axis=1)>half
    climbing=(a['command'][:,0]>.1)&(level.max(axis=1)>0)&(level.min(axis=1)<count)
    first_collision=min((entry['first_time'] for entry in s['nonwheel_terrain_contacts'].values()),default=float('inf'))
    before_collision=climbing&~outside&(t<first_collision)
    air=before_collision[:,None]&(force[:,:2]<5)
    severe=air&((ext[:,:2]<tuning['front_swing_min_extension'])|
                (knee[:,:2]>tuning['front_swing_max_knee']))
    max_i=int(a['root_pos'][:,0].argmax())
    q=a['root_quat']; w,x,y,z=q.T
    yaw=np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
    completed=(a['wheel_pos'][:,:,0]>=s['goal_x']+tuning['wheel_radius']).all(axis=1)
    finish_i=int(np.flatnonzero(completed)[0]) if completed.any() else None
    case=dict(forward_command_m_s=s.get('forward_command_m_s',float(a['command'][:,0].max())),
        tread_depth_m=s['terrain_parameters']['tread_depth'][0],
        completed=bool(s['cleared_stairs']),fell=bool(s['fell_in_policy']),
        max_wheel_levels=level.max(axis=0).tolist(),final_xyz=a['root_pos'][-1].tolist(),
        base_max_x_m=float(a['root_pos'][max_i,0]),base_y_at_max_x_m=float(a['root_pos'][max_i,1]),
        yaw_at_max_x_deg=float(np.degrees(yaw[max_i])),yaw_final_deg=float(np.degrees(yaw[-1])),
        first_wheel_outside_time_s=float(t[outside][0]) if outside.any() else None,
        first_all_wheels_crossed_time_s=float(t[finish_i]) if finish_i is not None else None,
        max_abs_wheel_y_until_finish_m=float(np.abs(a['wheel_pos'][:finish_i+1,:,1]).max()) if finish_i is not None else None,
        pre_collision_air_front_samples=int(air.sum()),pre_collision_extreme_air_front_samples=int(severe.sum()),
        pre_collision_extreme_air_fraction=float(severe.sum()/air.sum()) if air.any() else None,
        stable_support={},landings=s['landings'],gait=s['gait'],snapshots=s['snapshots'],
        nonwheel_terrain_contacts=s['nonwheel_terrain_contacts'])
    for pair,ids,grace,angle in [('front',[0,1],tuning['front_support_grace'],tuning['front_max_knee']),
                               ('rear',[2,3],tuning['support_grace'],tuning['rear_max_knee'])]:
        valid=climbing[:,None]&(a['contact_time'][:,ids]>=grace)
        case['stable_support'][pair]=dict(samples=int(valid.sum()),
            below_extension_fraction=float(((ext[:,ids]<tuning[f'{pair}_min_extension'])&valid).sum()/valid.sum()) if valid.any() else None,
            over_knee_fraction=float(((knee[:,ids]>angle)&valid).sum()/valid.sum()) if valid.any() else None)
    result['cases'][label]=case
(OUT/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
