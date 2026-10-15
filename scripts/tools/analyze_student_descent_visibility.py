#!/usr/bin/env python3
"""Split the observed visibility effect without changing deployed sensor code.

Uses a scoped visibility-mask override only for the FOV-only diagnostic. All
physics, policy, RNG seeds and metrics use the frozen first-round runner.
"""
from copy import deepcopy
import argparse
import json
from pathlib import Path
import shutil
from unittest.mock import patch

import analyze_student_descent_noise as parent


def fov_only(model, data, base_id, hits, sensor_offset, horizontal_fov_deg=360., rear_sensor_offset=None):
    import numpy as np
    from student_v0_adapter import horizontal_fov_mask
    rotation=data.xmat[base_id].reshape(3,3)
    front=data.xpos[base_id]+rotation@np.asarray(sensor_offset)
    visible=horizontal_fov_mask(hits,front,rotation,horizontal_fov_deg)
    if rear_sensor_offset is not None:
        rear=data.xpos[base_id]+rotation@np.asarray(rear_sensor_offset)
        visible|=horizontal_fov_mask(hits,rear,rotation,horizontal_fov_deg,rear=True)
    return visible


def verify(plan):
    for filename,digest in plan['input_sha256'].items():
        if parent.file_hash(filename)!=digest:raise ValueError(f'Frozen input changed: {filename}')


def freeze(source, output):
    plan=json.loads(source.read_text());verify(plan)
    base=deepcopy(plan['v0_profiles']['student-v1'])
    both=deepcopy(plan['v0_profiles']['only_visibility'])
    geometry=deepcopy(both);geometry['extras']['lidar_horizontal_fov_deg']=360.
    plan['v1']=None
    plan['v0_profiles']={'baseline':base,'both_visibility':both,'fov_only':deepcopy(both),'occlusion_only':geometry}
    plan['cases']=[{'model':'v0','profile':profile,'seed':seed,'scene':scene['name']}
                   for scene in plan['scenes'] for seed in plan['seed_list'] for profile in plan['v0_profiles']]
    plan['visibility_split']={'baseline':'full grid, no geometric occlusion',
                             'both_visibility':'native front/rear 135-degree sectors AND geometric occlusion',
                             'fov_only':'same sectors, scoped deterministic angular mask without geometric ray test',
                             'occlusion_only':'native geometric ray test with front/rear sectors set to 360 degrees'}
    plan['input_sha256'][str(Path(__file__).resolve())]=parent.file_hash(Path(__file__))
    plan['input_sha256'][str(source.resolve())]=parent.file_hash(source)
    output.mkdir(parents=True,exist_ok=False)
    path=output/'plan.json';path.write_text(json.dumps(plan,indent=2)+'\n');path.chmod(0o444)
    print(json.dumps({'plan':str(path),'case_count':len(plan['cases'])}),flush=True)


def execute(path):
    import torch
    import student_v0_adapter as adapter
    torch.set_num_threads(1)
    plan=json.loads(path.read_text());verify(plan)
    actor,meta,_=adapter.load_student(Path(plan['v0']))
    rows=[]
    for case in plan['cases']:
        if shutil.disk_usage(path.parent).free<plan['resource_budget']['disk_reserve_bytes']:
            raise RuntimeError('Output reserve reached')
        target=path.parent/('_'.join(str(case[k]) for k in ('model','scene','profile','seed')))
        if case['profile']=='fov_only':
            with patch.object(adapter,'visible_terrain',side_effect=fov_only):
                result=parent.run_case(plan,case,target,{'v0':actor},{'v0':meta})
        else:
            result=parent.run_case(plan,case,target,{'v0':actor},{'v0':meta})
        result['visibility_mode']=plan['visibility_split'][case['profile']]
        (target/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
        rows.append(result)
        (path.parent/'summary.json').write_text(json.dumps({'cases':rows},indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-plan',type=Path);parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--run-plan',type=Path)
    args=parser.parse_args()
    if args.run_plan:
        if args.source_plan or args.output_dir:parser.error('--run-plan accepts no overrides')
        execute(args.run_plan)
    elif args.source_plan and args.output_dir:freeze(args.source_plan,args.output_dir)
    else:parser.error('Use source-plan/output-dir, or run-plan')
