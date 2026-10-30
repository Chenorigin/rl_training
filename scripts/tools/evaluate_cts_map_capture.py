#!/usr/bin/env python3
"""Bounded CPU MuJoCo comparison of CTS map support, without optimizer updates."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'source/rl_training'))
sys.path.insert(0,str(ROOT/'deploy/deploy_mujoco'))

import mujoco
import numpy as np
import psutil
import torch
import deploy_cts_student as deploy
from task_support.cts.configuration import default_settings


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--terrain-xml',type=Path)
    p.add_argument('--padding',type=float,nargs='+',default=[0.,default_settings()['map_capture']['padding_m']])
    p.add_argument('--seeds',type=int,nargs='+',default=[42,43,44])
    p.add_argument('--steps',type=int,default=400)
    p.add_argument('--hold-steps',type=int,default=100)
    p.add_argument('--command-x',type=float,default=.7)
    p.add_argument('--noise-level',type=float)
    p.add_argument('--wall-seconds',type=int,default=300)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(argv)
    if not 0 <= a.hold_steps < a.steps <= 1024 or not 0 < a.wall_seconds <= 600:
        p.error('Require 0 <= hold < steps <=1024 and wall-seconds in [1,600]')
    if (not 1 <= len(a.padding)*len(a.seeds) <= 12 or len(set(a.padding)) != len(a.padding)
            or len(set(a.seeds)) != len(a.seeds) or any(not 0 <= x <= 2 for x in a.padding)
            or any(x < 0 for x in a.seeds) or not math.isfinite(a.command_x)):
        p.error('Invalid/duplicate conditions, padding or command; maximum 12 cases')
    if a.noise_level is not None and not 0 <= a.noise_level <= 1:
        p.error('Noise level must be in [0,1]')
    host=psutil.virtual_memory()
    reserve=2*1024**3  # Prior exact CTS loader + MuJoCo measured <=1.2GB RSS.
    disk=psutil.disk_usage(a.output.parent if a.output.parent.exists() else ROOT)
    if host.total-host.available+reserve > .85*host.total or disk.free < reserve:
        raise RuntimeError('Insufficient host/disk headroom for bounded CPU map comparison')
    a.output.mkdir(parents=True,exist_ok=False)
    plan={'checkpoint':str(a.checkpoint.resolve()),'checkpoint_sha256':digest(a.checkpoint),
          'model':str(a.model.resolve()),'model_sha256':digest(a.model),'tool_sha256':digest(__file__),
          'terrain_xml':str(a.terrain_xml.resolve()) if a.terrain_xml else None,
          'terrain_sha256':digest(a.terrain_xml) if a.terrain_xml else None,
          'conditions':vars(a)|{'checkpoint':str(a.checkpoint),'model':str(a.model),
                               'terrain_xml':str(a.terrain_xml) if a.terrain_xml else None,'output':str(a.output)},
          'host_available_bytes':host.available,'host_total_bytes':host.total,'load_average':list(__import__('os').getloadavg()),
          'cpu_increment_reserve_bytes':reserve,'disk_free_bytes':disk.free,'optimizer_updates':0,
          'resource_basis':'Previous exact loader/MuJoCo RSS about1.2GB; reserve2GiB, one numerical thread, CPU only',
          'source_sha256':{name:digest(ROOT/name) for name in (
              'source/rl_training/task_support/student_v1/perception.py',
              'source/rl_training/task_support/student_v1/configuration.py',
              'deploy/deploy_mujoco/student_v1_adapter.py','deploy/deploy_mujoco/deploy_mujoco.py',
              'deploy/deploy_mujoco/deploy_cts_student.py')},
          'semantics':'Same checkpoint and saved landing controller. Capture padding only; larger source changes per-point noise RNG consumption. No exact full-noise pairing claim.',
          'criteria':'Report all cases: current-input rows, actual collider clearance, forward progress and physical failures. Coverage improvement alone is not behavioral acceptance.'}
    (a.output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');(a.output/'plan.json').chmod(0o444)
    torch.set_num_threads(1)
    actor,control=deploy.load_policy(a.checkpoint)
    started=time.monotonic();cases=[];traces={}
    for seed in a.seeds:
        for padding in a.padding:
            torch.manual_seed(seed);np.random.seed(seed)
            args=SimpleNamespace(checkpoint=a.checkpoint.resolve(),terrain=None if a.terrain_xml else 'flat',
                terrain_xml=a.terrain_xml,stair_height=None,tread_depth=None,stair_count=None,stair_start=None,
                noise_profile='student-v1',profile_file=None,noise_level=a.noise_level,seed=seed,
                map_capture_padding=padding)
            model=deploy.quiet.baseline.load_model(a.model,args)
            contract=deploy.quiet.baseline.M20Contract(model,args.terrain_config)
            data=mujoco.MjData(model);contract.reset(data)
            sensor=deploy.StudentPerception(contract,args,actor.student_provenance)
            controller=deploy.quiet.QuietExecution(model,contract,control,actor)
            state=sensor.map_adapter.map
            previous=np.zeros(16,dtype=np.float32);rows=[];fresh=[];clearance=[];root=[];ages=[];fell=False
            for step in range(a.steps):
                if time.monotonic()-started > a.wall_seconds:raise TimeoutError('Bounded map comparison wall limit')
                command=np.array([a.command_x if step >= a.hold_steps else 0.,0.,0.],dtype=np.float32)
                obs=sensor.observe(data,command,previous,step)
                valid=sensor.map_adapter.last_channels[1].reshape(11,17)>.5
                rows.append(int(valid[:,12:].any(0).sum()))
                fresh.append(int(sensor.map_adapter.last_visibility.reshape(11,17)[:,12:].any(0).sum()))
                capture=int(state.capture[0]);ages.append((step-capture)*.02 if capture>=0 else None)
                controller.observe(data,sensor.map_adapter)
                with torch.inference_mode():action=actor(controller.policy_inputs(obs,command,actor,data,sensor))[0].numpy().clip(-100.,100.)
                applied=sensor.action(action)
                for _ in range(4):
                    mujoco.mj_step1(model,data)
                    data.ctrl[contract.actuator_ids]=controller.apply(data,applied,command)
                    mujoco.mj_step2(model,data);controller.after_step(data)
                mujoco.mj_fwdPosition(model,data);mujoco.mj_fwdVelocity(model,data)
                values=[]
                for geom in controller.adapter.geoms:
                    axis=data.geom_xmat[geom].reshape(3,3)[:,2];radius,half=model.geom_size[geom,:2]
                    centre=data.geom_xpos[geom]
                    bottom=centre[2]-radius*math.sqrt(max(0.,1-axis[2]**2))-half*abs(axis[2])
                    values.append(bottom-contract.ground_height(data,float(centre[0]),float(centre[1])))
                clearance.append(values);root.append(data.xpos[contract.base_id].copy());previous=action.astype(np.float32)
                pos=root[-1];ground=contract.ground_height(data,float(pos[0]),float(pos[1]))
                fell=not np.isfinite(data.qpos).all() or pos[2]-ground<.22 or data.xmat[contract.base_id].reshape(3,3)[2,2]<.35
                if fell:break
            row=np.asarray(rows);z=np.asarray(clearance);position=np.asarray(root);moving=np.arange(len(rows))>=a.hold_steps
            record={'seed':seed,'padding_m':padding,'capture_points':len(state.capture_grid),'output_dim':len(obs),
                'selected_zero_delivery':bool(state.zero_delivery[0]),'steps_completed':len(rows),'fell':bool(fell),
                'moving_frames':int(moving.sum()),'moving_rows_histogram':{str(i):int(np.sum(row[moving]==i)) for i in range(6)},
                'wheel_clearance_max_m':z[moving].max(0).tolist() if moving.any() else None,
                'wheel_clearance_p95_m':np.quantile(z[moving],.95,axis=0).tolist() if moving.any() else None,
                'airborne_fraction_gt_025':(z[moving]>.025).mean(0).tolist() if moving.any() else None,
                'progress_m':float(position[-1,0]-position[max(0,min(a.hold_steps-1,len(position)-1)),0]),
                'controller_physics_substeps':controller.applied_substeps,'perception':asdict(sensor.cfg)}
            key=f'seed{seed}_padding{padding:g}'
            traces[key]={'rows':rows,'fresh_rows':fresh,'clearance_m':clearance,'root_pos_m':position.tolist(),'newest_packet_age_s':ages}
            cases.append(record)
            (a.output/'cases.json').write_text(json.dumps(cases,indent=2)+'\n')
            print(json.dumps({k:v for k,v in record.items() if k!='perception'}),flush=True)
    result={'cases':len(cases),'wall_seconds':time.monotonic()-started,
            'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            'optimizer_updates':0,'checkpoint_sha256_unchanged':digest(a.checkpoint)==plan['checkpoint_sha256']}
    (a.output/'observations.json').write_text(json.dumps(traces)+'\n')
    (a.output/'execution.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    return result


if __name__=='__main__':
    main()
