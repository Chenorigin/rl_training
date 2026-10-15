"""Frozen three-policy CPU MuJoCo benchmark. No training or production edits."""
from collections import deque
from pathlib import Path
from types import SimpleNamespace
import json
import math
import resource
import sys
import time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(ROOT / 'deploy/deploy_mujoco'))
import mujoco
import numpy as np
import torch
from tensordict import TensorDict
import deploy_mujoco as baseline
from quiet_landing_metrics import QuietMujocoMetrics
from student_v0_adapter import load_student, profile_config, StudentMapAdapter, sha256


def verify_sources(criteria):
    for name, path in criteria['checkpoints'].items():
        if sha256(path) != criteria['checkpoint_sha256'][name]:
            raise ValueError(f'Checkpoint changed after freeze: {name}')
    for key in ('model', 'terrain'):
        if sha256(criteria[key]) != criteria[key + '_sha256']:
            raise ValueError(f'{key} changed after freeze')
    for path, digest in criteria['source_sha256'].items():
        if sha256(ROOT / path) != digest:
            raise ValueError(f'Source changed after freeze: {path}')


def command_for(data, contract, config):
    r = data.xmat[contract.base_id].reshape(3, 3)
    yaw = math.atan2(r[1,0], r[0,0])
    target = contract.terrain.spawn_yaw
    side = float(np.clip(-config['lateral_gain'] * (data.xpos[contract.base_id,1] - contract.terrain.spawn_xy[1]),
                         -config['lateral_cap_m_s'], config['lateral_cap_m_s']))
    vx = config['forward_world_m_s'] * math.cos(target) - side * math.sin(target)
    vy = config['forward_world_m_s'] * math.sin(target) + side * math.cos(target)
    error = math.atan2(math.sin(target-yaw),math.cos(target-yaw))
    return np.asarray([math.cos(yaw)*vx+math.sin(yaw)*vy, -math.sin(yaw)*vx+math.cos(yaw)*vy,
                       np.clip(config['yaw_gain']*error,-config['yaw_cap_rad_s'],config['yaw_cap_rad_s'])], dtype=np.float32)


def run(criteria, name, case, profile):
    label = f'{name}_{profile}_{case["name"]}'
    folder = OUT / label
    folder.mkdir(exist_ok=False)
    started = time.monotonic()
    torch.manual_seed(case['seed'])
    is_student = name == 'student'
    path = Path(criteria['checkpoints'][name])
    if is_student:
        actor, provenance, identity = load_student(path)
    else:
        actor = baseline.load_actor(path)
        actor.requires_grad_(False)
        provenance, identity = {}, {'policy_path':str(path), 'policy_sha256':criteria['checkpoint_sha256'][name]}
    args = SimpleNamespace(terrain=None, terrain_xml=Path(criteria['terrain']))
    model = baseline.load_model(Path(criteria['model']), args)
    contract = baseline.M20Contract(model, args.terrain_config)
    data = mujoco.MjData(model)
    contract.reset(data)
    adr = int(contract.root_joint.qposadr[0])
    data.qpos[adr+1] += case['y_offset_m']
    yaw = contract.terrain.spawn_yaw + case['yaw_offset_rad']
    data.qpos[adr+3:adr+7] = [math.cos(yaw/2),0,0,math.sin(yaw/2)]
    mujoco.mj_forward(model,data)
    initial_qpos, initial_qvel = data.qpos.copy(), data.qvel.copy()
    if is_student:
        cfg, extras = profile_config(profile,provenance)
        adapter = StudentMapAdapter(contract,cfg,extras,case['seed'])
        delay_steps = extras.action_delay_steps
        perception = adapter.summary()
    else:
        delay_steps = 0
        perception = {'perception':'oracle 244D height scan, no sensor/action noise'}
    metrics = QuietMujocoMetrics(model,contract)
    contract.height_scan(data)
    previous = np.zeros(16,dtype=np.float32)
    queue = deque([previous.copy() for _ in range(delay_steps)])
    telemetry = {k:[] for k in ('time_s','command','action','applied_action','base_position','valid_fraction')}
    fell, boundary = False, False
    stopped = 'timeout'
    c = criteria['control']
    if not math.isclose(model.opt.timestep,c['physics_dt_s']):
        raise ValueError('Physics timestep mismatch')
    substeps = round(c['policy_dt_s']/model.opt.timestep)
    max_steps = round(c['max_duration_s']/c['policy_dt_s'])
    for step in range(max_steps):
        command = command_for(data,contract,c)
        obs = adapter.observe(data,command,previous,step) if is_student else contract.observe(data,command,previous)
        with torch.inference_mode():
            tensor = torch.from_numpy(obs)[None]
            action = (actor(tensor) if is_student else actor(TensorDict({'policy':tensor},batch_size=[1])))[0].numpy()
        if action.shape != (16,) or not np.isfinite(action).all():
            raise ValueError(f'Nonfinite or malformed policy output: {label}')
        action = np.clip(action,-100,100).astype(np.float32)
        queue.append(action.copy()); applied = queue.popleft()
        metrics.command_xy = command[:2].copy()
        metrics.tracker.begin_step()
        before_xy = data.xpos[contract.base_id,:2].copy()
        for _ in range(substeps):
            contract.apply_pd(data,applied)
            mujoco.mj_step1(model,data)
            snapshot = metrics.pre_step(data)
            mujoco.mj_step2(model,data)
            metrics.post_step(data,snapshot,record=True)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise ValueError(f'Nonfinite physics state: {label}')
        post = metrics.kinematics
        position = post.xpos[contract.base_id].copy()
        metrics.tracker.distance[0] += float(np.linalg.norm(position[:2]-before_xy))
        previous = action.copy()
        for key,value in (('time_s',float(data.time)),('command',command),('action',action),
                          ('applied_action',applied),('base_position',position),
                          ('valid_fraction',float(adapter.last_channels[1].mean()) if is_student else 1.)):
            telemetry[key].append(value)
        rzz = float(post.xmat[contract.base_id].reshape(3,3)[2,2])
        wheels = post.xpos[contract.wheel_body_ids]
        boundary |= bool(np.any(np.abs(wheels[:,1]) > contract.terrain.course_width/2))
        local_z = contract.ground_height(post,float(position[0]),float(position[1]))
        fell |= bool(position[2]-local_z < criteria['fall']['min_body_ground_height_m'] or rzz < criteria['fall']['min_Rzz'])
        if fell:
            stopped = 'fall';break
        if boundary:
            stopped = 'boundary_exit';break
        if contract.crossing_summary(post,fell)['cleared_terrain']:
            stopped = 'finish';break
        if step and step % 1000 == 0:
            print(f'PROGRESS {label} t={data.time:.2f}s x={position[0]:.3f}',flush=True)
    result = {'label':label,'model':name,'case':case,'profile':profile,**identity,
              'physics':'closed-loop mj_step1/mj_step2, contact solver active', 'physics_dt_s':model.opt.timestep,
              'policy_dt_s':c['policy_dt_s'],'weights_updated':False,'controller':name+'_only',
              'duration_s':float(data.time),'stop_reason':stopped,'fell':fell,'ever_boundary_exit':boundary,
              'complete':stopped=='finish' and not fell and not boundary,'terrain':args.terrain_config.summary(),
              'perception':perception,'initial_qpos':initial_qpos.tolist(),'initial_qvel':initial_qvel.tolist(),
              'max_x_m':float(np.asarray(telemetry['base_position'])[:,0].max()),
              'valid_fraction_mean':float(np.mean(telemetry['valid_fraction'])),
              'wall_time_s':time.monotonic()-started,'quiet_aggregate':metrics.summary(),
              'peak_process_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    arrays = {k:np.asarray(v) for k,v in metrics.records.items()}
    arrays.update({k:np.asarray(v) for k,v in metrics.extended_records.items()})
    # This explicit definition supersedes the old generic sampler metadata string.
    np.savez_compressed(folder/'quiet.npz',**arrays,metadata=json.dumps({**result,
                        'wheel_masses_kg':metrics.wheel_masses.tolist(),
                        'energy_definition':'pre-impact normal closing speed squared / 2; J/kg proxy',
                        'force_definition':'summed actual wheel/static terrain contact vector at each 5ms solve'}))
    np.savez_compressed(folder/'telemetry.npz',**{k:np.asarray(v) for k,v in telemetry.items()})
    (folder/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('RESULT',json.dumps({k:result[k] for k in ('label','complete','duration_s','stop_reason','max_x_m','wall_time_s','peak_process_rss_kib')}),flush=True)
    return result


def main():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    criteria = json.loads((OUT/'criteria.json').read_text())
    verify_sources(criteria)
    script_sha = sha256(__file__)
    (OUT/'execution.json').write_text(json.dumps({'started_at':time.time(),'script_sha256':script_sha,
                        'mujoco_version':mujoco.__version__,'torch_version':torch.__version__,
                        'criteria_sha256':sha256(OUT/'criteria.json'),'cuda_visible':False},indent=2)+'\n')
    results=[]
    for case in criteria['cases']:
        for name in criteria['primary']['models']:
            results.append(run(criteria,name,case,'clean'))
            (OUT/'suite_summary.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
    for profile in criteria['secondary']['noise_profiles']:
        results.append(run(criteria,'student',criteria['cases'][0],profile))
        (OUT/'suite_summary.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
    verify_sources(criteria)
    print('SUITE_DONE',len(results),flush=True)


if __name__ == '__main__':
    main()
