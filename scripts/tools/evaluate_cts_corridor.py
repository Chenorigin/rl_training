#!/usr/bin/env python3
"""Bounded CPU measurement through the actual CTS entry; no optimizer or action overrides."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
from unittest.mock import patch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--terrain-xml', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=300)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--command-x', type=float, default=.3)
    p.add_argument('--command-yaw', type=float, default=0.)
    p.add_argument('--spawn-xy', nargs=2, type=float)
    p.add_argument('--noise-profile', default='student-v1')
    a = p.parse_args()
    import math
    if not 2 <= a.steps <= 1000 or not all(math.isfinite(v) for v in (a.command_x,a.command_yaw)):
        p.error('Require 2..1000 steps and finite commands')
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    root = a.project_root.resolve()
    sys.path[:0] = [str(root/'deploy/deploy_mujoco'), str(root/'source/rl_training')]
    import numpy as np
    import torch
    import mujoco
    import deploy_cts_student as entry
    from task_support.cts.joint_limits_mujoco import PositionLimitError
    from task_support.student_v3_5.v3_5_feedback import stair_scan_mask
    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    a.output.mkdir(parents=True, exist_ok=False)
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    initial_hashes = {str(q.resolve()):digest(q) for q in (a.checkpoint,a.model,a.terrain_xml)}
    rows, policies, meta = [], [], {}
    current = {}
    original_factory = entry.quiet.execution_factory
    original_perception = entry.StudentPerception

    class WallContactStop(Exception):
        pass

    class Perception(original_perception):
        def observe(self, data, command, previous_action, step):
            obs = super().observe(data, command, previous_action, step)
            height, valid, confidence = self.map_adapter.last_channels
            usable = (valid > .5) & (confidence > 0)
            grid = torch.tensor(self.contract.grid, dtype=torch.float32)
            z = torch.tensor(np.where(usable, -height, np.nan), dtype=torch.float32)
            hits = torch.cat((grid, z[:,None]), -1)[None]
            current.update(step=step, raw=np.array(command).copy(), effective=obs[6:9].copy(),
                stairs=bool(stair_scan_mask(hits,grid)[0]),
                valid=float(usable.mean()), high=float(((height < -.25) & usable).mean()),
                confidence=float(confidence[usable].mean()) if usable.any() else 0.,
                map_height=height.copy(), map_valid=valid.copy(), map_confidence=confidence.copy())
            meta.setdefault('perception',self.summary())
            return obs

    def recording_factory(document, actor=None):
        factory = original_factory(document, actor)
        def build(model, contract):
            meta['terrain'] = contract.terrain.summary()
            meta['dt'] = float(model.opt.timestep)
            wall_ids = {i for i in contract.terrain_geom_ids if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX}
            wheel_geoms = []
            for body in contract.wheel_body_ids:
                ids = [i for i in range(model.ngeom) if model.geom_bodyid[i] == body
                    and model.geom_type[i] == mujoco.mjtGeom.mjGEOM_CYLINDER
                    and (model.geom_contype[i] or model.geom_conaffinity[i])]
                if len(ids) != 1:
                    raise RuntimeError('Expected one physical cylinder per wheel')
                wheel_geoms.append(ids[0])
            meta['friction'] = {str(i):model.geom_friction[i].tolist() for i in contract.terrain_geom_ids}
            kin = mujoco.MjData(model)

            class Recorder:
                def __init__(self):
                    self.delegate = factory(model,contract)
                def __getattr__(self,key):
                    return getattr(self.delegate,key)
                def policy_inputs(self, observation, command, actor, data, perception=None, trot_control=None):
                    inputs = self.delegate.policy_inputs(observation,command,actor,data,perception,trot_control)
                    assert np.array_equal(inputs['motion_control'][0].numpy(),command)
                    policies.append({**current,'time':float(data.time)})
                    return inputs
                def check_issued(self,data,action):
                    current['action'] = np.array(action).copy()
                    if policies: policies[-1]['action'] = np.array(action).copy()
                    self.delegate.check_issued(data,action)
                def after_step(self,data):
                    self.delegate.after_step(data)
                    kin.qpos[:] = data.qpos
                    kin.qvel[:] = data.qvel
                    mujoco.mj_fwdPosition(model,kin)
                    mujoco.mj_fwdVelocity(model,kin)
                    rot = kin.xmat[contract.base_id].reshape(3,3)
                    attitude = np.array([math.atan2(rot[2,1],rot[2,2]),
                        math.asin(np.clip(-rot[2,0],-1.,1.)),math.atan2(rot[1,0],rot[0,0])])
                    velocity = np.zeros(6)
                    mujoco.mj_objectVelocity(model,kin,mujoco.mjtObj.mjOBJ_BODY,contract.base_id,velocity,1)
                    extent = []
                    for geom in wheel_geoms:
                        az = abs(kin.geom_xmat[geom].reshape(3,3)[2,2])
                        extent.append(model.geom_size[geom,0]*math.sqrt(max(0.,1-az*az))+model.geom_size[geom,1]*az)
                    wall_force, wall_contacts = 0., 0
                    for ci in range(data.ncon):
                        c = data.contact[ci]
                        if c.efc_address < 0 or not ({int(c.geom1),int(c.geom2)} & wall_ids): continue
                        f = np.zeros(6)
                        mujoco.mj_contactForce(model,data,ci,f)
                        wall_force += abs(float(f[0]))
                        wall_contacts += int(c.dist <= 0 and f[0] > 1.)
                    rows.append(dict(time=float(data.time),step=current['step'],
                        position=kin.xpos[contract.base_id].copy(),attitude=attitude,velocity=velocity,
                        joint_position=data.qpos[contract.qpos_ids].copy(),
                        joint_velocity=data.qvel[contract.qvel_ids].copy(),
                        wheel_clearance=kin.geom_xpos[wheel_geoms,2]-np.asarray(extent),
                        support=self.delegate.feedback_force.copy(),
                        corrected_target=np.array(self.delegate.diagnostics['landing_leg_target']).reshape(-1).copy(),
                        wall_force=wall_force,wall_contacts=wall_contacts))
                    if wall_contacts:
                        raise WallContactStop()
            return Recorder()
        return build

    argv = ['deploy_cts_student.py','--checkpoint',str(a.checkpoint.resolve()),'--model',str(a.model.resolve()),
        '--terrain-xml',str(a.terrain_xml.resolve()),'--command-x',str(a.command_x),
        '--command-yaw',str(a.command_yaw),'--noise-profile',a.noise_profile,'--seed',str(a.seed),
        '--steps',str(a.steps),'--stop-on-fall','--no-spawn-mat']
    if a.spawn_xy is not None:
        argv += ['--spawn-xy',*(str(v) for v in a.spawn_xy)]
    fault, result = None, None
    with patch.object(sys,'argv',argv),patch.object(entry,'StudentPerception',Perception),patch.object(entry.quiet,'execution_factory',recording_factory):
        try:
            result = entry.main()
        except PositionLimitError as e:
            fault = {'type':'software_joint_limit',**e.record}
        except WallContactStop:
            fault = {'type':'first_wall_contact','time_s':rows[-1]['time']}
    telemetry = {k:np.asarray([row[k] for row in rows]) for k in rows[0]} if rows else {}
    policy_data = {k:np.asarray([row[k] for row in policies]) for k in policies[0]} if policies else {}
    if telemetry: np.savez_compressed(a.output/'physics.npz',**telemetry)
    if policy_data: np.savez_compressed(a.output/'policy.npz',**policy_data)
    assert all(digest(path) == sha for path,sha in initial_hashes.items()), 'Runtime input changed during evaluation'
    modules = {}
    for module in list(sys.modules.values()):
        file = getattr(module,'__file__',None)
        if file and Path(file).is_file() and str(Path(file).resolve()).startswith(str(root)+os.sep) and Path(file).suffix == '.py':
            modules[str(Path(file).resolve().relative_to(root))] = digest(file)
    report = dict(argv=argv,checkpoint_sha256=digest(a.checkpoint),input_sha256=initial_hashes,
        evaluator_sha256=digest(__file__),source_sha256=modules,seed=a.seed,policy_step_cap=a.steps,
        physics_samples=len(rows),policy_samples=len(policies),fault=fault,result=result,**meta,
        training_updates=0,hardware='NOT_RUN',initialization='Actual headless deployment reset; no GUI stand transition',
        measurement='Read-only hooks; controller/limits unchanged; stop on first force-bearing wall contact')
    (a.output/'receipt.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'samples':len(rows),'policies':len(policies),'fault':fault}))


if __name__ == '__main__':
    main()
