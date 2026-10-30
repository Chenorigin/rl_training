#!/usr/bin/env python3
"""Bounded CPU MuJoCo ascent measurement through the actual CTS deploy entry.

Record each wheel separately; a small world-Z swing peak alone cannot diagnose
motor weakness. No optimizer updates, action overrides or software-limit bypass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


def summarize_crossings(position, axis, forces, edges, radius, half, margin, *, envelope='cylinder'):
    """Independent edge instrument: upward front-surface crossings only.

    Include supported and unsupported contacts. Positive underside margin is
    deliberately conservative: rolling over a lip may succeed with less margin.
    Interpolate the crossing to avoid policy sampling bias; count each edge once
    per wheel (backtracking cannot inflate successful crossing counts).
    """
    import numpy as np
    if envelope not in ('capsule','cylinder'):
        raise ValueError('Unknown wheel envelope')
    def extent(component):
        projection = np.abs(axis[...,component]).clip(0.,1.)
        radial = np.sqrt(1-projection**2) if envelope == 'cylinder' else np.ones_like(projection)
        return radius*radial+half*projection
    bottom = position[..., 2]-extent(2)
    front = position[..., 0]+extent(0)
    result = {}
    for leg, name in enumerate(('FL','FR','HL','HR')):
        events = []
        for edge, top in edges:
            hits = np.flatnonzero((front[:-1, leg] < edge) & (front[1:, leg] >= edge))
            if not len(hits):
                continue
            k = int(hits[0])
            alpha = (edge-front[k,leg])/(front[k+1,leg]-front[k,leg])
            clearance = float(bottom[k,leg]*(1-alpha)+bottom[k+1,leg]*alpha-top)
            events.append(dict(edge_x_m=float(edge), top_m=float(top),
                underside_margin_m=clearance, meets_training_margin=clearance >= margin,
                opposing_force_N=float(max(0., -min(forces[k,leg,0],forces[k+1,leg,0])))))
        values = [e['underside_margin_m'] for e in events]
        result[name] = dict(crossings=len(events), events=events,
            margin_min_m=min(values) if values else None,
            margin_median_m=float(np.median(values)) if values else None,
            below_training_margin=sum(v < margin for v in values),
            below_tread_top=sum(v < 0 for v in values))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New measurement directory')
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--stair-height', type=float, default=.17)
    parser.add_argument('--tread-depth', type=float, default=.30)
    parser.add_argument('--stair-count', type=int, default=8)
    parser.add_argument('--command-x', type=float, default=.5)
    parser.add_argument('--noise-profile', default='student-v1')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--friction', type=float, help='Explicit sliding friction override in a retained evaluation XML')
    a = parser.parse_args(argv)
    if not 2 <= a.steps <= 5000 or not 1 <= a.stair_count <= 20:
        parser.error('Require 2..5000 policy steps and 1..20 stairs')
    for name in ('stair_height','tread_depth','command_x'):
        if not __import__('math').isfinite(getattr(a,name)) or getattr(a,name) <= 0:
            parser.error('Geometry and speed must be finite and positive')
    if a.friction is not None and (not __import__('math').isfinite(a.friction) or a.friction <= 0):
        parser.error('Friction must be finite and positive')
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
        os.environ[name] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.path[:0] = [str(ROOT/'deploy/deploy_mujoco'),str(ROOT/'source/rl_training')]
    import numpy as np
    import torch
    import mujoco
    import deploy_cts_student as entry
    from gait_metrics import GaitMetrics
    from terrain_loader import terrain_contact_bodies
    from task_support.cts.configuration import default_settings
    from task_support.cts.joint_limits_mujoco import PositionLimitError
    from task_support.stair_quiet_v1_width.settings import WidthSettings
    torch.set_num_threads(1)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    tail = WidthSettings()
    settings = default_settings()['all_terrain']['hind_clearance']
    a.output.mkdir(parents=True, exist_ok=False)
    rows, fault, metadata = [], None, {}
    factory = entry.quiet.execution_factory

    def recording_factory(document, actor=None):
        delegate_factory = factory(document, actor)
        def build(model, contract):
            metadata.update(terrain=contract.terrain.summary())
            ids = contract.wheel_body_ids
            wheel_geoms = []
            for body in ids:
                geoms = [i for i in range(model.ngeom) if model.geom_bodyid[i] == body
                         and model.geom_type[i] == mujoco.mjtGeom.mjGEOM_CYLINDER
                         and (model.geom_contype[i] or model.geom_conaffinity[i])]
                if len(geoms) != 1:
                    raise ValueError('Measurement requires one physical cylinder per wheel')
                geom = geoms[0]
                if not np.allclose(model.geom_pos[geom],0.,atol=1.e-7) or not np.allclose(
                        model.geom_size[geom,:2],[tail.wheel_radius,tail.wheel_half_length],atol=1.e-6):
                    raise ValueError('Wheel dimensions/origin differ from the expected M20 geometry')
                wheel_geoms.append(geom)
            metadata['wheel_collision'] = dict(shape='cylinder',radius_m=tail.wheel_radius, half_length_m=tail.wheel_half_length)
            metadata['training_envelope'] = 'capsule with same radius/half length; separately reported, conservative for cylinders'
            metrics = GaitMetrics(contract)
            kin = mujoco.MjData(model)
            class Recorder:
                def __init__(self):
                    self.delegate = delegate_factory(model, contract)
                def __getattr__(self, key): return getattr(self.delegate, key)
                def after_step(self, data):
                    self.delegate.after_step(data)
                    kin.qpos[:] = data.qpos; kin.qvel[:] = data.qvel
                    mujoco.mj_fwdPosition(model, kin)
                    ids = contract.wheel_body_ids
                    for ci in range(data.ncon):
                        contact = data.contact[ci]
                        b1,b2 = terrain_contact_bodies(model, contact, metrics.flex_ids)
                        if contact.efc_address >= 0 and ((b1 in ids and b2 == 0) or (b2 in ids and b1 == 0)):
                            metadata.setdefault('effective_contact_friction',[]).append(float(contact.friction[0]))
                    rows.append(dict(time=float(data.time), position=kin.xpos[ids].copy(),
                        axis=kin.geom_xmat[wheel_geoms].reshape(4,3,3)[...,2].copy(),
                        force=metrics.forces(data), effort=data.actuator_force[contract.actuator_ids].copy(),
                        joint_pos=data.qpos[contract.qpos_ids].copy()))
            return Recorder()
        return build

    args = ['deploy_cts_student.py','--checkpoint',str(a.checkpoint),'--model',str(a.model),
        '--terrain','ascent','--stair-height',str(a.stair_height),'--tread-depth',str(a.tread_depth),
        '--stair-count',str(a.stair_count),'--command-x',str(a.command_x),
        '--noise-profile',a.noise_profile,'--seed',str(a.seed),'--steps',str(a.steps),
        '--stop-on-fall','--no-spawn-mat','--output',str(a.output/'playback.npz')]
    if a.friction is not None:
        import xml.etree.ElementTree as ET
        xml = ET.parse(ROOT/'deploy/deploy_mujoco/terrains/stairs_ascent.xml')
        friction = xml.find("./custom/numeric[@name='friction']")
        old = friction.get('data').split()
        friction.set('data', ' '.join([str(a.friction), *old[1:]]))
        path = (a.output/'terrain.xml').resolve()
        xml.write(path, encoding='utf-8', xml_declaration=True)
        offset = args.index('--terrain')
        args[offset:offset+2] = ['--terrain-xml', str(path)]
    result = None
    with patch.object(sys,'argv',args), patch.object(entry.quiet,'execution_factory',recording_factory):
        try:
            result = entry.main()
        except PositionLimitError as error:
            fault = error.record
    telemetry = {k:np.asarray([row[k] for row in rows]) for k in rows[0]} if rows else {}
    if telemetry:
        np.savez_compressed(a.output/'wheel_telemetry.npz', **telemetry)
    terrain = metadata.get('terrain',{})
    mu = metadata.pop('effective_contact_friction',[])
    metadata['effective_contact_friction'] = dict(samples=len(mu), minimum=min(mu) if mu else None,
        maximum=max(mu) if mu else None)
    p = terrain.get('terrain_parameters',{})
    # TerrainConfig.summary stores all geometry as lists. Do not guess the
    # default stair-start from training settings or infer it from wheel motion.
    start = float(p['start_x'][0]) if 'start_x' in p else None
    if start is None:
        raise RuntimeError('MuJoCo terrain must expose actual start_x')
    edges = [(start+i*a.tread_depth,(i+1)*a.stair_height) for i in range(a.stair_count)]
    report = dict(checkpoint=str(a.checkpoint.resolve()),
        checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
        model=str(a.model.resolve()), model_sha256=hashlib.sha256(a.model.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        seed=a.seed, policy_step_cap=a.steps, samples=len(rows), fault=fault, playback=result,
        sample_scope='Every physics substep; integrated positions refreshed on a separate MjData; forces from completed solver step',
        telemetry_dt_s=float(rows[1]['time']-rows[0]['time']) if len(rows)>1 else None,
        **metadata, training_margin_m=settings['margin_m'],
        wheels=summarize_crossings(telemetry['position'],telemetry['axis'],telemetry['force'],edges,
            tail.wheel_radius,tail.wheel_half_length,settings['margin_m']) if len(rows)>1 else {},
        training_capsule_wheels=summarize_crossings(telemetry['position'],telemetry['axis'],telemetry['force'],edges,
            tail.wheel_radius,tail.wheel_half_length,settings['margin_m'],envelope='capsule') if len(rows)>1 else {},
        training_updates=0, hardware_validation='NOT_RUN',
        scope='Actual cylinder and training capsule underside at their leading-surface edge crossings; deficit is not proof of riser collision or motor torque weakness')
    (a.output/'clearance.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ('samples','fault','wheels')},indent=2),flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
