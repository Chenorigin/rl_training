#!/usr/bin/env python3
"""CPU CTS ONNX closed-loop contact ablation using the original deploy adapters.

Explicit experiment only: estimated-force validity is a mathematical hypothesis,
and off bypasses landing control. Neither option modifies production admission.
"""
import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import resource
import signal
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
DEPLOY_ROOT = ROOT.parent / 'm20_student_deploy'
sys.path[:0] = [str(ROOT / 'deploy/deploy_mujoco'), str(ROOT / 'source/rl_training'),
                str(DEPLOY_ROOT / 'src/m20_heightmap')]

import mujoco
import numpy as np
import torch

import deploy_mujoco as baseline
from deploy_student_v3_safe import StudentPerception
from terrain_loader import terrain_contact_bodies
from task_support.stair_quiet_v1.impact_events import ImpactEvents
from task_support.stair_quiet_v1.landing import LandingSettings
from task_support.stair_quiet_v1.mujoco_landing import MujocoLanding
from task_support.stair_quiet_v1.traversal import TopLanding
from m20_heightmap.contact_estimator import ContactModel, ORDER
from m20_heightmap.cts import CTSInference

spec = importlib.util.spec_from_file_location('contact_capture', DEPLOY_ROOT / 'scripts/capture_cts_contact.py')
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def contacts(model, data, contract):
    wheels = {int(body): i for i, body in enumerate(contract.wheel_body_ids)}
    net = np.zeros((4, 3))
    normal_support = np.zeros(4)
    groups, points = {}, []
    nonwheel, side = 0, 0
    for ci in range(data.ncon):
        contact = data.contact[ci]
        if contact.efc_address < 0:
            continue
        first, second = terrain_contact_bodies(model, contact, contract.terrain_flex_ids)
        if first in wheels and second == 0:
            wi, sign, geom = wheels[first], -1., int(contact.geom2)
        elif second in wheels and first == 0:
            wi, sign, geom = wheels[second], 1., int(contact.geom1)
        else:
            nonwheel += int((first == 0) != (second == 0))
            continue
        local = np.zeros(6)
        mujoco.mj_contactForce(model, data, ci, local)
        if local[0] <= 0:
            continue
        frame = contact.frame.reshape(3, 3)
        net[wi] += sign * frame.T @ local[:3]
        nz = sign * frame[0, 2]
        if nz >= .8:
            normal_support[wi] += local[0] * nz
        else:
            side += 1
        points.append((wi, contact.pos.copy(), nz, float(local[0])))
        axis = int(np.argmax(np.abs(frame[0])))
        key = f'{wi}/{geom}/{axis}'
        row = groups.setdefault(key, dict(normal_force_N=0., position_m=np.zeros(3),
                                         wheel=('fl', 'fr', 'hl', 'hr')[wi], surface=f'{geom}/{axis}'))
        row['normal_force_N'] += float(local[0])
        row['position_m'] += contact.pos * local[0]
    for row in groups.values():
        row['position_m'] /= row['normal_force_N']
    return np.maximum(net[:, 2], 0.), normal_support, groups, points, nonwheel, side


def observer_payload(state, torque, diag, stamp):
    header = {'stamp': dict(sec=stamp // 10**9, nanosec=stamp % 10**9)}
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, state['base_rotation'].reshape(-1))
    imu = dict(header=header, orientation=dict(zip(('w', 'x', 'y', 'z'), quat.tolist())),
               angular_velocity=dict(zip('xyz', state['omega_body'].tolist())),
               linear_acceleration=dict(zip('xyz', state['base_specific_force'].tolist())))
    offset = diag.offset.copy()
    data = dict(header=header, data={'joints_data': [
        dict(name=list(str(i).encode()) + [0],
             position=float((state['q'][i] - offset[i]) * diag.direction[i]),
             velocity=float(state['dq'][i] * diag.direction[i]),
             torque=float(torque[i] * diag.direction[i])) for i in range(16)]})
    return imu, data


def run_case(protocol, session, provenance, contract_document, role, scene, seed, folder):
    start = time.monotonic()
    torch.manual_seed(seed)
    np.random.seed(seed)
    args = SimpleNamespace(terrain=scene, terrain_xml=None, stair_height=None, tread_depth=None,
        stair_count=None, stair_start=None, seed=seed, noise_profile='student-v1',
        profile_file=None, noise_level=None, checkpoint=Path(protocol['checkpoint']))
    if scene != 'flat':
        args.stair_height = protocol['stair_height_m']
        args.tread_depth = protocol['tread_depth_m']
        args.stair_count = protocol['stair_count']
    model = baseline.load_model(Path(protocol['model']), args)
    c = baseline.M20Contract(model, args.terrain_config)
    data = mujoco.MjData(model)
    c.reset(data)
    perception = StudentPerception(c, args, provenance)
    landing = MujocoLanding(model, c, LandingSettings(**contract_document['quiet_execution_control']['settings']), baseline)
    landing.contact_rng = np.random.default_rng(seed)
    geometry = json.loads((DEPLOY_ROOT / 'deploy/m20_contact_geometry.json').read_text())
    robot = json.loads((DEPLOY_ROOT / 'deploy/robot.json').read_text())
    diagnostic = capture.Diagnostics(robot, geometry)
    core = ContactModel(geometry)
    sdk_qpos = np.array([model.joint(name).qposadr[0] for name in ORDER])
    sdk_dof = np.array([model.joint(name).dofadr[0] for name in ORDER])
    dt = float(model.opt.timestep)
    substeps = round(protocol['policy_dt_s'] / dt)
    previous = np.zeros(16, dtype=np.float32)
    feedback = np.zeros(4)
    feedback_valid = np.zeros(4, bool)
    feedback_stamp = None
    feedback_oracle = np.zeros(4)
    feedback_comparable = False
    records = {key: [] for key in ('time_s', 'root_position_m', 'wheel_position_m', 'upright',
        'roll_pitch_rad', 'true_force_N', 'upward_normal_force_N', 'feedback_force_N', 'feedback_valid',
        'feedback_oracle_N', 'oracle_contact_state', 'contact_state', 'kp_scale', 'kd_scale', 'correction_rad', 'modified', 'terrain_valid',
        'nominal_action', 'applied_action', 'effort_Nm', 'side_contacts', 'nonwheel_contacts')}
    force_errors = []
    oracle_state, oracle_pending = np.zeros(4, bool), np.zeros(4)
    extractor = ImpactEvents(dt)
    top = TopLanding()
    top_height = protocol['stair_count'] * protocol['stair_height_m'] if scene == 'ascent' else 0.
    start_x = float(data.qpos[0])
    fell = completed = bypass = False
    max_steps = round(protocol['scenes'][scene] / protocol['policy_dt_s'])
    command = np.zeros(3, dtype=np.float32)
    for policy_tick in range(max_steps):
        if data.time >= protocol['command_start_s'] - 1e-9:
            command[:] = protocol['command']
        # Refresh integrated kinematics; leave the preceding solved feedback
        # in our immutable arrays rather than reading mismatched data fields.
        mujoco.mj_fwdPosition(model, data)
        mujoco.mj_fwdVelocity(model, data)
        obs = perception.observe(data, command, previous, policy_tick)
        landing.observe(data, perception.map_adapter)
        nominal = np.clip(session(obs[None], command[None])[0], -100., 100.).astype(np.float32)
        action = perception.action(nominal)
        for _ in range(substeps):
            mujoco.mj_step1(model, data)
            if role == 'off':
                c.apply_pd(data, action)
                diag = dict(landing_kp_scale=np.ones(4), landing_kd_scale=np.ones(4),
                    landing_correction=np.zeros((4, 3)), landing_modified=np.zeros(4, bool),
                    landing_terrain_valid=np.zeros(4, bool), landing_phase=np.zeros(4, int))
            else:
                age = dt if feedback_stamp is None else float(data.time) - feedback_stamp
                effort, diag = landing.apply(data, action, feedback, feedback_valid,
                    bool(np.max(np.abs(command)) > landing.settings.command_active_threshold), dt, age)
                data.ctrl[c.actuator_ids] = effort
            selected_feedback = feedback.copy()
            selected_valid = feedback_valid.copy()
            selected_oracle = feedback_oracle.copy()
            wants = feedback_comparable & (selected_oracle >= np.where(oracle_state,
                landing.settings.contact_off_force_N, landing.settings.contact_on_force_N))
            oracle_pending = np.where(wants, oracle_pending + dt, 0.)
            oracle_state = wants & (oracle_state | (oracle_pending >= landing.settings.contact_confirm_s))
            if feedback_comparable:
                force_errors.extend((feedback - feedback_oracle)[feedback_valid].tolist())
            # Pre-step state paired with the torque/IMU/contact from this solve.
            source_time = float(data.time)
            rotation = data.xmat[c.base_id].reshape(3, 3).copy()
            before_q = data.qpos[sdk_qpos].copy()
            before_dq = data.qvel[sdk_dof].copy()
            before_omega = data.qvel[3:6].copy()
            effort = data.ctrl[c.actuator_ids].copy()
            mujoco.mj_step2(model, data)
            true_force, upward, groups, touch_points, nonwheel, side = contacts(model, data, c)
            extractor.update(source_time, groups)
            if role == 'oracle':
                feedback = true_force.copy()
                feedback_valid.fill(True)
            elif role.startswith('estimate_'):
                state = dict(q=before_q, dq=before_dq, base_rotation=rotation,
                    base_specific_force=rotation.T @ (data.qacc[:3] - model.opt.gravity),
                    omega_body=before_omega)
                torque = data.qfrc_actuator[sdk_dof].copy()
                if role == 'estimate_current':
                    stamp = 10**9 + round(source_time * 10**9)
                    imu, joints = observer_payload(state, torque, diagnostic, stamp)
                    diagnostic.consume('/IMU', imu, stamp)
                    estimated = diagnostic.consume('/JOINTS_DATA', joints, stamp)
                else:
                    # These end-of-interval velocities are observed only now.
                    # The resulting force belongs to source_time, age=dt.
                    state['ddq'] = (data.qvel[sdk_dof] - before_dq) / dt
                    state['alpha_body'] = (data.qvel[3:6] - before_omega) / dt
                    estimated = core.diagnose(torque, **state)
                force = np.full((4, 3), np.nan) if estimated is None else np.asarray(estimated['estimated_force_world_N'], dtype=float)
                feedback_valid = np.isfinite(force).all(-1)
                feedback = np.where(feedback_valid, np.maximum(force[:, 2], 0.), 0.)
            feedback_stamp = source_time
            feedback_oracle = true_force.copy()
            feedback_comparable = role != 'off'
            # Outcome positions are refreshed separately from solved contacts.
            mujoco.mj_fwdPosition(model, data)
            mujoco.mj_fwdVelocity(model, data)
            root = data.xpos[c.base_id].copy()
            wheels = data.xpos[c.wheel_body_ids].copy()
            r = data.xmat[c.base_id].reshape(3, 3)
            upright = float(r[2, 2])
            roll = math.atan2(r[2, 1], r[2, 2])
            pitch = math.atan2(-r[2, 0], math.hypot(r[2, 1], r[2, 2]))
            ground = c.ground_height(data, root[0], root[1])
            fell |= root[2] - ground < .22 or upright < .35
            touched = np.zeros(4, bool)
            for wi, point, nz, force in touch_points:
                if point[0] >= c.terrain.goal_x and abs(point[2] - top_height) <= top.settings.top_contact_height_tolerance_m and nz >= top.settings.top_contact_normal_z_min and force >= extractor.settings.contact_on_N:
                    touched[wi] = True
            if scene != 'flat':
                in_stairs = (wheels[:, 0] >= float(c.terrain.parameters['start_x'][0])) & (wheels[:, 0] <= c.terrain.goal_x)
                bypass |= bool((in_stairs & (np.abs(wheels[:, 1]) > c.terrain.course_width / 2)).any())
                beyond = wheels[:, 0] >= c.terrain.goal_x + baseline.WHEEL_RADIUS
                inside = bool((np.abs(wheels[:, 1]) <= c.terrain.course_width / 2).all())
                completed = top.update(float(data.time), beyond, touched, invalid=fell or bypass or not inside)
            row = dict(time_s=float(data.time), root_position_m=root, wheel_position_m=wheels,
                upright=upright, roll_pitch_rad=[roll, pitch], true_force_N=true_force,
                upward_normal_force_N=upward, feedback_force_N=selected_feedback,
                feedback_valid=selected_valid, feedback_oracle_N=selected_oracle,
                oracle_contact_state=oracle_state.copy(), contact_state=landing.control.contact[0].numpy().copy(),
                kp_scale=diag['landing_kp_scale'], kd_scale=diag['landing_kd_scale'],
                correction_rad=diag['landing_correction'], modified=diag['landing_modified'],
                terrain_valid=diag['landing_terrain_valid'], nominal_action=nominal.copy(), applied_action=action.copy(),
                effort_Nm=effort, side_contacts=side, nonwheel_contacts=nonwheel)
            for key, value in row.items():
                records[key].append(value)
            if fell or completed:
                break
        previous = nominal.copy()  # Preserve training's pre-execution history.
        if fell or completed:
            break
    events = extractor.finish()
    arrays = {key: np.asarray(value) for key, value in records.items()}
    command_window = arrays['time_s'] >= protocol['command_start_s']
    angles = arrays['roll_pitch_rad'][command_window]
    active_events = [event for event in events if event['time_s'] >= protocol['command_start_s']]
    peaks = [event['peak_normal_force_N'] for event in active_events]
    impulses = [event['normal_impulse_N_s'] for event in active_events]
    progress = float(arrays['root_position_m'][-1, 0] - start_x)
    success = (not fell and progress >= 3.) if scene == 'flat' else completed
    summary = dict(role=role, scene=scene, seed=seed, success=bool(success), fell=bool(fell),
        bypass=bool(bypass), completed=bool(completed), progress_m=progress,
        simulated_s=float(data.time), physics_steps=len(arrays['time_s']),
        roll_pitch_rms_deg=float(np.sqrt(np.mean(angles ** 2)) * 180 / np.pi) if len(angles) else None,
        max_tilt_deg=float(np.max(np.arccos(np.clip(arrays['upright'], -1., 1.))) * 180 / np.pi),
        peak_contact_force_N=float(np.max(arrays['true_force_N'])),
        event_count=len(active_events), impact_peak_p95_N=float(np.quantile(peaks, .95)) if peaks else None,
        impact_impulse_p95_Ns=float(np.quantile(impulses, .95)) if impulses else None,
        impact_impulse_total_Ns=float(sum(impulses)), impact_impulse_per_m_Ns=float(sum(impulses) / max(progress, .01)),
        modified_wheel_samples=int(arrays['modified'].sum()),
        modified_fraction=float(arrays['modified'].mean()),
        max_joint_correction_rad=float(np.max(np.abs(arrays['correction_rad']))),
        min_kp_scale=float(arrays['kp_scale'].min()), min_kd_scale=float(arrays['kd_scale'].min()),
        usable_feedback_fraction=float(arrays['feedback_valid'].mean()),
        contact_fsm_agreement=float((arrays['contact_state'] == arrays['oracle_contact_state']).mean()) if role != 'off' else None,
        false_support_wheel_samples=int((arrays['contact_state'] & ~arrays['oracle_contact_state']).sum()) if role != 'off' else None,
        missed_support_wheel_samples=int((~arrays['contact_state'] & arrays['oracle_contact_state']).sum()) if role != 'off' else None,
        terrain_valid_fraction=float(arrays['terrain_valid'].mean()),
        side_contact_frames=int((arrays['side_contacts'] > 0).sum()),
        nonwheel_contact_frames=int((arrays['nonwheel_contacts'] > 0).sum()),
        source_time_force_error_p95_N=float(np.quantile(np.abs(force_errors), .95)) if force_errors else None,
        final_root_position_m=arrays['root_position_m'][-1].tolist(),
        all_wheels_touched_final_platform=top.seen.tolist(), terrain=c.terrain.summary(),
        perception=perception.summary(), elapsed_wall_s=time.monotonic() - start,
        hardware_admission_granted=False, estimator_validity_assumed_for_offline_test=role.startswith('estimate_'))
    name = f'{scene}_{role}_{seed}'
    np.savez_compressed(folder / (name + '.npz'), **arrays)
    write_json(folder / (name + '.events.json'), events)
    write_json(folder / (name + '.json'), summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    p = json.loads(args.protocol.read_text())
    for path, expected in p['source_hashes'].items():
        if digest(path) != expected:
            raise ValueError('Frozen source drift: ' + path)
    available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))) // 1024
    if available < p['resources']['expected_rss_mb']:
        raise RuntimeError('Insufficient physical memory for declared CPU job')
    resource.setrlimit(resource.RLIMIT_AS, (p['resources']['virtual_address_space_mb'] * 1024**2,) * 2)
    signal.alarm(p['resources']['wall_timeout_s'])
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'protocol.json').write_bytes(args.protocol.read_bytes())
    saved = torch.load(p['checkpoint'], map_location='cpu', weights_only=True)
    provenance = saved['cts']
    document = json.loads(Path(p['contract']).read_text())
    if provenance['quiet_execution_control'] != document['quiet_execution_control']:
        raise ValueError('Checkpoint and deployed execution settings differ')
    del saved
    session = CTSInference(Path(p['onnx']), Path(p['contract']))
    before_model = digest(p['onnx'])
    report = dict(schema=p['schema'], protocol_sha256=digest(args.protocol), script_sha256=digest(__file__),
        model_sha256=before_model, mujoco_version=mujoco.__version__, cpu_only=True,
        hardware_commands=0, optimizer_updates=0, cases=[], steps=0, completed=False)
    start = time.monotonic()
    for scene in p['scenes']:
        for seed in p['seeds']:
            for role in p['roles']:
                summary = run_case(p, session, provenance, document, role, scene, seed, args.output)
                report['cases'].append(summary)
                report['steps'] += summary['physics_steps']
                if report['steps'] > p['resources']['max_physics_steps']:
                    raise RuntimeError('Frozen physical step budget exceeded')
                write_json(args.output / 'results.json', report)
                print(json.dumps({key: summary[key] for key in ('scene', 'role', 'seed', 'success', 'fell',
                    'progress_m', 'simulated_s', 'modified_wheel_samples', 'impact_peak_p95_N', 'elapsed_wall_s')}), flush=True)
    report.update(completed=True, policy_unchanged=digest(p['onnx']) == before_model,
                  elapsed_wall_s=time.monotonic() - start, max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write_json(args.output / 'results.json', report)


if __name__ == '__main__':
    main()
