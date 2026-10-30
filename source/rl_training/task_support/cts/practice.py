"""Bounded, domain-randomized online reverse practice; no simulator imports.

Terrain contacts use the original five physical profiles. Robot mass, COM,
inertia, joints, gains, sensor errors and actuator delay retain the source
task's randomizers. Mild practice-only pushes and reset jitter are additional.
"""
from copy import deepcopy
from collections import UserDict
from functools import wraps
import math

import torch


def flat_patch_mesh(generator, length):
    """CPU geometry shared by collision construction and regression checks."""
    import numpy as np
    import trimesh
    rows, cols = generator.num_rows, generator.num_cols
    width = generator.size[1]
    origins = np.zeros((rows, cols, 3), dtype=np.float32)
    parts = []
    for row in range(rows):
        for col in range(cols):
            x = -rows*generator.size[0]/2-(row+.5)*length
            y = -cols*width/2+(col+.5)*width
            origins[row, col] = (x, y, 0.)
            box = trimesh.creation.box((length, width, .2))
            box.apply_translation((x, y, -.1))
            parts.append(box)
    return trimesh.util.concatenate(parts), origins


def cohort_mask(n, fraction, seed, device='cpu'):
    generator = torch.Generator(device='cpu').manual_seed(seed)
    ids = torch.randperm(n, generator=generator)[:math.floor(n*fraction)]
    mask = torch.zeros(n, dtype=torch.bool)
    mask[ids] = True
    return mask.to(device)


def practice_mask(teacher, fraction, seed, excluded=None):
    selected = torch.zeros_like(teacher)
    if not fraction:
        return selected
    generator = torch.Generator(device='cpu').manual_seed(seed+1)
    for mask in (teacher, ~teacher):
        total = int(mask.sum())
        eligible = mask if excluded is None else mask & ~excluded
        ids = eligible.cpu().nonzero().flatten()
        count = min(total, max(1, round(total*fraction)))
        if count > len(ids):
            raise ValueError('CTS environment count cannot contain disjoint practice cohorts')
        ids = ids[torch.randperm(len(ids), generator=generator)[:count]]
        selected[ids.to(teacher.device)] = True
    return selected


def validate_practice(cfg):
    # Historical checkpoint schema only; current CTS has one unified motion
    # group and no active backward_practice section in its numerical ledger.
    expected = {'fraction','speed_range','initial_hold_s','reverse_s','stop_s','patch_length',
                'boundary_margin','reset_xy_range','reset_yaw_range','reset_velocity_range',
                'push_interval_range_s','push_velocity_range','tracking_std','linear_reward',
                'yaw_reward','tilt_cost','action_rate_cost','effort_cost','failure_cost'}
    if not isinstance(cfg, dict) or set(cfg) != set(expected):
        raise ValueError('Unknown or missing CTS backward practice settings')
    for key, value in cfg.items():
        values = value if isinstance(value, (tuple, list)) else [value]
        if any(type(x) not in (int, float) or not math.isfinite(x) for x in values):
            raise ValueError('Nonfinite/non-numeric CTS backward practice setting: '+key)
        if isinstance(value, (tuple, list)) and len(value) != 2:
            raise ValueError('CTS backward ranges require two endpoints')
    if not 0 <= cfg['fraction'] < .5 or not cfg['speed_range'][0] < cfg['speed_range'][1] < 0:
        raise ValueError('CTS backward practice requires a minority and negative speed range')
    for key in ('initial_hold_s', 'reverse_s', 'stop_s', 'patch_length', 'boundary_margin', 'tracking_std'):
        if cfg[key] <= 0:
            raise ValueError('CTS backward practice requires positive '+key)
    if (not 0 < cfg['push_interval_range_s'][0] <= cfg['push_interval_range_s'][1] < cfg['reverse_s']
            or not cfg['push_velocity_range'][0] <= 0 <= cfg['push_velocity_range'][1]
            or min(cfg['reset_xy_range']) < 0 or cfg['reset_yaw_range'] < 0 or cfg['reset_velocity_range'] < 0
            or any(cfg[k] < 0 for k in ('linear_reward','yaw_reward','tilt_cost','action_rate_cost','effort_cost','failure_cost'))):
        raise ValueError('Invalid CTS backward randomization/reward ranges')
    reach = abs(cfg['speed_range'][0])*cfg['reverse_s'] + cfg['reset_xy_range'][0]+cfg['boundary_margin']
    if reach >= cfg['patch_length']/2:
        raise ValueError('CTS flat patch cannot contain the nominal backward episode')
    return cfg


def requested_command(age, speed, cfg):
    moving = (age >= cfg['initial_hold_s']) & (age < cfg['initial_hold_s']+cfg['reverse_s'])
    return torch.where(moving, speed, 0.)


def stair_only(function):
    """Wrap a resolved manager callable without losing its entity parameters."""
    @wraps(function)
    def call(env, **kwargs):
        value = function(env, **kwargs)
        mask = getattr(env, 'cts_practice_mask', None)
        if mask is None:
            mask = env.cts_backward.mask  # Historical isolated backward tests.
        return torch.where(mask, torch.zeros_like(value), value)
    if hasattr(function, 'reset'):
        call.reset = function.reset
    return call


def explicit_pd_gains(env, mask):
    """Pinned landing uses explicit delayed PD; PhysX drive gains are zero."""
    entries = env.stair_quiet_landing._gains
    return (torch.cat([entry['base_kp'][mask] for entry in entries], -1),
            torch.cat([entry['base_kd'][mask] for entry in entries], -1))


class FlatCourseState(UserDict):
    """CTS-local view over the original cache; source teacher files stay frozen.

    Shared legacy helpers rebuild per-env geometry repeatedly. Intercept reads
    of those derived arrays, retaining the same backing dictionary (including
    fields installed after its initial assignment). Never change source tables.
    """
    def __init__(self, data, practice, *others):
        self.data, self.practice = data, practice
        self.practices = (practice, *others)

    def __getitem__(self, key):
        value = self.data[key]
        for practice in self.practices:
            practice.override_field(key, value)
        return value


class PracticeBase:
    origins_key = 'cts_flat_origins'

    def __init__(self, env, cfg, algorithm, *, validator=validate_practice, mask=None):
        self.env, self.cfg = env, deepcopy(validator(cfg))
        n, device = env.cfg.scene.num_envs, env.cfg.sim.device
        self.teacher = cohort_mask(n, algorithm['teacher_fraction'], algorithm['cohort_seed'], device)
        self.mask = practice_mask(self.teacher, cfg['fraction'], algorithm['cohort_seed']) if mask is None else mask
        self.speed = torch.zeros(n, device=device)
        self.push_at = torch.zeros(n, device=device)
        self.resets = torch.zeros(n, dtype=torch.long, device=device)
        self.pushes = torch.zeros(n, dtype=torch.long, device=device)
        self.completed = torch.zeros(n, dtype=torch.long, device=device)
        self.failures = torch.zeros(n, dtype=torch.long, device=device)
        self.profile_counts = torch.zeros(5, dtype=torch.long, device=device)
        self.vx_error_sum = torch.zeros(n, device=device)
        self.reverse_samples = torch.zeros(n, dtype=torch.long, device=device)
        self.height_error_max = 0.
        self.invalid_flat_rays = 0
        self.ledger_checks = 0

    @property
    def movement_duration(self):
        return self.cfg['reverse_s']

    def reset_origins(self, ids, types):
        origins = getattr(self.env.scene.terrain, self.origins_key)
        return origins[ids % origins.shape[0], types]

    def override_field(self, key, value):
        mask, cfg = self.mask, self.cfg
        if key in ('height', 'steps', 'goal_height'):
            value[mask] = 0
        elif key == 'width':
            value[mask] = self.env.cfg.stair_tail.tile_width
        elif key == 'spans':
            value[mask] = 0
            value[mask, 0, 0] = -cfg['patch_length']/2
            value[mask, 0, 1] = cfg['patch_length']/2
        elif key == 'round_trip':
            # The shared geometry kernel uses this flag to select explicit
            # finite spans. These spans/goal have zero height, with no stairs.
            value[mask] = True
        elif key in ('goal', 'ascent_end'):
            value[mask] = cfg['patch_length']/2

    def flat_state(self, state):
        for key in ('height','steps','goal_height','width','spans','round_trip','goal','ascent_end'):
            if key in state:
                self.override_field(key, state[key])
        return state

    def reset(self, env_ids):
        env, cfg = self.env, self.cfg
        ids = env_ids[self.mask[env_ids]]
        if not len(ids):
            return
        from task_support.stair_quiet_v1_width.settings import PROFILES
        terrain, robot = env.scene.terrain, env.scene['robot']
        # Cycle all contact domains independently of ascent material promotion.
        profile = (self.resets[ids]+ids) % len(PROFILES)
        types = terrain.terrain_types[ids]//len(PROFILES)*len(PROFILES)+profile
        terrain.terrain_types[ids] = types
        origin = self.reset_origins(ids, types)
        terrain.env_origins[ids] = origin
        env.scene.env_origins[ids] = origin
        position = robot.data.default_root_state[ids, :3].clone()+origin
        for axis in (0, 1):
            position[:, axis] += torch.empty(len(ids), device=env.device).uniform_(-cfg['reset_xy_range'][axis], cfg['reset_xy_range'][axis])
        yaw = torch.empty(len(ids), device=env.device).uniform_(-cfg['reset_yaw_range'], cfg['reset_yaw_range'])
        quat = torch.zeros(len(ids), 4, device=env.device)
        quat[:, 0], quat[:, 3] = torch.cos(yaw/2), torch.sin(yaw/2)
        robot.write_root_pose_to_sim(torch.cat((position, quat), -1), env_ids=ids)
        velocity = torch.zeros(len(ids), 6, device=env.device)
        velocity[:, :2].uniform_(-cfg['reset_velocity_range'], cfg['reset_velocity_range'])
        robot.write_root_velocity_to_sim(velocity, env_ids=ids)
        self.speed[ids] = torch.empty(len(ids), device=env.device).uniform_(*cfg['speed_range'])
        # Exercise disturbances even when a weak reverse policy falls before
        # a conventional long interval. Subsequent intervals are randomized.
        self.push_at[ids] = cfg['initial_hold_s']
        self.resets[ids] += 1
        self.profile_counts += torch.bincount(profile, minlength=len(PROFILES))
        self.flat_state(env._tail_course)

    def command(self, term, ids):
        ids = torch.arange(self.mask.numel(), device=self.mask.device)[ids]
        ids = ids[self.mask[ids]]
        age = self.env.episode_length_buf[ids]*self.env.step_dt
        vx = requested_command(age, self.speed[ids], self.cfg)
        term.vel_command_b[ids] = 0.
        term.vel_command_b[ids, 0] = vx
        term.is_standing_env[ids] = vx == 0
        term.is_heading_env[ids] = False

    def push(self):
        env, cfg = self.env, self.cfg
        age = env.episode_length_buf*env.step_dt
        due = self.mask & (age >= self.push_at) & (age < cfg['initial_hold_s']+self.movement_duration)
        ids = due.nonzero().flatten()
        if len(ids):
            robot = env.scene['robot']
            velocity = torch.cat((robot.data.root_lin_vel_w[ids], robot.data.root_ang_vel_w[ids]), -1).clone()
            velocity[:, :2] += torch.empty(len(ids), 2, device=env.device).uniform_(*cfg['push_velocity_range'])
            robot.write_root_velocity_to_sim(velocity, env_ids=ids)
            self.pushes[ids] += 1
            self.push_at[ids] = age[ids]+torch.empty(len(ids), device=env.device).uniform_(*cfg['push_interval_range_s'])

    def completed_episodes(self, ids):
        ids = ids[self.mask[ids] & (self.env.episode_length_buf[ids] > 0)]
        from task_support.stair_quiet_v1_width.signals import failure_mask
        if len(ids):
            done = self.env.termination_manager.terminated | self.env.termination_manager.time_outs
            ids = ids[done[ids]]
            self.completed[ids] += 1
            self.failures[ids] += failure_mask(self.env)[ids].long()

    def diagnostics(self):
        selected = self.mask
        return dict(selected=int(selected.sum()), teacher_selected=int((selected & self.teacher).sum()),
                    student_selected=int((selected & ~self.teacher).sum()),
                    actual_fraction=float(selected.float().mean()), resets=int(self.resets.sum()),
                    completed=int(self.completed.sum()), failures=int(self.failures.sum()), pushes=int(self.pushes.sum()),
                    profile_counts=self.profile_counts.tolist(),
                    reverse_samples=int(self.reverse_samples.sum()),
                    vx_mse=float(self.vx_error_sum.sum()/self.reverse_samples.sum().clamp_min(1)),
                    flat_ray_height_error_max=self.height_error_max,
                    invalid_flat_rays=self.invalid_flat_rays, stair_ledger_checks=self.ledger_checks)

    def randomization_receipt(self):
        robot = self.env.scene['robot']
        ids = self.mask.cpu().nonzero().flatten()
        if not len(ids):
            return {'enabled': False}
        masses = robot.root_physx_view.get_masses()[ids]
        com = robot.root_physx_view.get_coms()[ids, :, :3]
        stiffness, damping = explicit_pd_gains(self.env, self.mask)
        def span(value):
            # Compare the SAME body/joint across environments. Comparing
            # different bodies would report a range even without any DR.
            return float((value.amax(0)-value.amin(0)).abs().max())
        return dict(body_mass_min_max=[float(masses.min()), float(masses.max())],
                    body_com_abs_max=float(com.abs().max()),
                    same_body_mass_env_span_max=span(masses),
                    same_body_com_env_span_max=span(com),
                    same_joint_stiffness_env_span_max=span(stiffness),
                    same_joint_damping_env_span_max=span(damping),
                    gain_source='landing_explicit_delayed_pd_baseline',
                    stiffness_min_max=[float(stiffness.min()), float(stiffness.max())],
                    damping_min_max=[float(damping.min()), float(damping.max())],
                    source_events={mode: list(names) for mode, names in self.env.event_manager.active_terms.items()},
                    perception=vars(self.env.cfg.student_perception),
                    contact_binding=self.env.scene.terrain.contact_receipt)

    def snapshot(self):
        return {key: getattr(self, key).detach().cpu().clone() for key in
                ('mask','resets','pushes','completed','failures','profile_counts','vx_error_sum','reverse_samples')}

    def restore(self, saved):
        from task_support.student_v3_safe.environment import checked_tree
        checked = checked_tree(saved, self.snapshot())
        if not torch.equal(checked['mask'].to(self.mask.device), self.mask):
            raise ValueError('CTS backward cohort allocation changed')
        if any((value < 0).any() for key, value in checked.items() if key != 'mask'):
            raise ValueError('Negative CTS backward accounting')
        for key, value in checked.items():
            getattr(self, key).copy_(value.to(self.mask.device))


BackwardPractice = PracticeBase  # Legacy import/checkpoint compatibility only.


def practice_reward(env):
    practice, robot = env.cts_backward, env.scene['robot'].data
    cfg = practice.cfg
    command = env.command_manager.get_command('base_velocity')
    error = (robot.root_lin_vel_b[:, :2]-command[:, :2]).square().sum(-1)
    yaw_error = (robot.root_ang_vel_b[:, 2]-command[:, 2]).square()
    rate = (env.action_manager.action-env.action_manager.prev_action).square().mean(-1)
    torque = robot.applied_torque.square().mean(-1)
    from task_support.stair_quiet_v1_width.signals import failure_mask
    value = (cfg['linear_reward']*torch.exp(-error/cfg['tracking_std']**2)
             +cfg['yaw_reward']*torch.exp(-yaw_error/cfg['tracking_std']**2)
             -cfg['tilt_cost']*robot.projected_gravity_b[:, :2].square().sum(-1)
             -cfg['action_rate_cost']*rate-cfg['effort_cost']*torque
             -cfg['failure_cost']*failure_mask(env))
    moving = practice.mask & (command[:, 0] < 0)
    practice.vx_error_sum += error.detach()*moving
    practice.reverse_samples += moving.long()
    return torch.where(practice.mask, value, 0.)


def practice_boundary(env):
    p = env.cts_backward
    root = env.scene['robot'].data.root_pos_w-env.scene.env_origins
    outside = ((root[:, 0].abs() > p.cfg['patch_length']/2-p.cfg['boundary_margin'])
               | (root[:, 1].abs() > env.cfg.stair_tail.tile_width/2-p.cfg['boundary_margin']))
    return p.mask & (outside | ~torch.isfinite(root).all(-1) | (root[:, 2] < .1))


def practice_timeout(env):
    cfg = env.cts_backward.cfg
    return env.cts_backward.mask & (env.episode_length_buf*env.step_dt >= cfg['initial_hold_s']+cfg['reverse_s']+cfg['stop_s'])
