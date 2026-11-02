"""Bounded inference checks and complete two-optimizer CTS persistence."""
from copy import deepcopy
import json
from pathlib import Path
import time

import torch

from m20_student_v0.output_storage import atomic_torch_save, require_disk_space
from task_support.student_v3_safe.environment import course_snapshot, restore_course
from .model import sample_observations, teacher_observations


def parity_check(models, teacher_source, student_source, *, initialized=True):
    from task_support.stair_quiet_v1_safe.actor import SafeResidualActor
    from m20_student_v1.model_io import actor_from_state
    obs = sample_observations(8)
    for key, value in obs.items():
        value.copy_(torch.randn_like(value))
    obs['motion_control'][0] = 0.
    config = deepcopy(teacher_source['m20_training_manifest']['actor_model_config'])
    config.pop('class_name')
    reference = SafeResidualActor(teacher_observations(obs), {'actor': ['policy']}, 'actor', 16, **config)
    reference.load_state_dict(teacher_source['actor_state_dict'], strict=True)
    state = student_source['student_state_dict']
    student = actor_from_state(state, 789, {'hidden_dims': [state[f'mlp.{i}.weight'].shape[0] for i in (0,2,4)],
                                          'activation': 'elu', 'obs_normalization': False})
    export = torch.jit.script(models.student_export().cpu().eval())
    with torch.no_grad():
        teacher_mask = torch.ones(8, dtype=torch.bool)
        teacher_out = models.mean(obs, teacher_mask)[0]
        student_out = models.mean(obs, ~teacher_mask)[0]
        teacher_error = float((teacher_out-reference(teacher_observations(obs))).abs().max())
        student_error = float((student_out-student(obs['policy'])).abs().max())
        export_error = float((export(obs['policy'], obs['motion_control'])-student_out).abs().max())
        # Independent source critic forward, including original raw-command input.
        value = torch.cat((obs['teacher_critic'], obs['motion_control']), -1)
        source = teacher_source['critic_state_dict']
        for index in (0,2,4,6):
            value = torch.nn.functional.linear(value, source[f'mlp.{index}.weight'], source[f'mlp.{index}.bias'])
            if index != 6:
                value = torch.nn.functional.elu(value)
        critic_error = float((models.value(obs, models.teacher_latent(obs))-value.squeeze(-1)).abs().max())
        critic64 = deepcopy(models.critic).double()
        value64 = torch.cat((obs['teacher_critic'], obs['motion_control']), -1).double()
        for index in (0,2,4,6):
            value64 = torch.nn.functional.linear(value64, source[f'mlp.{index}.weight'].double(), source[f'mlp.{index}.bias'].double())
            if index != 6:
                value64 = torch.nn.functional.elu(value64)
        critic_error64 = float((critic64(obs['teacher_critic'].double(), obs['motion_control'].double(),
                                       models.teacher_latent(obs).double())-value64.squeeze(-1)).abs().max())
    if export_error > 1.e-6 or not all(__import__('math').isfinite(x) for x in (teacher_error, student_error, critic_error, export_error)):
        raise RuntimeError('CTS student export parity failed')
    # Expanding a GEMM with zero columns changes float32 accumulation order;
    # double precision distinguishes that rounding from a bad state migration.
    if initialized and (max(teacher_error, student_error) > 1.e-5 or critic_error64 > 1.e-9):
        raise RuntimeError('CTS warm start changed source actor/critic behavior')
    return {'teacher_source_max_action_error': teacher_error, 'student_source_max_action_error': student_error,
            'critic_source_max_value_error': critic_error, 'student_export_max_action_error': export_error,
            'critic_source_float64_max_value_error': critic_error64,
            'source_parity_required': initialized}


def replay_behavior_error(models, data, num_envs):
    """Recheck the collection ABI with its original batch shape.

    Changing CUDA GEMM from N to T*N rows changes float32 reduction order.
    With a narrow fixed exploration std this amplifies a few action ULPs into
    log-prob differences. Keep the original batch when verifying behavior;
    report the alternative full-batch error separately without changing PPO.
    """
    means = []
    for start in range(0, len(data['actions']), num_envs):
        ids = slice(start, start+num_envs)
        obs = {key: value[ids] for key, value in data['obs'].items()}
        mean, _ = models.mean(obs, data['teacher'][ids])
        means.append(mean)
    mean = torch.cat(means)
    distribution = torch.distributions.Normal(mean, data.get('std', models.log_std.exp()))
    return float((distribution.log_prob(data['actions']).sum(-1)-data['log_prob']).abs().max())


class CTSRunner:
    def __init__(self, env, algorithm, metadata, initialization, directory, resource_report):
        self.env, self.alg = env, algorithm
        self.metadata, self.initialization = deepcopy(metadata), initialization
        self.directory, self.resource_report = Path(directory), resource_report

    def all_terrain_course(self):
        cfg = self.metadata.get('config', {}).get('all_terrain')
        if not cfg or not cfg.get('enabled'):
            return None
        value = getattr(self.env.unwrapped, 'cts_all_terrain', None)
        if value is None:
            raise RuntimeError('CTS all-terrain metadata requires the live course controller')
        return value

    def save(self, name):
        required = self.resource_report['min_free_disk_bytes'] + self.resource_report['checkpoint_estimate_bytes']
        require_disk_space(self.directory, required)
        state = {'cts': deepcopy(self.metadata), 'initialization': self.initialization,
                 'algorithm': self.alg.save()}
        course = self.all_terrain_course()
        if course is not None:
            if course.update_id != self.alg.num_updates:
                raise RuntimeError('CTS all-terrain iteration ledger differs from the optimizer')
            state['all_terrain_course'] = course.snapshot()
        else:
            state.update(course=course_snapshot(self.env.unwrapped),
                         platform_practice=self.env.unwrapped.cts_platform.snapshot(),
                         motion_practice=self.env.unwrapped.cts_motion.snapshot())
        path = self.directory / name
        atomic_torch_save(state, path)
        return path

    def load(self, saved):
        from .entry import validate_metadata
        validate_metadata(saved['cts'])
        if saved['cts'] != self.metadata:
            raise ValueError('CTS runner metadata changed during resume')
        # Rebuild a fresh physics episode deterministically from saved RNG.
        # This is not a claim to restore an in-flight simulator trajectory.
        course = self.all_terrain_course()
        if course is not None:
            state = saved.get('all_terrain_course')
            if not isinstance(state, dict) or state.get('update_id') != saved['algorithm']['num_updates']:
                raise ValueError('CTS all-terrain checkpoint iteration/course contract changed')
            course.restore(state)
            self.alg.load(saved['algorithm'])
            with torch.inference_mode():
                self.env.reset()
            return
        else:
            self.env.unwrapped.cts_platform.restore(saved['platform_practice'])
            self.env.unwrapped.cts_motion.restore(saved['motion_practice'])
            restore_course(self.env.unwrapped, saved['course'])
        self.alg.load(saved['algorithm'])

    def export(self):
        from task_support.stair_quiet_v1.deployment import execution_contract
        execution_contract(self.metadata, required=True)
        folder = self.directory / 'exported'
        folder.mkdir(exist_ok=True)
        model = torch.jit.script(self.alg.models.student_export().cpu().eval())
        model.save(str(folder/'cts_student.pt'))
        (folder/'deployment.json').write_text(json.dumps({
            'deployment_family': 'm20_cts_student',
            'inputs': {'policy': ['batch',789], 'raw_command': ['batch',3]},
            'output': ['batch',16], 'actor_scope': 'nominal action before mandatory landing controller',
            'standalone_robot_controller': False, 'provenance': self.metadata,
        }, indent=2)+'\n')
        return folder

    def validate(self, steps):
        before = {k: v.clone() for k,v in self.alg.models.state_dict().items()}
        updates = self.alg.num_updates
        with torch.inference_mode():
            obs = self.env.get_observations()
        total_reward = torch.zeros(self.alg.num_envs, device=self.alg.device)
        ratio_error, episodes, checked_samples, gae_rollouts = 0., 0, 0, 0
        auxiliary_checks = []
        course = self.all_terrain_course()
        wall_hold = getattr(course, 'wall_hold', None)
        wall_hold_spawn = wall_hold.validate_initial() if wall_hold is not None else None
        wall_hold_samples = torch.zeros(2, dtype=torch.long, device=self.alg.device)
        wall_hold_command_max = 0.
        hold_spawn = course.stair_hold.validate_initial() if course is not None and course.stair_hold is not None else None
        from .all_terrain_config import COURSES
        num_modes = course.num_modes if course is not None else 9
        course_samples = torch.zeros(2,len(COURSES),num_modes,dtype=torch.long,device=self.alg.device)
        course_delays = torch.zeros(2,len(COURSES),num_modes,2,dtype=torch.long,device=self.alg.device)
        hold_samples = torch.zeros(2,2,dtype=torch.long,device=self.alg.device)
        hold_command_max = 0.
        bands = len(self.metadata['config']['platform_practice']['height_bands'])
        platform_samples = torch.zeros(bands,2,3,dtype=torch.long,device=self.alg.device)
        sample_frames = torch.zeros(2,dtype=torch.long,device=self.alg.device)
        from .motion import TERRAINS, COMMANDS
        motion_samples = torch.zeros(2,len(TERRAINS),len(COMMANDS),dtype=torch.long,device=self.alg.device)
        delay_samples = torch.zeros(*motion_samples.shape,2,dtype=torch.long,device=self.alg.device)
        from task_support.student_v1.environment import sensor_state
        from .perception import delay_diagnostics
        batched_ratio_error = 0.
        stop_samples = 0
        scale_error = 0.
        with torch.inference_mode():
            for _ in range(steps):
                action = self.alg.act(obs)
                stop_samples += int((obs['motion_control'].abs().amax(-1)<=1.e-4).sum())
                scale_error = max(scale_error, float((self.alg.pending['std']-self.alg.action_std(obs)).abs().max()))
                if scale_error != 0.:raise RuntimeError('CTS exploration collection/replay scale mismatch')
                if course is not None:
                    cell = ((~self.alg.teacher_mask).long()*len(COURSES)+course.course_ids)*num_modes+course.mode
                    original = torch.ones_like(course.teacher) if course.stair_hold is None else ~course.stair_hold.active
                    if wall_hold is not None:original &= ~wall_hold.active
                    course_samples.view(-1).add_(torch.bincount(cell[original],minlength=course_samples.numel()))
                    if course.stair_hold is not None and course.stair_hold.active.any():
                        ids=course.stair_hold.active.nonzero().flatten()
                        hold_cell=(~course.teacher[ids]).long()*2+(course.course_ids[ids]==2).long()
                        hold_samples.view(-1).add_(torch.bincount(hold_cell,minlength=4))
                        hold_command_max=max(hold_command_max,float(obs['motion_control'][ids].abs().max()))
                        if hold_command_max>1.e-4:raise RuntimeError('Stair hold delivered nonzero motion command')
                    if wall_hold is not None:
                        ids = wall_hold.active.nonzero().flatten()
                        wall_hold_samples.add_(torch.bincount((~course.teacher[ids]).long(), minlength=2))
                        wall_hold_command_max = max(wall_hold_command_max, float(obs['motion_control'][ids].abs().max()))
                        if wall_hold_command_max > 1.e-4:raise RuntimeError('Wall hold delivered nonzero motion command')
                    zero = sensor_state(self.env.unwrapped).zero_delivery.long()
                    course_delays.view(-1).add_(torch.bincount((cell*2+zero)[original],minlength=course_delays.numel()))
                    if checked_samples % (self.alg.num_envs*self.alg.num_steps_per_env) == 0:
                        course.check_rays()
                else:
                    practice = self.env.unwrapped.cts_platform
                    motion = self.env.unwrapped.cts_motion
                    raw = self.alg.pending['obs']['motion_control']
                    moving_ids = (motion.mask & (raw.abs().amax(-1) > 1.e-4)).nonzero().flatten()
                    cell = ((~self.alg.teacher_mask[moving_ids]).long()*len(TERRAINS)+motion.kind[moving_ids])*len(COMMANDS)+motion.mode[moving_ids]
                    motion_samples.view(-1).add_(torch.bincount(cell,minlength=motion_samples.numel()))
                    zero = sensor_state(self.env.unwrapped).zero_delivery[moving_ids].long()
                    delay_samples.view(-1).add_(torch.bincount(cell*2+zero,minlength=delay_samples.numel()))
                    sample_mask = self.env.unwrapped.cts_sample.mask
                    for teacher in (0,1):
                        sample_frames[teacher] += (sample_mask & (self.alg.teacher_mask == bool(teacher))).sum()
                    for band in range(bands):
                        for teacher in (0,1):
                            mask = practice.mask & (practice.band == band) & (self.alg.teacher_mask == bool(teacher))
                            command = self.alg.pending['obs']['motion_control'][:,0]
                            for direction,selected in enumerate((command == 0,command > 0,command < 0)):
                                platform_samples[band,teacher,direction] += (mask & selected).sum()
                if not torch.isfinite(action).all():
                    raise RuntimeError('Nonfinite CTS inference actions')
                # Check every transition, including a validation shorter than
                # one rollout; an unexecuted check must never report zero error.
                mean, _ = self.alg.models.mean(self.alg.pending['obs'], self.alg.teacher_mask)
                distribution = torch.distributions.Normal(mean, self.alg.action_std(self.alg.pending['obs']))
                error = (distribution.log_prob(action).sum(-1)-self.alg.pending['log_prob']).abs().max()
                ratio_error = max(ratio_error, float(error))
                checked_samples += self.alg.num_envs
                if ratio_error > 1.e-4:
                    raise RuntimeError('CTS sampled behavior probability mismatch')
                obs, reward, done, extras = self.env.step(action)
                self.alg.process_env_step(obs, reward, done, extras)
                total_reward += reward
                episodes += int(done.sum())
                if len(self.alg.rollout) == self.alg.num_steps_per_env:
                    self.alg.compute_returns(obs)
                    gae_rollouts += 1
                    data = self.alg._data()
                    mean, latent = self.alg.models.mean(data['obs'], data['teacher'])
                    distribution = torch.distributions.Normal(mean, self.alg.action_std(data['obs']))
                    if not torch.equal(distribution.scale, data['std']):
                        raise RuntimeError('CTS stored exploration scale differs from behavior observations')
                    error = (distribution.log_prob(data['actions']).sum(-1)-data['log_prob']).abs().max()
                    batched_ratio_error = max(batched_ratio_error, float(error))
                    ratio_error = max(ratio_error, replay_behavior_error(self.alg.models, data, self.alg.num_envs))
                    if not torch.isfinite(self.alg.returns).all() or ratio_error > 1.e-4:
                        raise RuntimeError('CTS stored behavior probability/returns mismatch: '
                            f'log_prob_max={ratio_error}, returns_finite={bool(torch.isfinite(self.alg.returns).all())}, '
                            f'mean_max={float(mean.abs().max())}, '
                            f'mean_recompute_error={float((mean-data["mean"]).abs().max())}')
                    # Exercise all enabled auxiliaries on real simulator data,
                    # with no backward pass or optimizer update.
                    selection = self.alg.auxiliary.prepare(self.alg.rollout)
                    teacher_latent = self.alg.models.teacher_latent(data['obs'])
                    target = self.alg.models.teacher_action(data['obs'], teacher_latent)
                    ids = torch.arange(min(len(target), self.alg.cfg['student_batch_size']), device=self.alg.device)
                    loss, diagnostics = self.alg.auxiliary.loss(data, ids, target, selection)
                    from .straight import straight_action_loss, straight_teacher_actions
                    clean_target = straight_teacher_actions(self.alg.models,data['obs'],teacher_latent)
                    straight_loss,straight_diagnostics = straight_action_loss(self.alg.models,data,clean_target,self.alg.straight_cfg)
                    if not torch.isfinite(straight_loss):raise RuntimeError('Nonfinite CTS straight inference validation')
                    diagnostics.update(straight_diagnostics)
                    from .clutter_consistency import loss as clutter_loss
                    clutter, clutter_metrics = clutter_loss(self.alg.models,data,ids,selection['regular'],
                        (self.alg.all_terrain_cfg or {}).get('clutter_consistency'),self.alg.action_safety)
                    if not torch.isfinite(clutter):raise RuntimeError('Nonfinite CTS clutter inference validation')
                    diagnostics.update(clutter_metrics)
                    diagnostics['clutter_loss'] = float(clutter)
                    from .wall_stability import settings as wall_settings, student_loss as wall_loss
                    wall, wall_metrics = wall_loss(self.alg.models, data, ids, wall_settings(self.alg.all_terrain_cfg), self.alg.action_safety)
                    if not torch.isfinite(wall):raise RuntimeError('Nonfinite CTS wall stability validation')
                    diagnostics.update(wall_metrics)
                    diagnostics['wall_stability_loss'] = float(wall)
                    ppo_retention, ppo_diagnostics = self.alg.auxiliary.retention_loss(detach_student=True)
                    if not torch.isfinite(loss) or not torch.isfinite(ppo_retention):
                        raise RuntimeError('Nonfinite CTS auxiliary inference validation')
                    auxiliary_checks.append({'loss': float(loss), **diagnostics,
                        'ppo_retention_loss': float(ppo_retention),
                        **{'ppo_'+key: value for key, value in ppo_diagnostics.items()},
                        'temporal_pair_count': selection['pairs_current'].numel(),
                        **self.alg.auxiliary.evaluate()})
                    self.alg.clear_rollout()
        self.alg.clear_rollout()
        if wall_hold is not None and wall_hold.handover is not None:
            if not (wall_hold.totals['handover_entries'][2:] > 0).all():
                raise RuntimeError('C handover validation did not enter policy in both cohorts')
        if any(not torch.equal(value, self.alg.models.state_dict()[key]) for key,value in before.items()):
            raise RuntimeError('Inference validation changed CTS weights')
        delay_receipt = delay_diagnostics(sensor_state(self.env.unwrapped),self.alg.teacher_mask)
        sensors = sensor_state(self.env.unwrapped)
        recovery_receipt = ({key: float(value) for key, value in sensors.diagnostics().items()}
                            if hasattr(sensors, 'recovery_cfg') else None)
        all_terrain_receipt = None
        if course is not None:
            with torch.inference_mode():
                course.check_rays()
            course.validate_actual_contacts()
            if course.ray_checks < 1:
                raise RuntimeError('CTS all-terrain validation did not check any physical rays')
            if (course.ray_checks_by_course < 1).any():
                missing = [name for name,count in zip(COURSES,course.ray_checks_by_course.tolist()) if count < 1]
                raise RuntimeError('All-terrain physical rays unchecked for courses: '+str(missing))
            all_terrain_receipt = course.diagnostics()
        checkpoint = self.save('validation_checkpoint.pt')
        self.load(torch.load(checkpoint, map_location='cpu', weights_only=True))
        if any(not torch.equal(value, self.alg.models.state_dict()[key]) for key,value in before.items()):
            raise RuntimeError('CTS checkpoint reload changed weights')
        folder = self.export()
        with torch.inference_mode():
            sample = self.env.get_observations()
            script = torch.jit.load(str(folder/'cts_student.pt'))
            policy, command = sample['policy'].cpu(), sample['motion_control'].cpu()
            expected_gpu = self.alg.models.student_export()(sample['policy'],sample['motion_control']).cpu()
            # Verify CPU serialization against the same CPU backend. Record
            # GPU/CPU arithmetic differences independently of export fidelity.
            expected = self.alg.models.student_export().cpu()(policy, command)
            actual = script(sample['policy'].cpu(), sample['motion_control'].cpu())
            export_error = float((actual-expected).abs().max())
            gpu_cpu_export_error = float((actual-expected_gpu).abs().max())
            if export_error > 1.e-5:
                raise RuntimeError(f'Saved CTS export changed actual environment actions: {export_error}')
        import resource
        result = {'task': self.metadata['task'], 'steps': steps, 'num_envs': self.alg.num_envs,
                  'teacher_environments': int(self.alg.teacher_mask.sum()),
                  'student_environments': int((~self.alg.teacher_mask).sum()),
                  'observation_shapes': {k:list(v.shape) for k,v in sample.items()},
                  'optimizer_updates': self.alg.num_updates-updates, 'weights_unchanged': True,
                  'behavior_log_prob_max_error': ratio_error, 'export_max_action_error': export_error,
                  'alternative_batch_log_prob_max_error': batched_ratio_error,
                  'gpu_cpu_export_max_action_error': gpu_cpu_export_error,
                  'behavior_log_prob_checked_samples': checked_samples,
                  'exploration_stop_samples': stop_samples,
                  'exploration_moving_samples': checked_samples-stop_samples,
                  'exploration_stop_scale': self.alg.cfg.get('stop_exploration_scale', 1.),
                  'exploration_scale_max_error': scale_error,
                  'gae_checked_rollouts': gae_rollouts,
                  'auxiliary_checked_rollouts': len(auxiliary_checks), 'auxiliary': auxiliary_checks,
                  'checkpoint_reload': 'passed', 'completed_episodes': episodes,
                  'physics_substeps': self.env.unwrapped._sim_step_counter,
                  'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                  'trained': False, 'performance_improvement': 'not_measured'}
        result['map_recovery'] = recovery_receipt
        result['perception_timing'] = {
            key:getattr(self.env.unwrapped.cfg.student_perception,key)
            for key in ('front_only','map_update_period_s','min_delay_s','max_delay_s','zero_delay_probability','action_delay_steps','continuous_delay')}
        if course is not None:
            result['all_terrain'] = all_terrain_receipt
            result['all_terrain']['physical_materials'] = self.env.unwrapped.scene.terrain.contact_validation
            result['all_terrain']['rollout_frames_cohort_course_command'] = course_samples.tolist()
            result['all_terrain']['every_course_sampled_by_both_policies'] = bool((course_samples.sum(-1) > 0).all())
            result['all_terrain']['course_names'] = list(COURSES)
            result['all_terrain']['reward_terms'] = {name:{'weight':float(self.env.unwrapped.reward_manager.get_term_cfg(name).weight),
                'callable':self.env.unwrapped.reward_manager.get_term_cfg(name).func.__module__+'.'+self.env.unwrapped.reward_manager.get_term_cfg(name).func.__name__}
                for name in self.env.unwrapped.reward_manager.active_terms}
        else:
            result['platform_practice'] = self.env.unwrapped.cts_platform.diagnostics()
            result['motion_practice'] = self.env.unwrapped.cts_motion.diagnostics()
            result['straight_practice'] = self.env.unwrapped.cts_motion.straight_diagnostics()
            result['motion_practice']['rollout_cell_frames_teacher_student'] = motion_samples.tolist()
            result['motion_practice']['every_cell_sampled_by_both_policies'] = bool((motion_samples > 0).all())
            result['motion_practice']['perception_timing'] = {
                key:getattr(self.env.unwrapped.cfg.student_perception,key)
                for key in ('front_only','map_update_period_s','min_delay_s','max_delay_s','zero_delay_probability','action_delay_steps')}
        sensor = sensor_state(self.env.unwrapped)
        from task_support.student_v1.environment import capture_hits
        hits = capture_hits(self.env.unwrapped)
        if hits.shape[1] != len(sensor.capture_grid) or sensor.last_visibility.shape[-1] != 187:
            raise RuntimeError('CTS expanded capture/output sensor contract mismatch')
        result['map_capture'] = {'padding_m':sensor.cfg.map_capture_padding_m,
            'source_grid_shape':list(sensor.capture_shape),'source_hit_shape':list(hits.shape),
            'packet_shape':list(sensor.packet_z.shape),'actor_points':187,
            'teacher_ray_shape':list(self.env.unwrapped.scene.sensors['height_scanner'].data.ray_hits_w.shape)}
        if sensor.cfg.support_surface_visibility:
            capture = self.env.unwrapped.scene.sensors['height_scanner_capture'].data
            normals = capture.ray_normals_w
            finite_hits = torch.isfinite(capture.ray_hits_w).all(-1)
            measured = torch.isfinite(normals).all(-1) & (normals.norm(dim=-1) > 1.e-6)
            if normals.shape != capture.ray_hits_w.shape or not measured[finite_hits].all():
                raise RuntimeError('Student support normals are missing for measured acquisition hits')
            result['support_surface_visibility'] = dict(enabled=True, normal_shape=list(normals.shape),
                finite_hits=int(finite_hits.sum()), measured_normals=int(measured.sum()),
                actor_channels_added=0, teacher_scanner_changed=False)
        if hold_spawn is not None:
            result['stair_hold_spawn'] = hold_spawn
            result['stair_hold_rollout_frames'] = hold_samples.tolist()
            result['stair_hold_command_max'] = hold_command_max
            result['stair_hold_practice'] = course.stair_hold.diagnostics()
        if wall_hold_spawn is not None:
            result['wall_hold_spawn'] = wall_hold_spawn
            result['wall_hold_rollout_frames'] = wall_hold_samples.tolist()
            result['wall_hold_command_max'] = wall_hold_command_max
            result['wall_hold_practice'] = wall_hold.diagnostics()
        result['map_delay_mixture'] = delay_receipt
        if course is not None:
            result['map_delay_mixture']['frames_cohort_course_command_delayed_zero'] = course_delays.tolist()
        else:
            result['map_delay_mixture']['motion_frames_cohort_terrain_command_delayed_zero'] = delay_samples.tolist()
            result['motion_randomization'] = self.env.unwrapped.cts_motion.randomization_receipt()
            result['sample_stairs'] = self.env.unwrapped.cts_sample.diagnostics()
            from task_support.student_v3_safe.terrain_course import course_state
            from .terrain import geometries
            state = course_state(self.env.unwrapped)
            table = geometries(self.env.unwrapped.cfg.student_terrain_course,self.metadata['config']['legacy_stairs'])
            from task_support.stair_quiet_v1_width.settings import PROFILES
            expected_steps = torch.tensor([g.steps for g in table for _ in PROFILES],device=state['steps_table'].device)
            if not torch.equal(state['steps_table'],expected_steps):
                raise RuntimeError('CTS physical geometry and tracker step counts differ')
            result['course_geometry'] = {'steps_table':state['steps_table'].tolist(),
                'goal_table':state['goal_table'].tolist(),'goal_height_table':state['goal_height_table'].tolist(),
                'ascent_end_table':state['ascent_end_table'].tolist()}
            result['sample_stairs']['rollout_student_teacher_samples'] = sample_frames.tolist()
            result['sample_stairs']['reward_terms'] = {name:{'weight':float(self.env.unwrapped.reward_manager.get_term_cfg(name).weight),
                'callable':self.env.unwrapped.reward_manager.get_term_cfg(name).func.__module__+'.'+self.env.unwrapped.reward_manager.get_term_cfg(name).func.__name__}
                for name in self.env.unwrapped.reward_manager.active_terms}
            result['platform_practice']['rollout_samples_band_teacher_direction'] = platform_samples.tolist()
            result['platform_randomization'] = self.env.unwrapped.cts_platform.randomization_receipt()
        if str(self.alg.device).startswith('cuda'):
            result['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated(self.alg.device)
        (self.directory/'validation.json').write_text(json.dumps(result, indent=2)+'\n')
        print('CTS_VALIDATION='+json.dumps(result), flush=True)
        return result

    def learn(self, iterations):
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(self.directory))
        with torch.inference_mode():
            obs = self.env.get_observations()
        update_in_progress = False
        try:
            # Always retain one completed joint state before any optimization.
            last_checkpoint = self.save(f'model_{self.alg.num_updates}.pt')
            for _ in range(iterations):
                course = self.all_terrain_course()
                if course is not None:
                    course.update_id = self.alg.num_updates
                start = time.monotonic()
                episode_reward = torch.zeros(self.alg.num_envs, device=self.alg.device)
                with torch.inference_mode():
                    for _ in range(self.alg.num_steps_per_env):
                        action = self.alg.act(obs)
                        obs, reward, done, extras = self.env.step(action)
                        self.alg.process_env_step(obs, reward, done, extras)
                        episode_reward += reward
                    self.alg.compute_returns(obs)
                update_in_progress = True
                metrics = self.alg.update()
                update_in_progress = False
                metrics.update(teacher_reward=float(episode_reward[self.alg.teacher_mask].mean()),
                               student_reward=float(episode_reward[~self.alg.teacher_mask].mean()),
                               iteration_seconds=time.monotonic()-start)
                step = self.alg.num_updates
                if course is not None:
                    course.update_id = step
                    for key,value in course.diagnostics().items():
                        writer.add_scalar('CTSAllTerrain/'+key, value, step)
                for key,value in metrics.items():
                    writer.add_scalar('CTS/'+key, value, step)
                for key,value in extras.get('log',{}).items():
                    if isinstance(value, (float,int)) or isinstance(value,torch.Tensor) and value.numel()==1:
                        writer.add_scalar(key, float(value), step)
                print('CTS_UPDATE='+json.dumps({'iteration':step,**metrics}), flush=True)
                if step % self.metadata['config']['save_interval'] == 0:
                    last_checkpoint = self.save(f'model_{step}.pt')
                # The output disk is shared with the running legacy job.
                require_disk_space(self.directory, self.resource_report['min_free_disk_bytes']+
                                   self.resource_report['checkpoint_estimate_bytes'])
            self.save(f'model_{self.alg.num_updates}.pt')
            self.export()
        except (KeyboardInterrupt, Exception):
            if update_in_progress:
                # A half-finished PPO/distillation update is not a valid joint
                # checkpoint. Resume the last atomically saved complete state.
                print('CTS_INCOMPLETE_UPDATE: resume '+str(last_checkpoint), flush=True)
            else:
                # Partial sampling has not touched weights/optimizer states.
                self.alg.clear_rollout()
                try:
                    self.save(f'recovery_{self.alg.num_updates}.pt')
                except Exception as error:
                    print('CTS_RECOVERY_SAVE_FAILED: '+str(error), flush=True)
            raise
        finally:
            writer.close()
