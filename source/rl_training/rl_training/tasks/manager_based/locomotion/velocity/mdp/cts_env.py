"""Concurrent teacher/student learning on the existing Safe student task."""
import torch
from .student_v3_safe_env import M20StudentV3SafeEnv


class M20CTSEnv(M20StudentV3SafeEnv):
    """Keep sensor-only actor inputs and expose the pretrained critic's inputs."""

    def __setattr__(self, name, value):
        if name == '_tail_course':
            from task_support.cts.practice import FlatCourseState
            if not isinstance(value, FlatCourseState):
                value = FlatCourseState(value, self.cts_platform,self.cts_motion,self.cts_sample)
        super().__setattr__(name, value)

    def __init__(self, cfg, **kwargs):
        from task_support.cts.practice import stair_only
        self.cfg = cfg
        from task_support.cts.platform import PlatformPractice
        self.cts_platform = PlatformPractice(self,cfg.cts_platform_practice,cfg.cts_cohort_config,torch.zeros(cfg.scene.num_envs,dtype=torch.bool,device=cfg.sim.device))
        from task_support.cts.motion import MotionPractice
        self.cts_motion = MotionPractice(self,cfg.cts_motion_practice,cfg.cts_cohort_config,
                                        self.cts_platform.mask,cfg.cts_straight_practice)
        self.cts_practice_mask = self.cts_platform.mask | self.cts_motion.mask
        from task_support.cts.sample_stairs import SampleStairsCourse
        self.cts_sample = SampleStairsCourse(self,cfg.cts_sample_stairs)
        super().__init__(cfg, **kwargs)
        self._cts_motion_leg_ids = self.scene['robot'].find_joints(cfg.leg_joint_names,preserve_order=True)[0]
        from task_support.cts.sample_stairs import SampleTerrainTracker
        self.tail_tracker = SampleTerrainTracker(self.num_envs,self.device,cfg.stair_tail,self.cts_sample)
        self.tail_tracker.reset(slice(None),self.scene['robot'].data.root_pos_w[:,0]-self.scene.env_origins[:,0])
        if self.observation_manager.group_obs_dim.get("teacher_critic") != (285,):
            raise ValueError("CTS requires the original 285-dimensional privileged critic group")
        # Manager parameters are already resolved to real entity indices here.
        # Preserve generic physical failures, replacing ascent-only termination
        # and all old reward objectives only on the selected flat environments.
        for name in self.reward_manager.active_terms:
            if name not in ('cts_platform','cts_motion'):
                term = self.reward_manager.get_term_cfg(name)
                term.func = stair_only(term.func)
                self.reward_manager.set_term_cfg(name, term)
        for name in self.termination_manager.active_terms:
            if name.startswith(('tail_', 'stair_')) or name == 'terrain_out_of_bounds':
                term = self.termination_manager.get_term_cfg(name)
                term.func = stair_only(term.func)
                self.termination_manager.set_term_cfg(name, term)

    def _reset_idx(self, env_ids):
        from task_support.student_v3_safe.terrain_course import record_episodes
        from task_support.student_v3_safe.terrain_signals import StudentTerrainTracker
        skip = getattr(self, '_tail_skip_episode_record', False)
        metrics = {}
        failed = None
        if isinstance(getattr(self, 'tail_tracker', None), StudentTerrainTracker) and not skip:
            from task_support.stair_quiet_v1_width.signals import failure_mask
            failed = failure_mask(self)[env_ids].clone()
            self.cts_platform.completed_episodes(env_ids)
            self.cts_motion.completed_episodes(env_ids)
            stairs = env_ids[~self.cts_practice_mask[env_ids]]
            done = self.termination_manager.terminated | self.termination_manager.time_outs
            expected = ((self.episode_length_buf[stairs] > 0) & done[stairs]).sum()
            before = self._tail_course['episodes'].sum().clone()
            metrics = record_episodes(self, stairs)
            self.cts_sample.advance_materials()
            if self._tail_course['episodes'].sum()-before != expected:
                raise RuntimeError('CTS practice episodes contaminated the stair course ledger')
            self.cts_platform.ledger_checks += 1
        self._tail_skip_episode_record = True
        try:
            super()._reset_idx(env_ids)
        finally:
            self._tail_skip_episode_record = skip
        if failed is not None:
            selected = self.cts_practice_mask[env_ids]
            self._student_failed[env_ids[selected]] = failed[selected]
        self.extras.setdefault('log', {}).update(metrics)

    def step(self, action):
        # Capture the state that generated action, before physics/reset can
        # move a terminal student into another episode.
        straight = self.cts_motion.straight_active()
        self.cts_motion.push()
        self.cts_platform.begin_step()
        self.cts_platform.push()
        result = super().step(action)
        self.extras['cts_straight'] = straight
        self.extras.setdefault('log',{}).update({'CTSStraight/'+k:v for k,v in self.cts_motion.straight_diagnostics().items()})
        self.extras.setdefault('log',{}).update({'CTSSampleStairs/'+key:value for key,value in self.cts_sample.diagnostics().items()})
        self.cts_platform.observe_sensors(result[0]['policy'])
        self.cts_motion.observe_sensors(result[0]['policy'])
        for name,practice in (('CTSPlatform',self.cts_platform),('CTSMotion',self.cts_motion)):
            for key, value in practice.diagnostics().items():
                if isinstance(value, (int, float)):
                    self.extras.setdefault('log', {})[name+'/'+key] = value
                elif isinstance(value,list) and all(isinstance(x,(int,float)) for x in value):
                    for index,item in enumerate(value):
                        self.extras.setdefault('log', {})[name+'/'+key+'/'+str(index)] = item
        from task_support.student_v1.environment import sensor_state
        from task_support.cts.perception import delay_diagnostics
        self.extras.setdefault('log',{}).update({'CTSDelay/'+k:v for k,v in
            delay_diagnostics(sensor_state(self),self.cts_motion.teacher,scalar_only=True).items()})
        return result

    def _after_quiet_physics_sample(self):
        super()._after_quiet_physics_sample()
        self.cts_platform.sample_physics()


from .stair_quiet_v1_env import M20StairQuietV1StudentEnv


class M20CTSAllTerrainEnv(M20StairQuietV1StudentEnv):
    """Shared physical cohorts with one versioned course scheduler."""
    def __init__(self,cfg,**kwargs):
        self.cts_all_terrain = None
        from task_support.cts.map_recovery import install as install_recovery
        install_recovery(self, cfg)
        super().__init__(cfg,**kwargs)
        if self.observation_manager.group_obs_dim.get('teacher_critic') != (285,):
            raise ValueError('All-terrain critic ABI changed')
        self._cts_failed_step = torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
        from task_support.cts.joint_limits_isaac import install
        install(self)

    def _reset_idx(self,env_ids):
        from task_support.cts.all_terrain_course import course
        state = course(self)
        if hasattr(self,'_cts_failed_step'):
            self._cts_failed_step[env_ids] = state.failed[env_ids] | (
                self.termination_manager.time_outs[env_ids] & ~state.success[env_ids])
        state.record(env_ids)
        super()._reset_idx(env_ids)
        guard = getattr(self, 'cts_limit_monitor', None)
        if guard is not None:
            guard.reset(env_ids)

    def step(self,action):
        from task_support.cts.all_terrain_course import course
        state = course(self)
        tags = state.tags()
        self._cts_failed_step.zero_()
        from task_support.cts.joint_limits_isaac import begin
        result = super().step(begin(self, action))
        self.extras.setdefault('log', {}).update(getattr(self, '_cts_limit_step_metrics', {}))
        self.extras.update(tags)
        self.extras['student_v1']['terminated'] = self._cts_failed_step.clone()
        if self.common_step_counter % 24 == 0:
            self.extras.setdefault('log',{}).update({'CTSAllTerrain/'+k:v for k,v in state.diagnostics().items()})
            sensors = getattr(self, '_m20_student_v1_state', None)
            if hasattr(sensors, 'recovery_cfg'):
                self.extras['log'].update({'CTSMapRecovery/'+k:v for k,v in sensors.diagnostics().items()})
            if 'startup_stability' in state.cfg:
                from task_support.cts.stability import stability_diagnostics
                self.extras['log'].update(stability_diagnostics(self))
        return result

    def _after_quiet_physics_sample(self):
        from task_support.cts.joint_limits_isaac import sample
        sample(self)

    def _finish_quiet_policy_step(self):
        from task_support.cts.joint_limits_isaac import snapshot
        snapshot(self)
