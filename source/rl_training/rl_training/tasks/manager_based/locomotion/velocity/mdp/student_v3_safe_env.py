"""Complete Safe dynamics, sensor-only student, privileged teacher labels."""
import torch

from .stair_quiet_v1_safe_env import M20StairQuietV1SafeEnv
from task_support.student_v3_safe import environment as glue


class M20StudentV3SafeEnv(M20StairQuietV1SafeEnv):
    def __init__(self, cfg, **kwargs):
        cfg.student_perception.validate()
        super().__init__(cfg, **kwargs)
        expected = {"policy": (789,), "critic": (789,), "teacher": (244,),
                    "motion_control": (3,), "locomotion_scan": (187,), "safety": (48,)}
        if any(self.observation_manager.group_obs_dim.get(k) != v for k, v in expected.items()):
            raise ValueError("student_v3_safe observation wiring changed")
        self._student_failed = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        from task_support.student_v3_safe.terrain_signals import StudentTerrainTracker
        self.tail_tracker=StudentTerrainTracker(self.num_envs,self.device,cfg.stair_tail)
        self.tail_tracker.reset(slice(None),self.scene['robot'].data.root_pos_w[:,0]-self.scene.env_origins[:,0])

    def _after_quiet_physics_sample(self):
        from task_support.student_v3_safe.terrain_observations import physical_sample
        self.tail_tracker.update(**physical_sample(self))

    def _reset_idx(self, env_ids):
        # Terminations have already been computed here; the trackers are still
        # the completed episode's trackers. Capture failures BEFORE reset.
        if hasattr(self, "_student_failed"):
            self._student_failed[env_ids] = glue.completed_failure(self)[env_ids]
        from task_support.student_v3_safe.terrain_signals import StudentTerrainTracker
        from task_support.student_v3_safe.terrain_course import record_episodes
        skip=getattr(self,'_tail_skip_episode_record',False)
        metrics=record_episodes(self,env_ids) if isinstance(getattr(self,'tail_tracker',None),StudentTerrainTracker) and not skip else {}
        self._tail_skip_episode_record=True
        try:
            super()._reset_idx(env_ids)
        finally:
            self._tail_skip_episode_record=skip
        self.extras.setdefault('log',{}).update(metrics)
        tracker = getattr(self, "student_diagonal_tracker", None)
        if tracker is not None:
            x = self.scene["robot"].data.root_pos_w[:, 0]-self.scene.env_origins[:, 0]
            tracker.reset(env_ids, x[env_ids])

    def step(self, action):
        self._student_diagonal_step = None
        glue.motion.student_observation(self)
        tags = glue.sampling_tags(self)
        self._student_failed.zero_()
        applied = glue.common.action_delay(self).push(action.to(self.device))
        obs, reward, terminated, timed_out, extras = super().step(applied)
        touchdown = self.stair_quiet_step_events.sum(-1) > 0
        tags["phase"] = torch.where(touchdown & (tags["phase"] != 3), 2, tags["phase"])
        # The legacy field name denotes FAILURE for the teacher-decay gate.
        # Safe success is a genuine termination, but must never count as failure.
        tags["terminated"] = self._student_failed.clone()
        extras["student_v1"] = tags
        extras.setdefault("log", {}).update(glue.sensor_metrics(self, obs))
        from task_support.student_v3_safe.start_stop import environment_masks
        selected, paused = environment_masks(self)
        extras["log"].update({"StudentPractice/selected_fraction": selected.float().mean(),
                              "StudentPractice/scheduled_pause_fraction": paused.float().mean()})
        from task_support.student_v3_safe.gait_reward import step_evidence
        evidence = step_evidence(self)
        extras["student_diagonal"] = evidence
        extras["log"].update({"StudentGait/reward_rate": evidence["reward"].mean(),
                              "StudentGait/eligible_fraction": evidence["eligible"].float().mean(),
                              "StudentGait/alternation_fraction": evidence["switch"].float().mean()})
        for key in ("safe", "diagonal", "sustained", "cycling"):
            extras["log"][f"StudentGait/{key}_eligible_fraction"] = (evidence[key] & evidence["eligible"]).float().mean()
        return obs, reward, terminated, timed_out, extras
