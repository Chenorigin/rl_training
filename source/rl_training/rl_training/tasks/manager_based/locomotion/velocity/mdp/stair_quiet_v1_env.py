"""Manager-based M20 dynamics with paired pre/post-physics impact sampling."""
import torch

from isaaclab.envs import ManagerBasedRLEnv
from task_support.stair_quiet_v1.signals import (
    LEGS, StairQuietTracker, scheduled_quiet_scale, validate_usd_wheel_geometry, wheel_axis_world,
)


class M20StairQuietV1Env(ManagerBasedRLEnv):
    """Same manager ordering; task-local landing control at each physics step."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        robot, sensor = self.scene["robot"], self.scene.sensors["contact_forces"]
        expected = [f"{leg}_wheel" for leg in LEGS]
        self._quiet_body_ids, names = robot.find_bodies(expected, preserve_order=True)
        self._quiet_contact_ids, contact_names = sensor.find_bodies(expected, preserve_order=True)
        if names != expected or contact_names != expected:
            raise RuntimeError("Stair quiet wheel body order must be FL/FR/HL/HR")
        dims = self.observation_manager.group_obs_dim
        student = getattr(cfg, "stair_quiet_student", False)
        expected_dims = (("policy", 789), ("critic", 789), ("teacher", 244), ("motion_control", 3)) if student else (
            ("policy", 244), ("critic", 285), ("motion_control", 3))
        if any(dims.get(name) != (size,) for name, size in expected_dims):
            raise RuntimeError(f"Stair quiet baseline observation contract changed: {dims}")
        if self.action_manager.total_action_dim != 16:
            raise RuntimeError("Stair quiet requires the baseline 16 actions")
        spawn_scale = getattr(cfg.scene.robot.spawn, "scale", None)
        if spawn_scale is not None and (len(spawn_scale) != 3 or any(
                not abs(value - 1.) <= cfg.stair_quiet.geometry_tolerance for value in spawn_scale)):
            raise ValueError("Stair quiet contact geometry requires unit robot spawn scale")
        self.stair_quiet_geometry = validate_usd_wheel_geometry(
            cfg.scene.robot.spawn.usd_path, cfg.stair_quiet)
        self.stair_quiet_tracker = StairQuietTracker(self.num_envs, self.device, cfg.stair_quiet)
        from task_support.stair_quiet_v1.isaac_landing import IsaacLandingAdapter
        self.stair_quiet_landing = IsaacLandingAdapter(self, self._quiet_body_ids, self._quiet_contact_ids)

    def step(self, action):
        self.extras.pop("log", None)
        if getattr(self.cfg, "stair_quiet_student", False):
            # Distillation does not optimize environment rewards. Report full
            # physical costs without inventing a PPO curriculum iteration.
            self.stair_quiet_iteration, self.stair_quiet_reward_scale = 0, 1.
        else:
            self.stair_quiet_iteration, self.stair_quiet_reward_scale = scheduled_quiet_scale(
                getattr(self, "_m20_stair_quiet_schedule", None), self.common_step_counter,
                self.stair_quiet_tracker.settings)
        # Keep this sequence aligned with ManagerBasedRLEnv.step: rewards before
        # resets, commands/events after resets, observation history last.
        self.stair_quiet_tracker.begin_step()
        landing = getattr(self, "stair_quiet_landing", None)
        if landing is not None:
            landing.begin_step()
        self.action_manager.process_action(action.to(self.device))
        self.recorder_manager.record_pre_step()
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        robot, sensor = self.scene["robot"], self.scene.sensors["contact_forces"]
        start_xy = robot.data.root_pos_w[:, :2].clone()
        for _ in range(self.cfg.decimation):
            # Snapshot all kinematics BEFORE the impulse changes them. The
            # post-step net normal force only supplies the estimated normal.
            # Support-point geometry is verified from the active USD; it is
            # still an aggregate-contact approximation, not a contact sensor.
            pre_velocity = robot.data.body_lin_vel_w[:, self._quiet_body_ids].clone()
            pre_omega = robot.data.body_ang_vel_w[:, self._quiet_body_ids].clone()
            pre_quaternion = robot.data.body_link_quat_w[:, self._quiet_body_ids].clone()
            # Validate this quaternion together with all physics samples in the
            # tracker, avoiding a separate device-to-host branch here.
            axis = wheel_axis_world(
                pre_quaternion, self.stair_quiet_tracker.settings, validate=False)
            link_minus_com = (robot.data.body_link_pos_w[:, self._quiet_body_ids]
                              - robot.data.body_com_pos_w[:, self._quiet_body_ids]).clone()
            self._sim_step_counter += 1
            self.action_manager.apply_action()
            if landing is not None:
                landing.apply(self.physics_dt)
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.recorder_manager.record_post_physics_decimation_step()
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)
            self.stair_quiet_tracker.update(
                pre_velocity, sensor.data.net_forces_w[:, self._quiet_contact_ids], self.physics_dt,
                pre_omega=pre_omega, wheel_axis=axis, link_minus_com=link_minus_com,
                wheel_quaternion=pre_quaternion,
            )
            self._after_quiet_physics_sample()
        self.stair_quiet_tracker.distance += torch.linalg.vector_norm(
            robot.data.root_pos_w[:, :2] - start_xy, dim=-1,
        )
        # A reset below clears episode state, not the evidence returned for the
        # just-completed policy step.
        self.stair_quiet_step_events = self.stair_quiet_tracker.step_events.clone()
        self._finish_quiet_policy_step()
        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
        if len(self.recorder_manager.active_terms) > 0:
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.recorder_manager.record_pre_reset(reset_env_ids)
            self._reset_idx(reset_env_ids)
            if self.sim.has_rtx_sensors():
                # Isaac Lab versions renamed the boolean to a render count.
                count = getattr(self.cfg, "num_rerenders_on_reset",
                                int(getattr(self.cfg, "rerender_on_reset", False)))
                for _ in range(count):
                    self.sim.render()
            self.recorder_manager.record_post_reset(reset_env_ids)
        self.command_manager.compute(dt=self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        self.obs_buf = self.observation_manager.compute(update_history=True)
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def _after_quiet_physics_sample(self):
        """Optional task extension; default preserves the v1 dynamics path."""

    def _finish_quiet_policy_step(self):
        """Optional label snapshot before reward calculation and episode reset."""

    def _reset_idx(self, env_ids):
        tracker = getattr(self, "stair_quiet_tracker", None)
        landing = getattr(self, "stair_quiet_landing", None)
        metrics = tracker.metrics(env_ids) if tracker is not None else {}
        if landing is not None:
            landing.before_reset(env_ids)
        super()._reset_idx(env_ids)
        # Preserve every baseline curriculum/traversal/fall/power diagnostic.
        self.extras.setdefault("log", {}).update(metrics)
        if hasattr(self, "stair_quiet_reward_scale"):
            self.extras["log"]["StairQuiet/curriculum_scale"] = self.stair_quiet_reward_scale
            self.extras["log"]["StairQuiet/quiet_task_iteration"] = self.stair_quiet_iteration
        if tracker is not None:
            tracker.reset(env_ids)
        if landing is not None:
            landing.reset(env_ids)


class M20StairQuietV1StudentEnv(M20StairQuietV1Env):
    """789D student ABI and a single shared landing/physics/history path."""

    def __init__(self, cfg, **kwargs):
        cfg.student_perception.validate()
        if not cfg.stair_quiet_student:
            raise ValueError("Quiet student requires its explicit student configuration")
        super().__init__(cfg, **kwargs)

    def step(self, action):
        from task_support.student_v1 import environment as glue
        from task_support.student_v3_5 import v3_5_environment as motion
        # Cached at the existing policy tick: this does not add a history entry.
        motion.student_observation(self)
        robot = self.scene["robot"]
        tracker = self.stair_quiet_tracker
        tags = glue.classify_state(
            self.scene.sensors["height_scanner"].data.ray_hits_w,
            robot.data.projected_gravity_b, self.command_manager.get_command("base_velocity"),
            torch.where(self.stair_quiet_landing.terrain_valid, self.stair_quiet_landing.clearance, torch.inf),
            robot.data.body_lin_vel_w[:, self._quiet_body_ids, 2], tracker.contact,
            self.scene.terrain.terrain_levels,
        )
        # Issued actions remain the policy's nominal actions; landing target
        # edits must never be reinterpreted as normalized policy actions.
        applied = glue.action_delay(self).push(action.to(self.device))
        obs, reward, terminated, timed_out, extras = super().step(applied)
        touchdown = self.stair_quiet_step_events.sum(-1) > 0
        tags["phase"] = torch.where(touchdown & (tags["phase"] != 3), 2, tags["phase"])
        tags["terminated"] = terminated.clone()
        extras["student_v1"] = tags
        return obs, reward, terminated, timed_out, extras
