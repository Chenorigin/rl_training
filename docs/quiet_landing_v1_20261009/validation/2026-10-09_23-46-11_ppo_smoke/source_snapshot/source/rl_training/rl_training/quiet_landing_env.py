"""Independent ManagerBasedRLEnv with explicit pre/post physics sampling.

The step lifecycle follows installed IsaacLab ManagerBasedRLEnv. Sampling is
inside its substep loop so rewards never confuse post-impact with pre-impact
velocity. No global hooks or changes to the teacher environments are used.
"""
import torch

from isaaclab.envs import ManagerBasedRLEnv
from m20_quiet.signals import LEGS, TouchdownTracker, map_clearance


class M20QuietLandingEnv(ManagerBasedRLEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        self.quiet_tracker = TouchdownTracker(self.num_envs, self.device, cfg.quiet_landing)
        robot = self.scene["robot"]
        sensor = self.scene.sensors["contact_forces"]
        self._quiet_body_ids, names = robot.find_bodies([f"{leg}_wheel" for leg in LEGS], preserve_order=True)
        self._quiet_contact_ids, contact_names = sensor.find_bodies([f"{leg}_wheel" for leg in LEGS], preserve_order=True)
        if names != [f"{leg}_wheel" for leg in LEGS] or contact_names != names:
            raise RuntimeError("Quiet landing wheel order does not match FL/FR/HL/HR")
        if self.observation_manager.group_obs_dim["policy"] != (244,) or self.action_manager.total_action_dim != 16:
            raise RuntimeError("Quiet V1 requires the baseline 244D actor / 16D action contract")

    def step(self, action):
        self.extras.pop("log", None)
        self.quiet_tracker.begin_step()
        self.action_manager.process_action(action.to(self.device))
        self.recorder_manager.record_pre_step()
        rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        robot = self.scene["robot"]
        sensor = self.scene.sensors["contact_forces"]
        # Original scanner only, at its original update frequency; no privileged rays.
        hits = self.scene.sensors["height_scanner"].data.ray_hits_w
        start_xy = robot.data.root_pos_w[:, :2].clone()
        for _ in range(self.cfg.decimation):
            pre_velocity = robot.data.body_lin_vel_w[:, self._quiet_body_ids].clone()
            wheels = robot.data.body_pos_w[:, self._quiet_body_ids]
            clearance, valid = map_clearance(wheels, hits, self.cfg.quiet_landing)
            self._sim_step_counter += 1
            self.action_manager.apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.recorder_manager.record_post_physics_decimation_step()
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and rendering:
                self.sim.render()
            self.scene.update(dt=self.physics_dt)
            post_force = sensor.data.net_forces_w[:, self._quiet_contact_ids]
            self.quiet_tracker.update(pre_velocity, post_force, self.physics_dt, clearance, valid)
        self.quiet_tracker.distance += torch.linalg.vector_norm(robot.data.root_pos_w[:, :2] - start_xy, dim=-1)
        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
        if len(self.recorder_manager.active_terms) > 0:
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()
        reset_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_ids) > 0:
            self.recorder_manager.record_pre_reset(reset_ids)
            self._reset_idx(reset_ids)
            rerenders = getattr(self.cfg, "num_rerenders_on_reset", int(getattr(self.cfg, "rerender_on_reset", False)))
            if self.sim.has_rtx_sensors():
                for _ in range(rerenders):
                    self.sim.render()
            self.recorder_manager.record_post_reset(reset_ids)
        self.command_manager.compute(dt=self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        self.obs_buf = self.observation_manager.compute(update_history=True)
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def _reset_idx(self, env_ids):
        tracker = getattr(self, "quiet_tracker", None)
        quiet = tracker.metrics(env_ids) if tracker is not None else {}
        super()._reset_idx(env_ids)
        # Keep all traversal rewards active, but do not carry their debugging
        # cards (gait_quality, angle/entry/leg pose, ascent_direction, etc.).
        old = self.extras["log"]
        kept = {key: value for key, value in old.items()
                if key.startswith(("Metrics/base_velocity/", "Episode_Termination/"))}
        reward_total = sum(value for key, value in old.items() if key.startswith("Episode_Reward/"))
        kept["Task/episode_reward_rate"] = reward_total
        for name in ("quiet_touchdown", "joint_acc_l2", "stair_joint_acceleration_cost",
                     "action_rate_l2", "action_smooth_l2"):
            key = "Episode_Reward/" + name
            if key in old:
                kept["QuietReward/" + name] = old[key]
        kept.update(quiet)
        self.extras["log"] = kept
        if tracker is not None:
            tracker.reset(env_ids)
