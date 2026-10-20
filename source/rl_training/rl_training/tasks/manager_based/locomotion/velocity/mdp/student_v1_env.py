"""V1 student commands and environment integration shared by later versions."""
import torch
from m20_student_v1 import environment as glue
from m20_student_v1.contract import student_dim
from .quiet_landing_env import M20QuietLandingEnv, StairTeacherVelocityCommand
from task_support.quiet_landing.signals import map_clearance


class StudentV1VelocityCommand(StairTeacherVelocityCommand):
    """Explicit forward/backward/lateral/yaw/idle exposure; no hidden heading loop."""
    def _resample_command(self, env_ids):
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        super()._resample_command(env_ids)
        n = len(env_ids)
        if not n:
            return
        category = torch.multinomial(torch.tensor([.10, .25, .15, .20, .15, .15], device=self.device), n, replacement=True)
        # Re-sample the parent's special-case zeros before assigning v1 categories.
        for column, limits in enumerate((self.cfg.ranges.lin_vel_x, self.cfg.ranges.lin_vel_y, self.cfg.ranges.ang_vel_z)):
            self.vel_command_b[env_ids, column] = torch.empty(n, device=self.device).uniform_(*limits)
        self.vel_command_b[env_ids[category == 0]] = 0.
        forward = env_ids[category == 1]
        backward = env_ids[category == 2]
        self.vel_command_b[forward, 0] = self.vel_command_b[forward, 0].abs()
        self.vel_command_b[backward, 0] = -self.vel_command_b[backward, 0].abs()
        self.vel_command_b[forward, 1:] = 0.
        self.vel_command_b[backward, 1:] = 0.
        lateral = env_ids[category == 3]
        yaw = env_ids[category == 4]
        self.vel_command_b[lateral, 0] = 0.
        self.vel_command_b[lateral, 2] = 0.
        self.vel_command_b[yaw, :2] = 0.
        self.is_standing_env[env_ids] = category == 0



class M20StudentV1Env(M20QuietLandingEnv):
    def __init__(self, cfg, **kwargs):
        cfg.student_perception.validate()
        self.POLICY_OBSERVATION_DIM = student_dim(cfg.student_history_steps)
        super().__init__(cfg, **kwargs)
        if self.observation_manager.group_obs_dim["teacher"] != (244,):
            raise RuntimeError("V1 requires the clean 244D teacher")

    def step(self, action):
        robot = self.scene["robot"]
        hits = self.scene.sensors["height_scanner"].data.ray_hits_w
        wheels = robot.data.body_pos_w[:, self._quiet_body_ids]
        clearance, valid = map_clearance(wheels, hits, self.cfg.quiet_landing)
        tags = glue.classify_state(hits, robot.data.projected_gravity_b,
                                    self.command_manager.get_command("base_velocity"),
                                    clearance, robot.data.body_lin_vel_w[:, self._quiet_body_ids, 2],
                                    self.quiet_tracker.contact,
                                    self.scene.terrain.terrain_levels)
        applied = glue.action_delay(self).push(action.to(self.device))
        obs, rewards, terminated, timed_out, extras = super().step(applied)
        touchdown = self.quiet_step_events.sum(-1) > 0
        tags["phase"] = torch.where(touchdown & (tags["phase"] != 3), 2, tags["phase"])
        tags["terminated"] = terminated.clone()
        extras["student_v1"] = tags
        state = glue.sensor_state(self)
        log = extras.setdefault("log", {})
        usable = obs["policy"][:, 244:431] > .5
        log.update({"StudentSensor/valid_fraction": usable.float().mean(),
                    "StudentSensor/confidence_mean": obs["policy"][:, 431:618].mean(),
                    "StudentSensor/visible_fraction": state.last_visibility.float().mean(),
                    "StudentSensor/level": torch.tensor(state.cfg.level, device=self.device),
                    "StudentSensor/oldest_valid_age_s": torch.where(usable, state.cell_age, 0.).max()})
        for category in range(3):
            log[f"StudentSensor/mixture_{category}_fraction"] = (state.mixture == category).float().mean()
        return obs, rewards, terminated, timed_out, extras
