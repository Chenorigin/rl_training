"""Width teacher with owned contact, safety and curriculum implementation."""
import torch
from .stair_quiet_v1_env import M20StairQuietV1Env
from task_support.stair_quiet_v1.signals import wheel_axis_world
from task_support.stair_quiet_v1_width.course import course_state, record_episodes
from task_support.stair_quiet_v1_width.signals import TailTracker
from task_support.stair_quiet_v1_width.terrain import neutral_wheel_material


class M20StairQuietV1WidthEnv(M20StairQuietV1Env):
    def __init__(self, cfg, **kwargs):
        cfg.stair_tail.validate()
        generator = cfg.scene.terrain.terrain_generator
        if generator.num_rows != len(cfg.stair_tail.widths_m) or tuple(generator.difficulty_range) != (0., 1.):
            raise ValueError("Width rows must match the curriculum and span difficulty [0,1]")
        if any(tuple(case.settings.widths_m) != tuple(cfg.stair_tail.widths_m)
               for case in generator.sub_terrains.values()):
            raise ValueError("Terrain mesh width ledger differs from curriculum")
        super().__init__(cfg, **kwargs)
        course_state(self)
        neutral_wheel_material(self)
        self.tail_tracker = TailTracker(self.num_envs, self.device, cfg.stair_tail)
        self.tail_tracker.best_x.copy_(self.scene["robot"].data.root_pos_w[:, 0]-self.scene.env_origins[:, 0])

    def step(self, action):
        self.tail_tracker.begin_step()
        return super().step(action)

    def _after_quiet_physics_sample(self):
        data, state = self.scene["robot"].data, course_state(self)
        ids = self._quiet_body_ids
        q = data.root_quat_w
        yaw = torch.atan2(2*(q[:, 0]*q[:, 3]+q[:, 1]*q[:, 2]), 1-2*(q[:, 2].square()+q[:, 3].square()))
        axis = wheel_axis_world(data.body_link_quat_w[:, ids], self.cfg.stair_quiet, validate=False)
        omega = data.body_ang_vel_w[:, ids]
        velocity = data.body_lin_vel_w[:, ids]+torch.linalg.cross(
            omega, data.body_link_pos_w[:, ids]-data.body_com_pos_w[:, ids], dim=-1)
        self.tail_tracker.update(position=data.body_link_pos_w[:, ids]-self.scene.env_origins[:, None, :],
            axis=axis, velocity=velocity, omega=omega,
            forces=self.scene.sensors["contact_forces"].data.net_forces_w[:, self._quiet_contact_ids],
            root=data.root_pos_w-self.scene.env_origins, root_velocity=data.root_lin_vel_w,
            yaw=yaw, upright=-data.projected_gravity_b[:, 2], height=state["height"], depth=state["depth"],
            width=state["width"], steps=state["steps"], dt=self.physics_dt)

    def _reset_idx(self, env_ids):
        tracker = getattr(self, "tail_tracker", None)
        metrics = (record_episodes(self, env_ids) if tracker is not None
                   and not getattr(self, "_tail_skip_episode_record", False) else {})
        if tracker is not None and hasattr(tracker, "last"):
            tracker.last["crossed"][env_ids] = False
        super()._reset_idx(env_ids)
        self.extras.setdefault("log", {}).update(metrics)
        if tracker is not None:
            tracker.reset(env_ids, self.scene["robot"].data.root_pos_w[env_ids, 0]-self.scene.env_origins[env_ids, 0])
