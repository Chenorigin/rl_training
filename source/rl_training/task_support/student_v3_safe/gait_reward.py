"""Training-only diagonal alternation score; no phase or reward input to actor."""
import math
import torch


def validate_settings(cfg):
    keys = {"weight", "min_hold_s", "max_half_cycle_s", "min_clearance_m", "progress_speed_m_s",
            "forward_min_m_s", "lateral_max_m_s", "yaw_max_rad_s", "upright_min", "heading_max_rad"}
    if (set(cfg) != keys or any(isinstance(v, bool) or not math.isfinite(v) or v < 0 for v in cfg.values())
            or any(cfg[k] == 0 for k in keys-{"weight"})):
        raise ValueError("Invalid diagonal reward settings")
    if cfg["min_hold_s"] >= cfg["max_half_cycle_s"] or cfg["upright_min"] >= 1:
        raise ValueError("Invalid diagonal reward timing/upright threshold")


class DiagonalAlternation:
    def __init__(self, n, device, cfg):
        validate_settings(cfg)
        self.cfg = dict(cfg)
        self.frontier = torch.full((n,), -torch.inf, device=device)
        self.last_tick = torch.full((n,), -1, dtype=torch.long, device=device)
        self.last_pair = torch.zeros(n, dtype=torch.long, device=device)
        self.candidate = self.last_pair.clone()
        self.hold = torch.zeros(n, device=device)
        self.age = torch.zeros(n, device=device)
        self.cycling = torch.zeros(n, dtype=torch.bool, device=device)
        self.reward = torch.zeros(n, device=device)
        self.switched = self.cycling.clone()

    def reset(self, ids, x):
        self.frontier[ids] = x
        self.last_tick[ids] = -1
        for value in (self.last_pair, self.candidate, self.hold, self.age, self.cycling, self.reward, self.switched):
            value[ids] = 0

    def step(self, tick, dt, x, supported, airborne, eligible, safe):
        if not math.isfinite(dt) or dt <= 0 or not torch.isfinite(x).all():
            raise ValueError("Invalid diagonal reward clock/position")
        fresh = self.last_tick != tick
        active = eligible & safe
        pair = torch.zeros_like(self.last_pair)
        pair[supported[:, 0] & supported[:, 3] & airborne[:, 1] & airborne[:, 2]] = 1
        pair[supported[:, 1] & supported[:, 2] & airborne[:, 0] & airborne[:, 3]] = 2
        # Advance frontier even while gated off: no banking progress for later.
        delta = torch.where(torch.isfinite(self.frontier), (x-self.frontier).clamp_min(0), 0.)
        self.frontier.copy_(torch.where(fresh, torch.maximum(self.frontier, x), self.frontier))
        # A transient edge-margin failure suppresses reward, but must not erase
        # the previous safe phase during a transfer. Flight/command exit do.
        invalid = ~eligible | (supported.sum(-1) == 0)
        expired = self.age + dt > self.cfg["max_half_cycle_s"]
        clear = fresh & (invalid | expired)
        self.last_pair[clear] = 0
        self.cycling[clear] = False
        # Dwell measures the contact phase, independently of instantaneous margins.
        self.hold.copy_(torch.where(fresh, torch.where(eligible & (pair != 0),
            torch.where(pair == self.candidate, self.hold + dt, dt), 0.), self.hold))
        self.candidate.copy_(torch.where(fresh, pair, self.candidate))
        qualified = active & (pair != 0) & (self.hold >= self.cfg["min_hold_s"])
        accept = fresh & qualified & (pair != self.last_pair)
        switched = accept & (self.last_pair != 0)
        self.cycling |= switched
        self.last_pair.copy_(torch.where(accept, pair, self.last_pair))
        self.age.copy_(torch.where(fresh, torch.where(accept, 0., self.age + dt), self.age))
        # A score is a bounded rate proportional to THIS step's new distance.
        # Overlap support windows are permitted but receive no positive score.
        score = (delta/(dt*self.cfg["progress_speed_m_s"])).clamp(0, 1)
        score *= active & qualified & self.cycling & ~invalid
        self.reward.copy_(torch.where(fresh, score, self.reward))
        self.switched.copy_(torch.where(fresh, switched, self.switched))
        self.last_tick[fresh] = tick
        return self.reward.clone()


def eligible(env, ascent_only=True):
    """Privileged training mask on the existing straight +X Safe course."""
    if hasattr(env.cfg,'student_terrain_course'):
        from .terrain_course import course_state
    else:
        from task_support.stair_quiet_v1_safe.course import course_state
    state = course_state(env)
    cfg = env.cfg.student_diagonal_reward
    robot = env.scene["robot"].data
    x = robot.root_pos_w[:, 0]-env.scene.env_origins[:, 0]
    y = robot.root_pos_w[:, 1]-env.scene.env_origins[:, 1]
    command = env.command_manager.get_command("base_velocity")
    start = env.cfg.stair_tail.start_length/2
    reach = env.cfg.stair_tail.body_half_length
    end=state.get('goal',start+state['steps']*state['depth'])
    phase=torch.ones_like(x,dtype=torch.bool)
    if ascent_only and 'round_trip' in state:
        phase=~state['round_trip'] | (x<=state['ascent_end'])
    return (phase & (command[:, 0] > cfg["forward_min_m_s"])
            & (command[:, 1].abs() < cfg["lateral_max_m_s"])
            & (command[:, 2].abs() < cfg["yaw_max_rad_s"])
            & (x + reach >= start) & (x - reach <= end)
            & (y.abs() < state["width"]/2) & (state["height"] > 0))


def step_evidence(env):
    """Weight-zero terms are skipped by Isaac; enabled missing terms must fail."""
    evidence = getattr(env, "_student_diagonal_step", None)
    if evidence is None:
        if env.cfg.student_diagonal_reward["weight"]:
            raise RuntimeError("Enabled diagonal reward did not execute this step")
        evidence = {"reward": torch.zeros(env.num_envs, device=env.device),
                    "switch": torch.zeros(env.num_envs, device=env.device, dtype=torch.bool),
                    "eligible": torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)}
        for key in ("safe", "diagonal", "sustained", "cycling"):
            evidence[key] = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        evidence["x"] = torch.zeros(env.num_envs, device=env.device)
        for key in ("supported", "airborne"):
            evidence[key] = torch.zeros((env.num_envs, 4), device=env.device, dtype=torch.bool)
    return evidence


def diagonal_reward(env):
    if hasattr(env.cfg,'student_terrain_course'):
        from .terrain_observations import physical_sample, safety_geometry
    else:
        from task_support.stair_quiet_v1_safe.observations import physical_sample, safety_geometry
    from task_support.student_v1.perception import quaternion_rotation
    d = physical_sample(env)
    g = safety_geometry(d, env.cfg.stair_tail)
    cfg = env.cfg.student_diagonal_reward
    tracker = getattr(env, "student_diagonal_tracker", None)
    if tracker is None:
        tracker = env.student_diagonal_tracker = DiagonalAlternation(env.num_envs, env.device, cfg)
    rotation = quaternion_rotation(env.scene["robot"].data.root_quat_w)
    local = torch.bmm(d["position"]-d["root"][:, None], rotation)
    no_cross = (local[:, [0, 2], 1] > 0).all(-1) & (local[:, [1, 3], 1] < 0).all(-1)
    safe_support = ((g["forward"] >= 0) & (g["side"] >= 0)) | ~g["supported"]
    safe = ((d["upright"] > cfg["upright_min"]) & (d["yaw"].abs() < cfg["heading_max_rad"])
            & no_cross & safe_support.all(-1) & torch.isfinite(g["clearance"]).all(-1))
    airborne = ~g["supported"] & (g["clearance"] >= cfg["min_clearance_m"])
    active = eligible(env)
    value = tracker.step(env.common_step_counter, env.step_dt, d["root"][:, 0],
                         g["supported"], airborne, active, safe)
    # Snapshot before auto-reset clears episode history.
    env._student_diagonal_step = {"reward": value.clone(), "switch": tracker.switched.clone(),
                                  "eligible": active.clone(), "safe": safe.clone(),
                                  "diagonal": (tracker.candidate != 0).clone(),
                                  "sustained": (tracker.hold >= cfg["min_hold_s"]).clone(),
                                  "cycling": tracker.cycling.clone(), "x": d["root"][:, 0].clone(),
                                  "supported": g["supported"].clone(), "airborne": airborne.clone()}
    return value
