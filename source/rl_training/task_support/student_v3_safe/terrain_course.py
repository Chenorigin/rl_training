"""Student-owned expanded case ledger; frozen teacher course remains untouched.

The original width/material accounting is retained with runtime-sized case axes.
"""
import torch
from task_support.stair_quiet_v1_width.settings import PROFILES
from task_support.stair_quiet_v1_width.signals import failure_mask
from task_support.stair_quiet_v1_width.curriculum import WidthCurriculum
from .terrain_geometry import cases, surfaces
FAMILIES = ("legacy", "round_trip", "open_ascent")

def _base_state(env):
    state = getattr(env, "_tail_course", None)
    if state is None:
        table = cases(env.cfg.student_terrain_course)
        state = {
            key: torch.tensor(values, device=env.device, dtype=dtype)
            for key, values, dtype in (
                ("height_table", [g.height for g, _ in table], torch.float),
                ("depth_table", [g.depth for g, _ in table], torch.float),
                ("width_table", [g.width for g, _ in table], torch.float),
                ("steps_table", [g.steps for g, _ in table], torch.long),
                ("stage_table", [p.stage for _, p in table], torch.long),
            )}
        state["speed"] = torch.full((env.num_envs,), .35, device=env.device)
        state["assign_count"] = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        state["episodes"] = torch.zeros(len(table), dtype=torch.long, device=env.device)
        state["successes"] = torch.zeros(len(table), dtype=torch.long, device=env.device)
        state["stage_episodes"] = torch.zeros(len(table), dtype=torch.long, device=env.device)
        state["stage_successes"] = torch.zeros(len(table), dtype=torch.long, device=env.device)
        state["case_metrics"] = {key: torch.zeros(len(table), device=env.device) for key in (
            "force_peak_sum", "force_peak_max", "impulse_peak_sum", "impulse_peak_max",
            "slip_seconds_sum", "unsafe_seconds_sum", "recovery_events_sum", "edge_failures_sum")}
        env._tail_course = state
        env.tail_stage = env.cfg.stair_tail.initial_stage
    if "width_curriculum" not in state:
        s = env.cfg.stair_tail
        state["width_curriculum"] = WidthCurriculum(env.num_envs, env.device, s)
        state["width_values"] = torch.tensor(s.widths_m, device=env.device)
        state["width_statistics"] = {key: torch.zeros(len(s.widths_m), state["height_table"].numel(),
            dtype=torch.long, device=env.device) for key in (
                "episodes", "successes", "edge_failures", "recovery_timeouts", "stall_timeouts", "other_failures")}
    if "family_table" not in state:
        table = cases(env.cfg.student_terrain_course)
        definitions = [surfaces(g, env.cfg.stair_tail) for g, _ in table]
        count = max(len(value[0]) for value in definitions)
        padded = torch.zeros(len(table), count, 3, device=env.device)
        for i, value in enumerate(definitions):
            padded[i, :len(value[0])] = torch.tensor(value[0], device=env.device)
        state["span_table"] = padded
        state["family_table"] = torch.tensor([FAMILIES.index(g.family) for g,_ in table], device=env.device)
        state["goal_table"] = torch.tensor([v[1] for v in definitions], device=env.device)
        state["goal_height_table"] = torch.tensor([v[2] for v in definitions], device=env.device)
        state["ascent_end_table"] = torch.tensor([env.cfg.stair_tail.start_length/2+g.steps*g.depth for g,_ in table],device=env.device)
    types = env.scene.terrain.terrain_types
    for name in ("height", "depth", "steps"):
        state[name] = state[name+"_table"][types]
    levels = env.scene.terrain.terrain_levels
    state["width"] = state["width_values"][levels]
    for key in ("family", "goal", "goal_height", "ascent_end"):
        state[key] = state[key+"_table"][types]
    state["spans"] = state["span_table"][types]
    state["round_trip"] = state["family"] == 1
    return state


SCALARS = ("elapsed_s", "unsafe_seconds", "slip_seconds", "recovery_events", "force_peak",
           "impulse_peak", "backward_distance_m")
FEET = ("support_foot_seconds", "slip_foot_seconds", "unsafe_foot_seconds", "slip_foot_distance_m",
        "touchdown_unsafe_count", "early_unsafe_foot_seconds", "predicted_unsafe_foot_seconds")


def course_state(env):
    state = _base_state(env)
    if "material_practice" not in state:
        s = env.cfg.stair_tail
        state["material_practice"] = torch.arange(env.num_envs, device=env.device)%s.material_practice_every == 0
        shape = (len(s.widths_m), state["height_table"].numel())
        state["mechanics"] = {k: torch.zeros((*shape, *((4,) if k in FEET else ())),
            dtype=torch.float64, device=env.device) for k in (*SCALARS, *FEET)}
        state["width_curriculum"].level[state["material_practice"]] = s.material_practice_level
    return state


def case_assignment(env_ids, reset_counts, allowed_count, practice_every, seed):
    # Practice indices are contiguous, and every reset advances by one. In
    # particular, every-fourth env with 4096 envs cannot alias onto even cases.
    rank = torch.where(env_ids%practice_every == 0, env_ids//practice_every, env_ids)
    return (rank+reset_counts+seed)%allowed_count


def reset_course(env, env_ids, pose_range, velocity_range, asset_cfg=None):
    state = course_state(env)
    s, terrain, practice = env.cfg.stair_tail, env.scene.terrain, state["material_practice"]
    state["width_curriculum"].level[practice] = s.material_practice_level
    state["width_curriculum"].success_streak[practice] = 0
    allowed = torch.where(state["stage_table"] <= env.tail_stage)[0]
    cycle = torch.tensor([FAMILIES.index(name) for name in env.cfg.student_terrain_course["family_cycle"]],device=env.device)
    family = cycle[(env_ids+state["assign_count"][env_ids]+int(env.cfg.seed)) % len(cycle)]
    chosen = torch.empty_like(env_ids)
    for code in range(len(FAMILIES)):
        mask = family == code
        pool = allowed[state["family_table"][allowed] == code]
        if not len(pool):
            raise RuntimeError("No terrain cases available for requested family")
        assignment = case_assignment(env_ids[mask], state["assign_count"][env_ids][mask],len(pool),s.material_practice_every,int(env.cfg.seed))
        chosen[mask] = pool[assignment]
    terrain.terrain_types[env_ids] = chosen
    levels = state["width_curriculum"].level[env_ids]
    terrain.terrain_levels[env_ids] = levels
    terrain.env_origins[env_ids] = terrain.terrain_origins[levels, terrain.terrain_types[env_ids]]
    env.scene.env_origins[env_ids] = terrain.env_origins[env_ids]
    state["assign_count"][env_ids] += 1
    robot = env.scene["robot" if asset_cfg is None else asset_cfg.name]
    pos = robot.data.default_root_state[env_ids, :3].clone()+env.scene.env_origins[env_ids]
    pos[:, 0] += torch.empty(len(env_ids), device=env.device).uniform_(-.05, .05)
    pos[:, 1] += torch.empty(len(env_ids), device=env.device).uniform_(-s.reset_y_range, s.reset_y_range)
    yaw = torch.empty(len(env_ids), device=env.device).uniform_(-s.reset_yaw_range, s.reset_yaw_range)
    quat = torch.zeros(len(env_ids), 4, device=env.device)
    quat[:, 0], quat[:, 3] = torch.cos(yaw/2), torch.sin(yaw/2)
    robot.write_root_pose_to_sim(torch.cat((pos, quat), -1), env_ids=env_ids)
    robot.write_root_velocity_to_sim(torch.zeros(len(env_ids), 6, device=env.device), env_ids=env_ids)
    course_state(env)


def record_outcomes(state, types, rows, succeeded, edge, recovery, stall, s):
    one = torch.ones_like(types)
    for key, value in (("episodes", one), ("successes", succeeded.long())):
        state[key].index_add_(0, types, value)
    # Material evidence is measured on sufficiently wide REAL stairs. Narrow
    # successes remain in width statistics and never substitute for this gate.
    eligible = rows <= s.material_practice_level
    state["stage_episodes"].index_add_(0, types[eligible], one[eligible])
    state["stage_successes"].index_add_(0, types[eligible], succeeded[eligible].long())
    index = rows*state["height_table"].numel()+types
    for key, value in (("episodes", one), ("successes", succeeded.long()), ("edge_failures", edge.long()),
            ("recovery_timeouts", recovery.long()), ("stall_timeouts", stall.long()),
            ("other_failures", (~succeeded & ~edge & ~recovery & ~stall).long())):
        state["width_statistics"][key].view(-1).index_add_(0, index, value)


def advance_materials(env, state):
    active = state["stage_table"] <= env.tail_stage
    s = env.cfg.stair_tail
    ready = ((state["stage_episodes"][active] >= s.stage_min_episodes)
        & (state["stage_successes"][active]/state["stage_episodes"][active].clamp_min(1)
           >= s.stage_success_fraction)).all()
    if s.automatic_stages and env.tail_stage < 2 and bool(ready):
        env.tail_stage += 1
        state["stage_episodes"].zero_()
        state["stage_successes"].zero_()


def pooled(result, label, total, count):
    result["Tail/__weighted_total__/"+label] = total
    result["Tail/__weighted_count__/"+label] = count
    result["Tail/"+label] = total/count.clamp_min(1.e-9) if isinstance(count, torch.Tensor) else total/max(count, 1)


def record_episodes(env, env_ids):
    state, tracker, s = course_state(env), env.tail_tracker, env.cfg.stair_tail
    finished = env.termination_manager.terminated | env.termination_manager.time_outs
    ids = env_ids[(env.episode_length_buf[env_ids] > 0) & finished[env_ids]]
    types, rows = env.scene.terrain.terrain_types[ids], env.scene.terrain.terrain_levels[ids].clone()
    yes = tracker.success[ids] & ~failure_mask(env)[ids]
    edge, recovery = tracker.edge_failed[ids], tracker.recovery_failed[ids]
    stall = tracker.stall_age[ids] > s.no_progress_timeout_s
    record_outcomes(state, types, rows, yes, edge, recovery, stall, s)
    index = rows*state["height_table"].numel()+types
    for name, ledger in state["mechanics"].items():
        ledger.flatten(0, 1).index_add_(0, index, getattr(tracker, name)[ids].double())
    # Preserve the existing case ledger for readers, but no inherited success
    # gate or inherited outcome accumulator is called by this environment.
    for name in ("force_peak", "impulse_peak", "slip_seconds", "unsafe_seconds", "recovery_events"):
        value = getattr(tracker, name)[ids]
        state["case_metrics"][name+"_sum"].index_add_(0, types, value)
        if name+"_max" in state["case_metrics"]:
            state["case_metrics"][name+"_max"].scatter_reduce_(0, types, value, reduce="amax", include_self=True)
    state["case_metrics"]["edge_failures_sum"].index_add_(0, types, edge.float())
    learn = ~state["material_practice"][ids]
    state["width_curriculum"].update(ids[learn], actual_levels=rows[learn], succeeded=yes[learn])
    advance_materials(env, state)
    result = {"Tail/stage": float(env.tail_stage), "Width/actual_mean_m": state["width"].mean(),
              "Width/target_row_fraction": (env.scene.terrain.terrain_levels == len(s.widths_m)-1).float().mean(),
              "Safe/material_practice_fraction": state["material_practice"].float().mean()}
    for i, width in enumerate(s.widths_m):
        for key, values in state["width_statistics"].items():
            result[f"Width/{width:g}m/{key}"] = values[i].sum().float()
    for code,name in enumerate(FAMILIES):
        columns = state["family_table"] == code
        result["StudentTerrain/"+name+"/episodes"] = state["episodes"][columns].sum().float()
        result["StudentTerrain/"+name+"/successes"] = state["successes"][columns].sum().float()
        result["StudentTerrain/"+name+"/active_fraction"] = (state["family"] == code).float().mean()
    if not len(ids):
        return result
    for name in SCALARS:
        pooled(result, name+"_episode_mean", getattr(tracker, name)[ids].sum(), len(ids))
    for name in ("edge_failed", "recovery_failed"):
        pooled(result, name+"_fraction", getattr(tracker, name)[ids].float().sum(), len(ids))
    for row, width in enumerate(s.widths_m):
        selected = ids[rows == row]
        if not len(selected):
            continue
        for foot, leg in enumerate(("fl", "fr", "hl", "hr")):
            prefix = f"safe/{width:g}m/{leg}/"
            support = tracker.support_foot_seconds[selected, foot].sum()
            for name, numerator, denominator in (
                ("slip_support_fraction", tracker.slip_foot_seconds[selected, foot].sum(), support),
                ("unsafe_support_fraction", tracker.unsafe_foot_seconds[selected, foot].sum(), support),
                ("slip_distance_m_per_episode", tracker.slip_foot_distance_m[selected, foot].sum(), len(selected)),
                ("unsafe_touchdowns_per_episode", tracker.touchdown_unsafe_count[selected, foot].sum(), len(selected))):
                pooled(result, prefix+name, numerator, denominator)
    for pi, profile in enumerate(PROFILES):
        selected = ids[types%len(PROFILES) == pi]
        if len(selected):
            pooled(result, "material/"+profile.name+"/slip_support_fraction",
                tracker.slip_foot_seconds[selected].sum(), tracker.support_foot_seconds[selected].sum())
    return result


def validate_counts(state, s):
    stats = state["width_statistics"]
    if (not torch.equal(stats["episodes"].sum(0), state["episodes"])
            or not torch.equal(stats["successes"].sum(0), state["successes"])
            or any((v < 0).any() or (v > stats["episodes"]).any() for v in stats.values())
            or (state["stage_successes"] > state["stage_episodes"]).any()):
        raise ValueError("Inconsistent safe width/material counters")
    eligible = slice(0, s.material_practice_level+1)
    total, success = stats["episodes"][eligible].sum(0), stats["successes"][eligible].sum(0)
    if ((state["stage_successes"] > success).any()
            or (state["stage_episodes"]-state["stage_successes"] > total-success).any()):
        raise ValueError("Material evidence does not match wider physical stairs")
