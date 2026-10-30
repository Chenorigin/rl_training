"""35-case physical routing; varied approaches and explicit per-protocol evidence."""
import torch
from task_support.stair_quiet_v1_width.curriculum import WidthCurriculum
from task_support.stair_quiet_v1_width.signals import failure_mask
from task_support.stair_quiet_v1_safe.course import SCALARS, FEET, pooled, advance_materials, validate_counts
from .settings import cases, PROTOCOLS, GEOMETRIES, PROFILES
from .protocol import EntryProtocol
from .signals import METRICS


def course_state(env):
    state = getattr(env, "_tail_course", None)
    if state is None:
        table, s = cases(), env.cfg.stair_tail
        state = {key: torch.tensor(values, dtype=dtype, device=env.device) for key, values, dtype in (
            ("height_table", [g.height for g, _ in table], torch.float),
            ("depth_table", [g.depth for g, _ in table], torch.float),
            ("steps_table", [g.steps for g, _ in table], torch.long),
            ("closed_table", [g.closed_riser for g, _ in table], torch.bool),
            ("stage_table", [p.stage for _, p in table], torch.long))}
        n, c, w = env.num_envs, len(table), len(s.widths_m)
        state.update(speed=torch.full((n,), .6, device=env.device),
            assign_count=torch.zeros(n, dtype=torch.long, device=env.device),
            width_curriculum=WidthCurriculum(n, env.device, s),
            width_values=torch.tensor(s.widths_m, device=env.device),
            material_practice=torch.arange(n, device=env.device)%s.material_practice_every == 0,
            protocol=EntryProtocol(n, env.device, s))
        for key in ("episodes", "successes", "stage_episodes", "stage_successes"):
            state[key] = torch.zeros(c, dtype=torch.long, device=env.device)
        state["width_statistics"] = {k: torch.zeros(w, c, dtype=torch.long, device=env.device)
            for k in ("episodes", "successes", "edge_failures", "recovery_timeouts", "stall_timeouts", "other_failures")}
        state["case_metrics"] = {k: torch.zeros(c, device=env.device) for k in (
            "force_peak_sum", "force_peak_max", "impulse_peak_sum", "impulse_peak_max", "slip_seconds_sum",
            "unsafe_seconds_sum", "recovery_events_sum", "edge_failures_sum")}
        state["mechanics"] = {k: torch.zeros(w, c, *((4,) if k in FEET else ()), dtype=torch.float64, device=env.device)
                              for k in (*SCALARS, *FEET)}
        state["transition_statistics"] = {k: torch.zeros(len(PROTOCOLS), c, dtype=torch.float64, device=env.device)
            for k in ("episodes", "successes", "edge_failures", *METRICS)}
        state["width_curriculum"].level[state["material_practice"]] = s.material_practice_level
        env._tail_course, env.tail_stage = state, s.initial_stage
    types = env.scene.terrain.terrain_types
    for key in ("height", "depth", "steps", "closed"):
        state[key] = state[key+"_table"][types]
    state["width"] = state["width_values"][env.scene.terrain.terrain_levels]
    return state


def reset_course(env, env_ids, pose_range, velocity_range, asset_cfg=None):
    state, s = course_state(env), env.cfg.stair_tail
    terrain, levels = env.scene.terrain, state["width_curriculum"].level
    practice = state["material_practice"]
    levels[practice] = s.material_practice_level
    state["width_curriculum"].success_streak[practice] = 0
    allowed = torch.where(state["stage_table"] <= env.tail_stage)[0]
    rank = torch.where(env_ids%s.material_practice_every == 0, env_ids//s.material_practice_every, env_ids)
    assignment = (rank+state["assign_count"][env_ids]+int(env.cfg.seed))%len(allowed)
    terrain.terrain_types[env_ids] = allowed[assignment]
    terrain.terrain_levels[env_ids] = levels[env_ids]
    terrain.env_origins[env_ids] = terrain.terrain_origins[levels[env_ids], terrain.terrain_types[env_ids]]
    env.scene.env_origins[env_ids] = terrain.env_origins[env_ids]
    state["assign_count"][env_ids] += 1
    state["protocol"].reset(env_ids, state["assign_count"][env_ids])
    state["speed"][env_ids] = torch.empty(len(env_ids), device=env.device).uniform_(*s.speed_range)
    robot = env.scene["robot"]
    pos = robot.data.default_root_state[env_ids, :3].clone()+env.scene.env_origins[env_ids]
    pos[:, 0] += torch.empty(len(env_ids), device=env.device).uniform_(*s.approach_x_range)
    pos[:, 1] += torch.empty(len(env_ids), device=env.device).uniform_(-s.reset_y_range, s.reset_y_range)
    yaw = torch.empty(len(env_ids), device=env.device).uniform_(-s.reset_yaw_range, s.reset_yaw_range)
    quat = torch.zeros(len(env_ids), 4, device=env.device)
    quat[:, 0], quat[:, 3] = torch.cos(yaw/2), torch.sin(yaw/2)
    robot.write_root_pose_to_sim(torch.cat((pos, quat), -1), env_ids=env_ids)
    # Build velocity through real approach motion, never teleport a stationary
    # wheel into a translating body and call that an entry-speed experiment.
    robot.write_root_velocity_to_sim(torch.zeros(len(env_ids), 6, device=env.device), env_ids=env_ids)
    course_state(env)



def record_episodes(env, env_ids):
    state, t, s = course_state(env), env.tail_tracker, env.cfg.stair_tail
    ids = env_ids[(env.episode_length_buf[env_ids] > 0) & (env.termination_manager.terminated[env_ids] | env.termination_manager.time_outs[env_ids])]
    types, rows = env.scene.terrain.terrain_types[ids], env.scene.terrain.terrain_levels[ids].clone()
    yes = t.success[ids] & ~failure_mask(env)[ids]
    edge, recovery = t.edge_failed[ids], t.recovery_failed[ids]
    stall = t.stall_age[ids] > s.no_progress_timeout_s
    for key, values in (("episodes", torch.ones_like(types)), ("successes", yes.long())):
        state[key].index_add_(0, types, values)
    eligible = rows <= s.material_practice_level
    state["stage_episodes"].index_add_(0, types[eligible], torch.ones_like(types[eligible]))
    state["stage_successes"].index_add_(0, types[eligible], yes[eligible].long())
    index = rows*len(cases())+types
    for key, values in (("episodes", torch.ones_like(types)), ("successes", yes.long()),
            ("edge_failures", edge.long()), ("recovery_timeouts", recovery.long()), ("stall_timeouts", stall.long()),
            ("other_failures", (~yes & ~edge & ~recovery & ~stall).long())):
        state["width_statistics"][key].flatten().index_add_(0, index, values)
    for key, ledger in state["mechanics"].items():
        ledger.flatten(0, 1).index_add_(0, index, getattr(t, key)[ids].double())
    for name in ("force_peak", "impulse_peak", "slip_seconds", "unsafe_seconds", "recovery_events"):
        state["case_metrics"][name+"_sum"].index_add_(0, types, getattr(t, name)[ids])
        if name+"_max" in state["case_metrics"]:
            state["case_metrics"][name+"_max"].scatter_reduce_(0, types, getattr(t, name)[ids], reduce="amax", include_self=True)
    state["case_metrics"]["edge_failures_sum"].index_add_(0, types, edge.float())
    modes = state["protocol"].mode[ids]
    mode_cases = modes*len(cases())+types
    for key, values in (("episodes", torch.ones_like(modes)), ("successes", yes), ("edge_failures", edge)):
        state["transition_statistics"][key].flatten().index_add_(0, mode_cases, values.double())
    for key in METRICS:
        state["transition_statistics"][key].flatten().index_add_(0, mode_cases, getattr(t, key)[ids].double())
    learn = ~state["material_practice"][ids]
    state["width_curriculum"].update(ids[learn], actual_levels=rows[learn], succeeded=yes[learn])
    advance_materials(env, state)
    result = {"Tail/stage": float(env.tail_stage), "Width/actual_mean_m": state["width"].mean()}
    for i, width in enumerate(s.widths_m):
        for key, values in state["width_statistics"].items():
            result[f"Width/{width:g}m/{key}"] = values[i].sum().float()
    for i, name in enumerate(PROTOCOLS):
        for key, values in state["transition_statistics"].items():
            result[f"Transition/{name}/{key}_total"] = values[i].sum()
        for gi, geometry in enumerate(GEOMETRIES):
            for key in ("episodes", "successes", "edge_failures", "riser_work_proxy_J", "riser_descent_seconds"):
                result[f"Transition/{name}/{geometry.name}/{key}_total"] = state["transition_statistics"][key][i, gi*len(PROFILES):(gi+1)*len(PROFILES)].sum()
    if len(ids):
        for key in (*SCALARS, *METRICS):
            pooled(result, "transition/"+key+"_episode_mean", getattr(t, key)[ids].sum(), len(ids))
    return result


def validate_transition_counts(state):
    validate_counts(state, state["width_curriculum"].s)
    values = state["transition_statistics"]
    if (not torch.equal(values["episodes"].sum(0), state["episodes"].double())
            or not torch.equal(values["successes"].sum(0), state["successes"].double())
            or not torch.equal(values["edge_failures"].sum(0), state["width_statistics"]["edge_failures"].sum(0).double())
            or (values["successes"] > values["episodes"]).any()
            or (values["edge_failures"] > values["episodes"]).any()
            or any(not torch.equal(values[key], values[key].round()) for key in ("episodes", "successes", "edge_failures"))):
        raise ValueError("Transition protocol evidence inconsistent with physical episodes")
