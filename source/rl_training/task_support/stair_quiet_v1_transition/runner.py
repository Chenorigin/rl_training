"""Complete Safe weight migration and versioned Transition-only resume."""
import copy
from pathlib import Path
import torch
from rsl_rl.runners import OnPolicyRunner
from task_support.idle_stop_motion_v1.runner import IdleStopMotionRunner, file_sha256
from task_support.stair_quiet_v1.baselines import PROJECT_ROOT, TEACHER, TEACHER_SHA256, checked_path
from task_support.stair_quiet_v1.runner import canonical
from task_support.stair_quiet_v1_safe.runner import (
    StairQuietSafeRunner, validate_model, validate_optimizer, digest,
    contract as safe_contract, sources as safe_sources, ACTOR as SAFE_ACTOR)
from task_support.stair_quiet_v1_safe.settings import SafeSettings
from task_support.stair_quiet_v1_safe.course import validate_counts as validate_safe_counts
from task_support.stair_quiet_v1_width.settings import cases as safe_cases
from task_support.stair_quiet_v1_width.runner import checked_counts
from task_support.stair_quiet_v1_width.curriculum import WidthCurriculum
from .course import course_state, validate_transition_counts
from .settings import GEOMETRIES

TASK = "stair_quiet_v1_transition"
VERSION = "m20_stair_quiet_v1_transition.v1"
ACTOR = "task_support.stair_quiet_v1_transition.actor:TransitionResidualActor"
ROUTING = "stair_scan_through_zero_v1"
DEFAULT_START = "logs/rsl_rl/deeprobotics_m20_stair_quiet_v1_safe/2026-10-22_13-29-16_teacher_safe_from_width10000/model_18400.pt"
DEFAULT_START_SHA256 = "18c5a16f6739f18d4d2920883ce49d36e63441d0a0253c1e3e19d17227a46498"
GROUPS = {"policy": 244, "motion_control": 3, "locomotion_scan": 187, "safety": 48}


def contract(env_cfg, train_cfg):
    result = safe_contract(env_cfg, train_cfg)
    result.update(task=TASK, geometries=canonical(GEOMETRIES), input_routing=ROUTING,
        initialization_source="complete_safe_actor_including_trained_safety_branch_and_critic",
        perception="raw244 safety branch; longitudinal187 base/residual on moving AND zero; privileged safety48",
        initial_episode_clock="reset_to_zero_not_randomized",
        transition_cost_semantics="geometry_gated_net_force_work_proxy_and_descending_swing_risk; no_new_hard_failure")
    return result


def validate_safe_source(saved, teacher, actor_state, critic_state, actor_config):
    """Validate the source's own 30-case contract before migrating its weights.

    Its statistics do not become evidence for the new entry/35-case task.
    No source object or current environment state is modified here.
    """
    m = saved.get("m20_training_manifest", {})
    noise = m.get("noise_refinement", {})
    expected_config = dict(actor_config, class_name=SAFE_ACTOR)
    frozen = noise.get("contract", {})
    if (m.get("refinement_version") != "m20_stair_quiet_v1_safe.v1"
            or noise.get("task") != "stair_quiet_v1_safe" or noise.get("version") != 1
            or noise.get("contract_sha256") != digest(frozen)
            or noise.get("source_sha256") != safe_sources()
            or m.get("actor_model_config") != expected_config or frozen.get("actor") != expected_config
            or m.get("actor_input_groups") != GROUPS
            or m.get("critic_input_groups") != {"critic": 285, "motion_control": 3}
            or m.get("motion_control_version") != "m20_raw_command_gate_20261017"
            or m.get("m20_profile") != "idle_stop_motion_v1" or m.get("control_profile") != "legacy_v1"):
        raise ValueError("Transition initialization requires an intact current Safe checkpoint/ABI/source")
    for key, expected in (("actor_state_dict", actor_state), ("critic_state_dict", critic_state),
                          ("preservation_reference_actor", actor_state)):
        validate_model(saved.get(key), expected)
    actor, reference = saved["actor_state_dict"], saved["preservation_reference_actor"]
    for k, v in teacher["actor_state_dict"].items():
        if (not torch.equal(reference[k], v)
                or (not k.startswith("residual_mlp.") and not torch.equal(actor[k], v))):
            raise ValueError("Safe source frozen base/distribution/reference changed")
    if any(reference[k].count_nonzero() for k in ("safety_mlp.4.weight", "safety_mlp.4.bias")):
        raise ValueError("Safe source preservation reference must have zero safety output")
    i, completed = saved.get("iter"), saved.get("preservation_updates")
    if (type(i) is not int or type(completed) is not int or i < 0
            or not (i == completed == 0 or completed == i+1)):
        raise ValueError("Safe source iteration counters disagree")
    tail, width = m.get("tail_curriculum", {}), m.get("width_curriculum", {})
    if type(tail.get("stage")) is not int or tail["stage"] not in (0, 1, 2):
        raise ValueError("Invalid Safe source material stage")
    s = SafeSettings(**frozen["environment"]["stair_tail"])
    s.validate()
    levels = width.get("level")
    if not isinstance(levels, list) or not levels:
        raise ValueError("Missing Safe source width targets")
    curriculum = WidthCurriculum(len(levels), "cpu", s)
    curriculum.load_state_dict(width)
    practice = torch.arange(len(levels))%s.material_practice_every == 0
    if (curriculum.level[practice] != s.material_practice_level).any():
        raise ValueError("Safe source material practice targets changed")
    c, w = len(safe_cases()), len(s.widths_m)
    if any(k not in tail for k in ("episodes", "successes", "stage_episodes", "stage_successes")):
        raise ValueError("Incomplete Safe source episode counters")
    counts = {k: checked_counts(tail.get(k), torch.zeros(c, dtype=torch.long))
              for k in ("episodes", "successes", "stage_episodes", "stage_successes")}
    stats = width.get("statistics", {})
    keys = ("episodes", "successes", "edge_failures", "recovery_timeouts", "stall_timeouts", "other_failures")
    if set(stats) != set(keys):
        raise ValueError("Incomplete Safe source width statistics")
    counts["width_statistics"] = {k: checked_counts(stats[k], torch.zeros(w, c, dtype=torch.long)) for k in keys}
    validate_safe_counts(counts, s)
    return m


def sources():
    result = safe_sources()
    base = PROJECT_ROOT/"source/rl_training/rl_training/tasks/manager_based/locomotion/velocity"
    cfg = base/"config/wheeled/deeprobotics_m20"
    paths = [*Path(__file__).parent.glob("*.py"), base/"mdp/stair_quiet_v1_transition_env.py",
        cfg/"stair_quiet_v1_transition_env_cfg.py", cfg/"agents/stair_quiet_v1_transition_ppo_cfg.py",
        cfg/"stair_quiet_v1_transition/__init__.py"]
    result.update({str(p.relative_to(PROJECT_ROOT)): file_sha256(p) for p in paths})
    return result


class StairQuietTransitionRunner(StairQuietSafeRunner):
    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        configured = copy.deepcopy(train_cfg)
        configured["actor"].update(class_name=ACTOR, residual_hidden_dims=[128, 64])
        configured["obs_groups"].update(actor=list(GROUPS), critic=["critic", "motion_control"])
        self._quiet_train_cfg = copy.deepcopy(configured)
        self._actor_model_config = copy.deepcopy(configured["actor"])
        OnPolicyRunner.__init__(self, env, configured, log_dir, device)
        self._restored = self._initialized = False
        self._cli_args, self._initialization, self._prior_manifest = {}, {}, {}
        self._guide_schedule = dict(guided_iterations=0, decay_iterations=0,
            base_weights={"entry_front_lead_weight": .5, "entry_front_progress_weight": .5, "entry_sequence_weight": .5})
        self._install_guide_schedule()
        self._install_quiet_schedule()

    def initialize_task(self, arguments):
        self._cli_args = copy.deepcopy(arguments)
        if any(arguments.get(k) for k in ("preserve_actor_from", "init_critic_from")):
            raise ValueError("Transition loads complete Safe actor and critic together")
        if self._restored:
            if arguments.get("init_actor_from"):
                raise ValueError("Transition resume cannot also initialize Safe weights")
            return
        if arguments.get("resume") or self.cfg.get("resume"):
            raise ValueError("Load a complete Transition checkpoint before resume")
        path = self._project_path(arguments.get("init_actor_from") or DEFAULT_START)
        sha = file_sha256(path)
        if not arguments.get("init_actor_from") and sha != DEFAULT_START_SHA256:
            raise ValueError("Default Safe18400 source changed")
        saved = torch.load(path, map_location="cpu", weights_only=True)
        teacher = torch.load(checked_path(TEACHER, TEACHER_SHA256), map_location="cpu", weights_only=True)
        prior = validate_safe_source(saved, teacher, self.alg.actor.state_dict(),
                                     self.alg.critic.state_dict(), self._actor_model_config)
        levels = prior["width_curriculum"]["level"]
        env, state = self.env.unwrapped, course_state(self.env.unwrapped)
        s = env.cfg.stair_tail
        if (not levels or any(type(x) is not int or not 0 <= x < len(s.widths_m) for x in levels)
                or tuple(prior["noise_refinement"]["contract"]["environment"]["stair_tail"]["widths_m"]) != tuple(s.widths_m)):
            raise ValueError("Safe source curriculum incompatible with Transition")
        self.alg.actor.load_state_dict(saved["actor_state_dict"], strict=True)
        self.alg.critic.load_state_dict(saved["critic_state_dict"], strict=True)
        self.alg.set_reference(saved["preservation_reference_actor"])
        self.alg.guard_updates = self.current_learning_iteration = 0
        if self.alg.optimizer.state:
            raise RuntimeError("Transition initialization requires a fresh optimizer")
        source = torch.tensor(levels, device=env.device)
        source_every = prior["noise_refinement"]["contract"]["environment"]["stair_tail"]["material_practice_every"]
        pool = source[torch.arange(len(levels), device=env.device)%source_every != 0]
        learning_ids = (~state["material_practice"]).nonzero().flatten()
        # Resample learning environments separately: striding all source IDs
        # can alias entirely onto every-fourth fixed-width material practice.
        if len(learning_ids):
            indices = torch.arange(len(learning_ids), device=env.device)*len(pool)//len(learning_ids)
            state["width_curriculum"].level[learning_ids] = pool[indices]
        state["width_curriculum"].level[state["material_practice"]] = s.material_practice_level
        state["width_curriculum"].success_streak.zero_()
        env.tail_stage = prior["tail_curriculum"]["stage"]
        self._initialization = dict(initial_actor_path=str(path), initial_actor_sha256=sha,
            initial_critic_path=str(path), initial_critic_sha256=sha,
            source_iteration=int(saved["iter"]), source_completed_updates=int(saved["preservation_updates"]),
            source_task="stair_quiet_v1_safe", source_material_stage=env.tail_stage,
            source_width_statistics=prior["width_curriculum"]["statistics"],
            critic_initialization="complete Safe critic; inherited critic warmup; fresh optimizer",
            continuation="complete Safe actor including trained safety branch, critic, preservation reference, width targets and material stage; fresh 35-case/protocol statistics, width streaks and optimizer",
            preservation_actor_sha256=TEACHER_SHA256)
        self._initialized = True
        self._reset_physics()
        self._install_quiet_schedule()

    def learn(self, *args, **kwargs):
        # train.py asks RSL for random episode lengths. A real entry/exit
        # sequence must not begin with a nearly expired episode timeout.
        kwargs["init_at_random_ep_len"] = False
        return super().learn(*args, **kwargs)

    def _record_manifest(self):
        manifest = IdleStopMotionRunner._record_manifest(self)
        env, state = self.env.unwrapped, course_state(self.env.unwrapped)
        frozen = contract(env.cfg, self._quiet_train_cfg)
        manifest.update(refinement_version=VERSION, input_routing_version=ROUTING,
            actor_model_config=copy.deepcopy(self._actor_model_config), actor_input_groups=dict(GROUPS))
        manifest["noise_refinement"] = dict(task=TASK, version=1, contract=frozen, contract_sha256=digest(frozen),
            source_sha256=sources(), teacher_baseline=dict(path=TEACHER, sha256=TEACHER_SHA256),
            wheel_collision_geometry=canonical(env.stair_quiet_geometry),
            control_contract=self._control_contract(), control_adapter=env.stair_quiet_landing.contract(),
            terrain_contacts=env.scene.terrain.contact_receipt, wheel_contacts=env.tail_wheel_material_receipt,
            acoustic_status="not_measured", deployment_status="requires_transition_actor_and_privileged_safe_adapter")
        manifest["tail_curriculum"] = dict(stage=env.tail_stage,
            case_metrics={k: v.cpu().tolist() for k, v in state["case_metrics"].items()},
            **{k: state[k].cpu().tolist() for k in ("episodes", "successes", "stage_episodes", "stage_successes", "assign_count")})
        manifest["width_curriculum"] = dict(**state["width_curriculum"].state_dict(),
            statistics={k: v.cpu().tolist() for k, v in state["width_statistics"].items()})
        manifest["safe_mechanics"] = {k: v.cpu().tolist() for k, v in state["mechanics"].items()}
        manifest["transition_statistics"] = {k: v.cpu().tolist() for k, v in state["transition_statistics"].items()}
        self.alg.m20_training_manifest = copy.deepcopy(manifest)
        return manifest

    def load(self, path, load_cfg=None, strict=True, map_location=None):
        if load_cfg is not None or not strict:
            raise ValueError("Transition resume requires complete strict state")
        saved = torch.load(path, map_location="cpu", weights_only=True)
        m = saved.get("m20_training_manifest", {})
        noise = m.get("noise_refinement", {})
        frozen = contract(self.env.unwrapped.cfg, self._quiet_train_cfg)
        if (m.get("refinement_version") != VERSION or m.get("input_routing_version") != ROUTING
                or m.get("actor_model_config") != self._actor_model_config or m.get("actor_input_groups") != GROUPS
                or noise.get("task") != TASK or noise.get("version") != 1
                or noise.get("contract") != frozen or noise.get("contract_sha256") != digest(frozen)
                or noise.get("source_sha256") != sources()):
            raise ValueError("Transition resume task/source/physics/scan/optimization contract differs")
        for key, expected in (("actor_state_dict", self.alg.actor.state_dict()),
                              ("critic_state_dict", self.alg.critic.state_dict()),
                              ("preservation_reference_actor", self.alg.actor.state_dict())):
            validate_model(saved.get(key), expected)
        if "optimizer_state_dict" not in saved:
            raise ValueError("Missing Transition optimizer")
        validate_optimizer(saved["optimizer_state_dict"])
        if noise.get("control_contract") != self._control_contract():
            raise ValueError("Transition landing controller changed")
        i, completed = saved.get("iter"), saved.get("preservation_updates")
        if (type(i) is not int or type(completed) is not int or i < 0
                or not (i == completed == 0 or completed == i+1)):
            raise ValueError("Transition iteration counters disagree")
        teacher = torch.load(checked_path(TEACHER, TEACHER_SHA256), map_location="cpu", weights_only=True)
        actor, reference = saved["actor_state_dict"], saved["preservation_reference_actor"]
        for k, v in teacher["actor_state_dict"].items():
            if (not torch.equal(reference[k], v)
                    or (not k.startswith("residual_mlp.") and not torch.equal(actor[k], v))):
                raise ValueError("Transition frozen base/distribution/reference changed")
        if any(reference[k].count_nonzero() for k in ("safety_mlp.4.weight", "safety_mlp.4.bias")):
            raise ValueError("Transition reference safety output must be zero")
        env, current = self.env.unwrapped, course_state(self.env.unwrapped)
        tail, width = m.get("tail_curriculum", {}), m.get("width_curriculum", {})
        if type(tail.get("stage")) is not int or tail["stage"] not in (0, 1, 2):
            raise ValueError("Invalid Transition material stage")
        restored = WidthCurriculum(env.num_envs, env.device, env.cfg.stair_tail)
        restored.load_state_dict(width)
        if (restored.level[current["material_practice"]] != env.cfg.stair_tail.material_practice_level).any():
            raise ValueError("Transition practice rows changed")
        staged = {k: checked_counts(tail[k], current[k]) for k in ("episodes", "successes", "stage_episodes", "stage_successes", "assign_count")}
        staged["width_curriculum"] = restored
        staged["width_statistics"] = {k: checked_counts(width["statistics"][k], v) for k, v in current["width_statistics"].items()}
        for group, raw in (("case_metrics", tail.get("case_metrics", {})),
                           ("mechanics", m.get("safe_mechanics", {})),
                           ("transition_statistics", m.get("transition_statistics", {}))):
            if set(raw) != set(current[group]):
                raise ValueError("Incomplete Transition "+group)
            staged[group] = {}
            for key, target in current[group].items():
                v = torch.as_tensor(raw[key], dtype=target.dtype, device=env.device)
                if v.shape != target.shape or not torch.isfinite(v).all() or (v < 0).any():
                    raise ValueError("Invalid Transition "+group+" evidence")
                staged[group][key] = v
        validate_transition_counts(staged)
        self.alg.load(saved, strict=True)
        self._prior_manifest = copy.deepcopy(m)
        self._initialization = dict(resumed_checkpoint=str(Path(path).resolve()), resumed_checkpoint_sha256=file_sha256(path))
        self._restored = self._initialized = True
        self.current_learning_iteration = int(self.alg.guard_updates)
        env.tail_stage = tail["stage"]
        for key, value in staged.items():
            if key == "width_curriculum":
                current[key].load_state_dict(value.state_dict())
            elif isinstance(value, dict):
                for name, tensor in value.items():
                    current[key][name].copy_(tensor)
            else:
                current[key].copy_(value)
        self._reset_physics()
        self._install_guide_schedule()
        self._install_quiet_schedule()
        return saved.get("infos")
