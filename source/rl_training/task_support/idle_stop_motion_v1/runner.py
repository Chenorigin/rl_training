"""Standalone idle-stop motion runner; all initialization stays task-specific."""
from __future__ import annotations

from contextlib import redirect_stdout
import copy
import hashlib
from io import StringIO
import json
import math
from pathlib import Path
import statistics
import time
from types import MethodType

import torch
from rsl_rl.runners import OnPolicyRunner

from .training import (ACTOR_CLASS, ACTOR_SOURCE_SHA256, BASELINE_CHECKPOINT_SHA256,
                       BASELINE_CONFIG_SHA256, CONTROL_PROFILE, LEGACY_ACTOR_CLASS,
                       MOTION_CONTROL_VERSION, PROFILE, VERSION, configure_models)

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_BASELINE = "deploy/models/idle_stop_v1/model_257100_idle_stop.pt"


def pool_episode_metrics(extras: list[dict]) -> list[dict]:
    """Resolve mean/RMS/min metadata, preserving unweighted ordinary metrics."""
    if not extras:
        return []
    kinds = {
        "weighted": ("__weighted_total__/", "__weighted_count__/"),
        "rms": ("__rms_total__/", "__rms_count__/"),
        "minimum": ("__minimum_value__/", "__minimum_count__/"),
    }
    fields = {}
    metadata = set()
    for row in extras:
        for key in row:
            for kind, (value_marker, count_marker) in kinds.items():
                for marker in (value_marker, count_marker):
                    token = "/" + marker
                    if key.startswith(marker):
                        prefix, suffix = "", key[len(marker):]
                    elif token in key:
                        prefix, suffix = key.split(token, 1)
                        prefix += "/"
                    else:
                        continue
                    target = prefix + suffix
                    fields[(kind, target)] = (prefix + value_marker + suffix,
                                              prefix + count_marker + suffix)
                    metadata.add(key)
    output = [{key: value for key, value in row.items() if key not in metadata} for row in extras]
    entries, pairs = [], []
    for (kind, target), (value_key, count_key) in fields.items():
        selected = [(row[value_key], row[count_key]) for row in extras
                    if value_key in row and count_key in row]
        entries.append((kind, target, len(pairs), len(selected)))
        pairs.extend(selected)
    if not pairs:
        return output
    reference = next((x for pair in pairs for x in pair if isinstance(x, torch.Tensor)), None)
    device = reference.device if reference is not None else "cpu"
    # One host transfer for all diagnostics, rather than synchronizing CUDA
    # twice per field. Metadata is detached; it never affects PPO gradients.
    packed = torch.stack([
        torch.stack([torch.as_tensor(pair[i], device=device).detach().reshape(()) for pair in pairs])
        for i in (0, 1)
    ]).cpu().tolist()
    targets, pooled = [], []
    for kind, target, start, size in entries:
        valid = [(v, c) for v, c in zip(packed[0][start:start + size], packed[1][start:start + size])
                 if c > 0 and math.isfinite(c) and math.isfinite(v)]
        if not valid:
            for row in output:
                row.pop(target, None)
            continue
        value = min(v for v, _ in valid) if kind == "minimum" else sum(v for v, _ in valid) / sum(c for _, c in valid)
        if kind == "rms":
            value = math.sqrt(max(0.0, value))
        if not math.isfinite(value):
            raise ValueError(f"Non-finite pooled diagnostic: {target}")
        targets.append(target)
        pooled.append(value)
    values = torch.tensor(pooled, device=device, dtype=torch.float32)
    # RSL's logger only iterates the first dict's keys. Replicate pooled values
    # so a measurement appearing in a later reset batch is still recorded.
    for target, value in zip(targets, values):
        for row in output:
            row[target] = value
    return output



def episode_means(rows):
    """Match RSL's concatenated means, with one packed GPU-to-host transfer."""
    if not rows:
        return {}
    parts, spans, host, count = [], {}, {}, {}
    cursor = 0
    for key in rows[0]:
        spans[key], host[key], count[key] = [], 0., 0
        for row in rows:
            if key not in row:
                continue
            value = torch.as_tensor(row[key]).detach().reshape(-1)
            count[key] += value.numel()
            if value.is_cuda:
                parts.append(value)
                spans[key].append((cursor, cursor+value.numel()))
                cursor += value.numel()
            else:
                host[key] += float(value.float().sum())
    packed = torch.cat(parts).float().cpu().tolist() if parts else []
    return {key: (host[key]+sum(sum(packed[a:b]) for a,b in spans[key]))/count[key]
            for key in spans if count[key]}



def write_iteration(logger, *, it, collect_time, learn_time, loss_dict, learning_rate, action_std, **kwargs):
    if logger.writer is None:
        return
    size = logger.cfg['num_steps_per_env']*logger.num_envs*logger.gpu_world_size
    logger.tot_timesteps += size
    logger.tot_time += collect_time+learn_time
    metrics = episode_means(logger.ep_extras)
    for key, value in metrics.items():
        logger.writer.add_scalar(key if '/' in key else 'Episode/'+key, value, it)
    for key, value in loss_dict.items():
        logger.writer.add_scalar('Loss/'+key, value, it)
    logger.writer.add_scalar('Loss/learning_rate', learning_rate, it)
    logger.writer.add_scalar('Policy/mean_std', float(action_std.mean()), it)
    logger.writer.add_scalar('Perf/total_fps', int(size/(collect_time+learn_time)), it)
    logger.writer.add_scalar('Perf/collection_time', collect_time, it)
    logger.writer.add_scalar('Perf/learning_time', learn_time, it)
    if logger.rewbuffer:
        reward, length = statistics.mean(logger.rewbuffer), statistics.mean(logger.lenbuffer)
        logger.writer.add_scalar('Train/mean_reward', reward, it)
        logger.writer.add_scalar('Train/mean_episode_length', length, it)
        if logger.logger_type != 'wandb':
            logger.writer.add_scalar('Train/mean_reward/time', reward, int(logger.tot_time))
            logger.writer.add_scalar('Train/mean_episode_length/time', length, int(logger.tot_time))
    logger.ep_extras.clear()



def _descent_guide_scale(iteration, guided_iterations, decay_iterations):
    if iteration < guided_iterations:
        return 1.0
    if decay_iterations <= 0:
        return 0.0
    return max(0.0, 1.0 - (iteration - guided_iterations) / decay_iterations)



def use_compact_console_logging(logger, fast=False, descent_guide_schedule=None):
    """Keep RSL-RL's full TensorBoard logging while shortening its console output."""
    original_log = logger.log

    def compact_log(self, *args, **kwargs):
        # Logger.log writes every scalar to TensorBoard before printing its full table.
        # Capture only that table; the writer and episode buffers remain untouched.
        log_start = time.perf_counter()
        self.ep_extras[:] = pool_episode_metrics(self.ep_extras)
        if fast and self.logger_type == 'tensorboard' and not self.cfg['algorithm']['rnd_cfg']:
            write_iteration(self, **kwargs)
        else:
            with redirect_stdout(StringIO()):
                original_log(*args, **kwargs)
        log_elapsed = time.perf_counter()-log_start
        if fast and self.writer is not None:
            self.writer.add_scalar('Perf/logging_time', log_elapsed, kwargs['it'])

        iteration = kwargs["it"]
        if descent_guide_schedule and self.writer is not None:
            guide_scale = _descent_guide_scale(
                iteration, descent_guide_schedule["guided_iterations"],
                descent_guide_schedule["decay_iterations"])
            self.writer.add_scalar("Curriculum/descent_entry_guide_scale", guide_scale, iteration)
            for name, weight in descent_guide_schedule["base_weights"].items():
                self.writer.add_scalar(f"Curriculum/descent_entry_guide_weight/{name}",
                                       weight * guide_scale, iteration)
        start_iteration = kwargs["start_it"]
        total_iterations = kwargs["total_it"]
        collect_time = kwargs["collect_time"]
        learn_time = kwargs["learn_time"]
        loss_dict = kwargs["loss_dict"]
        action_std = kwargs["action_std"]

        samples = self.cfg["num_steps_per_env"] * self.num_envs * self.gpu_world_size
        iteration_time = collect_time + learn_time
        fps = int(samples / iteration_time) if iteration_time > 0 else 0
        mean_reward = statistics.mean(self.rewbuffer) if self.rewbuffer else None
        mean_episode_length = statistics.mean(self.lenbuffer) if self.lenbuffer else None
        completed = iteration + 1 - start_iteration
        remaining = total_iterations - iteration - 1
        eta_seconds = self.tot_time / completed * remaining if completed > 0 else 0.0

        def metric(value, precision=3):
            return f"{value:.{precision}f}" if value is not None else "n/a"

        print(
            f"[train] {iteration + 1}/{total_iterations}"
            f" | reward={metric(mean_reward, 2)}"
            f" | ep_len={metric(mean_episode_length, 0)}"
            f" | fps={fps}"
            f" | action_std={metric(action_std.mean().item(), 3)}"
            f" | value_loss={metric(loss_dict.get('value'), 4)}"
            f" | policy_loss={metric(loss_dict.get('surrogate'), 4)}"
            f" | iter={iteration_time:.2f}s"
            f" | ETA={eta_seconds / 3600:.1f}h"
        )

    logger.log = MethodType(compact_log, logger)




def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_resume_manifest(manifest):
    """Accept original callable names while enforcing the recorded residual ABI."""
    expected = {
        "refinement_version": VERSION, "m20_profile": PROFILE,
        "control_profile": CONTROL_PROFILE, "motion_control_version": MOTION_CONTROL_VERSION,
        "actor_input_groups": {"policy": 244, "motion_control": 3},
        "critic_input_groups": {"critic": 285, "motion_control": 3},
        "baseline_actor_sha256": BASELINE_CHECKPOINT_SHA256,
        "baseline_config_sha256": BASELINE_CONFIG_SHA256,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"IdleStopMotion resume manifest differs: {key}")
    actor = manifest.get("actor_model_config", {})
    if actor.get("class_name") not in (LEGACY_ACTOR_CLASS, ACTOR_CLASS):
        raise ValueError("IdleStopMotion requires its complete residual actor")
    expected_actor = {"hidden_dims": [512, 256, 128], "residual_hidden_dims": [128, 64],
                      "activation": "elu", "obs_normalization": False,
                      "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "log"}}
    if any(actor.get(key) != value for key, value in expected_actor.items()):
        raise ValueError("IdleStopMotion actor architecture differs from its recorded contract")
    if manifest.get("source_sha256", {}).get("motion_residual_actor.py") != ACTOR_SOURCE_SHA256:
        raise ValueError("IdleStopMotion actor source identity differs")
    if file_sha256(Path(__file__).with_name("motion_residual_actor.py")) != ACTOR_SOURCE_SHA256:
        raise ValueError("IdleStopMotion actor implementation changed")
    return copy.deepcopy(manifest)


class IdleStopMotionRunner(OnPolicyRunner):
    """Frozen baseline initialization, strict full resume and guide provenance."""

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        configured = configure_models(train_cfg)
        # RSL-RL consumes class_name entries while constructing its models.
        self._actor_model_config = copy.deepcopy(configured["actor"])
        super().__init__(env, configured, log_dir, device)
        self._restored = False
        self._initialized = False
        self._cli_args = {}
        self._initialization = {}
        self._prior_manifest = {}
        self._guide_schedule = {
            "guided_iterations": int(self.cfg.get("descent_guide_iters", 20000)),
            "decay_iterations": int(self.cfg.get("descent_guide_decay_iters", 10000)),
            "base_weights": {"entry_front_lead_weight": 0.5,
                             "entry_front_progress_weight": 0.5, "entry_sequence_weight": 0.5},
        }
        if min(self._guide_schedule["guided_iterations"], self._guide_schedule["decay_iterations"]) < 0:
            raise ValueError("Descent guide iteration counts must be non-negative")
        self._install_guide_schedule()

    def _install_guide_schedule(self):
        self.env.unwrapped._m20_descent_entry_guide_schedule = {
            **copy.deepcopy(self._guide_schedule), "steps_per_iteration": int(self.cfg["num_steps_per_env"]),
            "iteration_offset": int(self.current_learning_iteration),
        }

    @staticmethod
    def _project_path(value):
        path = Path(value).expanduser()
        return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()

    def initialize_task(self, arguments):
        """Called by train.py instead of generic actor/critic/reference initialization."""
        self._cli_args = copy.deepcopy(arguments)
        if arguments.get("preserve_actor_from"):
            raise ValueError("IdleStopMotion freezes its initialized baseline; omit --preserve_actor_from")
        if self._restored:
            if arguments.get("init_actor_from") or arguments.get("init_critic_from"):
                raise ValueError("Full IdleStopMotion resume cannot also initialize weights")
            self._initialized = True
            return
        if arguments.get("resume") or self.cfg.get("resume"):
            raise RuntimeError("IdleStopMotion resume requires runner.load before initialization")
        path = self._project_path(arguments.get("init_actor_from") or self.cfg.get("baseline_checkpoint", DEFAULT_BASELINE))
        if file_sha256(path) != BASELINE_CHECKPOINT_SHA256:
            raise ValueError("IdleStopMotion must initialize from the recorded idle-stop checkpoint")
        saved = torch.load(path, map_location="cpu", weights_only=True)
        self.alg.actor.load_baseline_state(saved["actor_state_dict"])
        self._initialization = {"initial_actor_path": str(path), "initial_actor_sha256": file_sha256(path),
                                "critic_initialization": "random; original 20-update critic warmup"}
        critic_path = arguments.get("init_critic_from") or self.cfg.get("initial_critic_checkpoint")
        if critic_path:
            critic_path = self._project_path(critic_path)
            critic = torch.load(critic_path, map_location="cpu", weights_only=True)
            # Match upstream: no dimension padding and no optimizer restoration.
            self.alg.critic.load_state_dict(critic["critic_state_dict"], strict=True)
            self._initialization.update(initial_critic_path=str(critic_path),
                                         initial_critic_sha256=file_sha256(critic_path),
                                         critic_initialization="explicit shape-compatible state; fresh optimizer")
        self._initialized = True

    def load(self, path, load_cfg=None, strict=True, map_location=None):
        if not strict:
            raise ValueError("IdleStopMotion full resume requires strict state loading")
        checkpoint = torch.load(path, map_location=map_location or "cpu", weights_only=True)
        required = {"actor_state_dict", "critic_state_dict", "optimizer_state_dict",
                    "iter", "preservation_updates", "m20_training_manifest"}
        missing = required.difference(checkpoint)
        if missing:
            raise ValueError(f"IdleStopMotion full resume is missing state: {sorted(missing)}")
        for counter in ("iter", "preservation_updates"):
            if type(checkpoint[counter]) is not int or checkpoint[counter] < 0:
                raise ValueError(f"IdleStopMotion resume requires a nonnegative integer {counter}")
        prior = validate_resume_manifest(checkpoint.get("m20_training_manifest") or {})
        load_iteration = self.alg.load(checkpoint, load_cfg, strict)
        if load_iteration:
            self.current_learning_iteration = checkpoint["iter"]
            self._restored = True
            self._initialized = True
            self._prior_manifest = prior
            self._guide_schedule = copy.deepcopy(prior.get("descent_entry_guide_schedule", {
                "guided_iterations": 0, "decay_iterations": 0,
                "base_weights": self._guide_schedule["base_weights"],
            }))
            self._initialization = {"resumed_checkpoint": str(Path(path).resolve()),
                                    "resumed_checkpoint_sha256": file_sha256(path)}
        self.alg.m20_training_manifest = prior
        self._install_guide_schedule()
        return checkpoint.get("infos")

    def record_initialization(self, arguments):
        self._cli_args = copy.deepcopy(arguments)
        if not self._initialized:
            self.initialize_task(arguments)
        self._record_manifest()

    def _record_manifest(self):
        from . import profile
        folder = Path(__file__).resolve().parent
        actor_config = copy.deepcopy(self._actor_model_config)
        env_cfg = self.env.unwrapped.cfg
        # Preserve the original manifest ABI consumed by existing deploy/student code.
        actor_config["class_name"] = LEGACY_ACTOR_CLASS
        manifest = copy.deepcopy(self._prior_manifest)
        manifest.update(
            refinement_version=VERSION, m20_profile=PROFILE, control_profile=CONTROL_PROFILE,
            motion_control_version=MOTION_CONTROL_VERSION, actor_model_config=actor_config,
            actor_input_groups={"policy": 244, "motion_control": 3},
            critic_input_groups={"critic": 285, "motion_control": 3},
            baseline_actor_sha256=BASELINE_CHECKPOINT_SHA256,
            baseline_config_sha256=BASELINE_CONFIG_SHA256,
            cli_args=self._cli_args, stair_tuning=copy.deepcopy(profile.TUNING),
            descent_entry_guide_schedule=copy.deepcopy(self._guide_schedule),
            teacher_detailed_diagnostics=bool(getattr(env_cfg, "teacher_detailed_diagnostics", False)),
            preservation_actor_sha256=manifest.get("preservation_actor_sha256"),
        )
        manifest.update(self._initialization)
        manifest["reward_weights"] = {name: term.weight for name, term in vars(env_cfg.rewards).items()
                                      if term is not None and hasattr(term, "weight")}
        manifest["source_sha256"] = copy.deepcopy(profile.SOURCE_SHA256)
        manifest["source_sha256"]["motion_residual_actor.py"] = file_sha256(folder / "motion_residual_actor.py")
        cfg = PROJECT_ROOT / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20"
        sources = [*folder.glob("*.py"), profile.PROFILE_PATH,
                   cfg / "idle_stop_motion_v1_env_cfg.py", cfg / "agents/idle_stop_motion_v1_ppo_cfg.py",
                   PROJECT_ROOT / "scripts/reinforcement_learning/rsl_rl/train.py"]
        manifest["integration_source_sha256"] = {
            str(path.relative_to(PROJECT_ROOT)): file_sha256(path) for path in sorted(set(sources))}
        manifest["train_script_sha256"] = file_sha256(PROJECT_ROOT / "scripts/reinforcement_learning/rsl_rl/train.py")
        self.alg.m20_training_manifest = json.loads(json.dumps(manifest, default=str))
        return self.alg.m20_training_manifest

    def learn(self, *args, **kwargs):
        if not self._initialized:
            self.initialize_task({})
        self._install_guide_schedule()
        manifest = self._record_manifest()
        use_compact_console_logging(self.logger, fast=True, descent_guide_schedule=self._guide_schedule)
        if self.logger.log_dir is not None:
            target = Path(self.logger.log_dir) / "params/launch_manifest.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(manifest, indent=2) + "\n")
        return super().learn(*args, **kwargs)
