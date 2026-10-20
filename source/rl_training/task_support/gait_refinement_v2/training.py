"""Gait refinement v2 PPO, replay preservation and runner provenance."""
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

from task_support.stair_teacher.ppo import StairGuardedPPO
from task_support.quiet_landing.runner import QuietLandingRunner
from rl_training.tasks.manager_based.locomotion.velocity.mdp.gait_refinement_v2_env import idle_mask, idle_preservation_loss

PROJECT_ROOT = Path(__file__).resolve().parents[4]


class GaitRefinementV2PPO(StairGuardedPPO):
    def __init__(self, *args, replay_coefficient=5.0, replay_batch_size=256,
                 freeze_actor_front=True, correction_replay_manifest=None, **kwargs):
        super().__init__(*args, **kwargs)
        if replay_coefficient < 0 or replay_batch_size <= 0:
            raise ValueError("Invalid v2 replay preservation settings")
        self.corrections = None  # Assigned from the final env_cfg by the runner.
        self.correction_replay_manifest = correction_replay_manifest
        self.replay_coefficient = replay_coefficient
        self.replay_batch_size = replay_batch_size
        self.replay_inputs = None
        self.replay_receipt = None
        if freeze_actor_front:
            for name, parameter in self.actor.named_parameters():
                if name.startswith(("mlp.0.", "mlp.2.", "distribution.")):
                    parameter.requires_grad_(False)

    def _actor_update_mask(self, observations):
        policy = observations["policy"]
        settings = self.corrections
        idle = idle_mask(policy, settings.idle.command_threshold, settings.idle.flat_range)
        straight = (torch.linalg.vector_norm(policy[:, 6:8], dim=-1) > settings.heading.translation_threshold)
        straight &= policy[:, 8].abs() < settings.heading.intentional_turn_threshold
        straight &= torch.isfinite(policy).all(-1)
        return (idle & settings.idle_stop_enabled) | (straight & settings.stair_heading_enabled)

    def set_replay_inputs(self, manifest_path):
        if self.reference_actor is None:
            raise RuntimeError("Initialize the fixed reference before loading v2 replay")
        manifest_path = Path(manifest_path).expanduser()
        if not manifest_path.is_absolute():
            manifest_path = PROJECT_ROOT / manifest_path
        manifest_path = manifest_path.resolve()
        manifest = json.loads(manifest_path.read_text())
        arrays, hashes = [], {}
        for name in manifest["inputs"]:
            path = Path(name)
            if not path.is_absolute():
                local = manifest_path.parent / path
                path = local if local.is_file() else PROJECT_ROOT / path
            path = path.resolve()
            data = np.load(path)
            value = data["policy"]
            if value.ndim != 2 or value.shape[1] != 244 or value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError(f"Invalid actual-observation replay: {path}")
            actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual_hash != manifest["input_sha256"][name]:
                raise ValueError(f"Replay identity mismatch: {path}")
            arrays.append(torch.from_numpy(value.copy()))
            hashes[name] = actual_hash
            data.close()
        self.replay_inputs = torch.cat(arrays).to(self.device)
        mask = idle_mask(self.replay_inputs, self.corrections.idle.command_threshold, self.corrections.idle.flat_range)
        if not mask.any() or mask.all():
            raise ValueError("Replay requires both stopping and moving observations")
        self.replay_idle_ids = torch.where(mask)[0]
        self.replay_moving_ids = torch.where(~mask)[0]
        self.replay_receipt = {"manifest": str(manifest_path), "input_sha256": hashes,
                               "samples": len(self.replay_inputs), "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()}

    def _preservation(self, observations, uneven_weight):
        with torch.no_grad():
            target = self.reference_actor(observations)
            std = self.reference_actor.distribution.log_std_param.exp()
        prediction = self.actor(observations)
        idle = idle_mask(observations["policy"], self.corrections.idle.command_threshold, self.corrections.idle.flat_range)
        return idle_preservation_loss(prediction, target, std,
            idle & self.corrections.idle_stop_enabled, self.corrections.idle, uneven_weight)

    def _anchor_loss(self, observations):
        policy = observations["policy"]
        uneven = (policy[:, 57:244].amax(-1) - policy[:, 57:244].amin(-1)) > self.corrections.heading.stair_height_range
        weight = torch.where(uneven, self.anchor_uneven_weight, 1.0)
        loss = self._preservation(observations, weight)
        if self.replay_inputs is not None and self.replay_coefficient:
            half = self.replay_batch_size // 2
            ids = torch.cat([
                self.replay_idle_ids[torch.randint(len(self.replay_idle_ids), (half,), device=self.device)],
                self.replay_moving_ids[torch.randint(len(self.replay_moving_ids), (self.replay_batch_size-half,), device=self.device)]])
            replay = TensorDict({"policy": self.replay_inputs[ids]}, batch_size=[len(ids)])
            loss = loss + self.replay_coefficient * self._preservation(replay, 1.0)
        return loss

    def update(self):
        if self.reference_actor is None:
            raise RuntimeError("v2 PPO requires an explicit task-capable preservation reference")
        return super().update()

    def save(self):
        data = super().save()
        data["gait_v2_replay"] = self.replay_receipt
        return data


class GaitRefinementV2Runner(QuietLandingRunner):
    """PPO/replay integration and embedded training provenance."""
    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        super().__init__(env, train_cfg, log_dir, device)
        self.alg.corrections = env.cfg.gait_corrections
        self.v2_provenance = {
            "revision": env.cfg.gait_corrections.revision,
            "corrections": asdict(env.cfg.gait_corrections),
            "algorithm": train_cfg["algorithm"],
            "actor_layout": "244->512->256->128->16 ELU",
            "training_method": "Isaac Lab PPO with idle targets and replay preservation",
        }
        root = PROJECT_ROOT
        cfg = root / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20"
        support = root / "source/rl_training/task_support"
        paths = [root / "source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/gait_refinement_v2_env.py",
                 cfg / "gait_refinement_v2_env_cfg.py", cfg / "agents/gait_refinement_v2_ppo_cfg.py",
                 Path(__file__).resolve(), support / "stair_teacher/ppo.py",
                 support / "quiet_landing/runner.py",
                 root / "scripts/reinforcement_learning/rsl_rl/train.py"]
        # Embed task configuration and implementation sources in the checkpoint.
        self.v2_provenance["source_text"] = {
            str(path.relative_to(root)): path.read_text() for path in paths}
        inherited = list((root / "source/rl_training/rl_training").rglob("*.py"))
        inherited += list(support.rglob("*.py"))
        inherited += [root / "scripts/reinforcement_learning/rsl_rl/cli_args.py"]
        self.v2_provenance["source_sha256"] = {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(set(paths + inherited))}

    def record_initialization(self, arguments):
        self.v2_provenance["task"] = arguments["task"]
        for key in ("init_actor_from", "init_critic_from", "preserve_actor_from"):
            if arguments.get(key):
                path = Path(arguments[key]).expanduser().resolve()
                self.v2_provenance[key] = {
                    "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def learn(self, *args, **kwargs):
        if self.alg.reference_actor is None:
            raise RuntimeError("v2 requires --init_actor_from or --preserve_actor_from through train.py")
        if self.alg.replay_coefficient:
            manifest = self.alg.correction_replay_manifest
            if not manifest:
                raise RuntimeError("v2 replay needs --correction_replay_manifest or its PPO configuration value")
            self.alg.set_replay_inputs(manifest)
            self.v2_provenance["replay"] = self.alg.replay_receipt
        return super().learn(*args, **kwargs)

    def load(self, path, load_cfg=None, strict=True, map_location=None):
        resolved = Path(path).resolve()
        payload = resolved.read_bytes()
        loaded = torch.load(io.BytesIO(payload), weights_only=False, map_location=map_location)
        load_iteration = self.alg.load(loaded, load_cfg, strict)
        if load_iteration:
            self.current_learning_iteration = loaded["iter"]
        self.v2_provenance["resumed_checkpoint"] = {
            "path": str(resolved), "sha256": hashlib.sha256(payload).hexdigest(),
            "iteration": loaded.get("iter"), "parent": loaded.get("gait_corrections_v2")}
        return loaded.get("infos")

    def save(self, path, infos=None):
        data = self.alg.save()
        data.update(iter=self.current_learning_iteration, infos=infos,
                    gait_corrections_v2=self.v2_provenance)
        torch.save(data, path)
        self.logger.save_model(path, self.current_learning_iteration)
