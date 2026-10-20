"""Deterministic online action distillation with bounded dataset aggregation."""

import math

import torch
from tensordict import TensorDict
from rsl_rl.algorithms import Distillation

from .contract import STUDENT_OBS_DIM, TEACHER_OBS_DIM, ACTION_DIM


class StudentV0Distillation(Distillation):
    def __init__(self, *args, teacher_ratio=0.5, batch_size=1024, replay_rollouts=4, **kwargs):
        super().__init__(*args, **kwargs)
        if self.is_multi_gpu:
            raise ValueError("V0 deterministic replay distillation supports one GPU/process")
        if not math.isfinite(teacher_ratio) or not 0 <= teacher_ratio <= 1:
            raise ValueError("teacher_ratio must be in [0, 1]")
        for name, value in (("batch_size", batch_size), ("replay_rollouts", replay_rollouts),
                            ("num_learning_epochs", self.num_learning_epochs)):
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.student.is_recurrent or self.teacher.is_recurrent:
            raise ValueError("V0 distillation is restricted to feedforward models")
        if self.student.obs_dim != STUDENT_OBS_DIM or self.teacher.obs_dim != TEACHER_OBS_DIM:
            raise ValueError("Expected student 618D / teacher 244D")
        if self.student.obs_normalization or self.teacher.obs_normalization:
            raise ValueError("V0 teacher warm start requires normalization disabled")
        self.teacher.requires_grad_(False)
        self.teacher.eval()
        if self.student.distribution is not None:
            self.student.distribution.requires_grad_(False)
        # Only the deterministic action network is optimized.
        self.optimizer = torch.optim.Adam(self.student.mlp.parameters(), lr=self.learning_rate)
        self.teacher_ratio, self.batch_size = teacher_ratio, batch_size
        self.rollout_capacity = self.storage.num_envs * self.storage.num_transitions_per_env
        self.capacity = self.rollout_capacity * replay_rollouts
        self.replay_obs = torch.zeros(self.capacity, STUDENT_OBS_DIM, device=self.device)
        self.replay_labels = torch.zeros(self.capacity, ACTION_DIM, device=self.device)
        self.replay_size = self.replay_cursor = self.rollout_samples = 0
        self.teacher_mask = None
        self.executed_teacher = self.executed_total = 0
        self.provenance = {}

    def act(self, obs):
        if not self.teacher_loaded:
            raise RuntimeError("Load and freeze a teacher before collecting labels")
        student_obs, teacher_obs = obs["policy"], obs["teacher"]
        if not torch.isfinite(student_obs).all() or not torch.isfinite(teacher_obs).all():
            raise ValueError("Nonfinite observation: no teacher label or action may be collected")
        teacher_action = self.teacher(obs).detach()
        student_action = self.student(obs).detach()
        if not torch.isfinite(teacher_action).all() or not torch.isfinite(student_action).all():
            raise ValueError("Nonfinite deterministic action")
        n = student_obs.shape[0]
        if self.teacher_mask is None:
            self.teacher_mask = torch.rand(n, device=self.device) < self.teacher_ratio
        actions = torch.where(self.teacher_mask[:, None], teacher_action, student_action)
        # Initialize distribution statistics for the existing RSL logger, without sampling.
        if self.student.distribution is not None:
            self.student.distribution.update(student_action)
        self.transition.actions = actions
        self.transition.privileged_actions = teacher_action
        self.transition.observations = obs.clone()
        self.executed_teacher += int(self.teacher_mask.sum())
        self.executed_total += n
        return actions

    def process_env_step(self, obs, rewards, dones, extras):
        # Commit only after a successful environment step. Labels correspond to
        # transition.observations (pre-step), never the returned next observation.
        before = self.transition.observations["policy"]
        labels = self.transition.privileged_actions
        n = before.shape[0]
        indices = (torch.arange(n, device=self.device) + self.replay_cursor) % self.capacity
        self.replay_obs[indices] = before
        self.replay_labels[indices] = labels
        self.replay_cursor = (self.replay_cursor + n) % self.capacity
        self.replay_size = min(self.capacity, self.replay_size + n)
        self.rollout_samples += n
        super().process_env_step(obs, rewards, dones, extras)
        ids = dones.flatten().bool().nonzero().flatten()
        if ids.numel():
            self.teacher_mask[ids] = torch.rand(ids.numel(), device=self.device) < self.teacher_ratio

    def update(self):
        if self.rollout_samples == 0:
            raise RuntimeError("No successfully executed samples in this rollout")
        recent_count = min(self.rollout_samples, self.replay_size)
        recent = (torch.arange(recent_count, device=self.device) + self.replay_cursor - recent_count) % self.capacity
        total = torch.zeros(3, device=self.device)
        updates = 0
        for _ in range(self.num_learning_epochs):
            for _ in range(math.ceil(self.rollout_samples / self.batch_size)):
                n = min(self.batch_size, self.rollout_samples)
                # Retain recent recovery states while rehearsing older states.
                current = recent[torch.randint(recent_count, (n//2,), device=self.device)]
                history = torch.randint(self.replay_size, (n-n//2,), device=self.device)
                ids = torch.cat((current, history))
                batch = TensorDict({"policy": self.replay_obs[ids]}, batch_size=[n])
                predicted = self.student(batch)
                error = (predicted - self.replay_labels[ids]).square()
                loss = error.mean()
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite distillation loss")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.student.mlp.parameters(), self.max_grad_norm or 1.0,
                                              error_if_nonfinite=True)
                self.optimizer.step()
                total += torch.stack((loss.detach(), error[:, :12].mean().detach(), error[:, 12:].mean().detach()))
                updates += 1
        self.num_updates += 1
        self.storage.clear()
        values = total / updates
        stats = {"behavior": float(values[0]), "leg_mse": float(values[1]), "wheel_mse": float(values[2]),
                 "teacher_executed_fraction": self.executed_teacher / self.executed_total,
                 "replay_samples": float(self.replay_size)}
        self.rollout_samples = self.executed_teacher = self.executed_total = 0
        return stats

    def save(self):
        state = super().save()
        state.update({
            # Actor alias allows downstream actor-only tools to read the model.
            "actor_state_dict": state["student_state_dict"],
            "student_v0": {"version": 1, "teacher_ratio": self.teacher_ratio,
                           "replay_capacity": self.capacity,
                           "num_updates": self.num_updates, "provenance": self.provenance,
                           "replay_size": self.replay_size, "replay_cursor": self.replay_cursor},
            "replay_obs": self.replay_obs[:self.replay_size].clone(),
            "replay_labels": self.replay_labels[:self.replay_size].clone(),
        })
        return state

    def load(self, loaded_dict, load_cfg=None, strict=True):
        resume = "student_state_dict" in loaded_dict
        if resume and load_cfg is None:
            load_cfg = {"student": True, "teacher": True, "optimizer": True, "iteration": True}
        result = super().load(loaded_dict, load_cfg, strict)
        self.teacher.requires_grad_(False)
        self.teacher.eval()
        if resume and load_cfg.get("student"):
            meta = loaded_dict["student_v0"]
            if meta["version"] != 1:
                raise ValueError("Unsupported student checkpoint version")
            saved_obs, saved_labels = loaded_dict["replay_obs"], loaded_dict["replay_labels"]
            size = meta["replay_size"]
            if meta["replay_capacity"] != self.capacity or size > self.capacity or saved_obs.shape != (size, STUDENT_OBS_DIM) or saved_labels.shape != (size, ACTION_DIM):
                raise ValueError("Replay checkpoint incompatible; keep num_envs, rollout size and replay capacity on resume")
            if not torch.isfinite(saved_obs).all() or not torch.isfinite(saved_labels).all():
                raise ValueError("Nonfinite replay checkpoint")
            self.replay_obs[:size] = saved_obs.to(self.device)
            self.replay_labels[:size] = saved_labels.to(self.device)
            self.replay_size = size
            self.replay_cursor = meta["replay_cursor"] % self.capacity
            self.num_updates = meta["num_updates"]
            self.provenance = meta["provenance"]
            self.teacher_mask = None  # New simulation episodes choose controllers afresh.
        return result
