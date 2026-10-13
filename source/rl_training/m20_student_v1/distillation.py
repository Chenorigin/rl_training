"""V1 dataset aggregation, phase/terrain/command balancing and group losses."""
import math

import torch
from tensordict import TensorDict
from rsl_rl.algorithms import Distillation


def stratified_indices(keys, count):
    """Equal expected mass per present stratum; no phantom/missing strata."""
    if keys.numel() == 0 or count < 1:
        raise ValueError("Cannot sample an empty training population")
    _, inverse, sizes = torch.unique(keys, return_inverse=True, return_counts=True)
    weight = 1./sizes[inverse].float()
    return torch.multinomial(weight, count, replacement=True)


def stratum_keys(tags):
    """Phase × local terrain × command direction × terrain difficulty."""
    return tags[:, 0]*24+tags[:, 1]*6+tags[:, 2]+tags[:, 3]*96


def grouped_loss(predicted, target, phase, leg_weight=1., wheel_weight=1.,
                 landing_weight=2., recovery_weight=2.):
    error = (predicted-target).square()
    legs, wheels = error[:, :12].mean(-1), error[:, 12:].mean(-1)
    weights = torch.where((phase == 1) | (phase == 2), landing_weight,
                          torch.where(phase == 3, recovery_weight, 1.))
    per_sample = (leg_weight*legs+wheel_weight*wheels)/(leg_weight+wheel_weight)
    return (weights*per_sample).sum()/weights.sum(), legs, wheels


def validate_resume_checkpoint(saved, obs_dim, capacity, hard_capacity, decay):
    meta = saved.get("student_v1", {})
    if meta.get("version") != "v1" or meta.get("obs_dim") != obs_dim or meta.get("capacity") != capacity or meta.get("hard_capacity") != hard_capacity:
        raise ValueError("Resume requires compatible v1 checkpoint; initialize v0 weights with --init-student")
    for key in ("num_updates", "student_episodes", "schedule_progress"):
        value = meta.get(key)
        if type(value) is not int or value < 0:
            raise ValueError(f"Invalid resume counter: {key}")
    if meta["schedule_progress"] > decay or not math.isfinite(meta["failure_ema"]) or not 0 <= meta["failure_ema"] <= 1:
        raise ValueError("Invalid resume teacher schedule/failure state")
    if not isinstance(meta.get("provenance"), dict):
        raise ValueError("Missing resume provenance")
    coverage = saved.get("coverage")
    if coverage is None or coverage.shape != (96,) or not torch.isfinite(coverage).all() or (coverage < 0).any():
        raise ValueError("Invalid resume coverage")
    for prefix, maximum in (("replay", capacity), ("hard", hard_capacity)):
        size, cursor = meta[prefix+"_size"], meta[prefix+"_cursor"]
        if type(size) is not int or type(cursor) is not int or not 0 <= size <= maximum or not 0 <= cursor < maximum:
            raise ValueError("Invalid replay size/cursor")
        for field, suffix, dtype in (("obs", (obs_dim,), torch.float32), ("labels", (16,), torch.float32),
                                    ("tags", (4,), torch.long), ("holdout", (), torch.bool)):
            value = saved[prefix+"_"+field]
            if value.shape != (size, *suffix) or value.dtype != dtype or not torch.isfinite(value).all():
                raise ValueError("Invalid replay checkpoint")
        tags = saved[prefix+"_tags"]
        if not ((tags >= 0).all() and (tags[:, 0] <= 3).all() and (tags[:, 1] <= 3).all() and (tags[:, 2] <= 5).all()):
            raise ValueError("Invalid replay tags")
        if prefix == "hard" and saved["hard_holdout"].any():
            raise ValueError("Held-out states cannot enter critical training replay")
    return meta


class StudentV1Distillation(Distillation):
    def __init__(self, *args, teacher_ratio=.5, teacher_ratio_end=0.,
                 teacher_warmup_updates=1000, teacher_decay_updates=7000,
                 student_failure_gate=.2, gate_min_episodes=32,
                 batch_size=1024, replay_rollouts=8, hard_replay_rollouts=2, save_replay=True,
                 leg_loss_weight=1., wheel_loss_weight=1., landing_loss_weight=2., recovery_loss_weight=2.,
                 stratified_fraction=.5, hard_fraction=.25,
                 noise_level_start=.5, noise_level_end=1., noise_ramp_updates=4000, **kwargs):
        super().__init__(*args, **kwargs)
        if self.is_multi_gpu or self.student.is_recurrent or self.teacher.is_recurrent:
            raise ValueError("V1 uses single-process feedforward models with optional explicit history")
        if self.teacher.obs_dim != 244 or self.student.obs_dim < 618 or (self.student.obs_dim-618) % 57:
            raise ValueError("Expected v1 actor dimensions and 244D teacher")
        if self.student.obs_normalization or self.teacher.obs_normalization:
            raise ValueError("V1 warm start requires normalization disabled")
        if self.storage.num_envs < 1:
            raise ValueError("Invalid rollout storage")
        if not isinstance(save_replay, bool):
            raise ValueError("save_replay must be a boolean")
        # True by default here preserves old run configurations without this key.
        self.save_replay = save_replay
        for name, value in (("teacher_ratio", teacher_ratio), ("teacher_ratio_end", teacher_ratio_end),
                            ("student_failure_gate", student_failure_gate), ("stratified_fraction", stratified_fraction),
                            ("hard_fraction", hard_fraction), ("noise_level_start", noise_level_start), ("noise_level_end", noise_level_end)):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Invalid {name}")
        if teacher_ratio_end > teacher_ratio or stratified_fraction+hard_fraction > 1:
            raise ValueError("Invalid teacher schedule or sampling fractions")
        for value in (batch_size, replay_rollouts, hard_replay_rollouts, teacher_decay_updates,
                      gate_min_episodes, noise_ramp_updates, self.num_learning_epochs):
            if not isinstance(value, int) or value < 1:
                raise ValueError("Schedule, replay and batch sizes must be positive integers")
        if not isinstance(teacher_warmup_updates, int) or teacher_warmup_updates < 0:
            raise ValueError("Invalid teacher warmup")
        for value in (leg_loss_weight, wheel_loss_weight, landing_loss_weight, recovery_loss_weight):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Loss weights must be finite and positive")
        self.teacher.requires_grad_(False).eval()
        if self.student.distribution is not None:
            self.student.distribution.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.student.mlp.parameters(), lr=self.learning_rate)
        self.obs_dim, self.batch_size = self.student.obs_dim, batch_size
        self.ratio_start, self.ratio_end = teacher_ratio, teacher_ratio_end
        self.warmup, self.decay = teacher_warmup_updates, teacher_decay_updates
        self.failure_gate, self.gate_min_episodes = student_failure_gate, gate_min_episodes
        self.schedule_progress = 0
        self.student_episodes = 0
        self.failure_ema = 0.
        self.teacher_ratio = teacher_ratio
        self.noise_start, self.noise_end, self.noise_ramp = noise_level_start, noise_level_end, noise_ramp_updates
        self.stratified_fraction, self.hard_fraction = stratified_fraction, hard_fraction
        self.loss_weights = (leg_loss_weight, wheel_loss_weight, landing_loss_weight, recovery_loss_weight)
        self.rollout_capacity = self.storage.num_envs*self.storage.num_transitions_per_env
        self.capacity, self.hard_capacity = self.rollout_capacity*replay_rollouts, self.rollout_capacity*hard_replay_rollouts
        for prefix, capacity in (("replay", self.capacity), ("hard", self.hard_capacity)):
            setattr(self, prefix+"_obs", torch.zeros(capacity, self.obs_dim, device=self.device))
            setattr(self, prefix+"_labels", torch.zeros(capacity, 16, device=self.device))
            setattr(self, prefix+"_tags", torch.zeros(capacity, 4, device=self.device, dtype=torch.long))
            setattr(self, prefix+"_holdout", torch.zeros(capacity, device=self.device, dtype=torch.bool))
            setattr(self, prefix+"_size", 0)
            setattr(self, prefix+"_cursor", 0)
        self.rollout_samples = self.executed_teacher = self.executed_total = 0
        self.teacher_mask = None
        self.coverage = torch.zeros(96, device=self.device)
        self.provenance = {}

    @property
    def noise_level(self):
        return self.noise_start+(self.noise_end-self.noise_start)*min(1., self.num_updates/self.noise_ramp)

    def act(self, obs):
        if not self.teacher_loaded:
            raise RuntimeError("A frozen teacher must be loaded before collection")
        if not all(torch.isfinite(obs[key]).all() for key in ("policy", "teacher")):
            raise ValueError("Nonfinite observations")
        teacher_action, student_action = self.teacher(obs).detach(), self.student(obs).detach()
        if not torch.isfinite(teacher_action).all() or not torch.isfinite(student_action).all():
            raise ValueError("Nonfinite actions")
        n = student_action.shape[0]
        if self.teacher_mask is None:
            self.teacher_mask = torch.rand(n, device=self.device) < self.teacher_ratio
        if self.teacher_ratio == 0:
            self.teacher_mask.zero_()
        action = torch.where(self.teacher_mask[:, None], teacher_action, student_action)
        if self.student.distribution is not None:
            self.student.distribution.update(student_action)
        self.transition.actions = action
        self.transition.privileged_actions = teacher_action
        self.transition.observations = obs.clone()
        self.executed_teacher += int(self.teacher_mask.sum())
        self.executed_total += n
        return action

    def _append(self, prefix, observations, labels, tags, holdout):
        n = observations.shape[0]
        capacity = self.capacity if prefix == "replay" else self.hard_capacity
        if n > capacity:
            observations, labels, tags, holdout = observations[-capacity:], labels[-capacity:], tags[-capacity:], holdout[-capacity:]
            n = capacity
        cursor = getattr(self, prefix+"_cursor")
        ids = (torch.arange(n, device=self.device)+cursor) % capacity
        for field, data in (("obs", observations), ("labels", labels), ("tags", tags), ("holdout", holdout)):
            getattr(self, prefix+"_"+field)[ids] = data
        setattr(self, prefix+"_size", min(capacity, getattr(self, prefix+"_size")+n))
        setattr(self, prefix+"_cursor", (cursor+n) % capacity)

    def process_env_step(self, obs, rewards, dones, extras):
        # Tags describe the pre-step state, plus touchdowns caused during this step.
        diagnostic = extras.get("student_v1")
        if diagnostic is None:
            raise RuntimeError("V1 environment must provide privileged sampling tags outside actor observations")
        n = self.transition.observations["policy"].shape[0]
        tags = torch.stack([diagnostic[name].to(self.device).long() for name in ("phase", "terrain", "direction", "terrain_level")], -1)
        if tags.shape != (n, 4) or not ((tags[:, 0] >= 0) & (tags[:, 0] <= 3) & (tags[:, 1] >= 0) & (tags[:, 1] <= 3) & (tags[:, 2] >= 0) & (tags[:, 2] <= 5) & (tags[:, 3] >= 0)).all():
            raise ValueError("Invalid v1 sampling tags")
        before, labels = self.transition.observations["policy"], self.transition.privileged_actions
        holdout = torch.rand(n, device=self.device) < .05
        # Keep at least one training example in a tiny offline fixture/rollout.
        if holdout.all():
            holdout[0] = False
        self._append("replay", before, labels, tags, holdout)
        critical = (tags[:, 0] != 0) & ~holdout
        self._append("hard", before[critical], labels[critical], tags[critical], holdout[critical])
        coarse_keys = tags[:, 0]*24+tags[:, 1]*6+tags[:, 2]
        self.coverage += torch.bincount(coarse_keys, minlength=96)
        self.rollout_samples += n
        done = dones.flatten().bool().to(self.device)
        finished_student = done & ~self.teacher_mask
        count = int(finished_student.sum())
        if count:
            failures = diagnostic["terminated"].to(self.device).bool()[finished_student].float().mean().item()
            blend = 1-math.exp(-count/32)
            self.failure_ema = (1-blend)*self.failure_ema+blend*failures
            self.student_episodes += count
        super().process_env_step(obs, rewards, dones, extras)
        self.teacher_mask[done] = torch.rand(int(done.sum()), device=self.device) < self.teacher_ratio

    def _batch(self, n):
        population = (~self.replay_holdout[:self.replay_size]).nonzero().flatten()
        if not population.numel():
            raise RuntimeError("No non-holdout training samples")
        hard_n = int(n*self.hard_fraction) if self.hard_size else 0
        balanced_n = int(n*self.stratified_fraction)
        recent_n = n-hard_n-balanced_n
        tags = self.replay_tags[population]
        keys = stratum_keys(tags)
        balanced = population[stratified_indices(keys, balanced_n)] if balanced_n else population[:0]
        recent_count = min(self.rollout_samples, self.replay_size)
        recent_ids = (torch.arange(recent_count, device=self.device)+self.replay_cursor-recent_count) % self.capacity
        recent_ids = recent_ids[~self.replay_holdout[recent_ids]]
        if not recent_ids.numel():
            recent_ids = population
        recent = recent_ids[torch.randint(recent_ids.numel(), (recent_n,), device=self.device)]
        ids = torch.cat((balanced, recent))
        observations, labels, sampled_tags = self.replay_obs[ids], self.replay_labels[ids], self.replay_tags[ids]
        if hard_n:
            ht = self.hard_tags[:self.hard_size]
            hk = stratum_keys(ht)
            hard_ids = stratified_indices(hk, hard_n)
            observations = torch.cat((observations, self.hard_obs[hard_ids]))
            labels = torch.cat((labels, self.hard_labels[hard_ids]))
            sampled_tags = torch.cat((sampled_tags, ht[hard_ids]))
        return observations, labels, sampled_tags

    def update(self):
        if not self.rollout_samples:
            raise RuntimeError("No collected v1 samples")
        total = torch.zeros(3, device=self.device)
        updates = 0
        for _ in range(self.num_learning_epochs):
            for _ in range(math.ceil(self.rollout_samples/self.batch_size)):
                observations, labels, tags = self._batch(min(self.batch_size, self.rollout_samples))
                predicted = self.student(TensorDict({"policy": observations}, batch_size=[len(observations)]))
                loss, legs, wheels = grouped_loss(predicted, labels, tags[:, 0], *self.loss_weights)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite distillation loss")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.student.mlp.parameters(), self.max_grad_norm or 1., error_if_nonfinite=True)
                self.optimizer.step()
                total += torch.stack((loss.detach(), legs.mean().detach(), wheels.mean().detach()))
                updates += 1
        self.num_updates += 1
        gate_open = self._advance_schedule()
        self.storage.clear()
        values = total/updates
        stats = {"behavior": float(values[0]), "leg_mse": float(values[1]), "wheel_mse": float(values[2]),
                 "teacher_ratio_target": self.teacher_ratio, "teacher_executed_fraction": self.executed_teacher/self.executed_total,
                 "teacher_decay_gate_open": float(gate_open), "student_failure_ema": self.failure_ema,
                 "student_completed_episodes": float(self.student_episodes), "noise_level": self.noise_level,
                 "replay_samples": float(self.replay_size), "critical_replay_samples": float(self.hard_size),
                 "present_strata": float(torch.unique(stratum_keys(self.replay_tags[:self.replay_size])).numel())}
        with torch.inference_mode():
            ids = self.replay_holdout[:self.replay_size].nonzero().flatten()[:self.batch_size]
            if ids.numel():
                predicted = self.student(TensorDict({"policy": self.replay_obs[ids]}, batch_size=[len(ids)]))
                residual = predicted-self.replay_labels[ids]
                error = residual.square().mean(-1)
                stats["heldout_action_mse"] = float(error.mean())
                leg_scale = torch.tensor([.125, .25, .25]*4, device=self.device)
                stats["heldout_leg_position_rmse_rad"] = float((residual[:, :12]*leg_scale).square().mean().sqrt())
                stats["heldout_wheel_velocity_rmse_rad_s"] = float(residual[:, 12:].square().mean().sqrt()*5.)
                for phase in range(4):
                    select = self.replay_tags[ids, 0] == phase
                    if select.any():
                        stats[f"heldout_phase_{phase}_mse"] = float(error[select].mean())
        current_tags = self.replay_tags[:self.replay_size]
        for column, prefix, categories in ((0, "phase", 4), (1, "terrain", 4), (2, "direction", 6)):
            for category in range(categories):
                stats[f"coverage_{prefix}_{category}"] = float((current_tags[:, column] == category).float().mean())
        for level in torch.unique(current_tags[:, 3]).tolist():
            stats[f"coverage_terrain_level_{level}"] = float((current_tags[:, 3] == level).float().mean())
        self.rollout_samples = self.executed_teacher = self.executed_total = 0
        return stats

    def _advance_schedule(self):
        """Control-state update exposed separately for no-optimization unit checks."""
        gate_open = self.student_episodes >= self.gate_min_episodes and self.failure_ema <= self.failure_gate
        if self.num_updates > self.warmup and gate_open:
            self.schedule_progress = min(self.decay, self.schedule_progress+1)
        self.teacher_ratio = self.ratio_start+(self.ratio_end-self.ratio_start)*self.schedule_progress/self.decay
        return gate_open

    def save(self):
        result = super().save()
        meta = {"version": "v1", "obs_dim": self.obs_dim, "num_updates": self.num_updates,
                "capacity": self.capacity, "hard_capacity": self.hard_capacity,
                "schedule_progress": self.schedule_progress, "student_episodes": self.student_episodes,
                "failure_ema": self.failure_ema, "provenance": self.provenance,
                "replay_saved": self.save_replay}
        for prefix in ("replay", "hard"):
            size = getattr(self, prefix+"_size") if self.save_replay else 0
            meta[prefix+"_size"] = size
            meta[prefix+"_cursor"] = getattr(self, prefix+"_cursor") if self.save_replay else 0
            for field in ("obs", "labels", "tags", "holdout"):
                # Clone even empty slices so serialization cannot retain the
                # backing storage of a multi-GiB replay tensor.
                result[prefix+"_"+field] = getattr(self, prefix+"_"+field)[:size].clone()
        result.update(student_v1=meta, actor_state_dict=result["student_state_dict"], coverage=self.coverage.clone())
        return result

    def load(self, loaded_dict, load_cfg=None, strict=True):
        resume = load_cfg is None and "student_state_dict" in loaded_dict
        if resume:
            meta = validate_resume_checkpoint(loaded_dict, self.obs_dim, self.capacity, self.hard_capacity, self.decay)
            load_cfg = {"student": True, "teacher": True, "optimizer": True, "iteration": True}
        result = super().load(loaded_dict, load_cfg, strict)
        self.teacher.requires_grad_(False).eval()
        if resume:
            for prefix in ("replay", "hard"):
                setattr(self, prefix+"_size", meta[prefix+"_size"])
                setattr(self, prefix+"_cursor", meta[prefix+"_cursor"])
                for field in ("obs", "labels", "tags", "holdout"):
                    getattr(self, prefix+"_"+field)[:meta[prefix+"_size"]] = loaded_dict[prefix+"_"+field].to(self.device)
            self.num_updates = meta["num_updates"]
            self.schedule_progress = meta["schedule_progress"]
            self.student_episodes = meta["student_episodes"]
            self.failure_ema = meta["failure_ema"]
            self.provenance = meta["provenance"]
            self.coverage.copy_(loaded_dict["coverage"].to(self.device))
            self.teacher_ratio = self.ratio_start+(self.ratio_end-self.ratio_start)*self.schedule_progress/self.decay
            self.teacher_mask = None
        return result
