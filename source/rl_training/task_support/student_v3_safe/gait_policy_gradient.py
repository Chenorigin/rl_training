"""Short-horizon on-policy gradient, combined with the existing distillation loss.

Fixed Gaussian exploration and clipped likelihood ratios; no learned critic.
Returns end at the rollout boundary (no bootstrap), so this is not full PPO.
"""
from copy import deepcopy
import math
import torch
from tensordict import TensorDict
from task_support.student_v1.distillation import StudentV1Distillation


def validate_pg(cfg):
    keys = {"weight", "leg_std", "wheel_std", "discount", "clip_ratio", "max_kl", "batch_size", "action_delay_steps"}
    if set(cfg) != keys or any(isinstance(v, bool) or not math.isfinite(v) for v in cfg.values()):
        raise ValueError("Invalid gait policy-gradient settings")
    if (any(cfg[k] <= 0 for k in ("leg_std", "wheel_std", "max_kl", "batch_size"))
            or cfg["weight"] < 0 or not 0 < cfg["discount"] <= 1 or not 0 < cfg["clip_ratio"] < 1
            or type(cfg["batch_size"]) is not int or type(cfg["action_delay_steps"]) is not int
            or cfg["action_delay_steps"] < 0):
        raise ValueError("Invalid gait policy-gradient bounds")


def delayed_returns(rewards, done, sampled, student, delay, discount):
    """Exclude unexecuted tail samples and resets inside the action queue delay."""
    total = torch.zeros_like(rewards)
    carry = torch.zeros_like(rewards[0])
    for t in range(len(rewards)-1, -1, -1):
        # Reward through non-eligible states stays: departure/failure still counts.
        carry = torch.where(student[t], rewards[t]+discount*carry*(~done[t]), 0.)
        total[t] = carry
    returns = torch.zeros_like(total)
    valid = sampled.clone()
    for t in range(len(rewards)):
        effect = t+delay
        if effect >= len(rewards):
            valid[t] = False
            continue
        survived = student[t:effect+1].all(0)
        if delay:
            survived &= ~done[t:effect].any(0)
        valid[t] &= survived
        returns[t] = total[effect]
    # Same-time, other-environment baseline; teacher rewards never participate.
    count = valid.sum(1, keepdim=True)
    others = (returns*valid).sum(1, keepdim=True)-returns*valid
    baseline = torch.where(count > 1, others/(count-1).clamp_min(1), 0.)
    advantages = torch.where(valid, returns-baseline, 0.)
    # A shared scale retains the advantage sign; do not centre each singleton.
    if valid.any():
        advantages /= advantages[valid].square().mean().sqrt().clamp_min(1.e-6)
    return returns, advantages, valid


def clipped_objective(mean, old_mean, actions, std, advantages, clip_ratio):
    delta = ((actions-old_mean)/std).square()-((actions-mean)/std).square()
    log_ratio = .5*delta.sum(-1)
    ratio = log_ratio.clamp(-20., 20.).exp()
    loss = -torch.minimum(ratio*advantages, ratio.clamp(1-clip_ratio, 1+clip_ratio)*advantages).mean()
    kl = .5*((mean-old_mean)/std).square().sum(-1).mean()
    return loss, kl, ((ratio-1).abs() > clip_ratio).float().mean()


class GaitStudentDistillation(StudentV1Distillation):
    def __init__(self, *args, gait_policy_gradient, **kwargs):
        validate_pg(gait_policy_gradient)
        super().__init__(*args, **kwargs)
        self.pg_cfg = deepcopy(gait_policy_gradient)
        self.pg_std = torch.tensor([self.pg_cfg["leg_std"]]*12+[self.pg_cfg["wheel_std"]]*4, device=self.device)
        shape = (self.storage.num_transitions_per_env, self.storage.num_envs)
        self.pg_mean = torch.zeros((*shape, 16), device=self.device)
        self.pg_action = torch.zeros_like(self.pg_mean)
        self.pg_sampled = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self.pg_student = torch.zeros_like(self.pg_sampled)
        self.pg_provider = None
        self.pg_ids = torch.empty((0, 2), dtype=torch.long, device=self.device)
        self.pg_stopped = False
        self.pg_teacher_absmax = 0.

    def act(self, obs):
        action = super().act(obs)
        step = self.storage.step
        if self.pg_cfg["weight"] and self.pg_provider is None:
            raise RuntimeError("Gait PG requires the actual Safe environment eligibility provider")
        student = ~self.teacher_mask
        eligible = self.pg_provider().to(self.device) if self.pg_cfg["weight"] else torch.zeros_like(student)
        if eligible.shape != student.shape or eligible.dtype != torch.bool:
            raise ValueError("Invalid gait exploration eligibility")
        sampled = student & eligible
        candidate = action+torch.randn_like(action)*self.pg_std if sampled.any() else action
        issued = torch.where(sampled[:, None], candidate, action)
        self.pg_mean[step].copy_(action)
        self.pg_action[step].copy_(issued)
        self.pg_sampled[step].copy_(sampled)
        self.pg_student[step].copy_(student)
        self.pg_teacher_absmax = max(self.pg_teacher_absmax, float(self.transition.privileged_actions.abs().max()))
        self.transition.actions = issued
        return issued

    def prepare_policy_gradient(self):
        steps = self.storage.step
        rewards = self.storage.rewards[:steps].squeeze(-1)
        done = self.storage.dones[:steps].squeeze(-1).bool()
        if not steps or not torch.isfinite(rewards).all():
            raise ValueError("Gait PG requires finite collected rewards")
        self.pg_returns, self.pg_advantages, valid = delayed_returns(
            rewards, done, self.pg_sampled[:steps] & self.stability_training_mask[:steps], self.pg_student[:steps],
            self.pg_cfg["action_delay_steps"], self.pg_cfg["discount"])
        self.pg_ids = valid.nonzero()
        self.pg_stopped = False

    def auxiliary_loss(self):
        zero = self.pg_mean.new_zeros(())
        stats = {"gait_pg_loss": zero, "gait_pg_kl": zero, "gait_pg_clip_fraction": zero}
        if not self.pg_cfg["weight"] or not self.pg_ids.numel() or self.pg_stopped:
            return zero, stats
        chosen = torch.randint(len(self.pg_ids), (min(self.pg_cfg["batch_size"], len(self.pg_ids)),), device=self.device)
        t, e = self.pg_ids[chosen].unbind(-1)
        obs = self.storage.observations["policy"][t, e]
        mean = self.student(TensorDict({"policy": obs}, batch_size=[len(obs)]))
        loss, kl, clipped = clipped_objective(mean, self.pg_mean[t, e], self.pg_action[t, e],
                                              self.pg_std, self.pg_advantages[t, e], self.pg_cfg["clip_ratio"])
        stats = {"gait_pg_loss": loss.detach(), "gait_pg_kl": kl.detach(), "gait_pg_clip_fraction": clipped.detach()}
        if float(kl.detach()) > self.pg_cfg["max_kl"]:
            self.pg_stopped = True
            return zero, stats
        return self.pg_cfg["weight"]*loss, stats

    def update(self):
        self.prepare_policy_gradient()
        count = len(self.pg_ids)
        result = super().update()
        result.update(gait_pg_samples=float(count), gait_pg_stopped=float(self.pg_stopped),
                      gait_pg_teacher_action_absmax=self.pg_teacher_absmax)
        # Monitor total movement from BC+retention+PG, not merely the PG term.
        with torch.inference_mode():
            kl = self.pg_mean.new_zeros(())
            for chunk in self.pg_ids.split(self.pg_cfg["batch_size"]):
                if not len(chunk):
                    continue
                t, e = chunk.unbind(-1)
                obs = self.storage.observations["policy"][t, e]
                mean = self.student(TensorDict({"policy": obs}, batch_size=[len(obs)]))
                kl += .5*((mean-self.pg_mean[t, e])/self.pg_std).square().sum()
            result["gait_pg_joint_final_kl"] = float(kl/max(count, 1))
        self.pg_ids = self.pg_ids[:0]
        self.pg_teacher_absmax = 0.
        return result

    def save(self):
        saved = super().save()
        saved["student_gait_pg"] = {"version": 1, "settings": deepcopy(self.pg_cfg)}
        return saved

    def load(self, saved, load_cfg=None, strict=True):
        if load_cfg is None and "student_state_dict" in saved:
            if saved.get("student_gait_pg") != {"version": 1, "settings": self.pg_cfg}:
                raise ValueError("Gait policy-gradient resume configuration changed")
        return super().load(saved, load_cfg, strict)
