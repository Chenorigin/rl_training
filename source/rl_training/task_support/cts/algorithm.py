"""CTS PPO on each cohort's own actions, followed by online latent supervision.

Teacher and student environments have fixed identities. PPO uses each
trajectory's actual behavior log probability; each cohort contributes half of
the policy/value objective regardless of its population. Student encoder
weights remain detached during every PPO epoch, while the shared decoder and
critic learn from both cohorts. Teacher targets are recomputed after PPO.
"""
from copy import deepcopy
import math

import torch
from torch import nn
from torch.distributions import Normal

from .model import OBS_DIMS, actor_settings, teacher_observations
from .configuration import TEACHER_TRAINING_KEYS, train_teacher_base


CONFIG_KEYS = {"teacher_fraction", "cohort_seed", "learning_rate", "student_learning_rate", "gamma", "lam", "clip_param",
               "num_learning_epochs", "num_mini_batches", "value_loss_coef", "max_grad_norm",
               "student_learning_epochs", "student_batch_size", "latent_loss_weight", "decoder_hidden_dims",
               "decoder_bound", "anchor_coefficient", "entropy_coef", "desired_kl"}


def validate_config(cfg):
    unknown = set(cfg)-CONFIG_KEYS-TEACHER_TRAINING_KEYS
    if unknown:
        raise ValueError(f"Unknown CTS algorithm configuration: {sorted(unknown)}")
    missing = CONFIG_KEYS-set(cfg)
    if missing:
        raise ValueError(f"Missing CTS algorithm configuration: {sorted(missing)}")
    train_teacher_base(cfg)
    if not 0 < cfg["teacher_fraction"] < 1:
        raise ValueError("CTS requires nonempty teacher and student cohorts")
    if type(cfg["cohort_seed"]) is not int or cfg["cohort_seed"] < 0:
        raise ValueError("CTS cohort_seed must be a nonnegative integer")
    for key in ("num_learning_epochs", "num_mini_batches", "student_learning_epochs", "student_batch_size"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("learning_rate", "student_learning_rate", "max_grad_norm", "clip_param", "latent_loss_weight", "desired_kl"):
        if not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    for key in ("anchor_coefficient", "value_loss_coef", "entropy_coef"):
        if not math.isfinite(cfg[key]) or cfg[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if not 0 < cfg["gamma"] <= 1 or not 0 <= cfg["lam"] <= 1:
        raise ValueError("Invalid CTS return discount")
    if not cfg["decoder_hidden_dims"] or any(type(x) is not int or x < 1 for x in cfg["decoder_hidden_dims"]):
        raise ValueError("Invalid CTS decoder hidden dimensions")
    if len(cfg["decoder_bound"]) != 16 or any(not math.isfinite(x) or x <= 0 for x in cfg["decoder_bound"]):
        raise ValueError("Invalid CTS decoder action bounds")


def balanced_mean(values, teacher_mask):
    """Equal contribution from both policy populations, never a silent empty group."""
    if not teacher_mask.any() or teacher_mask.all():
        raise ValueError("Balanced CTS batches require both cohorts")
    return .5*(values[teacher_mask].mean()+values[~teacher_mask].mean())


def clipped_policy_objective(new_log_prob, old_log_prob, advantages, clip_param):
    ratio = (new_log_prob-old_log_prob).exp()
    return torch.maximum(-advantages*ratio, -advantages*ratio.clamp(1-clip_param, 1+clip_param))


def domain_anchor_loss(per_sample_error, known_domain, new_domain_scale):
    """Lower an unreliable old teacher's pull, without renormalizing it away."""
    weight = torch.where(known_domain, torch.ones_like(per_sample_error),
                         torch.full_like(per_sample_error, new_domain_scale))
    return (per_sample_error * weight).mean()


def ordinary_clone(tensor):
    # The native runner collects under inference_mode. Replay tensors used by
    # autograd must be ordinary tensors, even when copied during collection.
    with torch.inference_mode(False):
        return tensor.detach().clone()


def validate_adam_state(saved, optimizer, learning_rate=None):
    """Validate an entire Adam restore before touching the live optimizer."""
    if not isinstance(saved, dict) or set(saved) != {"state", "param_groups"}:
        raise ValueError("Invalid CTS Adam checkpoint structure")
    expected = optimizer.state_dict()["param_groups"]
    rates = ([learning_rate]*len(expected) if isinstance(learning_rate, (int, float))
             else learning_rate)
    if rates is not None and len(rates) != len(expected):
        raise ValueError('CTS Adam learning-rate groups changed')
    groups = saved["param_groups"]
    if not isinstance(groups, list) or len(groups) != len(expected) or not isinstance(saved["state"], dict):
        raise ValueError("Invalid CTS Adam parameter groups")
    parameter_map = {}
    for index, (group, reference, live) in enumerate(zip(groups, expected, optimizer.param_groups)):
        if rates is not None:
            reference = {**reference, "lr": rates[index]}
        if not isinstance(group, dict) or set(group) != set(reference):
            raise ValueError("CTS Adam group keys changed")
        for name, value in reference.items():
            if group[name] != value:
                raise ValueError(f"CTS Adam fixed hyperparameter changed: {name}")
        if any(type(identifier) is not int for identifier in group["params"]):
            raise ValueError("Invalid CTS Adam parameter identifiers")
        parameter_map.update(zip(group["params"], live["params"]))
    if len(parameter_map) != sum(len(group["params"]) for group in groups):
        raise ValueError("Duplicate CTS Adam parameters")
    for identifier, state in saved["state"].items():
        if type(identifier) is not int or identifier not in parameter_map:
            raise ValueError("Unknown CTS Adam state parameter")
        if not isinstance(state, dict) or set(state) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("Invalid CTS Adam moment state")
        step = state["step"]
        if (not isinstance(step, torch.Tensor) or step.layout != torch.strided or step.shape != torch.Size([])
                or not step.is_floating_point() or not torch.isfinite(step) or step < 0 or step != step.floor()):
            raise ValueError("Invalid CTS Adam step counter")
        parameter = parameter_map[identifier]
        for name in ("exp_avg", "exp_avg_sq"):
            value = state[name]
            if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
                    or value.shape != parameter.shape or value.dtype != parameter.dtype
                    or not torch.isfinite(value).all()):
                raise ValueError(f"Invalid CTS Adam {name} tensor")
        if (state["exp_avg_sq"] < 0).any():
            raise ValueError("Negative CTS Adam squared moment")


class CTSAlgorithm:
    def __init__(self, models, cfg, num_envs, num_steps_per_env, device="cpu", *, auxiliary_cfg=None, perception=None, straight_cfg=None, all_terrain_cfg=None, joint_limit_cfg=None):
        validate_config(cfg)
        if num_envs < 2 or num_steps_per_env < 1:
            raise ValueError("CTS requires at least two environments and one rollout step")
        self.models, self.cfg = models, deepcopy(cfg)
        from .joint_limits import TorchJointLimits
        self.joint_limits = TorchJointLimits(joint_limit_cfg, device) if joint_limit_cfg is not None else None
        self.straight_cfg = deepcopy(straight_cfg)
        self.all_terrain_cfg = deepcopy(all_terrain_cfg)
        self.actor_cfg = actor_settings(all_terrain_cfg)
        from .all_terrain_config import COURSES
        self.num_courses = len(COURSES)
        has_idle = hasattr(models.teacher, 'idle_mlp') and models.decoder.idle_mlp is not None
        if (self.actor_cfg is not None) != has_idle:
            raise ValueError('CTS all-terrain architecture and algorithm contract differ')
        if models.clean_teacher_decoder != (self.actor_cfg is not None):
            raise ValueError('CTS all-terrain clean teacher decoder contract differs')
        if self.actor_cfg is not None and (models.teacher.idle_leg_scale != self.actor_cfg['idle_leg_scale']
                                          or models.decoder.idle_leg_scale != self.actor_cfg['idle_leg_scale']):
            raise ValueError('CTS all-terrain idle bounds differ from the algorithm contract')
        self.num_envs, self.num_steps_per_env, self.device = num_envs, num_steps_per_env, device
        # Practice and terrain assignment use environment index patterns. A
        # separately seeded permutation avoids aliasing a 3:1 split with the
        # idle-practice every-fourth-environment selection.
        from .practice import cohort_mask
        self.teacher_mask = cohort_mask(num_envs, cfg['teacher_fraction'], cfg['cohort_seed'], device)
        if not self.teacher_mask.any() or self.teacher_mask.all():
            raise ValueError("Environment count cannot represent both CTS cohorts")
        self.ppo_parameters = [p for module in (models.teacher, models.decoder, models.critic)
                               for p in module.parameters() if p.requires_grad]
        self.student_parameters = list(models.student.mlp.parameters())
        if train_teacher_base(cfg):
            teacher_parameters = [p for p in models.teacher.parameters() if p.requires_grad]
            teacher_ids = {id(p) for p in teacher_parameters}
            groups = [dict(params=teacher_parameters, lr=cfg['teacher_learning_rate'], name='teacher'),
                      dict(params=[p for p in self.ppo_parameters if id(p) not in teacher_ids],
                           lr=cfg['learning_rate'], name='decoder_critic')]
            self.optimizer = torch.optim.Adam(groups, lr=cfg['learning_rate'])
        else:
            self.optimizer = torch.optim.Adam(self.ppo_parameters, lr=cfg["learning_rate"])
        self.student_optimizer = torch.optim.Adam(self.student_parameters, lr=cfg["student_learning_rate"])
        self.num_updates = 0
        self.rollout = []
        self.pending = None
        self.returns = self.advantages = None
        self.provenance = {}
        from .auxiliary import AuxiliaryObjectives
        self.auxiliary = AuxiliaryObjectives(models, auxiliary_cfg, perception, device)

    def _checked_obs(self, obs):
        result = {}
        for key, dim in OBS_DIMS.items():
            value = obs[key]
            if value.shape != (self.num_envs, dim) or not torch.isfinite(value).all():
                raise ValueError(f"Invalid CTS observation {key}")
            result[key] = value
        return result

    @torch.no_grad()
    def act(self, obs):
        if self.pending is not None or len(self.rollout) >= self.num_steps_per_env:
            raise RuntimeError("CTS rollout sequencing error")
        obs = self._checked_obs(obs)
        mean, latent = self.models.mean(obs, self.teacher_mask)
        distribution = Normal(mean, self.models.log_std.exp())
        actions = distribution.sample()
        values = self.models.value(obs, latent)
        self.pending = {"obs": {key: ordinary_clone(value) for key, value in obs.items()},
                        "actions": ordinary_clone(actions), "mean": ordinary_clone(mean),
                        "values": ordinary_clone(values),
                        "log_prob": ordinary_clone(distribution.log_prob(actions).sum(-1))}
        return actions

    @torch.no_grad()
    def process_env_step(self, obs, rewards, dones, extras):
        if self.pending is None:
            raise RuntimeError("CTS process_env_step requires a sampled action")
        rewards, dones = rewards.reshape(-1), dones.reshape(-1)
        if (rewards.shape != (self.num_envs,) or dones.shape != rewards.shape or not torch.isfinite(rewards).all()
                or not ((dones == 0) | (dones == 1)).all()):
            raise ValueError("Invalid CTS reward/done shape")
        reward = ordinary_clone(rewards)
        timeouts = extras.get("time_outs")
        if timeouts is not None:
            timeouts = torch.as_tensor(timeouts, device=self.device).reshape(-1).bool()
            if timeouts.shape != dones.shape or (timeouts & ~dones.bool()).any():
                raise ValueError("CTS timeout must be a completed transition")
            # Matches installed RSL 5.0.1 / Isaac timeout semantics: post-step
            # observations are reset states, so bootstrap with pre-step V.
            reward += self.cfg["gamma"]*self.pending["values"]*timeouts
        regular = self.auxiliary.regular_phase(extras, self.num_envs)
        straight = extras.get('cts_straight') if self.straight_cfg is not None else torch.zeros_like(dones,dtype=torch.bool)
        if not isinstance(straight,torch.Tensor) or straight.shape != dones.shape or straight.dtype != torch.bool:
            raise ValueError('CTS straight supervision requires pre-step course tags')
        anchor = extras.get('cts_anchor') if self.all_terrain_cfg is not None else torch.ones_like(dones, dtype=torch.bool)
        course = extras.get('cts_course') if self.all_terrain_cfg is not None else torch.full_like(dones, -1, dtype=torch.long)
        if not isinstance(anchor, torch.Tensor) or anchor.shape != dones.shape or anchor.dtype != torch.bool:
            raise ValueError('CTS domain anchor requires pre-step cts_anchor boolean tags')
        if (not isinstance(course, torch.Tensor) or course.shape != dones.shape or course.dtype != torch.long
                or (self.all_terrain_cfg is not None and ((course < 0) | (course >= self.num_courses)).any())):
            raise ValueError('CTS all-terrain requires pre-step cts_course integer tags')
        self.pending.update(rewards=reward, dones=ordinary_clone(dones.bool()), regular_phase=ordinary_clone(regular),
                            straight=ordinary_clone(straight.to(self.device) & regular),
                            anchor=ordinary_clone(anchor.to(self.device)), course=ordinary_clone(course.to(self.device)))
        self.rollout.append(self.pending)
        self.pending = None

    @torch.no_grad()
    def compute_returns(self, obs):
        if self.pending is not None or len(self.rollout) != self.num_steps_per_env:
            raise RuntimeError("CTS returns require a complete rollout")
        obs = self._checked_obs(obs)
        latent = self.models.latent(obs, self.teacher_mask)
        next_value = self.models.value(obs, latent)
        advantage = torch.zeros_like(next_value)
        returns = []
        for step in reversed(self.rollout):
            alive = (~step["dones"]).float()
            delta = step["rewards"] + self.cfg["gamma"]*next_value*alive-step["values"]
            advantage = delta+self.cfg["gamma"]*self.cfg["lam"]*alive*advantage
            returns.append(advantage+step["values"])
            next_value = step["values"]
        self.returns = ordinary_clone(torch.stack(list(reversed(returns))))
        advantages = self.returns-torch.stack([s["values"] for s in self.rollout])
        # Normalize separately: different cohorts need not have equal reward scales.
        for mask in (self.teacher_mask, ~self.teacher_mask):
            values = advantages[:, mask]
            advantages[:, mask] = (values-values.mean())/(values.std(unbiased=False)+1.e-8)
        self.advantages = ordinary_clone(advantages)

    def _data(self):
        if self.returns is None:
            raise RuntimeError("CTS update requires computed returns")
        return {"obs": {key: torch.cat([step["obs"][key] for step in self.rollout]) for key in OBS_DIMS},
                **{key: torch.cat([step[key] for step in self.rollout]) for key in ("actions", "mean", "values", "log_prob")},
                "returns": self.returns.flatten(), "advantages": self.advantages.flatten(),
                "teacher": self.teacher_mask.repeat(self.num_steps_per_env),
                **{key: torch.cat([step[key] for step in self.rollout]) for key in ('straight', 'anchor', 'course')}}

    def ppo_update(self):
        data, cfg = self._data(), self.cfg
        teacher_ids = data["teacher"].nonzero().flatten()
        student_ids = (~data["teacher"]).nonzero().flatten()
        batches = min(cfg["num_mini_batches"], len(teacher_ids), len(student_ids))
        metrics = {key: 0. for key in ("policy_loss", "value_loss", "anchor_loss", "kl")}
        updates = 0
        self.student_optimizer.zero_grad(set_to_none=True)
        for _ in range(cfg["num_learning_epochs"]):
            teachers = teacher_ids[torch.randperm(len(teacher_ids), device=self.device)].tensor_split(batches)
            students = student_ids[torch.randperm(len(student_ids), device=self.device)].tensor_split(batches)
            for tids, sids in zip(teachers, students):
                ids = torch.cat((tids, sids))
                obs, mask = {key: value[ids] for key, value in data["obs"].items()}, data["teacher"][ids]
                mean, latent = self.models.mean(obs, mask, detach_student=True)
                distribution = Normal(mean, self.models.log_std.exp())
                log_prob = distribution.log_prob(data["actions"][ids]).sum(-1)
                with torch.no_grad():
                    kl = .5*((mean-data["mean"][ids])/self.models.log_std.exp()).square().sum(-1)
                    scalar_kl = float(balanced_mean(kl, mask))
                # Check the already accumulated policy change BEFORE another
                # optimizer step, rather than crossing the gate once more.
                if scalar_kl > 2*cfg["desired_kl"]:
                    break
                surrogate = clipped_policy_objective(log_prob, data["log_prob"][ids], data["advantages"][ids], cfg["clip_param"])
                policy_loss = balanced_mean(surrogate, mask)
                value = self.models.value(obs, latent)
                clipped = data["values"][ids]+(value-data["values"][ids]).clamp(-cfg["clip_param"], cfg["clip_param"])
                value_error = torch.maximum((value-data["returns"][ids]).square(), (clipped-data["returns"][ids]).square())
                value_loss = balanced_mean(value_error, mask)
                with torch.no_grad():
                    target = self.models.reference_teacher(teacher_observations(obs))
                teacher_proposal = self.models.teacher_latent(obs)
                teacher_mean = self.models.teacher_action(obs, teacher_proposal)
                anchor_error = ((teacher_mean-target)/self.models.log_std.exp().clamp_min(.05)).square().mean(-1)
                anchor = domain_anchor_loss(anchor_error, data['anchor'][ids],
                    self.actor_cfg['new_domain_anchor_scale'] if self.actor_cfg is not None else 1.)
                entropy = distribution.entropy().sum(-1).mean()
                retention_loss, retention_metrics = self.auxiliary.retention_loss(detach_student=True)
                loss = (policy_loss+cfg["value_loss_coef"]*value_loss+cfg["anchor_coefficient"]*anchor
                        -cfg["entropy_coef"]*entropy+retention_loss)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite CTS PPO loss")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.ppo_parameters, cfg["max_grad_norm"], error_if_nonfinite=True)
                self.optimizer.step()
                for key, value in (("policy_loss", policy_loss), ("value_loss", value_loss), ("anchor_loss", anchor), ("kl", scalar_kl)):
                    metrics[key] += float(value)
                for key, value in retention_metrics.items():
                    name = 'ppo_'+key
                    metrics[name] = metrics.get(name, 0.)+value
                updates += 1
            else:
                continue
            break
        if any(p.grad is not None for p in self.student_parameters):
            raise RuntimeError("CTS PPO leaked a gradient into the student encoder")
        rates = ({'teacher_learning_rate': self.optimizer.param_groups[0]['lr'],
                  'decoder_critic_learning_rate': self.optimizer.param_groups[1]['lr']}
                 if train_teacher_base(cfg) else {})
        return {**{key: value/max(updates, 1) for key, value in metrics.items()},
                "ppo_minibatches": updates, **rates}

    def distill_update(self):
        # Auxiliaries must differentiate THROUGH the complete shared decoder,
        # but the decoder's parameters belong exclusively to the PPO Adam.
        self.optimizer.zero_grad(set_to_none=True)
        frozen = [(p, p.requires_grad) for p in self.models.decoder.parameters()]
        try:
            for parameter, _ in frozen:
                parameter.requires_grad_(False)
            result = self._distill_update()
            if any(p.grad is not None for p in self.ppo_parameters):
                raise RuntimeError('CTS student objectives leaked gradients into PPO parameters')
            return result
        finally:
            for parameter, trainable in frozen:
                parameter.requires_grad_(trainable)

    def _distill_update(self):
        data, cfg = self._data(), self.cfg
        # The teacher has just changed; no stale replay labels are retained.
        # Supervise all collected states, including the student occupancy.
        with torch.no_grad():
            target = ordinary_clone(self.models.teacher_latent(data["obs"]))
            target_actions = ordinary_clone(self.models.teacher_action(data['obs'], target))
            from .straight import straight_teacher_actions
            straight_targets = (ordinary_clone(straight_teacher_actions(self.models,data['obs'],target))
                                if self.straight_cfg is not None else target_actions)
        selection = self.auxiliary.prepare(self.rollout)
        total, updates, metrics = 0., 0, {}
        n = len(target)
        for _ in range(cfg["student_learning_epochs"]):
            order = torch.randperm(n, device=self.device)
            for ids in order.split(cfg["student_batch_size"]):
                obs = {key: value[ids] for key, value in data["obs"].items()}
                predicted = self.models.student_latent(obs)
                latent_loss = cfg["latent_loss_weight"]*(predicted-target[ids]).square().mean()
                auxiliary_loss, auxiliary_metrics = self.auxiliary.loss(data, ids, target_actions, selection)
                from .straight import straight_action_loss
                straight_loss, straight_metrics = straight_action_loss(self.models,data,straight_targets,self.straight_cfg)
                limit_loss = (self.joint_limits.student_loss(
                    self.models.student_action(obs['policy'], obs['motion_control']))
                    if self.joint_limits is not None else latent_loss.new_zeros(()))
                loss = latent_loss + auxiliary_loss + straight_loss + limit_loss
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite CTS student supervision/auxiliary loss")
                self.student_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.student_parameters, cfg["max_grad_norm"], error_if_nonfinite=True)
                self.student_optimizer.step()
                total += float(latent_loss.detach())
                for name, value in {**auxiliary_metrics, **straight_metrics, 'joint_limit_loss': float(limit_loss.detach()), 'total_loss': float(loss.detach())}.items():
                    metrics[name] = metrics.get(name, 0.) + value
                updates += 1
        return {"latent_loss": total/max(updates, 1), "student_minibatches": updates,
                **{name: value/max(updates, 1) for name, value in metrics.items()},
                'temporal_pair_count': selection['pairs_current'].numel(),
                **self.auxiliary.evaluate()}

    def update(self):
        metrics = {**self.ppo_update(), **self.distill_update()}
        command = torch.cat([step['obs']['motion_control'] for step in self.rollout])
        # Expose the actual training distribution: an offline retention loss
        # must not be mistaken for online backward experience.
        for name, mask in (('backward', command[:, 0] < -1.e-4),
                           ('forward', command[:, 0] > 1.e-4),
                           ('idle', command.abs().amax(-1) <= 1.e-4)):
            metrics['online_'+name+'_fraction'] = float(mask.float().mean())
        if self.all_terrain_cfg is not None:
            data = self._data()
            for name, cohort in (('teacher', data['teacher']), ('student', ~data['teacher'])):
                metrics['online_'+name+'_known_domain_fraction'] = float(data['anchor'][cohort].float().mean())
                metrics['online_'+name+'_new_domain_fraction'] = float((~data['anchor'][cohort]).float().mean())
                for course in data['course'].unique().tolist():
                    metrics[f'online_{name}_course_{course}_fraction'] = float((data['course'][cohort] == course).float().mean())
        self.num_updates += 1
        self.clear_rollout()
        metrics["teacher_fraction"] = float(self.teacher_mask.float().mean())
        return metrics

    def clear_rollout(self):
        self.rollout.clear()
        self.pending = self.returns = self.advantages = None

    def save(self):
        if self.rollout or self.pending is not None:
            raise RuntimeError("CTS checkpoints must be saved at rollout boundaries")
        return {"cts_version": 3, "models_state_dict": self.models.state_dict(),
                "straight_contract": deepcopy(self.straight_cfg),
                "all_terrain_contract": deepcopy(self.all_terrain_cfg),
                "auxiliary_contract": deepcopy(self.auxiliary.contract),
                "optimizer_state_dict": self.optimizer.state_dict(), "student_optimizer_state_dict": self.student_optimizer.state_dict(),
                "num_updates": self.num_updates, "teacher_mask": self.teacher_mask.clone(), "algorithm_config": deepcopy(self.cfg),
                "num_envs": self.num_envs, "num_steps_per_env": self.num_steps_per_env,
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state(self.device) if str(self.device).startswith("cuda") else None,
                "provenance": deepcopy(self.provenance)}

    def load(self, saved):
        mask = saved.get("teacher_mask")
        if (saved.get("cts_version") != 3 or saved.get("algorithm_config") != self.cfg
                or saved.get('straight_contract') != self.straight_cfg
                or saved.get('all_terrain_contract') != self.all_terrain_cfg
                or saved.get('auxiliary_contract') != self.auxiliary.contract
                or saved.get("num_envs") != self.num_envs or saved.get("num_steps_per_env") != self.num_steps_per_env
                or not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.shape != self.teacher_mask.shape
                or not torch.equal(mask.to(self.device), self.teacher_mask)
                or type(saved.get("num_updates")) is not int or saved["num_updates"] < 0):
            raise ValueError("CTS resume architecture/optimization/cohort contract changed")
        def finite(value):
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("Nonfinite CTS resume state")
            if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                raise ValueError("Nonfinite CTS resume state")
            if isinstance(value, dict):
                for item in value.values(): finite(item)
            elif isinstance(value, (tuple, list)):
                for item in value: finite(item)
        finite(saved)
        expected = self.models.state_dict()
        state = saved.get("models_state_dict")
        if not isinstance(state, dict) or set(state) != set(expected):
            raise ValueError("CTS resume model keys changed")
        for key, value in state.items():
            if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
                    or value.shape != expected[key].shape or value.dtype != expected[key].dtype):
                raise ValueError(f"CTS resume model shape/dtype changed: {key}")
            if key == "decoder.bounds":
                bounds = value.new_tensor(self.cfg["decoder_bound"])
                if not torch.equal(value, bounds):
                    raise ValueError("CTS resume changed its configured decoder bounds")
            frozen_prefixes = ("reference_teacher.", "teacher.distribution.", "student.distribution.")
            if not train_teacher_base(self.cfg):
                frozen_prefixes += ("teacher.mlp.",)
            if key == "log_std" or key.startswith(frozen_prefixes):
                if not torch.equal(value.to(expected[key].device), expected[key]):
                    raise ValueError("CTS resume changed its frozen teacher baseline/reference/std")
        rates = ([self.cfg['teacher_learning_rate'], self.cfg['learning_rate']]
                 if train_teacher_base(self.cfg) else self.cfg['learning_rate'])
        validate_adam_state(saved.get("optimizer_state_dict"), self.optimizer, rates)
        validate_adam_state(saved.get("student_optimizer_state_dict"), self.student_optimizer, self.cfg["student_learning_rate"])
        if not isinstance(saved.get("provenance", {}), dict):
            raise ValueError("Invalid CTS provenance")
        for name, device in (("torch_rng_state", "cpu"), ("cuda_rng_state", self.device)):
            rng = saved.get(name)
            if name == "cuda_rng_state" and rng is None and not str(self.device).startswith("cuda"):
                continue
            if not isinstance(rng, torch.Tensor) or rng.dtype != torch.uint8 or rng.ndim != 1 or not rng.numel():
                raise ValueError(f"Invalid CTS {name}")
            if name == "torch_rng_state" or str(self.device).startswith("cuda"):
                try:
                    torch.Generator(device=device).set_state(rng.cpu())
                except RuntimeError as error:
                    raise ValueError(f"Invalid CTS {name}") from error
        # All rejection paths above precede mutation of models, optimizers,
        # counters, provenance, and global RNG state.
        self.models.load_state_dict(state, strict=True)
        self.optimizer.load_state_dict(saved["optimizer_state_dict"])
        self.student_optimizer.load_state_dict(saved["student_optimizer_state_dict"])
        self.num_updates = saved["num_updates"]
        self.provenance = deepcopy(saved.get("provenance", {}))
        torch.set_rng_state(saved["torch_rng_state"].cpu())
        if str(self.device).startswith("cuda"):
            torch.cuda.set_rng_state(saved["cuda_rng_state"].cpu(), self.device)
        self.clear_rollout()
