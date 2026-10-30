# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# SPDX-License-Identifier: BSD-3-Clause
"""RSL-RL 5.0.1 PPO with an explicit frozen-policy preservation loss.

_guarded_update is vendored from the installed official PPO.update, with one
additional loss term. This keeps anchoring inside the same clipped-gradient
optimizer step rather than performing an untracked post-PPO policy update.
"""
from __future__ import annotations
import copy
import math
import torch
from torch import nn
from rsl_rl.algorithms import PPO


class StairGuardedPPO(PPO):
    def __init__(self, *args, anchor_coefficient=0.1, critic_warmup_iterations=50,
                 reference_std_multiplier=1.05, anchor_stair_weight=0.02,
                 anchor_stop_weight=0.02, anchor_stop_command_epsilon=1.e-4,
                 stop_teacher_coefficient=0.0, stop_regularization_coefficient=0.0,
                 motion_regularization_coefficient=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        if self.actor.is_recurrent or self.symmetry or self.rnd or self.is_multi_gpu:
            raise ValueError("StairGuardedPPO currently supports plain single-device MLP PPO only")
        if anchor_coefficient < 0 or critic_warmup_iterations < 0 or reference_std_multiplier < 1:
            raise ValueError("Invalid preservation configuration")
        self.anchor_coefficient = anchor_coefficient
        self.critic_warmup_iterations = critic_warmup_iterations
        self.reference_std_multiplier = reference_std_multiplier
        if min(anchor_stair_weight, anchor_stop_weight, anchor_stop_command_epsilon) < 0:
            raise ValueError("Preservation sample weights/epsilon must be nonnegative")
        self.anchor_stair_weight = anchor_stair_weight
        self.anchor_stop_weight = anchor_stop_weight
        self.anchor_stop_command_epsilon = anchor_stop_command_epsilon
        self.reference_actor = None
        if stop_teacher_coefficient < 0:
            raise ValueError("Stop teacher coefficient must be nonnegative")
        self.stop_teacher_coefficient = stop_teacher_coefficient
        self.stop_regularization_coefficient = stop_regularization_coefficient
        if motion_regularization_coefficient < 0:
            raise ValueError("Motion residual regularization must be nonnegative")
        self.motion_regularization_coefficient = motion_regularization_coefficient
        self.stop_teacher_actor = None
        self._stop_teacher_metrics = []
        self.guard_updates = 0
        self._anchor_losses = []

    def set_reference(self, actor_state):
        """Freeze a task-capable actor; do not anchor against a damaged checkpoint."""
        self.reference_actor = copy.deepcopy(self.actor)
        self.reference_actor.load_state_dict(actor_state, strict=True)
        self.reference_actor.eval().requires_grad_(False)
        if self.actor.obs_dim != 244:
            raise ValueError("Preservation mask requires the stair teacher 57+187 layout")
        if not hasattr(self.actor.distribution, 'log_std_param'):
            raise ValueError("Preservation requires state-independent log Gaussian std")
        self.guard_updates = 0

    def act(self, obs):
        if hasattr(self.actor, "set_update_iteration"):
            self.actor.set_update_iteration(self.guard_updates)
        actions = super().act(obs)
        if hasattr(self.actor, "record_rollout_sample"):
            self.actor.record_rollout_sample(actions)
        return actions

    def set_stop_teacher(self, actor_state):
        """Preserve a validated quiet leg policy; never copy its wheel drift."""
        self.stop_teacher_actor = copy.deepcopy(self.actor)
        self.stop_teacher_actor.load_state_dict(actor_state, strict=True)
        self.stop_teacher_actor.eval().requires_grad_(False)

    def _stop_teacher_loss(self, observations):
        if self.stop_teacher_actor is None:
            raise RuntimeError("v1_stop_v4 requires its frozen model900 stop teacher")
        labels = observations["stop_support"]
        stop = observations["stop_metadata"].squeeze(-1) > .5
        mask = stop & (labels[:, 0] > .5) & (labels[:, 2] > .5)
        with torch.no_grad():
            clean = observations["stop_teacher"].clone()
            # Only physically planar support is supervised. Removing the
            # distant edge from the teacher scan preserves the quiet flat
            # behavior when a riser is visible but not yet being traversed.
            clean[:, 57:244] = labels[:, 3:4]
            target = self.stop_teacher_actor({"policy": clean})[:, :12]
            std = self.stop_teacher_actor.distribution.log_std_param[:12].exp().clamp_min(.05)
        predicted = self.actor(observations)[:, :12]
        error = ((predicted-target)/std).square().mean(-1)
        loss = (error*mask).sum()/mask.sum().clamp_min(1)
        self._stop_teacher_metrics.append((float(loss.detach()), float(mask.float().mean()),
                                           float((mask & (labels[:, 1] > .5)).float().mean())))
        return loss

    def _anchor_loss(self, observations):
        with torch.no_grad():
            target = self.reference_actor(observations)
            std = self.reference_actor.distribution.log_std_param.exp().detach().clamp_min(0.05)
            # Allow the old flat-ground stop policy to change. The actor keeps
            # its 244D input, so a small effective yaw correction is treated as
            # a zero request when translation is absent.
            height = observations['policy'][:, 57:244]
            uneven = (height.amax(-1)-height.amin(-1)) > 0.07
            command = observations['policy'][:, 6:9]
            stopping = (command[:, :2].abs().amax(-1) < self.anchor_stop_command_epsilon) & (command[:, 2].abs() <= 0.51)
            if 'stop_metadata' in observations:
                # v2 stores the raw request, distinguishing automatic heading
                # correction from a deliberate small manual turn.
                stopping = observations['stop_metadata'].squeeze(-1) > .5
            # Do not preserve the very stair posture / idle actions being
            # corrected by this refinement. Normal flat motion keeps a guard.
            weight = torch.where(uneven, self.anchor_stair_weight, 1.0)
            weight = torch.where(stopping, self.anchor_stop_weight, weight)
        predicted = self.actor(observations)
        error = ((predicted-target)/std).square().mean(-1)
        return (weight*error).mean()

    def update(self):
        self._anchor_losses = []
        self._stop_teacher_metrics = []
        warmup = (self.reference_actor is not None or getattr(self.actor, 'frozen_motion', False)) and self.guard_updates < self.critic_warmup_iterations
        parameters = list(self.actor.parameters())
        flags = [p.requires_grad for p in parameters]
        if warmup:
            for p in parameters:
                p.requires_grad_(False)
        try:
            result = self._guarded_update()
        finally:
            for p, flag in zip(parameters, flags):
                p.requires_grad_(flag)
        if self.reference_actor is not None:
            with torch.no_grad():
                limit = self.reference_actor.distribution.log_std_param + math.log(self.reference_std_multiplier)
                self.actor.distribution.log_std_param.copy_(torch.minimum(self.actor.distribution.log_std_param, limit))
        self.guard_updates += 1
        result['anchor'] = sum(self._anchor_losses)/max(len(self._anchor_losses), 1)
        result['critic_warmup'] = float(warmup)
        if self._stop_teacher_metrics:
            for i, name in enumerate(("stop_teacher", "stop_teacher_sample_fraction", "stop_teacher_preview_fraction")):
                result[name] = sum(row[i] for row in self._stop_teacher_metrics)/len(self._stop_teacher_metrics)
        if hasattr(self.actor, "take_rollout_metrics"):
            result.update(self.actor.take_rollout_metrics())
        return result

    def auxiliary_actor_loss(self, observations):
        """Task extension inside the same PPO optimizer step; legacy default is zero."""
        return observations["policy"].new_zeros(())

    def save(self):
        state = super().save()
        if self.reference_actor is not None:
            state['preservation_reference_actor'] = self.reference_actor.state_dict()
        state['preservation_updates'] = self.guard_updates
        if self.stop_teacher_actor is not None:
            state['stop_teacher_actor'] = self.stop_teacher_actor.state_dict()
        manifest = getattr(self, "m20_training_manifest", None)
        if manifest is not None:
            state["m20_training_manifest"] = copy.deepcopy(manifest)
        return state

    def load(self, loaded_dict, load_cfg=None, strict=True):
        result = super().load(loaded_dict, load_cfg, strict)
        # Optimizer.load_state_dict restores its old LR. With a fixed schedule
        # PPO never overwrites it, so explicitly honor the selected new LR.
        if self.schedule == 'fixed':
            for group in self.optimizer.param_groups:
                group['lr'] = self.learning_rate
        # Restore auxiliary state only for a full resume, never for actor-only initialization.
        if load_cfg is None or load_cfg.get('iteration', False):
            if 'stop_teacher_actor' in loaded_dict:
                self.set_stop_teacher(loaded_dict['stop_teacher_actor'])
            elif self.stop_teacher_coefficient > 0:
                raise ValueError("v4 full resume checkpoint is missing its frozen stop teacher")
            reference = loaded_dict.get('preservation_reference_actor')
            if reference is not None:
                self.set_reference(reference)
                self.guard_updates = int(loaded_dict.get('preservation_updates', 0))
            elif not getattr(self.actor, 'frozen_motion', False):
                # Legacy checkpoints also support play/full resume. Freeze the
                # actor being loaded, rather than silently disabling every guard.
                # Training should load a task-capable checkpoint; the CLI can
                # explicitly replace this reference with --preserve_actor_from.
                self.set_reference(loaded_dict['actor_state_dict'])
            if getattr(self.actor, 'frozen_motion', False):
                self.guard_updates = int(loaded_dict.get('preservation_updates', 0))
        return result

    def _guarded_update(self) -> dict[str, float]:
        """Run optimization epochs over stored batches and return mean losses."""
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        # RND loss
        mean_rnd_loss = 0 if self.rnd else None
        # Symmetry loss
        mean_symmetry_loss = 0 if self.symmetry else None

        # Get mini batch generator
        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # Iterate over batches
        for batch in generator:
            original_batch_size = batch.observations.batch_size[0]

            # Check if we should normalize advantages per mini batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)  # type: ignore

            # Perform symmetric augmentation
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                # Augmentation using symmetry
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                # Returned shape: [batch_size * num_aug, ...]
                batch.observations, batch.actions = data_augmentation_func(
                    env=self.symmetry["_env"],
                    obs=batch.observations,
                    actions=batch.actions,
                )
                # Compute number of augmentations per sample
                num_aug = int(batch.observations.batch_size[0] / original_batch_size)
                # Repeat the rest of the batch
                batch.old_actions_log_prob = batch.old_actions_log_prob.repeat(num_aug, 1)
                batch.values = batch.values.repeat(num_aug, 1)
                batch.advantages = batch.advantages.repeat(num_aug, 1)
                batch.returns = batch.returns.repeat(num_aug, 1)

            # Recompute actions log prob and entropy for current batch of transitions
            # Note: We need to do this because we updated the policy with the new parameters
            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)  # type: ignore
            values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
            # Note: We only keep the distribution parameters and entropy of the first augmentation (the original one)
            distribution_params = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
            entropy = self.actor.output_entropy[:original_batch_size]

            # Compute KL divergence and adapt the learning rate
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)  # type: ignore
                    kl_mean = torch.mean(kl)

                    # Reduce the KL divergence across all GPUs
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    # Update the learning rate only on the main process
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    # Update the learning rate for all GPUs
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    # Update the learning rate for all parameter groups
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))  # type: ignore
            surrogate = -torch.squeeze(batch.advantages) * ratio  # type: ignore
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(  # type: ignore
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()
            if self.reference_actor is not None and self.guard_updates >= self.critic_warmup_iterations:
                anchor_loss = self._anchor_loss(batch.observations)
                loss = loss + self.anchor_coefficient * anchor_loss
                self._anchor_losses.append(float(anchor_loss.detach()))
            if self.stop_teacher_coefficient > 0 and self.guard_updates >= self.critic_warmup_iterations:
                loss = loss + self.stop_teacher_coefficient*self._stop_teacher_loss(batch.observations)
            if self.stop_regularization_coefficient > 0 and self.guard_updates >= self.critic_warmup_iterations:
                loss = loss+self.stop_regularization_coefficient*self.actor.stop_regularization()
            if self.motion_regularization_coefficient > 0 and self.guard_updates >= self.critic_warmup_iterations:
                loss = loss+self.motion_regularization_coefficient*self.actor.motion_regularization()
            if self.guard_updates >= self.critic_warmup_iterations:
                loss = loss + self.auxiliary_actor_loss(batch.observations)

            # Symmetry loss
            if self.symmetry:
                # Obtain the symmetric actions
                # Note: If we did augmentation before then we don't need to augment again
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    batch.observations, _ = data_augmentation_func(
                        obs=batch.observations, actions=None, env=self.symmetry["_env"]
                    )

                # Actions predicted by the actor for symmetrically-augmented observations
                mean_actions = self.actor(batch.observations.detach().clone())

                # Compute the symmetrically augmented actions
                # Note: We are assuming the first augmentation is the original one. We do not use the batch.actions from
                # earlier since that action was sampled from the distribution. However, the symmetry loss is computed
                # using the mean of the distribution.
                action_mean_orig = mean_actions[:original_batch_size]
                _, actions_mean_symm = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
                )

                # Compute the loss
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions[original_batch_size:], actions_mean_symm.detach()[original_batch_size:]
                )
                # Add the loss to the total loss
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            # RND loss
            if self.rnd:
                # Extract the rnd_state
                with torch.no_grad():
                    rnd_state = self.rnd.get_rnd_state(batch.observations[:original_batch_size])  # type: ignore
                    rnd_state = self.rnd.state_normalizer(rnd_state)
                # Predict the embedding and the target
                predicted_embedding = self.rnd.predictor(rnd_state)
                target_embedding = self.rnd.target(rnd_state).detach()
                # Compute the loss as the mean squared error
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            # Compute the gradients for PPO
            self.optimizer.zero_grad()
            loss.backward()
            # Compute the gradients for RND
            if self.rnd:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters()

            # Apply the gradients for PPO
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            # Apply the gradients for RND
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            # RND loss
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            # Symmetry loss
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        # Divide the losses by the number of updates
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates

        # Clear the storage
        self.storage.clear()

        # Construct the loss dictionary
        loss_dict = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss

        return loss_dict


PROFILE = "idle_stop_motion_v1"
VERSION = "m20_idle_stop_motion_v1_20261017"
CONTROL_PROFILE = "legacy_v1"
MOTION_CONTROL_VERSION = "m20_raw_command_gate_20261017"
LEGACY_ACTOR_CLASS = "rl_training.motion_residual_actor:MotionResidualMLPModel"
ACTOR_CLASS = "task_support.idle_stop_motion_v1.motion_residual_actor:MotionResidualMLPModel"
# Upstream names this actor_sha256, but validates the complete checkpoint file.
BASELINE_CHECKPOINT_SHA256 = "0e6d0c93560b5a3f7ce9219c7b6a0cea84bd4a2d3f5dba33c02d79f7d4eb5f34"
BASELINE_CONFIG_SHA256 = "e7bce00a4d49cdceaf4935a678b83de7854aa84a49c828ba3b35d732f02092ec"
ACTOR_SOURCE_SHA256 = "c15973be1f5ddfd4c1888f402d7f70a5387fc866b57f237306ee9a7f5fb05129"


def configure_models(train_cfg):
    """Use the original 244D base and the explicit raw-command residual ABI."""
    configured = copy.deepcopy(train_cfg)
    configured["actor"].update(class_name=ACTOR_CLASS, residual_hidden_dims=[128, 64])
    configured["obs_groups"].update(actor=["policy", "motion_control"],
                                    critic=["critic", "motion_control"])
    return configured
