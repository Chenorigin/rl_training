"""CTS auxiliaries on complete, decoded nominal actions.

The decoder is frozen by the caller, not detached from its input. Offline
rehearsal uses the original collection commands from a pinned sidecar; sensor
heading feedback is never substituted for the requested command.
"""
from copy import deepcopy
import json
import math
from pathlib import Path

import torch

from task_support.student_v1.stability import (
    action_group_mse, adjacent_training_pairs, perturb_sensors,
    sensor_noise_std, temporal_residual_loss,
)
from task_support.student_v1.retention import (
    CLASSES, ROOT, MotionRetention, checked_manifest, retention_bytes as data_bytes,
)
from task_support.student_v1.model_io import file_hash


# A model-only fixture or caller must opt in explicitly. Production defaults
# are supplied by configs/distillation/cts.json via the real entry point.
DISABLED = dict(temporal_loss_weight=0., spatial_loss_weight=0.,
                stability_batch_size=256, spatial_noise_scale=.5,
                retention_loss_weight=0., retention_manifest=None, retention_sha256=None,
                retention_command_manifest=None, retention_command_sha256=None,
                retention_batch_size=256, leg_loss_weight=1., wheel_loss_weight=1.)


def validate_config(cfg):
    if not isinstance(cfg, dict) or set(cfg) != set(DISABLED):
        raise ValueError('Unknown or missing CTS auxiliary configuration keys')
    for key in ('temporal_loss_weight', 'spatial_loss_weight', 'retention_loss_weight',
                'spatial_noise_scale', 'leg_loss_weight', 'wheel_loss_weight'):
        value = cfg[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f'Invalid CTS auxiliary {key}')
    if min(cfg['leg_loss_weight'], cfg['wheel_loss_weight']) <= 0:
        raise ValueError('CTS auxiliary action group weights must be positive')
    for key, minimum in (('stability_batch_size', 1), ('retention_batch_size', len(CLASSES))):
        if type(cfg[key]) is not int or cfg[key] < minimum:
            raise ValueError(f'Invalid CTS auxiliary {key}')
    if cfg['spatial_loss_weight'] and cfg['spatial_noise_scale'] <= 0:
        raise ValueError('CTS spatial loss requires a positive perturbation scale')
    if cfg['retention_loss_weight']:
        for key in ('retention_manifest', 'retention_command_manifest'):
            if not isinstance(cfg[key], str) or not cfg[key]:
                raise ValueError(f'CTS auxiliary requires {key}')
        for key in ('retention_sha256', 'retention_command_sha256'):
            value = cfg[key]
            if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError(f'Invalid CTS auxiliary {key}')
    return cfg


def validate_context(cfg, perception):
    """Derive perturbations without reading training data (safe for deployment)."""
    validate_config(cfg)
    if not cfg['spatial_loss_weight']:
        return None
    if not isinstance(perception, dict):
        raise ValueError('CTS spatial loss requires the actual perception configuration')
    from m20_student_v1.configuration import StudentV1PerceptionCfg
    sensor = StudentV1PerceptionCfg(**perception)
    sensor.validate()
    return sensor_noise_std(sensor)


def command_contract(cfg):
    _, _, manifest = checked_manifest(cfg['retention_manifest'], cfg['retention_sha256'])
    path = Path(cfg['retention_command_manifest']).expanduser()
    path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_file() or file_hash(path) != cfg['retention_command_sha256']:
        raise ValueError('CTS retention command manifest missing or hash changed')
    commands = json.loads(path.read_text())
    if (commands.get('schema') != 'm20_cts_retention_commands_v1'
            or commands.get('classes') != list(CLASSES)
            or commands.get('retention_manifest_sha256') != cfg['retention_sha256']
            or commands.get('payload_sha256') != manifest['payload_sha256']
            or commands.get('source_checkpoint_sha256') != manifest['source_checkpoint_sha256']):
        raise ValueError('CTS retention command/data provenance mismatch')
    table = torch.as_tensor(commands.get('commands'), dtype=torch.float32)
    if table.shape != (len(CLASSES), 3) or not torch.isfinite(table).all():
        raise ValueError('Invalid CTS retention commands')
    # Five pure collection commands in the declared class order. In particular
    # a zero request must never enable the decoder's moving gate by accident.
    expected = torch.tensor([[-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
    if not torch.equal(table.sign(), expected) or not (table.abs().amax(-1) > 1.e-4).all():
        raise ValueError('CTS retention command classes/directions changed')
    return table, {'command_manifest': str(path),
                   'command_manifest_sha256': cfg['retention_command_sha256'],
                   'commands': table.tolist(),
                   'source_plan_sha256': commands.get('source_plan_sha256')}


def retention_bytes(cfg):
    validate_config(cfg)
    if not cfg['retention_loss_weight']:
        return 0
    command_contract(cfg)
    _, _, manifest = checked_manifest(cfg['retention_manifest'], cfg['retention_sha256'])
    # The phase sampler holds one bool per row and a second partition of row
    # indices; all network batches remain bounded by retention_batch_size.
    phases = manifest['sample_count']*9 if manifest.get('sampling') is not None else 0
    return data_bytes(cfg['retention_manifest'], cfg['retention_sha256']) + len(CLASSES)*3*4 + phases


def context_contract(cfg, perception):
    # Small CPU data load also validates split/class counts and array shapes.
    return AuxiliaryObjectives(None, cfg, perception, 'cpu').contract


class AuxiliaryObjectives:
    def __init__(self, models, cfg=None, perception=None, device='cpu'):
        self.models, self.device = models, device
        self.cfg = deepcopy(DISABLED if cfg is None else cfg)
        self.noise_std = validate_context(self.cfg, perception)
        self.retention = self.commands = None
        self.memory_bytes = retention_bytes(self.cfg)
        receipt = None
        if self.cfg['retention_loss_weight']:
            table, commands = command_contract(self.cfg)
            self.retention = MotionRetention(self.cfg['retention_manifest'], self.cfg['retention_sha256'], device)
            self.commands = table.to(device)
            receipt = {**self.retention.receipt, **commands}
        self.startup = self.phase_pools = None
        if self.retention is not None:
            _, _, manifest = checked_manifest(self.cfg['retention_manifest'], self.cfg['retention_sha256'])
            sampling = manifest.get('sampling')
            if sampling is not None:
                if (not isinstance(sampling, dict) or sampling.get('mode') != 'class_startup_balanced_v1'
                        or type(sampling.get('startup_fraction')) not in (int, float)
                        or not 0 < sampling['startup_fraction'] < 1
                        or type(sampling.get('startup_policy_steps')) is not int
                        or sampling['startup_policy_steps'] < 1):
                    raise ValueError('Invalid CTS retention startup sampling contract')
                indices = sampling.get('startup_indices')
                n = len(self.retention.obs)
                if (not isinstance(indices, list) or not indices
                        or any(type(i) is not int or not 0 <= i < n for i in indices)
                        or len(set(indices)) != len(indices) or self.cfg['retention_batch_size'] < 2*len(CLASSES)):
                    raise ValueError('Invalid CTS retention startup indices/batch size')
                self.startup = torch.zeros(n, device=device, dtype=torch.bool)
                self.startup[indices] = True
                self.startup_fraction = sampling['startup_fraction']
                self.phase_pools = {split: [(pool[self.startup[pool]], pool[~self.startup[pool]])
                                           for pool in pools] for split, pools in self.retention.pools.items()}
                if any(not len(pool) for pairs in self.phase_pools.values() for pair in pairs for pool in pair):
                    raise ValueError('Every CTS class/split requires startup and later-motion samples')
                receipt['sampling'] = deepcopy(sampling)
        self.contract = {'version': 1, 'config': deepcopy(self.cfg),
                         'sensor_noise_std': deepcopy(self.noise_std), 'retention': receipt}

    def regular_phase(self, extras, n):
        if not (self.cfg['temporal_loss_weight'] or self.cfg['spatial_loss_weight']):
            return torch.ones(n, dtype=torch.bool, device=self.device)
        phase = extras.get('student_v1', {}).get('phase')
        if (not isinstance(phase, torch.Tensor) or phase.shape != (n,)
                or phase.dtype not in (torch.int32, torch.int64)
                or not ((phase >= 0) & (phase <= 3)).all()):
            raise ValueError('CTS stability requires valid pre-step student_v1 phase tags')
        return (phase.to(self.device) == 0).detach().clone()

    def sample_retention(self, *, heldout=False):
        if self.phase_pools is None:
            return (*self.retention.sample(self.cfg['retention_batch_size'], heldout=heldout), None)
        count = self.cfg['retention_batch_size']
        chunks = []
        for index, pair in enumerate(self.phase_pools['heldout' if heldout else 'train']):
            size = count//len(CLASSES)+int(index < count % len(CLASSES))
            first = min(size-1, max(1, round(size*self.startup_fraction)))
            for pool, quota in zip(pair, (first, size-first)):
                chunks.append(pool[torch.randint(len(pool), (quota,), device=self.device)])
        ids = torch.cat(chunks)
        return self.retention.obs[ids], self.retention.labels[ids], self.retention.classes[ids], self.startup[ids]

    def prepare(self, rollout):
        regular = torch.stack([step['regular_phase'] for step in rollout])
        empty = torch.empty(0, dtype=torch.long, device=self.device)
        current, previous = empty, empty
        if self.cfg['temporal_loss_weight']:
            policy = torch.stack([step['obs']['policy'] for step in rollout])
            commands = torch.stack([step['obs']['motion_control'] for step in rollout])
            done = torch.stack([step['dones'] for step in rollout])
            # CTS has no online held-out split: all on-policy states may train.
            time, env = adjacent_training_pairs(policy, done, torch.ones_like(regular), regular, commands)
            current = time*regular.shape[1]+env
            previous = (time-1)*regular.shape[1]+env
        return {'regular': regular.flatten(), 'pairs_current': current, 'pairs_previous': previous}

    def retention_loss(self, *, detach_student=False):
        """Rehearse the deployed action in both optimization stages.

        PPO owns the decoder and must protect it on non-forward states too.
        Detaching only the encoder preserves this gradient while keeping the
        two optimizers disjoint. Distillation instead freezes decoder weights
        at its caller and differentiates through its input as before.
        """
        cfg = self.cfg
        if self.retention is None:
            return self.models.log_std.new_zeros(()), {'retention_mse': 0., 'retention_training_samples': 0}
        policy, labels, classes, startup = self.sample_retention()
        command = self.commands[classes]
        if detach_student:
            with torch.no_grad():
                latent = self.models.student_latent({'policy': policy})
            actions = self.models.decoder(policy[:, :57], latent.detach(), command)
        else:
            actions = self.models.student_action(policy, command)
        residual = actions-labels
        safety = getattr(self, 'action_safety', None)
        accepted = safety.accept(labels) if safety is not None else None
        if accepted is not None:
            residual = torch.where(accepted[:, None], residual, torch.zeros_like(residual))
        weights = (cfg['leg_loss_weight'], cfg['wheel_loss_weight'])
        value = action_group_mse(residual, *weights)
        metrics = dict(retention_mse=float(value.detach()), retention_training_samples=len(classes))
        for index, name in enumerate(CLASSES):
            metrics[f'retention_{name}_mse'] = float(action_group_mse(residual[classes == index], *weights).detach())
            if startup is not None:
                for phase, mask in (('startup', startup), ('later', ~startup)):
                    metrics[f'retention_{name}_{phase}_mse'] = float(action_group_mse(residual[(classes == index) & mask], *weights).detach())
        penalty = value.new_zeros(())
        if accepted is not None:
            metrics['retention_rejected_fraction'] = float((~accepted).float().mean())
            penalty = safety.cfg['student_tail_weight']*safety.tail_cost(actions)
        return cfg['retention_loss_weight']*value + penalty, metrics

    def loss(self, data, ids, target_actions, selection):
        cfg, models = self.cfg, self.models
        weights = (cfg['leg_loss_weight'], cfg['wheel_loss_weight'])
        total = target_actions.new_zeros(())
        metrics = {name: 0. for name in ('temporal_residual_mse', 'spatial_action_mse', 'retention_mse',
                                        'temporal_samples', 'spatial_samples', 'retention_training_samples')}
        pair_count = selection['pairs_current'].numel()
        if cfg['temporal_loss_weight'] and pair_count:
            selected = torch.randint(pair_count, (min(cfg['stability_batch_size'], pair_count),), device=self.device)
            current, previous = (selection[name][selected] for name in ('pairs_current', 'pairs_previous'))
            both = torch.cat((current, previous))
            actions = models.student_action(data['obs']['policy'][both], data['obs']['motion_control'][both])
            value = temporal_residual_loss(*actions.chunk(2), target_actions[current], target_actions[previous], *weights)
            value = value*selection.get('pair_safe_fraction',1.)
            total = total + cfg['temporal_loss_weight']*value
            metrics.update(temporal_residual_mse=float(value.detach()), temporal_samples=len(selected))
        if cfg['spatial_loss_weight']:
            selected = ids[selection['regular'][ids]][:cfg['stability_batch_size']]
            if selected.numel():
                policy, command = data['obs']['policy'][selected], data['obs']['motion_control'][selected]
                perturbed = perturb_sensors(policy, self.noise_std, cfg['spatial_noise_scale'])
                value = action_group_mse(models.student_action(perturbed, command)-models.student_action(policy, command), *weights)
                total = total + cfg['spatial_loss_weight']*value
                metrics.update(spatial_action_mse=float(value.detach()), spatial_samples=len(selected))
        if self.retention is not None:
            retention, retention_metrics = self.retention_loss()
            total = total + retention
            metrics.update(retention_metrics)
        return total, metrics

    @torch.no_grad()
    def evaluate(self):
        if self.retention is None:
            return {}
        policy, labels, classes, startup = self.sample_retention(heldout=True)
        residual = self.models.student_action(policy, self.commands[classes])-labels
        weights = (self.cfg['leg_loss_weight'], self.cfg['wheel_loss_weight'])
        metrics = {'retention_heldout_mse': float(action_group_mse(residual, *weights))}
        for index, name in enumerate(CLASSES):
            metrics[f'retention_{name}_heldout_mse'] = float(action_group_mse(residual[classes == index], *weights))
            if startup is not None:
                for phase, mask in (('startup', startup), ('later', ~startup)):
                    metrics[f'retention_{name}_{phase}_heldout_mse'] = float(action_group_mse(residual[(classes == index) & mask], *weights))
            for split in ('train', 'heldout'):
                metrics[f'retention_{name}_{split}_population'] = self.retention.pools[split][index].numel()
        return metrics
