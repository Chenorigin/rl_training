"""Named software-position contract shared by CTS learning and simulation.

These are SDK corrected-coordinate limits, not raw encoder or USD soft limits.
Targets are rejected, never silently clipped. Wheels have no position limit.
"""
from copy import deepcopy
import math


LEG_NAMES = tuple(f'{leg}_{joint}_joint' for leg in ('fl', 'fr', 'hl', 'hr')
                  for joint in ('hipx', 'hipy', 'knee'))
KEYS = {'version', 'joint_names', 'target_bounds_rad', 'feedback_bounds_rad',
        'default_position_rad', 'action_scale', 'normalized_action_clip',
        'margin_fraction', 'target_cost_weight', 'feedback_cost_weight',
        'student_loss_weight', 'source_robot_sha256'}


def validate_limits(cfg):
    if not isinstance(cfg, dict) or set(cfg) != KEYS or cfg['version'] != 1:
        raise ValueError('Invalid CTS software joint-limit contract')
    if tuple(cfg['joint_names']) != LEG_NAMES:
        raise ValueError('CTS limit joint order must be FL/FR/HL/HR hipx/hipy/knee')
    def numbers(values, length):
        return (isinstance(values, (list, tuple)) and len(values) == length
                and all(type(x) in (int, float) and math.isfinite(x) for x in values))
    for key in ('target_bounds_rad', 'feedback_bounds_rad'):
        rows = cfg[key]
        if (not isinstance(rows, (list, tuple)) or len(rows) != 12
                or any(not numbers(row, 2) or row[0] >= row[1] for row in rows)):
            raise ValueError('Invalid CTS position bounds: ' + key)
    for key in ('default_position_rad', 'action_scale'):
        if not numbers(cfg[key], 12):
            raise ValueError('Invalid CTS action mapping: ' + key)
    if any(x <= 0 for x in cfg['action_scale']):
        raise ValueError('CTS limit action scales must be positive')
    clip = cfg['normalized_action_clip']
    if not numbers(clip, 2) or list(clip) != [-100., 100.]:
        raise ValueError('CTS normalized action clip must match the deployed [-100, 100] ABI')
    for target, feedback, default in zip(cfg['target_bounds_rad'], cfg['feedback_bounds_rad'],
                                         cfg['default_position_rad']):
        if not feedback[0] <= target[0] < default < target[1] <= feedback[1]:
            raise ValueError('CTS defaults/target/feedback position bounds disagree')
    for key in ('margin_fraction', 'target_cost_weight', 'feedback_cost_weight', 'student_loss_weight'):
        if not numbers([cfg[key]], 1) or cfg[key] <= 0:
            raise ValueError('CTS limit costs and margin must be positive')
    if cfg['margin_fraction'] >= .5:
        raise ValueError('CTS limit margin must be less than half of range')
    sha = cfg['source_robot_sha256']
    if not isinstance(sha, str) or len(sha) != 64 or any(x not in '0123456789abcdef' for x in sha):
        raise ValueError('CTS position limits require a source robot SHA256')
    return cfg


class TorchJointLimits:
    """Batched float32 SDK comparisons and differentiable interior-margin cost."""
    def __init__(self, cfg, device='cpu'):
        import torch
        self.cfg = deepcopy(validate_limits(cfg))
        self.names = tuple(cfg['joint_names'])
        self.target = torch.tensor(cfg['target_bounds_rad'], device=device, dtype=torch.float32)
        self.feedback = torch.tensor(cfg['feedback_bounds_rad'], device=device, dtype=torch.float32)
        self.default = torch.tensor(cfg['default_position_rad'], device=device, dtype=torch.float32)
        self.scale = torch.tensor(cfg['action_scale'], device=device, dtype=torch.float32)

    def targets(self, action):
        if action.shape[-1] != 16:
            raise ValueError('CTS requires 16 policy actions')
        return self.default + action[..., :12].clamp(*self.cfg['normalized_action_clip']) * self.scale

    @staticmethod
    def violations(value, bounds):
        import torch
        if value.shape[-1] != 12:
            raise ValueError('CTS position checks require 12 named leg joints')
        return (~torch.isfinite(value) | (value < bounds[:, 0]) | (value > bounds[:, 1]))

    def cost(self, value, bounds):
        import torch
        margin = (bounds[:, 1]-bounds[:, 0])*self.cfg['margin_fraction']
        distance = ((bounds[:, 0]+margin-value).clamp_min(0)
                    +(value-bounds[:, 1]+margin).clamp_min(0))/margin
        return torch.nan_to_num(distance.square().mean(-1), nan=1., posinf=1., neginf=1.)

    def student_loss(self, action):
        return self.cfg['student_loss_weight']*self.cost(self.targets(action), self.target).mean()


class JointLimitMonitor:
    """Latch each substep's failure until reset; keep the terminal costs intact."""
    def __init__(self, cfg, num_envs, device):
        import torch
        self.limits = TorchJointLimits(cfg, device)
        self.target_fault = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.feedback_fault = self.target_fault.clone()
        self.corrected_fault = self.target_fault.clone()
        self.target_cost = torch.zeros(num_envs, device=device)
        self.feedback_cost = self.target_cost.clone()

    @property
    def faulted(self):
        return self.target_fault | self.feedback_fault | self.corrected_fault

    def begin(self, action, position):
        import torch
        target = self.limits.targets(action)
        self.target_fault |= self.limits.violations(target, self.limits.target).any(-1) | ~torch.isfinite(action).all(-1)
        self.target_cost.copy_(self.limits.cost(target, self.limits.target))
        self.feedback_cost.zero_()
        self.sample(position)

    def sample(self, position, corrected_target=None):
        import torch
        self.feedback_fault |= self.limits.violations(position, self.limits.feedback).any(-1)
        self.feedback_cost.copy_(torch.maximum(self.feedback_cost, self.limits.cost(position, self.limits.feedback)))
        if corrected_target is not None:
            self.corrected_fault |= self.limits.violations(corrected_target, self.limits.target).any(-1)
            self.target_cost.copy_(torch.maximum(self.target_cost, self.limits.cost(corrected_target, self.limits.target)))

    def reset(self, ids):
        for value in (self.target_fault, self.feedback_fault, self.corrected_fault,
                      self.target_cost, self.feedback_cost):
            value[ids] = 0

    def reward_cost(self):
        cfg = self.limits.cfg
        return cfg['target_cost_weight']*self.target_cost + cfg['feedback_cost_weight']*self.feedback_cost


def fault_damping(effort, velocity, damping, effort_limit, failed):
    """Override even delayed commands after actuator compute, before PhysX."""
    import torch
    safe = (-damping*torch.nan_to_num(velocity)).clamp(-effort_limit, effort_limit)
    return torch.where(failed[:, None], safe, effort)
