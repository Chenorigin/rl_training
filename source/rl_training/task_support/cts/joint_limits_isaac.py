"""CTS-only Isaac hooks; retain legacy actors and simulator implementations."""
from copy import deepcopy

import torch
from isaaclab.actuators import DelayedPDActuator

from .joint_limits import JointLimitMonitor, fault_damping, validate_limits


class JointLimitPDActuator(DelayedPDActuator):
    """Do not let a queued position command bypass a latched software fault."""
    limit_monitor = None

    def compute(self, control_action, joint_pos, joint_vel):
        output = super().compute(control_action, joint_pos, joint_vel)
        if self.limit_monitor is None:
            raise RuntimeError('CTS joint-limit actuator is missing its monitor')
        output.joint_efforts = fault_damping(output.joint_efforts, joint_vel,
            self.fault_damping, self.effort_limit, self.limit_monitor.faulted)
        self.applied_effort.copy_(output.joint_efforts)
        return output


class LimitCheckedLanding:
    """Preserve the frozen teacher's physics loop; guard its existing adapter slot."""
    def __init__(self, env, delegate):
        self.env, self.delegate = env, delegate

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def apply(self, dt):
        self.delegate.apply(dt)
        sample(self.env, targets=True)


def configure(cfg, limits):
    cfg.cts_joint_limits = deepcopy(validate_limits(limits))
    for actuator in cfg.scene.robot.actuators.values():
        if actuator.class_type is not DelayedPDActuator:
            raise ValueError('CTS software limits require explicit DelayedPD actuators')
        actuator.class_type = JointLimitPDActuator


def install(env):
    cfg = getattr(env.cfg, 'cts_joint_limits', None)
    if cfg is None:
        return
    robot = env.scene['robot']
    guard = JointLimitMonitor(cfg, env.num_envs, env.device)
    ids, names = robot.find_joints(list(guard.limits.names), preserve_order=True)
    if tuple(names) != guard.limits.names or tuple(env.cfg.leg_joint_names) != guard.limits.names:
        raise ValueError('CTS limit/action named-joint order differs')
    if not torch.allclose(robot.data.default_joint_pos[:, ids], guard.limits.default.expand(env.num_envs, -1), atol=1.e-6, rtol=0):
        raise ValueError('CTS limit default pose differs from the actual robot')
    # Check the real action term rather than trusting a second declaration.
    term = env.action_manager.get_term('joint_pos')
    if tuple(term._joint_names) != guard.limits.names:
        raise ValueError('CTS position action order differs from software limits')
    scale = torch.as_tensor(term._scale, device=env.device).expand(env.num_envs, 12)
    offset = torch.as_tensor(term._offset, device=env.device).expand(env.num_envs, 12)
    if not torch.allclose(scale, guard.limits.scale.expand_as(scale), atol=1.e-6, rtol=0) or not torch.allclose(offset, guard.limits.default.expand_as(offset), atol=1.e-6, rtol=0):
        raise ValueError('CTS action scale/offset differs from software-limit contract')
    env.cts_limit_monitor, env.cts_limit_leg_ids = guard, ids
    env.stair_quiet_landing = LimitCheckedLanding(env, env.stair_quiet_landing)
    for actuator in robot.actuators.values():
        if not isinstance(actuator, JointLimitPDActuator):
            raise ValueError('CTS fault damping is not connected to every actuator')
        actuator.limit_monitor = guard
        actuator.fault_damping = actuator.damping.clone()


def begin(env, action):
    guard = getattr(env, 'cts_limit_monitor', None)
    if guard is None:
        return action
    action = action.to(env.device)
    guard.begin(action, env.scene['robot'].data.joint_pos[:, env.cts_limit_leg_ids])
    # Faulted episodes are terminal. No invalid request enters the delay queue;
    # actuator output additionally switches to damping for ALL remaining substeps.
    return torch.where(guard.faulted[:, None], torch.zeros_like(action), action)


def sample(env, *, targets=False):
    guard = getattr(env, 'cts_limit_monitor', None)
    if guard is not None:
        data, ids = env.scene['robot'].data, env.cts_limit_leg_ids
        guard.sample(data.joint_pos[:, ids], data.joint_pos_target[:, ids] if targets else None)


def snapshot(env):
    guard = getattr(env, 'cts_limit_monitor', None)
    if guard is not None:
        env._cts_limit_step_metrics = {
            'CTSJointLimits/target_fault_fraction': float(guard.target_fault.float().mean()),
            'CTSJointLimits/feedback_fault_fraction': float(guard.feedback_fault.float().mean()),
            'CTSJointLimits/corrected_target_fault_fraction': float(guard.corrected_fault.float().mean()),
            'CTSJointLimits/target_margin_cost': float(guard.target_cost.mean()),
            'CTSJointLimits/feedback_margin_cost': float(guard.feedback_cost.mean()),
        }
