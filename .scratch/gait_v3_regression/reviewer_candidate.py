#!/usr/bin/env python3
"""Targeted CPU-only review of the repaired legacy load and config isolation."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage

ROOT = Path(__file__).resolve().parents[2]
torch.set_num_threads(1)


def package(name, path):
    value = types.ModuleType(name)
    value.__path__ = [str(path)]
    sys.modules[name] = value
    return value


def source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


guard_path = ROOT/'source/rl_training/rl_training/stair_guarded_ppo.py'
Guard = source('review_stair_guard', guard_path).StairGuardedPPO
obs = TensorDict({'policy': torch.zeros(4,244), 'critic': torch.zeros(4,285)}, [4])
groups = {'actor': ['policy'], 'critic': ['critic']}


def make(lr, warmup=50):
    actor = MLPModel(obs, groups, 'actor', 16, hidden_dims=[8], activation='elu',
        distribution_cfg={'class_name':'GaussianDistribution','init_std':.35,'std_type':'log'})
    critic = MLPModel(obs, groups, 'critic', 1, hidden_dims=[8], activation='elu')
    return Guard(actor, critic, RolloutStorage('rl',4,2,obs,[16]),
                 learning_rate=lr, schedule='fixed', critic_warmup_iterations=warmup)


legacy = make(.000170859375).save()
repaired = make(1e-5)
repaired.load(legacy, None, True)
assert repaired.learning_rate == 1e-5
assert all(group['lr'] == 1e-5 for group in repaired.optimizer.param_groups)
assert repaired.reference_actor is not None and repaired.guard_updates == 0
assert all(torch.equal(value, repaired.reference_actor.state_dict()[key])
           for key,value in legacy['actor_state_dict'].items())
assert all(not param.requires_grad for param in repaired.reference_actor.parameters())
partial = make(1e-5)
partial.load(legacy, {'actor':True}, True)
assert partial.reference_actor is None  # train.py supplies explicit reference on init
repaired.guard_updates = 63
saved = repaired.save()
resumed = make(1e-5)
resumed.load(saved, None, True)
assert resumed.guard_updates == 63 and resumed.reference_actor is not None
resumed.set_reference(legacy['actor_state_dict'])
assert resumed.guard_updates == 0

# The loss is differentiable into the actor and cannot update its frozen target.
with torch.no_grad(): repaired.actor.mlp[-1].bias.add_(.1)
loss_flat = repaired._anchor_loss(obs)
loss_flat.backward()
assert repaired.actor.mlp[-1].bias.grad.abs().sum() > 0
assert all(param.grad is None for param in repaired.reference_actor.parameters())
uneven = obs.clone()
uneven['policy'][:,57] = .2
loss_uneven = repaired._anchor_loss(uneven)
assert abs(float(loss_flat/loss_uneven) - 10) < 1e-4

# Use actual IsaacLab pure-Python configclass and RSL config definitions,
# loading their files without importing the simulator package initializer.
isaac = Path('/home/ubuntu/xyproject/m20_wzh/IsaacLab/source')
package('isaaclab', isaac/'isaaclab/isaaclab')
utils = package('isaaclab.utils', isaac/'isaaclab/isaaclab/utils')
utils.configclass = source('isaaclab.utils.configclass', isaac/'isaaclab/isaaclab/utils/configclass.py').configclass
package('isaaclab_rl', isaac/'isaaclab_rl/isaaclab_rl')
rl_pkg = package('isaaclab_rl.rsl_rl', isaac/'isaaclab_rl/isaaclab_rl/rsl_rl')
cfg = source('isaaclab_rl.rsl_rl.rl_cfg', isaac/'isaaclab_rl/isaaclab_rl/rsl_rl/rl_cfg.py')
for attr in ('RslRlOnPolicyRunnerCfg','RslRlPpoActorCriticCfg','RslRlPpoAlgorithmCfg'):
    setattr(rl_pkg, attr, getattr(cfg, attr))
agents = ROOT/'source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/agents'
package('review_config', agents)
stair = source('review_config.stair_teacher_ppo_cfg', agents/'stair_teacher_ppo_cfg.py')
pre = source('review_config.pre_teacher_ppo_cfg', agents/'pre_teacher_ppo_cfg.py')
stair_cfg = stair.DeeproboticsM20StairTeacherPPORunnerCfg()
pre_cfg = pre.DeeproboticsM20PreTeacherPPORunnerCfg()
assert stair_cfg.algorithm.class_name == 'rl_training.stair_guarded_ppo:StairGuardedPPO'
assert stair_cfg.algorithm.learning_rate == 1e-5 and stair_cfg.algorithm.schedule == 'fixed'
assert stair_cfg.algorithm.entropy_coef == 0 and stair_cfg.algorithm.clip_param == .1
assert stair_cfg.algorithm.critic_warmup_iterations == 50
assert pre_cfg.algorithm.class_name == 'PPO'
assert pre_cfg.algorithm.learning_rate == 1e-3 and pre_cfg.algorithm.schedule == 'adaptive'
assert pre_cfg.algorithm.entropy_coef == .003 and pre_cfg.algorithm.clip_param == .2
assert pre_cfg.experiment_name == 'deeprobotics_m20_pre_teacher'
# Instance mutation must not leak through configclass inherited defaults.
stair_cfg.algorithm.learning_rate = .123
assert pre.DeeproboticsM20PreTeacherPPORunnerCfg().algorithm.learning_rate == 1e-3
assert stair.DeeproboticsM20StairTeacherPPORunnerCfg().algorithm.learning_rate == 1e-5

output = {
    'guard_sha256':hashlib.sha256(guard_path.read_bytes()).hexdigest(),
    'legacy_resume_fixed_lr_actual':repaired.optimizer.param_groups[0]['lr'],
    'legacy_reference_frozen_and_warmup_reset':True,
    'full_resume_counter_restored':63,
    'explicit_reference_replacement_restarts_warmup':True,
    'partial_actor_load_requires_explicit_train_reference':True,
    'anchor_actor_gradient_nonzero_reference_gradient_none':True,
    'flat_vs_uneven_anchor_weight_ratio':float(loss_flat/loss_uneven),
    'official_configclass_stair_guard_pre_ppo_isolated':True,
    'pre_algorithm':pre_cfg.algorithm.to_dict(),
}
path = ROOT/'.scratch/gait_v3_regression/reviewer_candidate.json'
path.write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps(output,indent=2))
