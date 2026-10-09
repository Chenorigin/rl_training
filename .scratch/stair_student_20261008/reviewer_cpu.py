#!/usr/bin/env python3
"""Read-only student scaffold reviewer; no simulator, training or GPU."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel

ROOT = Path(__file__).resolve().parents[2]
FILE = ROOT / 'source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/mdp/stair_student.py'
CFG = ROOT / 'source/rl_training/rl_training/tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/stair_student_env_cfg.py'
PPO = CFG.parent / 'agents/stair_student_ppo_cfg.py'
REG = CFG.parent / '__init__.py'
torch.set_num_threads(1)
spec = importlib.util.spec_from_file_location('reviewer_student_module', FILE)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

hits = torch.zeros(4, 187, 3)
hits[:, :, 2] = torch.linspace(-.2, .3, 187)
base = torch.tensor([[0., 0., .57]]).repeat(4, 1)
clean = module.StairStudentPerceptionCfg()
height, validity, confidence = module.build_terrain_channels(hits, base, clean)
teacher_height = (base[:, 2, None] - hits[..., 2] - .5).clamp(-1, 1)
assert torch.equal(height, teacher_height) and validity.eq(1).all() and confidence.eq(1).all()
out = {'clean_height_teacher_parity': True, 'height_shape': list(height.shape)}

bad_hits = hits.clone()
bad_hits[0, 2, 2] = torch.nan
bad_hits[1, 3, 0] = torch.inf
bad_base = base.clone()
bad_base[2, 2] = torch.nan
h, v, c = module.build_terrain_channels(bad_hits, bad_base, module.StairStudentPerceptionCfg(height_noise_std=.025, dropout_probability=.2))
assert all(torch.isfinite(t).all() for t in (h, v, c))
assert h[0, 2] == v[0, 2] == c[0, 2] == 0
assert h[1, 3] == v[1, 3] == c[1, 3] == 0
assert h[2].eq(0).all() and v[2].eq(0).all() and c[2].eq(0).all()
assert torch.equal(c > 0, v > 0)
out['nan_inf_student_channels_finite'] = True
out['teacher_raw_height_nan_survives_clip'] = bool(torch.isnan((bad_base[:, 2, None] - bad_hits[..., 2] - .5).clamp(-1, 1)).any())

h, v, c = module.build_terrain_channels(hits, base, module.StairStudentPerceptionCfg(dropout_probability=1.))
assert all(t.eq(0).all() for t in (h, v, c))
out['full_dropout_zero_placeholders'] = True

sensor = SimpleNamespace(data=SimpleNamespace(ray_hits_w=hits.clone(), pos_w=base.clone()))
env = SimpleNamespace(common_step_counter=9, episode_length_buf=torch.full((4,), 3),
    cfg=SimpleNamespace(student_perception=module.StairStudentPerceptionCfg(height_noise_std=.025, dropout_probability=.2)),
    scene=SimpleNamespace(sensors={'height_scanner': sensor}))
entity = SimpleNamespace(name='height_scanner')
a = module.student_height_scan(env, entity)
v = module.student_height_validity(env, entity)
c = module.student_height_confidence(env, entity)
assert torch.equal(module.student_height_scan(env, entity), a)
assert torch.equal(module.student_height_validity(env, entity), v)
assert torch.equal(module.student_height_confidence(env, entity), c)
a.fill_(999)
assert not module.student_height_scan(env, entity).eq(999).any()
env.episode_length_buf[0] = 0
before = module.student_height_scan(env, entity)
env.common_step_counter += 1
assert not torch.equal(module.student_height_scan(env, entity), before)
out['atomic_reuse_clone_and_step_refresh'] = True

# Reproduce observation-manager dimension probing then initial reset: the
# global step and zero episode lengths are identical, but reset changes pose.
env.cfg.student_perception = clean
env.common_step_counter = 0
env.episode_length_buf.zero_()
env._m20_student_terrain_cache = None
sensor.data.ray_hits_w.zero_()
sensor.data.pos_w.copy_(base)
probe = module.student_height_scan(env, entity)
sensor.data.ray_hits_w[:, :, 2] = .1
after_reset = module.student_height_scan(env, entity)
fresh = module.build_terrain_channels(sensor.data.ray_hits_w, sensor.data.pos_w, clean)[0]
out['same_step_zero_episode_reset_without_event'] = dict(cache_return=float(after_reset[0, 0]),
    correct_fresh_height=float(fresh[0, 0]), max_error=float((after_reset - fresh).abs().max()),
    stale=bool(torch.equal(probe, after_reset) and not torch.equal(fresh, after_reset)))
assert out['same_step_zero_episode_reset_without_event']['stale']
module.reset_student_perception(env, torch.arange(4))
with_event = module.student_height_scan(env, entity)
assert torch.equal(with_event, fresh)
out['same_step_zero_episode_reset_with_event'] = dict(cache_return=float(with_event[0, 0]),
    correct_fresh_height=float(fresh[0, 0]), max_error=float((with_event - fresh).abs().max()))
cfg_tree = ast.parse(CFG.read_text())
reset_assignment = next(node for node in ast.walk(cfg_tree)
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Attribute) and t.attr == 'student_perception_reset' for t in node.targets))
assert isinstance(reset_assignment.value, ast.Call)
assert any(k.arg == 'mode' and isinstance(k.value, ast.Constant) and k.value.value == 'reset' for k in reset_assignment.value.keywords)
assert any(k.arg == 'func' and isinstance(k.value, ast.Attribute) and k.value.attr == 'reset_student_perception' for k in reset_assignment.value.keywords)
out['reset_event_wiring_static'] = True

teacher = MLPModel(TensorDict({'teacher': torch.zeros(2, 244)}, batch_size=[2]),
    {'actor': ['teacher']}, 'actor', 16, hidden_dims=[512, 256, 128], activation='elu', obs_normalization=False,
    distribution_cfg={'class_name': 'rsl_rl.modules.distribution:GaussianDistribution', 'init_std': 1., 'std_type': 'log'})
student = MLPModel(TensorDict({'policy': torch.zeros(2, 618)}, batch_size=[2]),
    {'actor': ['policy']}, 'actor', 16, hidden_dims=[512, 256, 128], activation='elu', obs_normalization=False,
    distribution_cfg={'class_name': 'rsl_rl.modules.distribution:GaussianDistribution', 'init_std': 1., 'std_type': 'log'})
source_state = teacher.state_dict()
source_copy = {k: t.clone() for k, t in source_state.items()}
module.initialize_student_actor(student, source_state)
x = torch.randn(8, 244)
y = torch.cat((x, torch.randn(8, 374)), -1)
with torch.no_grad():
    t_out = teacher(TensorDict({'teacher': x}, batch_size=[8]))
    s_out = student(TensorDict({'policy': y}, batch_size=[8]))
error = float((t_out - s_out).abs().max())
assert error < 1e-6
assert student.state_dict()['mlp.0.weight'][:, 244:].eq(0).all()
assert all(torch.equal(source_copy[k], source_state[k]) for k in source_state)
out['real_rsl_mlp_actor_expansion'] = dict(max_action_error=error, extra_columns_zero=True, source_unchanged=True,
    actor_keys=list(source_state), action_shape=list(s_out.shape))

sa = torch.randn(3, 16, requires_grad=True)
ta = torch.randn(3, 16, requires_grad=True)
loss = module.action_distillation_loss(sa, ta)
loss.backward()
assert sa.grad is not None and ta.grad is None
out['distillation_teacher_grad_detached'] = True

# Execute actual registration calls with a local registry recorder; no task
# package import or simulator/environment creation is needed.
tree = ast.parse(REG.read_text())
tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
registrations = []
exec(compile(tree, str(REG), 'exec'), {'gym': SimpleNamespace(register=lambda **kw: registrations.append(kw)),
    '__name__': 'rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20',
    'agents': SimpleNamespace(__name__='rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.agents')})
entry = next(r for r in registrations if r['id'] == 'Rough-Deeprobotics-M20-StairStudent-v0')
assert entry['kwargs']['env_cfg_entry_point'].endswith('stair_student_env_cfg:DeeproboticsM20StairStudentEnvCfg')
assert entry['kwargs']['rsl_rl_cfg_entry_point'].endswith('stair_student_ppo_cfg:DeeproboticsM20StairStudentPPORunnerCfg')
out['registration_entry'] = entry
out['source_sha256'] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in (FILE, CFG, PPO, REG)}
Path(__file__).with_suffix('.json').write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps(out, indent=2))
