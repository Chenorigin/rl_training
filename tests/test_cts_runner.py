"""Recovery must never publish a partially applied joint optimizer update."""
from pathlib import Path
import sys
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'source/rl_training'))
from task_support.cts.runner import CTSRunner, replay_behavior_error
from task_support.cts.entry import validate_student_source


@pytest.mark.parametrize('phase', ['rollout', 'update'])
def test_interrupted_joint_update_does_not_create_a_resumable_partial_checkpoint(phase):
    alg = SimpleNamespace(num_envs=4, device='cpu', num_updates=0, num_steps_per_env=1,
                          act=Mock(return_value=torch.zeros(4,16)), process_env_step=Mock(),
                          compute_returns=Mock(), clear_rollout=Mock(),
                          update=Mock(side_effect=KeyboardInterrupt))
    env = SimpleNamespace(get_observations=Mock(return_value={}),
                          step=Mock(return_value=({},torch.zeros(4),torch.zeros(4),{})))
    if phase == 'rollout':
        env.step.side_effect = KeyboardInterrupt
    runner = CTSRunner(env, alg, {}, {}, '.', {})
    runner.save = Mock(side_effect=lambda name: Path(name))
    with patch('torch.utils.tensorboard.SummaryWriter'), pytest.raises(KeyboardInterrupt):
        runner.learn(1)
    names = [call.args[0] for call in runner.save.call_args_list]
    assert names == (['model_0.pt'] if phase == 'update' else ['model_0.pt','recovery_0.pt'])


def test_changed_student_activation_rejected_before_warm_start():
    metadata = {'contract':{}, 'training_config':{'student':{'hidden_dims':[512,256,128],
                'activation':'elu','obs_normalization':False}}, 'command_feedback':{'tuning':{'a':1}}}
    source = {'student_v1':{'provenance':metadata}}
    manifest = {'stair_tuning':{'a':1}}
    with patch('task_support.student_v3_safe.deployment.validate_provenance'), \
         patch('m20_student_v1.contract.validate_contract'):
        validate_student_source(source, manifest)
        metadata['training_config']['student']['activation'] = 'tanh'
        with pytest.raises(ValueError, match='ABI'):
            validate_student_source(source, manifest)


def test_behavior_verifier_accepts_original_batch_and_rejects_changed_probability():
    from test_cts_algorithm import fixture, collect
    models, alg, obs, *_ = fixture()
    collect(alg, obs)
    data = alg._data()
    with torch.inference_mode():
        assert replay_behavior_error(models, data, alg.num_envs) < 1.e-4
        data['log_prob'] = data['log_prob']+.5
        assert replay_behavior_error(models, data, alg.num_envs) > .49


def test_fatal_entry_cannot_be_turned_into_exit_zero_by_kit_close():
    # A separate CPU process exercises the real fatal-exit branch. The fake
    # simulator's close deliberately exits zero, reproducing Kit's failure mode.
    code = '''
import sys, os, types, importlib
from task_support.cts import entry
entry.prepare=lambda args:(None, {}, {}, {})
entry.preflight=lambda *args:{}
entry.configure_environment=lambda *args:types.SimpleNamespace(scene=types.SimpleNamespace(terrain=types.SimpleNamespace()))
app=types.SimpleNamespace(close=lambda:os._exit(0))
for name in ('isaaclab','isaaclab.app','isaaclab_rl','isaaclab_rl.rsl_rl','gymnasium'):
    sys.modules[name]=types.ModuleType(name)
sys.modules['isaaclab.app'].AppLauncher=lambda *args:types.SimpleNamespace(app=app)
sys.modules['isaaclab_rl.rsl_rl'].RslRlVecEnvWrapper=lambda value,**kwargs:value
def fail(*args,**kwargs):
    raise RuntimeError('controlled simulator construction failure')
sys.modules['gymnasium'].make=fail
real_import=importlib.import_module
importlib.import_module=lambda name,*a,**k:None if name=='rl_training.tasks' else real_import(name,*a,**k)
entry.main(['--resume','unused.pt','--validate-only'])
'''
    result = subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,timeout=20)
    assert result.returncode == 1
    assert 'controlled simulator construction failure' in result.stderr
