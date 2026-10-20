"""V3.5 full teacher labels, checkpoint roundtrip and old v3 isolation. No updates."""
from copy import deepcopy
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from tensordict import TensorDict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'source/rl_training'))
sys.path.insert(0, str(ROOT/'deploy/deploy_mujoco'))
from m20_student_v1.v3_5_teacher import (TASK, teacher_manifest, initialization_source, _module)
from m20_student_v1.v3_5_feedback import LegacyHeadingFeedback, SensorHeadingFeedback, stair_scan_mask
from m20_student_v1.distillation import StudentV1Distillation
from m20_student_v1.model_io import initialize_student
from m20_student_v1.contract import observation_contract
from m20_student_v1.perception import grid_points
from student_v1_adapter import load_student, profile_config
from student_v1_adapter import StudentMapAdapter

spec = importlib.util.spec_from_file_location('v35_entry', ROOT/'scripts/reinforcement_learning/rsl_rl/distill_student.py')
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)
CHECKPOINT = ROOT/'deploy/models/idle_stop_motion_v1/idle_stop_motion_v1_46400.pt'
torch.set_num_threads(1)


@unittest.skipUnless(CHECKPOINT.is_file(), 'real model_46400 checkpoint required')
class ResidualDistillationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.saved = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
        cls.manifest = teacher_manifest(cls.saved)

    def configuration(self):
        args = entry.parse_args(['--task', TASK, '--teacher-checkpoint', str(CHECKPOINT), '--device', 'cpu', '--num-envs', '4'])
        entry.configure_run(args, self.saved)
        training, perception, _, _, _ = entry.settings_v1(args, self.saved)
        return args, training, perception

    def construct(self):
        _, cfg, _ = self.configuration()
        cfg['multi_gpu'] = None
        cfg['num_steps_per_env'] = 1
        obs = TensorDict({'policy': torch.randn(4, 789), 'teacher': torch.randn(4, 244),
                          'motion_control': torch.tensor([[0.,0.,0.],[.5,0.,0.],[0.,0.,.6],[-.5,0.,0.]])}, [4])
        alg = StudentV1Distillation.construct_algorithm(obs, SimpleNamespace(num_envs=4, num_actions=16), deepcopy(cfg), 'cpu')
        alg.load(self.saved, {'teacher': True, 'iteration': False}, True)
        initialize_student(alg.student, initialization_source(self.saved))
        return alg, obs, cfg

    def test_task_noise_and_dimensions(self):
        args, cfg, perception = self.configuration()
        old = json.loads((ROOT/'configs/distillation/student_v3_front_only.json').read_text())
        self.assertEqual(args.log_root, ROOT/'logs/rsl_rl/deeprobotics_m20_student_v3_5')
        self.assertEqual(args.config, ROOT/'configs/distillation/student_v3_5.json')
        self.assertEqual(cfg['history_steps'], 4)
        self.assertEqual(asdict(perception), old['perception'])
        self.assertEqual(cfg['student']['hidden_dims'], [512,256,128])
        bad = entry.parse_args(['--task', TASK, '--teacher-checkpoint', str(CHECKPOINT), '--history-steps', '1'])
        entry.configure_run(bad, self.saved)
        with self.assertRaisesRegex(ValueError, '789D'):
            entry.settings_v1(bad, self.saved)

    def test_teacher_full_parity_raw_zero_and_strict_source(self):
        alg, obs, cfg = self.construct()
        packed = TensorDict({'policy': obs['teacher'], 'motion_control': obs['motion_control']}, [4])
        config = deepcopy(cfg['teacher']);config.pop('class_name')
        reference = _module.MotionResidualMLPModel(packed, {'actor':['policy']}, 'actor',16,**config)
        reference.load_state_dict(self.saved['actor_state_dict'], strict=True)
        # Effective yaw does not activate the zero-raw residual branch.
        obs['teacher'][0,8] = .5
        with torch.inference_mode():
            actual = alg.teacher(obs)
            self.assertTrue(torch.equal(actual, reference(packed)))
            self.assertTrue(torch.equal(actual[0], alg.teacher.mlp(obs['teacher'])[0]))
            self.assertGreater(float((actual[1:]-alg.teacher.mlp(obs['teacher'])[1:]).abs().max()), 0.)
        self.assertFalse(any(p.requires_grad for p in alg.teacher.parameters()))
        self.assertTrue(set(alg.teacher.state_dict()) == set(self.saved['actor_state_dict']))
        bad = deepcopy(self.saved);bad['m20_training_manifest']['source_sha256'].pop('motion_residual_actor.py')
        with self.assertRaisesRegex(ValueError,'hash'):
            teacher_manifest(bad)
        bad_weights = deepcopy(self.saved)
        bad_weights['actor_state_dict'].pop('residual_mlp.0.weight')
        with self.assertRaises(RuntimeError):
            alg.load(bad_weights, {'teacher':True}, True)
        with self.assertRaisesRegex(ValueError, 'init-student'):
            initialization_source(self.saved, self.saved)

    def test_real_collection_save_resume_deploy_without_optimizer_step(self):
        with patch.object(torch.optim.Adam, 'step', side_effect=AssertionError('no training authorized')):
            alg, obs, cfg = self.construct()
            _, _, perception = self.configuration()
            alg.provenance = {'task':TASK, 'task_version':'v3.5', 'num_envs':4,
                'teacher_manifest':self.manifest, 'training_config':cfg,
                'perception':asdict(perception), 'contract':observation_contract(4),
                'command_feedback':{'version':'legacy_v1_v3_5','tuning':self.manifest['stair_tuning']}}
            with torch.inference_mode():
                labels = alg.teacher(obs).clone()
                alg.act(obs)
                self.assertTrue(torch.equal(alg.transition.privileged_actions, labels))
                tags = {k:torch.zeros(4,dtype=torch.long) for k in ('phase','terrain','direction','terrain_level')}
                tags['terminated'] = torch.zeros(4,dtype=torch.bool)
                alg.process_env_step(obs, torch.zeros(4), torch.zeros(4), {'student_v1':tags})
            self.assertTrue(torch.equal(alg.replay_labels[:4], labels))
            state = alg.save()
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp)/'model_0.pt';torch.save(state,path)
                state = torch.load(path,map_location='cpu',weights_only=True)
                restored, _, _ = self.construct();restored.load(state)
                with torch.inference_mode():
                    self.assertTrue(torch.equal(restored.teacher(obs),labels))
                self.assertEqual(restored.num_updates,0)
                self.assertEqual(len(restored.optimizer.state),0)
                args = entry.parse_args(['--resume',str(path),'--device','cpu'])
                entry.configure_run(args,state)
                training, _, task, _, _ = entry.settings_v1(args,state)
                self.assertEqual(task,TASK)
                self.assertEqual(training['teacher'],cfg['teacher'])
                # Exercise the real CPU resume preflight, not only alg.load.
                entry.preflight_v1(args, state, training, perception, 4, path)
                policy, provenance, identity = load_student(path)
                _, extras = profile_config('student-v1',provenance)
                self.assertEqual(identity['actor_input_dim'],789)
                self.assertIsNotNone(extras.command_feedback)
                with torch.inference_mode():
                    self.assertTrue(torch.equal(policy(obs['policy']),alg.student(obs)))

    def test_feedback_same_tick_and_reset(self):
        controller = LegacyHeadingFeedback(2,'cpu',self.manifest['stair_tuning'])
        raw = torch.tensor([[.5,0.,0.],[0.,0.,0.]])
        zeros = torch.zeros(2);stairs = torch.tensor([False,True])
        controller.update(raw,zeros,zeros,stairs,0)
        a = controller.update(raw,torch.tensor([.2,.2]),zeros,stairs,1)
        self.assertLess(a[0,2].item(),0.)
        self.assertTrue(torch.equal(a,controller.update(raw,torch.tensor([.2,.2]),zeros,stairs,1)))
        controller.reset(torch.tensor([0]))
        b = controller.update(raw,torch.tensor([.2,.2]),zeros,stairs,1)
        self.assertEqual(b[0,2].item(),0.)
        self.assertLess(b[1,2].item(),0.)
        grid = grid_points('cpu')
        hits = torch.cat((grid,torch.zeros(187,1)),-1)[None]
        self.assertFalse(stair_scan_mask(hits,grid)[0])
        hits[0,grid[:,0]>.2,2] = .15
        self.assertTrue(stair_scan_mask(hits,grid)[0])

    def test_feedback_has_only_delivered_sensor_information(self):
        controller = SensorHeadingFeedback(1,'cpu',self.manifest['stair_tuning'])
        raw = torch.tensor([[.5,0.,0.]])
        zero = torch.zeros(1)
        height = torch.zeros(1,187)
        missing = (height,torch.zeros_like(height),torch.zeros_like(height))
        controller(raw,zero,zero,missing,0)
        # Missing/occluded map cannot reveal a staircase through command slots.
        flat = controller(raw,torch.tensor([.2]),zero,missing,1)
        self.assertAlmostEqual(flat[0,2].item(),-.16,places=6)
        height[:,grid_points('cpu')[:,0]>.2] = -.15
        delivered = (height,torch.ones_like(height),torch.ones_like(height))
        stairs = controller(raw,torch.tensor([.2]),zero,delivered,2)
        self.assertAlmostEqual(stairs[0,2].item(),-.32,places=6)

    def test_training_and_deployment_sensor_feedback_with_noise_and_reset(self):
        from m20_student_v1 import v3_5_environment as glue
        from m20_student_v1 import environment as common
        from m20_student_v1.perception import quaternion_rotation
        provenance = {'perception':json.loads((ROOT/'configs/distillation/student_v3_5.json').read_text())['perception'],
            'contract':observation_contract(4), 'task_version':'v3.5',
            'command_feedback':{'version':'legacy_v1_v3_5','tuning':self.manifest['stair_tuning']}}
        cfg, extras = profile_config('student-v1',provenance)
        p = torch.zeros(1,57);p[:,5] = -1.
        grid = grid_points('cpu')
        hits = torch.cat((grid,torch.zeros(187,1)),-1)[None]
        hits[:,grid[:,0]>.2,2] = .15
        robot = SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.tensor([[0.,0.,.5]]),root_quat_w=torch.tensor([[1.,0.,0.,0.]])))
        sensor = SimpleNamespace(data=SimpleNamespace(ray_hits_w=hits))
        class Scene(dict):
            sensors = {'height_scanner':sensor}
        scene = Scene(robot=robot)
        spans = dict(base_ang_vel=(0,3),projected_gravity=(3,6),velocity_commands=(6,9),joint_pos=(9,25),joint_vel=(25,41),actions=(41,57))
        terms = {}
        for name,(a,b) in spans.items():
            terms[name] = SimpleNamespace(func=lambda env,a=a,b=b:p[:,a:b],params={},clip=None,scale=None,modifiers=None,history_length=0)
        terms['velocity_commands'].func = glue.effective_velocity_commands
        env = SimpleNamespace(num_envs=1,device='cpu',step_dt=.02,scene=scene,common_step_counter=0,
            cfg=SimpleNamespace(student_perception=cfg,student_history_steps=4,motion_teacher_tuning=self.manifest['stair_tuning'],observations=SimpleNamespace(teacher=SimpleNamespace(**terms))),
            command_manager=SimpleNamespace(get_command=lambda _:p[:,6:9]))
        contract = SimpleNamespace(base_id=0,last_scan_hits=hits[0].numpy())
        def observe(data,command,previous):
            clean=p[0].numpy().copy();clean[6:9]=command;clean[41:57]=previous
            return np.concatenate((clean,np.zeros(187,dtype=np.float32)))
        contract.observe=observe
        adapter = StudentMapAdapter(contract,cfg,extras)
        for step in range(40):
            if step == 23:
                common.sensor_state(env).reset(torch.tensor([0]));glue.reset_feedback(env,torch.tensor([0]));adapter.reset()
            env.common_step_counter=step
            angle=.005*step
            robot.data.root_quat_w=torch.tensor([[np.cos(angle/2),0.,0.,np.sin(angle/2)]],dtype=torch.float32)
            rotation=quaternion_rotation(robot.data.root_quat_w)
            data=SimpleNamespace(xpos=robot.data.root_pos_w.numpy(),xmat=rotation.numpy().reshape(1,9))
            p[:,6:9]=torch.tensor([.5 if step < 30 else 0.,0.,0.])
            p[:,41:57]=.001*step
            rng=torch.get_rng_state()
            expected=glue.student_observation(env)
            after=torch.get_rng_state();torch.set_rng_state(rng)
            actual=adapter.observe(data,p[0,6:9].numpy(),p[0,41:57].numpy(),step)
            torch.set_rng_state(after)
            np.testing.assert_allclose(actual,expected[0].numpy(),atol=2e-6,rtol=1e-6)
            self.assertTrue(torch.equal(glue.effective_velocity_commands(env),expected[:,6:9]))
            self.assertTrue(torch.equal(glue.raw_motion_control(env),p[:,6:9]))


if __name__ == '__main__':
    unittest.main()
