"""Reward counterexamples and actual reward-to-actor gradient wiring, CPU only."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"source/rl_training"))
from task_support.student_v3_safe.configuration import default_settings
from task_support.student_v3_safe.gait_reward import DiagonalAlternation, diagonal_reward, eligible, step_evidence
from task_support.student_v3_safe.gait_policy_gradient import GaitStudentDistillation, delayed_returns, clipped_objective
from task_support.student_v1.configuration import DeeproboticsM20StudentV1DistillationCfg

torch.set_num_threads(1)
CFG = default_settings()["training"]["diagonal_reward"]
PG = default_settings()["training"]["algorithm"]["gait_policy_gradient"]
A, B = [True, False, False, True], [False, True, True, False]


def sequence(patterns, *, moving=True, active=True, safe=True):
    tracker = DiagonalAlternation(1, "cpu", CFG)
    tracker.reset(slice(None), 0.)
    values = []
    for tick, pattern in enumerate(patterns):
        contact = torch.tensor([pattern])
        values.append(tracker.step(tick, .02, torch.tensor([.01*(tick+1) if moving else 0.]),
            contact, ~contact, torch.tensor([active]), torch.tensor([safe])).item())
    return values, tracker


def fixture(reward_scale=1., teacher_only=False):
    torch.manual_seed(673)
    obs = TensorDict({"policy": torch.randn(4, 789)*.05, "teacher": torch.randn(4, 244)*.05,
                      "motion_control": torch.tensor([[.7, 0., 0.]]*4)}, batch_size=[4])
    cfg = DeeproboticsM20StudentV1DistillationCfg().to_dict()
    models = []
    for key in ("student", "teacher"):
        c = deepcopy(cfg[key]); c.pop("class_name"); c["hidden_dims"] = [16, 8]
        models.append(MLPModel(obs, cfg["obs_groups"], key, 16, **c))
    pg = dict(PG, max_kl=.03, batch_size=32)
    alg = GaitStudentDistillation(*models, RolloutStorage("distillation", 4, 8, obs, [16], "cpu"),
        gait_policy_gradient=pg, teacher_ratio=1. if teacher_only else 0., teacher_ratio_end=0.,
        batch_size=32, num_learning_epochs=1, replay_rollouts=1, hard_replay_rollouts=1,
        stratified_fraction=0., hard_fraction=0.)
    alg.teacher_loaded = True
    alg.pg_provider = lambda: torch.ones(4, dtype=torch.bool)
    tags = {k:torch.zeros(4, dtype=torch.long) for k in ("phase", "terrain", "direction", "terrain_level")}
    tags["terminated"] = torch.zeros(4, dtype=torch.bool)
    with torch.inference_mode():
        for t in range(8):
            alg.act(obs)
            reward = torch.tensor([1., -1., 2., -2.])*(t+1)*reward_scale
            alg.process_env_step(obs, reward, torch.zeros(4), {"student_v1":tags})
    return alg


class RewardTests(unittest.TestCase):
    def test_command_course_gate_and_disabled_evidence(self):
        class Scene(dict):
            env_origins = torch.zeros(9, 3)
        position = torch.tensor([[1., 0., 0.]]*9)
        position[6, 0] = -5.
        position[7, 1] = 2.
        scene = Scene(robot=SimpleNamespace(data=SimpleNamespace(root_pos_w=position)))
        commands = torch.tensor([[.5,0.,0.],[-.5,0.,0.],[0.,.5,0.],
                                 [0.,0.,.5],[0.,0.,0.],[.5,0.,.5],
                                 [.5,0.,0.],[.5,0.,0.],[.5,0.,0.]])
        env = SimpleNamespace(scene=scene,num_envs=9,device='cpu',
            cfg=SimpleNamespace(student_diagonal_reward=dict(CFG),
                                stair_tail=SimpleNamespace(start_length=2.,body_half_length=.4)),
            command_manager=SimpleNamespace(get_command=lambda _:commands))
        state = dict(steps=torch.ones(9)*3,depth=torch.ones(9)*.3,
                     width=torch.ones(9),height=torch.ones(9)*.15)
        state['height'][8]=0.
        with patch('task_support.stair_quiet_v1_safe.course.course_state',return_value=state):
            self.assertEqual(eligible(env).tolist(),[True]+[False]*8)
        with self.assertRaisesRegex(RuntimeError,'did not execute'):
            step_evidence(env)
        env.cfg.student_diagonal_reward['weight']=0.
        self.assertTrue(all(not value.any() for value in step_evidence(env).values()))

    def test_progressing_alternation_positive_all_static_and_wrong_pairs_zero(self):
        values, _ = sequence([A]*4+[B]*4+[A]*4)
        self.assertEqual(sum(values[:7]), 0.)
        self.assertGreater(sum(values[7:]), 0.)
        for patterns in ([A]*60, [[True]*4]*12, [[False]*4]*12,
                         [[True,False,True,False]]*4+[[False,True,False,True]]*4,
                         [[True,True,False,False]]*4+[[False,False,True,True]]*4):
            self.assertEqual(sum(sequence(patterns)[0]), 0.)
        for kwargs in ({"moving":False}, {"active":False}, {"safe":False}):
            self.assertEqual(sum(sequence([A]*4+[B]*4+[A]*4, **kwargs)[0]), 0.)

    def test_no_banked_progress_and_gate_reset_does_not_reset_frontier(self):
        tracker = DiagonalAlternation(1, "cpu", CFG)
        tracker.reset(slice(None), 0.)
        for tick in range(12):
            contact = torch.tensor([A if tick < 4 else B if tick < 8 else A])
            score = tracker.step(tick, .02, torch.tensor([1.]), contact, ~contact,
                                 torch.tensor([tick >= 4]), torch.tensor([True]))
            self.assertEqual(score.item(), 0.)
        self.assertEqual(tracker.frontier.item(), 1.)

    def test_unsafe_transfer_scores_zero_without_erasing_previous_safe_phase(self):
        tracker = DiagonalAlternation(1, 'cpu', CFG)
        tracker.reset(slice(None), 0.)
        for tick in range(10):
            c = torch.tensor([A if tick < 4 else [True]*4 if tick < 6 else B])
            safe = torch.tensor([tick < 4 or tick >= 8])
            value = tracker.step(tick,.02,torch.tensor([tick*.01]),c,~c,
                                 torch.ones(1,dtype=torch.bool),safe)
            if not safe.item():
                self.assertEqual(value.item(),0.)
        self.assertGreater(value.item(),0.)

    def test_single_support_transfer_is_not_flight_and_cannot_score(self):
        values,_=sequence([A]*4+[[True,False,False,False]]*2+[B]*4)
        self.assertEqual(sum(values[4:6]),0.)
        self.assertGreater(values[-1],0.)

    def test_cache_partial_reset_and_flight_break_sequence(self):
        tracker = DiagonalAlternation(2, "cpu", CFG)
        tracker.reset(slice(None), 0.)
        for tick in range(8):
            c = torch.tensor([A if tick < 4 else B]*2)
            value = tracker.step(tick, .02, torch.tensor([tick*.01]*2), c, ~c,
                                 torch.ones(2,dtype=torch.bool), torch.ones(2,dtype=torch.bool))
        self.assertTrue((value > 0).all())
        age = tracker.age[1].clone()
        tracker.reset(torch.tensor([0]), torch.tensor([.07]))
        repeat = tracker.step(7, .02, torch.tensor([.07,.07]), c, ~c,
                              torch.ones(2,dtype=torch.bool), torch.ones(2,dtype=torch.bool))
        self.assertEqual(repeat[0], 0.)
        self.assertEqual(repeat[1], value[1])
        self.assertEqual(tracker.age[1], age)
        tracker.step(8,.02,torch.tensor([.08]*2), torch.zeros_like(c), torch.ones_like(c),
                     torch.ones(2,dtype=torch.bool), torch.ones(2,dtype=torch.bool))
        self.assertFalse(tracker.cycling.any())

    def test_real_reward_wrapper_midline_and_clearance_guards(self):
        env = SimpleNamespace(cfg=SimpleNamespace(student_diagonal_reward=CFG,stair_tail=object()),
            num_envs=1,device="cpu",step_dt=.02,common_step_counter=0,
            scene={"robot":SimpleNamespace(data=SimpleNamespace(root_quat_w=torch.tensor([[1.,0.,0.,0.]])))})
        d={"root":torch.zeros(1,3),"position":torch.tensor([[[.3,.2,.1],[.3,-.2,.1],[-.3,.2,.1],[-.3,-.2,.1]]]),
           "upright":torch.ones(1),"yaw":torch.zeros(1)}
        g={"supported":torch.tensor([A]),"forward":torch.ones(1,4),"side":torch.ones(1,4),"clearance":torch.ones(1,4)*.04}
        with patch('task_support.stair_quiet_v1_safe.observations.physical_sample',return_value=d), \
             patch('task_support.stair_quiet_v1_safe.observations.safety_geometry',return_value=g), \
             patch('task_support.student_v3_safe.gait_reward.eligible',return_value=torch.ones(1,dtype=torch.bool)):
            for t in range(12):
                env.common_step_counter=t;d['root'][0,0]=t*.01
                g['supported']=torch.tensor([A if t<4 or t>=8 else B])
                value=diagonal_reward(env)
            self.assertGreater(value.item(),0.)
            env.common_step_counter+=1;d['root'][0,0]+=.01;d['position'][0,0,1]=-.2
            self.assertEqual(diagonal_reward(env).item(),0.)


class PolicyGradientTests(unittest.TestCase):
    def test_delay_tail_teacher_exclusion_and_reset_credit(self):
        r=torch.tensor([[1.,100.],[2.,200.],[3.,300.]])
        d=torch.zeros(3,2,dtype=torch.bool);student=torch.tensor([[True,False]]*3)
        ret,adv,valid=delayed_returns(r,d,torch.ones_like(d),student,1,1.)
        torch.testing.assert_close(ret[:,0],torch.tensor([5.,3.,0.]))
        self.assertEqual(valid[:,0].tolist(),[True,True,False])
        self.assertFalse(valid[:,1].any());self.assertGreater(adv[0,0],0.)
        d[1,0]=True
        ret,_,valid=delayed_returns(r,d,torch.ones_like(d),student,1,1.)
        self.assertEqual(ret[0,0],2.);self.assertFalse(valid[1,0])
        # Later loss of PG eligibility must still carry a terminal penalty.
        d.zero_();r[2,0]=-100.;sample=torch.zeros_like(d);sample[0,0]=True
        ret,_,_=delayed_returns(r,d,sample,student,1,1.)
        self.assertEqual(ret[0,0],-98.)

    def test_reward_changes_actual_update_gradient_teacher_rewards_do_not(self):
        gradients=[]
        for teacher_only in (False,True):
            for scale in (1.,-1.):
                alg=fixture(scale,teacher_only)
                before={k:v.clone() for k,v in alg.student.state_dict().items()}
                teacher={k:v.clone() for k,v in alg.teacher.state_dict().items()}
                captured=[]
                # Execute full update/backprop wiring, intercept optimizer writes.
                alg.optimizer.step=lambda: captured.append(torch.cat([p.grad.flatten().clone() for p in alg.student.mlp.parameters()]))
                torch.manual_seed(82);stats=alg.update()
                gradients.append(captured[0])
                self.assertTrue(all(torch.equal(v,alg.student.state_dict()[k]) for k,v in before.items()))
                self.assertTrue(all(torch.equal(v,alg.teacher.state_dict()[k]) for k,v in teacher.items()))
                self.assertEqual(stats['gait_pg_samples']==0,teacher_only)
        self.assertFalse(torch.equal(gradients[0],gradients[1]))
        self.assertTrue(torch.equal(gradients[2],gradients[3]))

    def test_original_issued_density_and_snapshot_not_current_teacher_mask(self):
        alg=fixture();original=alg.pg_student.clone();alg.teacher_mask=torch.ones_like(alg.teacher_mask)
        alg.prepare_policy_gradient();self.assertGreater(len(alg.pg_ids),0)
        self.assertTrue(torch.equal(original,alg.pg_student))
        t,e=alg.pg_ids.unbind(-1)
        self.assertGreater((alg.pg_action[t,e]-alg.pg_mean[t,e]).abs().sum(),0.)
        # Reward data in older replay buffers never participates.
        base=alg.pg_advantages.clone();alg.replay_labels.fill_(1.e5);alg.prepare_policy_gradient()
        self.assertTrue(torch.equal(base,alg.pg_advantages))

    def test_clipping_gradient_guard_and_resume_contract(self):
        mean=torch.zeros(2,16,requires_grad=True);old=torch.zeros_like(mean)
        loss,kl,_=clipped_objective(mean,old,torch.ones_like(mean)*.04,torch.ones(16)*.04,
                                    torch.tensor([1.,2.]),.2)
        loss.backward();self.assertGreater(mean.grad.abs().sum(),0.);self.assertEqual(kl.item(),0.)
        alg=fixture();alg.prepare_policy_gradient()
        with torch.no_grad():
            list(alg.student.mlp.parameters())[-1].add_(100.)
        loss,_=alg.auxiliary_loss();self.assertEqual(loss.item(),0.);self.assertTrue(alg.pg_stopped)
        saved=alg.save();restored=fixture();restored.load(saved)
        saved['student_gait_pg']['settings']['discount']=.5
        with self.assertRaisesRegex(ValueError,'resume configuration'):
            restored.load(saved)


if __name__=='__main__':
    unittest.main()
