"""CPU behavioral checks; no Isaac startup, formal training or robot access."""

from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import unittest

import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.runners import DistillationRunner

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/rl_training"))
from m20_student_v0.perception import StudentV0PerceptionCfg, SyntheticMapState, confidence_from_quality
from m20_student_v0.distillation import StudentV0Distillation
from m20_student_v0.configuration import DeeproboticsM20StudentV0DistillationCfg

helper_path = ROOT/"source/rl_training/task_support/stair_student/perception.py"
spec = importlib.util.spec_from_file_location("stair_student_helpers", helper_path)
helpers = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = helpers
spec.loader.exec_module(helpers)
torch.set_num_threads(1)


def fixture(n=2):
    x = torch.arange(-8, 9)*.1
    y = torch.arange(-5, 6)*.1
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    hits = torch.stack((xx, yy, .2*xx), -1).reshape(1,187,3).repeat(n,1,1)
    base = torch.tensor([[0.,0.,.5]]).repeat(n,1)
    yaw = torch.zeros(n)
    return hits, base, yaw


def quiet_cfg(**kwargs):
    options = dict(level=1., height_noise_std=0., height_bias_std=0., xy_bias_std=0.,
                   yaw_bias_std=0., point_dropout=0., occlusion_probability=0.,
                   packet_dropout=0., max_delay_s=0., map_update_period_s=1.,
                   variance_growth_per_s=0., max_map_age_s=2.)
    options.update(kwargs)
    return StudentV0PerceptionCfg(**options)


class MapTests(unittest.TestCase):
    def test_clean_and_nonfinite_rays(self):
        hits, base, yaw = fixture()
        hits[1,10,2] = torch.nan
        state = SyntheticMapState(2,"cpu",.02,StudentV0PerceptionCfg(level=0))
        h,m,c = state.snapshot(hits,base,yaw,0)
        self.assertTrue(torch.allclose(h[0], -hits[0,:,2]))
        self.assertTrue(torch.equal(m[0],torch.ones(187)))
        self.assertTrue(torch.equal(c[0],torch.ones(187)))
        self.assertEqual((h[1,10].item(),m[1,10].item(),c[1,10].item()), (0.,0.,0.))
        self.assertTrue(all(torch.isfinite(v).all() for v in (h,m,c)))

    def test_reprojection_translation_and_vertical_motion(self):
        hits, base, yaw = fixture()
        state = SyntheticMapState(2,"cpu",.02,quiet_cfg())
        original = state.snapshot(hits,base,yaw,0)[0]
        moved = base.clone()
        moved[:,0] += .1
        moved[:,2] += .1
        h,m,_ = state.snapshot(hits,moved,yaw,1)
        self.assertTrue(torch.allclose(h[0].reshape(11,17)[:,:-1], original[0].reshape(11,17)[:,1:]+.1,atol=1e-6))
        self.assertTrue((m.reshape(2,11,17)[:,:,-1]==0).all())
        # A naive stale ego vector would fail this known-positive expectation.
        self.assertFalse(torch.allclose(h[0],original[0]))

    def test_yaw_reprojection(self):
        hits, base, yaw = fixture()
        state = SyntheticMapState(2,"cpu",.02,quiet_cfg())
        state.snapshot(hits,base,yaw,0)
        rotated = torch.full((2,),torch.pi/2)
        h,m,_ = state.snapshot(hits,base,rotated,1)
        expected = .2*hits[0,:,1]
        self.assertTrue(torch.allclose(h[0][m[0]>0],expected[m[0]>0],atol=1e-6))
        self.assertTrue((m==0).any())

    def test_cache_and_partial_reset_do_not_leak_old_episode(self):
        hits,base,yaw = fixture()
        state = SyntheticMapState(2,"cpu",.02,quiet_cfg())
        a = state.snapshot(hits,base,yaw,0)
        a[0].fill_(999)
        self.assertLess(state.snapshot(hits,base,yaw,0)[0].abs().max(),1)
        changed = hits.clone()
        changed[0,:,2] += .2
        state.reset(torch.tensor([0]))
        b = state.snapshot(changed,base,yaw,0)
        self.assertTrue(torch.allclose(b[0][0],-changed[0,:,2]))
        self.assertTrue(torch.allclose(b[0][1],-hits[1,:,2]))

    def test_dropout_expiry_delay_and_stale_packet_order(self):
        hits,base,yaw = fixture()
        dropped = SyntheticMapState(2,"cpu",.02,quiet_cfg(packet_dropout=1))
        self.assertTrue((dropped.snapshot(hits,base,yaw,0)[1]==0).all())
        state = SyntheticMapState(2,"cpu",.02,quiet_cfg(max_map_age_s=.03))
        state.snapshot(hits,base,yaw,0)
        self.assertTrue((state.snapshot(hits,base,yaw,2)[1]==0).all())
        delayed = SyntheticMapState(2,"cpu",.02,quiet_cfg(max_delay_s=.04))
        delayed.snapshot(hits,base,yaw,0)
        self.assertTrue((delayed.snapshot(hits,base,yaw,2)[1]==1).all())
        # Inject a late older packet after a newer packet has already arrived.
        delayed.capture[:] = 2
        delayed.queue_capture[0] = 0
        delayed.queue_arrival[0] = 3
        delayed.snapshot(hits,base,yaw,3)
        self.assertTrue((delayed.capture==2).all())

    def test_quality_and_contiguous_occlusion(self):
        cfg = quiet_cfg(occlusion_probability=1.,occlusion_width_m=.8,occlusion_height_m=.6)
        hits,base,yaw = fixture()
        torch.manual_seed(42)
        state = SyntheticMapState(2,"cpu",.02,cfg)
        _,m,_ = state.snapshot(hits,base,yaw,0)
        invalid = (m[0]==0).reshape(11,17)
        self.assertTrue((invalid[:,1:]&invalid[:,:-1]).any())
        score = confidence_from_quality(torch.ones(4),torch.tensor([0.,.01,torch.nan,0.]),
                                        torch.tensor([0.,0.,0.,3.]),cfg)
        self.assertGreater(score[0],score[1])
        self.assertEqual(score[2].item(),0)
        self.assertEqual(score[3].item(),0)
        with self.assertRaises(ValueError):
            StudentV0PerceptionCfg(level=float("nan")).validate()

    def test_configured_persistent_bias_contributes_to_quality(self):
        hits,base,yaw = fixture()
        state = SyntheticMapState(2,"cpu",.02,quiet_cfg(height_bias_std=.1))
        _,m,c = state.snapshot(hits,base,yaw,0)
        self.assertTrue((m==1).all())
        self.assertTrue(torch.allclose(state.variance,torch.full_like(state.variance,.01)))
        self.assertTrue(torch.allclose(c,torch.full_like(c,.2)))


def make_algorithm(ratio=0.0):
    torch.manual_seed(42)
    teacher_obs = torch.randn(4,244)*.1
    obs = TensorDict({"policy":torch.cat((teacher_obs,torch.ones(4,374)),1),"teacher":teacher_obs},batch_size=[4])
    model = dict(hidden_dims=[512,256,128],activation="elu",obs_normalization=False,
                 distribution_cfg={"class_name":"GaussianDistribution","init_std":1.,"std_type":"log"})
    teacher = MLPModel(obs,{"teacher":["teacher"]},"teacher",16,**deepcopy(model))
    student = MLPModel(obs,{"student":["policy"]},"student",16,**deepcopy(model))
    helpers.initialize_student_actor(student,teacher.state_dict())
    storage = RolloutStorage("distillation",4,2,obs,[16],"cpu")
    alg = StudentV0Distillation(student,teacher,storage,teacher_ratio=ratio,batch_size=4,
                               replay_rollouts=2,learning_rate=1e-4,max_grad_norm=1.,num_learning_epochs=1)
    alg.teacher_loaded = True
    return alg,obs


class DistillationTests(unittest.TestCase):
    def test_native_runner_inference_collection_and_unit_update(self):
        # Tensor-only VecEnv: exercise the installed runner lifecycle without
        # Isaac, GPU, output files or a robot-policy training invocation.
        base_alg,obs = make_algorithm()
        obs["policy"][:,57:244] += .1
        obs["critic"] = obs["policy"].clone()

        class TensorEnv:
            cfg = {}
            num_envs, num_actions, device = 4,16,"cpu"

            def get_observations(self):
                return obs.clone()

            def step(self,action):
                self.assert_finite = bool(torch.isfinite(action).all())
                return self.get_observations(),torch.zeros(4),torch.zeros(4),{}

        cfg = DeeproboticsM20StudentV0DistillationCfg().to_dict()
        cfg["num_steps_per_env"] = 2
        cfg["algorithm"].update(batch_size=4,num_learning_epochs=1,replay_rollouts=2)
        env = TensorEnv()
        runner = DistillationRunner(env,cfg,log_dir=None,device="cpu")
        runner.alg.load({"actor_state_dict":base_alg.teacher.state_dict()},None,True)
        helpers.initialize_student_actor(runner.alg.student,base_alg.teacher.state_dict())
        frozen = {k:v.clone() for k,v in runner.alg.teacher.state_dict().items()}
        runner.learn(1)
        self.assertTrue(env.assert_finite)
        self.assertEqual(runner.alg.replay_size,8)
        self.assertEqual(runner.alg.num_updates,1)
        self.assertEqual(runner.alg.storage.step,0)
        self.assertTrue(torch.isfinite(runner.alg.student.output_std).all())
        self.assertTrue(all(torch.equal(v,runner.alg.teacher.state_dict()[k]) for k,v in frozen.items()))

    def test_warm_start_and_controller_selection(self):
        alg,obs = make_algorithm()
        self.assertTrue(torch.equal(alg.student(obs),alg.teacher(obs)))
        obs["policy"][:,57:244] += .1
        with torch.no_grad():
            action = alg.act(obs)
        self.assertTrue(torch.equal(action,alg.student(obs)))
        self.assertFalse(torch.allclose(action,alg.teacher(obs)))
        other,_ = make_algorithm(1)
        with torch.no_grad():
            self.assertTrue(torch.equal(other.act(obs),other.teacher(obs)))
        self.assertFalse(any(p.requires_grad for p in alg.teacher.parameters()))

    def test_labels_are_pre_step_and_replay_is_owned(self):
        alg,obs = make_algorithm()
        with torch.no_grad():
            label = alg.teacher(obs).clone()
            alg.act(obs)
            before = obs["policy"].clone()
            obs["policy"].add_(100)
            alg.process_env_step(obs,torch.zeros(4),torch.zeros(4),{})
        self.assertTrue(torch.equal(alg.replay_obs[:4],before))
        self.assertTrue(torch.equal(alg.replay_labels[:4],label))

    def test_unit_update_resume_and_export(self):
        alg,obs = make_algorithm()
        frozen = {k:v.clone() for k,v in alg.teacher.state_dict().items()}
        obs["policy"][:,57:244] += .1
        for _ in range(2):
            with torch.no_grad():
                alg.act(obs)
                alg.process_env_step(obs,torch.zeros(4),torch.zeros(4),{})
        stats = alg.update()  # Bounded CPU unit fixture; no simulated optimization run.
        self.assertEqual(alg.num_updates,1)
        self.assertTrue(all(torch.equal(v,alg.teacher.state_dict()[k]) for k,v in frozen.items()))
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in stats.values()))
        saved = alg.save()
        restored,_ = make_algorithm()
        restored.load(saved,None,True)
        self.assertTrue(torch.equal(restored.replay_obs,alg.replay_obs))
        self.assertTrue(torch.equal(restored.student(obs),alg.student(obs)))
        exported = torch.jit.script(restored.student.as_jit())
        self.assertTrue(torch.allclose(exported(obs["policy"]),restored.student(obs),atol=1e-6))
        bad = deepcopy(saved)
        bad["student_v0"]["replay_capacity"] += 1
        with self.assertRaises(ValueError):
            restored.load(bad,None,True)

    def test_missing_teacher_and_invalid_label_fail_closed(self):
        alg,obs = make_algorithm()
        alg.teacher_loaded = False
        with self.assertRaises(RuntimeError):
            alg.act(obs)
        alg.teacher_loaded = True
        obs["teacher"][0,0] = torch.nan
        with self.assertRaises(ValueError):
            alg.act(obs)


if __name__ == "__main__":
    unittest.main()
