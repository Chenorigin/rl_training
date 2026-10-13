"""Small CPU fixtures exercising real TensorBoard files and failed disk writes."""

from collections import namedtuple
from copy import deepcopy
import errno
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from tensordict import TensorDict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"source/rl_training"))
from m20_student_v0.configuration import DeeproboticsM20StudentV0DistillationCfg
from m20_student_v0.disk_runner import StorageDistillationRunner
from m20_student_v0.output_storage import GIB, LowDiskSpace, atomic_torch_save, output_budget, require_disk_space

torch.set_num_threads(1)
Usage = namedtuple("Usage","total used free")


def runner_fixture(directory):
    cfg = DeeproboticsM20StudentV0DistillationCfg().to_dict()
    cfg["num_steps_per_env"] = 2
    cfg["algorithm"].update(batch_size=4,num_learning_epochs=1,replay_rollouts=2)
    teacher_obs = torch.zeros(4,244)
    obs = TensorDict({"policy":torch.cat((teacher_obs,torch.ones(4,374)),1),
                      "teacher":teacher_obs},batch_size=[4])
    obs["critic"] = obs["policy"].clone()

    class TensorEnv:
        cfg = {}
        num_envs,num_actions,device = 4,16,"cpu"

        def get_observations(self):
            return obs.clone()

        def step(self,action):
            assert torch.isfinite(action).all()
            return obs.clone(),torch.zeros(4),torch.zeros(4),{"log":{"fixture/reward":torch.tensor(1.)}}

    runner = StorageDistillationRunner(TensorEnv(),deepcopy(cfg),log_dir=str(directory),device="cpu",
                                       min_free_disk_bytes=2*GIB,checkpoint_estimate_bytes=16*1024**2)
    runner.alg.teacher_loaded = True
    runner.logger.git_status_repos = []  # No unrelated repository snapshot in this unit fixture.
    return runner


class OutputTests(unittest.TestCase):
    def test_output_budget_covers_actual_full_replay_checkpoint_and_schedule(self):
        cfg=DeeproboticsM20StudentV0DistillationCfg().to_dict()
        fresh=output_budget(cfg,64,5000)
        resume=output_budget(cfg,64,4599,401)
        self.assertGreater(fresh["checkpoint_estimate_bytes"],27758531)
        self.assertEqual(fresh["checkpoint_count"],51)
        self.assertEqual(resume["checkpoint_count"],46)
        self.assertGreater(fresh["required_free_bytes"],2*GIB)

    def test_insufficient_disk_fails_closed_on_destination_volume(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("m20_student_v0.output_storage.shutil.disk_usage",return_value=Usage(10*GIB,9*GIB,GIB)) as probe:
                with self.assertRaises(LowDiskSpace):require_disk_space(Path(directory)/"new/run",2*GIB)
                self.assertEqual(probe.call_args.args[0],Path(directory))

    def test_atomic_write_failure_preserves_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            target=Path(directory)/"model.pt"
            atomic_torch_save({"value":torch.tensor(7)},target)
            original=target.read_bytes()
            def failed_save(state,stream):
                stream.write(b"partial")
                raise OSError(errno.ENOSPC,"No space left on device")
            with self.assertRaises(OSError):atomic_torch_save({},target,failed_save)
            self.assertEqual(target.read_bytes(),original)
            self.assertEqual(torch.load(target,weights_only=True)["value"].item(),7)
            self.assertFalse(list(Path(directory).glob("*.tmp")))

    def test_real_tensorboard_logging_and_atomic_native_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            runner=runner_fixture(directory)
            runner.learn(1)
            events=EventAccumulator(directory).Reload()
            self.assertIn("fixture/reward",events.Tags()["scalars"])
            self.assertEqual(len(events.Scalars("Loss/behavior")),1)
            checkpoint=torch.load(Path(directory)/"model_0.pt",map_location="cpu",weights_only=True)
            self.assertEqual(checkpoint["student_v0"]["num_updates"],1)
            self.assertEqual(checkpoint["student_v0"]["replay_size"],8)
            self.assertFalse(list(Path(directory).glob("*.tmp")))

    def test_runtime_low_space_saves_latest_update_before_log_write(self):
        with tempfile.TemporaryDirectory() as directory:
            runner=runner_fixture(directory)
            readings=iter((8*GIB,GIB,GIB))  # Start succeeds, log guard trips, recovery fits.
            with patch("m20_student_v0.output_storage.shutil.disk_usage",side_effect=lambda _:Usage(10*GIB,0,next(readings))):
                with self.assertRaisesRegex(LowDiskSpace,"Recovery checkpoint saved"):
                    runner.learn(2)
            checkpoint=torch.load(Path(directory)/"model_disk_stop_1.pt",map_location="cpu",weights_only=True)
            self.assertEqual(checkpoint["student_v0"]["num_updates"],1)
            self.assertEqual(runner.alg.num_updates,1)
            self.assertFalse((Path(directory)/"model_0.pt").exists())
            self.assertFalse(list(Path(directory).glob("*.tmp")))

    def test_failed_or_interrupted_run_closes_tensorboard_worker(self):
        for error_type in (ValueError,RuntimeError,KeyboardInterrupt):
            with self.subTest(error=error_type),tempfile.TemporaryDirectory() as directory:
                runner=runner_fixture(directory)
                def failed_log(*args,**kwargs):raise error_type("injected error")
                runner._native_log=failed_log
                with self.assertRaises(error_type):runner.learn(1)
                self.assertIsNone(runner.logger.writer.file_writer)


if __name__ == "__main__":unittest.main()
