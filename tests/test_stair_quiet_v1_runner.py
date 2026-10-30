"""Real checkpoint compatibility and resume failure cases, without optimization."""
from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/rl_training"))
sys.path.insert(0, str(ROOT / "tests"))
from test_idle_stop_motion_v1 import ObservationEnv, train_cfg
from task_support.stair_quiet_v1.baselines import TEACHER, TEACHER_SHA256, checked_path
from task_support.stair_quiet_v1.runner import StairQuietRunner, canonical
from task_support.student_v3_5.v3_5_teacher import MotionTeacherModel, teacher_config, validate_manifest

torch.set_num_threads(1)


def make_runner():
    cfg = train_cfg()
    cfg.update(descent_guide_iters=0, descent_guide_decay_iters=0)
    cfg["algorithm"].update(learning_rate=3.e-5, anchor_coefficient=.5,
                            anchor_stair_weight=.1, anchor_stop_weight=1.,
                            motion_regularization_coefficient=0.)
    env = ObservationEnv()
    with redirect_stdout(StringIO()):
        return StairQuietRunner(env, cfg, device="cpu")


def test_complete_teacher_and_critic_warm_start_freeze_and_zero_command():
    runner = make_runner()
    runner.initialize_task({})
    saved = torch.load(checked_path(TEACHER, TEACHER_SHA256), weights_only=True, map_location="cpu")
    for kind in ("actor", "critic"):
        assert all(torch.equal(v, getattr(runner.alg, kind).state_dict()[k])
                   for k, v in saved[kind + "_state_dict"].items())
    assert runner.current_learning_iteration == runner.alg.guard_updates == 0
    assert not runner.alg.optimizer.state
    assert not any(p.requires_grad for p in runner.alg.actor.mlp.parameters())
    assert not any(p.requires_grad for p in runner.alg.actor.distribution.parameters())
    assert all(p.requires_grad for p in runner.alg.actor.residual_mlp.parameters())
    assert not any(p.requires_grad for p in runner.alg.reference_actor.parameters())
    obs = runner.env.obs
    with torch.no_grad():
        before = runner.alg.actor(obs).clone()
        assert torch.equal(before, runner.alg.reference_actor(obs))
        # A deliberate residual change must affect moving outputs but never raw-zero outputs.
        runner.alg.actor.residual_mlp[-1].bias.add_(1.)
        after = runner.alg.actor(obs)
    assert torch.equal(before[0], after[0])
    assert not torch.equal(before[1:], after[1:])
    runner.alg.actor(obs).square().mean().backward()
    assert runner.alg.actor.residual_mlp[-1].weight.grad is not None
    assert all(p.grad is None for p in runner.alg.actor.mlp.parameters())
    guide = runner.env._m20_descent_entry_guide_schedule
    assert guide["guided_iterations"] == guide["decay_iterations"] == 0


def test_task_manifest_loads_complete_distillation_teacher():
    runner = make_runner()
    runner.initialize_task({})
    manifest = runner._record_manifest()
    validate_manifest(manifest)
    cfg = teacher_config(manifest)
    cfg.pop("class_name")
    obs = runner.env.obs
    packed = TensorDict({"teacher": obs["policy"], "motion_control": obs["motion_control"]}, [4])
    with redirect_stdout(StringIO()):
        teacher = MotionTeacherModel(packed, {"teacher": ["teacher"]}, "teacher", 16, **cfg)
    teacher.load_state_dict(runner.alg.actor.state_dict(), strict=True)
    with torch.no_grad():
        assert torch.equal(teacher(packed), runner.alg.actor(obs))
    assert manifest["noise_refinement"]["acoustic_status"] == "not_measured"
    assert manifest["noise_refinement"]["teacher_baseline"]["sha256"] == TEACHER_SHA256


def test_quiet_schedule_anchors_resume_iteration_to_current_environment_step():
    runner = make_runner()
    runner.env.common_step_counter = 123
    runner.initialize_task({})
    schedule = runner.env._m20_stair_quiet_schedule
    assert schedule["iteration_offset"] == 0  # Never the source teacher's 46400.
    assert schedule["step_offset"] == 123
    assert schedule["steps_per_iteration"] == runner.cfg["num_steps_per_env"]
    runner.current_learning_iteration = 6
    runner.alg.guard_updates = 7
    runner.env.common_step_counter = 456
    runner._install_quiet_schedule()
    assert runner.env._m20_stair_quiet_schedule == {
        "iteration_offset": 7, "step_offset": 456,
        "steps_per_iteration": runner.cfg["num_steps_per_env"],
    }


def test_full_resume_and_incompatible_contract_rejection(tmp_path):
    original = make_runner()
    original.initialize_task({})
    original._record_manifest()
    original.alg.guard_updates = 7
    saved = original.alg.save()
    saved["iter"] = 6  # Actual RSL semantics: iter is last zero-based update.
    target = tmp_path / "candidate.pt"
    torch.save(saved, target)
    resumed = make_runner()
    resumed.load(target)
    resumed.initialize_task({"resume": True})
    assert resumed.current_learning_iteration == resumed.alg.guard_updates == 7
    assert resumed.env._m20_stair_quiet_schedule["iteration_offset"] == 7
    malformed = deepcopy(saved)
    malformed["iter"] = 7
    invalid = tmp_path / "bad_clock.pt"
    torch.save(malformed, invalid)
    with pytest.raises(ValueError, match="counters disagree"):
        make_runner().load(invalid)
    for key, value in original.alg.reference_actor.state_dict().items():
        assert torch.equal(resumed.alg.reference_actor.state_dict()[key], value)
    bad = make_runner()
    bad.env.cfg.rewards = SimpleNamespace(new_objective=SimpleNamespace(weight=-3.))
    with pytest.raises(ValueError, match="contract changed"):
        bad.load(target)
    saved.pop("preservation_reference_actor")
    torch.save(saved, target)
    with pytest.raises(ValueError, match="reference actor"):
        make_runner().load(target)
    with pytest.raises(ValueError, match="not as a StairQuiet resume"):
        make_runner().load(ROOT / TEACHER)


def test_resume_rejects_changed_reference_frozen_weights_and_source(tmp_path):
    original = make_runner()
    original.initialize_task({})
    original._record_manifest()
    saved = original.alg.save()
    saved["iter"] = 0
    target = tmp_path / "candidate.pt"
    altered = deepcopy(saved)
    altered["preservation_reference_actor"]["residual_mlp.0.weight"].add_(1.)
    torch.save(altered, target)
    with pytest.raises(ValueError, match="reference differs"):
        make_runner().load(target)
    altered = deepcopy(saved)
    altered["actor_state_dict"]["mlp.0.weight"].add_(1.)
    torch.save(altered, target)
    with pytest.raises(ValueError, match="frozen baseline"):
        make_runner().load(target)
    altered = deepcopy(saved)
    altered["m20_training_manifest"]["noise_refinement"]["source_sha256"] = {}
    torch.save(altered, target)
    with pytest.raises(ValueError, match="source changed"):
        make_runner().load(target)


def test_wrong_source_and_ambiguous_initialization_rejected(tmp_path):
    wrong = tmp_path / "wrong.pt"
    wrong.write_bytes(b"not the pinned baseline")
    with pytest.raises(ValueError, match="SHA256"):
        make_runner().initialize_task({"init_actor_from": str(wrong)})
    with pytest.raises(ValueError, match="pinned full teacher"):
        make_runner().initialize_task({"init_critic_from": str(wrong)})
    with pytest.raises(RuntimeError, match="complete checkpoint"):
        make_runner().initialize_task({"resume": True})


def test_config_function_identity_is_stable():
    assert canonical({"fn": make_runner}) == {"fn": f"{__name__}:make_runner"}
