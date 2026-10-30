"""Teacher preset imports, real sensor corruption and unchanged teacher ABI."""
from dataclasses import asdict
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deploy/deploy_mujoco"))
import deploy_mujoco as teacher
import deploy_quiet_landing as quiet
import teacher_noise_adapter as noise
from m20_student_v1.configuration import StudentV1PerceptionCfg
from m20_student_v1.perception import map_output_mask
from m20_student_v1.profiles import profile_config

torch.set_num_threads(1)


@pytest.fixture
def noise_args(tmp_path):
    cfg = StudentV1PerceptionCfg(level=1., clean_probability=0., stress_probability=0.,
                               map_update_period_s=.02, update_jitter_s=0., min_delay_s=0., max_delay_s=0.,
                               terrain_visibility=False, action_delay_steps=3)
    reference = tmp_path / "deployment.json"
    reference.write_text(json.dumps({"training_provenance": {"perception": asdict(cfg)}}))
    return SimpleNamespace(noise_profile="student-v1", noise_reference=reference,
                           profile_file=None, noise_level=None, seed=42)


def flat_contract():
    grid = teacher.scan_grid()
    hits = np.column_stack((grid, np.zeros(187)))
    contract = SimpleNamespace(grid=grid, last_scan_hits=hits, base_id=0)
    data = SimpleNamespace(xpos=np.array([[0., 0., .55]]), xmat=np.eye(3).reshape(1, 9))

    def observe(data, command, previous):
        clean = np.zeros(244, dtype=np.float32)
        clean[5], clean[6:9], clean[41:57], clean[57:] = -1., command, previous, .05
        return clean

    contract.observe = observe
    return contract, data


@pytest.mark.parametrize("entry", [teacher, quiet])
@pytest.mark.parametrize("name,expected", [("clean", "teacher"), ("realistic", "student-v0"),
                                          ("train", "student-v1"), ("front_only", "front_only"),
                                          ("front_only_30cm", "front_only_30cm"), ("stress", "stress")])
def test_teacher_clis_accept_existing_profiles_and_aliases(monkeypatch, entry, name, expected):
    monkeypatch.setattr(sys, "argv", ["runtime.py", "--model", "unused.xml", "--viewer", "--noise-profile", name])
    args = entry.parse_args()
    assert args.noise_profile == expected
    assert args.steps is None


@pytest.mark.parametrize("entry", [teacher, quiet])
@pytest.mark.parametrize("level", ["nan", "-1", "1.1"])
def test_invalid_level_is_not_silently_accepted(monkeypatch, entry, level):
    monkeypatch.setattr(sys, "argv", ["runtime.py", "--model", "unused.xml", "--noise-level", level])
    with pytest.raises(SystemExit):
        entry.parse_args()


@pytest.mark.parametrize("name", noise.V1_PROFILE_NAMES)
def test_every_existing_profile_uses_saved_values(noise_args, name):
    noise_args.noise_profile = name
    contract, _ = flat_contract()
    adapter = noise.TeacherNoiseAdapter(contract, noise_args)
    provenance, _ = noise.read_reference(noise_args.noise_reference)
    expected = profile_config(name, provenance)
    assert asdict(adapter.cfg) == asdict(expected)
    assert adapter.map_adapter.extras.command_feedback is None
    assert adapter.map_adapter.extras.history_steps == 1


def test_clean_default_keeps_exact_old_observations(noise_args):
    contract, _ = flat_contract()
    original = contract.observe
    noise_args.noise_profile = "teacher"
    noise_args.noise_reference = Path("missing.pt")
    assert noise.make_noise_adapter(noise_args, contract) is None
    assert contract.observe is original


def test_existing_json_overrides_and_reference_formats(noise_args, tmp_path):
    contract, _ = flat_contract()
    noise_args.noise_profile = "student-v0"
    noise_args.profile_file = tmp_path / "profiles.json"
    noise_args.profile_file.write_text(json.dumps({"realistic": {"height_noise_std": .123}}))
    adapter = noise.TeacherNoiseAdapter(contract, noise_args)
    assert adapter.cfg.height_noise_std == .123
    provenance, _ = noise.read_reference(noise_args.noise_reference)
    checkpoint = tmp_path / "reference.pt"
    torch.save({"student_v1": {"provenance": provenance}}, checkpoint)
    assert noise.read_reference(checkpoint)[0] == provenance
    noise_args.profile_file.write_text(json.dumps({"teacher": {}}))
    with pytest.raises(ValueError, match="Missing noise profile"):
        noise.TeacherNoiseAdapter(contract, noise_args)
    bad = tmp_path / "bad_reference.json"
    bad.write_text('{}')
    with pytest.raises(ValueError, match="saved v1"):
        noise.read_reference(bad)


def test_noisy_front_map_never_fills_missing_cells_with_truth(noise_args):
    contract, data = flat_contract()
    noise_args.noise_profile = "front_only"
    adapter = noise.TeacherNoiseAdapter(contract, noise_args)
    keep = map_output_mask(torch.as_tensor(contract.grid, dtype=torch.float32), adapter.cfg).numpy()
    previous = np.linspace(-.2, .2, 16, dtype=np.float32)
    rows = []
    for step in range(30):
        row = adapter.observe(data, np.array([.6, 0., 0.], dtype=np.float32), previous, step)
        assert row.shape == (244,)
        np.testing.assert_array_equal(row[41:57], previous)
        np.testing.assert_allclose(row[6:9], [.6, 0., 0.])
        assert not row[57:][~keep].any()
        rows.append(row)
    heights, valid, confidence = adapter.map_adapter.last_channels
    assert np.any(valid[keep] > 0)
    assert not np.allclose(np.array(rows)[:, :3], 0.)
    points = adapter.terrain_points(data)
    assert np.isnan(points[~keep]).all()
    assert (heights[valid == 0] == 0).all()
    assert (confidence[~keep] == 0).all()
    assert adapter.map_adapter.map.cell_age.shape == (1, 187)


def test_action_delay_and_handover_clear_old_commands(noise_args):
    contract, _ = flat_contract()
    adapter = noise.TeacherNoiseAdapter(contract, noise_args)
    values = [adapter.action(np.full(16, i, dtype=np.float32)) for i in range(1, 5)]
    assert [float(v[0]) for v in values] == [0., 0., 0., 1.]
    adapter.reset()
    np.testing.assert_array_equal(adapter.action(np.ones(16)), np.zeros(16))
    assert adapter.map_adapter.last_channels is None


def test_trot_gate_uses_noisy_missing_map_instead_of_clean_rays():
    from stair_trot_adapter import StairTrotMujocoController
    from task_support.stair_trot_v1.signals import TrotSettings
    grid = teacher.scan_grid()
    heights = (grid[:, 0] > .05).astype(float) * .15
    hits = np.column_stack((grid, heights))
    contract = SimpleNamespace(model=SimpleNamespace(nflex=0), terrain=SimpleNamespace(kind="flat"),
                               base_id=0, wheel_body_ids=np.arange(1, 5), grid=grid, last_scan_hits=hits)
    actor = SimpleNamespace(m20_training_manifest={"trot_settings": asdict(TrotSettings())})
    control = StairTrotMujocoController(contract, actor, .02)
    control.sample_contacts = lambda data: None
    control.forces[:, 2] = 100.
    wheels = np.array([[.3, .2, .24], [.3, -.2, .24], [-.3, .2, .09], [-.3, -.2, .09]])
    data = SimpleNamespace(time=0., xpos=np.vstack(([0, 0, .55], wheels)), xmat=np.tile(np.eye(3).reshape(1, 9), (5, 1)))
    assert control.control(data, [.6, 0, 0])[0] > 0
    data.time = .02
    assert not control.control(data, [.6, 0, 0], terrain_points=np.full_like(hits, np.nan)).any()
    data.time = .04
    assert control.control(data, [.6, 0, 0])[0] > 0


@pytest.mark.parametrize("entry", [teacher, quiet])
def test_clean_front_cli_keeps_manual_control(monkeypatch, entry):
    flags = ["runtime.py", "--model", "unused.xml", "--viewer", "--noise-profile", "front_only_clean"]
    monkeypatch.setattr(sys, "argv", flags)
    args = entry.parse_args()
    assert args.noise_profile == "front_only_clean"
    assert args.steps is None
    monkeypatch.setattr(sys, "argv", flags + ["--noise-level", "1"])
    with pytest.raises(SystemExit):
        entry.parse_args()


def test_clean_front_is_exact_spatial_ablation_and_seed_independent(noise_args):
    contract, data = flat_contract()
    # Exercise changing terrain/pose and arbitrary proprioception, rather than
    # only a flat map whose erased zero heights would be unobservable.
    clean = np.linspace(-.7, .8, 244, dtype=np.float32)
    contract.observe = lambda *args: clean.copy()
    noise_args.noise_profile = "front_only_clean"
    first = noise.make_noise_adapter(noise_args, contract)
    provenance, _ = noise.read_reference(noise_args.noise_reference)
    expected_mask = map_output_mask(torch.as_tensor(contract.grid, dtype=torch.float32),
                                    profile_config("front_only", provenance)).numpy()
    assert int(expected_mask.sum()) == 45
    for step in range(5):
        clean[:] += np.float32(.001)
        data.xpos[0] += [.01, -.01, .001]
        actual = first.observe(data, np.zeros(3), np.zeros(16), step)
        np.testing.assert_array_equal(actual[:57], clean[:57])
        np.testing.assert_array_equal(actual[57:][expected_mask], clean[57:][expected_mask])
        assert not actual[57:][~expected_mask].any()
        heights, valid, confidence = first.map_adapter.last_channels
        np.testing.assert_array_equal(valid, expected_mask.astype(np.float32))
        np.testing.assert_array_equal(confidence, valid)
        assert (first.map_adapter.map.cell_age[0, expected_mask] == 0).all()
        assert torch.isinf(first.map_adapter.map.cell_age[0, ~expected_mask]).all()
    noise_args.seed = 123
    second = noise.make_noise_adapter(noise_args, contract)
    np.testing.assert_array_equal(actual, second.observe(data, np.zeros(3), np.zeros(16), 99))
    nominal = np.linspace(-1., 1., 16, dtype=np.float32)
    np.testing.assert_array_equal(first.action(nominal), nominal)
    assert first.cfg.action_delay_steps == 0
    assert first.cfg.min_delay_s == first.cfg.max_delay_s == 0
    assert first.summary()["noise_added"] is False
    first.reset()
    assert first.map_adapter.last_channels is None


def test_clean_front_cannot_reintroduce_noise_through_profile_file(noise_args, tmp_path):
    contract, _ = flat_contract()
    noise_args.noise_profile = "front_only_clean"
    noise_args.profile_file = tmp_path / "front_clean.json"
    noise_args.profile_file.write_text(json.dumps({"front_only_clean": {"height_noise_std": .1}}))
    with pytest.raises(ValueError, match="overrides only"):
        noise.make_noise_adapter(noise_args, contract)
    noise_args.profile_file.write_text(json.dumps({"front_only_clean": {"lidar_horizontal_fov_deg": 90.}}))
    adapter = noise.make_noise_adapter(noise_args, contract)
    assert adapter.cfg.lidar_horizontal_fov_deg == 90.
