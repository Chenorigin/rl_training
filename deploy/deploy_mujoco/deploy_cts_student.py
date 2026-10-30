#!/usr/bin/env python3
"""CTS student MuJoCo playback with its decoder and mandatory landing control.

The deployed policy receives sensor789 and the raw requested command3. Teacher
privileged inputs and the training critic are never requested during rollout.
"""
from pathlib import Path
import sys

import torch

import deploy_quiet_landing as quiet
from deploy_student_v3_safe import StudentPerception
from task_support.cts.entry import validate_metadata
from task_support.cts.configuration import LEGACY_VERSION, algorithm_version
from task_support.cts.model import build_models
from task_support.stair_quiet_v1.deployment import execution_contract


class CTSStudentActor(torch.nn.Module):
    uses_motion_control = True

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, observation):
        return self.policy(observation["policy"], observation["motion_control"])


def load_policy(checkpoint):
    """Load evolved CTS weights without constructing rollout or optimizers."""
    saved = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    meta = validate_metadata(saved.get("cts", {}), resume=False)
    if meta.get("student_inputs") != {"policy": 789, "motion_control": 3}:
        raise ValueError("CTS deployment requires sensor789 and raw-command3 inputs")
    feedback = meta.get("command_feedback", {})
    if (feedback.get("version") != "legacy_v1_v3_5"
            or feedback.get("tuning") != meta.get("teacher_manifest", {}).get("stair_tuning")):
        raise ValueError("CTS deployment command feedback differs from the teacher contract")
    control = execution_contract(meta, required=True)
    initialization = saved["initialization"]
    algorithm = saved["algorithm"]
    expected_version = algorithm_version(meta.get('version', LEGACY_VERSION))
    if (algorithm.get("cts_version") != expected_version
            or algorithm.get("algorithm_config") != meta["config"]["algorithm"]
            or algorithm.get('auxiliary_contract') != meta.get('auxiliary_contract')):
        raise ValueError("CTS deployment model/configuration contract mismatch")
    state = algorithm["models_state_dict"]
    if not isinstance(state, dict) or not all(
            isinstance(value, torch.Tensor) and torch.isfinite(value).all() for value in state.values()):
        raise ValueError("Invalid or nonfinite CTS model state")
    all_terrain = meta["config"].get("all_terrain")
    models = build_models(initialization["teacher"], initialization["student"],
                          meta["config"]["algorithm"], device="cpu",
                          all_terrain_cfg=all_terrain if all_terrain and all_terrain.get('enabled') else None)
    models.load_state_dict(state, strict=True)
    expected_bounds = torch.tensor(meta["config"]["algorithm"]["decoder_bound"], dtype=torch.float32)
    if not torch.equal(models.decoder.bounds, expected_bounds):
        raise ValueError("CTS decoder bounds differ from the saved configuration")
    actor = CTSStudentActor(models.student_export()).eval().requires_grad_(False)
    actor.student_provenance = meta
    # Old checkpoints retain their weights/action ABI, but are now evaluated
    # against the same software gate as the real robot. No silent action shaping.
    from task_support.cts.configuration import default_settings
    actor.joint_limit_contract = meta['config'].get('joint_limits', default_settings()['joint_limits'])
    with torch.inference_mode():
        output = actor({"policy": torch.zeros(1, 789), "motion_control": torch.zeros(1, 3)})
    if output.shape != (1, 16) or not torch.isfinite(output).all():
        raise ValueError("Invalid CTS student policy output")
    return actor, control


def sensor_heading_controller(*_args, **_kwargs):
    """StudentPerception already applies the sensor-based heading feedback."""
    return None


def main():
    args = quiet.parse_args()
    if not any(arg == "--checkpoint" or arg.startswith("--checkpoint=") for arg in sys.argv[1:]):
        raise ValueError("Specify --checkpoint for the trained CTS student")
    if not any(arg == "--noise-profile" or arg.startswith("--noise-profile=") for arg in sys.argv[1:]):
        args.noise_profile = "student-v1"
    actor, _ = load_policy(args.checkpoint)
    if args.map_capture_padding is None:
        from task_support.cts.configuration import default_settings
        args.map_capture_padding = actor.student_provenance['perception'].get(
            'map_capture_padding_m',default_settings()['map_capture']['padding_m'])
    print(f'[CTS MAP] capture padding={args.map_capture_padding:g} m; output=187 cells; saved timing/noise retained',flush=True)
    factory = lambda args, contract: StudentPerception(contract, args, actor.student_provenance)
    # These hooks are scoped to this entry's process; legacy programs and their
    # source contracts remain intact. In particular uses_motion_control must
    # pack raw command3 without also enabling the teacher's heading controller.
    quiet.load_policy = load_policy
    quiet.make_noise_adapter = factory
    quiet.baseline.make_noise_adapter = factory
    quiet.baseline.heading_controller = sensor_heading_controller
    from task_support.cts.joint_limits_mujoco import PositionLimitError
    try:
        return quiet.main(args)
    except PositionLimitError as error:
        import json
        record = {**error.record, 'checkpoint': str(args.checkpoint.resolve()),
                  'software_joint_limits': actor.joint_limit_contract}
        if args.output is not None:
            path = args.output.with_suffix('.limit_failure.json')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
        raise


if __name__ == "__main__":
    main()
