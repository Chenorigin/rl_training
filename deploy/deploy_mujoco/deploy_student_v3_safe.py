#!/usr/bin/env python3
"""Student v3 Safe MuJoCo playback with the mandatory shared landing controller.

Uses the existing Quiet viewer/headless lifecycle. Only the observation adapter
and actor loader change; privileged Safe teacher features are never constructed.
"""
import sys

import numpy as np
import torch

import deploy_quiet_landing as quiet
from student_v1_adapter import load_student, profile_config, StudentMapAdapter, sha256
from teacher_noise_adapter import TeacherNoiseAdapter
from task_support.student_v3_safe.configuration import TASK
from task_support.student_v3_safe.deployment import validate_provenance


class StudentActor(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, observation):
        return self.policy(observation["policy"])


def load_policy(checkpoint):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    provenance = saved.get("student_v1", {}).get("provenance", {})
    if provenance.get("task") != TASK:
        raise ValueError("This entry requires a student_v3_safe checkpoint")
    control = validate_provenance(provenance)
    policy, _, _ = load_student(checkpoint, execution_control=control)
    actor = StudentActor(policy).eval().requires_grad_(False)
    actor.student_provenance = provenance
    return actor, control


class StudentPerception(TeacherNoiseAdapter):
    def __init__(self, contract, args, provenance):
        self.contract, self.profile, self.seed = contract, args.noise_profile, args.seed
        self.cfg, extras = profile_config(self.profile, provenance, args.profile_file, args.noise_level)
        padding = getattr(args,'map_capture_padding',None)
        if padding is not None:
            self.cfg.map_capture_padding_m = padding
            self.cfg.validate()
        self.reference = {"path": str(args.checkpoint.resolve()), "sha256": sha256(args.checkpoint)}
        self.profile_file = ({"path": str(args.profile_file.resolve()), "sha256": sha256(args.profile_file)}
                             if args.profile_file else None)
        torch.manual_seed(self.seed)
        self.map_adapter = StudentMapAdapter(contract, self.cfg, extras, self.seed)
        self.steps = self.valid_samples = 0
        self.reset()

    def observe(self, data, command, previous_action, step):
        observation = self.map_adapter.observe(data, command, previous_action, step)
        self.steps += 1
        self.valid_samples += int(np.count_nonzero(self.map_adapter.last_channels[1]))
        if observation.shape != (789,) or not np.isfinite(observation).all():
            raise RuntimeError("Invalid Safe student sensor observation")
        return observation

    def summary(self):
        result = super().summary()
        result.pop("teacher_input_dim")
        result["student_input_dim"] = 789
        return result


def main():
    args = quiet.parse_args()
    if not any(arg == "--checkpoint" or arg.startswith("--checkpoint=") for arg in sys.argv[1:]):
        raise ValueError("Specify --checkpoint for the trained Safe student")
    if not any(arg == "--noise-profile" or arg.startswith("--noise-profile=") for arg in sys.argv[1:]):
        args.noise_profile = "student-v1"
    actor, _ = load_policy(args.checkpoint)
    factory = lambda args, contract: StudentPerception(contract, args, actor.student_provenance)
    # This is a separate process/entry. Existing teacher and student programs
    # keep their original module objects and adapters.
    quiet.load_policy = load_policy
    quiet.make_noise_adapter = factory
    quiet.baseline.make_noise_adapter = factory
    return quiet.main(args)


if __name__ == "__main__":
    main()
