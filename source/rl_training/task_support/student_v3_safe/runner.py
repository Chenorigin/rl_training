"""Existing distillation runner with explicit Safe course save and restore."""
import torch

from m20_student_v1.runner import StudentV1Runner
from m20_student_v0.output_storage import atomic_torch_save
from .environment import course_snapshot, restore_course


class StudentV3SafeRunner(StudentV1Runner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from .gait_reward import eligible
        self.alg.pg_provider = lambda: eligible(self.env.unwrapped, ascent_only=False)

    def save(self, path, infos=None, *, recovery=False):
        self._check_space(recovery=recovery)
        state = self.alg.save()
        state.update(iter=self.current_learning_iteration, infos=infos,
                     student_v3_safe_course=course_snapshot(self.env.unwrapped))
        atomic_torch_save(state, path)
        # Inference validation never starts a logging writer. RSL's
        # save_model accesses writer unconditionally, so call it only in learn.
        if getattr(self.logger, "writer", None) is not None:
            self.logger.save_model(path, self.current_learning_iteration)

    def load(self, path, load_cfg=None, strict=True, map_location=None):
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if "student_v3_safe_course" not in saved:
            raise ValueError("student_v3_safe resume requires its completed-episode course state")
        result = super().load(path, load_cfg, strict, map_location)
        restore_course(self.env.unwrapped, saved["student_v3_safe_course"])
        return result
