"""Storage-safe runner plus shared sensor curriculum at rollout boundaries."""
from copy import deepcopy

from m20_student_v0.disk_runner import StorageDistillationRunner


class StudentV1Runner(StorageDistillationRunner):
    def _sync_sensor(self):
        env = self.env.unwrapped
        env.cfg.student_perception.level = self.alg.noise_level
        if self.alg.provenance:
            self.alg.provenance["perception"] = deepcopy(vars(env.cfg.student_perception))

    def learn(self, *args, **kwargs):
        self._sync_sensor()
        return super().learn(*args, **kwargs)

    def _guarded_log(self, *args, **kwargs):
        result = super()._guarded_log(*args, **kwargs)
        self._sync_sensor()
        return result
