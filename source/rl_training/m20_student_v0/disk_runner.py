"""Native RSL runner with per-iteration disk checks and atomic checkpoint writes."""

import errno
from pathlib import Path

from rsl_rl.runners import DistillationRunner

from .output_storage import LowDiskSpace, atomic_torch_save, require_disk_space


class StorageDistillationRunner(DistillationRunner):
    def __init__(self,*args,min_free_disk_bytes,checkpoint_estimate_bytes,**kwargs):
        super().__init__(*args,**kwargs)
        self.min_free_disk_bytes = min_free_disk_bytes
        self.checkpoint_estimate_bytes = checkpoint_estimate_bytes
        self._native_log = self.logger.log
        self.logger.log = self._guarded_log

    def _check_space(self,recovery=False):
        if self.logger.log_dir is not None:
            reserve = 1024**2 if recovery else self.min_free_disk_bytes
            require_disk_space(self.logger.log_dir,reserve+self.checkpoint_estimate_bytes)

    def _guarded_log(self,*args,**kwargs):
        self._check_space()
        return self._native_log(*args,**kwargs)

    def save(self,path,infos=None,*,recovery=False):
        self._check_space(recovery=recovery)
        state = self.alg.save()
        state.update(iter=self.current_learning_iteration,infos=infos)
        atomic_torch_save(state,path)
        self.logger.save_model(path,self.current_learning_iteration)

    def _close_writer(self,*,suppress=False):
        writer = getattr(self.logger,"writer",None)
        if writer is None:
            return
        try:
            self.logger.stop_logging_writer()
            if self.logger.logger_type == "tensorboard":
                writer.close()
        except Exception as error:
            if not suppress:
                raise
            # Preserve the original storage failure during failed async flush.
            print(f"TensorBoard close failed: {error}",flush=True)

    def learn(self,*args,**kwargs):
        self._check_space()
        try:
            result = super().learn(*args,**kwargs)
            self._close_writer()
            return result
        except LowDiskSpace as error:
            path = Path(self.logger.log_dir)/f"model_disk_stop_{self.alg.num_updates}.pt"
            try:
                self.save(path,recovery=True)
                recovery = f"Recovery checkpoint saved: {path}. Resume it with --resume."
            except Exception as save_error:
                recovery = f"Recovery save failed: {save_error}. Resume a previous complete checkpoint."
            self._close_writer(suppress=True)
            raise LowDiskSpace(f"Training stopped before the next log write. {error}. {recovery}") from error
        except OSError as error:
            self._close_writer(suppress=True)
            if error.errno == errno.ENOSPC:
                raise LowDiskSpace(f"Output disk filled between checks. Existing checkpoints were not deleted; resume the latest complete checkpoint in {self.logger.log_dir} after freeing space.") from error
            raise
        except BaseException:
            # Includes interrupted runs and PyTorch serialization errors that
            # can wrap a filesystem failure as RuntimeError.
            self._close_writer(suppress=True)
            raise
