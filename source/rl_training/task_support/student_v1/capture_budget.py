"""Extra live-map memory beyond the archived actor-sized acquisition budget."""
import math
from .configuration import StudentV1PerceptionCfg
from .perception import capture_grid_shape
from .contract import HEIGHT_DIM


def capture_memory_budget(perception, num_envs):
    cfg = StudentV1PerceptionCfg(**perception)
    cfg.validate()
    rows,cols = capture_grid_shape(cfg.map_capture_padding_m)
    points = rows*cols
    slots = math.ceil((cfg.max_map_age_s+cfg.max_delay_s)/max(.02,cfg.map_update_period_s-cfg.update_jitter_s))+3
    extra = num_envs*(points-HEIGHT_DIM)
    # z/valid/variance (9 bytes) plus float32 three-channel sampling stack (12).
    queues = extra*slots*21
    # Three position-sized visibility temporaries, four scalar work buffers per
    # line sample, plus measured-noise/quality intermediates (32 float buffers).
    visibility = extra*(cfg.visibility_samples*52+128)
    # Extra ray origins/directions/hits and ray result arrays, two copies.
    rays = num_envs*points*88 if cfg.map_capture_padding_m else 0
    return {'capture_grid_shape':[rows,cols],'capture_points':points,'actor_map_points':HEIGHT_DIM,
            'extra_capture_queue_stack_bytes':queues,'extra_capture_work_bytes':visibility,
            'extra_capture_ray_bytes':rays,'extra_capture_reserve_bytes':queues+visibility+rays}
