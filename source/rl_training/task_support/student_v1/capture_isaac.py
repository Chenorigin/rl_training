"""Measured source-map rays; the frozen teacher/output sensor remains unchanged."""
from copy import deepcopy
from .contract import GRID_RESOLUTION
from .perception import capture_grid_shape


def configure_capture(cfg):
    padding = cfg.student_perception.map_capture_padding_m
    surface = cfg.student_perception.support_surface_visibility
    if not padding and not surface:
        cfg.scene.height_scanner_capture = None
        return
    rows,cols = capture_grid_shape(padding)
    sensor = deepcopy(cfg.scene.height_scanner)
    if surface:
        from .surface_ray_caster import SurfaceRayCaster
        sensor.class_type = SurfaceRayCaster
    sensor.pattern_cfg.size = ((cols-1)*GRID_RESOLUTION,(rows-1)*GRID_RESOLUTION)
    sensor.pattern_cfg.ordering = 'xy'
    sensor.debug_vis = False
    cfg.scene.height_scanner_capture = sensor
