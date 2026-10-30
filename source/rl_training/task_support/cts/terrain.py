"""CTS-owned geometry overrides; pretrained teacher/student sources stay frozen."""
from dataclasses import replace

from task_support.student_v3_safe.terrain_geometry import geometries as source_geometries


def validate_legacy(cfg):
    if not isinstance(cfg,dict) or set(cfg) != {'steps'} or type(cfg['steps']) is not int or cfg['steps'] < 1:
        raise ValueError('CTS legacy stairs require a positive integer step count')
    return cfg


def geometries(course,legacy):
    steps = validate_legacy(legacy)['steps']
    return tuple(replace(g,steps=steps) if g.family == 'legacy' else g for g in source_geometries(course))
