"""The crossing instrument must separate deliberate clearance from rolling."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location('cts_clearance_eval',
    Path(__file__).resolve().parents[1]/'scripts/tools/evaluate_cts_stair_clearance.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_crossings_interpolate_underside_and_count_each_edge_once():
    position = np.zeros((5,4,3))
    position[...,0] = np.array([.8,1.,.8,1.,1.2])[:,None]
    position[...,2] = .09+.2+np.array([.06,.06,.0,.0])
    axis = np.zeros_like(position);axis[...,1] = 1.
    force = np.zeros_like(position)
    report = MODULE.summarize_crossings(position,axis,force,[(1.,.2)],.09,.027,.05)
    assert all(row['crossings'] == 1 for row in report.values())
    assert report['FL']['below_training_margin'] == 0
    assert report['HR']['below_training_margin'] == 1
    assert report['HR']['margin_min_m'] == pytest.approx(0.,abs=1.e-12)


def test_unvisited_edges_report_no_samples_not_a_success():
    position = np.zeros((2,4,3)); axis=np.zeros_like(position);force=np.zeros_like(position)
    report=MODULE.summarize_crossings(position,axis,force,[(1.,.2)],.09,.027,.05)
    assert all(row['crossings'] == 0 and row['margin_min_m'] is None for row in report.values())


def test_vertical_cylinder_and_capsule_have_distinct_exact_extents():
    position=np.zeros((2,4,3));position[...,0]=np.array([.8,1.])[:,None];position[...,2]=.3
    axis=np.zeros_like(position);axis[...,2]=1.
    force=np.zeros_like(position)
    cylinder=MODULE.summarize_crossings(position,axis,force,[(1.,.2)],.09,.027,.05)
    capsule=MODULE.summarize_crossings(position,axis,force,[(1.,.2)],.09,.027,.05,envelope='capsule')
    assert cylinder['FL']['margin_min_m'] == pytest.approx(.3-.027-.2)
    assert capsule['FL']['margin_min_m'] == pytest.approx(.3-.09-.027-.2)
