"""Tests for aiflow_nodes.CrackLoopNodes (staging, 2026-11-17); no LAMMPS runs."""

import numpy as np
import pytest
from ase.build import bulk
from ase.constraints import FixAtoms

from aiflow_nodes.controls_loops import ProceedWhileBelow
from aiflow_nodes.CrackLoopNodes import (
    CrackLoadStep,
    CrackStopDistance,
    CrackTipDistance,
    IncrementKField,
    MinimizationStats,
    StepDirectory,
)
from aiflow_nodes.CrackNodes import CutCylinder


def _cylinder(R_nm=2.0, a=3.165):
    block = bulk("W", "bcc", a=a, cubic=True).repeat((20, 20, 4))
    return CutCylinder._original_func(structure=block, radius_nm=R_nm)


def test_stop_distance_matches_wcrack_auto():
    assert CrackStopDistance._original_func(radius_nm=4.0, ring_width_A=12.0) == (
        20.0,
        15.0,
    )
    assert CrackStopDistance._original_func(radius_nm=40.0, ring_width_A=40.0) == (
        352.0,
        347.0,
    )
    assert CrackStopDistance._original_func(radius_nm=2.2, ring_width_A=4.0) == (
        10.0,
        5.0,
    )  # max(10, mobile - 8)
    with pytest.raises(ValueError, match="15 A"):
        CrackStopDistance._original_func(radius_nm=2.0, ring_width_A=12.0)


def _ringed(n=6):
    at = bulk("W", "bcc", a=3.165, cubic=True).repeat((4, 4, 2))
    at.pbc = (False, False, True)
    at.set_constraint(FixAtoms(indices=list(range(n))))
    return at


def test_increment_all_boundary_and_first_step():
    at = _ringed()
    U = np.full((len(at), 3), [0.1, 0.2, 0.0])
    s, K = IncrementKField._original_func(
        structure=at, U=U, K=1.7, K_prev=0.0, predictor="boundary"
    )
    assert K == 1.7
    np.testing.assert_allclose(
        s.positions - at.positions, 1.7 * U
    )  # first step: all atoms
    s2, _ = IncrementKField._original_func(
        structure=s, U=U, K=1.8, K_prev=1.7, predictor="boundary"
    )
    d = s2.positions - s.positions
    np.testing.assert_allclose(d[:6], 0.1 * U[:6], atol=1e-12)  # ring moved by dK U
    np.testing.assert_allclose(d[6:], 0.0, atol=1e-15)  # mobile atoms kept
    s3, _ = IncrementKField._original_func(
        structure=s, U=U, K=1.8, K_prev=1.7, predictor="all"
    )
    np.testing.assert_allclose(s3.positions - s.positions, 0.1 * U, atol=1e-12)
    assert sorted(int(i) for c in s3.constraints for i in c.get_indices()) == list(
        range(6)
    )


def test_increment_wraps_z_and_raises_outside_the_cell():
    at = _ringed()
    Lz = at.cell[2, 2]
    U = np.zeros((len(at), 3))
    U[:, 2] = Lz  # one full period along z
    s, _ = IncrementKField._original_func(structure=at, U=U, K=1.5, K_prev=0.0)
    assert (s.positions[:, 2] >= 0).all() and (s.positions[:, 2] < Lz).all()
    U = np.zeros((len(at), 3))
    U[:, 0] = -1e3
    with pytest.raises(RuntimeError, match="outside"):
        IncrementKField._original_func(structure=at, U=U, K=1.0, K_prev=0.0)


def test_step_directory():
    assert StepDirectory._original_func(run_root="/r", step=7) == ("/r/step_0007", 8)


class _Bundle:
    def __init__(self, wd):
        self.working_directory = wd


def test_minimization_stats(tmp_path):
    (tmp_path / "log.lammps").write_text(
        "Minimization stats:\n  Stopping criterion = force tolerance\n"
        "  Iterations, force evaluations = 300 312\n"
    )
    (tmp_path / "dump.out").write_text(
        "ITEM: TIMESTEP\n0\nx\nITEM: TIMESTEP\n100\nx\nITEM: TIMESTEP\n300\nx\n"
    )
    assert MinimizationStats._original_func(io_bundle=_Bundle(str(tmp_path))) == (
        "force tolerance",
        300,
    )
    (tmp_path / "dump.out").write_text("ITEM: TIMESTEP\n0\nITEM: TIMESTEP\n200\n")
    with pytest.raises(RuntimeError, match="last dump frame"):
        MinimizationStats._original_func(io_bundle=_Bundle(str(tmp_path)))


def test_tip_distance_on_a_perfect_cylinder():
    cyl = _cylinder(R_nm=2.0)
    # interior bcc atoms have 14 neighbours within 1.207 a0 (8 + 6), so only the outer surface counts
    tip, n = CrackTipDistance._original_func(structure=cyl, analysis_radius_A=15.0)
    assert (tip, n) == (0.0, 0)
    tip, n = CrackTipDistance._original_func(structure=cyl, analysis_radius_A=25.0)
    assert n > 0 and 17.0 < tip <= 20.0


def test_proceed_while_below_inclusive():
    f = ProceedWhileBelow._original_func
    assert f(value=15.0, threshold=15.0) is False
    assert f(value=15.0, threshold=15.0, inclusive=True) is True
    assert f(value=15.1, threshold=15.0, inclusive=True) is False


def test_group_builds_with_the_expected_ports():
    from aiflow.core import Workflow

    from aiflow_nodes.controls import Code

    at = _ringed()
    # U wired from a node, as in the workflow (a numpy array given directly to a group input
    # hits aiflow's `port._value == value` in groups.py; known, see CLAUDE.md)
    wf = Workflow("crack_load_step_ports")
    wf.U = Code(x=len(at), code="__import__('numpy').zeros((x, 3))")
    wf.Step = CrackLoadStep(
        structure=at, U=wf.U, potential="dummy", run_root="/tmp/never_used"
    )
    g = wf.Step
    assert {
        "structure",
        "U",
        "potential",
        "run_root",
        "K",
        "K_prev",
        "step",
        "predictor",
    } <= set(g.inputs.ports)
    assert {
        "structure",
        "K_used",
        "step_next",
        "energy",
        "iterations",
        "criterion",
        "fmax_free",
        "ring_motion",
        "tip_distance",
        "working_directory",
    } <= set(g.outputs.ports)
