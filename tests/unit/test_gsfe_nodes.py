"""Tests for GSFENodes: the GSFESlipPoint group's wiring, GSFEFit on a
synthetic gamma curve and PlotGSFE (no LAMMPS)."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from ase.build import bulk

from aiflow_nodes.GSFENodes import (
    _EV_PER_A2_IN_MJ_PER_M2,
    GSFEFit,
    GSFESlipPoint,
    PlotGSFE,
)

FRACTIONS = [0.0, 0.25, 0.5, 0.75, 1.0]
GAMMA = [0.0, 120.0, 180.0, 90.0, 40.0]  # mJ/m^2: USFE at 0.5, SSFE at 1.0
NY = 2
E0 = -1000.0


@pytest.fixture
def relaxed():
    # stands in for the (111) slab: only the y cell vector and the area matter
    return bulk("Cu", "fcc", a=3.6, orthorhombic=True).repeat((1, NY, 1))


def _df(relaxed, gamma=GAMMA, fractions=FRACTIONS, column="final"):
    area = np.linalg.norm(np.cross(relaxed.cell[0], relaxed.cell[1]))
    finals = [
        SimpleNamespace(
            energy=E0 + g * area / _EV_PER_A2_IN_MJ_PER_M2,
            structure=relaxed,
            force=np.zeros((len(relaxed), 3)),
        )
        for g in gamma
    ]
    if column == "out":
        finals = [SimpleNamespace(final=f) for f in finals]
    return pd.DataFrame({"fraction": fractions, column: finals})


class TestGSFEFit:
    def test_usfe_and_ssfe(self, relaxed):
        usfe, ssfe, f_usfe, f_ssfe, table = GSFEFit._original_func(
            _df(relaxed), relaxed, NY
        )
        assert (usfe, f_usfe) == (pytest.approx(180.0), 0.5)
        assert (ssfe, f_ssfe) == (pytest.approx(40.0), 1.0)
        assert np.allclose(table["gamma_mJ_m2"], GAMMA)

    def test_displacement_is_the_partial_burgers_vector(self, relaxed):
        *_, table = GSFEFit._original_func(_df(relaxed), relaxed, NY)
        b = np.linalg.norm(relaxed.cell[1]) / (3 * NY)
        assert np.allclose(table["displacement"], np.array(FRACTIONS) * b)

    def test_usfe_at_the_last_point(self, relaxed, capsys):
        usfe, ssfe, f_usfe, f_ssfe, _ = GSFEFit._original_func(
            _df(relaxed, gamma=[0, 50, 100, 150, 200]), relaxed, NY
        )
        assert usfe == ssfe == pytest.approx(200.0)
        assert f_usfe == f_ssfe == 1.0
        assert "USFE is at the last point" in capsys.readouterr().out

    def test_accepts_an_unexpanded_out_column(self, relaxed):
        usfe, *_ = GSFEFit._original_func(_df(relaxed, column="out"), relaxed, NY)
        assert usfe == pytest.approx(180.0)

    def test_failed_points_are_dropped(self, relaxed, capsys):
        df = _df(relaxed)
        df.loc[len(df)] = {"fraction": 1.25, "final": SimpleNamespace(energy=None)}
        *_, table = GSFEFit._original_func(df, relaxed, NY)
        assert len(table) == len(FRACTIONS)
        assert "1 of 6 slip points failed" in capsys.readouterr().out

    def test_needs_the_unfaulted_reference(self, relaxed):
        df = _df(relaxed, gamma=GAMMA[1:], fractions=FRACTIONS[1:])
        with pytest.raises(ValueError, match="fraction=0.0"):
            GSFEFit._original_func(df, relaxed, NY)

    def test_rejects_a_dataframe_without_results(self, relaxed):
        with pytest.raises(ValueError, match="neither a 'final' nor an 'out'"):
            GSFEFit._original_func(pd.DataFrame({"fraction": [0.0]}), relaxed, NY)

    def test_rejects_an_all_failed_sweep(self, relaxed):
        df = pd.DataFrame({"fraction": [0.0], "final": [None]})
        with pytest.raises(ValueError, match="No successful slip points"):
            GSFEFit._original_func(df, relaxed, NY)


def test_plot_gsfe_draws_the_curve_and_both_points(relaxed):
    from matplotlib.figure import Figure

    usfe, ssfe, f_usfe, f_ssfe, table = GSFEFit._original_func(
        _df(relaxed), relaxed, NY
    )
    fig = PlotGSFE._original_func(table, usfe, ssfe, f_usfe, f_ssfe, figsize="4, 3")
    assert isinstance(fig, Figure)
    assert tuple(fig.get_size_inches()) == (4.0, 3.0)
    (ax,) = fig.axes
    curve, usfe_star, ssfe_star = ax.get_lines()
    assert np.allclose(curve.get_ydata(), GAMMA)
    assert usfe_star.get_ydata()[0] == pytest.approx(180.0)
    b = np.linalg.norm(relaxed.cell[1]) / (3 * NY)
    assert usfe_star.get_xdata()[0] == pytest.approx(0.5 * b)
    assert ssfe_star.get_xdata()[0] == pytest.approx(1.0 * b)


def test_a_missing_force_gives_a_nan_fmax(relaxed):
    df = _df(relaxed)
    df["final"][1].force = None
    *_, table = GSFEFit._original_func(df, relaxed, NY)
    assert np.isnan(table["fmax"][1]) and table["fmax"][0] == 0.0


def test_gsfe_slip_point_wires_slip_then_z_only_minimize():
    group = GSFESlipPoint()
    assert list(group.outputs.ports) == ["out"]
    assert {"PartialsPerCellY", "ShiftY", "LayerShift", "SlipDirectory", "LammpsFix",
            "LammpsMinimize", "ParseLammpsOutput"} <= set(group.subgraph.nodes)  # fmt: skip
