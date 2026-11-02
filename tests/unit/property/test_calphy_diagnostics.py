"""Checks for the calphy run diagnostics.

The free-energy nodes used to return only ``free_energy`` and ``temperature``,
so a run whose reversible-scaling path had drifted (structure change, switching
too fast) was indistinguishable from a clean one -- which is how the demo
workflow came to report a melting temperature roughly a third of the true
value without a single warning.  ``_rs_forward_backward`` recovers the forward
and backward G(T) branches that ``calphy.integrators.integrate_rs`` averages
away, and their separation is the quantity that exposes such a run.

Nothing here launches LAMMPS: the helpers only read ``ts.forward_*.dat`` /
``ts.backward_*.dat`` and the log, all of which are written by hand into
``tmp_path``.  The one test that touches a real simulation folder skips itself
when that folder is absent.
"""

import os

import numpy as np
import pytest

pytest.importorskip("calphy")

from aiflow_nodes.atomistic.property.calphy import (  # noqa: E402
    CalphyDiagnosticsTable,
    PlotCalphyHysteresis,
    _collect_diagnostics,
    _read_calphy_warnings,
    _rs_forward_backward,
)

N_POINTS = 51
LAMBDA_START, LAMBDA_STOP = 1.0, 0.5


def _write_sweep(folder, slope_forward, slope_backward, iteration=1):
    """Write one forward/backward pair whose work integral is analytic.

    The columns are ``(dx, energy, volume, lambda)``.  Putting
    ``dx = slope * lambda`` makes the integrand ``dx / lambda`` a constant, so
    ``int dx/lambda dlambda`` is exactly ``slope * (lambda - lambda_start)``.
    Volumes are zero so the pressure term drops out.
    """
    flam = np.linspace(LAMBDA_START, LAMBDA_STOP, N_POINTS)
    blam = flam[::-1]
    zeros = np.zeros_like(flam)
    np.savetxt(
        os.path.join(folder, f"ts.forward_{iteration}.dat"),
        np.column_stack([slope_forward * flam, zeros, zeros, flam]),
    )
    np.savetxt(
        os.path.join(folder, f"ts.backward_{iteration}.dat"),
        np.column_stack([slope_backward * blam, zeros, zeros, blam]),
    )
    return flam


def test_rs_forward_backward_matches_analytic_integral(tmp_path):
    from calphy.integrators import kb

    slope_f, slope_b = 0.3, 0.2
    f0, t0 = -3.5, 700.0
    flam = _write_sweep(tmp_path, slope_f, slope_b)

    temperature, fe_fwd, fe_bwd, e_diss = _rs_forward_backward(
        str(tmp_path), f0=f0, t0=t0, natoms=500, pressure=0.0, n_iterations=1
    )

    expected_t = t0 / flam
    base = f0 / flam + 1.5 * kb * expected_t * np.log(flam)
    np.testing.assert_allclose(temperature, expected_t)
    np.testing.assert_allclose(fe_fwd, base + slope_f * (flam - LAMBDA_START) / flam)
    np.testing.assert_allclose(fe_bwd, base + slope_b * (flam - LAMBDA_START) / flam)

    # The two branches are driven apart by exactly the slope difference; the
    # extreme is at the far end of the lambda path.
    assert e_diss == pytest.approx(abs(slope_f - slope_b) / 2)


def test_rs_forward_backward_zero_hysteresis_for_identical_branches(tmp_path):
    _write_sweep(tmp_path, 0.3, 0.3)
    _, fe_fwd, fe_bwd, e_diss = _rs_forward_backward(
        str(tmp_path), f0=-3.5, t0=700.0, natoms=500, pressure=0.0, n_iterations=1
    )
    np.testing.assert_allclose(fe_fwd, fe_bwd)
    assert e_diss == pytest.approx(0.0, abs=1e-15)


def test_rs_forward_backward_averages_curves_but_reports_worst_dissipation(tmp_path):
    """Curves are the mean over runs; the dissipation is the worst single run.

    calphy logs ``np.min(es)`` instead, so a single drifting iteration among
    several leaves no trace in its output -- which is exactly the failure these
    diagnostics exist to catch.
    """
    _write_sweep(tmp_path, 0.3, 0.2, iteration=1)  # dissipation 0.05
    _write_sweep(tmp_path, 0.5, 0.1, iteration=2)  # dissipation 0.20
    _, fe_fwd, fe_bwd, e_diss = _rs_forward_backward(
        str(tmp_path), f0=-3.5, t0=700.0, natoms=500, pressure=0.0, n_iterations=2
    )
    _, fe_fwd_1, _, _ = _rs_forward_backward(
        str(tmp_path), f0=-3.5, t0=700.0, natoms=500, pressure=0.0, n_iterations=1
    )
    # mean slope 0.4 vs. the first iteration's 0.3 -> the averaged curve sits
    # strictly further from the reference everywhere but the start point.
    assert np.all(np.abs(fe_fwd[1:] - fe_fwd_1[1:]) > 0)

    assert e_diss == pytest.approx(0.20)
    # Not the mean-curve hysteresis (0.125) and not calphy's minimum (0.05).
    assert e_diss != pytest.approx(abs(0.4 - 0.15) / 2)
    assert e_diss != pytest.approx(0.05)


def test_read_calphy_warnings(tmp_path):
    (tmp_path / "calphy.log").write_text(
        "2026-08-18 20:39:12 calphy.helpers INFO     Reference phase is liquid\n"
        "2026-08-18 20:39:35 calphy.helpers WARNING  Found max energy dissipation\n"
        "2026-08-18 20:39:36 calphy.helpers INFO     done\n"
    )
    assert _read_calphy_warnings(str(tmp_path)) == ["Found max energy dissipation"]


def test_read_calphy_warnings_without_log(tmp_path):
    assert _read_calphy_warnings(str(tmp_path)) == []


class _FakeJob:
    """The handful of attributes ``_collect_diagnostics`` reads off a calphy job."""

    fe = -3.5
    ferr = 1e-3
    w = -3.3
    fref = -0.07
    fideal = 0.0
    pv = 0.0
    k = [0.077, 0.705, 0.077]
    rho = 0.058
    vol = 8624.6
    natoms = 500
    report = {"results": {"einstein_crystal": -0.070, "com_correction": -0.001}}


class _FakeCalc:
    _temperature = 700.0
    _pressure = 0.0
    n_iterations = 1


def _bundle(tmp_path, **kwargs):
    return _collect_diagnostics(
        _FakeJob(), _FakeCalc(), str(tmp_path), "solid", **kwargs
    )


def test_collect_diagnostics_populates_bundle(tmp_path):
    flam = _write_sweep(tmp_path, 0.3, 0.2)
    (tmp_path / "calphy.log").write_text("x calphy.helpers WARNING  something odd\n")

    sweep_t = 700.0 / flam
    diag = _bundle(
        tmp_path,
        temperature=sweep_t,
        free_energy=np.linspace(-3.5, -3.9, N_POINTS),
        fe_error=np.full(N_POINTS, 1e-3),
    )

    assert diag.phase == "solid"
    assert diag.natoms == 500
    assert diag.fe_free_energy == pytest.approx(-3.5)
    assert diag.vol_atom == pytest.approx(8624.6 / 500)
    assert diag.einstein_crystal == pytest.approx(-0.070)
    assert diag.spring_constants == pytest.approx([0.077, 0.705, 0.077])

    assert len(diag.free_energy_forward) == N_POINTS
    assert len(diag.free_energy_backward) == N_POINTS
    np.testing.assert_allclose(
        diag.hysteresis,
        (np.asarray(diag.free_energy_forward) - np.asarray(diag.free_energy_backward))
        / 2,
    )
    assert diag.rs_max_dissipation == pytest.approx(0.05)

    # calphy's own warning survives, and the dissipation (0.05 >> 1e-4) earns
    # one of ours on top of it.
    assert "something odd" in diag.calphy_warnings
    assert any("dissipation" in w for w in diag.calphy_warnings)


def test_collect_diagnostics_clean_run_adds_no_warning(tmp_path):
    flam = _write_sweep(tmp_path, 0.3, 0.3)
    diag = _bundle(
        tmp_path,
        temperature=700.0 / flam,
        free_energy=np.linspace(-3.5, -3.9, N_POINTS),
    )
    assert diag.calphy_warnings == []
    assert diag.free_energy_error == []


def test_collect_diagnostics_survives_missing_sweep_files(tmp_path):
    """A parse failure must degrade to a warning, never kill an expensive run."""
    diag = _bundle(
        tmp_path,
        temperature=np.linspace(700, 1200, 5),
        free_energy=np.linspace(-3.5, -3.9, 5),
    )
    assert type(diag).__name__ == "CalphyDiagnostics"
    assert any("diagnostics collection failed" in w for w in diag.calphy_warnings)


def test_collect_diagnostics_without_sweep_keeps_ti_block(tmp_path):
    """``fe``-mode nodes have no sweep; the integration block is still filled."""
    diag = _bundle(tmp_path)
    assert diag.fe_free_energy == pytest.approx(-3.5)
    assert len(diag.temperature) == 0
    assert len(diag.free_energy_forward) == 0
    assert diag.rs_max_dissipation == 0.0
    assert diag.calphy_warnings == []


def test_display_nodes_accept_the_bundle(tmp_path):
    import matplotlib

    matplotlib.use("Agg")

    flam = _write_sweep(tmp_path, 0.3, 0.2)
    diag = _bundle(
        tmp_path,
        temperature=700.0 / flam,
        free_energy=np.linspace(-3.5, -3.9, N_POINTS),
    )

    df = CalphyDiagnosticsTable(diagnostics=diag).run()
    assert set(df.columns) == {"quantity", "value", "unit"}
    quantities = set(df["quantity"])
    assert "max switching hysteresis" in quantities
    assert "WARNING" in quantities  # the 0.05 eV/atom dissipation is flagged
    hysteresis = df.loc[df["quantity"] == "max switching hysteresis", "value"].iloc[0]
    assert hysteresis == pytest.approx(0.05)

    assert PlotCalphyHysteresis(diagnostics=diag).run() is not None
    # An fe-mode bundle has no curves to draw but must not raise.
    assert PlotCalphyHysteresis(diagnostics=_bundle(tmp_path)).run() is not None


REAL_RUNS = {
    # folder -> max dissipation calphy itself logged for that run
    "notebooks/calphy/ts-gc7mze9.dat-solid-300-0": 2.1755e-03,
    "notebooks/calphy/ts-mqbg9vq.dat-liquid-300-0": 4.3791e-04,
}


@pytest.mark.parametrize("folder,expected", sorted(REAL_RUNS.items()))
def test_dissipation_matches_calphy_log_on_real_runs(folder, expected):
    """Regression against runs already on disk, if this checkout has them.

    calphy prints the same number at the end of ``calphy.log``; reproducing it
    is what pins ``_rs_forward_backward`` to calphy's own arithmetic.
    """
    root = os.path.join(os.path.dirname(__file__), "..", "..", "..", folder)
    if not os.path.isdir(root):
        pytest.skip("simulation folder not present in this checkout")

    sweep = np.loadtxt(os.path.join(root, "temperature_sweep.dat"), unpack=True)
    _, fe_fwd, fe_bwd, e_diss = _rs_forward_backward(
        root, f0=sweep[1][0], t0=sweep[0][0], natoms=500, pressure=0.0, n_iterations=1
    )
    assert e_diss == pytest.approx(expected, rel=1e-3)
    # The mean of the two branches is what calphy wrote to temperature_sweep.dat.
    np.testing.assert_allclose((fe_fwd + fe_bwd) / 2, sweep[1], atol=1e-12)
