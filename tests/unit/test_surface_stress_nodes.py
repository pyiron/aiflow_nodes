"""Tests for SurfaceStressNodes: the per-strain directory, the two group nodes'
wiring, SurfaceStressFit on a synthetic sweep with known tau and gamma, and
SurfaceStressFromPressures (no LAMMPS)."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from ase.build import bulk

from aiflow_nodes.SurfaceStressNodes import (
    LammpsMinimizeCalc,
    StrainWorkingDirectory,
    SurfaceStressFit,
    SurfaceStressFromPressures,
    SurfaceStressStrainPoint,
)

TAU, GAMMA = 0.08, 0.11  # eV/Å^2
STRAINS = [-0.004, -0.002, 0.0, 0.002, 0.004]
E_BULK = -400.0


def _state(structure, energy, stress=None):
    return SimpleNamespace(
        final=SimpleNamespace(
            energy=energy,
            structure=structure,
            force=np.zeros((len(structure), 3)),
            stress=stress,
        )
    )


def _sweep(strains=STRAINS, stress=None):
    """A slab whose excess energy is A0 (gamma + tau * strain), strained along x."""
    slab = bulk("Cu", "fcc", a=3.6, cubic=True).repeat((2, 2, 4))
    slab.cell[2, 2] += 10.0
    a0 = 2 * np.linalg.norm(np.cross(slab.cell[0], slab.cell[1]))
    rows = []
    for eps in strains:
        strained = slab.copy()
        strained.set_cell(
            np.diag([1 + eps, 1, 1]) @ np.asarray(slab.cell), scale_atoms=True
        )
        surface_energy = E_BULK + a0 * (GAMMA + TAU * eps)
        rows.append(
            {
                "strain": eps,
                "bulk": _state(strained, E_BULK),
                "surface": _state(strained, surface_energy, stress),
            }
        )
    return pd.DataFrame(rows)


def test_strain_working_directory_names_strain_and_stage():
    assert (
        StrainWorkingDirectory._original_func("runs", -0.0015, "surface")
        == "runs/strain_-0.001500/surface"
    )


class TestSurfaceStressFit:
    def test_recovers_known_tau_and_gamma(self):
        tau, gamma, tau_err, gamma_err, table = SurfaceStressFit._original_func(
            _sweep()
        )
        assert tau == pytest.approx(TAU)
        assert gamma == pytest.approx(GAMMA)
        assert tau_err == pytest.approx(0, abs=1e-9)
        assert gamma_err == pytest.approx(0, abs=1e-9)
        assert list(table["strain"]) == STRAINS

    def test_degree_two_beyond_the_notebooks_threshold(self):
        strains = [-0.02, -0.01, 0.0, 0.01, 0.02]
        tau, gamma, *_ = SurfaceStressFit._original_func(_sweep(strains))
        assert tau == pytest.approx(TAU)
        assert gamma == pytest.approx(GAMMA)

    def test_failed_cells_are_skipped(self):
        df = _sweep()
        df.loc[len(df)] = {"strain": 0.006, "bulk": np.nan, "surface": np.nan}
        df.loc[len(df)] = {
            "strain": 0.008,
            "bulk": _state(df["bulk"][0].final.structure, None),
            "surface": _state(df["bulk"][0].final.structure, 1.0),
        }
        *_, table = SurfaceStressFit._original_func(df)
        assert len(table) == len(STRAINS)

    def test_a_missing_force_gives_a_nan_fmax(self):
        df = _sweep()
        df["surface"][0].final.force = None
        *_, table = SurfaceStressFit._original_func(df)
        assert np.isnan(table["fmax_surface"][0]) and table["fmax_bulk"][0] == 0.0

    def test_needs_the_zero_strain_point(self):
        with pytest.raises(ValueError, match="zero-strain point is missing"):
            SurfaceStressFit._original_func(_sweep([0.001, 0.002, 0.003]))

    def test_needs_at_least_one_successful_point(self):
        df = pd.DataFrame({"strain": [0.0], "bulk": [None], "surface": [None]})
        with pytest.raises(ValueError, match="No successful strain points"):
            SurfaceStressFit._original_func(df)


class TestSurfaceStressFromPressures:
    @pytest.fixture
    def relaxed(self):
        return bulk("Cu", "fcc", a=3.6, cubic=True).repeat((2, 2, 4))

    def test_full_tensor(self, relaxed):
        pressure = np.diag([1.0, 1.0, 3.0])  # GPa
        df = _sweep(stress=pressure)
        tau = SurfaceStressFromPressures._original_func(df, relaxed)
        # (V / 2A)(pzz - p_t), V = c_relaxed * A; with the full tensor p_t is
        # scaled by the slab/bulk y-z area ratio, here (c + vacuum) / c
        slab_cell = df["surface"][2].final.structure.cell
        ratio = slab_cell[2, 2] / relaxed.cell[2, 2]
        expected = relaxed.cell[2, 2] / 2 * (3.0 - 1.0 * ratio) / 160.21766208
        assert ratio > 1
        assert tau == pytest.approx(expected)

    def test_diagonal_only(self, relaxed):
        tau = SurfaceStressFromPressures._original_func(
            _sweep(stress=np.array([1.0, 1.0, 3.0])), relaxed
        )
        assert tau == pytest.approx(relaxed.cell[2, 2] / 2 * 2.0 / 160.21766208)

    def test_rejects_another_pressure_shape(self, relaxed):
        with pytest.raises(ValueError, match="Unexpected pressure shape"):
            SurfaceStressFromPressures._original_func(
                _sweep(stress=np.zeros(6)), relaxed
            )


def test_lammps_minimize_calc_wires_one_minimize_chain():
    group = LammpsMinimizeCalc()
    assert list(group.outputs.ports) == ["out"]
    assert {
        "CreateLammpsStructure",
        "LammpsMinimize",
        "AssembleLammpsInput",
        "RunLammpsBase",
        "ParseLammpsOutput",
    } == (  # fmt: skip
        set(group.subgraph.nodes)
        - {"LammpsInit", "LammpsPotential", "LammpsDump", "LammpsThermo"}
    )


def test_surface_stress_strain_point_wires_bulk_then_surface():
    group = SurfaceStressStrainPoint()
    assert list(group.outputs.ports) == ["bulk", "surface"]
    assert group.inputs["vacuum"].value == 10.0
    nodes = set(group.subgraph.nodes)
    assert {"ApplyVoigtStrain", "BulkParseLammpsOutput", "AddVacuum",
            "SurfaceParseLammpsOutput"} <= nodes  # fmt: skip
