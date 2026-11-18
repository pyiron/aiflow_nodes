"""Tests for BulkCijNodes: strain states, the deformation gradient, the
CijStrainPoint group's wiring and CubicCijFit on a synthetic sweep built from a
known cubic tensor (no LAMMPS)."""

import numpy as np
import pandas as pd
import pytest
from ase.build import bulk

from aiflow_nodes.BulkCijNodes import (
    _EV_PER_A3_IN_GPA,
    ApplyDeformationGradient,
    CijStrainPoint,
    CubicCijFit,
    StrainStateDirectory,
    VoigtStrainStates,
    deformation_gradient,
)

C11, C12, C44 = 200.0, 120.0, 60.0
E0 = -100.0


def _cubic_c():
    C = np.zeros((6, 6))
    C[:3, :3] = C12
    C[[0, 1, 2], [0, 1, 2]] = C11
    C[[3, 4, 5], [3, 4, 5]] = C44
    return C


def _voigt_to_tensor(v):
    xx, yy, zz, yz, xz, xy = v
    return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])


def _sweep(relaxed, strains=(0.001, 0.002), failed=0):
    """IterToDataFrame-shaped rows of a linear-elastic cubic crystal."""
    C, V0 = _cubic_c(), relaxed.get_volume()
    rows = []
    for state in VoigtStrainStates._original_func(
        strains=", ".join(str(s) for s in strains)
    ):
        e = np.zeros(6)
        e[state["component"]] = state["strain"]  # engineering strain, Voigt
        sigma = C @ e
        energy = E0 + 0.5 * V0 * e @ C @ e / _EV_PER_A3_IN_GPA
        rows.append(
            {
                **state,
                "stress": -_voigt_to_tensor(sigma),  # pressure = -sigma
                "energy": energy,
                "structure": relaxed,
                "force": np.zeros((len(relaxed), 3)),
            }
        )
    rows += [
        {"component": 0, "strain": 0.003, "stress": None, "energy": None,
         "structure": None, "force": None}
    ] * failed  # fmt: skip
    return pd.DataFrame(rows)


@pytest.fixture
def relaxed():
    return bulk("Cu", "fcc", a=3.6, cubic=True)


class TestDeformationGradient:
    def test_normal_component_stretches_one_axis(self):
        F = deformation_gradient(1, 0.01)
        assert np.allclose(F, np.diag([1.0, 1.01, 1.0]))

    @pytest.mark.parametrize(
        "component, entry", [(3, (1, 2)), (4, (0, 2)), (5, (0, 1))]
    )
    def test_shear_is_a_single_upper_entry(self, component, entry):
        F = deformation_gradient(component, 0.02)
        expected = np.eye(3)
        expected[entry] = 0.02
        assert np.allclose(F, expected)
        assert np.allclose(np.tril(F, -1), 0)  # upper-triangular

    @pytest.mark.parametrize("component", [-1, 6])
    def test_rejects_components_outside_voigt(self, component):
        with pytest.raises(ValueError, match="0..5"):
            deformation_gradient(component, 0.01)


class TestVoigtStrainStates:
    def test_every_component_at_plus_and_minus_each_strain(self):
        states = VoigtStrainStates._original_func(strains="0.001, 0.002")
        assert len(states) == 24
        assert {(s["component"], s["strain"]) for s in states} == {
            (j, sign * s) for j in range(6) for s in (0.001, 0.002) for sign in (1, -1)
        }

    @pytest.mark.parametrize("strains", ["", "0.001, -0.002", "0"])
    def test_rejects_empty_or_non_positive(self, strains):
        with pytest.raises(ValueError, match="positive"):
            VoigtStrainStates._original_func(strains=strains)


class TestApplyDeformationGradient:
    def test_normal_strain_scales_cell_and_atoms(self, relaxed):
        out = ApplyDeformationGradient._original_func(
            relaxed, component=0, magnitude=0.01
        )
        assert np.allclose(out.cell.lengths(), [3.636, 3.6, 3.6])
        assert np.allclose(out.get_scaled_positions(), relaxed.get_scaled_positions())
        assert out is not relaxed

    def test_shear_keeps_the_cell_lower_triangular(self, relaxed):
        out = ApplyDeformationGradient._original_func(
            relaxed, component=5, magnitude=0.01
        )
        cell = np.asarray(out.cell)
        assert cell[1, 0] == pytest.approx(0.036)
        assert np.allclose([cell[0, 1], cell[0, 2], cell[1, 2]], 0)

    def test_rejects_a_cell_not_in_lammps_form(self, relaxed):
        tilted = relaxed.copy()
        tilted.set_cell([[3.6, 0.5, 0], [0, 3.6, 0], [0, 0, 3.6]])
        with pytest.raises(ValueError, match="lower-triangular"):
            ApplyDeformationGradient._original_func(tilted, component=0, magnitude=0.01)


def test_strain_state_directory_names_component_and_signed_strain():
    assert StrainStateDirectory._original_func("runs", 4, -0.002) == "runs/c4_-0.002000"


def test_cij_strain_point_wires_a_static_run():
    group = CijStrainPoint()
    assert list(group.outputs.ports) == ["out"]
    for name in ("relaxed_bulk", "component", "strain", "potential", "base_dir"):
        assert name in group.inputs.ports
    nodes = set(group.subgraph.nodes)
    assert {"Strain", "StateDirectory", "AssembleLammpsInput", "RunLammpsBase",
            "ParseLammpsOutput"} <= nodes  # fmt: skip


class TestCubicCijFit:
    def test_recovers_a_known_cubic_tensor(self, relaxed):
        summary, C, table = CubicCijFit._original_func(
            _sweep(relaxed), relaxed_bulk=relaxed, reference_energy=E0
        )
        assert np.allclose(C, _cubic_c())
        assert summary["C11"] == pytest.approx(C11)
        assert summary["C12"] == pytest.approx(C12)
        assert summary["C44"] == pytest.approx(C44)
        assert summary["B"] == pytest.approx((C11 + 2 * C12) / 3)
        assert summary["C_prime"] == pytest.approx((C11 - C12) / 2)
        assert summary["zener_A"] == pytest.approx(2 * C44 / (C11 - C12))
        assert summary["born_stable"] is True
        assert summary["a0"] == pytest.approx(3.6)
        assert summary["strains"] == [0.001, 0.002]
        assert summary["n_runs"] == 24 == len(table)

    def test_quality_checks_are_zero_for_exact_data(self, relaxed):
        summary, _, _ = CubicCijFit._original_func(
            _sweep(relaxed), relaxed_bulk=relaxed, reference_energy=E0
        )
        for key in ("qc_cubic_dev", "qc_asym", "qc_strain_dep"):
            assert summary[key] == pytest.approx(0, abs=1e-9)
        # the energy route sees the same diagonal
        assert summary["C11_energy"] == pytest.approx(C11, rel=1e-6)
        assert summary["C44_energy"] == pytest.approx(C44, rel=1e-6)
        assert summary["fmax_eV_A"] == 0.0

    def test_directional_moduli_match_the_compliance(self, relaxed):
        summary, _, _ = CubicCijFit._original_func(
            _sweep(relaxed), relaxed_bulk=relaxed, reference_energy=E0
        )
        S = np.linalg.inv(_cubic_c())
        assert summary["E100"] == pytest.approx(1 / S[0, 0])
        # E110/E100 = 1 + nu(110,001) - nu(110,-110): an identity of the cubic S
        assert summary["rho_identity_res"] == pytest.approx(0, abs=1e-9)

    def test_failed_cells_are_dropped(self, relaxed, capsys):
        summary, _, table = CubicCijFit._original_func(
            _sweep(relaxed, failed=2), relaxed_bulk=relaxed, reference_energy=E0
        )
        assert len(table) == 24
        assert summary["C11"] == pytest.approx(C11)
        assert "2 of 26 sweep cells failed" in capsys.readouterr().out

    def test_one_magnitude_leaves_the_strain_check_undefined(self, relaxed):
        summary, _, _ = CubicCijFit._original_func(
            _sweep(relaxed, strains=(0.001,)), relaxed_bulk=relaxed, reference_energy=E0
        )
        assert np.isnan(summary["qc_strain_dep"])
        assert summary["C44"] == pytest.approx(C44)
