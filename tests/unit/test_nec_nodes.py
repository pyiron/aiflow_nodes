"""Tests for NECNodes: the NECStrainPoint group's wiring and NECFit on a
synthetic uniaxial sweep with known C2..C4 (no LAMMPS)."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from ase.build import bulk

from aiflow_nodes.NECNodes import _EV_PER_A3_IN_GPA, NECFit, NECStrainPoint, _fmax

CONSTANTS = {"C2": 250.0, "C3": -1500.0, "C4": 6000.0}  # GPa
STRAINS = np.linspace(-0.04, 0.04, 9)
U0 = -50.0


def _final(relaxed, eps, axis=2):
    """A relaxed state strained by `eps` along `axis` of a hyperelastic solid
    with W(E) = sum_n Cn/n E^n, E the Green-Lagrange strain (no Poisson)."""
    F = np.eye(3)
    F[axis, axis] += eps
    E = 0.5 * ((1 + eps) ** 2 - 1)
    C = list(CONSTANTS.values())
    S = sum(c * E ** (n - 1) for n, c in zip((2, 3, 4), C))  # PK2 [GPa]
    W = sum(c / n * E**n for n, c in zip((2, 3, 4), C))  # [GPa]
    sigma = np.zeros((3, 3))
    sigma[axis, axis] = S * (1 + eps) ** 2 / np.linalg.det(F)  # Cauchy
    structure = relaxed.copy()
    structure.set_cell(np.asarray(relaxed.cell) @ F.T, scale_atoms=True)
    return SimpleNamespace(
        energy=U0 + relaxed.get_volume() * W / _EV_PER_A3_IN_GPA,
        stress=-sigma,
        structure=structure,
        force=np.zeros((len(relaxed), 3)),
    )


@pytest.fixture
def relaxed():
    return bulk("Cu", "fcc", a=3.6, cubic=True)


def _df(relaxed, column="final", axis=2):
    finals = [_final(relaxed, e, axis) for e in STRAINS]
    if column == "out":
        finals = [SimpleNamespace(final=f) for f in finals]
    return pd.DataFrame({"strain": STRAINS, column: finals})


class TestNECFit:
    @pytest.mark.parametrize("method", ["energy", "stress"])
    def test_recovers_known_constants(self, relaxed, method):
        moduli, _, _ = NECFit._original_func(_df(relaxed), relaxed, degree=4)
        for name, value in CONSTANTS.items():
            assert moduli[method]["constants"][name] == pytest.approx(value, rel=1e-6)
        assert moduli[method]["r_squared"] == pytest.approx(1.0)

    def test_summary_and_table(self, relaxed):
        _, summary, table = NECFit._original_func(_df(relaxed), relaxed, degree=4)
        assert list(summary.index) == ["energy", "stress"]
        assert {"C2", "C3", "C4", "C2_err", "r_squared"} <= set(summary.columns)
        assert len(table) == len(STRAINS)
        assert table["green_lagrange_strain"].is_monotonic_increasing
        assert table["dU_by_V0_GPa"].min() == pytest.approx(0.0)

    def test_another_deformation_axis(self, relaxed):
        moduli, _, _ = NECFit._original_func(
            _df(relaxed, axis=0), relaxed, deformation_axis=0, degree=4
        )
        assert moduli["stress"]["constants"]["C2"] == pytest.approx(250.0, rel=1e-6)

    def test_accepts_an_unexpanded_out_column(self, relaxed):
        moduli, _, _ = NECFit._original_func(_df(relaxed, "out"), relaxed, degree=4)
        assert moduli["stress"]["constants"]["C2"] == pytest.approx(250.0, rel=1e-6)

    def test_failed_points_are_dropped(self, relaxed, capsys):
        df = _df(relaxed)
        df.loc[len(df)] = {"strain": 0.05, "final": SimpleNamespace(energy=None)}
        moduli, _, table = NECFit._original_func(df, relaxed, degree=4)
        assert len(table) == len(STRAINS)
        assert "1 of 10 strain points failed" in capsys.readouterr().out
        assert moduli["energy"]["constants"]["C3"] == pytest.approx(-1500.0, rel=1e-6)

    def test_rejects_a_dataframe_without_results(self, relaxed):
        with pytest.raises(ValueError, match="neither a 'final' nor an 'out'"):
            NECFit._original_func(pd.DataFrame({"strain": [0.0]}), relaxed)

    def test_rejects_an_all_failed_sweep(self, relaxed):
        df = pd.DataFrame({"strain": [0.0], "final": [None]})
        with pytest.raises(ValueError, match="No successful strain points"):
            NECFit._original_func(df, relaxed)

    def test_rejects_degree_below_two(self, relaxed):
        with pytest.raises(ValueError, match="degree must be >= 2"):
            NECFit._original_func(_df(relaxed), relaxed, degree=1)

    def test_rejects_too_few_points_for_the_degree(self, relaxed):
        with pytest.raises(ValueError, match="need at least"):
            NECFit._original_func(_df(relaxed).iloc[:2], relaxed, degree=5)


def test_fmax_is_the_largest_force_norm():
    assert _fmax(None) != _fmax(None)  # NaN
    assert _fmax([[3.0, 4.0, 0.0], [0.0, 0.0, 1.0]]) == pytest.approx(5.0)


def test_nec_strain_point_wires_a_poisson_relaxed_minimize():
    group = NECStrainPoint()
    assert list(group.outputs.ports) == ["out"]
    assert group.inputs["pressure"].value == "0.0, 0.0, None"
    assert group.inputs["deformation_axis"].value == 2
    assert {"ApplyVoigtStrain", "StrainDirectory", "LammpsMinimize",
            "ParseLammpsOutput"} <= set(group.subgraph.nodes)  # fmt: skip
