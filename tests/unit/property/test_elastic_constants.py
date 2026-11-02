"""Physics check for the rewritten elastic-constant nodes.

``atomistic/property/elastic.py`` used to depend on a ``symmetry`` module that
exists nowhere in this repo or its history, so every path through it raised
``NameError`` — and the module was unreachable anyway, because a stale in-class
import in its sibling ``phonons.py`` took the whole ``atomistic.property``
package down at import time.  The replacement derives the elastic tensor from
stress slopes under small Voigt strains, so its correctness is a *numerical*
property, not an importability one, and asserting it needs a real calculation.

EMT on fcc Cu is the cheapest calculator that gives meaningful cubic constants
(no potential files, no GPU, ~1 s).  The windows below are the ones the agentic
benchmark's copper variant grades against, so a regression here is a regression
in a graded task.
"""

import pytest

pytest.importorskip("ase")

from aiflow_nodes.atomistic.engine.ase import EMT  # noqa: E402
from aiflow_nodes.atomistic.property.elastic import (  # noqa: E402
    ApplyVoigtStrain,
    ComputeElasticConstants,
    StressStrainElasticConstants,
    voigt_strain_matrix,
)
from aiflow_nodes.atomistic.structure.build import Bulk  # noqa: E402

#: The benchmark's grading windows for fcc copper, in GPa.
CU_WINDOWS = {"C11": (140.0, 200.0), "C12": (90.0, 160.0), "C44": (55.0, 100.0)}


@pytest.fixture(scope="module")
def cu_elastic():
    """``StressStrainElasticConstants`` on fcc Cu with EMT, reference relaxed."""
    structure = Bulk(name="Cu", crystalstructure="fcc", cubic=True).run()
    engine = EMT().run()
    return StressStrainElasticConstants(
        structure=structure, engine=engine, eps=0.005, relax_reference=True
    ).run()


def test_voigt_strain_matrix_puts_engineering_shear_in_both_off_diagonals():
    # Voigt components 3-5 are engineering shears: gamma = 2 * epsilon_ij, so
    # each tensor entry carries half the requested magnitude.  Getting this
    # wrong silently doubles C44.
    normal = voigt_strain_matrix(0, 0.01)
    assert normal[0, 0] == pytest.approx(0.01)
    assert normal[1, 1] == pytest.approx(0.0)

    shear = voigt_strain_matrix(3, 0.01)
    assert shear[1, 2] == pytest.approx(0.005)
    assert shear[2, 1] == pytest.approx(0.005)
    assert shear[0, 0] == pytest.approx(0.0)


def test_apply_voigt_strain_scales_the_cell_and_keeps_the_atoms():
    structure = Bulk(name="Cu", crystalstructure="fcc", cubic=True).run()
    strained = ApplyVoigtStrain(structure=structure, component=0, magnitude=0.01).run()

    assert len(strained) == len(structure)
    a0 = structure.get_cell()[0, 0]
    assert strained.get_cell()[0, 0] == pytest.approx(a0 * 1.01)
    # only the strained direction moves
    assert strained.get_cell()[1, 1] == pytest.approx(structure.get_cell()[1, 1])


def test_elastic_tensor_is_symmetric_and_cubic(cu_elastic):
    C = cu_elastic[0].C
    assert C.shape == (6, 6)

    # symmetrisation is applied inside the node
    assert pytest.approx(C.T, abs=1e-8) == C

    # cubic symmetry: the three normal and the three shear diagonals agree
    assert C[1, 1] == pytest.approx(C[0, 0], rel=0.05)
    assert C[2, 2] == pytest.approx(C[0, 0], rel=0.05)
    assert C[4, 4] == pytest.approx(C[3, 3], rel=0.05)
    assert C[5, 5] == pytest.approx(C[3, 3], rel=0.05)

    # cubic off-diagonal block: C13 == C12, and the normal/shear blocks decouple
    assert C[0, 2] == pytest.approx(C[0, 1], rel=0.05)
    assert C[0, 3] == pytest.approx(0.0, abs=1.0)


@pytest.mark.parametrize("name", sorted(CU_WINDOWS))
def test_copper_constants_land_in_the_benchmark_window(cu_elastic, name):
    _, C11, C12, C44 = cu_elastic
    value = {"C11": C11, "C12": C12, "C44": C44}[name]
    low, high = CU_WINDOWS[name]
    assert low <= value <= high, f"{name} = {value:.1f} GPa, expected {low}-{high} GPa"


def test_moduli_follow_voigt_reuss_hill(cu_elastic):
    analysis = cu_elastic[0]
    # The Hill average lies between the Voigt and Reuss bounds.  For a cubic
    # crystal the bulk-modulus bounds coincide exactly, so compare with a
    # floating-point tolerance rather than strict ordering.
    tol = 1e-9
    assert analysis.BR <= analysis.BH + tol <= analysis.BV + 2 * tol
    assert analysis.GR <= analysis.GH + tol <= analysis.GV + 2 * tol
    assert analysis.BH > 0 and analysis.GH > 0


def test_compute_elastic_constants_delegates_and_keeps_its_ports(cu_elastic):
    """The public node kept its name and ``(structure, engine)`` signature."""
    structure = Bulk(name="Cu", crystalstructure="fcc", cubic=True).run()
    engine = EMT().run()
    node = ComputeElasticConstants(
        structure=structure, engine=engine, relax_reference=True
    )
    assert list(node.outputs.keys()) == [
        "elastic_analysis",
        "C11",
        "C12",
        "C44",
    ]

    _, C11, C12, C44 = node.run()
    _, ref11, ref12, ref44 = cu_elastic
    assert pytest.approx((ref11, ref12, ref44), rel=1e-6) == (C11, C12, C44)
