"""``plot_phase_diagram`` against the installed landau.

landau 1.7 dropped ``make_poly`` and ``make_concave_poly``; the plot imported
them inside the function, so it failed only when it ran.
"""

import warnings

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
landau = pytest.importorskip("landau")

from landau.calculate import calc_phase_diagram  # noqa: E402

from aiflow_nodes.atomistic.thermodynamics.landau.plot import (  # noqa: E402
    plot_phase_diagram,
)


@pytest.fixture(scope="module", params=[False, True], ids=["plain", "refined"])
def phase_data(request):
    """A solid solution with an ordered compound at c = 0.5, on a coarse grid."""
    a = landau.LinePhase("A", 0, 0.0, 1e-4)
    b = landau.LinePhase("B", 1, 0.0, 1e-4)
    compound = landau.LinePhase("AB", 0.5, -0.1, 0)
    solution = landau.IdealSolution("sol", a, b)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return calc_phase_diagram(
            [solution, compound],
            np.linspace(300, 1500, 8),
            np.linspace(-1, 1, 21),
            refine=request.param,
        )


@pytest.mark.parametrize("poly_method", ["concave", "segments"])
def test_plot_phase_diagram_draws_phase_polygons(phase_data, poly_method):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig = plot_phase_diagram(phase_data, poly_method=poly_method, tielines=True)
    ax = fig.axes[0]
    assert ax.patches, "no phase polygon was drawn"
    assert {p.get_label() for p in ax.patches} <= {"sol", "AB", "sol'", "AB'"}
    assert ax.get_xlim() == (0, 1)
    matplotlib.pyplot.close(fig)
