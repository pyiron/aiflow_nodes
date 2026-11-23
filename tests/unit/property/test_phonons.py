"""The phonopy group hands phonopy one force set per displaced supercell.

It sweeps ``Static`` over the supercells with ``iterate``, so this holds
whether symmetry leaves one supercell or several.  EMT on Al needs no
potential files.
"""

import pytest

pytest.importorskip("ase")
pytest.importorskip("phonopy")

from aiflow.core import Workflow  # noqa: E402
from ase.build import bulk  # noqa: E402

from aiflow_nodes.atomistic.engine.ase import EMT  # noqa: E402
from aiflow_nodes.atomistic.property.phonons import (  # noqa: E402
    PhonopyParameters,
    phonopy,
)


@pytest.mark.slow
@pytest.mark.parametrize("plusminus", [False, True], ids=["one", "two"])
def test_phonopy_gets_the_forces_of_every_supercell(plusminus):
    structure = bulk("Al", cubic=True)
    wf = Workflow("phonons")
    wf.engine = EMT()
    wf.params = PhonopyParameters(is_plusminus=plusminus)
    wf.ph = phonopy(structure=structure, engine=wf.engine, phonopy_parameters=wf.params)
    wf.run()
    ph = wf.ph.outputs.phonopy.value
    n_supercells = len(ph.supercells_with_displacements)
    assert n_supercells == (2 if plusminus else 1)
    assert ph.forces.shape == (n_supercells, len(ph.supercell), 3)
