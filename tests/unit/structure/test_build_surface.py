"""``Surface`` with a name that is not an ase.build surface function.

It used to fall through to ``return surface`` with ``surface`` unassigned and
raise ``UnboundLocalError``.
"""

import pytest

from aiflow_nodes.atomistic.structure.build import Surface


def test_surface_builds_a_slab():
    slab = Surface(element="Al", surface_type="fcc111", size="2 2 3").run()
    assert len(slab) == 12
    assert slab.pbc.all()


def test_surface_rejects_unknown_surface_type():
    with pytest.raises(ValueError, match="Unknown surface_type"):
        Surface._original_func(element="Al", surface_type="fcc999")
