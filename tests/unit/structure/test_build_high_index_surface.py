"""``HighIndexSurface`` returns a plain ASE slab, without pyiron."""

from ase import Atoms

from aiflow_nodes.atomistic.structure.build import HighIndexSurface


def test_high_index_surface_returns_ase_atoms():
    slab = HighIndexSurface(
        element="Ni",
        crystal_structure="fcc",
        lattice_constant=3.52,
        terrace_orientation=[1, 1, 1],
        step_orientation=[1, 1, 0],
        kink_orientation=[1, 0, 1],
        step_down_vector=[1, 1, 0],
        length_step=2,
        length_terrace=3,
        length_kink=1,
    ).run()
    assert isinstance(slab, Atoms)
    assert not type(slab).__module__.startswith("pyiron")
    assert len(slab) > 0
    assert slab.pbc.all()
    assert slab.positions[:, 2].min() == 0
