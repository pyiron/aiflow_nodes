from __future__ import annotations

from aiflow.core import group_node


@group_node("chemical_potential")
def GetChemicalPotential(
    element: str,
    engine=None,
):
    from aiflow.core import Workflow

    from aiflow_nodes.atomistic.calculator.ase import Minimize
    from aiflow_nodes.atomistic.calculator.output import GetEnergyLast
    from aiflow_nodes.atomistic.structure.build import Bulk
    from aiflow_nodes.atomistic.structure.calc import NumberOfAtoms
    from aiflow_nodes.math_utils import Divide

    wf = Workflow("GetChemicalPotential")
    wf.bulk = Bulk(name=element)
    wf.minimize = Minimize(structure=wf.bulk, engine=engine)  # pressure = 0
    wf.n_atoms = NumberOfAtoms(structure=wf.bulk)
    wf.energy = GetEnergyLast(calculator=wf.minimize)

    wf.chemical_potential = Divide(x=wf.energy, y=wf.n_atoms)

    return wf.chemical_potential
