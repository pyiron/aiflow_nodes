from aiflow.core import as_function_node
from ase.atoms import Atoms

from aiflow_nodes.atomistic.calculator.data import OutputCalcMD


@as_function_node
def structure_to_calc_md(structure: Atoms) -> OutputCalcMD.dataclass_type:
    import numpy as np

    calc_md = OutputCalcMD.pure_dataclass()
    calc_md.positions = np.array([structure.positions])
    calc_md.cells = np.array([structure.cell * np.eye(3)])
    calc_md.indices = np.array([structure.indices])
    calc_md.species = [e.Abbreviation for e in structure.species]

    return calc_md
