from aiflow.core import Workflow, group_node

from aiflow_nodes.atomistic.calculator.data import InputCalcMD, OutputCalcMD
from aiflow_nodes.atomistic.structure.transform import FixSpecies
from aiflow_nodes.atomistic.structure.view import Animate

# ── Group node factories ─────────────────────────────


@group_node("structure")
def Structure(name, cubic=False, repeat_scalar=1):
    from aiflow.core import Workflow

    from aiflow_nodes.atomistic.structure.build import Bulk
    from aiflow_nodes.atomistic.structure.transform import Repeat

    inner_wf = Workflow("Structure")
    inner_wf.Bulk = Bulk(name=name, cubic=cubic)
    inner_wf.Repeat = Repeat(structure=inner_wf.Bulk, repeat_scalar=repeat_scalar)
    return inner_wf.Repeat


@group_node("element")
def GetPotential(structure, index):
    from aiflow.core import Workflow

    from aiflow_nodes.atomistic.engine.lammps import ListPotentials
    from aiflow_nodes.controls import pick_element

    inner_wf = Workflow("GetPotential")
    inner_wf.ListPotentials = ListPotentials(structure=structure)
    inner_wf.pick_element = pick_element(lst=inner_wf.ListPotentials, index=index)
    return inner_wf.pick_element


@group_node("out")
def Lammps(calc_dataclass, structure, potential, working_directory="."):
    from aiflow.core import Workflow

    from aiflow_nodes.atomistic.engine.lammps import (
        CreateLammpsMDInput,
        CreateLammpsStructure,
        ParseLammpsOutput,
        RunLammpsCalculation,
    )

    inner_wf = Workflow("Lammps")
    inner_wf.CreateLammpsStructure = CreateLammpsStructure(
        structure=structure, potential=potential, working_directory=working_directory
    )
    inner_wf.CreateLammpsMDInput = CreateLammpsMDInput(
        io_bundle=inner_wf.CreateLammpsStructure, calc_dataclass=calc_dataclass
    )
    inner_wf.RunLammpsCalculation = RunLammpsCalculation(
        io_bundle=inner_wf.CreateLammpsMDInput
    )
    inner_wf.ParseLammpsOutput = ParseLammpsOutput(
        io_bundle=inner_wf.RunLammpsCalculation.outputs.io_bundle
    )
    return inner_wf.ParseLammpsOutput


wf = Workflow("lammps_md_group_py")

wf.InputCalcMD = InputCalcMD()

wf.Structure = Structure(name="Al", cubic=True, repeat_scalar=2)

wf.FixSpecies = FixSpecies(structure=wf.Structure)

wf.GetPotential = GetPotential(structure=wf.Structure, index=0)

wf.Lammps = Lammps(
    calc_dataclass=wf.InputCalcMD,
    structure=wf.FixSpecies,
    potential=wf.GetPotential,
    working_directory="./test",
)

wf.OutputCalcMD = OutputCalcMD(input=wf.Lammps)

wf.Animate = Animate(trajectory=wf.Lammps)
