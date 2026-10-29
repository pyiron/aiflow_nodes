from aiflow.core import Workflow

from aiflow_nodes.atomistic.calculator.data import InputCalcMD, OutputCalcMD
from aiflow_nodes.atomistic.engine.lammps import (
    CreateLammpsMDInput,
    CreateLammpsStructure,
    ListPotentials,
    ParseLammpsOutput,
    RunLammpsCalculation,
)
from aiflow_nodes.atomistic.structure.build import Bulk
from aiflow_nodes.atomistic.structure.transform import Repeat
from aiflow_nodes.controls import pick_element
from aiflow_nodes.plotting import Plot

wf = Workflow("lammps_md_basic")

wf.Bulk = Bulk(name="Al", cubic=True)

wf.InputCalcMD = InputCalcMD(initial_temperature=600, langevin=True)

wf.ListPotentials = ListPotentials(structure=wf.Bulk)

wf.Repeat = Repeat(structure=wf.Bulk, repeat_scalar="4")

wf.pick_element = pick_element(lst=wf.ListPotentials, index=5)

wf.CreateLammpsStructure = CreateLammpsStructure(
    structure=wf.Repeat,
    potential=wf.pick_element,
    working_directory="./lammps_md_basic",
)

wf.CreateLammpsMDInput = CreateLammpsMDInput(
    io_bundle=wf.CreateLammpsStructure,
    calc_dataclass=wf.InputCalcMD,
    read_restart_filename="",
    write_restart_filename="",
)

wf.RunLammpsCalculation = RunLammpsCalculation(
    io_bundle=wf.CreateLammpsMDInput, debug=False, executor=None
)

wf.ParseLammpsOutput = ParseLammpsOutput(
    io_bundle=wf.RunLammpsCalculation.outputs.io_bundle
)

wf.OutputCalcMD = OutputCalcMD(input=wf.ParseLammpsOutput)

wf.Plot = Plot(y=wf.OutputCalcMD.outputs.temperatures)

wf.Plot_1 = Plot(y=wf.OutputCalcMD.outputs.energies_pot)
