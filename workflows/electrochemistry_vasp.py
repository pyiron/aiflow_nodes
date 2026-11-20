from aiflow.core import Workflow

from aiflow_nodes.atomistic.calculator.data import InputMDVASP, InputSCF
from aiflow_nodes.atomistic.engine.vasp_new import (
    CreateVaspInputResources,
    MergeVaspInput,
    RunVaspCalculation,
)
from aiflow_nodes.atomistic.structure.build import Surface
from aiflow_nodes.atomistic.structure.view import Plot3d
from aiflow_nodes.electrochemistry.add_potential.vasp import CCESetup, ParsePotential
from aiflow_nodes.electrochemistry.structure.build import AddNeonLayer, AddWaterFilm

wf = Workflow("electrochemistry_vasp")
wf.storage_enabled = True

wf.Surface = Surface(element="Al", size="3 4 4", vacuum=20, orthogonal=True)

wf.md = InputMDVASP(n_ionic_steps=100)

wf.scf = InputSCF(kpoints="1 1 1")

wf.AddWaterFilm = AddWaterFilm(electrode=wf.Surface)

wf.calc = MergeVaspInput(scf=wf.scf, md=wf.md)

wf.AddNeonLayer = AddNeonLayer(structure=wf.AddWaterFilm)

wf.Plot3d = Plot3d(structure=wf.AddNeonLayer)

wf.CCESetup = CCESetup(
    structure=wf.AddNeonLayer,
    electrode=wf.Surface,
    calc=wf.calc,
    potcar_lib_path="/cmmc/u/system/SLES12/soft/pyiron/dev/pyiron-resources-cmmc/vasp/potentials/potpaw_PBE",
)

wf.CreateVaspInputResources = CreateVaspInputResources(
    structure=wf.CCESetup.outputs.structure,
    calc=wf.CCESetup.outputs.calc,
    potcar_lib_path="/cmmc/u/system/SLES12/soft/pyiron/dev/pyiron-resources-cmmc/vasp/potentials/potpaw_PBE",
    plugin_data=wf.CCESetup.outputs.plugin_data,
    working_directory="./electrochemistry_vasp_run",
)

wf.RunVaspCalculation = RunVaspCalculation(
    io_bundle=wf.CreateVaspInputResources, debug=True
)

wf.ParsePotential = ParsePotential(io_bundle=wf.RunVaspCalculation.outputs.io_bundle)
