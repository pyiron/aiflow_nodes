from aiflow.core import Workflow

from aiflow_nodes.atomistic.calculator.data import InputMinimizationVASP, InputSCF
from aiflow_nodes.atomistic.engine.vasp_new import (
    CreateVaspInputResources,
    MergeVaspInput,
    ParseVaspOutput,
    RunVaspCalculation,
)
from aiflow_nodes.atomistic.structure.build import Bulk

wf = Workflow("vasp_bulk_relax")
wf.storage_enabled = True

wf.Bulk = Bulk(name="Fe", crystalstructure="bcc", a=2.89, cubic=True)

wf.minimization = InputMinimizationVASP(max_ionic_steps=50)

wf.scf = InputSCF(kpoints="6 6 6", smearing_type="methfessel-paxton")

wf.calc = MergeVaspInput(scf=wf.scf, minimization=wf.minimization)

wf.CreateVaspInputResources = CreateVaspInputResources(
    structure=wf.Bulk,
    calc=wf.calc,
    potcar_lib_path="/cmmc/u/system/SLES12/soft/pyiron/dev/pyiron-resources-cmmc/vasp/potentials/potpaw_PBE",
    working_directory="./vasp_bulk_relax_run",
)

wf.RunVaspCalculation = RunVaspCalculation(
    io_bundle=wf.CreateVaspInputResources,
    vasp_version="6.4.0",
    threads_per_core=1,
    debug=False,
)

wf.ParseVaspOutput = ParseVaspOutput(io_bundle=wf.RunVaspCalculation.outputs.io_bundle)
