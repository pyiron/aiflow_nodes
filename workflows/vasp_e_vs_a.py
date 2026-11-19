from aiflow.core import Workflow, group_node

from aiflow_nodes.controls import IterToDataFrame
from aiflow_nodes.executors import SlurmExecutor
from aiflow_nodes.math_utils import Linspace
from aiflow_nodes.plotting import PlotDataFrameXY

# ── Group node factories ─────────────────────────────


@group_node("energy_last")
def VASP(
    kpoints,
    name,
    smearing_type="gaussian",
    crystalstructure=None,
    a=None,
    cubic=False,
    potcar_lib_path="",
    working_directory=None,
):
    from aiflow.core import Workflow

    from aiflow_nodes.atomistic.calculator.data import InputSCF
    from aiflow_nodes.atomistic.calculator.output import GetEnergyLast
    from aiflow_nodes.atomistic.engine.vasp_new import (
        CreateVaspInputResources,
        MergeVaspInput,
        ParseVaspOutput,
        RunVaspCalculation,
    )
    from aiflow_nodes.atomistic.structure.build import Bulk

    inner_wf = Workflow("VASP")
    inner_wf.Bulk = Bulk(name=name, crystalstructure=crystalstructure, a=a, cubic=cubic)
    inner_wf.scf = InputSCF(kpoints=kpoints, smearing_type=smearing_type)
    inner_wf.calc = MergeVaspInput(scf=inner_wf.scf)
    inner_wf.CreateVaspInputResources = CreateVaspInputResources(
        structure=inner_wf.Bulk,
        calc=inner_wf.calc,
        potcar_lib_path=potcar_lib_path,
        working_directory=working_directory,
    )
    inner_wf.RunVaspCalculation = RunVaspCalculation(
        io_bundle=inner_wf.CreateVaspInputResources, debug=False
    )
    inner_wf.ParseVaspOutput = ParseVaspOutput(
        io_bundle=inner_wf.RunVaspCalculation.outputs.io_bundle
    )
    inner_wf.GetEnergyLast = GetEnergyLast(
        calculator=inner_wf.ParseVaspOutput.outputs.out, store=False
    )
    return inner_wf.GetEnergyLast.outputs.energy_last


wf = Workflow("vasp_e_vs_a")
wf.storage_enabled = True

wf.Linspace = Linspace(x_min=2.1, x_max=4.0, num_points=21)

wf.SlurmExecutor = SlurmExecutor(
    partition="s.cmmg",
    pysqa_config_directory="/cmmc/u/system/SLES12/soft/pyiron/dev/pyiron-resources-cmmc/queues",
    cache_directory="/u/neugebau/cache_slurm",
)

wf.VASP = VASP(
    kpoints="6 6 6",
    name="Fe",
    smearing_type="methfessel-paxton",
    crystalstructure="bcc",
    a=2.89,
    cubic=True,
    potcar_lib_path="/cmmc/u/system/SLES12/soft/pyiron/dev/pyiron-resources-cmmc/vasp/potentials/potpaw_PBE",
    working_directory="./vasp_bulk_relax_run",
)

wf.IterToDataFrame = IterToDataFrame(
    node=wf.VASP,
    input_label="a",
    values=wf.Linspace,
    debug=False,
    executor=wf.SlurmExecutor,
    store=True,
)

wf.PlotDataFrameXY = PlotDataFrameXY(df=wf.IterToDataFrame)
