"""
PROD: generalized stacking fault energy (GSFE) of an FCC metal (LAMMPS)
=======================================================================
Production workflow (from gsfe_full_test.py, 2026-11-17; logic unchanged).

  Cell (oriented FCC, Miller 111) -> BulkRelax (box/relax aniso, P = 0; one
  SLURM job) -> Slab (10 A vacuum along z)
  -> Sweep: IterToDataFrame over GSFESlipPoint, fractions 0..1.05 of the
     Shockley partial a/sqrt(6), one SLURM job per fraction (fixed cell,
     z-only relaxation)
  -> GSFEFit (gamma curve in mJ/m^2, USFE, SSFE) -> PlotGSFE.

Settings as in the source file: Cu, a = 3.65 A (the source docstring said Ag;
the cell it builds is Cu), Miller 111, 22 fractions, ny = 6, vacuum 10 A,
etol 0, ftol 1e-9, cg, 1 h per job. Potential: ListPotentials placeholder
(index 0).

Needs the aiflow fix `fix/groupnode-pickle-nonscalar-inputs-v2` (PR open, until merged) (sweep template
groups keep non-scalar inputs). Executor on the sweep, not on the template;
the template's `fraction` stays unset. Use a fresh RUN_DIR / cache per run
(executorlib replays cached results, failed ones too).
"""

from aiflow.core import Workflow

from aiflow_nodes.atomistic.engine.lammps import ListPotentials
from aiflow_nodes.atomistic.structure.build import FCCOrientedCell
from aiflow_nodes.atomistic.structure.transform import AddVacuum
from aiflow_nodes.controls import GetAttribute, IterToDataFrame, pick_element
from aiflow_nodes.executors import SlurmExecutor
from aiflow_nodes.GSFENodes import GSFEFit, GSFESlipPoint, PlotGSFE
from aiflow_nodes.math_utils import Linspace
from aiflow_nodes.SurfaceStressNodes import LammpsMinimizeCalc

RUN_DIR = "/cmmc/u/sanand/pyiron/projects/00_53_Hackathon_test_merge/gsfe_runs/PROD_1"

wf = Workflow("PROD_gsfe")
wf.storage_enabled = True

wf.Cell = FCCOrientedCell(element="Cu", lattice_constant=3.65)

wf.Fractions = Linspace(x_max=1.05, num_points=22)

wf.SlurmExecutor = SlurmExecutor(
    partition="s.cmmg",
    run_time_max=3600,
    threads_per_core=32,
    pysqa_config_directory="/cmmc/u/system/SLES12/soft/pyiron/dev/pyiron-resources-cmmc/queues",
    cache_directory="./cache_PROD_gsfe_1",
    job_name="PROD_gsfe",
)

wf.ListPotentials = ListPotentials(structure=wf.Cell)

wf.pick_element = pick_element(lst=wf.ListPotentials, index=0)

wf.BulkRelax = LammpsMinimizeCalc(
    structure=wf.Cell,
    potential=wf.pick_element,
    working_directory=f"{RUN_DIR}/relax",
    e_tol=0.0,
    f_tol=1e-09,
    style="cg",
    pressure="0.0, 0.0, 0.0",
    executor=wf.SlurmExecutor,
)

wf.BulkRelaxFinal = GetAttribute(obj=wf.BulkRelax, attr="final")

wf.RelaxedBulk = GetAttribute(obj=wf.BulkRelaxFinal, attr="structure")

wf.Slab = AddVacuum(structure=wf.RelaxedBulk)

wf.SlipPoint = GSFESlipPoint(
    relaxed_slab=wf.Slab,
    ny=6,
    potential=wf.pick_element,
    base_dir=RUN_DIR,
    e_tol=0.0,
    f_tol=1e-09,
    style="cg",
)

wf.Sweep = IterToDataFrame(
    node=wf.SlipPoint,
    input_label="fraction",
    values=wf.Fractions,
    debug=False,
    executor=wf.SlurmExecutor,
    store=True,
)

wf.Fit = GSFEFit(df=wf.Sweep, relaxed_bulk=wf.RelaxedBulk, ny=6)

wf.PlotGSFE = PlotGSFE(
    table=wf.Fit.outputs.table,
    usfe=wf.Fit.outputs.usfe,
    ssfe=wf.Fit.outputs.ssfe,
    usfe_fraction=wf.Fit.outputs.usfe_fraction,
    ssfe_fraction=wf.Fit.outputs.ssfe_fraction,
)
