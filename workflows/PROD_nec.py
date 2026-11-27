"""
PROD: nonlinear elastic constants (NEC) of an FCC metal (LAMMPS)
================================================================
Production workflow (from nec_full_test.py, 2026-11-17; logic unchanged).

  Cell (oriented FCC) -> BulkRelax (box/relax aniso, P = 0; one SLURM job)
  -> Sweep: IterToDataFrame over NECStrainPoint, one SLURM job per strain
     (uniaxial along z, x/y relaxed to P = 0; columns strain / out)
  -> NECFit (C2, C3, C4 by the energy and the PK2-stress methods).

Settings: Ag, a = 4.089 A, Miller 100, 7x7x7, 11 strains in -0.05..0.05 along
z, etol 0, ftol 1e-9 (relax) / 1e-8 (strain points), cg, fit degree 5.
Potential: ListPotentials placeholder (index 0).

Needs the aiflow fix `fix/groupnode-pickle-nonscalar-inputs-v2` (PR open, until merged) (sweep template
groups keep non-scalar inputs). Executor on the sweep, not on the template;
the template's `strain` stays unset. Use a fresh RUN_DIR / cache per run
(executorlib replays cached results, failed ones too).
"""

from aiflow.core import Workflow

from aiflow_nodes.atomistic.engine.lammps import ListPotentials
from aiflow_nodes.atomistic.structure.build import FCCOrientedCell
from aiflow_nodes.controls import GetAttribute, IterToDataFrame, pick_element
from aiflow_nodes.executors import SlurmExecutor
from aiflow_nodes.math_utils import Linspace
from aiflow_nodes.NECNodes import NECFit, NECStrainPoint
from aiflow_nodes.SurfaceStressNodes import LammpsMinimizeCalc

RUN_DIR = "/cmmc/u/sanand/pyiron/projects/00_53_Hackathon_test_merge/nec_runs/PROD_1"

wf = Workflow("PROD_nec")
wf.storage_enabled = True

wf.Cell = FCCOrientedCell(
    element="Ag", lattice_constant=4.089, miller_indices="100", supercell="7,7,7"
)

wf.SlurmExecutor = SlurmExecutor(
    partition="s.cmmg",
    run_time_max=1800,
    threads_per_core=32,
    pysqa_config_directory="/cmmc/u/system/SLES12/soft/pyiron/dev/pyiron-resources-cmmc/queues",
    cache_directory="./cache_PROD_nec_1",
    job_name="PROD_nec",
)

wf.Strains = Linspace(x_min=-0.05, x_max=0.05, num_points=11)

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

wf.StrainPoint = NECStrainPoint(
    relaxed_bulk=wf.RelaxedBulk,
    potential=wf.pick_element,
    base_dir=RUN_DIR,
    deformation_axis=2,
    pressure="0.0, 0.0, None",
    e_tol=0.0,
    f_tol=1e-08,
    style="cg",
)

wf.Sweep = IterToDataFrame(
    node=wf.StrainPoint,
    input_label="strain",
    values=wf.Strains,
    debug=False,
    executor=wf.SlurmExecutor,
    store=True,
)

wf.Fit = NECFit(df=wf.Sweep, relaxed_bulk=wf.RelaxedBulk, degree=5)
