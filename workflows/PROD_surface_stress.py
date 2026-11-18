"""
PROD: surface stress and surface energy of an FCC metal (LAMMPS)
================================================================
Production workflow (from surface_stress_full_test.py, 2026-11-17; logic unchanged).

  Cell (oriented FCC) -> BulkRelax (box/relax, P = 0; one SLURM job)
  -> Sweep: IterToDataFrame over SurfaceStressStrainPoint, one SLURM job per
     in-plane strain (columns strain / bulk / surface)
  -> Fit (polynomial: tau, gamma) and TauFromPressures (independent tau from
     the zero-strain slab pressures).

Settings: Ag, a = 4.089 A, 11 strains in +-0.001 (linear fit, degree 1),
etol 0, ftol 1e-8 (relax) / 1e-9 (strain points), cg. The zero-strain point
must succeed: it defines the reference area and the pressure-based tau.
Potential: ListPotentials placeholder (index 0).

Needs the aiflow fix `fix/groupnode-pickle-nonscalar-inputs-v2` (PR open, until merged) (GroupNode
pickling keeps non-scalar inputs) in the GUI kernel and on PYTHONPATH for the
jobs. Executor on the sweep, not on the template; the template's `strain`
stays unset. Use a fresh RUN_DIR / cache per run (executorlib replays cached
results, failed ones too).
"""

from aiflow.core import Workflow

from aiflow_nodes.atomistic.engine.lammps import ListPotentials
from aiflow_nodes.atomistic.structure.build import FCCOrientedCell
from aiflow_nodes.controls import GetAttribute, IterToDataFrame, pick_element
from aiflow_nodes.executors import SlurmExecutor
from aiflow_nodes.math_utils import Linspace
from aiflow_nodes.SurfaceStressNodes import (
    LammpsMinimizeCalc,
    SurfaceStressFit,
    SurfaceStressFromPressures,
    SurfaceStressStrainPoint,
)

RUN_DIR = "/cmmc/u/sanand/pyiron/projects/00_53_Hackathon_test_merge/surface_stress_runs/PROD_1"

wf = Workflow("PROD_surface_stress")

wf.Cell = FCCOrientedCell(element="Ag", lattice_constant=4.089)

wf.ListPotentials = ListPotentials(structure=wf.Cell)
wf.pick_element = pick_element(lst=wf.ListPotentials, index=0)

wf.SlurmExecutor = SlurmExecutor(
    partition="s.cmmg",
    run_time_max=1800,
    threads_per_core=32,
    pysqa_config_directory="/cmmc/u/system/SLES12/soft/pyiron/dev/pyiron-resources-cmmc/queues",
    cache_directory="./cache_PROD_surface_stress_1",
    job_name="PROD_surface_stress",
)

wf.BulkRelax = LammpsMinimizeCalc(
    structure=wf.Cell,
    potential=wf.pick_element,
    working_directory=f"{RUN_DIR}/relax",
    e_tol=0.0,
    f_tol=1e-08,
    style="cg",
    pressure=0.0,
    executor=wf.SlurmExecutor,
)

wf.BulkRelaxFinal = GetAttribute(obj=wf.BulkRelax, attr="final")
wf.RelaxedBulk = GetAttribute(obj=wf.BulkRelaxFinal, attr="structure")

wf.Strains = Linspace(x_min=-0.001, x_max=0.001, num_points=11)

# Template: no `strain` and no executor - the sweep sets the strain and
# submits each cell.
wf.StrainPoint = SurfaceStressStrainPoint(
    relaxed_bulk=wf.RelaxedBulk,
    potential=wf.pick_element,
    base_dir=RUN_DIR,
    e_tol=0.0,
    f_tol=1e-09,
    style="cg",
)

wf.Sweep = IterToDataFrame(
    node=wf.StrainPoint,
    input_label="strain",
    values=wf.Strains,
    debug=False,
    executor=wf.SlurmExecutor,
    store=False,
)

wf.Fit = SurfaceStressFit(df=wf.Sweep)

wf.TauFromPressures = SurfaceStressFromPressures(
    df=wf.Sweep, relaxed_bulk=wf.RelaxedBulk
)
