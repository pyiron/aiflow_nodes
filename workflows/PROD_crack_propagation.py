"""
PROD: K-controlled crack propagation at 0 K (incremental loading, Onur's method)
================================================================================
Production workflow (from crack_K_loop_test.py, 2026-11-17; logic unchanged).

  MisorientedAxes -> OrientedLatticeBlock -> CrackOriginShift -> CutCylinder
  -> CutNotch -> FreezeOuterRing (FixAtoms on X0, r > R - ring)
  -> Order (no-op unless an atom order is set; used for validation)
  -> StrohKField (unit anisotropic K field U(X0))
  -> WhileLoop(body = CrackLoadStep, feed = K schedule, condition = early stop):
       each step: positions += (K - K_prev) U(X0) (predictor "all"), ring
       frozen (group ring id ... + fix setforce 0 0 0), one FIRE minimisation
       (neighbor 1.0 bin, min_modify norm two, timestep 1 fs, ftol 1e-6),
       relaxed positions carried to the next step; stop once the crack-tip
       distance exceeds the stop distance (wcrack: analysis radius - 5 A).

Validated 2026-11-17 against Onur's ProductionRun_Onur notebook (wcrack) for
this case (R 4 nm, ring 12 A, K 1.70/1.71/1.72): relaxed positions agree to
1.25e-11 A at every step, energies / iterations / stopping criteria identical,
coordination identical to LAMMPS coord/atom. NOT yet validated: the early stop
with real crack advance (needs a schedule past Griffith, ~1.74 here); see
PROJECT_MEMORY.md.

Inputs are the constants below. Each K step runs in RUN_DIR/step_NNNN.
Executor: `executor_for_body` = one SLURM job per K step, decisions in the GUI
(the validated mode). For one job for the whole schedule, wire the executor
into the WhileLoop's own `executor` port instead and drop executor_for_body.
Use a fresh RUN_DIR / cache per run (executorlib replays cached results).
"""

from aiflow.core import Workflow

from aiflow_nodes.controls_loops import ProceedWhileBelow, WhileLoop
from aiflow_nodes.CrackLoopNodes import CrackLoadStep, CrackStopDistance
from aiflow_nodes.CrackNodes import (
    WCRACK_DIR,
    CrackOriginShift,
    CutCylinder,
    CutNotch,
    FreezeOuterRing,
    MisorientedAxes,
    OrientedLatticeBlock,
    ReorderAtoms,
    StrohKField,
)
from aiflow_nodes.executors import SlurmExecutor
from aiflow_nodes.PotentialNodes import InputPotential

# ---- inputs -----------------------------------------------------------------
ELEMENT = "W"
A0_A = 3.165  # lattice constant of the potential [A]
C_GPA = "525.2, 192.5, 177.5"  # C11, C12, C44 of the potential [GPa]
R_NM = 4.0  # specimen radius [nm]
RING_A = 12.0  # frozen ring width [A] (>= 2 x cutoff = 11.6 A)
SIZE_XY_A = 110.0  # lattice block side [A] (>= 2R + 30 A)
NZ = 4  # periods along the front [001] (Lz = 12.66 A)
K_SCHEDULE = "1.70, 1.71, 1.72"  # K_I [MPa m^1/2], one LAMMPS minimisation each
PREDICTOR = "all"  # "all" or "boundary" (wcrack)
CORES = 32  # MPI ranks per step
RUN_DIR = "/cmmc/u/sanand/pyiron/projects/00_53_Hackathon_test_merge/crack_runs/PROD_1"
ENV = "/cmmc/ptmp/pyironhb/mambaforge/envs/pyiron_mpie_cmti_2026-11-17/bin"
LMP_COMMAND = f"{ENV}/mpiexec -n {CORES} --oversubscribe {ENV}/lmp_mpi -in lmp.in"

wf = Workflow("PROD_crack_propagation")

wf.SlurmExecutor = SlurmExecutor(
    partition="s.cmmg",
    run_time_max=3600,
    threads_per_core=CORES,
    pysqa_config_directory="/cmmc/u/system/SLES12/soft/pyiron/dev/pyiron-resources-cmmc/queues",
    cache_directory="./cache_PROD_crack_propagation_1",
    job_name="PROD_crack_K_step",
)

# --- specimen X0 with the frozen ring ----------------------------------------
wf.Axes = MisorientedAxes(mode="rational_front", lattice_constant=A0_A)
wf.Block = OrientedLatticeBlock(
    element=ELEMENT,
    lattice_constant=A0_A,
    x=wf.Axes.outputs.x,
    y=wf.Axes.outputs.y,
    z=wf.Axes.outputs.z,
    size_xy_A=SIZE_XY_A,
    nz=NZ,
)
wf.OriginShift = CrackOriginShift(structure=wf.Block)
wf.Cylinder = CutCylinder(structure=wf.OriginShift.outputs.structure, radius_nm=R_NM)
wf.Notch = CutNotch(structure=wf.Cylinder)
wf.Ring = FreezeOuterRing(structure=wf.Notch, radius_nm=R_NM, ring_width_A=RING_A)
wf.Order = ReorderAtoms(
    structure=wf.Ring.outputs.structure
)  # no-op; validation sets wcrack's order
wf.KField = StrohKField(
    structure=wf.Order,
    elastic_constants=C_GPA,
    x=wf.Axes.outputs.x,
    y=wf.Axes.outputs.y,
    z=wf.Axes.outputs.z,
)

wf.Potential = InputPotential(
    name="W-P_MEAM_wcrack",
    filename=f"{WCRACK_DIR}/potentials/WP.library.meam, {WCRACK_DIR}/potentials/WP.parameter.meam",
    species=ELEMENT,
    config="pair_style meam; pair_coeff * * WP.library.meam W P WP.parameter.meam W",
)
wf.Stop = CrackStopDistance(radius_nm=R_NM, ring_width_A=RING_A)

# --- the loading loop ---------------------------------------------------------
wf.Step = CrackLoadStep(
    structure=wf.Order,
    U=wf.KField.outputs.U,
    potential=wf.Potential,
    run_root=RUN_DIR,
    K=1.70,
    K_prev=0.0,
    step=0,
    predictor=PREDICTOR,
    lmp_command=LMP_COMMAND,
    cores=CORES,
    analysis_radius_A=wf.Stop.outputs.analysis_radius_A,
    lattice_constant=A0_A,
    surface_coordination_max=11,
)
wf.EarlyStop = ProceedWhileBelow(
    threshold=wf.Stop.outputs.stop_tip_distance_A, inclusive=True
)
wf.KLoop = WhileLoop(
    body=wf.Step,
    condition=wf.EarlyStop,
    carry="structure -> structure, K_used -> K_prev, step_next -> step",
    feed=K_SCHEDULE,
    feed_input="K",
    record="K_used, energy, iterations, criterion, fmax_free, ring_motion, tip_distance, working_directory",
    condition_map="tip_distance -> value",
    max_iterations=1000,
    executor_for_body=wf.SlurmExecutor,
)
