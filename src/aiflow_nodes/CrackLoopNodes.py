"""Staging module: the K-loading step of a crack specimen, as a body for
controls_loops.WhileLoop (new, 2026-11-17).

One step mirrors Onur's `load_case_job` iteration (ProductionRun_Onur.ipynb,
cell 5, method of `wcrack`):
  positions += (K - K_prev) * U(X0) on the moving atoms (predictor "all": every
  atom; "boundary": the frozen ring only; the first step always moves all),
  z wrapped, the ring held by FixAtoms -> one FIRE minimisation (the validated
  modular LAMMPS chain) -> relaxed positions (lammpsparser's unwrapped
  positions, as Onur), checks (ring motion < 1e-6 A, last dump frame at the
  minimiser's last iteration) -> crack-tip distance for the early stop.

Defaults are wcrack's (wcrack/config.py DEFAULTS, read 2026-11-17):
predictor "all"; coord cutoff 1.207 a0; surface_coordination_max 11;
analysis radius max(10, R - ring - 8) A; stop distance analysis radius - 5 A
(CrackStopDistance).

Function nodes may move to their proper place once tested; the group
(CrackLoadStep) stays here (group nodes are recorded in workflow save files).
"""

from __future__ import annotations

import re
from typing import Literal

from aiflow.core import as_function_node, group_node


def _fixed_mask(structure):
    import numpy as np
    from ase.constraints import FixAtoms

    mask = np.zeros(len(structure), bool)
    for c in structure.constraints:
        if isinstance(c, FixAtoms):
            mask[c.get_indices()] = True
    return mask


def _tip(structure):
    """Crack tip = the cell's x-y centre (crack nodes' convention), (x, y, 0)."""
    import numpy as np

    cell = np.array(structure.cell)
    c = 0.5 * (cell[0] + cell[1])
    return np.array([c[0], c[1], 0.0])


def coordination(structure, cutoff_A: float):
    """Tip-relative positions (z wrapped) and the number of neighbours within
    `cutoff_A` of each atom, periodic along z only (as LAMMPS compute
    coord/atom in the f f p box). Plain function, also for validation."""
    import numpy as np
    from scipy.spatial import cKDTree

    x = structure.get_positions() - _tip(structure)
    Lz = float(np.array(structure.cell)[2, 2])
    x[:, 2] %= Lz
    x[x[:, 2] >= Lz, 2] = 0.0
    off = 1.0e5
    pts = x.copy()
    pts[:, :2] += off
    tree = cKDTree(pts, boxsize=[4 * off, 4 * off, Lz])
    coord = tree.query_ball_point(pts, r=float(cutoff_A), return_length=True) - 1
    return x, coord


@as_function_node(["analysis_radius_A", "stop_tip_distance_A"])
def CrackStopDistance(radius_nm: float, ring_width_A: float):
    """wcrack's 'auto' analysis radius and early-stop distance:
    mobile = R - ring; analysis radius = max(10, mobile - 8) A;
    stop distance = analysis radius - 5 A."""
    mobile = 10.0 * float(radius_nm) - float(ring_width_A)
    if mobile < 15.0:
        raise ValueError(
            f"mobile radius R - ring = {mobile:g} A is below wcrack's minimum of 15 A"
        )
    analysis_radius_A = max(10.0, mobile - 8.0)
    stop_tip_distance_A = analysis_radius_A - 5.0
    return analysis_radius_A, stop_tip_distance_A


@as_function_node(["structure", "K_used"])
def IncrementKField(
    structure,
    U,
    K: float,
    K_prev: float = 0.0,
    predictor: Literal["all", "boundary"] = "all",
):
    """Positions + (K - K_prev) * U on the moving atoms; U is the unit field at
    the undeformed X0 (StrohKField, same atom order), never recomputed.

    predictor "all" moves every atom, "boundary" only the FixAtoms ring; at
    the first step (K_prev == 0) every atom moves. z is wrapped into [0, Lz);
    atoms leaving the x/y cell raise (wcrack: "increase MARGIN"). The FixAtoms
    constraint is kept (and bypassed for this displacement).
    Returns the displaced structure and K (to carry into K_prev).
    """
    import numpy as np

    if predictor not in ("all", "boundary"):
        raise ValueError(f"predictor must be 'all' or 'boundary', got {predictor!r}")
    U = np.asarray(U, float)
    n = len(structure)
    if U.shape != (n, 3):
        raise ValueError(f"U has shape {U.shape}, expected ({n}, 3)")
    fixed = _fixed_mask(structure)
    move = np.ones(n, bool) if (float(K_prev) == 0.0 or predictor == "all") else fixed
    dK = float(K) - float(K_prev)
    pos = structure.get_positions().copy()
    pos[move] += dK * U[move]
    cell = np.array(structure.cell)
    Lz = cell[2, 2]
    pos[:, 2] %= Lz
    if (
        (pos[:, :2] < 0).any()
        or (pos[:, 0] >= cell[0, 0]).any()
        or (pos[:, 1] >= cell[1, 1]).any()
    ):
        raise RuntimeError(
            f"K = {K}: atoms outside the x/y cell after the increment; use a larger margin"
        )
    structure = structure.copy()
    structure.set_positions(pos, apply_constraint=False)
    K_used = float(K)
    return structure, K_used


@as_function_node(["working_directory", "step_next"])
def StepDirectory(run_root: str, step: int = 0):
    """`{run_root}/step_{step:04d}` (as Onur's lmp/step_NNNN) and step + 1."""
    import os

    working_directory = os.path.join(str(run_root), f"step_{int(step):04d}")
    step_next = int(step) + 1
    return working_directory, step_next


@as_function_node(["structure", "energy", "fmax_free", "fnorm", "ring_motion"])
def CrackStepResult(io_bundle, input_structure, ring_tolerance_A: float = 1e-6):
    """The relaxed structure of one minimisation, as Onur's step does it:
    lammpsparser's last `unwrapped_positions` frame, forces and potential
    energy (same parser), atom order and FixAtoms of `input_structure`.

    Raises if the atom count changed or a frozen ring atom moved more than
    `ring_tolerance_A` (z by minimum image). Returns the structure, the energy
    [eV], the largest |force component| over the free atoms, the force norm,
    and the ring motion [A].
    """
    import numpy as np

    from aiflow_nodes.atomistic.engine.lammps import _parse_lammps_raw

    g = _parse_lammps_raw(io_bundle, "dump.h5", "dump.out", "log.lammps")
    new = np.asarray(g["unwrapped_positions"][-1], float)
    f = np.asarray(g["forces"][-1], float)
    n = len(input_structure)
    if len(new) != n:
        raise RuntimeError(f"{len(new)} atoms returned, {n} expected")
    fixed = _fixed_mask(input_structure)
    Lz = float(np.array(input_structure.cell)[2, 2])
    dr = new[fixed] - input_structure.get_positions()[fixed]
    dr[:, 2] -= Lz * np.rint(dr[:, 2] / Lz)
    ring_motion = float(np.abs(dr).max()) if fixed.any() else 0.0
    if ring_motion > float(ring_tolerance_A):
        raise RuntimeError(
            f"frozen ring atoms moved by {ring_motion:.2e} A (constraint not applied)"
        )
    energy = float(g["energy_pot"][-1])
    fmax_free = float(np.abs(f[~fixed]).max())
    fnorm = float(np.linalg.norm(f))
    structure = input_structure.copy()
    structure.set_positions(
        new, apply_constraint=False
    )  # unwrapped, as Onur carries them
    return structure, energy, fmax_free, fnorm, ring_motion


@as_function_node(["criterion", "iterations"])
def MinimizationStats(
    io_bundle, log_file: str = "log.lammps", dump_file: str = "dump.out"
):
    """Stopping criterion and iteration count of the last minimisation in
    log.lammps ("Stopping criterion = ...", "Iterations, force evaluations =
    ..."). Raises if the last dump frame is not at that iteration (the
    relaxed positions would then not be the final state)."""
    import os

    wd = io_bundle.working_directory
    with open(os.path.join(wd, log_file)) as fh:
        log = fh.read()
    crit = re.findall(r"Stopping criterion\s*=\s*(.+)", log)
    its = re.findall(r"Iterations, force evaluations\s*=\s*(\d+)\s+(\d+)", log)
    if not crit or not its:
        raise RuntimeError(f"no minimization stats in {os.path.join(wd, log_file)}")
    criterion = crit[-1].strip()
    iterations = int(its[-1][0])
    last_frame = None
    with open(os.path.join(wd, dump_file)) as fh:
        for line in fh:
            if line.startswith("ITEM: TIMESTEP"):
                last_frame = int(next(fh).split()[0])
    if last_frame != iterations:
        raise RuntimeError(
            f"last dump frame at timestep {last_frame}, minimization ended after {iterations} iterations"
        )
    return criterion, iterations


@as_function_node(["tip_distance", "n_surface"])
def CrackTipDistance(
    structure,
    analysis_radius_A: float,
    lattice_constant: float = 3.165,
    coord_cutoff_A: float | None = None,
    surface_coordination_max: int = 11,
):
    """wcrack's early-stop measure: the farthest free-surface atom ahead of the
    original tip. Coordination = neighbours within `coord_cutoff_A` (None:
    1.207 a0, wcrack's 'auto', as LAMMPS compute coord/atom; periodic along z
    only). Surface atoms: coordination <= `surface_coordination_max`. The tip
    distance is the largest r = |(x, y) - tip| among surface atoms with x > 0
    and r < `analysis_radius_A` (0 if none). Tip = the cell's x-y centre.
    Returns it and the number of such surface atoms.
    """
    import numpy as np

    cutoff = (
        1.207 * float(lattice_constant)
        if coord_cutoff_A is None
        else float(coord_cutoff_A)
    )
    x, coord = coordination(structure, cutoff)
    rr = np.hypot(x[:, 0], x[:, 1])
    sel = (
        (coord <= int(surface_coordination_max))
        & (x[:, 0] > 0)
        & (rr < float(analysis_radius_A))
    )
    tip_distance = float(rr[sel].max()) if sel.any() else 0.0
    n_surface = int(sel.sum())
    return tip_distance, n_surface


@group_node(
    "structure",
    "K_used",
    "step_next",
    "energy",
    "iterations",
    "criterion",
    "fmax_free",
    "ring_motion",
    "tip_distance",
    "working_directory",
)
def CrackLoadStep(
    structure,
    U,
    potential,
    run_root,
    K=1.7,
    K_prev=0.0,
    step=0,
    predictor="all",
    lmp_command=None,
    cores=32,
    analysis_radius_A=20.0,
    lattice_constant=3.165,
    surface_coordination_max=11,
):
    """One K step of the crack loading (body for WhileLoop):
    IncrementKField -> CreateLammpsStructure -> Init / Potential / Neighbor
    (skin 1.0) / GroupFromStructure (ring) / Fix (freeze, setforce 0 0 0) /
    Dump / Thermo / Minimize (FIRE, norm two, timestep 1 fs, ftol 1e-6) ->
    AssembleLammpsInput -> RunLammpsBase (no executor: the loop or
    executor_for_body places it) -> CrackStepResult, MinimizationStats,
    CrackTipDistance.

    Carry in WhileLoop: "structure -> structure, K_used -> K_prev,
    step_next -> step"; feed the K schedule into "K". Each step runs in
    `{run_root}/step_NNNN`. `lmp_command` None: `mpiexec -n cores lmp_mpi`.
    """
    from aiflow.core import Workflow

    from aiflow_nodes.atomistic.engine.lammps import (
        CreateLammpsStructure,
        RunLammpsBase,
    )
    from aiflow_nodes.atomistic.engine.lammps_input import (
        AssembleLammpsInput,
        LammpsDump,
        LammpsFix,
        LammpsInit,
        LammpsMinimize,
        LammpsPotential,
        LammpsThermo,
    )
    from aiflow_nodes.LammpsRegionGroupNodes import (
        LammpsGroupFromStructure,
        LammpsNeighbor,
    )

    wf = Workflow("crack_load_step")
    wf.Increment = IncrementKField(
        structure=structure, U=U, K=K, K_prev=K_prev, predictor=predictor
    )
    wf.StepDir = StepDirectory(run_root=run_root, step=step)
    wf.LammpsStructure = CreateLammpsStructure(
        structure=wf.Increment.outputs.structure,
        potential=potential,
        working_directory=wf.StepDir.outputs.working_directory,
    )
    wf.Init = LammpsInit(io_bundle=wf.LammpsStructure)
    wf.PotentialSection = LammpsPotential(io_bundle=wf.LammpsStructure)
    wf.Neighbor = LammpsNeighbor(io_bundle=wf.LammpsStructure, skin=1.0)
    wf.Group = LammpsGroupFromStructure(io_bundle=wf.LammpsStructure, group_id="ring")
    wf.Freeze = LammpsFix(
        io_bundle=wf.LammpsStructure,
        fix_id="freeze",
        group=wf.Group.outputs.group_id,
        style="setforce",
        args="0.0 0.0 0.0",
    )
    wf.Dump = LammpsDump(io_bundle=wf.LammpsStructure, n_print=100)
    wf.Thermo = LammpsThermo(io_bundle=wf.LammpsStructure, n_print=100)
    wf.Minimize = LammpsMinimize(
        io_bundle=wf.LammpsStructure,
        style="fire",
        e_tol=0.0,
        f_tol=1e-6,
        max_iter=100000,
        max_eval=10000000,
        min_modify="norm two",
        timestep=1.0,
    )
    wf.Assemble = AssembleLammpsInput(
        io_bundle=wf.LammpsStructure,
        init=wf.Init,
        potential=wf.PotentialSection.outputs.potential,
        potential_file_content=wf.PotentialSection.outputs.potential_file_content,
        neighbor=wf.Neighbor,
        group=wf.Group.outputs.group,
        dump=wf.Dump,
        thermo=wf.Thermo,
        fix=wf.Freeze.outputs.fix,
        fix_ids=wf.Freeze.outputs.fix_ids,
        calculation=wf.Minimize,
        mode="minimize",
    )
    wf.Run = RunLammpsBase(
        io_bundle=wf.Assemble, lmp_command=lmp_command, threads_per_core=cores
    )
    wf.Result = CrackStepResult(
        io_bundle=wf.Run.outputs.io_bundle,
        input_structure=wf.Increment.outputs.structure,
    )
    wf.Stats = MinimizationStats(io_bundle=wf.Run.outputs.io_bundle)
    wf.Tip = CrackTipDistance(
        structure=wf.Result.outputs.structure,
        analysis_radius_A=analysis_radius_A,
        lattice_constant=lattice_constant,
        surface_coordination_max=surface_coordination_max,
    )
    return (
        wf.Result.outputs.structure,
        wf.Increment.outputs.K_used,
        wf.StepDir.outputs.step_next,
        wf.Result.outputs.energy,
        wf.Stats.outputs.iterations,
        wf.Stats.outputs.criterion,
        wf.Result.outputs.fmax_free,
        wf.Result.outputs.ring_motion,
        wf.Tip.outputs.tip_distance,
        wf.StepDir.outputs.working_directory,
    )
