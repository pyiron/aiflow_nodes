"""Nodes for the generalized stacking-fault energy (GSFE) workflow (ported from
99_Workflow_Updation/FromRaven/WORK_ModifiedStackingFaultWorkflow_Ag_polymlp.ipynb).

Staging module: holds only what the existing aiflow_nodes cannot already do,
plus a group node composing existing nodes. Once tested, move each piece to
its proper place in the tree.

Reused, not redefined here:
    FCCOrientedCell, AddVacuum, StrainWorkingDirectory, LammpsMinimizeCalc
    (bulk relax) from SurfaceStressNodes; LayerShift (the notebook's
    apply_slip: upper half shifted along y); Multiply/Divide; LammpsFix
    (z-only relaxation, `setforce 0.0 0.0 NULL`).

Genuinely new:
    GSFESlipPoint - group: slip + fixed-cell z-only minimize, one SLURM job
                    per fraction when swept
    GSFEFit       - gamma(fraction) in mJ/m^2, USFE and SSFE (notebook
                    archive_gamma_surface)

Only the (111) family is meaningful: the slip b = a/sqrt(6) along <112> is
specific to it, as in the notebook.
"""

import numpy as np
import pandas as pd
from aiflow.core import as_function_node, group_node
from ase import Atoms

# --------------------------------------------------------------------------- #
#  LAMMPS: composition of existing nodes                                       #
# --------------------------------------------------------------------------- #


def _add_fixed_minimize_chain(
    inner_wf,
    prefix,
    structure,
    potential,
    working_directory,
    e_tol=None,
    f_tol=None,
    max_iter=None,
    style=None,
    pressure=None,
    lmp_command=None,
    run=0,
):
    """SurfaceStressNodes._add_minimize_chain plus a LammpsFix with its
    defaults (`fix constraintxy all setforce 0.0 0.0 NULL`): atoms relax only
    along z. Rendered before the minimize; `run` (default 0, a single-point
    evaluation of the minimized state) follows it, and the fix is unfixed
    after that run. A plain helper, not a nested group (see CLAUDE.md).
    """
    from aiflow_nodes.atomistic.engine.lammps import (
        CreateLammpsStructure,
        ParseLammpsOutput,
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

    def add(name, node):
        setattr(inner_wf, prefix + name, node)
        return node

    create = add(
        "CreateLammpsStructure",
        CreateLammpsStructure(
            structure=structure,
            potential=potential,
            working_directory=working_directory,
        ),
    )
    init = add("LammpsInit", LammpsInit(io_bundle=create))
    pot = add("LammpsPotential", LammpsPotential(io_bundle=create))
    dump = add("LammpsDump", LammpsDump(io_bundle=create))
    thermo = add("LammpsThermo", LammpsThermo(io_bundle=create))
    fix = add("LammpsFix", LammpsFix(io_bundle=create))
    minimize = add(
        "LammpsMinimize",
        LammpsMinimize(
            io_bundle=create,
            e_tol=e_tol,
            f_tol=f_tol,
            max_iter=max_iter,
            style=style,
            pressure=pressure,
        ),
    )
    assemble = add(
        "AssembleLammpsInput",
        AssembleLammpsInput(
            io_bundle=create,
            init=init,
            potential=pot.outputs.potential,
            potential_file_content=pot.outputs.potential_file_content,
            dump=dump,
            thermo=thermo,
            fix=fix.outputs.fix,
            fix_ids=fix.outputs.fix_ids,
            calculation=minimize,
            run=run,
            mode="minimize",
        ),
    )
    run = add(
        "RunLammpsBase", RunLammpsBase(io_bundle=assemble, lmp_command=lmp_command)
    )
    return add("ParseLammpsOutput", ParseLammpsOutput(io_bundle=run.outputs.io_bundle))


@group_node("out")
def GSFESlipPoint(
    relaxed_slab,
    fraction,
    ny,
    potential,
    base_dir,
    e_tol=None,
    f_tol=None,
    max_iter=None,
    style=None,
    lmp_command=None,
):
    """One GSFE point: rigidly slip the upper half of `relaxed_slab` along y by
    `fraction` of the Shockley partial b = a/sqrt(6) (notebook `apply_slip`),
    then minimize with a fixed cell and atoms free along z only. Runs in
    `{base_dir}/strain_{fraction:+.6f}/slip`.

    `ny` is the supercell repeat along y (FCCOrientedCell's `supercell`
    "nx,ny,nz"). The (111) cell's y vector is ny * a*sqrt(6)/2 long (ASE reduces
    [1,1,-2] to its lattice period), so shift_y = fraction / (3*ny) in fractional
    units is exactly fraction*a/sqrt(6).

    Returns the `OutputCalcMinimize`. Sweep `fraction` with IterToDataFrame;
    with a SlurmExecutor on the sweep each fraction is one job.
    """
    from aiflow.core import Workflow

    from aiflow_nodes.atomistic.structure.transform import LayerShift
    from aiflow_nodes.math_utils import Divide, Multiply
    from aiflow_nodes.SurfaceStressNodes import StrainWorkingDirectory

    inner_wf = Workflow("GSFESlipPoint")
    inner_wf.PartialsPerCellY = Multiply(x=3, y=ny)
    inner_wf.ShiftY = Divide(x=fraction, y=inner_wf.PartialsPerCellY)
    inner_wf.LayerShift = LayerShift(
        structure=relaxed_slab, shift_fraction=0.5, shift_x=0.0, shift_y=inner_wf.ShiftY
    )
    inner_wf.SlipDirectory = StrainWorkingDirectory(
        base_dir=base_dir, strain=fraction, stage="slip"
    )
    parse = _add_fixed_minimize_chain(
        inner_wf,
        "",
        structure=inner_wf.LayerShift,
        potential=potential,
        working_directory=inner_wf.SlipDirectory,
        e_tol=e_tol,
        f_tol=f_tol,
        max_iter=max_iter,
        style=style,
        pressure=None,
        lmp_command=lmp_command,
    )
    return parse


# --------------------------------------------------------------------------- #
#  Analysis                                                                    #
# --------------------------------------------------------------------------- #

_EV_PER_A2_IN_MJ_PER_M2 = 16021.766208  # 1000 * constants.electron_volt / 1e-20


def _fmax(force) -> float:
    if force is None:
        return np.nan
    return float(np.max(np.linalg.norm(np.asarray(force), axis=1)))


def _gsfe_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-fraction energy / in-plane area from IterToDataFrame(GSFESlipPoint).
    Failed cells are dropped, as in the notebook. Accepts the expanded `final`
    column or an unexpanded `out` column (as NECFit does)."""
    if "final" in df.columns:
        finals = df["final"]
    elif "out" in df.columns:
        finals = df["out"].map(lambda out: getattr(out, "final", None))
    else:
        raise ValueError(
            "Sweep DataFrame has neither a 'final' nor an 'out' column; "
            f"columns: {list(df.columns)}"
        )

    rows, n_failed = [], 0
    for (_, r), final in zip(df.iterrows(), finals):
        if (
            getattr(final, "energy", None) is None
            or getattr(final, "structure", None) is None
        ):
            n_failed += 1
            continue
        cell = np.array(final.structure.cell, dtype=float)
        rows.append(
            {
                "fraction": float(r["fraction"]),
                "energy": float(final.energy),
                "area": float(np.linalg.norm(np.cross(cell[0], cell[1]))),
                "fmax": _fmax(final.force),
            }
        )
    if not rows:
        raise ValueError(
            f"No successful slip points in the sweep DataFrame ({n_failed} of {len(df)} failed)."
        )
    if n_failed:
        print(f"GSFEFit: {n_failed} of {len(df)} slip points failed and are excluded.")
    return pd.DataFrame(rows).sort_values("fraction").reset_index(drop=True)


@as_function_node(["usfe", "ssfe", "usfe_fraction", "ssfe_fraction", "table"])
def GSFEFit(df: pd.DataFrame, relaxed_bulk: Atoms, ny: int):
    """GSFE curve and its special points (notebook archive_gamma_surface):

    - gamma = (E - E0) / |a x b| [mJ/m^2], E0 and the area from fraction 0
      (the unfaulted relaxed slab; the cell is fixed so the area is common)
    - USFE = max gamma; SSFE = min gamma at or after the USFE point (if the
      USFE is the last point, SSFE = USFE with a warning, as in the notebook)
    - displacement = fraction * b, b = |relaxed_bulk.cell[1]| / (3*ny) = a/sqrt(6),
      i.e. the slip GSFESlipPoint actually applied

    Returns usfe, ssfe [mJ/m^2], their fractions, and the per-fraction table.
    """
    table = _gsfe_table(df)
    is_ref = np.isclose(table["fraction"].to_numpy(), 0.0)
    if not is_ref.any():
        raise ValueError(
            "fraction=0.0 (the unfaulted reference) is missing or failed: "
            "cannot compute gamma without a reference energy/area."
        )
    ref = table[is_ref].iloc[0]

    gamma = (
        (table["energy"].to_numpy() - ref["energy"])
        / ref["area"]
        * _EV_PER_A2_IN_MJ_PER_M2
    )
    fractions = table["fraction"].to_numpy()

    usfe_idx = int(np.argmax(gamma))
    if usfe_idx < len(gamma) - 1:
        ssfe_idx = usfe_idx + int(np.argmin(gamma[usfe_idx:]))
    else:
        print("GSFEFit: USFE is at the last point, cannot find SSFE after it.")
        ssfe_idx = usfe_idx

    b_partial = float(np.linalg.norm(np.array(relaxed_bulk.cell)[1])) / (3 * int(ny))
    table = table.assign(displacement=fractions * b_partial, gamma_mJ_m2=gamma)
    return (
        float(gamma[usfe_idx]),
        float(gamma[ssfe_idx]),
        float(fractions[usfe_idx]),
        float(fractions[ssfe_idx]),
        table,
    )


# --------------------------------------------------------------------------- #
#  Plotting                                                                    #
# --------------------------------------------------------------------------- #


@as_function_node("fig")
def PlotGSFE(
    table: pd.DataFrame,
    usfe: float,
    ssfe: float,
    usfe_fraction: float,
    ssfe_fraction: float,
    title: str = "GSFE",
    figsize: str = "6, 5",
):
    """GSFE curve, gamma [mJ/m^2] vs slip displacement [Å], from GSFEFit's
    outputs, with the USFE and SSFE points starred (notebook
    archive_gamma_surface, raw plot). The notebook's normalized plot
    (gamma - gamma_0) is the same curve, since gamma_0 = 0 at fraction 0.

    `figsize` is "width, height" in inches, explicit so the size does not
    depend on matplotlib's global rcParams. Plain matplotlib for now; to be
    moved onto SATemplates later.
    """
    from matplotlib import pyplot as plt

    width, height = (float(v) for v in str(figsize).split(","))
    table = table.sort_values("fraction")
    fractions = table["fraction"].to_numpy()
    displacement = table["displacement"].to_numpy()
    gamma = table["gamma_mJ_m2"].to_numpy()
    # slip per unit fraction (b = a/sqrt(6)), from the table itself, so the
    # stars sit exactly on the curve
    nonzero = fractions > 0
    b_partial = (
        float(np.median(displacement[nonzero] / fractions[nonzero]))
        if nonzero.any()
        else 0.0
    )

    fig, ax = plt.subplots(figsize=(width, height))
    ax.plot(
        displacement,
        gamma,
        marker="D",
        markersize=7,
        markerfacecolor="white",
        markeredgecolor="blue",
        markeredgewidth=1.5,
        linestyle="-",
        linewidth=1,
        color="blue",
        label="γ",
    )
    ax.plot(
        usfe_fraction * b_partial,
        usfe,
        marker="*",
        markersize=15,
        linestyle="none",
        color="red",
        label=f"USFE = {usfe:.3f} mJ/m$^2$",
    )
    ax.plot(
        ssfe_fraction * b_partial,
        ssfe,
        marker="*",
        markersize=15,
        linestyle="none",
        color="green",
        label=f"SSFE = {ssfe:.3f} mJ/m$^2$",
    )
    ax.set_xlabel(r"Displacement ($\mathrm{\AA}$)")
    ax.set_ylabel(r"$\gamma$ (mJ/m$^2$)")
    ax.set_title(title)
    ax.grid(True, which="both", linewidth=0.5, alpha=0.7)
    ax.minorticks_on()
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.legend(frameon=True, edgecolor="black", loc="best")
    fig.tight_layout()
    plt.close(fig)
    return fig
