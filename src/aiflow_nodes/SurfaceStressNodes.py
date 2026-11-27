"""Nodes for the surface-stress workflow (ported from
99_Workflow_Updation/FromRaven/WORK_SurfaceStress_LAMMPS*.ipynb).

Staging module: holds only what the existing aiflow_nodes cannot already do, plus
two group nodes that compose existing nodes. Once tested, move each piece to its
proper place in the tree (structure/build, structure/transform, atomistic/property,
...).

Moved into the tree, re-exported here:
    FCCOrientedCell           - notebook `Struct_gen`  -> atomistic/structure/build.py
    AddVacuum                 - notebook `add_vacuum`  -> atomistic/structure/transform.py

Genuinely new function nodes:
    StrainWorkingDirectory    - per-strain run directory (the notebook encoded it in
                                `ForceMinimization_bulk`'s `comment`)
    SurfaceStressFit          - notebook `compute_surface_stress_from_fit` + the
                                per-strain gamma/area bookkeeping from `run_surface_stress`
    SurfaceStressFromPressures - notebook `compute_surface_stress_from_pressures`

Group nodes (existing nodes only):
    LammpsMinimizeCalc        - replaces `ForceMinimization_bulk` (pyiron_atomistics
                                `lammps_function`) with the modular LAMMPS input nodes
    SurfaceStressStrainPoint  - Stage 1 (strained bulk) + Stage 2 (strained slab) for
                                one strain value; one SLURM job per strain when swept
"""

import numpy as np
import pandas as pd
from aiflow.core import as_function_node, group_node
from ase import Atoms

# --------------------------------------------------------------------------- #
#  Structure                                                                   #
# --------------------------------------------------------------------------- #
# Moved into the tree (2026-11-16); re-exported here so existing imports and
# saved workflows that reference aiflow_nodes.SurfaceStressNodes keep working.
from aiflow_nodes.atomistic.structure.build import FCCOrientedCell  # noqa: E402,F401
from aiflow_nodes.atomistic.structure.transform import AddVacuum  # noqa: E402,F401


@as_function_node("working_directory")
def StrainWorkingDirectory(base_dir: str, strain: float, stage: str = "bulk") -> str:
    """`{base_dir}/strain_{strain:+.6f}/{stage}`: one LAMMPS directory per strain and
    stage, so parallel sweep cells never write into the same files."""
    import os

    working_directory = os.path.join(
        str(base_dir), f"strain_{float(strain):+.6f}", str(stage)
    )
    return working_directory


# --------------------------------------------------------------------------- #
#  LAMMPS: composition of existing nodes                                       #
# --------------------------------------------------------------------------- #


def _add_minimize_chain(
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
):
    """Add the modular-LAMMPS minimize chain to `inner_wf` (node labels prefixed
    with `prefix`) and return its ParseLammpsOutput node.

    A plain helper rather than a nested group node on purpose: aiflow drops a
    group input that only feeds a *nested* group (it is never exposed on the
    outer group). Adding the chain's nodes directly to the calling group keeps
    every input wired to a function-node port.
    """
    from aiflow_nodes.atomistic.engine.lammps import (
        CreateLammpsStructure,
        ParseLammpsOutput,
        RunLammpsBase,
    )
    from aiflow_nodes.atomistic.engine.lammps_input import (
        AssembleLammpsInput,
        LammpsDump,
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
            calculation=minimize,
            mode="minimize",
        ),
    )
    run = add(
        "RunLammpsBase", RunLammpsBase(io_bundle=assemble, lmp_command=lmp_command)
    )
    return add("ParseLammpsOutput", ParseLammpsOutput(io_bundle=run.outputs.io_bundle))


@group_node("out")
def LammpsMinimizeCalc(
    structure,
    potential,
    working_directory,
    e_tol=None,
    f_tol=None,
    max_iter=None,
    style=None,
    pressure=None,
    lmp_command=None,
):
    """Minimize with the modular LAMMPS input nodes; returns `OutputCalcMinimize`.

    `pressure=None` keeps the cell fixed; a value (GPa) adds `box/relax iso`.
    Unset tolerances/style/max_iter inherit `InputCalcMinimize` defaults.
    `lmp_command=None` uses RunLammpsBase's default command. Note RunLammpsBase
    does not write `io_bundle.lammps_potential_string` to disk, so a DataFrame
    potential rendered as `include potential.inp` needs that file another way.
    """
    from aiflow.core import Workflow

    from aiflow_nodes.SurfaceStressNodes import _add_minimize_chain

    inner_wf = Workflow("LammpsMinimizeCalc")
    parse = _add_minimize_chain(
        inner_wf,
        "",
        structure=structure,
        potential=potential,
        working_directory=working_directory,
        e_tol=e_tol,
        f_tol=f_tol,
        max_iter=max_iter,
        style=style,
        pressure=pressure,
        lmp_command=lmp_command,
    )
    return parse


@group_node("bulk", "surface")
def SurfaceStressStrainPoint(
    relaxed_bulk,
    strain,
    potential,
    base_dir,
    vacuum=10.0,
    vacuum_direction="z",
    deformation_axis=0,
    e_tol=None,
    f_tol=None,
    style=None,
    lmp_command=None,
):
    """One strain point: strain the relaxed bulk along `deformation_axis` (Voigt
    0..2 = xx, yy, zz), minimize at fixed cell (Stage 1), add `vacuum` Å (default
    10) along `vacuum_direction` to *that* relaxed result, minimize the slab at
    fixed cell (Stage 2).

    Returns the two `OutputCalcMinimize`s. Sweep `strain` with IterToDataFrame;
    with a SlurmExecutor each strain point is one job running both stages.
    """
    from aiflow.core import Workflow

    from aiflow_nodes.atomistic.property.elastic import ApplyVoigtStrain
    from aiflow_nodes.controls import GetAttribute
    from aiflow_nodes.SurfaceStressNodes import (
        StrainWorkingDirectory,
        _add_minimize_chain,
    )

    inner_wf = Workflow("SurfaceStressStrainPoint")
    inner_wf.ApplyVoigtStrain = ApplyVoigtStrain(
        structure=relaxed_bulk, component=deformation_axis, magnitude=strain
    )
    inner_wf.BulkDirectory = StrainWorkingDirectory(
        base_dir=base_dir, strain=strain, stage="bulk"
    )
    bulk = _add_minimize_chain(
        inner_wf,
        "Bulk",
        structure=inner_wf.ApplyVoigtStrain,
        potential=potential,
        working_directory=inner_wf.BulkDirectory,
        e_tol=e_tol,
        f_tol=f_tol,
        style=style,
        lmp_command=lmp_command,
    )
    inner_wf.BulkFinal = GetAttribute(obj=bulk, attr="final")
    inner_wf.BulkFinalStructure = GetAttribute(obj=inner_wf.BulkFinal, attr="structure")
    inner_wf.AddVacuum = AddVacuum(
        structure=inner_wf.BulkFinalStructure, vacuum=vacuum, direction=vacuum_direction
    )
    inner_wf.SurfaceDirectory = StrainWorkingDirectory(
        base_dir=base_dir, strain=strain, stage="surface"
    )
    surface = _add_minimize_chain(
        inner_wf,
        "Surface",
        structure=inner_wf.AddVacuum,
        potential=potential,
        working_directory=inner_wf.SurfaceDirectory,
        e_tol=e_tol,
        f_tol=f_tol,
        style=style,
        lmp_command=lmp_command,
    )
    return bulk, surface


# --------------------------------------------------------------------------- #
#  Analysis                                                                    #
# --------------------------------------------------------------------------- #


def _fmax(force) -> float:
    if force is None:
        return np.nan
    return float(np.max(np.linalg.norm(np.asarray(force), axis=1)))


def _strain_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-strain energies/areas/gamma from IterToDataFrame(SurfaceStressStrainPoint).

    Failed cells (non-dataclass entries, e.g. NaN) are dropped, as the notebook
    excluded failed strains from the fit. gamma = (E_surface - E_bulk) / area with
    area = 2 x in-plane cell area (two free surfaces), as in the notebook.
    """
    rows = []
    for _, r in df.iterrows():
        bulk, surf = r.get("bulk"), r.get("surface")
        if not (hasattr(bulk, "final") and hasattr(surf, "final")):
            continue
        if bulk.final.energy is None or surf.final.energy is None:
            continue
        cell = np.array(surf.final.structure.cell)
        area = 2 * np.linalg.norm(np.cross(cell[0], cell[1]))
        rows.append(
            {
                "strain": float(r["strain"]),
                "energy_bulk": float(bulk.final.energy),
                "energy_surface": float(surf.final.energy),
                "area": area,
                "gamma": (surf.final.energy - bulk.final.energy) / area,
                "fmax_bulk": _fmax(bulk.final.force),
                "fmax_surface": _fmax(surf.final.force),
                "pressure_surface": surf.final.stress,
                "cell_surface": cell,
            }
        )
    if not rows:
        raise ValueError("No successful strain points in the sweep DataFrame.")
    return pd.DataFrame(rows).sort_values("strain").reset_index(drop=True)


def _zero_strain_row(table: pd.DataFrame) -> pd.Series:
    zero = table[np.isclose(table["strain"], 0.0, atol=1e-6)]
    if zero.empty:
        raise ValueError(
            "The zero-strain point is missing (failed or not in the strain list); "
            "it defines the reference surface area."
        )
    return zero.iloc[0]


@as_function_node(["tau", "gamma", "tau_err", "gamma_err", "table"])
def SurfaceStressFit(df: pd.DataFrame, degree: int | None = None):
    """Surface stress tau and surface energy gamma [eV/Å^2] from a polynomial fit
    of A*gamma vs strain (notebook `compute_surface_stress_from_fit`).

    `degree=None` picks it as the notebook did: 2 if max(strain) > 0.009, else 1.
    The reference area is the zero-strain point's `area` (= 2 x in-plane area).
    `table` is the per-strain data the fit used.
    """
    table = _strain_table(df)
    surface_area = _zero_strain_row(table)["area"]
    strains = table["strain"].to_numpy()
    a_gamma = (table["area"] * table["gamma"]).to_numpy()

    if degree is None:
        degree = 2 if strains.max() > 0.009 else 1
    coeffs, cov = np.polyfit(strains, a_gamma, deg=int(degree), cov=True)
    # last two coefficients: linear term (tau) and constant term (gamma)
    tau, gamma = coeffs[-2:] / surface_area
    tau_err, gamma_err = np.sqrt(np.diag(cov)[-2:]) / surface_area
    return float(tau), float(gamma), float(tau_err), float(gamma_err), table


@as_function_node("tau_from_pressures")
def SurfaceStressFromPressures(df: pd.DataFrame, relaxed_bulk: Atoms) -> float:
    """Independent tau [eV/Å^2] from the zero-strain slab's pressure tensor
    (notebook `compute_surface_stress_from_pressures`). Pressures are taken to be
    in GPa, as in the notebook (pyiron/lammpsparser convention)."""
    from scipy import constants

    table = _strain_table(df)
    zero = _zero_strain_row(table)
    surface_area = zero["area"]
    cell_surface = np.asarray(zero["cell_surface"])
    relaxed_cell = np.array(relaxed_bulk.cell)
    pressure = np.asarray(zero["pressure_surface"])

    ev_per_a3_in_gpa = constants.electron_volt / 1e-30 / 1e9
    ayz_surface = np.linalg.norm(np.cross(cell_surface[1], cell_surface[2]))
    ayz_relaxed = np.linalg.norm(np.cross(relaxed_cell[1], relaxed_cell[2]))
    if pressure.shape == (3,):
        pxx, pyy, pzz = pressure / ev_per_a3_in_gpa
        p_t = (pxx + pyy) / 2
    elif pressure.shape == (3, 3):
        pxx, pyy, pzz = np.diag(pressure) / ev_per_a3_in_gpa
        p_t = ((pxx + pyy) / 2) * (ayz_surface / ayz_relaxed)
    else:
        raise ValueError(f"Unexpected pressure shape {pressure.shape}")

    volume = relaxed_cell[2, 2] * surface_area
    tau_from_pressures = (volume / (2 * surface_area)) * (pzz - p_t)
    return float(tau_from_pressures)
