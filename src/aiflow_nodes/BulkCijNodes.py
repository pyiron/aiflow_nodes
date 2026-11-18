"""Bulk 0 K elastic tensor (C11, C12, C44) of a cubic crystal, after
04_01_Nanowires/12_BulkElasticConstants (BulkElasticConstants_Workflow.md).

Staging module. Reused, not redefined: FCCOrientedCell, ListPotentials,
pick_element, LammpsMinimizeCalc (box/relax "0.0, 0.0, 0.0"), IterToDataFrame,
the modular LAMMPS section nodes.

New:
    VoigtStrainStates        - the 6 x 2 x 2 = 24 (component, strain) sweep cells
    ApplyDeformationGradient - upper-triangular F, so the cell stays in LAMMPS
                               form and the stress comes back in the cubic frame
    StrainStateDirectory     - one run directory per state
    CijStrainPoint           - group: strain -> static LAMMPS run (run 0)
    CubicCijFit              - central-difference C, cubic averages, moduli, QC
"""

import numpy as np
import pandas as pd
from aiflow.core import as_function_node, group_node
from ase import Atoms

_EV_PER_A3_IN_GPA = 160.21766208
# Voigt j -> (row, col) of the single non-zero off-diagonal entry of F - I
_SHEAR_ENTRY = {3: (1, 2), 4: (0, 2), 5: (0, 1)}  # yz, xz, xy


def deformation_gradient(component: int, magnitude: float) -> np.ndarray:
    """Upper-triangular F for one Voigt component (0..5 = xx yy zz yz xz xy).
    Shears are simple shears with engineering strain gamma = magnitude."""
    component = int(component)
    if not 0 <= component <= 5:
        raise ValueError(f"Voigt component must be in 0..5, got {component}")
    F = np.eye(3)
    if component < 3:
        F[component, component] += magnitude
    else:
        F[_SHEAR_ENTRY[component]] = magnitude
    return F


@as_function_node("states")
def VoigtStrainStates(strains: str = "0.001, 0.002"):
    """All sweep cells for IterToDataFrame(input_label="component,strain"):
    every Voigt component 0..5 at +s and -s for each s in `strains`
    (6 x 2 x len(strains) cells; 24 for the default)."""
    magnitudes = [float(s) for s in str(strains).split(",") if s.strip()]
    if not magnitudes or any(s <= 0 for s in magnitudes):
        raise ValueError(f"strains must be positive magnitudes, got {strains!r}")
    states = [
        {"component": j, "strain": sign * s}
        for j in range(6)
        for s in magnitudes
        for sign in (+1, -1)
    ]
    return states


@as_function_node("structure")
def ApplyDeformationGradient(structure, component: int = 0, magnitude: float = 0.001):
    """h' = h F^T (cell rows = lattice vectors), atoms scaled along.
    The input cell must be lower-triangular (LAMMPS form; true for the cubic
    cell); F upper-triangular keeps it so, so LAMMPS applies no rotation and
    the parsed stress is in the input frame. Asserted."""
    from aiflow_nodes.atomistic.structure._atoms import _resolve_atoms

    atoms = _resolve_atoms(structure).copy()
    cell = np.asarray(atoms.get_cell()) @ deformation_gradient(component, magnitude).T
    upper = (cell[0, 1], cell[0, 2], cell[1, 2])
    if max(abs(v) for v in upper) > 1e-10 * np.abs(cell).max():
        raise ValueError(
            "Strained cell is not lower-triangular (LAMMPS would rotate it and the "
            f"stress frame would change); h01, h02, h12 = {upper}"
        )
    atoms.set_cell(cell, scale_atoms=True)
    structure = atoms
    return structure


@as_function_node("working_directory")
def StrainStateDirectory(base_dir: str, component: int, strain: float) -> str:
    """`{base_dir}/c{component}_{strain:+.6f}`: one directory per sweep cell."""
    import os

    working_directory = os.path.join(
        str(base_dir), f"c{int(component)}_{float(strain):+.6f}"
    )
    return working_directory


def _add_static_chain(
    inner_wf, prefix, structure, potential, working_directory, lmp_command=None
):
    """Modular-LAMMPS static (run 0) chain added into `inner_wf`; returns the
    ParseLammpsOutput node. Plain helper, not a nested group (see CLAUDE.md)."""
    from aiflow_nodes.atomistic.engine.lammps import (
        CreateLammpsStructure,
        ParseLammpsOutput,
        RunLammpsBase,
    )
    from aiflow_nodes.atomistic.engine.lammps_input import (
        AssembleLammpsInput,
        LammpsDump,
        LammpsInit,
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
    assemble = add(
        "AssembleLammpsInput",
        AssembleLammpsInput(
            io_bundle=create,
            init=init,
            potential=pot.outputs.potential,
            potential_file_content=pot.outputs.potential_file_content,
            dump=dump,
            thermo=thermo,
            run=0,
            mode="static",
        ),
    )
    run = add(
        "RunLammpsBase", RunLammpsBase(io_bundle=assemble, lmp_command=lmp_command)
    )
    return add("ParseLammpsOutput", ParseLammpsOutput(io_bundle=run.outputs.io_bundle))


@group_node("out")
def CijStrainPoint(
    relaxed_bulk, component, strain, potential, base_dir, lmp_command=None
):
    """One strained state: ApplyDeformationGradient -> static LAMMPS run in
    `{base_dir}/c{component}_{strain:+.6f}`. Returns OutputCalcStatic.
    Sweep `component,strain` with IterToDataFrame + SlurmExecutor on the sweep;
    leave both unset on the template."""
    from aiflow.core import Workflow

    from aiflow_nodes.BulkCijNodes import (
        ApplyDeformationGradient,
        StrainStateDirectory,
        _add_static_chain,
    )

    inner_wf = Workflow("CijStrainPoint")
    inner_wf.Strain = ApplyDeformationGradient(
        structure=relaxed_bulk, component=component, magnitude=strain
    )
    inner_wf.StateDirectory = StrainStateDirectory(
        base_dir=base_dir, component=component, strain=strain
    )
    parse = _add_static_chain(
        inner_wf,
        "",
        structure=inner_wf.Strain,
        potential=potential,
        working_directory=inner_wf.StateDirectory,
        lmp_command=lmp_command,
    )
    return parse


def _run_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per successful sweep cell: component, strain, Voigt Cauchy
    stress sigma = -pressure [GPa] (xx yy zz yz xz xy), energy, volume, fmax."""
    rows, n_failed = [], 0
    for _, r in df.iterrows():
        p, e, s = r.get("stress"), r.get("energy"), r.get("structure")
        if p is None or e is None or s is None or (np.isscalar(p) and np.isnan(p)):
            n_failed += 1
            continue
        p = np.asarray(p, dtype=float).reshape(3, 3)
        sigma = -np.array([p[0, 0], p[1, 1], p[2, 2], p[1, 2], p[0, 2], p[0, 1]])
        force = r.get("force")
        rows.append(
            {
                "component": int(r["component"]),
                "strain": float(r["strain"]),
                **{
                    f"sigma_{k}": v
                    for k, v in zip(("xx", "yy", "zz", "yz", "xz", "xy"), sigma)
                },
                "energy_eV": float(e),
                "volume_A3": float(s.get_volume()),
                "fmax_eV_A": float(np.abs(force).max())
                if force is not None
                else np.nan,
                "n_atoms": len(s),
            }
        )
    if n_failed:
        print(f"CubicCijFit: {n_failed} of {len(df)} sweep cells failed.")
    return (
        pd.DataFrame(rows).sort_values(["component", "strain"]).reset_index(drop=True)
    )


def _cij_at(table: pd.DataFrame, s: float) -> np.ndarray:
    """6x6 C [GPa]: column j = (sigma(+s e_j) - sigma(-s e_j)) / 2s."""
    cols = [f"sigma_{k}" for k in ("xx", "yy", "zz", "yz", "xz", "xy")]
    C = np.full((6, 6), np.nan)
    for j in range(6):
        sub = table[table["component"] == j]
        plus = sub[np.isclose(sub["strain"], +s)]
        minus = sub[np.isclose(sub["strain"], -s)]
        if len(plus) == 1 and len(minus) == 1:
            C[:, j] = (plus[cols].to_numpy()[0] - minus[cols].to_numpy()[0]) / (2 * s)
    return C


@as_function_node(["summary", "C", "table"])
def CubicCijFit(df: pd.DataFrame, relaxed_bulk: Atoms, reference_energy: float):
    """C11, C12, C44 and derived moduli from the CijStrainPoint sweep.

    C is the central difference at the smallest |strain| (s1); the next
    magnitude (s2) only feeds qc_strain_dep. Cubic averages define C_cub and
    S = C_cub^-1, S0 = S11 - S12 - S44/2. `reference_energy` is the relaxed
    cell's energy (eV), for the energy-based diagonal check.
    Returns `summary` (dict), `C` (6x6 at s1, GPa), `table` (per-run rows).
    """
    table = _run_table(df)
    mags = sorted({round(abs(s), 12) for s in table["strain"]})
    s1 = mags[0]
    C = _cij_at(table, s1)

    C11 = float(np.mean([C[0, 0], C[1, 1], C[2, 2]]))
    C12 = float(np.mean([C[0, 1], C[0, 2], C[1, 2], C[1, 0], C[2, 0], C[2, 1]]))
    C44 = float(np.mean([C[3, 3], C[4, 4], C[5, 5]]))
    C_cub = np.zeros((6, 6))
    C_cub[:3, :3] = C12
    C_cub[[0, 1, 2], [0, 1, 2]] = C11
    C_cub[[3, 4, 5], [3, 4, 5]] = C44
    S = np.linalg.inv(C_cub)
    S11, S12, S44 = S[0, 0], S[0, 1], S[3, 3]
    S0 = S11 - S12 - S44 / 2
    E100, E110, E111 = (1 / (S11 - 2 * S0 * t) for t in (0.0, 0.25, 1 / 3))
    nu_110_001 = -S12 * E110
    nu_110_m110 = -(S12 + S0 / 2) * E110

    # energy-based diagonal: (E+ + E- - 2 E0) / (V0 s^2)
    V0 = float(relaxed_bulk.get_volume())
    CE = []
    for j in range(6):
        sub = table[table["component"] == j]
        ep = sub[np.isclose(sub["strain"], +s1)]["energy_eV"]
        em = sub[np.isclose(sub["strain"], -s1)]["energy_eV"]
        CE.append(
            (ep.iloc[0] + em.iloc[0] - 2 * reference_energy)
            / (V0 * s1**2)
            * _EV_PER_A3_IN_GPA
            if len(ep) and len(em)
            else np.nan
        )
    C11_E, C44_E = float(np.mean(CE[:3])), float(np.mean(CE[3:]))

    qc_strain_dep = np.nan
    if len(mags) > 1:
        qc_strain_dep = float(np.nanmax(np.abs(C - _cij_at(table, mags[1]))) / C11)

    summary = {
        "a0": float((4 * V0 / len(relaxed_bulk)) ** (1 / 3)),  # FCC conventional
        "E0_eV_atom": float(reference_energy) / len(relaxed_bulk),
        "C11": C11,
        "C12": C12,
        "C44": C44,
        "B": (C11 + 2 * C12) / 3,
        "C_prime": (C11 - C12) / 2,
        "zener_A": 2 * C44 / (C11 - C12),
        "E100": E100,
        "E110": E110,
        "E111": E111,
        "nu_110_001": nu_110_001,
        "nu_110_m110": nu_110_m110,
        "born_stable": bool(C11 - C12 > 0 and C11 + 2 * C12 > 0 and C44 > 0),
        "qc_cubic_dev": float(np.nanmax(np.abs(C - C_cub)) / C11),
        "qc_asym": float(np.nanmax(np.abs(C - C.T)) / C11),
        "qc_strain_dep": qc_strain_dep,
        "C11_energy": C11_E,
        "C44_energy": C44_E,
        "qc_energy_C11": abs(C11_E - C11) / C11,
        "qc_energy_C44": abs(C44_E - C44) / C11,
        "fmax_eV_A": float(table["fmax_eV_A"].max()),
        "rho_identity_res": E110 / E100 - (1 + nu_110_001 - nu_110_m110),
        "strains": mags,
        "n_runs": len(table),
    }
    return summary, C, table
