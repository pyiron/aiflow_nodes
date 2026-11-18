"""Nodes for the nonlinear-elastic-constant (NEC) workflow (ported from
99_Workflow_Updation/FromRaven/WORK_NonlinearElasticModulus_LAMMPS*.ipynb).

Staging module: holds only what the existing aiflow_nodes cannot already do,
plus a group node composing existing nodes. Once tested, move each piece to
its proper place in the tree (atomistic/property, ...).

Reused, not redefined here:
    FCCOrientedCell, StrainWorkingDirectory, LammpsMinimizeCalc (bulk relax)
    and _add_minimize_chain from SurfaceStressNodes; ApplyVoigtStrain from
    atomistic/property/elastic. The Poisson relaxation is LammpsMinimize's
    per-component `pressure` string ("0.0, 0.0, None").

Genuinely new:
    NECStrainPoint - group: strain + Poisson-relaxed minimize, one SLURM job
                     per strain when swept
    NECFit         - notebook `extract_elastic_moduli` (PK2 stress /
                     Green-Lagrange strain fits, C2..C_degree)
"""

import numpy as np
import pandas as pd
from aiflow.core import as_function_node, group_node
from ase import Atoms

# --------------------------------------------------------------------------- #
#  LAMMPS: composition of existing nodes                                       #
# --------------------------------------------------------------------------- #


@group_node("out")
def NECStrainPoint(
    relaxed_bulk,
    strain,
    potential,
    base_dir,
    deformation_axis=2,
    pressure="0.0, 0.0, None",
    e_tol=None,
    f_tol=None,
    max_iter=None,
    style=None,
    lmp_command=None,
):
    """One NEC strain point: strain the relaxed bulk by `strain` along
    `deformation_axis` (Voigt 0/1/2 = xx/yy/zz), then minimize with
    LammpsMinimize(pressure=...). The default "0.0, 0.0, None" relaxes x and y
    to zero pressure with z fixed: the notebook's Poisson relaxation for
    `deformation_axis=2`. Strain along another axis needs the matching
    pressure, e.g. "None, 0.0, 0.0" for axis 0. Runs in
    `{base_dir}/strain_{strain:+.6f}/bulk`.

    Returns the `OutputCalcMinimize`. Sweep `strain` with IterToDataFrame;
    with a SlurmExecutor on the sweep each strain is one job.
    """
    from aiflow.core import Workflow

    from aiflow_nodes.atomistic.property.elastic import ApplyVoigtStrain
    from aiflow_nodes.SurfaceStressNodes import (
        StrainWorkingDirectory,
        _add_minimize_chain,
    )

    inner_wf = Workflow("NECStrainPoint")
    inner_wf.ApplyVoigtStrain = ApplyVoigtStrain(
        structure=relaxed_bulk, component=deformation_axis, magnitude=strain
    )
    inner_wf.StrainDirectory = StrainWorkingDirectory(
        base_dir=base_dir, strain=strain, stage="bulk"
    )
    parse = _add_minimize_chain(
        inner_wf,
        "",
        structure=inner_wf.ApplyVoigtStrain,
        potential=potential,
        working_directory=inner_wf.StrainDirectory,
        e_tol=e_tol,
        f_tol=f_tol,
        max_iter=max_iter,
        style=style,
        pressure=pressure,
        lmp_command=lmp_command,
    )
    return parse


# --------------------------------------------------------------------------- #
#  Analysis                                                                    #
# --------------------------------------------------------------------------- #

_EV_PER_A3_IN_GPA = 160.21766208  # constants.electron_volt / 1e-30 / 1e9


def _fmax(force) -> float:
    if force is None:
        return np.nan
    return float(np.max(np.linalg.norm(np.asarray(force), axis=1)))


def _nec_strain_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-strain energy / pressure tensor / final cell from
    IterToDataFrame(NECStrainPoint). Failed cells are dropped, as the notebook
    excluded failed strains from the fit.

    NECStrainPoint's single output is an OutputCalcMinimize dataclass, which
    IterToDataFrame expands into one column per field, so the final state is
    the `final` column. An unexpanded `out` column (`.final`) is accepted too.
    """
    if "final" in df.columns:
        finals = df["final"]
    elif "out" in df.columns:
        finals = df["out"].map(lambda out: getattr(out, "final", None))
    else:
        raise ValueError(
            "Sweep DataFrame has neither a 'final' nor an 'out' column; "
            f"columns: {list(df.columns)}"
        )

    rows = []
    n_failed = 0
    for (_, r), final in zip(df.iterrows(), finals):
        if (
            getattr(final, "energy", None) is None
            or getattr(final, "stress", None) is None
            or getattr(final, "structure", None) is None
        ):
            n_failed += 1
            continue
        rows.append(
            {
                "strain": float(r["strain"]),
                "energy": float(final.energy),
                "pressure": np.asarray(final.stress, dtype=float),
                "cell": np.array(final.structure.cell, dtype=float),
                "fmax": _fmax(final.force),
            }
        )
    if not rows:
        raise ValueError(
            f"No successful strain points in the sweep DataFrame "
            f"({n_failed} of {len(df)} rows failed or have no final energy/stress/structure)."
        )
    if n_failed:
        print(
            f"NECFit: {n_failed} of {len(df)} strain points failed and are excluded from the fit."
        )
    return pd.DataFrame(rows).sort_values("strain").reset_index(drop=True)


def _poly(x, coeffs, first_power):
    x = np.asarray(x, dtype=float)
    return sum(c * x ** (i + first_power) for i, c in enumerate(coeffs))


def _r_squared(y, y_fit):
    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0


@as_function_node(["moduli", "summary", "table"])
def NECFit(
    df: pd.DataFrame,
    relaxed_bulk: Atoms,
    deformation_axis: int = 2,
    degree: int = 4,
):
    """Nonlinear elastic constants C2..C_degree [GPa] along `deformation_axis`
    (notebook `extract_elastic_moduli`, same maths):

    - F = cell @ inv(reference cell), J = det F
    - PK2 S = J F^-1 sigma F^-T with Cauchy sigma = -pressure tensor [GPa]
    - Green-Lagrange E = (F^T F - I) / 2
    - energy fit:  (U - U0)/V0 = sum_n c_n E^n (n = 2..degree), Cn = n c_n
      (converted eV/Å^3 -> GPa); U0 = energy at the point with |E| smallest
    - stress fit:  S = sum_n Cn E^(n-1)

    `relaxed_bulk` is the reference (zero-strain relaxed) structure; V0 is its
    cell volume. Returns `moduli` (the notebook's nested dict: energy/stress ->
    constants, errors, r_squared), `summary` (one row per method, Cn and Cn_err
    columns) and `table` (per-strain data incl. E and S along the axis).
    """
    from scipy.optimize import curve_fit

    table = _nec_strain_table(df)
    axis = int(deformation_axis)
    n_params = int(degree) - 1
    if n_params < 1:
        raise ValueError(f"degree must be >= 2, got {degree}")
    if len(table) < n_params:
        raise ValueError(
            f"{len(table)} successful strain points; need at least {n_params} "
            f"for degree {degree}."
        )

    ref_cell = np.array(relaxed_bulk.cell, dtype=float)
    ref_inv = np.linalg.inv(ref_cell)
    V0 = abs(np.linalg.det(ref_cell))

    gl_strain, pk2 = [], []
    for cell, pressure in zip(table["cell"], table["pressure"]):
        F = cell @ ref_inv
        J = np.linalg.det(F)
        F_inv = np.linalg.inv(F)
        sigma = -np.asarray(pressure).reshape(3, 3)
        S = J * F_inv @ sigma @ F_inv.T
        E = 0.5 * (F.T @ F - np.eye(3))
        gl_strain.append(E[axis, axis])
        pk2.append(S[axis, axis])
    gl_strain = np.array(gl_strain)
    pk2 = np.array(pk2)
    energies = table["energy"].to_numpy()

    E0 = energies[np.argmin(np.abs(gl_strain))]
    dU_by_V = (energies - E0) / V0

    popt_e, pcov_e = curve_fit(
        lambda x, *c: _poly(x, c, 2), gl_strain, dU_by_V, p0=np.ones(n_params)
    )
    popt_s, pcov_s = curve_fit(
        lambda x, *c: _poly(x, c, 1), gl_strain, pk2, p0=np.ones(n_params)
    )
    perr_e = np.sqrt(np.diag(pcov_e))
    perr_s = np.sqrt(np.diag(pcov_s))

    energy_constants, energy_errors, stress_constants, stress_errors = {}, {}, {}, {}
    for i in range(n_params):
        n = i + 2
        energy_constants[f"C{n}"] = float(popt_e[i] * _EV_PER_A3_IN_GPA * n)
        energy_errors[f"C{n}_err"] = float(perr_e[i] * _EV_PER_A3_IN_GPA * n)
        stress_constants[f"C{n}"] = float(popt_s[i])
        stress_errors[f"C{n}_err"] = float(perr_s[i])

    moduli = {
        "energy": {
            "constants": energy_constants,
            "errors": energy_errors,
            "r_squared": _r_squared(dU_by_V, _poly(gl_strain, popt_e, 2)),
        },
        "stress": {
            "constants": stress_constants,
            "errors": stress_errors,
            "r_squared": _r_squared(pk2, _poly(gl_strain, popt_s, 1)),
        },
    }

    summary = pd.DataFrame(
        [
            {
                "method": method,
                **moduli[method]["constants"],
                **moduli[method]["errors"],
                "r_squared": moduli[method]["r_squared"],
            }
            for method in ("energy", "stress")
        ]
    ).set_index("method")

    table = table.assign(
        green_lagrange_strain=gl_strain,
        pk2_stress=pk2,
        dU_by_V0_GPa=dU_by_V * _EV_PER_A3_IN_GPA,
    )
    return moduli, summary, table
