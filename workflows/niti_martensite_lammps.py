from aiflow.core import Workflow, as_function_node

from aiflow_nodes.atomistic.calculator.ase import GenericOptimizerSettings, Relax
from aiflow_nodes.atomistic.calculator.data import InputCalcMD
from aiflow_nodes.atomistic.engine.ase import GRACE
from aiflow_nodes.atomistic.engine.lammps import (
    CreateLammpsMDInput,
    CreateLammpsStructure,
    GetPotential,
    LammpsStageAverages,
    ParseLammpsOutput,
    RunLammpsCalculation,
)
from aiflow_nodes.atomistic.structure.transform import Repeat

# ── Local node definitions ──────────────────────


@as_function_node("structure")
def B2UnitCell(species_a: str = "Ni", species_b: str = "Ti", a: float = 3.01):
    """Cubic two-atom B2 (CsCl-type) cell: species_a at (0,0,0), species_b at (1/2,1/2,1/2)."""
    from ase import Atoms

    structure = Atoms(
        symbols=[species_a, species_b],
        scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        cell=[a, a, a],
        pbc=True,
    )
    return structure


@as_function_node
def CoolingHeatingSchedule(
    t_high: float = 600.0, t_low: float = 100.0, t_step: float = 25.0
):
    """Temperature ladder t_high -> t_low (cooling) followed by t_low+t_step -> t_high (heating)."""
    n = int(round((t_high - t_low) / t_step))
    cooling = [round(float(t_high - i * t_step), 6) for i in range(n + 1)]
    heating = [round(float(t_low + i * t_step), 6) for i in range(1, n + 1)]
    temperatures = cooling + heating
    branches = ["cooling"] * len(cooling) + ["heating"] * len(heating)
    return temperatures, branches


@as_function_node
def TransformationTemperatures(df):
    """Forward (cooling) and reverse (heating) transformation temperatures from the
    largest excess jump in mean potential energy between neighbouring temperatures."""
    import numpy as np

    results = []
    for branch, sign in (("cooling", -1.0), ("heating", 1.0)):
        sub = df[df["branch"] == branch].sort_values("stage")
        e = sub["epot_eV_per_atom"].to_numpy()
        t = sub["T_target_K"].to_numpy()
        if len(e) < 3:
            results.append((float("nan"), 0.0))
            continue
        d = np.diff(e)
        excess = sign * (d - np.median(d))
        k = int(np.argmax(excess))
        results.append((float(0.5 * (t[k] + t[k + 1])), float(1000.0 * excess[k])))
    T_cooling_K, cooling_jump_meV = results[0]
    T_heating_K, heating_jump_meV = results[1]
    return T_cooling_K, T_heating_K, cooling_jump_meV, heating_jump_meV


@as_function_node("fig")
def PlotCoolingHeatingCycle(
    df, T_cooling_K: float | None = None, T_heating_K: float | None = None
):
    """Hysteresis loops: potential energy, volume and maximum cell-angle deviation vs T."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    quantities = [
        ("epot_eV_per_atom", "E_pot (eV/atom)"),
        ("volume_A3_per_atom", "V (Å³/atom)"),
        ("max_angle_deviation_deg", "max |angle − 90°| (deg)"),
    ]
    for ax, (col, ylabel) in zip(axes, quantities):
        for branch, style in (("cooling", "b-o"), ("heating", "r-s")):
            sub = df[df["branch"] == branch]
            ax.plot(sub["T_target_K"], sub[col], style, ms=4, label=branch)
        for t, c in ((T_cooling_K, "b"), (T_heating_K, "r")):
            if t is not None and t == t:
                ax.axvline(t, color=c, ls="--", lw=0.8)
        ax.set_xlabel("T (K)")
        ax.set_ylabel(ylabel)
        ax.legend()
    fig.suptitle(
        "B2 <-> martensite cooling/heating cycle (zero pressure, flexible cell)"
    )
    fig.tight_layout()
    return fig


@as_function_node
def MartensiteEnergyGain(b2_relaxation, martensite_relaxation):
    """E(relaxed low-T phase) - E(relaxed B2), per atom, in meV/atom."""
    from aiflow_nodes.atomistic.structure._atoms import to_ase

    n_b2 = len(to_ase(b2_relaxation.structure))
    n_m = len(to_ase(martensite_relaxation.structure))
    delta_e_meV_per_atom = 1000.0 * (
        martensite_relaxation.energy / n_m - b2_relaxation.energy / n_b2
    )
    return delta_e_meV_per_atom


@as_function_node("structure")
def ColdestStructure(df, structures: list):
    """Last MD frame of the coldest stage: the low-temperature (martensite) phase."""
    structure = structures[int(df["T_target_K"].idxmin())]
    return structure


wf = Workflow("niti_martensite_lammps")
wf.storage_enabled = True

wf.b2_unit_cell = B2UnitCell()

wf.foundation_model = GRACE(model="GRACE-2L-OMAT-large-ft-AM")

# 10 ps per stage (n_ionic_steps must be a multiple of n_print); pressure 0 GPa.
wf.md_params = InputCalcMD(
    temperature=600.0, pressure=0.0, n_ionic_steps=10_000, n_print=100
)

wf.relax_settings = GenericOptimizerSettings(max_steps=100)

wf.temperature_schedule = CoolingHeatingSchedule(t_step=50.0)

# 6x6x6 = 432 atoms: a 4x4x4 cell is too constrained for B19' variants to form.
wf.b2_supercell = Repeat(structure=wf.b2_unit_cell, repeat_scalar=6)

# Ko et al. (2015) 2NN-MEAM for Ni-Ti, fitted to the B2 <-> B19' transformation.
wf.potential = GetPotential(
    structure=wf.b2_supercell, name_filter="2015--Ko-W-S--Ni-Ti"
)

wf.lammps_structure = CreateLammpsStructure(
    structure=wf.b2_supercell,
    potential=wf.potential.outputs.potential_name,
    working_directory="niti_cooling_heating",
)

wf.md_input = CreateLammpsMDInput(
    io_bundle=wf.lammps_structure,
    calc_dataclass=wf.md_params,
    temperature_schedule=wf.temperature_schedule.outputs.temperatures,
    flexible_cell=True,
)

wf.md_run = RunLammpsCalculation(io_bundle=wf.md_input, threads_per_core=4)

wf.md_output = ParseLammpsOutput(io_bundle=wf.md_run.outputs.io_bundle)

wf.cycle_table = LammpsStageAverages(
    io_bundle=wf.md_run.outputs.io_bundle,
    md_output=wf.md_output,
    branches=wf.temperature_schedule.outputs.branches,
)

wf.martensite_structure = ColdestStructure(
    df=wf.cycle_table.outputs.df, structures=wf.cycle_table.outputs.stage_structures
)

wf.transformation_temperatures = TransformationTemperatures(
    df=wf.cycle_table.outputs.df
)

wf.hysteresis_plot = PlotCoolingHeatingCycle(
    df=wf.cycle_table.outputs.df,
    T_cooling_K=wf.transformation_temperatures.outputs.T_cooling_K,
    T_heating_K=wf.transformation_temperatures.outputs.T_heating_K,
)

wf.martensite_relaxation = Relax(
    structure=wf.martensite_structure,
    engine=wf.foundation_model,
    opt_parameters=wf.relax_settings,
    opt_mode="full",
    store=True,
)

wf.b2_reference_relaxation = Relax(
    structure=wf.b2_supercell,
    engine=wf.foundation_model,
    opt_parameters=wf.relax_settings,
    opt_mode="full",
    store=False,
)

wf.martensite_energy_gain = MartensiteEnergyGain(
    b2_relaxation=wf.b2_reference_relaxation,
    martensite_relaxation=wf.martensite_relaxation,
)
