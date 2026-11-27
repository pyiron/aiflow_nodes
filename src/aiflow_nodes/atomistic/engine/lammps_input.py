from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from aiflow.core import as_function_node
from lammpsparser.units import LAMMPS_UNIT_CONVERSIONS

from aiflow_nodes.atomistic.calculator.data import InputCalcMD, InputCalcMinimize
from aiflow_nodes.atomistic.engine.lammps import LammpsIOBundle

_TEMPLATE_DIR = Path(__file__).parent / "templates"


def _get_env():
    """Lazily build the Jinja2 environment (defers the import for optional-dep hygiene)."""
    from jinja2 import Environment, FileSystemLoader, StrictUndefined

    return Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
        undefined=StrictUndefined,
    )


_env = None


def _render(template_name: str, **context) -> str:
    """Render a section template.

    Always returns a string ending with '\\n' so that the core port's
    _coerce() method (which strips leading/trailing quotes) never truncates
    LAMMPS format strings that end with '"'.
    """
    global _env
    if _env is None:
        _env = _get_env()
    rendered = _env.get_template(template_name).render(**context)
    return rendered.rstrip("\n") + "\n"


def _fix_ids_in(text: str) -> list[str]:
    """Fix ids named in the `fix <id> ...` lines of a rendered section."""
    return [
        line.split()[1]
        for line in text.splitlines()
        if line.startswith("fix ") and len(line.split()) > 1
    ]


def _inherit(value, io_bundle: LammpsIOBundle, field: str, *calc_classes):
    """Resolve one section parameter.

    The section node's own port wins when it is set (not None). Otherwise the
    value comes from `io_bundle.calc_input`, if that is an instance of one of
    `calc_classes` (so e.g. an InputCalcMD's `pressure` never leaks into a
    minimize section). Failing that, it falls back to the class default of the
    first of `calc_classes`, so built-in defaults live only in the dataclasses.
    """
    if value is not None:
        return value
    calc = io_bundle.calc_input
    if any(isinstance(calc, cls._original_dataclass) for cls in calc_classes):
        return getattr(calc, field)
    return getattr(calc_classes[0]._original_dataclass(), field)


# fix box/relax keywords for the pressure components [xx, yy, zz, xy, xz, yz]
_BOX_RELAX_KEYWORDS = ("x", "y", "z", "xy", "xz", "yz")


def _pressure_array(pressure) -> np.ndarray | None:
    """Pressure [GPa] as a float array of 1..6 components, NaN = component
    fixed; None (unset) stays None.

    Accepts a string ("0.0", "0.0, 0.0, None", "[0, 0, None]"), a number, or
    a list/tuple/array with None entries.
    """
    if pressure is None:
        return None
    if isinstance(pressure, str):
        items = [s.strip() for s in pressure.strip("[]() ").split(",") if s.strip()]
        if not items or items == ["None"]:
            return None
        pressure = [None if s.lower() == "none" else float(s) for s in items]
    values = np.array(
        [
            np.nan if p is None else float(p)
            for p in np.atleast_1d(np.asarray(pressure, dtype=object))
        ],
        dtype=float,
    )
    if not 1 <= len(values) <= 6:
        raise ValueError(
            f"pressure needs 1..6 components [xx, yy, zz, xy, xz, yz], got {len(values)}"
        )
    if np.all(np.isnan(values)):
        raise ValueError(
            f"pressure {pressure!r} sets no component; leave it unset for a fixed cell."
        )
    return values


@as_function_node
def SetLammpsCalcInput(
    io_bundle: LammpsIOBundle,
    calc_input: InputCalcMD | InputCalcMinimize,
) -> LammpsIOBundle:
    """Attach a calculation dataclass to `io_bundle` for the section nodes to
    inherit from.

    For the first stage this can also be passed directly to
    `CreateLammpsStructure(calc_input=...)`. For a later stage of a chained
    input, attach that stage's dataclass to the bundle coming out of the
    previous `AssembleLammpsInput`, and wire the result into that stage's
    section nodes.
    """
    import copy

    io_bundle = copy.deepcopy(io_bundle)
    io_bundle.calc_input = calc_input
    return io_bundle


@as_function_node("init")
def LammpsInit(
    io_bundle: LammpsIOBundle,
    read_restart: str | None = None,
) -> str:
    """Initialization block: units, dimension, boundary, atom_style,
    read_data/read_restart.

    This is typically the first section of a LAMMPS input and is common to all
    calculation types (MD, minimize, static). Feed the result into
    `AssembleLammpsInput(init=...)`.

    Everything except `read_restart` comes from `io_bundle`: `units` and
    `atom_style` as set by `CreateLammpsStructure` (including from a DataFrame
    potential's Config), `dimension`/`boundary` from `io_bundle.structure.pbc`,
    and the data file name from `io_bundle.lammps_structure_filename`.
    """
    pbc = io_bundle.structure.pbc
    return _render(
        "init.j2",
        units=io_bundle.units,
        dim=3,  # always 3D; pbc only sets the boundary (sum(pbc) gave 1 for a F,F,T cylinder)
        boundary=" ".join("p" if p else "f" for p in pbc),
        atom_style=io_bundle.atom_style,
        read_restart=read_restart,
        structure_filename=io_bundle.lammps_structure_filename,
    )


@as_function_node(["potential", "potential_file_content"])
def LammpsPotential(
    io_bundle: LammpsIOBundle,
) -> tuple[str, str]:
    """Potential block: pair_style, pair_coeff (or `include potential.inp`).

    The potential, its resource path and the include file name all come from
    `io_bundle`.

    For DataFrame potentials the full potential text is returned as
    `potential_file_content` and the input file references it via `include`.
    For string potentials the commands are inlined and `potential_file_content`
    is empty.

    Connect `potential_file_content` to `AssembleLammpsInput` so the file gets
    written to disk before the LAMMPS run.
    """
    from lammpsparser.compatibility.file import _get_potential

    potential = io_bundle.potential
    potential_list, _, _ = _get_potential(
        potential=potential, resource_path=io_bundle.resource_path
    )

    if isinstance(potential, pd.DataFrame):
        section = _render(
            "potential_include.j2",
            potential_filename=io_bundle.lammps_potential_filename,
        )
        potential_file_content = "".join(potential_list)
    else:
        commands = "\n".join(line.rstrip() for line in potential_list if line.strip())
        section = _render("potential_inline.j2", commands=commands)
        potential_file_content = ""

    return section, potential_file_content


@as_function_node("dump")
def LammpsDump(
    io_bundle: LammpsIOBundle,
    n_print: int | None = None,
    filename: str = "dump.out",
    extra_columns: str = "",
) -> str:
    """Dump block: trajectory output configuration.

    Writes wrapped fractional coordinates (xsu ysu zsu) plus forces and
    velocities in the standard pyiron format.

    `n_print` left as None is inherited from `io_bundle.calc_input`
    (InputCalcMD or InputCalcMinimize), else the dataclass default.

    `extra_columns` (raw LAMMPS dump-custom column words, space separated,
    e.g. "c_stressperatom[*]") are appended after vz; the compute/fix they
    name must be defined (e.g. via LammpsComputes into AssembleLammpsInput's
    `compute` port, which renders before the dump). Since a wildcard's column
    count isn't known here, the format is then set with
    `format float %20.15g` instead of a per-column `format line`: the same
    output (%d for id/type, %20.15g for every float column). Empty (default)
    leaves the block unchanged. lammpsparser returns a `c_<name>[k]` column
    as `generic["<name>[k]"]`.
    """
    n_print = _inherit(n_print, io_bundle, "n_print", InputCalcMD, InputCalcMinimize)
    extra_columns = " ".join(str(extra_columns or "").split())
    return _render(
        "dump.j2", n_print=n_print, filename=filename, extra_columns=extra_columns
    )


@as_function_node("thermo")
def LammpsThermo(
    io_bundle: LammpsIOBundle,
    n_print: int | None = None,
) -> str:
    """Thermo block: output frequency, style, and format for log.lammps.

    `n_print` left as None is inherited from `io_bundle.calc_input`
    (InputCalcMD or InputCalcMinimize), else the dataclass default.
    """
    n_print = _inherit(n_print, io_bundle, "n_print", InputCalcMD, InputCalcMinimize)
    return _render("thermo.j2", n_print=n_print)


@as_function_node(["ensemble", "fix_ids"])
def LammpsEnsemble(
    io_bundle: LammpsIOBundle,
    temperature: float | None = None,
    pressure: float | None = None,
    temperature_damping: float | None = None,
    pressure_damping: float | None = None,
    langevin: bool | None = None,
    seed: int | None = None,
    ensemble: Literal["nve", "nvt", "npt"] | None = None,
) -> tuple[str, list[str]]:
    """Ensemble block: fix NVE / NVT / NPT (Nosé–Hoover or Langevin).

    Parameters are in pyiron natural units:
    - temperature [K], pressure [GPa], damping timescales [fs].
    Unit conversion to `io_bundle.units` is applied automatically.

    Every parameter left as None is inherited from an InputCalcMD attached as
    `io_bundle.calc_input` (`temperature_damping`/`pressure_damping` map to its
    `*_damping_timescale` fields), else from InputCalcMD's defaults.

    `ensemble` picks the integrator explicitly. Left as None it follows from
    the resolved values:
    - pressure set                   → NPT (isotropic)
    - temperature set, pressure None → NVT
    - both None                      → NVE
    - langevin=True                  → NVE+Langevin (NVT) or NPH+Langevin (NPT)
    Because temperature always inherits a value, NVE needs `ensemble="nve"`
    unless the attached dataclass has `temperature=None`. Likewise
    `ensemble="nvt"` ignores an inherited pressure.

    This fix stays active until a `run` executes the dynamics — potentially
    in a separate
    `AssembleLammpsInput` call. So `fix_ids` (the LAMMPS fix ID(s) this
    section creates) is returned alongside the rendered block: connect it to
    `AssembleLammpsInput`'s `calculation_fix_ids` so the matching `unfix` gets
    emitted after the `run` that uses it.
    """
    temperature = _inherit(temperature, io_bundle, "temperature", InputCalcMD)
    pressure = _inherit(pressure, io_bundle, "pressure", InputCalcMD)
    temperature_damping = _inherit(
        temperature_damping, io_bundle, "temperature_damping_timescale", InputCalcMD
    )
    pressure_damping = _inherit(
        pressure_damping, io_bundle, "pressure_damping_timescale", InputCalcMD
    )
    langevin = _inherit(langevin, io_bundle, "langevin", InputCalcMD)
    seed = _inherit(seed, io_bundle, "seed", InputCalcMD)

    if ensemble is None:
        if pressure is not None:
            ensemble = "npt"
        elif temperature is not None:
            ensemble = "nvt"
        else:
            ensemble = "nve"
    if ensemble not in ("nve", "nvt", "npt"):
        raise ValueError(
            f"Unknown ensemble: {ensemble!r} — must be 'nve', 'nvt', or 'npt'"
        )
    if ensemble != "nve" and temperature is None:
        raise ValueError(f"ensemble={ensemble!r} needs a temperature")
    if ensemble == "npt" and pressure is None:
        raise ValueError("ensemble='npt' needs a pressure")

    conversions = LAMMPS_UNIT_CONVERSIONS[io_bundle.units]
    t_damp = temperature_damping * conversions["time"]
    p_damp = pressure_damping * conversions["time"]

    t = p = None
    if ensemble == "nve":
        mode = "nve"
    elif ensemble == "npt":
        t = temperature * conversions["temperature"]
        p = pressure * conversions["pressure"]
        mode = "nph_langevin" if langevin else "npt"
    else:
        t = temperature * conversions["temperature"]
        mode = "nve_langevin" if langevin else "nvt"

    fix_ids = ["ensemble", "langevin"] if "langevin" in mode else ["ensemble"]

    ensemble = _render(
        "ensemble.j2",
        mode=mode,
        t=t,
        p=p,
        t_damp=t_damp,
        p_damp=p_damp,
        seed=seed,
    )
    return ensemble, fix_ids


@as_function_node("velocity")
def LammpsVelocity(
    io_bundle: LammpsIOBundle,
    temperature: float | None = None,
    initial_temperature: float | None = None,
    seed: int | None = None,
    time_step: float | None = None,
) -> str:
    """Velocity + timestep block.

    Sets the integration timestep and initialises velocities from a Gaussian
    distribution at `initial_temperature`, which defaults to 2×temperature
    (equipartition: half kinetic, half potential).

    Parameters: temperature [K], initial_temperature [K], time_step [fs].
    Every parameter left as None is inherited from an InputCalcMD attached as
    `io_bundle.calc_input`, else from InputCalcMD's defaults. Unit conversion
    to `io_bundle.units` is applied automatically.
    """
    temperature = _inherit(temperature, io_bundle, "temperature", InputCalcMD)
    initial_temperature = _inherit(
        initial_temperature, io_bundle, "initial_temperature", InputCalcMD
    )
    seed = _inherit(seed, io_bundle, "seed", InputCalcMD)
    time_step = _inherit(time_step, io_bundle, "time_step", InputCalcMD)
    if initial_temperature is None:
        initial_temperature = 2.0 * temperature

    conversions = LAMMPS_UNIT_CONVERSIONS[io_bundle.units]
    ts = time_step * conversions["time"]
    init_temp = initial_temperature * conversions["temperature"]

    return _render("velocity.j2", time_step=ts, init_temp=init_temp, seed=seed)


@as_function_node("minimize")
def LammpsMinimize(
    io_bundle: LammpsIOBundle,
    e_tol: float | None = None,
    f_tol: float | None = None,
    max_iter: int | None = None,
    style: str | None = None,
    pressure: str | None = None,
    min_modify: str | None = None,
    timestep: float | None = None,
    max_eval: int | None = None,
) -> str:
    """Minimization block: optional box relaxation, min_style, minimize.

    Optional extras, each omitted (None) unless set; none is inherited:
    - `min_modify`: raw `min_modify` arguments, e.g. "norm two".
    - `timestep` [fs]: written as `timestep` in `io_bundle.units` (FIRE and the
      other damped-dynamics minimizers use it), e.g. 1.0 -> `timestep 0.001` (metal).
    - `max_eval`: the 4th `minimize` argument (max force evaluations);
      defaults to `max_iter` as before.

    Parameters: e_tol [eV], f_tol [eV/Å], pressure [GPa].
    Every parameter left as None is inherited from an InputCalcMinimize
    attached as `io_bundle.calc_input`, else from InputCalcMinimize's
    defaults. Unit conversion to `io_bundle.units` is applied automatically.

    `pressure` is a string, converted to an array first (`_pressure_array`):
    - one value, e.g. "0.0": `fix ensemble all box/relax iso 0.0`
    - 2..6 comma-separated components [xx, yy, zz, xy, xz, yz], `None` keeping
      that component fixed, e.g. "0.0, 0.0, None" (x, y relax to 0 GPa, z fixed):
      `fix ensemble all box/relax x 0.0 y 0.0 couple none`. Shear components
      need a triclinic (non-orthogonal) cell.
    A float or list (e.g. an inherited InputCalcMinimize.pressure) is accepted
    the same way.

    When a box relaxation is set, `fix ensemble ... box/relax` goes before the
    minimize command. The block holds no `unfix`: AssembleLammpsInput (with
    `mode="minimize"`) reads the fix id from the block and writes
    `unfix ensemble` after the stage's `run N` line, or right after the
    minimize block when the stage has no `run`.
    """
    pressure = _pressure_array(pressure)
    e_tol = _inherit(e_tol, io_bundle, "e_tol", InputCalcMinimize)
    f_tol = _inherit(f_tol, io_bundle, "f_tol", InputCalcMinimize)
    max_iter = _inherit(max_iter, io_bundle, "max_iter", InputCalcMinimize)
    style = _inherit(style, io_bundle, "style", InputCalcMinimize)
    if pressure is None:
        pressure = _pressure_array(
            _inherit(None, io_bundle, "pressure", InputCalcMinimize)
        )

    box_relax = []
    couple_none = False
    if pressure is not None:
        pressure = pressure * LAMMPS_UNIT_CONVERSIONS[io_bundle.units]["pressure"]
        if len(pressure) == 1:
            box_relax = [("iso", float(pressure[0]))]
        else:
            box_relax = [
                (keyword, float(p))
                for keyword, p in zip(_BOX_RELAX_KEYWORDS, pressure)
                if not np.isnan(p)
            ]
            couple_none = True
            shear = [k for k, _ in box_relax if k in ("xy", "xz", "yz")]
            if shear and io_bundle.structure is not None:
                cell = np.array(io_bundle.structure.cell)
                if np.allclose(cell - np.diag(np.diag(cell)), 0.0):
                    raise ValueError(
                        f"box/relax {'/'.join(shear)} needs a triclinic box in "
                        "LAMMPS; the structure's cell is orthogonal."
                    )

    min_modify = str(min_modify).strip() if min_modify is not None else ""
    if timestep is not None:
        timestep = float(timestep) * LAMMPS_UNIT_CONVERSIONS[io_bundle.units]["time"]
    max_eval = max_iter if max_eval is None else int(max_eval)

    return _render(
        "minimize.j2",
        box_relax=box_relax,
        couple_none=couple_none,
        style=style,
        e_tol=e_tol,
        f_tol=f_tol,
        max_iter=max_iter,
        max_eval=max_eval,
        min_modify=min_modify,
        timestep=timestep,
    )


@as_function_node(["fix", "fix_ids"])
def LammpsFix(
    io_bundle: LammpsIOBundle,
    fix_id: str = "constraintxy",
    group: str = "all",
    style: str = "setforce",
    args: str = "0.0 0.0 NULL",
):
    """A single LAMMPS fix: `fix <fix_id> <group> <style> <args>`.

    Generic, for fixes no other section node sets up, e.g. the default
    `fix constraintxy all setforce 0.0 0.0 NULL` (atoms move only along z, as
    in a rigid-shift GSFE minimization). `args` is the raw LAMMPS argument
    string, written as given: no unit conversion, so it must already be in
    `io_bundle.units`. Leave `args` empty for a fix style that takes none.

    Returns the rendered line and `fix_ids` (`[fix_id]`), like LammpsEnsemble,
    so the fix can be unfixed after the run/minimize that uses it.
    """
    fix_id = str(fix_id).strip()
    group = str(group).strip()
    style = str(style).strip()
    args = "" if args is None else str(args).strip()
    for name, value in (("fix_id", fix_id), ("group", group), ("style", style)):
        if not value or any(c.isspace() for c in value):
            raise ValueError(
                f"LammpsFix {name} must be a single non-empty word, got {value!r}"
            )

    fix = _render("fix.j2", fix_id=fix_id, group=group, style=style, args=args)
    fix_ids = [fix_id]
    return fix, fix_ids


@as_function_node
def AssembleLammpsInput(
    io_bundle: LammpsIOBundle,
    init: str = "",
    potential: str = "",
    potential_file_content: str = "",
    neighbor: str = "",
    region: str = "",
    group: str = "",
    compute: str = "",
    dump: str = "",
    thermo: str = "",
    calculation: str = "",
    calculation_fix_ids: list[str] = [],
    velocity: str = "",
    fix: str = "",
    fix_ids: list[str] = [],
    run: int | None = None,
    write_restart: str | None = None,
    mode: str | None = None,
) -> LammpsIOBundle:
    """Assemble named sections into a LAMMPS input, in the fixed order LAMMPS
    requires (init, potential, region, group, compute, dump, thermo, velocity, fix,
    calculation, run), and inject/append it into `io_bundle` — mirroring how
    `CreateLammpsMDInput` builds `lammps_input_string` and returns the updated
    bundle. No separate bridge node is needed to reach `RunLammpsCalculation`.

    `neighbor` takes neighbor-list settings (`neighbor`/`neigh_modify` lines,
    e.g. from LammpsNeighbor), rendered right after the potential.

    `region`/`group` take region and group lines (e.g. a frozen ring:
    `region ring cylinder ...` and `group ring region ring`). They come after
    read_data/pair_style and before any dump/fix that uses the group. Unlike
    `init`/`potential` they are allowed on later stages too, since regions
    and groups persist in LAMMPS; a group's membership is set when its line
    runs.

    `compute` takes compute lines (e.g. from LammpsComputes, several computes
    in one block), rendered after `group` and before `dump`/`thermo`, so a
    dump or thermo can use `c_<id>`. Allowed on later stages too.

    `fix`/`fix_ids` take LammpsFix's (or LammpsFixes') outputs (an extra fix, e.g. setforce).
    The fix is rendered just before `calculation`. Its ids join the pending
    fix ids and are unfixed after the `run` (this stage's or a later one's),
    like LammpsEnsemble's. Only with `mode="minimize"` (io_bundle.mode) and no
    `run` in this stage is it unfixed right after the calculation block, since
    minimize runs by itself and no `run` follows. An empty `fix_ids`
    falls back to the ids named in the `fix` lines.

    Chainable across multiple simulation stages: call this node again on the
    same `io_bundle` to append a further stage (e.g. an NVT equilibration
    followed by an NPT production run). The first call (`io_bundle`'s
    `lammps_input_string` still empty) is where `init`/`potential` apply;
    later calls ignore them (with a printed note) since read_data/pair_style
    belong to the first stage only — just leave those two unconnected for
    stage 2+.

    `calculation` is a single slot fed by *either* `LammpsEnsemble`'s or
    `LammpsMinimize`'s output — the two are mutually exclusive ways to set up
    what the run actually does (an MD ensemble fix, or a box/relax fix plus
    `minimize`), so they share one port rather than needing two.
    `LammpsMinimize`'s box/relax fix (if any) needs no wiring: with
    `mode="minimize"` its id is read from the block's `fix` line, and it is
    unfixed after this stage's `run` (or right after the minimize block when
    there is no `run`), together with any LammpsFix fix. `LammpsEnsemble`'s fix(es) stay active until a `run` actually executes the
    dynamics, so its second output, `fix_ids`, should be connected to
    `calculation_fix_ids` — this node remembers them on `io_bundle` and emits
    the matching `unfix` line(s) right after the `run` (and any
    `write_restart`) that uses them, even if that `run` arrives in a later
    call than the one that set up `calculation`.

    Omit a string section (leave it as the default empty string) to skip it,
    e.g. leave `calculation`/`velocity` empty for a static run.

    `run` is the number of MD steps to execute (0 for a static single-point
    evaluation), rendered directly here rather than via a separate node since
    every input string is already a fully-rendered LAMMPS section. Leave
    `run=None` to omit the run command for this stage — but an ensemble
    `calculation` (one with `calculation_fix_ids`) left without a `run` is
    only valid as the *last* stage: if a later call supplies another
    `calculation` before this one ever got a `run` (and its `unfix`), that fix
    was set up and then abandoned, so this raises a `ValueError`.

    `write_restart` does *not* require `run` to be set — leaving `run=None`
    while supplying `write_restart` still emits `write_restart <filename>`
    (with no preceding `run` line), so a restart snapshot can be requested for
    a stage that has no MD steps of its own (e.g. right after a minimize).
    Pending ensemble fixes are only `unfix`ed after an actual `run`: a
    `write_restart` without one leaves them pending for a later stage's `run`.

    `potential_file_content` (from `LammpsPotential`'s second output) is
    written to `io_bundle.lammps_potential_string` when non-empty — needed
    for DataFrame potentials that produce an `include potential.inp` line.

    `mode` ('static', 'md', or 'minimize') sets `io_bundle.mode`, which
    `ParseLammpsOutput` requires to know how to interpret the dump/log files
    (it refuses to run with `mode=None`). The old monolithic
    `CreateLammpsMDInput`/`CreateLammpsStaticInput`/`CreateLammpsMinimizeInput`
    set this implicitly since each one only ever produces one mode; the
    modular section nodes (`LammpsMinimize`, `LammpsEnsemble`, ...) don't
    know what the other sections are doing, so the workflow author states it
    explicitly here — pass `mode="minimize"` alongside a `LammpsMinimize`
    `calculation`, or `mode="md"`/`"static"` alongside a `LammpsEnsemble` one
    (static = `run=0`). Leaving it `None` reproduces the previous behaviour
    (`ParseLammpsOutput` raises until something sets `io_bundle.mode`).
    Like `init`/`potential`, only meaningful confusion would come from
    changing it mid-chain, so later stages may simply repeat the same value.
    """
    import copy

    io_bundle = copy.deepcopy(io_bundle)
    is_first_stage = not io_bundle.lammps_input_string

    if mode is not None:
        if mode not in ("static", "md", "minimize"):
            raise ValueError(
                f"Unknown mode: {mode!r} — must be 'static', 'md', or 'minimize'"
            )
        io_bundle.mode = mode

    if calculation and io_bundle.lammps_pending_fix_ids:
        raise ValueError(
            "The previous `calculation` section's fix(es) "
            f"({io_bundle.lammps_pending_fix_ids}) were never `unfix`ed via a "
            "`run` before this new stage — add a `run` to that stage before "
            "adding another `calculation`."
        )

    if not is_first_stage and (init or potential):
        print(
            "AssembleLammpsInput: ignoring `init`/`potential` — they only "
            "apply to the first stage of an io_bundle; later stages reuse "
            "the existing read_data/pair_style setup."
        )
        init = ""
        potential = ""

    # Fix ids introduced by this stage: the calculation's (LammpsEnsemble's
    # wired `calculation_fix_ids`, or in minimize mode LammpsMinimize's
    # box/relax fix, read from its `fix` lines) and the extra fix's
    # (LammpsFix; an unwired `fix_ids` port falls back to its `fix` lines).
    # A minimize stage: mode="minimize", or a calculation block holding a
    # `minimize` command (LammpsMinimize's output, also when `mode` is unset).
    is_minimize = io_bundle.mode == "minimize" or any(
        line.startswith("minimize ") for line in calculation.splitlines()
    )
    pending = [i for i in io_bundle.lammps_pending_fix_ids.split(",") if i]
    new_ids = []
    if calculation:
        # any earlier pending ids raised above, so this stage starts clean
        pending = []
        calc_ids = list(calculation_fix_ids)
        if not calc_ids and is_minimize:
            calc_ids = _fix_ids_in(calculation)
        new_ids += calc_ids
    if fix:
        new_ids += list(fix_ids) or _fix_ids_in(fix)
    fix_unfix = ""
    if is_minimize and run is None:
        # minimize is self-executing and no `run` follows in this stage:
        # unfix right after the calculation block.
        if new_ids:
            fix_unfix = _render("unfix.j2", fix_ids=new_ids)
    else:
        # unfixed after this stage's `run`, or a later stage's
        pending += new_ids
    io_bundle.lammps_pending_fix_ids = ",".join(pending)

    render_run_section = run is not None or write_restart
    # Pending fixes are only unfixed after a `run` that actually used them: a
    # write_restart-only section leaves them pending for a later stage's run.
    unfix_pending = run is not None and bool(io_bundle.lammps_pending_fix_ids)
    run_section = (
        _render(
            "run.j2",
            n_steps=run,
            write_restart=write_restart,
            fix_ids=(
                io_bundle.lammps_pending_fix_ids.split(",") if unfix_pending else []
            ),
        )
        if render_run_section
        else ""
    )
    new_block = _render(
        "master.j2",
        init=init,
        potential=potential,
        neighbor=neighbor,
        region=region,
        group=group,
        compute=compute,
        dump=dump,
        thermo=thermo,
        calculation=calculation,
        velocity=velocity,
        fix=fix,
        fix_unfix=fix_unfix,
        run=run_section,
    )
    if is_first_stage:
        io_bundle.lammps_input_string = new_block
    else:
        io_bundle.lammps_input_string += "\n" + new_block

    if run is not None:
        io_bundle.lammps_pending_fix_ids = ""

    if potential_file_content:
        io_bundle.lammps_potential_string = potential_file_content
    return io_bundle


@as_function_node
def SetLammpsInputString(
    io_bundle: LammpsIOBundle,
    lammps_input_string: str,
    potential_file_content: str = "",
) -> LammpsIOBundle:
    """Bridge: inject an assembled input string into an existing LammpsIOBundle.

    Connects `AssembleLammpsInput`'s output to the existing RunLammpsCalculation
    pipeline. Optionally also sets the potential file content (needed when using
    DataFrame potentials that produce an `include potential.inp` line).
    """
    import copy

    io_bundle = copy.deepcopy(io_bundle)
    io_bundle.lammps_input_string = lammps_input_string
    if potential_file_content:
        io_bundle.lammps_potential_string = potential_file_content
    return io_bundle


@as_function_node("path")
def WriteFile(io_bundle: LammpsIOBundle, filename: str = "output.txt") -> str:
    """Write `io_bundle.lammps_input_string` to a file and return the file path."""
    with open(filename, "w") as f:
        f.write(io_bundle.lammps_input_string)
    path = filename
    return path
