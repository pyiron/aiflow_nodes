"""Nodes for K-controlled crack loading at 0 K (Onur's `wcrack`,
/cmmc/u/sanand/pyiron/projects/18_Onur/W_crack_K0).

Staging module. Thin nodes that call the `wcrack` package (option a: reuse,
not port). The whole K schedule runs as ONE LAMMPS input, as wcrack's CLI route
does (`python -m wcrack run`): every step is `displace_atoms` by dK*U(X0) +
`minimize` with the outer ring held by `setforce 0 0 0`, and the early stop is
tested inside LAMMPS (`if "$(c_TIPR) > d" then ... "jump SELF wcrack_end"`).
So the workflow has no loop node: one case = one LAMMPS run = one SLURM job.

Nodes:
    CrackConfig         - wcrack configuration for one case (the production
                          notebook's `build_cfg`: DEFAULTS + expand_cases + validate)
    BuildCrackSpecimen  - specimen.build + loading.schedule (writes specimen.data,
                          reference.npz, geometry.json); never loading.build_case,
                          which can shutil.rmtree the case folder
    CrackLoadingInput   - lammps_io.loading_header + loading_steps for the full
                          schedule, as a LammpsIOBundle for the existing RunLammpsBase
    AnalyzeCrackCase    - analysis.analyze_case (steps table, K_c bracket, event,
                          figures in <case>/analysis)
    CrackCase           - group: the four above + RunLammpsBase, one case end to end

BuildCrackSpecimen deconstructed (see the "Specimen from separate nodes"
section): MisorientedAxes, CylinderRepeats, OrientedCubicCell, CenterXY,
CrackOriginShift, CutCylinder, CutNotch, StrohKField, CrackSpecimenFiles, plus
SetPBC. The fixed outer ring is still pending (to be studied), so CrackCase
still uses BuildCrackSpecimen.

Nothing here deletes files. Where wcrack's CLI would delete or overwrite an
earlier history (it unlinks progress.txt on a fresh run), these nodes raise
instead: use a new run_dir or case name.

`wcrack` is imported inside each node body from `wcrack_dir` (a port), so the
module imports without it and SLURM workers find it without a PYTHONPATH entry.
"""

from typing import Literal

from aiflow.core import as_function_node, group_node

WCRACK_DIR = "/cmmc/u/sanand/pyiron/projects/18_Onur/W_crack_K0"


def _wcrack(wcrack_dir: str):
    import sys

    wcrack_dir = str(wcrack_dir)
    if wcrack_dir not in sys.path:
        sys.path.insert(0, wcrack_dir)
    import wcrack

    return wcrack


def _floats(text, n=None, name="value") -> list:
    values = [
        float(v)
        for v in str(text).replace("[", "").replace("]", "").split(",")
        if v.strip()
    ]
    if n is not None and len(values) != n:
        raise ValueError(f"{name} needs {n} comma-separated numbers, got {text!r}")
    return values


def _ints(text, n=3, name="value") -> list:
    values = [
        int(v)
        for v in str(text).replace("[", "").replace("]", "").split(",")
        if v.strip()
    ]
    if len(values) != n:
        raise ValueError(f"{name} needs {n} comma-separated integers, got {text!r}")
    return values


def _refs(cfg) -> dict:
    r = cfg["references"]
    return {
        "a0_A": float(r["lattice_parameter_A"]),
        "C_GPa": [float(c) for c in r["elastic_constants_GPa"]],
        "surface_energy_J_m2": {},
    }


# --------------------------------------------------------------------------- #
#  Configuration                                                               #
# --------------------------------------------------------------------------- #


@as_function_node(["cfg", "case"])
def CrackConfig(
    run_dir: str,
    case_name: str = "c110_001",
    # material: bcc only; a0 and C must be the potential's own 0 K values
    element: str = "W",
    mass: float = 183.84,
    lattice_constant: float = 3.165,
    elastic_constants: str = "525.2, 192.5, 177.5",
    # potential (MEAM: library file, its elements, parameter file)
    meam_library: str = f"{WCRACK_DIR}/potentials/WP.library.meam",
    meam_parameters: str = f"{WCRACK_DIR}/potentials/WP.parameter.meam",
    library_elements: str = "W, P",
    cutoff: float = 5.8,
    # geometry
    plane: str = "1, 1, 0",
    front: str = "0, 0, 1",
    rotation_deg: float = 0.0,
    radius_nm: float = 4.0,
    fixed_ring_nm: float = 1.2,
    front_length_nm: float = 1.2,
    crack: Literal["slit", "notch"] = "slit",
    notch_radius_A: float = 0.0,
    # loading
    mode: str = "1, 0, 0",
    K_start: float | None = 1.70,
    K_step: float = 0.01,
    K_stop: float = 1.80,
    K_values: str = "",
    surface_energy_J_m2: float = 0.0,
    predictor: Literal["all", "boundary"] = "all",
    early_stop: bool = True,
    # minimiser
    min_style: Literal["fire", "cg", "sd", "quickmin"] = "fire",
    ftol: float = 1e-6,
    min_norm: Literal["two", "max", "inf"] = "two",
    maxiter: int = 100000,
    wcrack_dir: str = WCRACK_DIR,
):
    """wcrack configuration with one case (production notebook `build_cfg`:
    wcrack DEFAULTS merged with these inputs, then expand_cases + validate).

    Vectors are comma-separated strings (`plane="1, 1, 0"`), so every port is a
    plain value that survives a GUI save or a sweep pickle. Case files go to
    `{run_dir}/cases/{case_name}`.

    K schedule: `K_start..K_stop` in steps of `K_step` [MPa m^1/2], or an
    explicit `K_values` list ("1.7, 1.75, 1.8") that overrides the ramp.
    `K_start=None` means wcrack's "auto" (0.9 x Griffith K), which needs
    `surface_energy_J_m2 > 0`. `mode` = weights (I, II, III).

    `min_norm` defaults to "two" as in the production notebook (wcrack's own
    default is "inf"). `early_stop=False` runs the whole schedule.

    Returns the full `cfg` dict and its single validated `case` dict.
    """
    wc = _wcrack(wcrack_dir)
    from wcrack import config

    cfg = {k: config._merge(v, {}) for k, v in config.DEFAULTS.items()}
    cfg["run"].update(name=str(case_name), dir=str(run_dir))
    cfg["potential"].update(
        library=str(meam_library),
        parameters=str(meam_parameters),
        library_elements=[
            e.strip() for e in str(library_elements).split(",") if e.strip()
        ],
        element=str(element),
        mass=float(mass),
        cutoff_A=float(cutoff),
    )
    cfg["references"].update(
        lattice_parameter_A=float(lattice_constant),
        elastic_constants_GPa=_floats(elastic_constants, 3, "elastic_constants"),
        surface_energy=False,
    )
    cfg["defaults"].update(
        ftol=float(ftol),
        min_norm=str(min_norm),
        maxiter=int(maxiter),
        min_style=str(min_style),
    )
    if not early_stop:
        cfg["defaults"]["stop_tip_distance_A"] = "never"
    cfg["_path"] = f"aiflow_nodes.CrackNodes (wcrack {wc.__version__})"
    cfg["_root"] = str(run_dir)

    raw = {
        "name": str(case_name),
        "plane": _ints(plane, 3, "plane"),
        "front": _ints(front, 3, "front"),
        "rotation_deg": float(rotation_deg),
        "radius_nm": float(radius_nm),
        "fixed_ring_nm": float(fixed_ring_nm),
        "front_length_nm": float(front_length_nm),
        "crack": str(crack),
        "notch_radius_A": float(notch_radius_A),
        "mode": _floats(mode, 3, "mode"),
        "K_start": "auto" if K_start is None else float(K_start),
        "K_step": float(K_step),
        "K_stop": float(K_stop),
        "K_values": _floats(K_values) if str(K_values).strip() else [],
        "surface_energy_J_m2": float(surface_energy_J_m2),
        "predictor": str(predictor),
    }
    cfg["cases"] = config.expand_cases(cfg, [raw])
    if len(cfg["cases"]) != 1:
        raise ValueError(f"expected one case, got {len(cfg['cases'])}")
    config.validate(cfg)
    case = cfg["cases"][0]
    return cfg, case


# --------------------------------------------------------------------------- #
#  Specimen                                                                    #
# --------------------------------------------------------------------------- #


@as_function_node("meta")
def BuildCrackSpecimen(cfg: dict, case: dict, wcrack_dir: str = WCRACK_DIR) -> dict:
    """Cracked cylinder for `case` (wcrack specimen.build): atoms within radius
    R around the front (z periodic), slit/notch on y = 0, x < 0, outer ring of
    width fixed_ring held on the K-field, unit Stroh field U(X0) per atom.

    Writes `specimen.data` (with the X0/U/FIXED Properties section),
    `reference.npz` and `geometry.json` into `{run_dir}/cases/{name}` and
    returns the geometry metadata with the K schedule
    (`meta['K_schedule_MPa_sqrtm']`).

    Raises instead of rebuilding when the case folder already holds loading
    output (`progress.txt`): the specimen of a started run is never replaced.
    """
    import json

    _wcrack(wcrack_dir)
    from wcrack import loading, specimen

    d = loading.case_dir(cfg, case)
    if (d / "progress.txt").exists():
        raise RuntimeError(
            f"{case['name']}: {d} already holds loading output (progress.txt); "
            "use a new run_dir or case_name (nothing is deleted)."
        )
    refs = _refs(cfg)
    meta = specimen.build(cfg, case, refs, d)
    meta["K_schedule_MPa_sqrtm"] = loading.schedule(
        case, meta["K_griffith_along_mode_MPa_sqrtm"]
    )
    meta["signature"] = loading.signature(case, refs)
    (d / "geometry.json").write_text(json.dumps(meta, indent=1))
    return meta


# --------------------------------------------------------------------------- #
#  Specimen from separate nodes (BuildCrackSpecimen deconstructed)             #
# --------------------------------------------------------------------------- #
#
#  MisorientedAxes (integer x, y, z for a misorientation phi/theta/zeta)
#  -> CylinderRepeats (size from the axes, before building) -> OrientedCubicCell(size)
#  -> CenterXY -> CrackOriginShift -> CutCylinder -> CutNotch
#  -> [fixed ring: pending, an ASE FixAtoms constraint]
#  -> StrohKField -> CrackSpecimenFiles
#
#  All three axes are rational, so the block is exactly periodic and nothing is
#  rotated afterwards; the cylinder cut removes x/y periodicity. SetPBC is
#  available for blocks rotated with RotateAxisAngle, but not used here.
#
#  CrackSpecimenFiles writes the same three files as wcrack's specimen.build
#  (specimen.data with the X0/U/FIXED Properties, reference.npz, geometry.json),
#  so CrackLoadingInput and AnalyzeCrackCase work unchanged.

_CUBIC_BUILDERS = {
    "bcc": "BodyCenteredCubic",
    "fcc": "FaceCenteredCubic",
    "sc": "SimpleCubic",
    "diamond": "Diamond",
}
_ATOMS_PER_CUBE = {"bcc": 2, "fcc": 4, "sc": 1, "diamond": 8}


def _lattice_period(v, lattice_constant: float, crystalstructure: str = "bcc") -> float:
    """Shortest lattice translation along the integer direction `v` (the cell
    edge ASE's ase.lattice.cubic gives that direction).

    v is first reduced to coprime integers. sc: a|v|. bcc: a|v|/2 if all
    components are odd (v/2 is a body-centring vector), else a|v|. fcc and
    diamond (fcc Bravais lattice): a|v|/2 if the component sum is even, else a|v|.
    """
    import numpy as np

    v = _reduce(v)
    length = float(lattice_constant) * float(np.linalg.norm(v))
    if crystalstructure == "bcc":
        half = all(c % 2 for c in v)
    elif crystalstructure in ("fcc", "diamond"):
        half = sum(v) % 2 == 0
    elif crystalstructure == "sc":
        half = False
    else:
        raise ValueError(f"unknown crystalstructure {crystalstructure!r}")
    return 0.5 * length if half else length


@as_function_node("structure")
def OrientedCubicCell(
    element: str,
    lattice_constant: float,
    crystalstructure: Literal["bcc", "fcc", "sc", "diamond"] = "bcc",
    x: str = "1, -1, 0",
    y: str = "1, 1, 0",
    z: str = "0, 0, 1",
    size: str = "1, 1, 1",
):
    """Orthogonal periodic cubic cell whose cell axes 0, 1, 2 lie along the
    lattice directions `x`, `y`, `z` (comma-separated Miller indices).

    The three directions must be mutually orthogonal and right-handed
    (x . (y cross z) > 0). Each cell edge is the lattice period along its
    direction (ASE `ase.lattice.cubic`), and `size` is "nx, ny, nz" repeats.
    Generalises FCCOrientedCell (fixed fcc families, z only) to any cubic
    lattice and all three axes.
    """
    import ase.lattice.cubic as cubic
    import numpy as np

    dirs = [_ints(x, 3, "x"), _ints(y, 3, "y"), _ints(z, 3, "z")]
    v = np.array(dirs, float)
    for i, j in ((0, 1), (0, 2), (1, 2)):
        if abs(np.dot(v[i], v[j])) > 1e-12:
            raise ValueError(f"axes {dirs[i]} and {dirs[j]} are not orthogonal")
    if np.dot(v[0], np.cross(v[1], v[2])) <= 0:
        raise ValueError(f"axes {dirs} are not right-handed (x . (y x z) <= 0)")
    builder = getattr(cubic, _CUBIC_BUILDERS[crystalstructure])
    structure = builder(
        directions=dirs,
        size=tuple(_ints(size, 3, "size")),
        symbol=str(element),
        latticeconstant=float(lattice_constant),
        pbc=(1, 1, 1),
    )
    return structure


def _bools(text, name="pbc") -> list:
    """ "True, True, False" / "T T F" / "1, 1, 0" / a single value for all three axes."""
    words = (
        str(text).replace("[", "").replace("]", "").replace("(", "").replace(")", "")
    )
    words = [w for w in words.replace(",", " ").split() if w]
    table = {
        "true": True,
        "t": True,
        "1": True,
        "yes": True,
        "false": False,
        "f": False,
        "0": False,
        "no": False,
    }
    try:
        values = [table[w.lower()] for w in words]
    except KeyError as exc:
        raise ValueError(f"{name}: cannot read {text!r} as booleans") from exc
    if len(values) == 1:
        values = values * 3
    if len(values) != 3:
        raise ValueError(f"{name} needs 1 or 3 values, got {text!r}")
    return values


@as_function_node("structure")
def SetPBC(structure, pbc: str = "True, True, True", wrap: bool = False):
    """Set the periodic boundary conditions of a structure; the cell is unchanged.

    `pbc` is one value for all axes or three ("True, True, False", "T T F",
    "1, 1, 0"), for cell vectors 0, 1, 2.

    `wrap=True` first wraps the atoms into the cell along every direction that
    is periodic *before* the change (ASE `Atoms.wrap` with the old pbc), and
    only then sets the new pbc. So a block that was built periodic and then
    rotated (atoms only, e.g. RotateAxisAngle) has its periodic images folded
    into the cell before those directions become non-periodic. With
    `wrap=False` the positions are left exactly as they are.
    """
    structure = structure.copy()
    if wrap:
        structure.wrap(pbc=structure.get_pbc())
    structure.set_pbc(_bools(pbc))
    return structure


_AXES = ("x", "y", "z")


def _reduce(v):
    import math

    g = 0
    for c in v:
        g = math.gcd(g, int(abs(c)))
    return [int(c) // g for c in v] if g else [int(c) for c in v]


def _pick_direction(candidates, target, tolerance_deg):
    """Lowest-index candidate within `tolerance_deg` of `target`, else the closest one.
    Ties go to the smaller index sum, then lexicographically."""
    import numpy as np

    cn = candidates / np.linalg.norm(candidates, axis=1)[:, None]
    ang = np.degrees(np.arccos(np.clip(cn @ target, -1.0, 1.0)))
    ok = ang <= tolerance_deg + 1e-9
    pool = np.flatnonzero(ok) if ok.any() else np.flatnonzero(ang <= ang.min() + 1e-9)
    index_sum = np.abs(candidates[pool]).sum(axis=1)
    best = min(
        pool[index_sum == index_sum.min()].tolist(),
        key=lambda k: [-c for c in candidates[k]],
    )
    return candidates[best].astype(int).tolist(), float(ang[best])


def _warn_tilted_front(z_actual, z_ideal, lattice_constant, crystalstructure, angles):
    """Warn when the front is no longer the ideal one (phi/theta tilted it): the
    specimen is periodic along the front only in multiples of its lattice period."""
    import warnings

    import numpy as np

    za, zi = _reduce(z_actual), _reduce(z_ideal)
    if np.allclose(np.cross(za, zi), 0):
        return
    p_new = _lattice_period(za, lattice_constant, crystalstructure)
    p_old = _lattice_period(zi, lattice_constant, crystalstructure)
    in_a = float(lattice_constant) == 1.0
    unit = "a" if in_a else "A"
    note = " Lengths in units of a; set lattice_constant for Angstrom." if in_a else ""
    phi_a, theta_a, zeta_a = angles
    warnings.warn(
        f"MisorientedAxes: the front is tilted from {zi} to {za} (actual phi {phi_a:+.3f}, theta {theta_a:+.3f}, "
        f"zeta {zeta_a:+.3f} deg). To be periodic along the front, the specimen thickness must be a multiple of "
        f"{p_new:.2f} {unit}: minimum Lz = {p_new:.2f} {unit} (nz = 1), then {2 * p_new:.2f}, {3 * p_new:.2f}, ... "
        f"(the ideal front {zi} has a period of {p_old:.2f} {unit}).{note}",
        UserWarning,
        stacklevel=2,
    )


@as_function_node(
    [
        "x",
        "y",
        "z",
        "phi_actual",
        "theta_actual",
        "zeta_actual",
        "deviation_deg",
        "periods_A",
        "plane_wcrack",
        "rotation_deg_wcrack",
    ]
)
def MisorientedAxes(
    plane: str = "1, 1, 0",
    front: str = "0, 0, 1",
    phi: float = 0.0,
    theta: float = 0.0,
    zeta: float = 0.0,
    mode: Literal["rational", "rational_front"] = "rational",
    priority: Literal["x", "y", "z"] = "z",
    secondary: Literal["x", "y", "z"] | None = None,
    max_index: int = 30,
    tolerance_deg: float = 0.5,
    crystalstructure: Literal["bcc", "fcc", "sc", "diamond"] = "bcc",
    lattice_constant: float = 1.0,
):
    """Integer lattice directions for lab axes x, y, z of a crack frame
    misoriented by (phi, theta, zeta) [deg], exactly orthogonal and
    right-handed, so the cell built from them (OrientedCubicCell) is periodic
    along every axis.

    Ideal frame: y = `plane` normal, z = `front`, x = y cross z. The target
    crystal is rotated by phi about lab x, then theta about lab y, then zeta
    about lab z (extrinsic; R = Rz(zeta) Ry(theta) Rx(phi)). zeta alone is
    wcrack's `rotation_deg` (rotation of the crystal about the front).

    The target axes are generally irrational, so they are replaced by integer
    directions with components up to `max_index`:
    1. `priority` axis: the lowest-index direction within `tolerance_deg` of
       its target (the closest one if none is that close);
    2. `secondary` axis (default: y if priority is z or x, z if priority is y):
       the same rule among the directions exactly perpendicular to the first;
    3. the third axis: their cross product (x = y cross z, y = z cross x,
       z = x cross y), reduced to the smallest integers.

    Returns the axes as strings ("1, -1, 0", for OrientedCubicCell /
    StrohKField / CrackConfig plane = y, front = z), the actual angles
    phi/theta/zeta of the integer frame in the same convention, and
    `deviation_deg`: the angle between each actual axis and its target
    ({"x": .., "y": .., "z": ..}), and `periods_A`: the lattice period along each
    axis for `crystalstructure` and `lattice_constant` (the default 1.0 gives
    them in units of a). High indices give long periods: the z period is the
    shortest possible front length, and CylinderRepeats checks the block size.

    The third axis (a cross product) can come out with components above
    `max_index`; that raises, so every axis respects the limit.

    mode="rational_front" (for cylinders, built with OrientedLatticeBlock):
    only the front z is made rational (lowest-index direction within
    `tolerance_deg` of the rotated front; `priority`/`secondary` are not used).
    x and y are the exact target directions, y projected perpendicular to the
    actual z and x = y cross z, as real-valued strings. So zeta and the in-plane
    part of phi/theta are met exactly; only the front is rounded. `periods_A`
    then has None for x and y (no lattice period along an irrational direction).

    In both modes a UserWarning is raised when phi/theta tilt the front away
    from `front`: it gives the new front and the minimum periodic thickness
    (one lattice period along it; thicknesses must be multiples of it). zeta
    alone keeps the front and does not warn.

    In both modes `plane_wcrack` and `rotation_deg_wcrack` describe the same
    frame the way wcrack does (CrackConfig plane, front = z, rotation_deg): an
    integer plane normal perpendicular to z (the y axis itself in "rational"
    mode, with rotation 0; else the lowest-index one) and the rotation about the
    front that takes it to the actual y.
    """
    import numpy as np
    from scipy.spatial.transform import Rotation

    y0 = np.array(_reduce(_ints(plane, 3, "plane")), float)
    z0 = np.array(_reduce(_ints(front, 3, "front")), float)
    if abs(np.dot(y0, z0)) > 1e-12:
        raise ValueError(
            f"plane normal {plane!r} is not perpendicular to the front {front!r}"
        )
    x0 = np.cross(y0, z0)
    R0 = np.array([x0, y0, z0])
    R0 /= np.linalg.norm(R0, axis=1)[:, None]

    Rm = Rotation.from_euler("xyz", [phi, theta, zeta], degrees=True).as_matrix()
    targets = dict(
        zip(_AXES, Rm @ R0)
    )  # rows: crystal directions of the rotated lab axes

    n = int(max_index)
    r = np.arange(-n, n + 1)
    grid = np.stack(np.meshgrid(r, r, r, indexing="ij"), -1).reshape(-1, 3)
    grid = grid[np.any(grid != 0, axis=1)]
    gcd = np.gcd.reduce(np.abs(grid), axis=1)
    grid = grid[gcd == 1]  # one primitive vector per direction

    if mode == "rational_front":
        zv, _ = _pick_direction(grid, targets["z"], tolerance_deg)
        zu = np.array(zv, float) / np.linalg.norm(zv)
        yu = targets["y"] - (targets["y"] @ zu) * zu
        yu /= np.linalg.norm(yu)
        xu = np.cross(yu, zu)
        A = np.array([xu, yu, zu])
        # wcrack form: lowest-index integer plane normal perpendicular to z, then rotation about z to y
        perp = grid[grid @ np.array(zv) == 0]
        y0 = perp[
            np.lexsort(
                (-perp[:, 2], -perp[:, 1], -perp[:, 0], np.abs(perp).sum(axis=1))
            )[0]
        ].tolist()
        y0u = np.array(y0, float) / np.linalg.norm(y0)
        x0u = np.cross(y0u, zu)
        rotation_deg_wcrack = float(np.degrees(np.arctan2(yu @ x0u, yu @ y0u)))
        plane_wcrack = ", ".join(str(c) for c in y0)
        phi_actual, theta_actual, zeta_actual = (
            float(v)
            for v in Rotation.from_matrix(A @ R0.T).as_euler("xyz", degrees=True)
        )
        deviation_deg = {
            a: float(np.degrees(np.arccos(np.clip(A[k] @ targets[a], -1.0, 1.0))))
            for k, a in enumerate(_AXES)
        }
        periods_A = [
            None,
            None,
            _lattice_period(zv, lattice_constant, crystalstructure),
        ]
        _warn_tilted_front(
            zv,
            z0.astype(int).tolist(),
            lattice_constant,
            crystalstructure,
            (phi_actual, theta_actual, zeta_actual),
        )
        x, y = (", ".join(f"{c:.15g}" for c in v) for v in (xu, yu))
        z = ", ".join(str(c) for c in zv)
        return (
            x,
            y,
            z,
            phi_actual,
            theta_actual,
            zeta_actual,
            deviation_deg,
            periods_A,
            plane_wcrack,
            rotation_deg_wcrack,
        )

    if secondary is None:
        secondary = "z" if priority == "y" else "y"
    if secondary == priority:
        raise ValueError(f"secondary axis must differ from priority ({priority!r})")
    third = next(a for a in _AXES if a not in (priority, secondary))

    axes = {}
    axes[priority], _ = _pick_direction(grid, targets[priority], tolerance_deg)
    perpendicular = grid[grid @ np.array(axes[priority]) == 0]
    axes[secondary], _ = _pick_direction(
        perpendicular, targets[secondary], tolerance_deg
    )
    i, j = {"x": ("y", "z"), "y": ("z", "x"), "z": ("x", "y")}[third]
    axes[third] = _reduce(np.cross(axes[i], axes[j]).tolist())
    if max(abs(c) for c in axes[third]) > n:
        raise ValueError(
            f"the {third} axis {axes[third]} (= {i} x {j}) has components above max_index = {n} "
            f"({priority} = {axes[priority]}, {secondary} = {axes[secondary]}). Raise max_index, "
            "use a larger tolerance_deg, or choose another priority/secondary axis."
        )

    A = np.array([axes[a] for a in _AXES], float)
    A /= np.linalg.norm(A, axis=1)[:, None]
    if not np.allclose(A @ A.T, np.eye(3), atol=1e-12) or np.linalg.det(A) <= 0:
        raise RuntimeError(
            f"internal error: {axes} is not a right-handed orthogonal set"
        )
    phi_actual, theta_actual, zeta_actual = (
        float(v) for v in Rotation.from_matrix(A @ R0.T).as_euler("xyz", degrees=True)
    )
    deviation_deg = {
        a: float(np.degrees(np.arccos(np.clip(A[k] @ targets[a], -1.0, 1.0))))
        for k, a in enumerate(_AXES)
    }
    periods_A = [
        _lattice_period(axes[a], lattice_constant, crystalstructure) for a in _AXES
    ]
    _warn_tilted_front(
        axes["z"],
        z0.astype(int).tolist(),
        lattice_constant,
        crystalstructure,
        (phi_actual, theta_actual, zeta_actual),
    )
    x, y, z = (", ".join(str(c) for c in axes[a]) for a in _AXES)
    return (
        x,
        y,
        z,
        phi_actual,
        theta_actual,
        zeta_actual,
        deviation_deg,
        periods_A,
        y,
        0.0,
    )


_CUBIC_BASIS = {
    "sc": [[0.0, 0.0, 0.0]],
    "bcc": [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    "fcc": [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]],
    "diamond": [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
        [0.25, 0.25, 0.25],
        [0.75, 0.75, 0.25],
        [0.75, 0.25, 0.75],
        [0.25, 0.75, 0.75],
    ],
}


def _lab_frame(x, y, z):
    """Unit rows (lab x, y, z in crystal coordinates) from three direction strings
    (real-valued allowed). Raises unless they are orthogonal and right-handed."""
    import numpy as np

    R = np.array([_floats(x, 3, "x"), _floats(y, 3, "y"), _floats(z, 3, "z")], float)
    R /= np.linalg.norm(R, axis=1)[:, None]
    if not np.allclose(R @ R.T, np.eye(3), atol=1e-9):
        raise ValueError(f"axes x={x!r}, y={y!r}, z={z!r} are not orthogonal")
    if np.linalg.det(R) <= 0:
        raise ValueError(f"axes x={x!r}, y={y!r}, z={z!r} are not right-handed")
    return R


def _xy_centre(structure):
    """Centre of the cell's x-y face, (x, y, 0): the crack tip / cylinder axis.

    The crack nodes keep the atoms inside an ordinary cell starting at (0, 0, 0)
    and put the tip at this point, so every node measures positions relative to
    it. CrackSpecimenFiles subtracts it again, because wcrack's files have the
    tip at (0, 0)."""
    import numpy as np

    cell = np.array(structure.cell)
    centre = 0.5 * (cell[0] + cell[1])
    return np.array([centre[0], centre[1], 0.0]) + np.asarray(
        structure.get_celldisp()
    ).ravel() * [1, 1, 0]


@as_function_node("structure")
def OrientedLatticeBlock(
    element: str,
    lattice_constant: float,
    crystalstructure: Literal["bcc", "fcc", "sc", "diamond"] = "bcc",
    x: str = "1, -1, 0",
    y: str = "1, 1, 0",
    z: str = "0, 0, 1",
    size_xy_A: float = 450.0,
    nz: int = 1,
    max_atoms: int = 2_000_000,
):
    """Block of lattice sites in the lab frame, filling the cell
    [0, L] x [0, L] x [0, Lz) with L = size_xy_A and Lz = nz * (lattice period
    along z). Periodic along z only.

    The sites are generated directly inside the box (cubic lattice points in
    crystal coordinates within the box's bounding region, rotated into the lab
    frame), not by repeating periodic cells. So x and y need no whole cells:
    the box is exactly `size_xy_A` wide, and the atom count (about
    size_xy_A^2 * Lz * density) does not depend on the lattice periods along x
    and y. x and y may be real-valued directions (e.g. from MisorientedAxes
    mode="rational_front"); z must be an integer direction, since the block is
    periodic along it.

    The lattice has a site at the centre of the cell's x-y face (L/2, L/2),
    which is the cylinder axis / crack tip, as wcrack's slab_points has one at
    its origin. The sites are generated around that centre and recorded inside
    the cell (no celldisp): x and y are filled because the sites are generated
    in the box (they have no lattice period there for real-valued axes, so they
    are not wrapped); z is wrapped into the cell (ASE wrap). Along z the interval
    is half-open: a site within 1e-8 A of Lz is put at z = 0. Returns pbc
    (False, False, True) and a cell of size_xy_A x size_xy_A x Lz. Raises above
    `max_atoms` before generating anything.
    """
    import math

    import numpy as np
    from ase import Atoms

    a = float(lattice_constant)
    zi = _ints(z, 3, "z")
    R = _lab_frame(x, y, z)
    Lz = int(nz) * _lattice_period(zi, a, crystalstructure)
    half = 0.5 * float(size_xy_A)
    basis = np.array(_CUBIC_BASIS[crystalstructure], float)
    n_estimate = int(round((2 * half) ** 2 * Lz * len(basis) / a**3))
    if n_estimate > int(max_atoms):
        raise ValueError(
            f"block of {2 * half:g} x {2 * half:g} x {Lz:.2f} A would hold about {n_estimate:,} atoms "
            f"> max_atoms = {int(max_atoms):,}"
        )

    corners = np.array(
        [
            [sx * half, sy * half, sz]
            for sx in (-1, 1)
            for sy in (-1, 1)
            for sz in (0.0, Lz)
        ]
    )
    cryst = corners @ R / a  # lab -> crystal (units of a): c = R^T l, row form l @ R
    lo = np.floor(cryst.min(axis=0)) - 1
    hi = np.ceil(cryst.max(axis=0)) + 1
    eps = 1e-6
    k = int(np.argmax(hi - lo))  # chunk along the longest crystal axis to bound memory
    n_cubes = float(np.prod(hi - lo))
    n_chunks = max(1, int(math.ceil(n_cubes * len(basis) / 4_000_000)))
    edges = np.linspace(lo[k], hi[k] + 1, n_chunks + 1).round().astype(int)
    parts = []
    for c0, c1 in zip(edges[:-1], edges[1:]):
        if c1 <= c0:
            continue
        ranges = [np.arange(int(lo[i]), int(hi[i]) + 1) for i in range(3)]
        ranges[k] = np.arange(c0, c1)
        cubes = (
            np.stack(np.meshgrid(*ranges, indexing="ij"), -1)
            .reshape(-1, 3)
            .astype(float)
        )
        sites = (cubes[:, None, :] + basis[None, :, :]).reshape(-1, 3) * a
        lab = sites @ R.T
        keep = (
            (np.abs(lab[:, 0]) <= half)
            & (np.abs(lab[:, 1]) <= half)
            & (lab[:, 2] >= -eps)
            & (lab[:, 2] < Lz - eps)
        )
        parts.append(lab[keep])
    pts = np.concatenate(parts) if parts else np.empty((0, 3))
    pts[:, :2] += half  # record the atoms inside the cell: centre (0, 0) -> (L/2, L/2)

    structure = Atoms(
        symbols=[str(element)] * len(pts),
        positions=pts,
        cell=np.diag([2 * half, 2 * half, Lz]),
        pbc=(False, False, True),
    )
    structure.wrap()  # along z (the periodic axis) into [0, Lz)
    positions = structure.get_positions()
    positions[positions[:, 2] >= Lz - 1e-8, 2] = 0.0
    positions[np.abs(positions[:, 2]) < 1e-8, 2] = 0.0
    structure.set_positions(positions)
    return structure


@as_function_node("tag")
def AxesTag(
    x: str, y: str, z: str, style: Literal["alireza", "signed"] = "alireza"
) -> str:
    """Name tag for a crystal frame (lab axes x, y, z as integer directions).

    "alireza": the convention of 16_Alireza's run folders, e.g. x = [0 10 1],
    y = [0 -1 10], z = [1 0 0] -> "X0101_Y0110_Z100": the absolute values of
    the components written one after the other, signs and separators dropped.
    This is ambiguous (X0101 could also be [0 1 01] / [01 0 1]), so use it only
    to compare with those runs.

    "signed": unambiguous, "X0_10_1__Ym1_0_10__Z1_0_0" ('m' = minus, '_'
    between components, '__' between axes).
    """
    vecs = [_ints(v, 3, name) for v, name in ((x, "x"), (y, "y"), (z, "z"))]
    if style == "alireza":
        tag = "_".join(a + "".join(str(abs(c)) for c in v) for a, v in zip("XYZ", vecs))
    else:
        tag = "__".join(
            a + "_".join(str(c).replace("-", "m") for c in v)
            for a, v in zip("XYZ", vecs)
        )
    return tag


@group_node(
    "tag",
    "x",
    "y",
    "z",
    "phi_actual",
    "theta_actual",
    "zeta_actual",
    "deviation_deg",
    "periods_A",
)
def MisorientationCase(
    plane="1, 1, 0",
    front="0, 0, 1",
    phi=0.0,
    theta=0.0,
    zeta=0.0,
    tolerance_deg=0.5,
    max_index=30,
    lattice_constant=3.165,
    style="alireza",
):
    """MisorientedAxes -> AxesTag for one misorientation, as a sweep template
    (IterToDataFrame over zeta, phi or theta, or several at once with
    input_label="phi, zeta"): the integer axes, their name tag, the actual
    angles, the deviation of each axis from its target and the lattice period
    along each axis (`periods_A`; the z period is the shortest front length)."""
    from aiflow.core import Workflow

    from aiflow_nodes.CrackNodes import AxesTag, MisorientedAxes

    inner_wf = Workflow("MisorientationCase")
    inner_wf.MisorientedAxes = MisorientedAxes(
        plane=plane,
        front=front,
        phi=phi,
        theta=theta,
        zeta=zeta,
        tolerance_deg=tolerance_deg,
        max_index=max_index,
        lattice_constant=lattice_constant,
    )
    inner_wf.AxesTag = AxesTag(
        x=inner_wf.MisorientedAxes.outputs.x,
        y=inner_wf.MisorientedAxes.outputs.y,
        z=inner_wf.MisorientedAxes.outputs.z,
        style=style,
    )
    return (
        inner_wf.AxesTag.outputs.tag,
        inner_wf.MisorientedAxes.outputs.x,
        inner_wf.MisorientedAxes.outputs.y,
        inner_wf.MisorientedAxes.outputs.z,
        inner_wf.MisorientedAxes.outputs.phi_actual,
        inner_wf.MisorientedAxes.outputs.theta_actual,
        inner_wf.MisorientedAxes.outputs.zeta_actual,
        inner_wf.MisorientedAxes.outputs.deviation_deg,
        inner_wf.MisorientedAxes.outputs.periods_A,
    )


@as_function_node(["size", "n_atoms_estimate", "periods_A"])
def CylinderRepeats(
    x: str,
    y: str,
    z: str,
    lattice_constant: float,
    crystalstructure: Literal["bcc", "fcc", "sc", "diamond"] = "bcc",
    radius_nm: float = 4.0,
    front_length_nm: float = 1.2,
    margin_A: float = 12.0,
    max_atoms: int = 2_000_000,
):
    """`size` ("nx, ny, nz") for OrientedCubicCell(x, y, z, size=...) so the block
    holds a cylinder of radius R along z, computed from the axes and the lattice
    *before* anything is built.

    The cell edges are the lattice periods along x, y, z (`periods_A`,
    computed analytically, as ASE builds them). x, y: the block covers a square
    of side 2 (R + margin); both counts are rounded up to even numbers, so after
    CenterXY a lattice site sits exactly on the axis (same lattice as wcrack's).
    z: the front length rounded up to whole periods, as wcrack's front_geometry
    does (nz = ceil(10 * front_length_nm / period_z)).

    `n_atoms_estimate` is the number of atoms in the block. Raises above
    `max_atoms` instead of building it: a high-index axis has a long period,
    and the block grows with it.
    """
    import math

    periods_A = [
        _lattice_period(_ints(v, 3, name), lattice_constant, crystalstructure)
        for v, name in ((x, "x"), (y, "y"), (z, "z"))
    ]
    side = 2.0 * (10.0 * float(radius_nm) + float(margin_A))
    nx = 2 * int(math.ceil(side / periods_A[0] / 2.0))
    ny = 2 * int(math.ceil(side / periods_A[1] / 2.0))
    nz = max(1, int(math.ceil(10.0 * float(front_length_nm) / periods_A[2] - 1e-9)))
    volume = nx * ny * nz * periods_A[0] * periods_A[1] * periods_A[2]
    n_atoms_estimate = int(
        round(volume * _ATOMS_PER_CUBE[crystalstructure] / float(lattice_constant) ** 3)
    )
    if n_atoms_estimate > int(max_atoms):
        raise ValueError(
            f"block of {nx} x {ny} x {nz} cells ({periods_A[0]:.1f} x {periods_A[1]:.1f} x {periods_A[2]:.1f} A each) "
            f"would hold about {n_atoms_estimate:,} atoms > max_atoms = {int(max_atoms):,}. Use lower-index axes "
            "(MisorientedAxes: smaller max_index or larger tolerance_deg) or raise max_atoms deliberately."
        )
    size = f"{nx}, {ny}, {nz}"
    return size, n_atoms_estimate, periods_A


@as_function_node("structure")
def CenterXY(structure):
    """Move the atoms so the centre of the cell's x-y face is at x = y = 0
    (the cylinder axis / crack front through the origin). The cell itself is
    unchanged; z is not touched."""
    import numpy as np

    structure = structure.copy()
    cell = np.array(structure.cell)
    centre = 0.5 * (cell[0] + cell[1])
    positions = structure.get_positions()
    positions[:, :2] -= centre[:2]
    structure.set_positions(positions)
    return structure


@as_function_node(["structure", "origin_shift"])
def CrackOriginShift(
    structure,
    snap_to_lattice: bool = True,
    tip_shift_A: str = "0.0, 0.0",
    wcrack_dir: str = WCRACK_DIR,
):
    """Shift the atoms in x-y so the crack plane lies midway between atomic
    layers and the tip midway between the columns of the first layer above it
    (wcrack crystal.crack_origin_shift), then by `tip_shift_A` ("dx, dy"). The
    tip is the centre of the cell's x-y face (the cell is not moved); the snap
    is computed relative to it. The atoms stay recorded inside the cell: they
    are wrapped along the periodic axes, and atoms shifted past a face along a
    non-periodic axis (x, y of an OrientedLatticeBlock) are dropped; they lie
    in the outer rim of the block, which CutCylinder removes anyway. Returns
    the structure and a dict with the applied shift, the layer / column gaps
    (geometry.json `origin_shift_A`, `lattice_gaps_A`) and the number of atoms
    dropped (`atoms_dropped_outside_cell`).
    """
    import numpy as np

    _wcrack(wcrack_dir)
    from wcrack import crystal

    structure = structure.copy()
    positions = structure.get_positions()
    if snap_to_lattice:
        shift, gaps = crystal.crack_origin_shift(positions - _xy_centre(structure))
    else:
        shift, gaps = np.zeros(2), {}
    shift = shift + np.array(_floats(tip_shift_A, 2, "tip_shift_A"))
    positions[:, :2] -= shift
    structure.set_positions(positions)
    # keep every atom recorded inside the cell: wrap along the periodic axes; along a non-periodic axis
    # (x, y of a block with real-valued axes) an atom shifted past a face can't be wrapped, so drop it
    if any(structure.pbc):
        structure.wrap()
    frac = structure.get_scaled_positions(wrap=False)
    eps = 1e-9
    outside = np.zeros(len(structure), bool)
    for i in range(3):
        if not structure.pbc[i]:
            outside |= (frac[:, i] < -eps) | (frac[:, i] > 1 + eps)
    if outside.any():
        del structure[[int(i) for i in np.flatnonzero(outside)]]
    origin_shift = {
        "origin_shift_A": shift.tolist(),
        "lattice_gaps_A": gaps,
        "atoms_dropped_outside_cell": int(outside.sum()),
    }
    return structure, origin_shift


@as_function_node("structure")
def CutCylinder(structure, radius_nm: float, margin_A: float = 15.0):
    """Keep the atoms within radius R of the cylinder axis, the line along z
    through the centre of the cell's x-y face (the crack tip), periodic along
    z only.

    The result has pbc (False, False, True), a square cell of side
    2 (R + margin) from (0, 0, 0) with the atoms inside it and the axis at its
    x-y centre (R + margin, R + margin), and the cell's z length unchanged.
    Atoms are sorted by z, then y, then x (positions relative to the axis,
    rounded to 1e-6 Å), as in wcrack, so atom ids are reproducible.
    """
    import numpy as np

    R = 10.0 * float(radius_nm)
    half = R + float(margin_A)
    positions = structure.get_positions() - _xy_centre(structure)
    Lz = float(np.array(structure.cell)[2, 2])
    positions[:, 2] = np.mod(positions[:, 2], Lz)
    # an atom on the z = 0 plane can come out at -1e-14 (-> Lz - 1e-14) or +1e-14 depending on rounding
    # noise; put every atom within 1e-8 A of Lz at z = 0, so the image along the front is deterministic
    positions[positions[:, 2] >= Lz - 1e-8, 2] = 0.0
    keep = np.hypot(positions[:, 0], positions[:, 1]) <= R
    pts = positions[keep]
    order = np.lexsort(
        (np.round(pts[:, 0], 6), np.round(pts[:, 1], 6), np.round(pts[:, 2], 6))
    )
    symbols = np.array(structure.get_chemical_symbols())[keep][order]

    from ase import Atoms

    structure = Atoms(
        symbols=list(symbols),
        positions=pts[order] + [half, half, 0.0],
        cell=np.diag([2.0 * half, 2.0 * half, Lz]),
        pbc=(False, False, True),
    )
    return structure


@as_function_node("structure")
def CutNotch(structure, notch_radius_A: float = 0.0, wcrack_dir: str = WCRACK_DIR):
    """Remove the atoms of a rounded notch whose root is at the crack tip, the
    centre of the cell's x-y face (wcrack crystal.notch_void, relative to the
    tip: a slot x <= -rho, |y| < rho plus a semicircular cap of radius rho).
    `notch_radius_A = 0` is a slit: nothing is removed. The order of the
    remaining atoms is kept."""
    _wcrack(wcrack_dir)
    from wcrack import crystal

    structure = structure.copy()
    rho = float(notch_radius_A)
    if rho > 0:
        void = crystal.notch_void(
            (structure.get_positions() - _xy_centre(structure))[:, :2], rho
        )
        del structure[[int(i) for i in void.nonzero()[0]]]
    return structure


@as_function_node(["structure", "n_fixed"])
def FreezeOuterRing(structure, radius_nm: float, ring_width_A: float = 20.0):
    """Fix the outer ring, r > R - ring_width_A from the crack tip (the cell's
    x-y centre), with an ASE FixAtoms constraint: selective dynamics
    [False, False, False] for those atoms.

    Apply it to the undisplaced specimen (after CutNotch, before StrohKField /
    ApplyKField), so the ring is decided on X0; ApplyKField's copy keeps the
    constraint. Downstream, LammpsGroupFromStructure turns it into
    `group ring id ...` and LammpsFix(setforce 0 0 0) freezes it;
    CrackSpecimenFiles reads it as the FIXED mask.

    `radius_nm` is CutCylinder's radius. Replaces any constraint the structure
    already has (a copy is returned). Returns the structure and the number of
    fixed atoms.
    """
    import numpy as np
    from ase.constraints import FixAtoms

    R = 10.0 * float(radius_nm)
    r_free = R - float(ring_width_A)
    if r_free <= 0.0:
        raise ValueError(
            f"ring_width_A {ring_width_A} >= radius {R} A: no mobile core left"
        )
    structure = structure.copy()
    if structure.constraints:
        print(
            f"FreezeOuterRing: replacing the structure's existing constraints {structure.constraints}"
        )
    rel = structure.get_positions() - _xy_centre(structure)
    mask = np.hypot(rel[:, 0], rel[:, 1]) > r_free
    if not mask.any():
        raise ValueError(
            f"no atom beyond r = {r_free} A: check radius_nm against the cut cylinder"
        )
    structure.set_constraint(FixAtoms(mask=mask))
    n_fixed = int(mask.sum())
    return structure, n_fixed


@as_function_node("structure")
def ReorderAtoms(structure, order: list | None = None):
    """The same atoms listed in another order: atom k of the result is atom
    order[k] of the input (constraints such as the FixAtoms ring follow their
    atoms). With `order` None (default) the structure is passed on unchanged.

    For validation: the atom order (= LAMMPS ids) of this chain and of wcrack
    can differ for a few atoms whose sort keys round differently at the 1e-6 A
    step; giving wcrack's order makes both LAMMPS inputs use the same ids.
    """
    import numpy as np

    if order is None:
        return structure
    order = np.asarray(order, int)
    n = len(structure)
    if order.shape != (n,) or not np.array_equal(np.sort(order), np.arange(n)):
        raise ValueError(f"order must be a permutation of 0..{n - 1}")
    structure = structure[order]
    return structure


@as_function_node(["U", "K_griffith", "K_griffith_mode_I", "stroh"])
def StrohKField(
    structure,
    elastic_constants: str = "525.2, 192.5, 177.5",
    x: str = "1, -1, 0",
    y: str = "1, 1, 0",
    z: str = "0, 0, 1",
    rotation_deg: float = 0.0,
    mode: str = "1, 0, 0",
    surface_energy_J_m2: float = 0.0,
    wcrack_dir: str = WCRACK_DIR,
):
    """Unit anisotropic crack-tip displacement field U(X0) per atom [Å per
    1 MPa m^1/2] for a straight crack on y = 0, x < 0 with the front along z
    (wcrack elasticity.StrohCrack, sextic Stroh, cubic C rotated into the lab
    frame), and the Griffith K.

    `x`, `y`, `z` are the crystal directions of the lab axes (as given to
    OrientedCubicCell or OrientedLatticeBlock; real-valued allowed); they must
    be orthogonal and right-handed, i.e. x parallel to y cross z.
    `rotation_deg` is an extra crystal rotation about the front (wcrack
    convention, as in crystal.orientation); 0 when x, y are already rotated.
    `mode` = weights (I, II, III). K_griffith (along `mode`) and
    K_griffith_mode_I need `surface_energy_J_m2 > 0`, else they are None.

    Raises if an atom lies on the crack plane behind the tip (|y| < 1e-6 Å at
    x < 0), where the field is undefined. Returns U, both K_G and the Stroh
    summary (geometry.json `stroh`).
    """
    import math

    import numpy as np

    _wcrack(wcrack_dir)
    from wcrack import elasticity

    R0 = _lab_frame(x, y, z)  # orthonormal + right-handed => x = y cross z
    th = math.radians(float(rotation_deg))
    c, s = math.cos(th), math.sin(th)
    # as wcrack crystal.orientation: the crystal rotated by +theta about z <=> lab x, y rotate by -theta
    R = np.array([c * R0[0] - s * R0[1], s * R0[0] + c * R0[1], R0[2]])

    positions = structure.get_positions() - _xy_centre(structure)  # relative to the tip
    flank = positions[:, 0] < 0
    if flank.any() and float(np.min(np.abs(positions[flank, 1]))) < 1e-6:
        raise ValueError(
            "an atom lies on the crack plane behind the tip; snap the origin to the lattice "
            "(CrackOriginShift) or use a different tip_shift_A."
        )

    crack = elasticity.StrohCrack(
        elasticity.lab_stiffness(_floats(elastic_constants, 3, "elastic_constants"), R)
    )
    wI, wII, wIII = _floats(mode, 3, "mode")
    mode_vector = np.array([wII, wI, wIII])  # Stroh order (K_II, K_I, K_III)
    U = crack.displacement(positions, mode_vector)
    gamma = float(surface_energy_J_m2)
    K_griffith = crack.griffith_K(gamma, mode_vector) if gamma > 0 else None
    K_griffith_mode_I = crack.griffith_K(gamma, (0.0, 1.0, 0.0)) if gamma > 0 else None
    stroh = crack.summary()
    return U, K_griffith, K_griffith_mode_I, stroh


@as_function_node("structure")
def ApplyKField(structure, U, K: float = 1.7, scale: float = 1.0):
    """The specimen with the crack-tip field applied, for viewing: positions
    X0 + K * scale * U (Å), with U from StrohKField (Å per 1 MPa m^1/2) and K
    in MPa m^1/2.

    With scale = 1 this is the configuration the first loading step starts
    from (wcrack applies K_start * U to all atoms of the unloaded specimen,
    then minimises); the minimised states are only in the run's dumps. A
    slit (notch radius 0) exists only through this field: its opening grows
    as sqrt(r) from the tip, below 1 Å at R = 4 nm for K ~ 1.7. scale > 1
    exaggerates the displacement to make the taper visible; that is a
    picture only, not a physical state.

    Nothing is moved in the input structure (a displaced copy is returned),
    and no atom is dropped. Along the periodic axis (z) the atoms are wrapped
    back into the cell; along x and y the cylinder's cell has a 15 Å margin,
    so they stay inside unless K * scale * |U| exceeds that.
    """
    import numpy as np

    U = np.asarray(U, float)
    if U.shape != (len(structure), 3):
        raise ValueError(
            f"U has shape {U.shape}, expected ({len(structure)}, 3): wire the U of the "
            "StrohKField computed on this same structure."
        )
    structure = structure.copy()
    # as wcrack's first step, every atom moves (ring included); keep a FixAtoms ring, but don't let it block this
    structure.set_positions(
        structure.get_positions() + float(K) * float(scale) * U, apply_constraint=False
    )
    if any(structure.pbc):
        structure.wrap()
    return structure


@as_function_node("meta")
def CrackSpecimenFiles(
    cfg: dict,
    case: dict,
    structure,
    U,
    K_griffith=None,
    K_griffith_mode_I=None,
    stroh: dict | None = None,
    origin_shift: dict | None = None,
    wcrack_dir: str = WCRACK_DIR,
) -> dict:
    """Write the specimen in wcrack's format into `{run_dir}/cases/{name}`:
    `specimen.data` (LAMMPS data with the X0/U/FIXED Properties section),
    `reference.npz` and `geometry.json`, exactly as wcrack specimen.build does,
    so CrackLoadingInput and AnalyzeCrackCase work unchanged.

    The fixed ring is read from the structure's ASE FixAtoms constraint;
    without one this raises. The structure must match `case`: its z length must
    be wcrack's front length for the case's front and lattice constant.
    Raises instead of overwriting a case folder that holds loading output.
    Returns the geometry metadata with the K schedule.
    """
    import json

    import numpy as np
    from ase.constraints import FixAtoms

    _wcrack(wcrack_dir)
    from wcrack import crystal, loading, specimen

    d = loading.case_dir(cfg, case)
    if (d / "progress.txt").exists():
        raise RuntimeError(
            f"{case['name']}: {d} already holds loading output (progress.txt); "
            "use a new run_dir or case_name (nothing is deleted)."
        )
    pts = structure.get_positions() - _xy_centre(
        structure
    )  # wcrack's files: tip at (0, 0)
    n = len(pts)
    fixed_idx = [
        i
        for c in structure.constraints
        if isinstance(c, FixAtoms)
        for i in c.get_indices()
    ]
    if not fixed_idx:
        raise ValueError(
            "the structure has no FixAtoms constraint: attach the fixed outer ring first."
        )
    fixed = np.zeros(n, bool)
    fixed[np.asarray(fixed_idx, int)] = True
    U = np.asarray(U, float)
    if U.shape != (n, 3):
        raise ValueError(f"U has shape {U.shape}, expected ({n}, 3)")

    refs = _refs(cfg)
    a = refs["a0_A"]
    o = crystal.orientation(
        case["plane"], case["front"], case["direction"] or None, case["rotation_deg"]
    )
    pz, nz, Lz = specimen.front_geometry(case, a)
    Lz_cell = float(np.array(structure.cell)[2, 2])
    if abs(Lz_cell - Lz) > 1e-6:
        raise ValueError(
            f"structure z length {Lz_cell:.6f} A differs from the case's front length {Lz:.6f} A "
            f"({nz} periods of {pz:.6f} A)"
        )
    R = 10.0 * case["radius_nm"]
    ring = 10.0 * case["fixed_ring_nm"]
    rho = float(case["notch_radius_A"]) if case["crack"] == "notch" else 0.0
    ids = np.arange(1, n + 1)
    flank = pts[:, 0] < 0
    min_abs_y = float(np.min(np.abs(pts[flank, 1]))) if flank.any() else float("nan")

    d.mkdir(parents=True, exist_ok=True)
    lo, hi = -(R + 15.0), R + 15.0
    with open(d / "specimen.data", "w") as f:
        f.write(
            f"# wcrack specimen {case['name']}: {o['label']}, a = {a:.8f} A, R = {R:g} A, Lz = {Lz:.6f} A\n\n"
        )
        f.write(f"{n} atoms\n1 atom types\n\n")
        f.write(
            f"{lo:.6f} {hi:.6f} xlo xhi\n{lo:.6f} {hi:.6f} ylo yhi\n0.0 {Lz:.12f} zlo zhi\n\n"
        )
        f.write(f"Masses\n\n1 {cfg['potential']['mass']}\n\nAtoms # atomic\n\n")
        np.savetxt(
            f, np.column_stack([ids, np.ones(n), pts]), fmt="%d %d %.12f %.12f %.12f"
        )
        f.write("\nProperties\n\n")
        np.savetxt(
            f,
            np.column_stack([ids, pts, U, fixed.astype(int)]),
            fmt="%d %.12f %.12f %.12f %.12e %.12e %.12e %d",
        )
    np.savez_compressed(
        d / "reference.npz", ids=ids, X0=pts, U=U, fixed=fixed, a=a, Lz=Lz
    )

    gamma = (
        float(case["surface_energy_J_m2"]) if case["surface_energy_J_m2"] > 0 else None
    )
    origin_shift = origin_shift or {}
    meta = {
        "case": case["name"],
        "orientation": {
            "x_propagation": o["x"],
            "y_plane_normal": o["y"],
            "z_front": o["z"],
            "rotation_about_front_deg": o["rotation_deg"],
            "rational_plane": o["rational_plane"],
            "x_after_rotation": o["x_rotated"],
            "y_after_rotation": o["y_rotated"],
            "label": o["label"],
            "R_lab_rows_in_crystal_frame": o["R"].tolist(),
        },
        "a0_A": a,
        "C_GPa": list(refs["C_GPa"]),
        "front_period_A": pz,
        "front_periods": nz,
        "Lz_A": Lz,
        "radius_A": R,
        "fixed_ring_A": ring,
        "mobile_radius_A": R - ring,
        "crack": case["crack"],
        "notch_radius_A": rho,
        "origin_shift_A": origin_shift.get("origin_shift_A", [0.0, 0.0]),
        "lattice_gaps_A": origin_shift.get("lattice_gaps_A", {}),
        "min_abs_y_on_flank_A": min_abs_y,
        "natoms": int(n),
        "nfixed": int(fixed.sum()),
        "nmobile": int((~fixed).sum()),
        "mode_weights_I_II_III": [float(v) for v in case["mode"]],
        "stroh": stroh or {},
        "surface_energy_J_m2": gamma,
        "surface_energy_source": "case setting surface_energy_J_m2"
        if gamma
        else "not available",
        "surface_energy_plane": specimen.plane_key(case),
        "K_griffith_mode_I_MPa_sqrtm": K_griffith_mode_I,
        "K_griffith_along_mode_MPa_sqrtm": K_griffith,
        "max_unit_displacement_A": float(np.abs(U).max()),
        "conventions": "tip at (0,0); crack on y=0, x<0; lab x=propagation, y=plane normal, z=front; "
        "U = displacement per 1 MPa m^1/2 of the ramp parameter (Stroh sextic, stationary tip).",
        "built_by": "aiflow_nodes.CrackNodes (separate nodes)",
    }
    meta["K_schedule_MPa_sqrtm"] = loading.schedule(case, K_griffith)
    meta["signature"] = loading.signature(case, refs)
    (d / "geometry.json").write_text(json.dumps(meta, indent=1))
    try:
        specimen.preview(pts, fixed, meta, d / "preview.png")
    except Exception as exc:  # plotting is optional
        meta["preview_error"] = str(exc)
    return meta


# --------------------------------------------------------------------------- #
#  LAMMPS input                                                                #
# --------------------------------------------------------------------------- #


@as_function_node("io_bundle")
def CrackLoadingInput(cfg: dict, case: dict, meta: dict, wcrack_dir: str = WCRACK_DIR):
    """The LAMMPS input of the whole K schedule (wcrack lammps_io.loading_header
    + loading_steps, as the CLI's `in.load`), wrapped in a LammpsIOBundle for
    the existing RunLammpsBase.

    Every step: `displace_atoms` by dK * U(X0), `minimize` (ring held by
    `setforce 0 0 0`), a WCRACK_STEP line in progress.txt, a dump in
    dumps/step_NNNN.dump and a restart file; then the early-stop test inside
    LAMMPS. The potential is inline (`pair_style meam`), boundaries `s s p`.

    The bundle's working directory is the case folder and its structure file is
    `specimen.data` (rewritten with the same content by RunLammpsBase). Creates
    `dumps/` and `restart/`; raises if the folder already holds a loading
    history (progress.txt), where wcrack's CLI would delete it.
    """
    import numpy as np
    from ase import Atoms

    _wcrack(wcrack_dir)
    from wcrack import lammps_io, loading

    from aiflow_nodes.atomistic.engine.lammps import LammpsIOBundle

    d = loading.case_dir(cfg, case)
    if (d / "progress.txt").exists():
        raise RuntimeError(
            f"{case['name']}: {d} already holds a loading history (progress.txt); "
            "use a new run_dir or case_name (nothing is deleted)."
        )
    (d / "dumps").mkdir(exist_ok=True)
    (d / "restart").mkdir(exist_ok=True)

    ks = meta["K_schedule_MPa_sqrtm"]
    header, group, dump_group = lammps_io.loading_header(cfg, case, meta)
    lammps_input_string = header + lammps_io.loading_steps(
        case, ks, 0, 0.0, group, dump_group
    )

    ref = np.load(d / "reference.npz")
    p = cfg["potential"]
    structure = Atoms(
        [p["element"]] * len(ref["X0"]),
        positions=ref["X0"],
        cell=np.diag([2 * meta["radius_A"], 2 * meta["radius_A"], meta["Lz_A"]]),
        pbc=[False, False, True],
    )
    io_bundle = LammpsIOBundle(
        structure=structure,
        potential=f"wcrack meam: {p['library']} {p['parameters']}",
        mode="minimize",
        working_directory=str(d),
        lammps_input_string=lammps_input_string,
        lammps_input_filename="lmp.in",
        lammps_structure_string=(d / "specimen.data").read_text(),
        lammps_structure_filename="specimen.data",
    )
    return io_bundle


# --------------------------------------------------------------------------- #
#  Analysis                                                                    #
# --------------------------------------------------------------------------- #


@as_function_node(["result", "steps"])
def AnalyzeCrackCase(
    cfg: dict, case: dict, io_bundle, figures: bool = True, wcrack_dir: str = WCRACK_DIR
):
    """wcrack analysis.analyze_case after the loading run: crack-tip tracking,
    plasticity indicator, critical-K bracket and event type per step.

    `io_bundle` is RunLammpsBase's output; it only orders this node after the
    run (the analysis reads the case folder: progress.txt, log.lammps, dumps).
    Writes analysis/steps.csv, crack_profile.csv, result.json and, with
    `figures`, K_curve.png and snapshots.png. Returns the result dict and the
    per-step table as a DataFrame.
    """
    import pandas as pd

    _wcrack(wcrack_dir)
    from wcrack import analysis, loading

    result = analysis.analyze_case(cfg, case, figures=bool(figures))
    steps = pd.read_csv(loading.case_dir(cfg, case) / "analysis" / "steps.csv")
    return result, steps


# --------------------------------------------------------------------------- #
#  One case end to end                                                         #
# --------------------------------------------------------------------------- #


@group_node("result", "steps")
def CrackCase(
    run_dir,
    case_name="c110_001",
    plane="1, 1, 0",
    front="0, 0, 1",
    radius_nm=4.0,
    fixed_ring_nm=1.2,
    front_length_nm=1.2,
    K_start=1.70,
    K_step=0.01,
    K_stop=1.80,
    mode="1, 0, 0",
    lattice_constant=3.165,
    elastic_constants="525.2, 192.5, 177.5",
    ftol=1e-6,
    min_norm="two",
    early_stop=True,
    lmp_command=None,
):
    """One crack case: CrackConfig -> BuildCrackSpecimen -> CrackLoadingInput ->
    RunLammpsBase (the whole K schedule in one LAMMPS run) -> AnalyzeCrackCase.

    Put the SlurmExecutor on this group for one SLURM job per case; sweep it
    with IterToDataFrame (executor on the sweep) for several cases in parallel.
    Ports not exposed here (element, potential files, rotation, notch, ...)
    take CrackConfig's defaults: W with Onur's W-P MEAM.
    """
    from aiflow.core import Workflow

    from aiflow_nodes.atomistic.engine.lammps import RunLammpsBase
    from aiflow_nodes.CrackNodes import (
        AnalyzeCrackCase,
        BuildCrackSpecimen,
        CrackConfig,
        CrackLoadingInput,
    )

    inner_wf = Workflow("CrackCase")
    inner_wf.CrackConfig = CrackConfig(
        run_dir=run_dir,
        case_name=case_name,
        plane=plane,
        front=front,
        radius_nm=radius_nm,
        fixed_ring_nm=fixed_ring_nm,
        front_length_nm=front_length_nm,
        K_start=K_start,
        K_step=K_step,
        K_stop=K_stop,
        mode=mode,
        lattice_constant=lattice_constant,
        elastic_constants=elastic_constants,
        ftol=ftol,
        min_norm=min_norm,
        early_stop=early_stop,
    )
    inner_wf.BuildCrackSpecimen = BuildCrackSpecimen(
        cfg=inner_wf.CrackConfig.outputs.cfg, case=inner_wf.CrackConfig.outputs.case
    )
    inner_wf.CrackLoadingInput = CrackLoadingInput(
        cfg=inner_wf.CrackConfig.outputs.cfg,
        case=inner_wf.CrackConfig.outputs.case,
        meta=inner_wf.BuildCrackSpecimen,
    )
    inner_wf.RunLammpsBase = RunLammpsBase(
        io_bundle=inner_wf.CrackLoadingInput, lmp_command=lmp_command
    )
    inner_wf.AnalyzeCrackCase = AnalyzeCrackCase(
        cfg=inner_wf.CrackConfig.outputs.cfg,
        case=inner_wf.CrackConfig.outputs.case,
        io_bundle=inner_wf.RunLammpsBase.outputs.io_bundle,
    )
    return (
        inner_wf.AnalyzeCrackCase.outputs.result,
        inner_wf.AnalyzeCrackCase.outputs.steps,
    )
