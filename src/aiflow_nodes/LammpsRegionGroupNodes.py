"""Staging module: LAMMPS region and group section nodes (new, untested).

For a crack cylinder's frozen outer ring:

    region ring cylinder z cx cy (R - w) INF INF side out units box   # LammpsCylinderRegion
    group ring region ring                                            # LammpsGroup
    fix freeze ring setforce 0.0 0.0 0.0                              # LammpsFix (existing)

Wire `region`/`group` into AssembleLammpsInput's `region`/`group` ports (rendered
after read_data/pair_style, before dump/fix/calculation), and `group_id` into
LammpsFix's `group`. Move into atomistic/engine/lammps_input.py once tested.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from aiflow.core import as_function_node

from aiflow_nodes.atomistic.engine.lammps import LammpsIOBundle
from aiflow_nodes.atomistic.engine.lammps_input import _render

# units whose distance unit is the Angstrom (positions/radii are written unconverted)
_ANGSTROM_UNITS = ("metal", "real")


def _single_word(name: str, value) -> str:
    value = str(value).strip()
    if not value or any(c.isspace() for c in value):
        raise ValueError(f"{name} must be a single non-empty word, got {value!r}")
    return value


@as_function_node(["region", "region_id"])
def LammpsCylinderRegion(
    io_bundle: LammpsIOBundle,
    radius_nm: float,
    ring_width_A: float = 20.0,
    region_id: str = "ring",
    side: Literal["in", "out"] = "out",
):
    """Cylinder region along z, centred on the crack tip, for a frozen outer ring:
    `region <region_id> cylinder z cx cy (R - ring_width) INF INF side <side> units box`.

    `radius_nm` is the specimen radius R, the same value given to CutCylinder.
    The region's radius is R - ring_width_A, so `side="out"` (default) selects
    the outer ring r > R - ring_width_A, and `side="in"` the mobile core.
    The axis (cx, cy) is the centre of the cell's x-y face of
    `io_bundle.structure`, which is the crack tip in the crack nodes'
    convention (CrackNodes._xy_centre). INF INF: the whole periodic z length.
    """
    if io_bundle.units not in _ANGSTROM_UNITS:
        raise ValueError(
            f"LammpsCylinderRegion writes Angstrom; units {io_bundle.units!r} not supported"
        )
    region_id = _single_word("region_id", region_id)
    if side not in ("in", "out"):
        raise ValueError(f"side must be 'in' or 'out', got {side!r}")
    radius = 10.0 * float(radius_nm) - float(ring_width_A)
    if radius <= 0.0:
        raise ValueError(
            f"ring_width_A {ring_width_A} >= radius {10.0 * float(radius_nm)} A: no mobile core left"
        )
    structure = io_bundle.structure
    cell = np.array(structure.cell)
    centre = 0.5 * (cell[0] + cell[1]) + np.asarray(structure.get_celldisp()).ravel()
    region = _render(
        "region.j2",
        region_id=region_id,
        cx=float(centre[0]),
        cy=float(centre[1]),
        radius=radius,
        side=side,
    )
    return region, region_id


@as_function_node("neighbor")
def LammpsNeighbor(
    io_bundle: LammpsIOBundle,
    skin: float = 1.0,
    style: Literal["bin", "nsq", "multi"] = "bin",
    every: int = 1,
    delay: int = 0,
    check: Literal["yes", "no"] = "yes",
):
    """Neighbor-list settings, for AssembleLammpsInput's `neighbor` port:

        neighbor <skin> <style>
        neigh_modify every <every> delay <delay> check <check>

    `skin` [A] is written unconverted (metal/real units). Without this node
    LAMMPS uses its defaults (metal: skin 2.0 bin), which changes the neighbor
    lists and so the order forces are summed in. The defaults here are the
    settings of Onur's production runs (wcrack loading_control).
    """
    if io_bundle.units not in _ANGSTROM_UNITS:
        raise ValueError(
            f"LammpsNeighbor writes the skin in A; units {io_bundle.units!r} not supported"
        )
    if float(skin) < 0:
        raise ValueError(f"skin must be >= 0, got {skin}")
    neighbor = _render(
        "neighbor.j2",
        skin=float(skin),
        style=_single_word("style", style),
        every=int(every),
        delay=int(delay),
        check=_single_word("check", check),
    )
    return neighbor


def _id_ranges(ids, per_line: int = 16) -> str:
    """1-based ids as LAMMPS `group ... id` arguments: consecutive runs written
    as `a:b`, wrapped with `&` continuation lines."""
    ids = np.unique(np.asarray(ids, int))
    breaks = np.flatnonzero(np.diff(ids) != 1) + 1
    tokens = [
        f"{run[0]}" if len(run) == 1 else f"{run[0]}:{run[-1]}"
        for run in np.split(ids, breaks)
    ]
    lines = [
        " ".join(tokens[i : i + per_line]) for i in range(0, len(tokens), per_line)
    ]
    return " &\n    ".join(lines)


@as_function_node(["group", "group_id"])
def LammpsGroupFromStructure(io_bundle: LammpsIOBundle, group_id: str = "ring"):
    """`group <group_id> id ...`: the atoms of `io_bundle.structure` held by an
    ASE FixAtoms constraint (selective dynamics [False, False, False]), e.g.
    the ring set by CrackNodes.FreezeOuterRing on the undisplaced specimen.

    LAMMPS ids are the structure's atom order + 1 (as CreateLammpsStructure
    writes the data file). Wire `group_id` into LammpsFix's `group`, e.g.
    LammpsFix(fix_id="freeze", style="setforce", args="0.0 0.0 0.0").
    Raises if there is no FixAtoms, or a constraint of another kind (partly
    fixed atoms are not handled here).
    """
    from ase.constraints import FixAtoms

    group_id = _single_word("group_id", group_id)
    if group_id == "all":
        raise ValueError("group_id 'all' is LAMMPS's built-in group; pick another name")
    constraints = io_bundle.structure.constraints
    other = [c for c in constraints if not isinstance(c, FixAtoms)]
    if other:
        raise ValueError(f"only FixAtoms is handled, the structure also has {other}")
    fixed = sorted({int(i) for c in constraints for i in c.get_indices()})
    if not fixed:
        raise ValueError(
            "the structure has no FixAtoms constraint: apply FreezeOuterRing first"
        )
    group = _render(
        "group_ids.j2", group_id=group_id, ids=_id_ranges(np.array(fixed) + 1)
    )
    return group, group_id


@as_function_node(["group", "group_id"])
def LammpsGroup(
    io_bundle: LammpsIOBundle,
    region_id: str = "ring",
    group_id: str = "ring",
):
    """`group <group_id> region <region_id>`: every atom inside the region at the
    point the command runs (membership is fixed then; it does not follow atoms
    that move later).

    Returns the line and `group_id`, to wire into LammpsFix's `group`.
    """
    region_id = _single_word("region_id", region_id)
    group_id = _single_word("group_id", group_id)
    if group_id == "all":
        raise ValueError("group_id 'all' is LAMMPS's built-in group; pick another name")
    group = _render("group.j2", group_id=group_id, region_id=region_id)
    return group, group_id
