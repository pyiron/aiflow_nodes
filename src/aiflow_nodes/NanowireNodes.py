"""Staging module: FCC nanowire construction as separate nodes (new, 2026-11-17).

`04_01_Nanowires/NanowireGenerator_v2.make_nanowire` broken into steps; the
arithmetic is taken line by line from `NanowireGenerator_v2.py` and
`NanowireHelper_v2.py`, so the wire is the same (same atoms, same order):

  NanowireAxes (direction matrix, rolled to wire_axis)
  -> NanowireSupercell (auto radial repeat counts from facet_distance)
  -> CrackNodes.OrientedCubicCell (fcc; the FaceCenteredCubic bulk block)  [existing]
  -> NanowireFacetDirections ({hkl} family -> radial facet normals in the 2D plane)
  -> WireAxisCenter (geometric centre snapped to the nearest atomic column)
  -> CutWireByFacets (whole layers within facet_distance spacings of the centre)
  -> TrimSurfaceLayers (remove n_trim layers per side along the radial directions)
  -> CrackNodes.SetPBC("True, False, False")                               [existing]
  -> NanowireMetadata (make_nanowire's metadata and inputs dicts, for recording)

Production settings of the 04_01 sweeps (RunNanowireSweep_auditdriven_v2):
miller "0, 1, 1" (-> x = [110] wire axis, y = [-110], z = [001]), wire_axis 0,
facets "1, 1, 1", supercell "20, 80, 80" with auto (margin 1.6, min 4),
layer_tolerance 0.05, trim_tolerance 0.001, (facet_distance, trim_layers) in
(12,3) (15,3) (18,3) (22,3) (24,4) (28,5) (30,5).

Vectors are comma-separated strings so they survive a GUI save. Move to its
proper place once tested.
"""

from __future__ import annotations

from aiflow.core import as_function_node

# NanowireHelper_v2.get_direction_matrix table (rows = x, y, z before the roll)
_DIRECTIONS = {
    "001": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    "110": [[-1, 1, 0], [0, 0, 1], [1, 1, 0]],
    "111": [[1, -1, 0], [1, 1, -2], [1, 1, 1]],
}


def _ints(text, n=3, name="value") -> list[int]:
    items = [
        s
        for s in str(text)
        .replace("(", " ")
        .replace(")", " ")
        .replace("[", " ")
        .replace("]", " ")
        .replace(",", " ")
        .split()
        if s
    ]
    if len(items) != n:
        raise ValueError(f"{name} needs {n} integers, got {text!r}")
    return [int(float(s)) for s in items]


def _vec_str(v) -> str:
    return ", ".join(str(int(c)) for c in v)


def _direction_matrix(miller_indices, wire_axis):
    """NanowireHelper_v2.get_direction_matrix, verbatim logic."""
    import numpy as np

    mi = tuple(miller_indices) if miller_indices else ()
    if mi in [(0, 0, 1), (1, 0, 0), (0, 1, 0)] or not mi:
        key = "001"
    elif mi in [(0, 1, 1), (1, 1, 0), (1, 0, 1)]:
        key = "110"
    else:
        key = "".join(str(i) for i in mi)
    if key not in _DIRECTIONS:
        raise ValueError(
            f"no direction matrix for miller_indices {mi} (supported: 001, 110, 111 families)"
        )
    matrix = np.array(_DIRECTIONS[key], dtype=int)
    shift = int(wire_axis) - 2
    if shift != 0:
        matrix = np.roll(matrix, shift, axis=0)
    return matrix


def _rows(x, y, z):
    import numpy as np

    return np.array([_ints(x, 3, "x"), _ints(y, 3, "y"), _ints(z, 3, "z")], dtype=int)


@as_function_node(["x", "y", "z"])
def NanowireAxes(miller_indices: str = "0, 1, 1", wire_axis: int = 0):
    """Lattice directions of the cell axes x, y, z for a wire along
    `miller_indices` (make_nanowire's direction matrix: 001, 110 or 111
    family), rolled so that the wire-axis direction is cell axis `wire_axis`.
    Default ("0, 1, 1", 0): x = [110] (wire axis), y = [-110], z = [001]."""
    m = _direction_matrix(_ints(miller_indices, 3, "miller_indices"), wire_axis)
    x, y, z = (_vec_str(r) for r in m)
    return x, y, z


@as_function_node(["size", "radial_size"])
def NanowireSupercell(
    element: str,
    acell: float,
    x: str,
    y: str,
    z: str,
    facet_indices: str = "1, 1, 1",
    facet_distance: float = 12,
    supercell: str = "20, 80, 80",
    wire_axis: int = 0,
    auto: bool = True,
    margin: float = 1.6,
    min_size: int = 4,
):
    """Repeat counts of the bulk block (for OrientedCubicCell `size`).

    With `auto` (make_nanowire auto_supercell): the two radial counts are
    max(min_size, ceil(2 * facet_distance * acell/|hkl| * margin / unit length))
    per axis, both set to the larger one (square transverse box); only the
    wire-axis entry of `supercell` is kept. Without `auto`: `supercell` as given.
    Returns the size string and the radial count."""
    import numpy as np
    from ase.lattice.cubic import FaceCenteredCubic

    dims = _ints(supercell, 3, "supercell")
    radial_axes = [i for i in range(3) if i != int(wire_axis)]
    if not auto:
        return _vec_str(dims), int(max(dims[i] for i in radial_axes))
    directions = _rows(x, y, z)
    ref = FaceCenteredCubic(
        directions=directions.tolist(),
        size=(1, 1, 1),
        symbol=element,
        latticeconstant=acell,
        pbc=(1, 1, 1),
    )
    unit_lengths = np.linalg.norm(ref.get_cell(), axis=1)
    fi = _ints(facet_indices, 3, "facet_indices")
    interplanar_spacing = acell / np.linalg.norm(fi)
    physical_facet_distance = facet_distance * interplanar_spacing
    target_diameter = 2 * physical_facet_distance * margin
    per_axis = {
        ax: max(int(min_size), int(np.ceil(target_diameter / unit_lengths[ax])))
        for ax in radial_axes
    }
    radial_size = max(per_axis.values())
    for ax in radial_axes:
        dims[ax] = radial_size
    size = _vec_str(dims)
    return size, int(radial_size)


@as_function_node(["facet_directions", "facet_hkl", "facet_angles_deg", "hkl_family"])
def NanowireFacetDirections(
    facet_indices: str, x: str, y: str, z: str, wire_axis: int = 0
):
    """Radial facet normals of the {facet_indices} family in the wire's cell
    frame: the family (all permutations and signs), each normal rotated into
    the frame and normalised, those with a radial component kept, projected on
    the radial plane and de-duplicated (n ~ -n). make_nanowire steps 2-4.
    Returns the unit 2D directions (radial axes in increasing order), their
    hkl, the angles in degrees (sorted, as make_nanowire's metadata) and the
    family."""
    from itertools import permutations, product

    import numpy as np

    directions = _rows(x, y, z).astype(float)
    h, k, l = (abs(i) for i in _ints(facet_indices, 3, "facet_indices"))  # noqa: E741 (Miller indices)
    family = set()
    for perm in set(permutations([h, k, l])):
        for sign in product([1, -1], repeat=3):
            member = tuple(s * p for s, p in zip(sign, perm))
            if member != (0, 0, 0):
                family.add(member)
    hkl_family = list(family)

    normals = []
    for hkl in hkl_family:
        n_cart = directions @ np.array(hkl, dtype=float)
        norm = np.linalg.norm(n_cart)
        if norm > 1e-10:
            normals.append(n_cart / norm)

    radial_axes = [i for i in range(3) if i != int(wire_axis)]
    radial = [
        np.array(n)
        for n in normals
        if np.linalg.norm([n[i] for i in radial_axes]) > 1e-6
    ]
    radial_hkl = [
        hkl
        for hkl, n in zip(hkl_family, normals)
        if any(np.allclose(n, rf) for rf in radial)
    ]

    unique, unique_hkl = [], []
    for n, hkl in zip(radial, radial_hkl):
        n2 = np.array([n[i] for i in radial_axes])
        n2 = n2 / np.linalg.norm(n2)
        if not any(
            np.allclose(n2, u, atol=1e-6) or np.allclose(n2, -u, atol=1e-6)
            for u in unique
        ):
            unique.append(n2)
            unique_hkl.append(hkl)
    facet_directions = [u.tolist() for u in unique]
    facet_hkl = [list(h_) for h_ in unique_hkl]
    facet_angles_deg = sorted(float(np.degrees(np.arctan2(u[1], u[0]))) for u in unique)
    return facet_directions, facet_hkl, facet_angles_deg, [list(m) for m in hkl_family]


@as_function_node(["center", "geometric_center"])
def WireAxisCenter(structure, wire_axis: int = 0, column_tolerance: float = 0.1):
    """Wire axis position in the radial plane: the geometric centre of the
    cell, 0.5 (a + b + c), snapped to the nearest atomic column (columns found
    by rounding the radial coordinates to `column_tolerance`). make_nanowire
    step 5 (find_atomic_center)."""
    import numpy as np

    radial_axes = [i for i in range(3) if i != int(wire_axis)]
    cell = np.array(structure.get_cell())
    center_3d = 0.5 * (cell[0] + cell[1] + cell[2])
    geometric = np.array([center_3d[ax] for ax in radial_axes])
    radial_positions = structure.get_positions()[:, radial_axes]
    tol = float(column_tolerance)
    columns = np.unique(np.round(radial_positions / tol) * tol, axis=0)
    center = columns[np.argmin(np.linalg.norm(columns - geometric, axis=1))]
    return center.tolist(), geometric.tolist()


@as_function_node(
    [
        "structure",
        "facet_layer_counts",
        "physical_facet_distance",
        "interplanar_spacing",
    ]
)
def CutWireByFacets(
    structure,
    facet_directions,
    center,
    facet_distance: float,
    acell: float,
    facet_indices: str = "1, 1, 1",
    wire_axis: int = 0,
    layer_tolerance: float = 0.05,
):
    """Keep the atoms within the facet planes, on whole atomic layers.

    `facet_distance` counts {facet_indices} interplanar spacings (acell/|hkl|),
    not Angstrom. Along each facet normal, atoms are grouped into layers by
    rounding their centre-relative distance to `layer_tolerance`; a layer is
    kept if its true mean distance is within the physical facet distance.
    make_nanowire step 6 (cut_wire_by_facets). Returns the wire (atom order
    of the block kept), the number of kept layers per facet direction, the
    physical facet distance and the interplanar spacing [A]."""
    import numpy as np

    radial_axes = [i for i in range(3) if i != int(wire_axis)]
    interplanar_spacing = float(acell) / np.linalg.norm(
        _ints(facet_indices, 3, "facet_indices")
    )
    distance = float(facet_distance) * interplanar_spacing
    radial_positions = structure.get_positions()[:, radial_axes]
    center = np.asarray(center, float)
    tol = float(layer_tolerance)
    mask = np.ones(len(structure), dtype=bool)
    counts = []
    for n in facet_directions:
        dist = (radial_positions - center) @ np.asarray(n, float)
        rounded = np.round(dist / tol) * tol
        layers = np.unique(rounded)
        mean_dist = np.array([dist[rounded == lv].mean() for lv in layers])
        keep = layers[np.abs(mean_dist) <= distance]
        mask &= np.isin(rounded, keep)
        counts.append(int(len(keep)))
    wire = structure[mask]
    return wire, counts, float(distance), float(interplanar_spacing)


@as_function_node("structure")
def TrimSurfaceLayers(
    structure, n_trim: int = 3, tolerance: float = 0.001, wire_axis: int = 0
):
    """Remove `n_trim` outer layers on each side along three directions normal
    to the wire axis (ase.geometry.get_layers, Miller indices relative to the
    cell): wire_axis 0 -> (0,1,1), (0,-1,1), (0,0,1); 1 -> (1,0,1), (-1,0,1),
    (1,0,0); 2 -> (1,1,0), (1,-1,0), (0,1,0). A direction with too few layers
    is skipped. n_trim 0 leaves the structure unchanged (as make_nanowire).
    make_nanowire step 7 (trim_surface_layers)."""
    import ase.geometry as ageo

    if int(n_trim) <= 0:
        return structure
    directions = {
        0: [(0, 1, 1), (0, -1, 1), (0, 0, 1)],
        1: [(1, 0, 1), (-1, 0, 1), (1, 0, 0)],
        2: [(1, 1, 0), (1, -1, 0), (0, 1, 0)],
    }
    if int(wire_axis) not in directions:
        raise ValueError(f"wire_axis must be 0, 1 or 2, got {wire_axis}")
    result = structure.copy()
    for miller in directions[int(wire_axis)]:
        tags, _ = ageo.get_layers(result, miller=miller, tolerance=tolerance)
        lo, hi = tags.min() + int(n_trim), tags.max() - int(n_trim)
        if lo > hi:
            continue
        result = result[(tags >= lo) & (tags <= hi)]
    structure = result
    return structure


@as_function_node(["metadata", "inputs"])
def NanowireMetadata(
    structure,
    element: str,
    acell: float,
    miller_indices: str,
    size: str,
    facet_indices: str,
    facet_distance: float,
    hkl_family,
    facet_hkl,
    facet_directions,
    facet_angles_deg,
    center,
    geometric_center,
    facet_layer_counts,
    physical_facet_distance: float,
    interplanar_spacing: float,
    wire_axis: int = 0,
    trim_layers: int = 3,
    trim_tolerance: float = 0.001,
    layer_tolerance: float = 0.05,
):
    """make_nanowire's `metadata` and `inputs` dicts (same keys and types), for
    recording the wire (e.g. DBRecorder.record_nanowire)."""
    import numpy as np

    metadata = {
        "hkl_family": [tuple(m) for m in hkl_family],
        "unique_facet_hkl": [tuple(m) for m in facet_hkl],
        "num_facets": len(facet_directions),
        "facet_angles_deg": list(facet_angles_deg),
        "facet_directions_2d": [np.asarray(d, float) for d in facet_directions],
        "n_atoms": len(structure),
        "geometric_center_2d": np.asarray(geometric_center, float),
        "atomic_center_2d": np.asarray(center, float),
        "interplanar_spacing": float(interplanar_spacing),
        "physical_facet_distance": float(physical_facet_distance),
        "facet_layer_counts": {
            str(tuple(h)): int(c) for h, c in zip(facet_hkl, facet_layer_counts)
        },
    }
    inputs = {
        "element": element,
        "acell": acell,
        "miller_indices": tuple(_ints(miller_indices, 3, "miller_indices")),
        "supercell_dims": _ints(size, 3, "size"),
        "facet_indices": tuple(_ints(facet_indices, 3, "facet_indices")),
        "facet_distance": facet_distance,
        "wire_axis": int(wire_axis),
        "trim_layers": int(trim_layers),
        "trim_tolerance": trim_tolerance,
        "layer_tolerance": layer_tolerance,
    }
    return metadata, inputs
