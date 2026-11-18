"""
PROD: FCC nanowire construction (no LAMMPS, no SLURM, no files written)
=======================================================================
Production workflow (from nanowire_construction_test.py, 2026-11-17; logic unchanged).
04_01_Nanowires `make_nanowire` (NanowireGenerator_v2) as separate nodes:

  NanowireAxes -> NanowireSupercell -> OrientedCubicCell (fcc block)
  -> NanowireFacetDirections, WireAxisCenter -> CutWireByFacets
  -> TrimSurfaceLayers -> SetPBC (True, False, False) -> NanowireMetadata

Validated 2026-11-17 (WORKFLOWS_new/DEBUG_nanowire_vs_make_nanowire.py, Ag,
a = 4.09 A): identical to make_nanowire for all 7 production (facet_distance,
trim_layers) pairs (12,3) (15,3) (18,3) (22,3) (24,4) (28,5) (30,5), 5780 to
47500 atoms: positions bit for bit in the same order, cell, pbc, resolved
supercell, facet layer counts, facet angles, centre.

Production settings of the 04_01 sweeps: <110> wire along x (miller "0, 1, 1",
wire_axis 0: x = [110], y = [-110], z = [001]), {111} facets, supercell
"20, 80, 80" with auto radial size (margin 1.6, min 4), layer tolerance 0.05,
trim tolerance 0.001. Inputs are the constants below; ELEMENT / ACELL must be
the potential's own values. Attach Plot3d to any `structure` output.
"""

from aiflow.core import Workflow

from aiflow_nodes.atomistic.structure.view import Plot3d
from aiflow_nodes.CrackNodes import OrientedCubicCell, SetPBC
from aiflow_nodes.NanowireNodes import (
    CutWireByFacets,
    NanowireAxes,
    NanowireFacetDirections,
    NanowireMetadata,
    NanowireSupercell,
    TrimSurfaceLayers,
    WireAxisCenter,
)

ELEMENT = "Ag"
ACELL = 4.09  # the potential's lattice constant [A] (Ag example)
MILLER = "0, 1, 1"
FACETS = "1, 1, 1"
WIRE_AXIS = 0
FACET_DISTANCE = 12  # {111} interplanar spacings from the axis to each facet
TRIM_LAYERS = 3

wf = Workflow("PROD_nanowire_construction")

wf.Axes = NanowireAxes(miller_indices=MILLER, wire_axis=WIRE_AXIS)
wf.Supercell = NanowireSupercell(
    element=ELEMENT,
    acell=ACELL,
    x=wf.Axes.outputs.x,
    y=wf.Axes.outputs.y,
    z=wf.Axes.outputs.z,
    facet_indices=FACETS,
    facet_distance=FACET_DISTANCE,
    supercell="20, 80, 80",
    wire_axis=WIRE_AXIS,
    auto=True,
    margin=1.6,
    min_size=4,
)
wf.Bulk = OrientedCubicCell(
    element=ELEMENT,
    lattice_constant=ACELL,
    crystalstructure="fcc",
    x=wf.Axes.outputs.x,
    y=wf.Axes.outputs.y,
    z=wf.Axes.outputs.z,
    size=wf.Supercell.outputs.size,
)
wf.Facets = NanowireFacetDirections(
    facet_indices=FACETS,
    x=wf.Axes.outputs.x,
    y=wf.Axes.outputs.y,
    z=wf.Axes.outputs.z,
    wire_axis=WIRE_AXIS,
)
wf.Center = WireAxisCenter(structure=wf.Bulk, wire_axis=WIRE_AXIS)
wf.Cut = CutWireByFacets(
    structure=wf.Bulk,
    facet_directions=wf.Facets.outputs.facet_directions,
    center=wf.Center.outputs.center,
    facet_distance=FACET_DISTANCE,
    acell=ACELL,
    facet_indices=FACETS,
    wire_axis=WIRE_AXIS,
    layer_tolerance=0.05,
)
wf.Trim = TrimSurfaceLayers(
    structure=wf.Cut.outputs.structure,
    n_trim=TRIM_LAYERS,
    tolerance=0.001,
    wire_axis=WIRE_AXIS,
)
wf.Wire = SetPBC(structure=wf.Trim, pbc="True, False, False")
wf.Metadata = NanowireMetadata(
    structure=wf.Wire,
    element=ELEMENT,
    acell=ACELL,
    miller_indices=MILLER,
    size=wf.Supercell.outputs.size,
    facet_indices=FACETS,
    facet_distance=FACET_DISTANCE,
    hkl_family=wf.Facets.outputs.hkl_family,
    facet_hkl=wf.Facets.outputs.facet_hkl,
    facet_directions=wf.Facets.outputs.facet_directions,
    facet_angles_deg=wf.Facets.outputs.facet_angles_deg,
    center=wf.Center.outputs.center,
    geometric_center=wf.Center.outputs.geometric_center,
    facet_layer_counts=wf.Cut.outputs.facet_layer_counts,
    physical_facet_distance=wf.Cut.outputs.physical_facet_distance,
    interplanar_spacing=wf.Cut.outputs.interplanar_spacing,
    wire_axis=WIRE_AXIS,
    trim_layers=TRIM_LAYERS,
    trim_tolerance=0.001,
    layer_tolerance=0.05,
)

wf.PlotBulk = Plot3d(structure=wf.Bulk)
wf.PlotCut = Plot3d(structure=wf.Cut.outputs.structure)
wf.PlotWire = Plot3d(structure=wf.Wire)
