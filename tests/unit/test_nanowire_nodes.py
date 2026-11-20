"""Tests for NanowireNodes: the make_nanowire steps as nodes, on a small Cu wire.

The chain was validated bit-identical against 04_01's make_nanowire for the
production sizes; that comparison needs 04_01's code, so here the steps are
checked for their geometric invariants, and the small wire's atom counts are
pinned as a regression guard.
"""

import numpy as np
import pytest

from aiflow_nodes.atomistic.structure.build import OrientedCubicCell
from aiflow_nodes.atomistic.structure.transform import SetPBC
from aiflow_nodes.NanowireNodes import (
    CutWireByFacets,
    NanowireAxes,
    NanowireFacetDirections,
    NanowireMetadata,
    NanowireSupercell,
    TrimSurfaceLayers,
    WireAxisCenter,
    _ints,
)

ACELL, FACET_DISTANCE = 3.6, 4


@pytest.fixture(scope="module")
def chain():
    """The default <110> wire, {111} facets, at a small facet distance."""
    x, y, z = NanowireAxes._original_func()
    size, radial = NanowireSupercell._original_func(
        "Cu", ACELL, x, y, z, facet_distance=FACET_DISTANCE, supercell="2, 8, 8"
    )
    block = OrientedCubicCell._original_func(
        "Cu", ACELL, crystalstructure="fcc", x=x, y=y, z=z, size=size
    )
    facets, facet_hkl, angles, family = NanowireFacetDirections._original_func(
        "1, 1, 1", x, y, z
    )
    center, geometric = WireAxisCenter._original_func(block)
    wire, counts, distance, spacing = CutWireByFacets._original_func(
        block, facets, center, FACET_DISTANCE, ACELL
    )
    trimmed = TrimSurfaceLayers._original_func(wire, n_trim=1)
    final = SetPBC._original_func(trimmed, pbc="True, False, False")
    return {
        "axes": (x, y, z),
        "size": size,
        "radial": radial,
        "block": block,
        "facets": facets,
        "facet_hkl": facet_hkl,
        "angles": angles,
        "family": family,
        "center": center,
        "geometric": geometric,
        "wire": wire,
        "counts": counts,
        "distance": distance,
        "spacing": spacing,
        "trimmed": trimmed,
        "final": final,
    }


class TestNanowireAxes:
    def test_default_puts_110_along_x(self):
        assert NanowireAxes._original_func() == ("1, 1, 0", "-1, 1, 0", "0, 0, 1")

    @pytest.mark.parametrize(
        "miller, axis", [("0, 1, 1", 0), ("0, 0, 1", 2), ("1, 1, 1", 1), ("1, 1, 1", 0)]
    )
    def test_axes_are_orthogonal_right_handed_and_the_wire_is_on_its_axis(
        self, miller, axis
    ):
        rows = np.array([_ints(v) for v in NanowireAxes._original_func(miller, axis)])
        gram = rows @ rows.T
        assert np.allclose(gram, np.diag(np.diag(gram)))
        assert np.dot(rows[0], np.cross(rows[1], rows[2])) > 0
        wire = rows[axis] / np.linalg.norm(rows[axis])
        target = np.array(_ints(miller), float)
        # the wire runs along a member of the requested family
        assert sorted(np.abs(wire * np.linalg.norm(target))) == pytest.approx(
            sorted(np.abs(target))
        )

    def test_rejects_an_unsupported_family(self):
        with pytest.raises(ValueError, match="no direction matrix"):
            NanowireAxes._original_func("1, 2, 3")

    def test_rejects_the_wrong_number_of_indices(self):
        with pytest.raises(ValueError, match="needs 3 integers"):
            NanowireAxes._original_func("1, 1")


class TestNanowireSupercell:
    def test_auto_gives_a_square_radial_box_and_keeps_the_wire_length(self, chain):
        dims = _ints(chain["size"])
        assert dims[0] == 2
        assert dims[1] == dims[2] == chain["radial"]
        block = chain["block"]
        lengths = block.cell.lengths()
        # the radial box covers 2 * facet distance * margin
        target = 2 * FACET_DISTANCE * ACELL / np.sqrt(3) * 1.6
        assert min(lengths[1:]) >= target

    def test_min_size_is_respected(self):
        x, y, z = NanowireAxes._original_func()
        _, radial = NanowireSupercell._original_func(
            "Cu", ACELL, x, y, z, facet_distance=0.5, min_size=6
        )
        assert radial == 6

    def test_without_auto_the_supercell_is_used_as_given(self):
        size, radial = NanowireSupercell._original_func(
            "Cu",
            ACELL,
            "1, 1, 0",
            "-1, 1, 0",
            "0, 0, 1",
            supercell="3, 5, 7",
            auto=False,
        )
        assert (size, radial) == ("3, 5, 7", 7)


class TestFacetsAndCut:
    def test_facet_directions_are_unit_radial_and_distinct(self, chain):
        dirs = np.array(chain["facets"])
        assert dirs.shape == (3, 2)
        assert np.allclose(np.linalg.norm(dirs, axis=1), 1)
        for i in range(3):
            for j in range(i + 1, 3):
                assert abs(abs(np.dot(dirs[i], dirs[j])) - 1) > 1e-6
        assert chain["angles"] == sorted(chain["angles"])
        assert len(chain["family"]) == 8  # {111}: 4 axes x 2 signs
        assert all(sorted(map(abs, h)) == [1, 1, 1] for h in chain["facet_hkl"])

    def test_center_is_an_atomic_column_near_the_cell_centre(self, chain):
        block, center = chain["block"], np.array(chain["center"])
        radial = block.get_positions()[:, 1:]
        assert np.min(np.linalg.norm(radial - center, axis=1)) < 0.1
        assert np.linalg.norm(center - chain["geometric"]) < ACELL

    def test_cut_keeps_whole_layers_inside_every_facet(self, chain):
        wire, center = chain["wire"], np.array(chain["center"])
        assert chain["spacing"] == pytest.approx(ACELL / np.sqrt(3))
        assert chain["distance"] == pytest.approx(FACET_DISTANCE * chain["spacing"])
        rel = wire.get_positions()[:, 1:] - center
        for n in chain["facets"]:
            assert np.all(np.abs(rel @ np.asarray(n)) <= chain["distance"] + 0.05)
        assert len(chain["counts"]) == 3

    def test_trim_removes_outer_layers_and_n_trim_0_is_a_no_op(self, chain):
        assert len(chain["trimmed"]) < len(chain["wire"])
        assert (
            TrimSurfaceLayers._original_func(chain["wire"], n_trim=0) is chain["wire"]
        )

    def test_trim_rejects_a_bad_wire_axis(self, chain):
        with pytest.raises(ValueError, match="wire_axis"):
            TrimSurfaceLayers._original_func(chain["wire"], n_trim=1, wire_axis=3)

    def test_trim_skips_a_direction_with_too_few_layers(self, chain):
        thin = TrimSurfaceLayers._original_func(chain["wire"], n_trim=50)
        assert len(thin) == len(chain["wire"])

    def test_small_wire_regression(self, chain):
        # pinned from this implementation (validated against make_nanowire)
        assert (len(chain["block"]), len(chain["wire"]), len(chain["trimmed"])) == (
            484,
            102,
            74,
        )
        assert list(chain["final"].pbc) == [True, False, False]


def test_metadata_matches_the_wire(chain):
    metadata, inputs = NanowireMetadata._original_func(
        chain["final"], "Cu", ACELL, "0, 1, 1", chain["size"], "1, 1, 1",
        FACET_DISTANCE, chain["family"], chain["facet_hkl"], chain["facets"],
        chain["angles"], chain["center"], chain["geometric"], chain["counts"],
        chain["distance"], chain["spacing"], trim_layers=1,
    )  # fmt: skip
    assert metadata["n_atoms"] == len(chain["final"])
    assert metadata["num_facets"] == 3
    assert set(metadata["facet_layer_counts"]) == {
        str(tuple(h)) for h in chain["facet_hkl"]
    }
    assert inputs["miller_indices"] == (0, 1, 1)
    assert inputs["supercell_dims"] == _ints(chain["size"])
    assert inputs["trim_layers"] == 1
