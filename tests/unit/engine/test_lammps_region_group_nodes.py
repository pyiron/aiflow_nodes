"""Tests for the region and group section nodes of LammpsRegionGroupNodes, and
the input checks of LammpsNeighbor / LammpsGroupFromStructure. The rendering of
LammpsNeighbor and the id ranges are in test_lammps_staging_nodes.py."""

import pytest
from ase import Atoms
from ase.build import bulk
from ase.constraints import FixAtoms, FixedLine

from aiflow_nodes.atomistic.engine.lammps import LammpsIOBundle
from aiflow_nodes.LammpsRegionGroupNodes import (
    LammpsCylinderRegion,
    LammpsGroup,
    LammpsGroupFromStructure,
    LammpsNeighbor,
)


def _bundle(structure=None, units="metal"):
    structure = structure if structure is not None else bulk("W", cubic=True)
    # these nodes never read the potential
    return LammpsIOBundle(structure=structure, potential="unused", units=units)


@pytest.fixture
def specimen():
    """An 80 x 60 Å cell, the crack tip at its x-y centre (40, 30)."""
    return Atoms("W", positions=[[40.0, 30.0, 1.0]], cell=[80.0, 60.0, 3.2], pbc=True)


class TestLammpsCylinderRegion:
    def test_outer_ring_around_the_cell_centre(self, specimen):
        region, region_id = LammpsCylinderRegion._original_func(
            _bundle(specimen), radius_nm=2.5, ring_width_A=5.0
        )
        assert region_id == "ring"
        assert region == (
            "region ring cylinder z 40.000000 30.000000 20.000000 "
            "INF INF side out units box\n"
        )

    def test_inner_side_and_custom_id(self, specimen):
        region, region_id = LammpsCylinderRegion._original_func(
            _bundle(specimen), radius_nm=2.5, region_id="core", side="in"
        )
        assert region_id == "core"
        assert "region core cylinder z" in region and "side in" in region

    def test_follows_a_legacy_celldisp(self, specimen):
        specimen.set_celldisp([1.0, -2.0, 0.0])
        region, _ = LammpsCylinderRegion._original_func(
            _bundle(specimen), radius_nm=2.5
        )
        assert "cylinder z 41.000000 28.000000" in region

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"units": "lj"}, "units 'lj' not supported"),
            ({"ring_width_A": 25.0}, "no mobile core left"),
            ({"side": "both"}, "side must be"),
            ({"region_id": "two words"}, "single non-empty word"),
            ({"region_id": " "}, "single non-empty word"),
        ],
    )
    def test_invalid_inputs(self, specimen, kwargs, match):
        units = kwargs.pop("units", "metal")
        with pytest.raises(ValueError, match=match):
            LammpsCylinderRegion._original_func(
                _bundle(specimen, units=units), radius_nm=2.5, **kwargs
            )


class TestLammpsGroup:
    def test_renders_a_region_group(self):
        group, group_id = LammpsGroup._original_func(
            _bundle(), region_id="ring", group_id="frozen"
        )
        assert (group, group_id) == ("group frozen region ring\n", "frozen")

    def test_rejects_the_builtin_all_group(self):
        with pytest.raises(ValueError, match="built-in group"):
            LammpsGroup._original_func(_bundle(), group_id="all")

    def test_rejects_a_multi_word_id(self):
        with pytest.raises(ValueError, match="single non-empty word"):
            LammpsGroup._original_func(_bundle(), region_id="a b")


class TestLammpsGroupFromStructureChecks:
    def test_rejects_the_builtin_all_group(self):
        structure = bulk("W", cubic=True)
        structure.set_constraint(FixAtoms(indices=[0]))
        with pytest.raises(ValueError, match="built-in group"):
            LammpsGroupFromStructure._original_func(_bundle(structure), group_id="all")

    def test_rejects_other_constraints(self):
        structure = bulk("W", cubic=True)
        structure.set_constraint(FixedLine(0, [0, 0, 1]))
        with pytest.raises(ValueError, match="only FixAtoms"):
            LammpsGroupFromStructure._original_func(_bundle(structure))

    def test_needs_a_fixatoms_constraint(self):
        with pytest.raises(ValueError, match="no FixAtoms"):
            LammpsGroupFromStructure._original_func(_bundle())


class TestLammpsNeighborChecks:
    def test_rejects_non_angstrom_units(self):
        with pytest.raises(ValueError, match="units 'lj'"):
            LammpsNeighbor._original_func(_bundle(units="lj"))

    def test_rejects_a_negative_skin(self):
        with pytest.raises(ValueError, match="skin must be >= 0"):
            LammpsNeighbor._original_func(_bundle(), skin=-0.5)

    def test_custom_settings(self):
        text = LammpsNeighbor._original_func(
            _bundle(units="real"), skin=2.0, style="nsq", every=5, delay=10, check="no"
        )
        assert text == "neighbor 2.0 nsq\nneigh_modify every 5 delay 10 check no\n"
