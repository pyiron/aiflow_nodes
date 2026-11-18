"""Tests for the staged LAMMPS/crack nodes added 2026-11-17 (LammpsRegionGroupNodes,
CrackNodes.FreezeOuterRing/ReorderAtoms) and the new LammpsMinimize/AssembleLammpsInput ports."""

import os
import sys
import unittest
from pathlib import Path

import numpy as np
from ase.build import bulk
from ase.constraints import FixAtoms

from aiflow_nodes.atomistic.engine.lammps import LammpsIOBundle
from aiflow_nodes.atomistic.engine.lammps_input import (
    AssembleLammpsInput,
    LammpsMinimize,
)
from aiflow_nodes.CrackNodes import ReorderAtoms
from aiflow_nodes.LammpsRegionGroupNodes import (
    LammpsGroupFromStructure,
    LammpsNeighbor,
    _id_ranges,
)

RESOURCE_PATH = os.environ.get(
    "IPRPY_RESOURCE_PATH", str(Path(sys.executable).parent.parent / "share" / "iprpy")
)


def _bundle(structure=None, **kw):
    return LammpsIOBundle(
        structure=structure if structure is not None else bulk("Al", cubic=True),
        potential="1999--Mishin-Y--Al--LAMMPS--ipr1",
        resource_path=RESOURCE_PATH,
        **kw,
    )


class TestLammpsMinimizeExtras(unittest.TestCase):
    def test_defaults_unchanged(self):
        block = LammpsMinimize._original_func(io_bundle=_bundle(), max_iter=1000)
        self.assertNotIn("min_modify", block)
        self.assertNotIn("timestep", block)
        self.assertIn("minimize 0.0 0.0001 1000 1000\n", block)

    def test_min_modify_timestep_max_eval(self):
        block = LammpsMinimize._original_func(
            io_bundle=_bundle(),
            style="fire",
            f_tol=1e-6,
            max_iter=100000,
            max_eval=10000000,
            min_modify="norm two",
            timestep=1.0,
        )
        lines = block.splitlines()
        self.assertEqual(
            lines,
            [
                "min_style fire",
                "min_modify norm two",
                "timestep 0.001",
                "minimize 0.0 1e-06 100000 10000000",
            ],
        )


class TestLammpsNeighbor(unittest.TestCase):
    def test_default_lines(self):
        text = LammpsNeighbor._original_func(io_bundle=_bundle())
        self.assertEqual(
            text, "neighbor 1.0 bin\nneigh_modify every 1 delay 0 check yes\n"
        )

    def test_rendered_after_potential(self):
        b = _bundle()
        s = AssembleLammpsInput._original_func(
            io_bundle=b,
            init="read_data lammps.data\n",
            potential="pair_style x\n",
            neighbor=LammpsNeighbor._original_func(io_bundle=b),
            group="group ring id 1\n",
        ).lammps_input_string
        self.assertLess(s.index("pair_style x"), s.index("neighbor 1.0 bin"))
        self.assertLess(s.index("neigh_modify"), s.index("group ring"))

    def test_rejects_non_angstrom_units(self):
        with self.assertRaises(ValueError):
            LammpsNeighbor._original_func(io_bundle=_bundle(units="lj"))


class TestGroupFromStructure(unittest.TestCase):
    def test_id_ranges(self):
        self.assertEqual(_id_ranges([1, 2, 3, 7, 9, 10]), "1:3 7 9:10")

    def test_group_ids_from_fixatoms(self):
        at = bulk("Al", cubic=True).repeat(2)
        at.set_constraint(FixAtoms(indices=[0, 1, 2, 5]))
        group, gid = LammpsGroupFromStructure._original_func(io_bundle=_bundle(at))
        self.assertEqual(group, "group ring id 1:3 6\n")
        self.assertEqual(gid, "ring")


class TestReorderAtoms(unittest.TestCase):
    def test_none_is_identity(self):
        at = bulk("Al", cubic=True)
        self.assertIs(ReorderAtoms._original_func(structure=at), at)

    def test_permutation_moves_fixatoms(self):
        at = bulk("Al", cubic=True).repeat(2)
        at.set_constraint(FixAtoms(indices=[0, 3]))
        order = np.random.default_rng(0).permutation(len(at))
        new = ReorderAtoms._original_func(structure=at, order=order.tolist())
        np.testing.assert_array_equal(new.positions, at.positions[order])
        fixed_new = sorted(int(i) for c in new.constraints for i in c.get_indices())
        self.assertEqual(
            fixed_new, sorted(int(k) for k in np.flatnonzero(np.isin(order, [0, 3])))
        )

    def test_rejects_non_permutation(self):
        with self.assertRaises(ValueError):
            ReorderAtoms._original_func(
                structure=bulk("Al", cubic=True), order=[0, 0, 1, 2]
            )


if __name__ == "__main__":
    unittest.main()
