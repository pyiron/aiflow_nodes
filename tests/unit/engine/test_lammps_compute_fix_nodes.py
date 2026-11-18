"""Tests for the staged multi-entry LammpsComputes/LammpsFixes nodes
(LammpsComputeFixNodes, added 2026-11-17) and AssembleLammpsInput's `compute` port."""

import os
import sys
import unittest
from pathlib import Path

from ase.build import bulk

from aiflow_nodes.atomistic.engine.lammps import LammpsIOBundle
from aiflow_nodes.atomistic.engine.lammps_input import AssembleLammpsInput, LammpsDump
from aiflow_nodes.LammpsComputeFixNodes import (
    LammpsComputes,
    LammpsFixes,
    _parse_entries,
)

RESOURCE_PATH = os.environ.get(
    "IPRPY_RESOURCE_PATH", str(Path(sys.executable).parent.parent / "share" / "iprpy")
)

computes = LammpsComputes._original_func
fixes = LammpsFixes._original_func
assemble = AssembleLammpsInput._original_func


def _bundle(**kw):
    return LammpsIOBundle(
        structure=bulk("Al", cubic=True),
        potential="1999--Mishin-Y--Al--LAMMPS--ipr1",
        resource_path=RESOURCE_PATH,
        **kw,
    )


class TestParseEntries(unittest.TestCase):
    def test_semicolon_and_newline(self):
        spec = "a all stress/atom NULL; b all pe/atom\nc all reduce sum c_a[1]"
        self.assertEqual(
            _parse_entries(spec, "compute"),
            [
                ("a", "all", "stress/atom", "NULL"),
                ("b", "all", "pe/atom", ""),
                ("c", "all", "reduce", "sum c_a[1]"),
            ],
        )

    def test_leading_keyword_dropped_and_blanks_skipped(self):
        self.assertEqual(
            _parse_entries(" ; fix f1 all setforce 0.0 NULL NULL ;\n\n", "fix"),
            [("f1", "all", "setforce", "0.0 NULL NULL")],
        )

    def test_empty_and_none(self):
        self.assertEqual(_parse_entries("", "fix"), [])
        self.assertEqual(_parse_entries(None, "fix"), [])

    def test_short_entry_raises(self):
        with self.assertRaises(ValueError):
            _parse_entries("a all", "compute")

    def test_duplicate_id_raises(self):
        with self.assertRaises(ValueError):
            _parse_entries("a all pe/atom; a all ke/atom", "compute")


class TestLammpsComputes(unittest.TestCase):
    def test_lines_and_ids(self):
        block, ids = computes(
            io_bundle=_bundle(),
            computes="stressperatom all stress/atom NULL; pe all pe/atom",
        )
        self.assertEqual(
            block,
            "compute stressperatom all stress/atom NULL\ncompute pe all pe/atom\n",
        )
        self.assertEqual(ids, ["stressperatom", "pe"])

    def test_empty(self):
        self.assertEqual(computes(io_bundle=_bundle(), computes=""), ("", []))

    def test_rendered_after_group_before_dump(self):
        b = _bundle()
        block, _ = computes(io_bundle=b, computes="s all stress/atom NULL")
        s = assemble(
            io_bundle=b,
            init="read_data lammps.data\n",
            potential="pair_style x\n",
            group="group ring id 1\n",
            compute=block,
            dump="dump 1 all custom 1 d c_s[*]\n",
            thermo="thermo 1\n",
        ).lammps_input_string
        self.assertLess(s.index("group ring"), s.index("compute s all"))
        self.assertLess(s.index("compute s all"), s.index("dump 1"))
        self.assertLess(s.index("compute s all"), s.index("thermo 1"))

    def test_omitted_when_empty(self):
        s = assemble(
            io_bundle=_bundle(), init="read_data lammps.data\n"
        ).lammps_input_string
        self.assertNotIn("compute", s)


class TestLammpsFixes(unittest.TestCase):
    def test_lines_and_ids(self):
        block, ids = fixes(
            io_bundle=_bundle(),
            fixes="freezex all setforce 0.0 NULL NULL\nfix ring top setforce 0 0 0",
        )
        self.assertEqual(
            block,
            "fix freezex all setforce 0.0 NULL NULL\nfix ring top setforce 0 0 0\n",
        )
        self.assertEqual(ids, ["freezex", "ring"])

    def test_all_unfixed_after_run(self):
        b = _bundle()
        block, ids = fixes(
            io_bundle=b, fixes="f1 all setforce 0 0 0; f2 all setforce NULL 0 0"
        )
        s = assemble(
            io_bundle=b,
            init="read_data lammps.data\n",
            fix=block,
            fix_ids=ids,
            run=0,
        ).lammps_input_string
        lines = s.splitlines()
        i_run = lines.index("run 0")
        self.assertIn("unfix f1", lines[i_run + 1 :])
        self.assertIn("unfix f2", lines[i_run + 1 :])

    def test_minimize_without_run_unfixes_after_block(self):
        b = _bundle()
        block, ids = fixes(
            io_bundle=b, fixes="f1 all setforce 0 0 0; f2 all setforce NULL 0 0"
        )
        out = assemble(
            io_bundle=b,
            init="read_data lammps.data\n",
            fix=block,
            fix_ids=ids,
            calculation="min_style cg\nminimize 0.0 1e-08 100 1000\n",
            mode="minimize",
        )
        lines = out.lammps_input_string.splitlines()
        i_min = lines.index("minimize 0.0 1e-08 100 1000")
        self.assertLess(lines.index("fix f1 all setforce 0 0 0"), i_min)
        self.assertGreater(lines.index("unfix f1"), i_min)
        self.assertGreater(lines.index("unfix f2"), i_min)
        self.assertEqual(out.lammps_pending_fix_ids, "")


class TestLammpsDumpExtraColumns(unittest.TestCase):
    DEFAULT = (
        "variable dumptime equal 100\n"
        "dump 1 all custom ${dumptime} dump.out id type xsu ysu zsu fx fy fz vx vy vz\n"
        'dump_modify 1 sort id format line "%d %d %20.15g %20.15g %20.15g %20.15g '
        '%20.15g %20.15g %20.15g %20.15g %20.15g"\n'
    )

    def test_default_unchanged(self):
        self.assertEqual(LammpsDump._original_func(io_bundle=_bundle()), self.DEFAULT)

    def test_extra_columns(self):
        block = LammpsDump._original_func(
            io_bundle=_bundle(), extra_columns="  c_stressperatom[*]   c_pe "
        )
        self.assertEqual(
            block.splitlines(),
            [
                "variable dumptime equal 100",
                "dump 1 all custom ${dumptime} dump.out id type xsu ysu zsu fx fy fz vx vy vz"
                " c_stressperatom[*] c_pe",
                "dump_modify 1 sort id format float %20.15g",
            ],
        )


class TestParseComputeHelpers(unittest.TestCase):
    def _write_dump(self, header):
        import tempfile

        d = tempfile.mkdtemp()
        with open(os.path.join(d, "dump.out"), "w") as f:
            f.write("ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n2\n")
            f.write("ITEM: ATOMS " + header + "\n1 1 0 0 0\n2 1 0 0 0\n")
        return d

    def test_header_columns(self):
        from aiflow_nodes.atomistic.engine.lammps import _dump_compute_columns

        d = self._write_dump("id type xsu ysu zsu c_s[1] c_s[2] c_pe f_mean_forces[1]")
        self.assertEqual(_dump_compute_columns(d, "dump.out"), ["s", "pe"])
        d = self._write_dump("id type xsu ysu zsu fx fy fz")
        self.assertEqual(_dump_compute_columns(d, "dump.out"), [])
        self.assertEqual(_dump_compute_columns(d, "missing.out"), [])

    def test_collect_and_frame(self):
        import numpy as np

        from aiflow_nodes.atomistic.engine.lammps import _collect_computes, _frame

        # 2 frames, 3 atoms; keys as lammpsparser writes them (index order scrambled)
        generic = {
            "s[2]": [[2, 2, 2], [20, 20, 20]],
            "s[1]": [[1, 1, 1], [10, 10, 10]],
            "pe": [[-1, -2, -3], [-4, -5, -6]],
            "pressures": [0, 0],
        }
        c = _collect_computes(generic, ["s", "pe"])
        self.assertEqual(c["s"].shape, (2, 3, 2))
        np.testing.assert_array_equal(c["s"][1, 0], [10, 20])
        self.assertEqual(c["pe"].shape, (2, 3))
        last = _frame(c, -1)
        np.testing.assert_array_equal(last["pe"], [-4, -5, -6])
        self.assertEqual(_collect_computes(generic, []), {})


if __name__ == "__main__":
    unittest.main()
