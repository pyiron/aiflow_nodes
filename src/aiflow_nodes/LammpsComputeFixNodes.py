"""Staging module: multi-entry LAMMPS compute and fix section nodes (new, untested).

One node per section; several entries in one string, separated by ';' or
newlines, each `<id> <group> <style> [args...]` (a leading `compute`/`fix`
keyword is accepted and dropped):

    LammpsComputes(io_bundle, computes="stressperatom all stress/atom NULL; pe all pe/atom")
      -> compute block -> AssembleLammpsInput(compute=...)
    LammpsFixes(io_bundle, fixes="freezex all setforce 0.0 NULL NULL")
      -> (fix, fix_ids) -> AssembleLammpsInput(fix=..., fix_ids=...)

Entries render in the order given (a compute that uses another, e.g.
`reduce sum c_stressperatom[1]`, goes after it). Args are raw LAMMPS text:
no unit conversion. Move into atomistic/engine/lammps_input.py once tested.
"""

from __future__ import annotations

from aiflow.core import as_function_node

from aiflow_nodes.atomistic.engine.lammps import LammpsIOBundle
from aiflow_nodes.atomistic.engine.lammps_input import _render


def _parse_entries(spec, keyword: str) -> list[tuple[str, str, str, str]]:
    """'id group style args; ...' -> [(id, group, style, args), ...]."""
    if spec is None:
        return []
    entries, seen = [], set()
    for raw in str(spec).replace(";", "\n").splitlines():
        words = raw.split()
        if not words:
            continue
        if words[0] == keyword:
            words = words[1:]
        if len(words) < 3:
            raise ValueError(
                f"{keyword} entry needs '<id> <group> <style> [args]', got {raw.strip()!r}"
            )
        entry_id, group, style, args = words[0], words[1], words[2], " ".join(words[3:])
        if entry_id in seen:
            raise ValueError(f"duplicate {keyword} id {entry_id!r}")
        seen.add(entry_id)
        entries.append((entry_id, group, style, args))
    return entries


@as_function_node(["compute", "compute_ids"])
def LammpsComputes(io_bundle: LammpsIOBundle, computes: str = ""):
    """Compute block: one `compute <id> <group> <style> <args>` line per entry
    of `computes` (';' or newline separated, in that order). Wire `compute`
    into AssembleLammpsInput's `compute` port (rendered after region/group,
    before dump/thermo, so a dump or thermo can use `c_<id>`). An empty
    `computes` gives an empty block (section omitted).
    """
    entries = _parse_entries(computes, "compute")
    compute_ids = [e[0] for e in entries]
    compute = _render("computes.j2", entries=entries) if entries else ""
    return compute, compute_ids


@as_function_node(["fix", "fix_ids"])
def LammpsFixes(io_bundle: LammpsIOBundle, fixes: str = ""):
    """Several fixes in one node: one `fix <id> <group> <style> <args>` line
    per entry of `fixes` (';' or newline separated, in that order). Wire
    `fix`/`fix_ids` into AssembleLammpsInput's existing `fix`/`fix_ids` ports;
    all ids are unfixed by its usual rule (after the next `run`, or after the
    minimize block when a minimize stage has no `run`).
    """
    entries = _parse_entries(fixes, "fix")
    fix_ids = [e[0] for e in entries]
    fix = _render("fixes.j2", entries=entries) if entries else ""
    return fix, fix_ids
