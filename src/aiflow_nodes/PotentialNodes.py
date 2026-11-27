"""Staging module: an interatomic potential input dataclass (new, untested).

Either a potential-table name, used as is, or four fields that fill a one-row
potential DataFrame (Name, Filename, Species, Config; Model = "Custom").
Wire it straight into CreateLammpsStructure's `potential`, which resolves it
(`resolve_potential`) into the bundle's potential; LammpsPotential and
ParseLammpsOutput then read that as before:

    wf.Potential = InputPotential(potential_name="1999--Mishin-Y--Al--LAMMPS--ipr1")
    # or
    wf.Potential = InputPotential(
        name="W-P_MEAM",
        filename="/path/WP.library.meam, /path/WP.parameter.meam",
        species="W",
        config="pair_style meam; pair_coeff * * WP.library.meam W P WP.parameter.meam W",
    )
    wf.LammpsStructure = CreateLammpsStructure(structure=..., potential=wf.Potential)

All fields are strings, so they survive a GUI save and a sweep pickle (lists
and DataFrames don't). Move to its proper place once tested.
"""

from __future__ import annotations

from aiflow.core import as_inp_dataclass_node


@as_inp_dataclass_node
class InputPotential:
    """An interatomic potential: set `potential_name`, or the four DataFrame fields.

    potential_name: a potential-table (iprpy) name, passed on unchanged.
    name: the DataFrame's Name (free text; "Custom" if empty).
    filename: comma-separated full paths of the potential files. As for table
        potentials, a token in `config` equal to a file's basename is replaced
        by its full path. Empty if `config` already holds full paths.
    species: comma-separated elements in LAMMPS type order ("W", "Ni, Al").
    config: the LAMMPS potential lines (pair_style, pair_coeff, ...), separated
        by newlines or ";".
    """

    potential_name: str | None = None
    name: str | None = None
    filename: str | None = None
    species: str | None = None
    config: str | None = None


def _items(text, sep=",") -> list[str]:
    return [s.strip() for s in str(text or "").split(sep) if s.strip()]


def resolve_potential(potential):
    """A potential as CreateLammpsStructure / lammpsparser take it.

    A string or DataFrame is returned unchanged. An InputPotential gives its
    `potential_name` string as is, or a one-row DataFrame (Name, Filename,
    Model = "Custom", Species, Config) built from the four fields. Raises if
    both or neither are given, or a listed file is missing.
    """
    import os

    import pandas as pd

    if isinstance(potential, (str, pd.DataFrame)):
        return potential
    if not isinstance(potential, InputPotential._original_dataclass):
        raise TypeError(
            f"potential must be a name string, a DataFrame or an InputPotential, got {type(potential)}"
        )

    p = potential
    fields = {k: getattr(p, k) for k in ("name", "filename", "species", "config")}
    given = [k for k, v in fields.items() if str(v or "").strip()]
    name_given = bool(str(p.potential_name or "").strip())
    if name_given and given:
        raise ValueError(
            f"set either potential_name or the DataFrame fields, not both (also given: {given})"
        )
    if name_given:
        return str(p.potential_name).strip()
    if not given:
        raise ValueError(
            "InputPotential is empty: set potential_name, or species and config"
        )

    species = _items(p.species)
    lines = [
        ln.strip()
        for chunk in str(p.config or "").splitlines()
        for ln in chunk.split(";")
        if ln.strip()
    ]
    if not species or not lines:
        raise ValueError("the DataFrame route needs species and config")
    files = _items(p.filename)
    missing = [f for f in files if not os.path.isfile(f)]
    if missing:
        raise FileNotFoundError(f"potential files not found: {missing}")
    by_base = {os.path.basename(f): f for f in files}
    config = [
        " ".join(by_base.get(tok, tok) for tok in ln.split()) + "\n" for ln in lines
    ]

    return pd.DataFrame(
        {
            "Name": [str(p.name).strip() if str(p.name or "").strip() else "Custom"],
            "Filename": [files],
            "Model": ["Custom"],
            "Species": [species],
            "Config": [config],
        }
    )
