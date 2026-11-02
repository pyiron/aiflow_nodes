"""Guard against the LAMMPS potential catalogue handing out unusable entries.

``pyiron_lammps.potential.update_potential_paths`` builds a
``{basename: full_path}`` substitution dict from each entry's ``Filename``.
OpenKIM entries carry ``Filename == ['']``, so the basename is ``''`` and
``line.replace('', resource_path)`` splices the resource path in between *every
character* of the ``pair_style`` and ``pair_coeff`` lines.  The result is a
LAMMPS deck that cannot possibly run, and because the damage happens inside
``get_potential_by_name`` no caller ever sees an intact ``Config`` to check.

That made 67 of the 136 aluminium potentials radioactive, and the trap was
camouflaged: every OpenKIM name starts with ``EAM_`` while the working ``ipr1``
names contain no ``eam`` substring at all, so the obvious "pick an EAM
potential" heuristic lands squarely on a broken one.  ``ListPotentials`` now
filters them out and ``get_usable_potential_by_name`` refuses them loudly.

These tests assert the *absence of the corruption* rather than the presence of
the filter, so they keep their meaning if upstream fixes the bug and the filter
is later relaxed.
"""

import pytest

pytest.importorskip("ase")
pytest.importorskip("pyiron_lammps")

from ase.build import bulk  # noqa: E402

from aiflow_nodes.atomistic.engine.lammps import (  # noqa: E402
    ListPotentials,
    _basenames_collide,
    _has_potential_files,
    _potential_rejection_reason,
    get_usable_potential_by_name,
)


def _resource_path():
    """The iprpy catalogue directory, or None when the data is not installed."""
    from pyiron_lammps.potential import get_resource_path_from_conda

    try:
        return get_resource_path_from_conda()
    except Exception:
        return None


requires_catalogue = pytest.mark.skipif(
    _resource_path() is None,
    reason="LAMMPS potential catalogue (iprpy resources) not installed",
)


@pytest.fixture(scope="module")
def al_potentials():
    """Names ``ListPotentials`` offers for a cubic fcc Al cell."""
    return ListPotentials(structure=bulk("Al", "fcc", a=4.05, cubic=True)).run()


@pytest.mark.parametrize(
    "filenames, expected",
    [
        ([""], False),  # the OpenKIM shape
        ([], False),
        (["  "], False),
        (None, False),
        (["potential_LAMMPS/x/Al.eam.alloy"], True),
        (["a.eam", ""], False),  # one blank name is enough to poison the lookup
    ],
)
def test_has_potential_files(filenames, expected):
    assert _has_potential_files(filenames) is expected


@pytest.mark.parametrize(
    "filenames, expected",
    [
        # the real MEAM shapes that break: library file embeds the element file
        (["p/library-Al.meam", "p/Al.meam"], True),
        (["p/library.AlCu.meam", "p/AlCu.meam"], True),
        (["p/Al.meam", "p/library-Al.meam"], True),  # order must not matter
        (["p/NiAlH_jea.eam.alloy"], False),
        (["p/library.meam", "p/Al.meam"], False),  # disjoint names are fine
        ([], False),
    ],
)
def test_basenames_collide(filenames, expected):
    assert _basenames_collide(filenames) is expected


def test_rejection_reason_distinguishes_the_two_failure_modes():
    assert _potential_rejection_reason(["p/Al.eam.alloy"]) is None
    assert "no parameter file" in _potential_rejection_reason([""])
    assert "substring" in _potential_rejection_reason(
        ["p/library-Al.meam", "p/Al.meam"]
    )


@requires_catalogue
def test_every_offered_potential_yields_an_intact_pair_style(al_potentials):
    """The real invariant: no listed name produces a corrupted LAMMPS deck."""
    from pyiron_lammps.potential import get_potential_by_name

    resource_path = _resource_path()
    corrupted = []
    for name in al_potentials:
        row = get_potential_by_name(potential_name=name)
        # Each parameter file contributes exactly one absolute path to the deck,
        # so the resource directory cannot appear more often than there are
        # files.  Splicing blows this bound wide open (~60 hits on one line),
        # while a legitimate multi-file MEAM potential names it exactly twice.
        max_paths = len(row["Filename"])
        config = row["Config"]
        if not config[0].strip().startswith("pair_style "):
            corrupted.append((name, config[0][:120]))
            continue
        # Later lines are free-form LAMMPS commands — COMB potentials add
        # `fix qeq ...`, some add `comm_modify ...` — so only the path budget
        # applies to them.
        for line in config:
            if line.count(resource_path) > max_paths:
                corrupted.append((name, line[:120]))
                break

    assert not corrupted, (
        f"{len(corrupted)} potential(s) corrupted, e.g. {corrupted[:2]}"
    )


@requires_catalogue
def test_openkim_entries_are_not_offered(al_potentials):
    # OpenKIM names carry the model-identifier infix; they have no local file.
    assert al_potentials, "no potentials offered for fcc Al at all"
    assert [n for n in al_potentials if "__MO_" in n] == []


@requires_catalogue
def test_first_offered_potential_is_unchanged_by_the_filter(al_potentials):
    """Several ``Workflows/h_diffusion_*.py`` select ``index=0`` positionally.

    The OpenKIM entries sort after the year-prefixed ones, so filtering must not
    move this name.  If it ever does, those workflows silently change physics.
    """
    assert al_potentials[0] == "1995--Angelo-J-E--Ni-Al-H--LAMMPS--ipr1"


@requires_catalogue
def test_get_usable_potential_by_name_rejects_an_openkim_entry():
    kim = "EAM_CubicNaturalSpline_ErcolessiAdams_1994_Al__MO_800509458712_002"
    with pytest.raises(ValueError, match="no parameter file"):
        get_usable_potential_by_name(kim)


@requires_catalogue
def test_get_usable_potential_by_name_passes_a_real_potential_through(al_potentials):
    potential = get_usable_potential_by_name(al_potentials[0])
    assert potential["Config"][0].strip().startswith("pair_style ")
    assert _has_potential_files(potential["Filename"])
