"""Checks for translating a catalogue potential into calphy's input.

calphy issues exactly one ``pair_coeff`` command and rewrites that single line
to scale the potential during reversible scaling.  A ``pair_style eam`` (funcfl)
catalogue entry carries one diagonal ``pair_coeff i i <file>`` line per element
instead, and the node used to keep only the first of them while still declaring
every element as an atom type -- so LAMMPS aborted mid-run with ``All pair
coeffs are not set``.  ``GetPotential(structure=Bulk("Cu"))`` selects exactly
such an entry (``1985--Foiles-S-M--Ni-Cu``), so this was reachable from a
one-line workflow.
"""

import pytest

pytest.importorskip("calphy")

from aiflow_nodes.atomistic.property.calphy import _single_pair_coeff  # noqa: E402

NI_CU = [
    "pair_coeff 1 1 /pot/1985--Foiles/Cu_smf7.eam",
    "pair_coeff 2 2 /pot/1985--Foiles/Ni_smf7.eam",
]


def test_a_single_pair_coeff_line_is_passed_through_untouched():
    """`eam/alloy` and friends already map every type in one line."""
    line = "pair_coeff * * /pot/Al99.eam.alloy Al"
    coeff, species = _single_pair_coeff("mishin", ["Al"], [line], {"Al"})
    assert coeff == line
    assert species == ["Al"]


def test_unused_species_are_not_dropped_from_a_single_line_potential():
    """A `* *` line names its own elements; the type list must keep matching it."""
    line = "pair_coeff * * /pot/NiAlH_jea.eam.alloy Ni Al H"
    coeff, species = _single_pair_coeff("angelo", ["Ni", "Al", "H"], [line], {"Al"})
    assert coeff == line
    assert species == ["Ni", "Al", "H"]


def test_per_type_potential_keeps_only_the_element_the_structure_uses():
    """Pure Cu in a Ni-Cu funcfl potential needs the Cu file and one type."""
    coeff, species = _single_pair_coeff("foiles", ["Cu", "Ni"], NI_CU, {"Cu"})
    assert coeff == "pair_coeff 1 1 /pot/1985--Foiles/Cu_smf7.eam"
    assert species == ["Cu"]


def test_the_surviving_type_is_renumbered_to_one():
    """Pure Ni is type 2 in that potential; with Cu gone it must become type 1."""
    coeff, species = _single_pair_coeff("foiles", ["Cu", "Ni"], NI_CU, {"Ni"})
    assert coeff == "pair_coeff 1 1 /pot/1985--Foiles/Ni_smf7.eam"
    assert species == ["Ni"]


def test_the_right_file_is_kept_from_a_six_element_potential():
    lines = [
        f"pair_coeff {i} {i} /pot/1986--Foiles/{el}_u3.eam"
        for i, el in enumerate(["Ag", "Au", "Cu", "Ni", "Pd", "Pt"], start=1)
    ]
    coeff, species = _single_pair_coeff(
        "foiles6", ["Ag", "Au", "Cu", "Ni", "Pd", "Pt"], lines, {"Cu"}
    )
    assert coeff == "pair_coeff 1 1 /pot/1986--Foiles/Cu_u3.eam"
    assert species == ["Cu"]


def test_an_alloy_structure_on_a_per_type_potential_raises():
    """Nothing can be reduced here, and a truncated input must not be built."""
    with pytest.raises(ValueError, match="one pair_coeff command"):
        _single_pair_coeff("foiles", ["Cu", "Ni"], NI_CU, {"Cu", "Ni"})


def test_a_structure_sharing_no_element_with_the_potential_raises():
    with pytest.raises(ValueError, match="one pair_coeff command"):
        _single_pair_coeff("foiles", ["Cu", "Ni"], NI_CU, {"Al"})


def test_the_error_names_a_usable_alternative():
    """The message has to say what to do next, not just what went wrong."""
    with pytest.raises(ValueError) as excinfo:
        _single_pair_coeff("foiles", ["Cu", "Ni"], NI_CU, {"Cu", "Ni"})
    message = str(excinfo.value)
    assert "eam/alloy" in message
    assert "ListPotentials" in message


def test_a_missing_diagonal_line_raises_rather_than_guessing():
    """An off-diagonal-only Config is not something to silently reinterpret."""
    with pytest.raises(ValueError, match="no single"):
        _single_pair_coeff(
            "odd",
            ["Cu", "Ni"],
            ["pair_coeff 1 2 /pot/a.eam", "pair_coeff 2 1 /pot/b.eam"],
            {"Cu"},
        )
