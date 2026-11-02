"""Checks for the two nodes that keep a calphy melting-point workflow consistent.

Both exist to remove a class of mistake that fails silently.  ``CalphySystem``
bundles the crystal, the melt seed and the potential so that the window estimate
and every leg are wired to the same three things: a window measured with one
potential against legs run with another is meaningless without raising, and a
liquid cell that is not the crystal's cell makes the two per-atom free energies
incomparable.  ``ApplyTemperatureWindow`` lets one ``InputClass`` carry the
sampling settings for every pass while each pass stamps its own window on a
copy, because free energies computed with different switching lengths cannot be
compared and two hand-written copies of the same settings can drift apart.

Nothing here launches LAMMPS or calphy.
"""

import dataclasses

import pytest
from ase.build import bulk

pytest.importorskip("calphy")

from aiflow_nodes.atomistic.property.calphy import (  # noqa: E402
    ApplyTemperatureWindow,
    BuildCalphySystem,
    CalphySystem,
    InputClass,
)

POTENTIAL = "1999--Mishin-Y--Al--LAMMPS--ipr1"


def _system():
    crystal = bulk("Al", "fcc", cubic=True)
    seed = crystal.copy()
    seed.rattle(stdev=0.5, seed=1)
    return crystal, seed


# ── the bundle ─────────────────────────────────────────────────────────────


def test_the_bundle_is_one_port_not_three():
    """A single port is the point: three cannot be wired inconsistently."""
    crystal, seed = _system()
    node = BuildCalphySystem(
        structure=crystal, liquid_structure=seed, potential=POTENTIAL
    )
    assert list(node.outputs.keys()) == ["system"]


def test_splitting_the_bundle_returns_what_went_in():
    crystal, seed = _system()
    system = BuildCalphySystem(
        structure=crystal, liquid_structure=seed, potential=POTENTIAL
    ).run()

    structure, liquid_structure, potential = CalphySystem(input=system).run()
    assert potential == POTENTIAL
    assert structure == crystal
    assert liquid_structure == seed
    # The seed has to be the crystal's cell, or the two free energies are
    # per-atom numbers for different systems.
    assert len(liquid_structure) == len(structure)


def test_the_split_ports_are_named_for_the_leg_nodes_that_consume_them():
    """``SolidFreeEnergyWithTemp`` takes ``structure`` and ``potential``."""
    crystal, seed = _system()
    system = BuildCalphySystem(
        structure=crystal, liquid_structure=seed, potential=POTENTIAL
    ).run()
    assert list(CalphySystem(input=system).outputs.keys()) == [
        "structure",
        "liquid_structure",
        "potential",
    ]


# ── stamping a window onto shared settings ─────────────────────────────────


def test_the_window_is_written_to_the_ports_calphy_reads():
    """``temperature`` is the thermostat setpoint, ``temperature_stop`` the top
    of the reversible-scaling sweep."""
    out = ApplyTemperatureWindow(
        inp=InputClass(temperature=1.0, temperature_stop=2.0).run(),
        temperature_min=900.0,
        temperature_max=1100.0,
    ).run()
    assert (out.temperature, out.temperature_stop) == (900.0, 1100.0)


def test_everything_except_the_window_survives_the_copy():
    """The settings are what makes two passes comparable, so they must not move
    when the window does."""
    settings = InputClass(
        n_equilibration_steps=2500,
        n_switching_steps=5000,
        n_iterations=1,
        pressure=0,
        npt=False,
        melting_cycle=False,
        equilibration_control="berendsen",
        cores=2,
    ).run()

    out = ApplyTemperatureWindow(
        inp=settings, temperature_min=900.0, temperature_max=1100.0
    ).run()

    moved = {
        f.name
        for f in dataclasses.fields(settings)
        if getattr(settings, f.name) != getattr(out, f.name)
    }
    assert moved == {"temperature", "temperature_stop"}


def test_the_shared_settings_are_not_mutated():
    """One ``InputClass`` feeds every pass; a pass that edited it in place would
    change the window of the passes around it."""
    settings = InputClass(temperature=1.0, temperature_stop=2.0).run()

    ApplyTemperatureWindow(
        inp=settings, temperature_min=900.0, temperature_max=1100.0
    ).run()

    assert (settings.temperature, settings.temperature_stop) == (1.0, 2.0)


def test_two_passes_stamp_independent_windows_on_the_same_settings():
    settings = InputClass(n_switching_steps=5000).run()

    first = ApplyTemperatureWindow(
        inp=settings, temperature_min=300.0, temperature_max=370.0
    ).run()
    second = ApplyTemperatureWindow(
        inp=settings, temperature_min=305.0, temperature_max=360.0
    ).run()

    assert (first.temperature, first.temperature_stop) == (300.0, 370.0)
    assert (second.temperature, second.temperature_stop) == (305.0, 360.0)
    assert first.n_switching_steps == second.n_switching_steps == 5000


def test_the_window_ports_are_named_after_the_nodes_that_produce_one():
    """``EstimateCalphyTemperatureRange`` and ``SuggestTemperatureWindow`` both
    hand back ``temperature_min`` / ``temperature_max``, so either drops in."""
    node = ApplyTemperatureWindow(inp=InputClass().run())
    assert "temperature_min" in node.inputs
    assert "temperature_max" in node.inputs
    assert list(node.outputs.keys()) == ["inp"]
