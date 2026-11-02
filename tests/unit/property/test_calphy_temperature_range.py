"""Checks for the calphy temperature-window estimator.

``EstimateCalphyTemperatureRange`` used to run its ramps at *constant volume*.
A crystal cannot melt without expanding by roughly 5 % linearly, so a fixed
cell superheats it enormously and the estimate came out far too high -- for
``1999--Mishin-Y--Al`` the returned window was 1095-2343 K against a melting
point near 1000 K, i.e. its lower bound already sat above the melting point and
no solid/liquid crossing existed inside it.  It also detected freezing from
atomic displacement, which cannot work: a cooling liquid's displacement falls
smoothly through any fixed threshold as its diffusivity drops.

The MD is far too expensive for a unit test, so what is covered here is the
decision logic (:func:`_highest_stable_temperature`, the window clamp, the two
error paths) plus the order parameter (:func:`_solid_fraction`), which is cheap
and is what the whole estimate now rests on.
"""

import numpy as np
import pytest

pytest.importorskip("calphy")
pytest.importorskip("pyscal3")

from aiflow_nodes.atomistic.property import calphy as calphy_nodes  # noqa: E402
from aiflow_nodes.atomistic.property.calphy import (  # noqa: E402
    EstimateCalphyTemperatureRange,
    _highest_stable_temperature,
    _solid_fraction,
)

# ── the bisection ──────────────────────────────────────────────────────────


def test_bisection_converges_on_the_stability_edge():
    """Four halvings of a 100-700 K bracket must land within ~40 K."""
    probed = []

    def is_stable(T):
        probed.append(T)
        return T < 500.0

    found = _highest_stable_temperature(is_stable, 100.0, 700.0, 4)
    assert found < 500.0  # never reports an unstable temperature as stable
    assert 500.0 - found <= (700.0 - 100.0) / 2**4
    assert len(probed) == 4  # one hold per refinement, no wasted MD


def test_bisection_returns_a_probed_temperature_not_an_interpolation():
    """The bound has to be a temperature that was actually simulated."""
    probed = []

    def is_stable(T):
        probed.append(T)
        return T < 480.0

    found = _highest_stable_temperature(is_stable, 100.0, 700.0, 5)
    assert found in probed


def test_bisection_reports_none_when_nothing_is_stable():
    """The untested lower bracket must never be returned as an answer.

    The node seeds the bracket at half the ramp's melting temperature, which
    it has not probed; returning that on total failure would hand calphy a
    ``temperature_stop`` above the superheating limit.
    """
    assert _highest_stable_temperature(lambda T: False, 100.0, 700.0, 4) is None


def test_bisection_with_no_refinements_probes_nothing_and_finds_nothing():
    assert _highest_stable_temperature(lambda T: True, 100.0, 700.0, 0) is None


# ── the order parameter ────────────────────────────────────────────────────


def test_solid_fraction_of_a_perfect_crystal_is_one():
    from ase.build import bulk

    assert _solid_fraction(bulk("Al", cubic=True).repeat(3)) == pytest.approx(1.0)


def test_solid_fraction_of_a_scrambled_cell_is_low():
    """The same cell with randomised positions must not read as crystalline."""
    from ase.build import bulk

    atoms = bulk("Al", cubic=True).repeat(3)
    rng = np.random.default_rng(0)
    atoms.set_positions(rng.uniform(0, atoms.get_cell()[0, 0], size=(len(atoms), 3)))
    assert _solid_fraction(atoms) < 0.7  # calphy's tolerance.solid_fraction


def test_solid_fraction_survives_a_non_orthorhombic_cell():
    """The primitive fcc cell is triclinic; the array path must still work."""
    from ase.build import bulk

    primitive = bulk("Al").repeat(4)
    assert not np.allclose(
        np.asarray(primitive.get_cell()),
        np.diag(np.diagonal(np.asarray(primitive.get_cell()))),
    )
    assert _solid_fraction(primitive) == pytest.approx(1.0)


# ── where the lower bound is placed ────────────────────────────────────────


@pytest.fixture
def stub_ramp(monkeypatch):
    """Melt on the first ramp check, then hand the bisection a fixed answer.

    Leaves only the window arithmetic to be observed, which is the part that
    decides whether the liquid leg survives.
    """
    calls = {"n": 0}

    def fake_fraction(atoms):
        calls["n"] += 1
        return 1.0 if calls["n"] == 1 else 0.0  # crystalline at t_start, then gone

    monkeypatch.setattr(calphy_nodes, "_solid_fraction", fake_fraction)
    monkeypatch.setattr(
        calphy_nodes, "_highest_stable_temperature", lambda *a, **k: 1000.0
    )


def _window(**kwargs):
    from ase.build import bulk

    t_min, t_max, _ = EstimateCalphyTemperatureRange(
        structure=bulk("Al", cubic=True).repeat(2),
        engine=_FrozenEngine(),
        t_start=100.0,
        t_stop=2000.0,
        heating_rate_k_per_ps=1e5,  # one check spans the whole range
        check_every=2,
        n_hold_steps=2,
        n_refinements=1,
        **kwargs,
    ).run()
    return t_min, t_max


def test_the_lower_bound_is_window_fraction_of_the_upper_one(stub_ramp):
    """Not ``min_lambda`` of it.

    The liquid leg is anchored at ``temperature_min``, so a window as wide as
    reversible scaling tolerates asks for a liquid at half the melting point.
    That is the reported Cu failure: 692-1384 K aborted with
    ``SolidifiedError`` before producing any free energy at all.
    """
    t_min, t_max = _window()
    assert t_max == pytest.approx(1000.0)
    assert t_min == pytest.approx(850.0)


def test_min_lambda_is_a_floor_under_window_fraction(stub_ramp):
    t_min, t_max = _window(window_fraction=0.3, min_lambda=0.5)
    assert t_min / t_max == pytest.approx(0.5)


def test_a_wider_window_can_still_be_asked_for(stub_ramp):
    t_min, t_max = _window(window_fraction=0.95)
    assert t_min / t_max == pytest.approx(0.95)


# ── the error paths ────────────────────────────────────────────────────────


class _FrozenEngine:
    """An engine whose atoms never move, so every plateau reads as solid.

    Enough to exercise the "no melting below t_stop" branch without MD that
    actually goes anywhere.
    """

    class _Calc:
        implemented_properties = ["energy", "forces", "stress"]

        def get_property(self, name, atoms=None, allow_calculation=True):
            if name == "energy":
                return 0.0
            if name == "forces":
                return np.zeros((len(atoms), 3))
            if name == "stress":
                return np.zeros(6)
            return None

        def calculate(self, *args, **kwargs):
            pass

        def check_state(self, atoms):
            return []

        def get_potential_energy(self, atoms=None, force_consistent=False):
            return 0.0

        def get_forces(self, atoms=None):
            return np.zeros((len(atoms), 3))

        def get_stress(self, atoms=None):
            return np.zeros(6)

    calculator = _Calc()


def test_no_melting_below_t_stop_raises_instead_of_guessing():
    """A silently-plausible window is the failure mode this replaces."""
    from ase.build import bulk

    with pytest.raises(ValueError, match="no melting between"):
        EstimateCalphyTemperatureRange(
            structure=bulk("Al", cubic=True).repeat(2),
            engine=_FrozenEngine(),
            t_start=100.0,
            t_stop=300.0,
            heating_rate_k_per_ps=1e5,  # one check spans the whole range
            check_every=2,
            n_hold_steps=2,
            n_refinements=1,
        ).run()


def test_disordered_start_raises():
    from ase.build import bulk

    atoms = bulk("Al", cubic=True).repeat(2)
    rng = np.random.default_rng(1)
    atoms.set_positions(rng.uniform(0, atoms.get_cell()[0, 0], size=(len(atoms), 3)))

    with pytest.raises(ValueError, match="already disordered at t_start"):
        EstimateCalphyTemperatureRange(
            structure=atoms,
            engine=_FrozenEngine(),
            t_start=100.0,
            t_stop=300.0,
            heating_rate_k_per_ps=1e5,  # one check spans the whole range
            check_every=2,
            n_hold_steps=2,
            n_refinements=1,
        ).run()
