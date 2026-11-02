"""Checks for the self-correcting temperature window in ``CalphyMeltingTemperatureSearch``.

A melting-point window has to contain the melting point, which means the solid
leg is superheated at the top of it and the liquid undercooled at the bottom.
calphy does not degrade gracefully when either goes too far: it aborts with
``MeltedError`` or ``SolidifiedError``, and neither says where the window should
have been.  Worse, a window can miss the melting point entirely *without*
failing — above T_m the liquid is stable at the bottom and the solid superheats
past the top — and then two perfectly clean legs produce curves that never
cross.

The node turns each of those outcomes into a bound on T_m and places the next
window inside the resulting bracket.  What is under test here is that search:
which way it moves, how far, that it never contradicts a bound it has already
established, and what it reports when it cannot get there.  The legs are faked,
because running real calphy legs to exercise it would cost hours per case.
"""

from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("calphy")

from calphy.errors import MeltedError, SolidifiedError  # noqa: E402

from aiflow_nodes.atomistic.property import calphy as calphy_nodes  # noqa: E402
from aiflow_nodes.atomistic.property.calphy import (  # noqa: E402
    CalphyDiagnostics,
    CalphyMeltingTemperatureSearch,
    InputClass,
)

T_TRUE = 1000.0
SLOPE_SOLID = -1.0e-3  # -S_solid;  the liquid's is steeper, as it must be
SLOPE_LIQUID = -1.5e-3


def _inp(temperature=900, temperature_stop=1100):
    """The starting window, exactly as the InputClass node hands it over."""
    return InputClass(temperature=temperature, temperature_stop=temperature_stop).run()


def _fake_legs(
    solid_limit=np.inf,
    liquid_limit=0.0,
    t_cross=T_TRUE,
    drift=None,
    parallel=False,
    error=0.0,
    seen=None,
    dumps_trajectory=False,
):
    """Stand in for one calphy reversible-scaling leg.

    ``solid_limit`` is the superheating limit: a window reaching above it melts.
    ``liquid_limit`` is the undercooling limit: a window starting below it
    freezes.  Otherwise both phases return straight G(T) lines crossing at
    ``t_cross`` -- or at ``centre + drift``, which models curves whose implied
    crossing keeps running away from the window that produced them.

    With ``dumps_trajectory`` the ``MeltedError`` carries a ``simfolder`` the way
    a real one does (``_run_routine`` attaches it), so the search can read back
    where the crystal went.  Use it with the ``melts_at`` fixture, which fakes
    the reader; writing real LAMMPS dumps would test ase.io, not the search.
    """

    def leg(inp, potential, structure, phase, working_directory):
        t_min, t_max = float(inp.temperature), float(inp.temperature_stop)
        if seen is not None:
            seen.append((t_min, t_max))
        if phase == "solid" and t_max > solid_limit:
            error_ = MeltedError("System melted, increase size or reduce temp!")
            if dumps_trajectory:
                error_.simfolder = "/fake/simfolder"
            raise error_
        if phase == "liquid" and t_min < liquid_limit:
            raise SolidifiedError("System solidified, increase temperature")

        crossing = 0.5 * (t_min + t_max) + drift if drift is not None else t_cross
        temperature = np.linspace(t_min, t_max, 50)
        if phase == "solid":
            free_energy = -4.0 + SLOPE_SOLID * (temperature - crossing)
        elif parallel:
            free_energy = -3.99 + SLOPE_SOLID * (temperature - crossing)
        else:
            free_energy = -4.0 + SLOPE_LIQUID * (temperature - crossing)
        diagnostics = CalphyDiagnostics.pure_dataclass(
            phase=phase, free_energy_error=[error] * len(temperature)
        )
        return list(free_energy), list(temperature), diagnostics

    return leg


@pytest.fixture
def fake_legs(monkeypatch):
    def install(**kwargs):
        monkeypatch.setattr(
            calphy_nodes, "_free_energy_with_temp", _fake_legs(**kwargs)
        )

    return install


@pytest.fixture
def melts_at(monkeypatch):
    """Make ``_melt_bound`` report a fixed temperature for the dumped sweep."""

    def install(temperature):
        monkeypatch.setattr(
            calphy_nodes,
            "_melt_bound",
            lambda exc, t_min, t_max: (
                (temperature, temperature)
                if getattr(exc, "simfolder", None)
                else (t_max, None)
            ),
        )

    return install


#: Output port order of CalphyMeltingTemperatureSearch, so the tests below name what
#: they read instead of indexing a twelve-tuple.
_PORTS = [
    "T_melt",
    "T_melt_error",
    "converged",
    "report",
    "temperature_min",
    "temperature_max",
    "temp_solid",
    "fe_solid",
    "temp_liquid",
    "fe_liquid",
    "solid_diagnostics",
    "liquid_diagnostics",
]


def _run(inp=None, **node_kwargs):
    node = CalphyMeltingTemperatureSearch(
        inp=inp if inp is not None else _inp(),
        structure=None,
        potential="fake",
        **node_kwargs,
    )
    assert list(node.outputs.keys()) == _PORTS  # keeps the names below honest
    return SimpleNamespace(**dict(zip(_PORTS, node.run())))


def test_a_window_that_already_brackets_the_crossing_converges_at_once(fake_legs):
    fake_legs()
    out = _run()
    assert out.converged is True
    assert out.T_melt == pytest.approx(T_TRUE, abs=1.0)
    assert list(out.report["outcome"]) == ["bracketed"]


def test_a_melted_solid_moves_the_window_down(fake_legs):
    """The window reaches above the superheating limit; only coming down helps."""
    fake_legs(solid_limit=1150.0)
    out = _run(inp=_inp(1200, 1400))
    assert list(out.report["outcome"])[0] == "solid melted"
    assert out.report["temperature_max"][1] < out.report["temperature_max"][0]
    assert out.converged is True
    assert out.T_melt == pytest.approx(T_TRUE, abs=1.0)


def test_a_frozen_liquid_moves_the_window_up(fake_legs):
    """This is the reported Cu failure: window bottom far below the melting point."""
    fake_legs(liquid_limit=900.0)
    out = _run(inp=_inp(700, 1100))
    assert list(out.report["outcome"])[0] == "liquid froze"
    assert out.report["temperature_min"][1] > out.report["temperature_min"][0]
    assert out.converged is True
    assert out.T_melt == pytest.approx(T_TRUE, abs=1.0)


# ── the silent miss: both legs run, no crossing exists ─────────────────────


def test_a_window_below_the_crossing_is_recognised_and_jumped(fake_legs):
    """Nothing fails; the solid is simply the stable phase throughout."""
    fake_legs()
    out = _run(inp=_inp(600, 800))
    assert list(out.report["outcome"])[0] == "solid stable throughout"
    assert out.converged is True  # re-centred on the extrapolate, then bracketed
    assert out.T_melt == pytest.approx(T_TRUE, abs=1.0)


def test_a_window_above_the_crossing_converges_in_two_attempts(fake_legs):
    """The Angelo case: the estimated window sits entirely above T_m.

    Both legs run — the liquid is stable at the bottom, the solid superheats
    past the top — so there is no exception to read a direction from, only two
    curves that never meet.  Extrapolating from them lands on the melting point
    in one move.  A fixed-step search would need eight, and ``max_attempts``
    defaults to six, which is what made this case fail.
    """
    seen = []
    fake_legs(seen=seen)
    out = _run(inp=_inp(1700, 1900))
    assert list(out.report["outcome"]) == ["liquid stable throughout", "bracketed"]
    assert out.converged is True
    assert out.T_melt == pytest.approx(T_TRUE, abs=1.0)
    assert seen[0] == (1700.0, 1900.0)
    assert seen[2] == (900.0, 1100.0)  # straight to the crossing, not 100 K down


# ── reading the next window off the failed run ─────────────────────────────


def test_a_dumped_trajectory_places_the_window_under_the_measured_melt(
    fake_legs, melts_at
):
    """Where the crystal actually went beats the fact that it went."""
    seen = []
    fake_legs(solid_limit=1150.0, seen=seen, dumps_trajectory=True)
    melts_at(1100.0)
    out = _run(inp=_inp(1200, 1400))
    assert list(out.report["outcome"])[0] == "solid melted @ 1100 K"
    # Window width 200 K, default back-off a tenth of it: the top lands 20 K
    # under the measured melt rather than on it.
    assert seen[1] == (880.0, 1080.0)
    assert out.converged is True


def test_the_back_off_from_a_bound_can_be_set(fake_legs, melts_at):
    seen = []
    fake_legs(solid_limit=1150.0, seen=seen, dumps_trajectory=True)
    melts_at(1100.0)
    _run(inp=_inp(1200, 1400), bound_backoff=50.0)
    assert seen[1] == (850.0, 1050.0)


def test_without_a_trajectory_the_search_still_moves_by_half_a_window(fake_legs):
    """An unreadable failure must not reduce the search to creeping.

    All it proves is "somewhere at or below t_max", which on its own would move
    the window by the back-off alone.  The half-window step is the fallback.
    """
    seen = []
    fake_legs(solid_limit=1050.0, seen=seen)  # no simfolder on the exception
    _run(inp=_inp(1100, 1300))
    assert seen[0] == (1100.0, 1300.0)
    assert seen[1] == (1000.0, 1200.0)


# ── the bracket ────────────────────────────────────────────────────────────


def test_a_bound_overrules_the_extrapolate(fake_legs):
    """What a phase demonstrated beats what a straight line predicts.

    Here the cell cannot hold its solid anywhere near the crossing — it melts
    at 1050 K while G_solid and G_liquid meet at 2000 K — so an attempt that
    survives reports a crossing 900 K above a temperature the solid has already
    been seen to fail at.  Following that extrapolate would spend every
    remaining attempt in windows the search has proof cannot work.
    """
    seen = []
    fake_legs(solid_limit=1050.0, t_cross=2000.0, seen=seen)
    out = _run(inp=_inp(900, 1100), max_attempts=8)
    assert 2000.0 in list(out.report["T_melt"])  # the extrapolate was made...
    for t_min, t_max in seen:  # ...and not followed
        assert 0.5 * (t_min + t_max) <= 1100.0
    assert list(out.report["outcome"])[-1] == "window search cycling"


def test_a_search_that_cycles_stops_instead_of_burning_its_attempts(fake_legs):
    """Superheating limit so close to T_m that no window both holds and brackets.

    The bracket then closes to less than one window width and the search starts
    proposing the same window twice.  Each attempt is two full calphy legs, so
    noticing the repeat matters more than the tidiness of the outcome.
    """
    seen = []
    fake_legs(solid_limit=1050.0, seen=seen)
    out = _run(inp=_inp(1100, 1300), max_attempts=20)
    assert out.converged is False
    assert list(out.report["outcome"])[-1] == "window search cycling"
    assert len(out.report) < 20
    # A window is seen at most twice -- once per phase.  A third sighting would
    # mean the repeat was only noticed after paying for it again.
    assert max(Counter(seen).values()) <= 2
    assert out.T_melt == pytest.approx(T_TRUE, abs=1.0)  # the edge crossing


def test_every_window_tried_respects_min_lambda(fake_legs):
    """A run of downward moves must not walk into unusable lambda."""
    seen = []
    fake_legs(solid_limit=10.0, seen=seen, liquid_limit=1e9)
    with pytest.raises(ValueError):
        _run(min_lambda=0.6, max_attempts=8)
    assert seen  # the search really did run
    for t_min, t_max in seen:
        assert t_min / t_max >= 0.6 - 1e-9


# ── what comes back ────────────────────────────────────────────────────────


def test_an_unconverged_run_hands_back_a_window_that_does_bracket(fake_legs):
    """The point of the exercise: a failed search still says what to run next."""
    fake_legs(drift=300.0)
    out = _run(max_attempts=2)
    assert out.converged is False
    assert out.temperature_min < out.T_melt < out.temperature_max
    assert out.temperature_max - out.temperature_min == pytest.approx(200.0)


def test_a_converged_run_recentres_the_window_on_the_answer(fake_legs):
    """A crossing near a window edge is worth re-centring before refining."""
    fake_legs()
    out = _run(inp=_inp(950, 1150))
    assert out.converged is True
    assert 0.5 * (out.temperature_min + out.temperature_max) == pytest.approx(
        out.T_melt, abs=1.0
    )


def test_no_window_survives_both_phases_raises_with_the_history(fake_legs):
    """A solid that melts and a liquid that freezes in every window."""
    fake_legs(solid_limit=900.0, liquid_limit=950.0)
    with pytest.raises(ValueError, match="no attempt completed both") as excinfo:
        _run()
    message = str(excinfo.value)
    assert "solid melted" in message and "liquid froze" in message
    assert "window_fraction" in message  # says what to change
    assert "known to lie between" in message  # and how far it did get


def test_running_out_of_attempts_returns_the_extrapolate_unconverged(fake_legs):
    """Curves whose implied crossing keeps running away from the window."""
    fake_legs(drift=300.0)
    out = _run(max_attempts=3)
    assert out.converged is False
    assert list(out.report["outcome"])[-1] == "gave up after 3 attempts"
    assert np.isfinite(out.T_melt)  # still a number, just not a trustworthy one


def test_parallel_curves_stop_the_search_rather_than_moving_forever(fake_legs):
    fake_legs(parallel=True)
    out = _run(inp=_inp(600, 800))
    assert out.converged is False
    assert list(out.report["outcome"])[-1] == "curves parallel, no crossing"
    assert np.isfinite(out.T_melt)


def test_the_error_bar_comes_from_calphys_free_energy_errors(fake_legs):
    fake_legs(error=1e-4)
    out = _run()
    # hypot(1e-4, 1e-4) / |slope difference| = 1.414e-4 / 5e-4
    assert out.T_melt_error == pytest.approx(np.hypot(1e-4, 1e-4) / 5e-4, rel=1e-6)
    assert out.T_melt == pytest.approx(T_TRUE, abs=1.0)


def test_no_errors_reported_gives_a_zero_error_bar_not_a_crash(fake_legs):
    """`n_iterations = 1` leaves calphy's error column identically zero."""
    fake_legs(error=0.0)
    assert _run().T_melt_error == 0.0


def test_an_inverted_starting_window_is_rejected(fake_legs):
    fake_legs()
    with pytest.raises(ValueError, match="must be above"):
        _run(inp=_inp(1100, 900))


def test_the_final_sweeps_and_diagnostics_come_from_the_last_attempt(fake_legs):
    fake_legs(solid_limit=1050.0)
    out = _run(inp=_inp(1100, 1300))
    # Not the 1100-1300 window that melted, but the one that ran.
    assert max(out.temp_solid) <= 1050.0
    assert min(out.temp_liquid) == pytest.approx(min(out.temp_solid))
    assert out.solid_diagnostics.phase == "solid"
    assert out.liquid_diagnostics.phase == "liquid"
