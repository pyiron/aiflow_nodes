"""Checks for ``SuggestTemperatureWindow``.

The node exists because a calphy temperature window can miss the melting point
without anything raising: above T_m the liquid is stable at ``temperature_min``
and a defect-free periodic solid superheats past ``temperature_max``, so both
legs finish and hand back two free-energy curves that never cross.  Those curves
still measure T_m — they just place it outside the range that was sampled — and
this node reads that off them together with the window that would have contained
it.

What matters is that it distinguishes the three cases: a crossing it can measure,
a crossing it can only predict, and a pair of legs that cannot support either.
The last one has to refuse rather than return a number, because a number here is
spent as several hours of calphy legs in the wrong place.
"""

import numpy as np
import pytest

pytest.importorskip("calphy")

from aiflow_nodes.atomistic.property.calphy import (  # noqa: E402
    SuggestTemperatureWindow,
)

T_TRUE = 1000.0
SLOPE_SOLID = -1.0e-3  # -S_solid;  the liquid's is steeper, as it must be
SLOPE_LIQUID = -1.5e-3


def _legs(t_min, t_max, t_cross=T_TRUE, parallel=False):
    """Two straight G(T) sweeps over the same window, crossing at *t_cross*."""
    temperature = np.linspace(t_min, t_max, 50)
    solid = -4.0 + SLOPE_SOLID * (temperature - t_cross)
    slope_liquid = SLOPE_SOLID if parallel else SLOPE_LIQUID
    offset = 0.01 if parallel else 0.0
    liquid = -4.0 + offset + slope_liquid * (temperature - t_cross)
    return {
        "temp_solid": list(temperature),
        "fe_solid": list(solid),
        "temp_liquid": list(temperature),
        "fe_liquid": list(liquid),
    }


def _suggest(t_min, t_max, t_cross=T_TRUE, parallel=False, **kwargs):
    return SuggestTemperatureWindow(
        **_legs(t_min, t_max, t_cross, parallel), **kwargs
    ).run()


def test_the_ports_are_named_to_drop_into_an_existing_graph():
    """``EstimateCalphyTemperatureRange`` uses these two names for its window."""
    node = SuggestTemperatureWindow(**_legs(900, 1100))
    assert list(node.outputs.keys()) == [
        "T_melt",
        "temperature_min",
        "temperature_max",
        "bracketed",
    ]


# ── a crossing that was measured ───────────────────────────────────────────


def test_a_bracketed_crossing_is_reported_as_measured():
    T_melt, t_min, t_max, bracketed = _suggest(900, 1100)
    assert bracketed is True
    assert T_melt == pytest.approx(T_TRUE, abs=1.0)


def test_a_bracketed_crossing_still_proposes_a_window_centred_on_it():
    """A crossing sitting near a window edge is worth re-centring before
    refining: the phase on that side spent the run barely metastable."""
    T_melt, t_min, t_max, _ = _suggest(950, 1150)
    assert 0.5 * (t_min + t_max) == pytest.approx(T_melt, abs=1.0)
    assert t_max - t_min == pytest.approx(200.0)  # the width that already worked


# ── a crossing that was only predicted ─────────────────────────────────────


def test_a_window_above_the_crossing_is_moved_down_onto_it():
    """The reported Angelo case, in miniature."""
    T_melt, t_min, t_max, bracketed = _suggest(1050, 1250)
    assert bracketed is False
    assert T_melt == pytest.approx(T_TRUE, abs=1.0)
    assert t_min < T_TRUE < t_max


def test_a_window_below_the_crossing_is_moved_up_onto_it():
    T_melt, t_min, t_max, bracketed = _suggest(750, 950)
    assert bracketed is False
    assert T_melt == pytest.approx(T_TRUE, abs=1.0)
    assert t_min < T_TRUE < t_max


def test_the_width_of_the_legs_is_kept_unless_another_is_asked_for():
    """Both phases survived that width; widening pushes them back to the edge."""
    _, t_min, t_max, _ = _suggest(1100, 1400)
    assert t_max - t_min == pytest.approx(300.0)
    _, t_min, t_max, _ = _suggest(1100, 1400, window_width=120.0)
    assert t_max - t_min == pytest.approx(120.0)


def test_a_high_fit_order_does_not_reach_into_the_extrapolation():
    """A polynomial fit diverges past the data rather than continuing.

    The crossing must come off a straight line whatever ``fit_order`` says, so
    an order the user picked to resolve curvature inside the window cannot turn
    an extrapolation into a wild one.
    """
    linear = _suggest(1050, 1250, fit_order=1)
    cubic = _suggest(1050, 1250, fit_order=3)
    assert cubic[0] == pytest.approx(linear[0], abs=1e-6)


# ── refusals ───────────────────────────────────────────────────────────────


def test_parallel_curves_are_refused_rather_than_extrapolated():
    """Equal slopes mean equal entropies, so no crossing exists anywhere."""
    with pytest.raises(ValueError, match="do not cross"):
        _suggest(900, 1100, parallel=True)


def test_a_crossing_extrapolated_below_zero_kelvin_is_refused():
    """Liquid below solid *and* falling faster: the lines meet at a negative
    temperature, which is a sign the two legs were not the phases they claim."""
    with pytest.raises(ValueError, match="above 0 K"):
        _suggest(800, 1000, t_cross=-500.0, max_extrapolation=10.0)


def test_a_crossing_too_far_outside_the_data_is_refused():
    """500 K beyond a 200 K window is a guess about unsampled temperatures."""
    with pytest.raises(ValueError, match="outside"):
        _suggest(1500, 1700)


def test_how_far_is_too_far_can_be_relaxed():
    T_melt, _, _, bracketed = _suggest(1500, 1700, max_extrapolation=5.0)
    assert bracketed is False
    assert T_melt == pytest.approx(T_TRUE, abs=1.0)


# ── the proposal has to be runnable ────────────────────────────────────────


def test_the_proposed_window_respects_min_lambda():
    """Reversible scaling reaches temperature_max by scaling the potential by
    λ = temperature_min/temperature_max; too small a λ is not a usable run."""
    _, t_min, t_max, _ = _suggest(800, 1000, t_cross=100.0, max_extrapolation=10.0)
    assert t_min / t_max == pytest.approx(0.5)


def test_min_lambda_can_be_tightened():
    _, t_min, t_max, _ = _suggest(
        800, 1000, t_cross=100.0, max_extrapolation=10.0, min_lambda=0.8
    )
    assert t_min / t_max == pytest.approx(0.8)


def test_a_window_wider_than_its_centre_still_starts_above_zero():
    """The ``min_lambda`` clamp raises ``temperature_min``; nothing may push it
    through zero, which is what a naive centre-minus-half-width would do."""
    _, t_min, t_max, _ = _suggest(
        800, 1000, t_cross=60.0, window_width=400.0, max_extrapolation=10.0
    )
    assert 0.0 < t_min < t_max
