"""Setting checks of controls_loops.WhileLoop and ProceedUntilConverged that
tests/test_controls_loops.py does not reach."""

import pytest
from aiflow.core import as_function_node

from aiflow_nodes.controls_loops import (
    ProceedUntilConverged,
    WhileLoop,
    _feed_values,
    _pairs,
)


@as_function_node("halved")
def Halve(number: float = 40.0):
    halved = number / 2
    return halved


@as_function_node(["first", "second"])
def TwoVerdicts(halved: float = 0.0):
    first = halved > 1
    second = halved > 2
    return first, second


def _run(**kw):
    return WhileLoop(body=Halve(), **kw).run()


@pytest.mark.parametrize(
    "kw, match",
    [
        ({"feed": [1], "feed_input": "number", "max_iterations": 0}, "max_iterations"),
        ({"feed": [1], "feed_input": "number", "on_max": "warn"}, "on_max"),
        (
            {"feed": [1], "feed_input": "number", "condition_sees": "all"},
            "condition_sees",
        ),
        (
            {"feed": [1], "feed_input": "number", "condition_scope": "all"},
            "condition_scope",
        ),
        ({"feed": [], "feed_input": "number"}, "feed is empty"),
        ({"condition": TwoVerdicts(), "carry": "halved -> number"}, "output 'proceed'"),
    ],
)
def test_bad_settings_raise(kw, match):
    with pytest.raises(ValueError, match=match):
        _run(**kw)


@pytest.mark.parametrize("text", ["halved ->", "-> number"])
def test_pairs_need_both_ends(text):
    with pytest.raises(ValueError, match="source -> target"):
        _pairs(text, "carry")


def test_feed_values():
    assert _feed_values(None) is None
    assert _feed_values("[1.7, 1.8]") == [1.7, 1.8]
    assert _feed_values((1, 2)) == [1, 2]
    with pytest.raises(TypeError, match="sequence or a comma-separated string"):
        _feed_values(5)


def test_converged_needs_a_positive_window():
    with pytest.raises(ValueError, match="window must be >= 1"):
        ProceedUntilConverged._original_func(values=[1.0, 1.0], window=0)


# ---- carry="auto": carry by matching port names -----------------------------
@as_function_node(["x", "energy"])
def NewtonStep(x: float = 1.0):
    """One Newton step for sqrt(2); output x has the name of input x."""
    x = (x + 2 / x) / 2
    energy = x
    return x, energy


@as_function_node(["x", "K_used"])
def Increment(x: float = 0.0, K: float = 0.0, K_prev: float = 0.0):
    x = x + (K - K_prev)
    K_used = K
    return x, K_used


@as_function_node("proceed")
def StillMoving(energy: float = 0.0, iteration: int = 0):
    proceed = iteration < 6 and abs(energy**2 - 2) > 1e-12
    return proceed


def test_auto_carries_outputs_named_like_inputs():
    outputs, _, n, reason = WhileLoop(
        body=NewtonStep(), condition=StillMoving(), carry="auto"
    ).run()
    assert outputs["x"] == pytest.approx(2**0.5)
    assert reason == "condition" and n <= 6


def test_auto_with_an_explicit_rename():
    # x carried by name, K_used -> K_prev by the pair; K is fed
    outputs, history, n, reason = WhileLoop(
        body=Increment(),
        carry="auto, K_used -> K_prev",
        feed="1.0, 1.5, 3.0",
        feed_input="K",
    ).run()
    assert reason == "schedule_end" and n == 3
    assert outputs["x"] == pytest.approx(3.0)  # the increments sum to the last K
    assert list(history["x"]) == pytest.approx([1.0, 1.5, 3.0])


def test_auto_leaves_the_fed_input_to_the_schedule():
    @as_function_node("K")
    def Echo(K: float = 0.0):
        return K

    _, history, n, _ = WhileLoop(
        body=Echo(), carry="auto, K -> K", feed="1, 2, 3", feed_input="K"
    ).run()
    assert n == 3 and list(history["K"]) == [1.0, 2.0, 3.0]
    # "auto" alone: K is the only match and it is fed, so nothing is left
    with pytest.raises(ValueError, match="carry='auto': no body output"):
        WhileLoop(body=Echo(), carry="auto", feed="1, 2", feed_input="K").run()


def test_auto_without_a_match_raises():
    with pytest.raises(ValueError, match="carry='auto': no body output"):
        _run(carry="auto", feed=[1], feed_input="number")
