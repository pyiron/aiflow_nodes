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
