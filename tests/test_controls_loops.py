"""Tests for aiflow_nodes.controls_loops.WhileLoop (staging, 2026-11-17)."""

import pandas as pd
import pytest
from aiflow.core import Workflow, as_function_node, group_node

from aiflow_nodes.controls import Code
from aiflow_nodes.controls_loops import (
    ProceedUntilConverged,
    ProceedWhileAbove,
    ProceedWhileBelow,
    WhileLoop,
)


# ---- body nodes ------------------------------------------------------------
@as_function_node("halved")
def Halve(number: float = 40.0):
    halved = number / 2
    return halved


@as_function_node(["x", "K_used"])
def Increment(x: float = 0.0, K: float = 0.0, K_prev: float = 0.0):
    """x grows by the K increment; reports the K it used (to carry into K_prev)."""
    x = x + (K - K_prev)
    K_used = K
    return x, K_used


@as_function_node(["x", "energy"])
def Relax(x: float = 1.0, rate: float = 0.5):
    """x -> x * rate; 'energy' = x**2 converges to 0."""
    x = x * rate
    energy = x**2
    return x, energy


@as_function_node("off")
def Constant(v: float = 10.0):
    off = v
    return off


@as_function_node("x")
def AddOffset(x: float = 0.0, off: float = 0.0):
    x = x + off
    return x


@as_function_node("x")
def AddOne(x: int = 0):
    x = x + 1
    return x


@group_node("x")
def AddTwoGroup(x=0):
    wf = Workflow("add_two")
    wf.a = AddOne(x=x)
    wf.b = AddOne(x=wf.a)
    return wf.b


# ---- condition nodes -------------------------------------------------------
@as_function_node("proceed")
def AtLeast(halved: float = 0.0, floor: float = 1.0):
    proceed = halved >= floor
    return proceed


@as_function_node("proceed")
def Below(x: float = 0.0, limit: float = 1.0):
    proceed = x < limit
    return proceed


@as_function_node("proceed")
def ValueAtLeast(value: float = 0.0, floor: float = 1.0):
    proceed = value >= floor
    return proceed


@as_function_node("proceed")
def KBelow(K: float = 0.0, limit: float = 1.0):
    proceed = limit > K
    return proceed


@as_function_node("proceed")
def NotConverged(energy=None, tol: float = 1e-3):
    """History scope: energy is the list so far."""
    proceed = not (len(energy) >= 2 and abs(energy[-1] - energy[-2]) < tol)
    return proceed


@as_function_node("proceed")
def FewerThan(iteration: int = 0, n: int = 3):
    proceed = iteration < n
    return proceed


@as_function_node("proceed")
def HistoryRows(history=None, n: int = 2):
    proceed = len(history) < n
    return proceed


@as_function_node("ratio")
def NotABool(halved: float = 0.0):
    ratio = halved / 2
    return ratio


def _run(**kw):
    return WhileLoop(**kw).run()


# ---- plain while (condition only) ------------------------------------------
def test_plain_while_with_carry():
    outputs, history, n, reason = _run(
        body=Halve(number=40.0), condition=AtLeast(floor=1.0), carry="halved -> number"
    )
    assert outputs == {"halved": 0.625}
    assert n == 6 and reason == "condition"
    assert list(history["halved"]) == [20.0, 10.0, 5.0, 2.5, 1.25, 0.625]
    assert list(history["iteration"]) == [1, 2, 3, 4, 5, 6]


def test_starting_value_from_body_input():
    _, _, n, _ = _run(
        body=Halve(number=4.0), condition=AtLeast(floor=1.0), carry="halved -> number"
    )
    assert n == 3  # 2, 1, 0.5


# ---- schedule (feed only) and both -----------------------------------------
def test_schedule_runs_to_the_end_and_carries_two_values():
    outputs, history, n, reason = _run(
        body=Increment(),
        feed=[1.0, 1.5, 3.0],
        feed_input="K",
        carry="x -> x, K_used -> K_prev",
    )
    assert reason == "schedule_end" and n == 3
    assert outputs == {"x": 3.0, "K_used": 3.0}  # increments 1.0 + 0.5 + 1.5
    assert list(history["K"]) == [1.0, 1.5, 3.0]


def test_feed_as_string():
    _, history, n, reason = _run(
        body=Increment(),
        feed="1.0, 2.0",
        feed_input="K",
        carry="x -> x, K_used -> K_prev",
    )
    assert reason == "schedule_end" and n == 2 and list(history["K"]) == [1.0, 2.0]


def test_schedule_with_early_stop():
    outputs, _, n, reason = _run(
        body=Increment(),
        condition=Below(limit=2.0),
        feed=[1.0, 1.5, 2.5, 3.0],
        feed_input="K",
        carry="x -> x, K_used -> K_prev",
    )
    assert reason == "condition" and n == 3 and outputs["x"] == 2.5


# ---- condition scope -------------------------------------------------------
def test_condition_map_renames():
    _, _, n, reason = _run(
        body=Halve(number=40.0),
        condition=ValueAtLeast(floor=1.0),
        carry="halved -> number",
        condition_map="halved -> value",
    )
    assert n == 6 and reason == "condition"


def test_inputs_plus_outputs_lets_the_condition_see_the_fed_value():
    _, _, n, reason = _run(
        body=Increment(),
        condition=KBelow(limit=1.5),
        condition_sees="inputs+outputs",
        feed=[1.0, 1.5, 2.0],
        feed_input="K",
        carry="x -> x, K_used -> K_prev",
    )
    assert reason == "condition" and n == 2


def test_outputs_only_hides_inputs():
    # K is a body input, not an output: with the default the condition keeps its own K (0.0 < 1.5)
    _, _, n, reason = _run(
        body=Increment(),
        condition=KBelow(limit=1.5),
        feed=[1.0, 1.5, 2.0],
        feed_input="K",
        carry="x -> x, K_used -> K_prev",
    )
    assert reason == "schedule_end" and n == 3


def test_history_scope_passes_the_series():
    _, history, n, reason = _run(
        body=Relax(x=1.0, rate=0.5),
        condition=NotConverged(tol=1e-3),
        carry="x -> x",
        condition_scope="history",
    )
    e = list(history["energy"])
    assert (
        reason == "condition"
        and abs(e[-1] - e[-2]) < 1e-3
        and abs(e[-2] - e[-3]) >= 1e-3
    )


def test_special_iteration_and_history_ports():
    _, _, n, _ = _run(
        body=Halve(number=1e6), condition=FewerThan(n=3), carry="halved -> number"
    )
    assert n == 3
    _, _, n, _ = _run(
        body=Halve(number=1e6), condition=HistoryRows(n=4), carry="halved -> number"
    )
    assert n == 4


# ---- record ----------------------------------------------------------------
def test_record_none_records_all_outputs_and_a_list_records_a_subset():
    _, history, _, _ = _run(
        body=Relax(), feed=[0, 0], feed_input="rate", carry="x -> x"
    )
    assert list(history.columns) == ["iteration", "rate", "x", "energy"]
    _, history, _, _ = _run(
        body=Relax(), feed=[0, 0], feed_input="rate", carry="x -> x", record="energy"
    )
    assert list(history.columns) == ["iteration", "rate", "energy"]
    assert isinstance(history, pd.DataFrame)


# ---- limits ----------------------------------------------------------------
def test_max_iterations_raise_and_stop():
    with pytest.raises(RuntimeError, match="did not finish"):
        _run(
            body=Halve(number=1e9),
            condition=AtLeast(floor=0.0),
            carry="halved -> number",
            max_iterations=3,
        )
    _, _, n, reason = _run(
        body=Halve(number=1e9),
        condition=AtLeast(floor=0.0),
        carry="halved -> number",
        max_iterations=3,
        on_max="stop",
    )
    assert n == 3 and reason == "max_iterations"


def test_schedule_end_at_the_limit_is_not_an_error():
    _, _, n, reason = _run(
        body=Increment(), feed=[1, 2, 3], feed_input="K", max_iterations=3
    )
    assert n == 3 and reason == "schedule_end"


# ---- errors ----------------------------------------------------------------
@pytest.mark.parametrize(
    "kw, match",
    [
        ({}, "nothing would stop it"),
        ({"feed": [1], "feed_input": "nope"}, "feed_input"),
        (
            {"feed": [1], "feed_input": "number", "carry": "nope -> number"},
            "not a body output",
        ),
        (
            {"feed": [1], "feed_input": "number", "carry": "halved -> nope"},
            "not a body input",
        ),
        (
            {"feed": [1], "feed_input": "number", "carry": "halved number"},
            "source -> target",
        ),
        ({"feed": [1], "feed_input": "number", "record": "nope"}, "record"),
        ({"condition": AtLeast(), "condition_map": "nope -> floor"}, "not visible"),
        (
            {"condition": AtLeast(), "condition_map": "halved -> nope"},
            "not a condition input",
        ),
    ],
)
def test_bad_settings_raise(kw, match):
    with pytest.raises(ValueError, match=match):
        _run(body=Halve(), **kw)


def test_condition_must_return_a_bool():
    with pytest.raises(TypeError, match="must be a bool"):
        _run(body=Halve(), condition=NotABool(), carry="halved -> number")


# ---- general condition nodes -----------------------------------------------
def test_proceed_nodes():
    assert ProceedWhileAbove._original_func(value=2.0, threshold=1.0) is True
    assert ProceedWhileAbove._original_func(value=1.0, threshold=1.0) is False
    assert ProceedWhileBelow._original_func(value=0.5, threshold=1.0) is True
    assert ProceedWhileBelow._original_func(value=1.0, threshold=1.0) is False
    f = ProceedUntilConverged._original_func
    assert f(values=[1.0], tol=0.1) is True
    assert f(values=[1.0, 0.95], tol=0.1) is False
    assert f(values=[1.0, 0.95], tol=0.1, window=2) is True
    assert f(values=[1.0, 0.95, 0.94], tol=0.1, window=2) is False
    with pytest.raises(ValueError, match="1D series"):
        f(values=[[1.0, 2.0]], tol=0.1)


def test_code_body_with_general_conditions():
    # the GUI test workflow's three loops (WORKFLOWS_new/while_loop_gui_test.py)
    _, h, n, reason = _run(
        body=Code(x=40.0, code="x/2"),
        condition=ProceedWhileAbove(threshold=1.0),
        carry="y -> x",
        condition_map="y -> value",
    )
    assert (n, reason, h["y"].iloc[-1]) == (6, "condition", 0.625)
    _, h, n, reason = _run(
        body=Code(x=0, code="x**2"),
        condition=ProceedWhileBelow(threshold=10.0),
        feed="1, 2, 3, 4, 5",
        feed_input="x",
        condition_map="y -> value",
    )
    assert (n, reason, list(h["y"])) == (4, "condition", [1.0, 4.0, 9.0, 16.0])
    _, h, n, reason = _run(
        body=Code(x=1.0, code="x*0.5"),
        condition=ProceedUntilConverged(tol=1e-3),
        carry="y -> x",
        condition_map="y -> values",
        condition_scope="history",
    )
    assert (n, reason) == (10, "condition")


# ---- executor_for_body -----------------------------------------------------
@as_function_node(["x", "pid"])
def AddOneWhere(x: int = 0):
    import os

    x = x + 1
    pid = os.getpid()
    return x, pid


@as_function_node("proceed")
def BelowHere(x: int = 0, limit: int = 3):
    import os

    proceed = x < limit
    # the condition must run in the calling process
    assert os.getpid() == int(os.environ["WHILELOOP_TEST_PID"])
    return proceed


@group_node("x", "pid")
def AddOneWhereGroup(x=0):
    wf = Workflow("add_one_where")
    wf.a = AddOneWhere(x=x)
    return wf.a.outputs.x, wf.a.outputs.pid


def test_executor_for_body_runs_the_body_elsewhere_and_the_condition_here(monkeypatch):
    import os

    from executorlib import (
        SingleNodeExecutor,  # executorlib, as the SLURM path (cloudpickle, own process)
    )

    monkeypatch.setenv("WHILELOOP_TEST_PID", str(os.getpid()))
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with SingleNodeExecutor(max_workers=1) as ex:
        outputs, history, n, reason = _run(
            body=AddOneWhere(x=0),
            condition=BelowHere(limit=3),
            carry="x -> x",
            executor_for_body=ex,
        )
    assert (outputs["x"], n, reason) == (3, 3, "condition")
    assert all(pid != os.getpid() for pid in history["pid"])


def test_executor_for_body_with_a_group_body(monkeypatch):
    import os
    from pathlib import Path

    from executorlib import (
        SingleNodeExecutor,  # executorlib, as the SLURM path (cloudpickle, own process)
    )

    monkeypatch.setenv("WHILELOOP_TEST_PID", str(os.getpid()))
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    # The worker rebuilds the group by importing its node classes by path
    # (tests.test_controls_loops.AddOneWhere), so it needs the repo root, which
    # holds the tests package, on its PYTHONPATH.
    repo_root = str(Path(__file__).resolve().parents[1])
    monkeypatch.setenv(
        "PYTHONPATH", os.pathsep.join([repo_root, os.environ.get("PYTHONPATH", "")])
    )
    with SingleNodeExecutor(max_workers=1) as ex:
        outputs, history, n, reason = _run(
            body=AddOneWhereGroup(x=0),
            condition=BelowHere(limit=2),
            carry="x -> x",
            executor_for_body=ex,
        )
    assert (outputs["x"], n, reason) == (2, 2, "condition")
    assert all(pid != os.getpid() for pid in history["pid"])


def test_executor_for_body_refused_inside_a_slurm_job(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    with (
        ThreadPoolExecutor(max_workers=1) as ex,
        pytest.raises(ValueError, match="inside a SLURM job"),
    ):
        _run(
            body=AddOne(x=0),
            condition=Below(limit=3),
            carry="x -> x",
            executor_for_body=ex,
        )


def test_executor_for_body_refuses_a_non_modal_executor(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    import aiflow.core.remote_cache as rc

    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with ThreadPoolExecutor(max_workers=1) as ex:
        monkeypatch.setattr(rc, "is_non_modal", lambda e: e is ex)
        with pytest.raises(ValueError, match="modal executor"):
            _run(
                body=AddOne(x=0),
                condition=Below(limit=3),
                carry="x -> x",
                executor_for_body=ex,
            )


# ---- group body, wired static inputs ---------------------------------------
def test_group_node_as_body():
    outputs, _, n, reason = _run(
        body=AddTwoGroup(x=0), condition=Below(limit=6), carry="x -> x"
    )
    assert outputs == {"x": 6} and n == 3 and reason == "condition"


def test_wired_static_inputs_in_a_workflow():
    wf = Workflow("whileloop_wired")
    wf.Off = Constant(v=10.0)
    wf.Limit = Constant(v=30.0)
    wf.Body = AddOffset(x=0.0, off=wf.Off)  # `off` is a wired static input of the body
    wf.Cond = Below(limit=wf.Limit)  # `limit` is a wired static input of the condition
    wf.Loop = WhileLoop(body=wf.Body, condition=wf.Cond, carry="x -> x")
    outputs, history, n, reason = wf.Loop.pull()
    assert outputs == {"x": 30.0} and n == 3 and reason == "condition"
    assert list(history["x"]) == [10.0, 20.0, 30.0]
