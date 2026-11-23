"""What ``iterate`` and ``IterToDataFrame`` return, at their node interface.

``input_label`` may name several ports, comma-separated; every swept value is
then a tuple with one entry per label.  Outputs of every kind are spread into
columns, and a failing cell is recorded in an ``error`` column.  These tests
characterize today's behaviour, including the parts marked as looking wrong.
"""

from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pytest
from aiflow.core import as_function_node, as_out_dataclass_node

from aiflow_nodes.controls import IterToDataFrame, iterate


@as_function_node
def Affine(x: float = 0.0, a: float = 1.0, b: float = 0.0):
    y = a * x + b
    return y


def _sweep(input_label, values, executor=None):
    sweep = IterToDataFrame(
        node=Affine(), input_label=input_label, values=values, executor=executor
    )
    sweep.run()
    return sweep.outputs.df.value


CELLS = [(1.0, 2.0, 0.0), (2.0, 2.0, 1.0), (3.0, -1.0, 0.5)]


@pytest.mark.parametrize("parallel", [False, True])
def test_one_column_per_label_in_order(parallel):
    pool = ThreadPoolExecutor(max_workers=2) if parallel else None
    try:
        df = _sweep("x, a,b", CELLS, executor=pool)
    finally:
        if pool is not None:
            pool.shutdown()

    assert list(df.columns) == ["x", "a", "b", "y"]
    assert df["x"].tolist() == [1.0, 2.0, 3.0]
    assert df["y"].tolist() == [2.0, 5.0, -2.5]
    assert "error" not in df.columns


def test_single_label_is_unchanged():
    df = _sweep("x", [1.0, 2.0])
    assert list(df.columns) == ["x", "y"]
    assert df["y"].tolist() == [1.0, 2.0]


@as_function_node
def Echo(v=None):
    out = v
    return out


@as_function_node
def Shift(y: float = 0.0, d: float = 0.0):
    y = y + d
    return y


def test_a_single_label_still_takes_a_tuple_whole():
    # Only a comma in input_label unpacks; a tuple swept on one port arrives
    # as the tuple, as it always did.
    sweep = IterToDataFrame(node=Echo(), input_label="v", values=[(1, 2)])
    sweep.run()
    df = sweep.outputs.df.value
    assert df["v"].tolist() == [(1, 2)]
    assert df["out"].tolist() == [(1, 2)]


def test_a_label_that_clashes_with_an_output_gets_the_input_prefix():
    sweep = IterToDataFrame(node=Shift(), input_label="y,d", values=[(1.0, 0.5)])
    sweep.run()
    df = sweep.outputs.df.value
    assert list(df.columns) == ["input_y", "d", "y"]
    assert df.iloc[0].tolist() == [1.0, 0.5, 1.5]


@pytest.mark.parametrize("bad", [[(1.0, 2.0)], [1.0], ["xy"]])
def test_wrong_tuple_length_fails_before_the_first_iteration(bad):
    calls = []

    class Spy:
        def submit(self, *args, **kwargs):  # pragma: no cover - must not be reached
            calls.append(args)
            raise AssertionError("submitted despite a malformed sweep")

    with pytest.raises(ValueError, match="does not match input_label"):
        iterate(node=Affine(), input_label="x,a,b", values=bad, executor=Spy()).run()
    assert calls == []


# ---------------------------------------------------------------------------
# Dict cells, and the swept ports belonging to the sweep
# ---------------------------------------------------------------------------


def test_dict_cells_fill_columns_in_label_order():
    cells = [{"b": 0.0, "x": 1.0, "a": 2.0}, {"a": -1.0, "b": 0.5, "x": 3.0}]
    df = _sweep("x,a,b", cells)
    assert list(df.columns) == ["x", "a", "b", "y"]
    assert df["y"].tolist() == [2.0, -2.5]


@pytest.mark.parametrize(
    "cell", [{"x": 1.0, "a": 2.0}, {"x": 1.0, "a": 2.0, "b": 0.0, "c": 1}]
)
def test_dict_keys_must_match_input_label(cell):
    with pytest.raises(ValueError, match="missing|extra"):
        iterate(node=Affine(), input_label="x,a,b", values=[cell]).run()


@pytest.mark.parametrize("parallel", [False, True])
def test_swept_ports_are_restored_after_the_sweep(parallel):
    template = Affine(a=5.0)  # `a` is swept below; the user's value comes back
    pool = ThreadPoolExecutor(max_workers=2) if parallel else None
    try:
        iterate(
            node=template,
            input_label="x,a",
            values=[(1.0, 2.0), (3.0, 4.0)],
            executor=pool,
        ).run()
    finally:
        if pool is not None:
            pool.shutdown()
    assert template.inputs.a.value == 5.0
    assert template.inputs.x.value == 0.0  # default, untouched by the last cell


def test_the_sweep_hash_does_not_depend_on_run_history():
    pdb = pytest.importorskip("aiflow.database")
    template = Affine(b=1.0)
    sweep = IterToDataFrame(
        node=template, input_label="x,a", values=[(1.0, 2.0), (3.0, 4.0)]
    )
    before = pdb.get_hash(template)
    sweep.run()
    assert pdb.get_hash(template) == before


def test_a_set_swept_port_is_reported_as_ignored(caplog):
    import logging

    with caplog.at_level(logging.WARNING):
        iterate(node=Affine(a=5.0), input_label="x,a", values=[(1.0, 2.0)]).run()
    ignored = [r for r in caplog.records if "is ignored" in r.getMessage()]
    assert len(ignored) == 1
    assert "'Affine.a' = 5.0" in ignored[0].getMessage()


def test_unset_swept_ports_are_not_reported(caplog):
    import logging

    with caplog.at_level(logging.WARNING):
        iterate(node=Affine(b=1.0), input_label="x,a", values=[(1.0, 2.0)]).run()
    assert not [r for r in caplog.records if "is ignored" in r.getMessage()]


# ---------------------------------------------------------------------------
# Outputs and failing cells
# ---------------------------------------------------------------------------


@as_function_node
def Square(x: int = 0):
    if x == 2:
        raise ValueError("two")
    y = x * x
    return y


@as_function_node
def Pair(x: int = 0):
    if x == 2:
        raise ValueError("two")
    s = x + 1
    p = x * 10
    return s, p


@as_out_dataclass_node
class Stats:
    lo: int = 0
    hi: int = 0


@as_function_node
def MakeStats(x: int = 0) -> Stats:
    if x == 2:
        raise ValueError("two")
    stats = Stats.pure_dataclass(lo=x, hi=2 * x)
    return stats


@as_function_node
def Frame(x: int = 0, rows: int = 1):
    if x == 2:
        raise ValueError("two")
    frame = pd.DataFrame({"a": [x] * rows, "b": [10 * x + i for i in range(rows)]})
    return frame


NAN = float("nan")
ERROR = ["", "ValueError: two", ""]

# The DataFrame IterToDataFrame returns over values [1, 2, 3], where the cell
# x=2 fails; over [1, 3] it is the same without that row and the error column.
OUTPUT_KINDS = {
    "scalar": (Square, {"x": [1, 2, 3], "y": [1, NAN, 9], "error": ERROR}),
    "tuple": (
        Pair,
        {"x": [1, 2, 3], "s": [2, NAN, 4], "p": [10, NAN, 30], "error": ERROR},
    ),
    "dataclass": (
        MakeStats,
        {"x": [1, 2, 3], "lo": [1, NAN, 3], "hi": [2, NAN, 6], "error": ERROR},
    ),
    # Looks wrong, pinned anyway: with DataFrame outputs the error column
    # comes before the output columns, not after them as for every other kind.
    "one-row frame": (
        Frame,
        {"x": [1, 2, 3], "error": ERROR, "a": [1, NAN, 3], "b": [10, NAN, 30]},
    ),
    "multi-row frame": (
        lambda: Frame(rows=2),
        {
            "x": [1, 1, 2, 3, 3],
            "error": ["", "", "ValueError: two", "", ""],
            "a": [1, 1, NAN, 3, 3],
            "b": [10, 11, NAN, 30, 31],
        },
    ),
}


@pytest.fixture(params=["sequential", "threads"])
def executor(request):
    if request.param == "sequential":
        yield None
        return
    pool = ThreadPoolExecutor(max_workers=3)
    yield pool
    pool.shutdown()


@pytest.mark.parametrize("kind", OUTPUT_KINDS)
@pytest.mark.parametrize("failing", [False, True], ids=["ok", "failing-cell"])
def test_outputs_are_expanded_into_columns_and_failures_recorded(
    kind, failing, executor
):
    template, expected = OUTPUT_KINDS[kind]
    expected = pd.DataFrame(expected)
    if not failing:
        expected = expected[expected["x"] != 2].drop(columns="error")
        expected = expected.reset_index(drop=True)
    values = [1, 2, 3] if failing else [1, 3]
    sweep = IterToDataFrame(
        node=template(), input_label="x", values=values, executor=executor
    )
    sweep.run()
    pd.testing.assert_frame_equal(sweep.outputs.df.value, expected, check_dtype=False)


@pytest.mark.parametrize("template", [Square, Pair, MakeStats])
def test_iterate_returns_what_the_template_returns_for_each_cell(template, executor):
    expected = [template(x=x).run() for x in (1, 3, 4)]
    out = iterate(
        node=template(), input_label="x", values=[1, 3, 4], executor=executor
    ).run()
    assert out == expected


def test_iterate_drops_a_failing_cell_when_sequential():
    # Looks wrong, pinned anyway: the failing cell vanishes from the result,
    # so the results no longer line up with the values; with an executor the
    # same cell raises (next test).
    assert iterate(node=Square(), input_label="x", values=[1, 2, 3]).run() == [1, 9]


def test_iterate_raises_a_failing_cell_with_an_executor():
    with (
        ThreadPoolExecutor(max_workers=2) as pool,
        pytest.raises(ValueError, match="two"),
    ):
        iterate(node=Square(), input_label="x", values=[1, 2, 3], executor=pool).run()


def test_iterate_over_one_value_returns_the_bare_result(executor):
    # Looks wrong, pinned anyway: one value gives its result, not a list of one.
    out = iterate(node=Pair(), input_label="x", values=[3], executor=executor).run()
    assert out == (4, 30)


@as_function_node
def Delayed(x: int = 0):
    import time

    time.sleep(0.05 * (3 - x))  # later cells finish first
    y = x * x
    return y


def test_results_are_in_input_order_whatever_order_cells_finish_in():
    with ThreadPoolExecutor(max_workers=3) as pool:
        out = iterate(
            node=Delayed(), input_label="x", values=[1, 2, 3], executor=pool
        ).run()
        df = IterToDataFrame(
            node=Delayed(), input_label="x", values=[1, 2, 3], executor=pool
        ).run()
    assert out == [1, 4, 9]
    assert df["x"].tolist() == [1, 2, 3]
    assert df["y"].tolist() == [1, 4, 9]
