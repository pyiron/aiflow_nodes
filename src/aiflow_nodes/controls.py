"""Sweeps that collect their cells, and small helpers for lists and arrays.

``IterToDataFrame`` and ``iterate`` run a template node once per value on
aiflow's sweep (:func:`aiflow.core.run_sweep`), which does the parallel
execution, the per-cell storage and the cluster recovery; this module only
arranges the results.  Control flow, executors and list nodes come from
:mod:`aiflow.core.stdlib`.
"""

from dataclasses import fields, is_dataclass

import numpy as np
import pandas as pd
from aiflow.core import Node, as_function_node, run_sweep, sweep_records


def _expand_dataclass_columns(columns: dict) -> dict:
    """Replace every column of dataclass instances by one column per field."""
    expanded = {}
    for key, column in columns.items():
        first = next(
            (v for v in column if is_dataclass(v) and not isinstance(v, type)), None
        )
        if first is None:
            expanded[key] = column
            continue
        for field in fields(first):
            expanded[field.name] = [
                getattr(v, field.name) if is_dataclass(v) else np.nan for v in column
            ]
    return expanded


def _expand_df_columns(data_dict: dict):
    """
    If any output column in *data_dict* contains ``pd.DataFrame`` values,
    replace that column with the individual columns of the inner DataFrame.

    Single-row inner DataFrames
        Each inner column is spread directly into the outer dict.  Returns a
        plain ``dict`` so the caller can still pass it to ``pd.DataFrame()``.

    Multi-row inner DataFrames
        Inner rows are concatenated; scalar column values are repeated for
        every inner row.  Returns a fully merged ``pd.DataFrame`` so the
        caller can return it immediately.

    The input-label column (the first column, a scalar) is left in place and
    is always the leftmost column in the result.
    """
    # Identify which columns hold DataFrame values (skip plain scalars/NaN).
    df_cols: dict[str, tuple] = {}
    for col, vals in data_dict.items():
        first_df = next((v for v in vals if isinstance(v, pd.DataFrame)), None)
        if first_df is not None:
            df_cols[col] = (vals, list(first_df.columns))

    if not df_cols:
        return data_dict  # nothing to expand — fast path

    scalar_cols = {col: vals for col, vals in data_dict.items() if col not in df_cols}
    n = len(next(iter(data_dict.values())))

    has_multi = any(
        isinstance(v, pd.DataFrame) and len(v) > 1
        for vals, _ in df_cols.values()
        for v in vals
    )

    if not has_multi:
        # ── single-row case: spread inner columns into the dict ──────────
        new_dict: dict[str, list] = dict(scalar_cols)
        for col, (vals, inner_cols) in df_cols.items():
            for icol in inner_cols:
                new_dict[icol] = []
            for v in vals:
                if isinstance(v, pd.DataFrame) and len(v) >= 1:
                    for icol in inner_cols:
                        new_dict[icol].append(v[icol].iloc[0])
                else:
                    for icol in inner_cols:
                        new_dict[icol].append(np.nan)
        return new_dict

    else:
        # ── multi-row case: concat inner frames, repeat scalar values ────
        frames = []
        for i in range(n):
            inner_parts = []
            for col, (vals, inner_cols) in df_cols.items():
                v = vals[i]
                if isinstance(v, pd.DataFrame):
                    inner_parts.append(v.reset_index(drop=True))
                else:
                    inner_parts.append(pd.DataFrame({c: [np.nan] for c in inner_cols}))

            n_inner = max((len(df) for df in inner_parts), default=1)

            parts = []
            if scalar_cols:
                parts.append(
                    pd.DataFrame(
                        {col: [vals[i]] * n_inner for col, vals in scalar_cols.items()}
                    )
                )
            parts.extend(p.reset_index(drop=True) for p in inner_parts)
            frames.append(pd.concat(parts, axis=1))

        return pd.concat(frames, ignore_index=True)


@as_function_node
def IterToDataFrame(
    node: Node,
    input_label: str,
    values: list | np.ndarray,
    debug: bool = False,
    executor: type = None,
    store: bool = False,
) -> pd.DataFrame:
    """
    Iterate over ``values`` feeding each element into ``node`` under the name
    ``input_label`` and collect the results in a pandas DataFrame.

    New feature:
        – If the node returns a *dataclass* instance, each field of the
          dataclass becomes its own column in the DataFrame.
        – Several inputs can be swept together: give ``input_label`` as a
          comma-separated list (``"task_index,model,rep"``) and make every
          element of ``values`` a tuple with one entry per label, or a dict
          keyed by label (the keys must match).  Each label gets its own
          column.  For a full grid, wire a node that returns the cartesian
          product (e.g. ``aiflow_evals.nodes.BenchmarkGrid``), or type the tuples into
          ``InputVector`` (``"[(0, 'a'), (1, 'b')]"``).
        – The swept ports belong to the sweep: they are set for every cell and
          restored afterwards, the canvas hides them on the template, and a
          value set on one in code is ignored with a warning.
        – ``store=True`` on the *template* stores every cell as its own row
          (linked to this sweep, restored on re-run).  On a cluster executor
          the cells are check-pointed regardless and can be collected after a
          crash; with ``non_modal`` the sweep submits them and returns.

    Parameters
    ----------
    node : Node
        The node that will be executed for each value.
    input_label : str
        Name of the input attribute on ``node`` that receives each element of
        ``values`` — or several names, comma-separated, to sweep them jointly.
    values : list | np.ndarray
        Iterable of input values; tuples of matching length when
        ``input_label`` names several ports.
    debug : bool, optional
        Unused, kept for existing workflows; ``iterate`` prints per-cell
        debugging output.
    executor : concurrent.futures.Executor, optional
        If supplied, the iteration runs in parallel using the executor.
    store : bool, optional
        Used by decorator to implement hash storage (if set to True)

    Returns
    -------
    pd.DataFrame
        DataFrame where each column corresponds to an input or an output
        field.  When the node returns a dataclass, each field of the
        dataclass is a separate column, and a DataFrame output is spread
        into its columns (and rows).  A failing cell leaves its outputs NaN
        and its message in an ``error`` column, which comes last and exists
        only when a cell failed.
    """
    records = sweep_records(node, input_label, values, executor=executor)
    output_labels = [label for label in node.outputs.keys() if label != "self"]  # noqa: SIM118  iterating yields ports
    errors = [record.pop("error") for record in records]
    for record, error in zip(records, errors):
        if error is not None:
            record.update(dict.fromkeys(output_labels, np.nan))
    columns = {
        key: [record[key] for record in records]
        for key in (records[0] if records else {})
    }
    columns = _expand_dataclass_columns(columns)
    if any(error is not None for error in errors):
        columns["error"] = ["" if error is None else error for error in errors]

    expanded = _expand_df_columns(columns)
    df = expanded if isinstance(expanded, pd.DataFrame) else pd.DataFrame(expanded)
    # The error column comes last, after every output column.
    if "error" in df.columns:
        df = df[[c for c in df.columns if c != "error"] + ["error"]]
    return df


@as_function_node
def iterate(
    node: Node,
    input_label: str,
    values: list | np.ndarray,
    debug: bool = False,
    executor: type = None,
):
    """Run ``node`` once per value and return its outputs, in the order of ``values``.

    ``input_label`` and ``values`` work as in :func:`IterToDataFrame`.  The
    result is a list with one entry per value, also for a single value; a
    failing cell raises its error.
    """
    out_lst = run_sweep(node, input_label, values, debug=debug, executor=executor)[0]
    return out_lst


@as_function_node
# pick a single element from a list
def pick_element(lst: list | np.ndarray, index: int) -> any:
    element = lst[index]
    return element


@as_function_node
def ExtractList(out_list: list, data_label: str, flatten: bool = True):
    import numpy as np

    collect = np.array([out.__getattribute__(data_label) for out in out_list])
    if flatten:
        collect = collect.flatten()
    return collect


@as_function_node
def InputVector(vec: str = ""):
    try:
        vector = eval(vec)
    except Exception:
        vector = None
    return vector


@as_function_node
def Slice(matrix, slice: str = "::"):
    try:
        result = eval(f"matrix[{slice}]")
    except Exception as e:
        result = None
        print("Slice failed: ", e)
    return result


@as_function_node
def Code(x, code: str = "x**2"):
    y = eval(code)
    return y


@as_function_node
def SetAttribute(obj, attr: str, val: str) -> any:
    """Set an attribute on an object."""
    try:
        obj.__setattr__(attr, val)
    except AttributeError:
        print(f"Attribute {attr} not found in object {obj}")
    return obj


@as_function_node
def GetMask(x: np.ndarray, index: int = 0):
    mask = np.array(x) == index
    return mask


@as_function_node
def Filter(x: np.ndarray, index_vec: np.ndarray):
    result = x[:, index_vec]
    return result


@as_function_node
def Sleep(seconds: float = 1.0, a: float = 0) -> None:
    """Sleep for a specified number of seconds.

    Parameters
    ----------
    seconds : float, optional
        Number of seconds to sleep (default: 1.0)

    Returns
    -------
    None
    """
    import time

    print("sleep: ", a)
    time.sleep(seconds)
    status = None
    return status
