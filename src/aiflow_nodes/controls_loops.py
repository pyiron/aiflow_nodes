"""Staging module: a general-purpose loop node (new, 2026-11-17).

`WhileLoop` runs a *body* node repeatedly. Between iterations, chosen body
outputs are carried back into body inputs (`carry`). An optional schedule
(`feed`) writes one value per iteration into a body input. An optional
*condition* node judges after each iteration whether another one runs (it
returns `proceed`). The existing `controls.while_loop` is left unchanged.

Inputs of the body are of three kinds:
- carried: set from the previous iteration's outputs (`carry="out -> in"`, or
  `carry="auto"`: every output with the same name as an input);
- fed: set from `feed[i]` (`feed_input`);
- static: everything else, set by the workflow as usual. aiflow resolves wired
  static inputs before the node reaches the loop.

Move to its proper place once tested.
"""

from __future__ import annotations

from typing import Literal

from aiflow.core import Node, as_function_node

# ports that steer execution, never data for the loop
_SKIP_PORTS = {"self", "store", "executor", "debug"}


def _pairs(text, name: str) -> list[tuple[str, str]]:
    """'a -> b, c -> d' -> [('a', 'b'), ('c', 'd')]."""
    pairs = []
    for item in str(text or "").split(","):
        item = item.strip()
        if not item:
            continue
        if "->" not in item:
            raise ValueError(f"{name}: {item!r} is not of the form 'source -> target'")
        a, b = (s.strip() for s in item.split("->", 1))
        if not a or not b:
            raise ValueError(f"{name}: {item!r} is not of the form 'source -> target'")
        pairs.append((a, b))
    return pairs


def _carry_pairs(text, body_out, body_in, fed: str = "") -> list[tuple[str, str]]:
    """`carry` as (body output, body input) pairs.

    The word "auto" (alone, or among explicit pairs: "auto, E -> E_prev")
    carries every body output whose name is also a body input, except the fed
    input `fed`, which the schedule sets. An explicit pair into the same input
    wins over the name match. "auto" matching nothing raises, so a misnamed
    port can't silently leave the loop without a carry.
    """
    items = [s.strip() for s in str(text or "").split(",") if s.strip()]
    auto = any(s.lower() == "auto" for s in items)
    explicit = _pairs(", ".join(s for s in items if s.lower() != "auto"), "carry")
    if not auto:
        return explicit
    taken = {target for _, target in explicit}
    matched = [
        (name, name)
        for name in body_out
        if name in body_in and name != fed and name not in taken
    ]
    if not matched and not explicit:
        raise ValueError(
            f"carry='auto': no body output has the name of a body input "
            f"(outputs: {list(body_out)}; inputs: {list(body_in)}). Rename the "
            f"ports, or give the pairs: 'out -> in'"
        )
    return matched + explicit


def _names(text) -> list[str]:
    return [s.strip() for s in str(text).split(",") if s.strip()]


def _feed_values(feed):
    """The schedule as a list: None stays None, a string '1.7, 1.8' becomes floats."""
    if feed is None:
        return None
    if isinstance(feed, str):
        items = [s.strip() for s in feed.strip("[]() ").split(",") if s.strip()]
        return [float(s) for s in items]
    try:
        return list(feed)
    except TypeError:
        raise TypeError(
            f"feed must be a sequence or a comma-separated string, got {type(feed)}"
        )


def _ports(collection) -> dict:
    return {
        label: port
        for label, port in collection.ports.items()
        if label not in _SKIP_PORTS
    }


def _outputs(node) -> dict:
    return {label: port.value for label, port in _ports(node.outputs).items()}


# --------------------------------------------------------------------------- #
#  General-purpose condition nodes for WhileLoop (output: proceed)            #
# --------------------------------------------------------------------------- #


@as_function_node("proceed")
def ProceedWhileAbove(
    value: float = 0.0, threshold: float = 0.0, inclusive: bool = False
):
    """proceed = value > threshold (run another iteration while above);
    `inclusive`: value >= threshold."""
    proceed = (
        bool(float(value) >= float(threshold))
        if inclusive
        else bool(float(value) > float(threshold))
    )
    return proceed


@as_function_node("proceed")
def ProceedWhileBelow(
    value: float = 0.0, threshold: float = 0.0, inclusive: bool = False
):
    """proceed = value < threshold (run another iteration while below);
    `inclusive`: value <= threshold. A crack loading as wcrack stops once the
    tip distance exceeds the stop distance, i.e. proceeds while tip <= stop:
    inclusive=True."""
    proceed = (
        bool(float(value) <= float(threshold))
        if inclusive
        else bool(float(value) < float(threshold))
    )
    return proceed


@as_function_node("proceed")
def ProceedUntilConverged(values=None, tol: float = 1e-6, window: int = 1):
    """For condition_scope="history": `values` is the series so far. Stops
    (proceed False) once each of the last `window` changes is below `tol` in
    magnitude; proceeds while fewer than window + 1 values exist."""
    import numpy as np

    v = np.asarray(values if values is not None else [], dtype=float)
    window = int(window)
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if v.ndim != 1:
        raise ValueError("values must be a 1D series: use condition_scope='history'")
    proceed = True
    if len(v) >= window + 1:
        proceed = not bool(np.all(np.abs(np.diff(v[-(window + 1) :])) < float(tol)))
    return proceed


@as_function_node(["outputs", "history", "n_iterations", "stop_reason"])
def WhileLoop(
    body: Node,
    condition: Node = None,
    carry: str = "",
    feed=None,
    feed_input: str = "",
    record: str = None,
    condition_map: str = "",
    condition_sees: Literal["outputs", "inputs+outputs"] = "outputs",
    condition_scope: Literal["last", "history"] = "last",
    max_iterations: int = 100,
    on_max: Literal["raise", "stop"] = "raise",
    executor_for_body=None,
):
    """Run `body` repeatedly until `condition` says stop, the `feed` schedule
    ends, or `max_iterations` is reached.

    Where things run:
    - `executor_for_body` (e.g. a SlurmExecutor's output): every body run is
      submitted with it (one job per iteration), waiting for each result. The
      condition, carry and feed stay in this process (the GUI kernel). The
      executor must be modal (not `non_modal=True`). Results are cached by
      executorlib under the body's inputs, so re-running the loop replays the
      iterations already done (including a failed one: use a fresh cache
      directory after a fix).
    - the reserved `executor` port of this node (wire an executor into
      WhileLoop itself): the whole loop runs as one job. Don't combine it with
      `executor_for_body` (that would submit jobs from inside a job; raises
      when it detects a SLURM job).

    Each iteration:
    1. with a feed, `feed[i]` is written into the body input `feed_input`;
    2. the body runs, and a history row is recorded: `iteration` (1, 2, ...),
       the fed value (column `feed_input`), and the outputs named in `record`
       (None: every body output);
    3. with a condition, its inputs are filled and it runs. Its output
       `proceed` (or its only output) must be a bool; True means run another
       iteration;
    4. the loop stops when `proceed` is False ("condition"), the feed is used
       up ("schedule_end", a normal end), or `max_iterations` is reached (an
       error with `on_max="raise"`, else "max_iterations");
    5. otherwise `carry` writes body outputs into body inputs, e.g.
       "x -> x, K -> K_prev". `carry="auto"` carries every body output that
       has the name of a body input (the fed input excepted), so a body whose
       ports are named alike needs no pairs; "auto, K -> K_prev" adds renames.
       Empty (the default): nothing is carried.

    What the condition gets: each condition input whose name matches a value
    (after renaming with `condition_map`, "body_name -> condition_input") is
    set; unmatched inputs keep their own (plain or wired) values.
    - `condition_sees`: "outputs" (the body's outputs) or "inputs+outputs"
      (also the body's inputs as used in that iteration; an output wins over
      an input of the same name).
    - `condition_scope`: "last" (the current value) or "history" (the list of
      that value over all iterations so far, oldest first).
    - Special names, if the condition has such a port: `iteration` (number of
      completed iterations) and `history` (the record table so far, a
      DataFrame).

    Strings (`carry`, `record`, `condition_map`, and `feed` as "1.7, 1.8")
    survive a GUI save. Returns the body's outputs of the last iteration
    (dict), the history (DataFrame, one row per iteration), the number of
    iterations, and the stop reason.
    """
    import numpy as np
    import pandas as pd

    feed = _feed_values(feed)
    if condition is None and feed is None:
        raise ValueError(
            "WhileLoop needs a condition node, a feed, or both: nothing would stop it"
        )
    if int(max_iterations) < 1:
        raise ValueError(f"max_iterations must be >= 1, got {max_iterations}")
    if on_max not in ("raise", "stop"):
        raise ValueError(f"on_max must be 'raise' or 'stop', got {on_max!r}")
    if condition_sees not in ("outputs", "inputs+outputs"):
        raise ValueError(
            f"condition_sees must be 'outputs' or 'inputs+outputs', got {condition_sees!r}"
        )
    if condition_scope not in ("last", "history"):
        raise ValueError(
            f"condition_scope must be 'last' or 'history', got {condition_scope!r}"
        )

    if executor_for_body is not None:
        import os

        from aiflow.core.node import set_reserved_port
        from aiflow.core.remote_cache import is_non_modal

        if os.environ.get("SLURM_JOB_ID"):
            raise ValueError(
                "WhileLoop runs inside a SLURM job (SLURM_JOB_ID is set): executor_for_body would submit "
                "jobs from inside a job. Use either the WhileLoop's own executor or executor_for_body."
            )
        if is_non_modal(executor_for_body):
            raise ValueError(
                "executor_for_body must wait for each iteration: use a modal executor "
                "(SlurmExecutor(non_modal=False))"
            )
        set_reserved_port(
            body, "executor", executor_for_body
        )  # body.run() then submits via aiflow

    body_in = _ports(body.inputs)
    body_out_labels = list(_ports(body.outputs))
    carry = _carry_pairs(
        carry, body_out_labels, body_in, feed_input if feed is not None else ""
    )
    for out_label, in_label in carry:
        if out_label not in body_out_labels:
            raise ValueError(
                f"carry: {out_label!r} is not a body output; outputs: {body_out_labels}"
            )
        if in_label not in body_in:
            raise ValueError(
                f"carry: {in_label!r} is not a body input; inputs: {list(body_in)}"
            )
    if feed is not None:
        if feed_input not in body_in:
            raise ValueError(
                f"feed_input {feed_input!r} is not a body input; inputs: {list(body_in)}"
            )
        if not feed:
            raise ValueError("feed is empty")
    recorded = body_out_labels if record is None else _names(record)
    unknown = [r for r in recorded if r not in body_out_labels]
    if unknown:
        raise ValueError(
            f"record: {unknown} are not body outputs; outputs: {body_out_labels}"
        )

    # condition wiring: which condition input receives which named value
    cond_targets = {}
    if condition is not None:
        cond_in = _ports(condition.inputs)
        cond_out = list(_ports(condition.outputs))
        proceed_label = (
            "proceed"
            if "proceed" in cond_out
            else (cond_out[0] if len(cond_out) == 1 else None)
        )
        if proceed_label is None:
            raise ValueError(
                f"condition needs an output 'proceed' (or a single output); outputs: {cond_out}"
            )
        renamed = dict(
            _pairs(condition_map, "condition_map")
        )  # body name -> condition input
        visible = set(body_out_labels) | (
            set(body_in) if condition_sees == "inputs+outputs" else set()
        )
        for src, dst in renamed.items():
            if src not in visible:
                raise ValueError(
                    f"condition_map: {src!r} is not visible to the condition "
                    f"(condition_sees={condition_sees!r}); visible: {sorted(visible)}"
                )
            if dst not in cond_in:
                raise ValueError(
                    f"condition_map: {dst!r} is not a condition input; inputs: {list(cond_in)}"
                )
        # matching names first, then the explicit renames, which win on a clash
        for name in sorted(visible):
            if (
                name not in renamed
                and name in cond_in
                and name not in ("iteration", "history")
            ):
                cond_targets[name] = name
        for src, dst in renamed.items():
            cond_targets[dst] = src

    rows, series = [], {name: [] for name in set(cond_targets.values())}
    stop_reason = None
    n_iterations = 0
    limit = int(max_iterations)
    n_total = len(feed) if feed is not None else None

    while True:
        if n_iterations >= limit:
            if on_max == "raise":
                raise RuntimeError(
                    f"WhileLoop did not finish within max_iterations={limit}"
                )
            stop_reason = "max_iterations"
            break
        i = n_iterations
        if feed is not None:
            body.inputs[feed_input].value = feed[i]
        used_inputs = {label: port.value for label, port in _ports(body.inputs).items()}
        body.run()
        n_iterations += 1
        outputs = _outputs(body)

        row = {"iteration": n_iterations}
        if feed is not None:
            row[feed_input] = feed[i]
        row.update({label: outputs[label] for label in recorded})
        rows.append(row)

        values = dict(used_inputs) if condition_sees == "inputs+outputs" else {}
        values.update(outputs)
        for name in series:
            series[name].append(values[name])

        proceed = True
        if condition is not None:
            for dst, name in cond_targets.items():
                condition.inputs[dst].value = (
                    list(series[name]) if condition_scope == "history" else values[name]
                )
            if "iteration" in condition.inputs.ports:
                condition.inputs["iteration"].value = n_iterations
            if "history" in condition.inputs.ports:
                condition.inputs["history"].value = pd.DataFrame(rows)
            condition.run()
            proceed = condition.outputs[proceed_label].value
            if not isinstance(proceed, (bool, np.bool_)):
                raise TypeError(
                    f"condition output {proceed_label!r} must be a bool, got {type(proceed)}"
                )
            proceed = bool(proceed)

        print(
            f"WhileLoop: iteration {n_iterations}"
            + (" (body on executor_for_body)" if executor_for_body is not None else "")
            + (f", {feed_input} = {feed[i]}" if feed is not None else "")
            + (f", proceed = {proceed}" if condition is not None else "")
        )
        if not proceed:
            stop_reason = "condition"
            break
        if n_total is not None and n_iterations >= n_total:
            stop_reason = "schedule_end"
            break
        for out_label, in_label in carry:
            body.inputs[in_label].value = outputs[out_label]

    history = pd.DataFrame(rows)
    return outputs, history, n_iterations, stop_reason
