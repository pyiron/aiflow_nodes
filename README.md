# aiflow_nodes

Node library for pyiron/aiflow workflows.

## Layout

- `src/aiflow_nodes/` – the installable package
- `workflows/` – example and reference workflows (not part of the package)
- `tutorial/` – tutorial material (not part of the package)
- `tests/` – test suite

Node-specific documentation sits next to the code, e.g.
[VASP nodes](src/aiflow_nodes/atomistic/engine/vasp_new_overview.md) and
[VASP electrochemistry (CCE)](src/aiflow_nodes/electrochemistry/add_potential/CCE-electrode_overview.md).

## Changes

### Changed (breaking)

aiflow_nodes no longer carries copies of aiflow code; use aiflow's public API
instead.  Node hashes change, so results cached by the removed nodes are not
found again.

- `aiflow_nodes.executors` is removed.  Use the nodes of the same name in
  `aiflow.core.stdlib` (`ThreadPoolExecutor`, `ProcessPoolExecutor`,
  `ForkExecutor`, `SingleNodeExecutor`, `FluxClusterExecutor`,
  `SlurmExecutor`, `SlurmAdvancedSettings`).  `ThreadPoolExecutorNode` becomes
  `ThreadPoolExecutor`, `ProcessPoolExecutorNode` becomes
  `ProcessPoolExecutor`; `fcc_metals` is gone.
- `aiflow_nodes.utilities` is removed.  `Prepend`, `Append`, `GetItem` and the
  `ListOf…` nodes are in `aiflow.core.stdlib`; `Index` becomes `GetItem`, and
  `List5` becomes `ListOf`.
- From `aiflow_nodes.controls`, `branch`, `while_loop`, `loop_until`,
  `recursive`, `GetAttribute`, `Print` and `Or` are removed.  Use `Branch`,
  `WhileLoop`, `GetAttribute`, `Print` and `Or` from `aiflow.core.stdlib`.
  aiflow's `GetAttribute` raises `AttributeError` for a missing attribute
  instead of returning `None`.
- `aiflow_nodes.databases.node_hash_db` is removed.  Its nodes are in
  `aiflow.database.nodes`, where `GetNode` returns an `aiflow.core.Graph`
  instead of a GUI graph.
- `iterate` and `IterToDataFrame` keep their signatures and run on aiflow's
  sweep (`aiflow.core.run_sweep`).  `IterToDataFrame` no longer prints
  anything for `debug=True`, and rejects a template with an output called
  `error`.

## Development

Dependencies are managed with [pixi](https://pixi.sh) in `pyproject.toml`; the
environment is locked in `pixi.lock`.

```bash
pixi install          # default environment, with aiflow_nodes installed editable
pixi run test         # unit tests with coverage
pixi run lint         # ruff check + ruff format --check
pixi run deptry       # imports vs. declared dependencies
pixi run hooks        # all pre-commit hooks on all files (as CI does)
pixi run pre-commit install  # once per clone: lint on commit, tests on push
```

The tests on push run in the active environment, so push from a `pixi shell`
(here, or in the aiflow_env workspace when you develop aiflow too).

The hooks strip notebook outputs and reject binary files and files over
500 kB; see `.pre-commit-config.yaml`.

aiflow is not published yet. pixi installs it from the private
`pyiron/aiflow` repository at the commit pinned in
`[tool.pixi.pypi-dependencies]`, so git needs read access to that repository
(gh credential helper or SSH). To move to a newer aiflow, change `rev` and run
`pixi install`.

Without pixi, `pip install -e ".[all]"` works once aiflow is installed; some
dependencies (e.g. the LAMMPS binary) only come from conda-forge.

`[project]` in `pyproject.toml` declares the requirements: the core in
`dependencies`, each node family's stack in an extra (`lammps`, `vasp`,
`structure`, `thermodynamics`, `ml`, `assyst`, `gpaw`, `pyiron`, `database`,
`viz`). `[tool.pixi.package.run-dependencies]` is their conda side, with the
same bounds, plus what only conda-forge provides; exact versions live in
`pixi.lock`. From PyPI, the `lammps` and `gpaw` extras install the Python side
only; the LAMMPS and GPAW programs come from conda-forge or a local build.

To hide mechanical commits (file moves, renames) from `git blame`:

```bash
git config blame.ignoreRevsFile .git-blame-ignore-revs
```
