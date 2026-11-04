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

## Development

Dependencies are managed with [pixi](https://pixi.sh) in `pyproject.toml`; the
environment is locked in `pixi.lock`.

```bash
pixi install          # default environment, with aiflow_nodes installed editable
pixi run test         # unit tests with coverage
pixi run lint         # ruff check + ruff format --check
pixi run deptry       # imports vs. declared dependencies
pixi run hooks        # all pre-commit hooks on all files (as CI does)
pixi run pre-commit install  # once per clone: run the hooks on every commit
```

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
