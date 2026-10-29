# aiflow_nodes

Node library for pyiron/aiflow workflows.

## Layout

- `src/aiflow_nodes/` – the installable package
- `workflows/` – example and reference workflows (not part of the package)
- `tutorial/` – tutorial material (not part of the package)
- `tests/` – test suite

## Development

Dependencies are managed with [pixi](https://pixi.sh) in `pyproject.toml`; the
environment is locked in `pixi.lock`.

```bash
pixi install          # default environment, with aiflow_nodes installed editable
pixi run test         # unit tests with coverage
pixi run lint         # ruff check + ruff format --check
pixi run deptry       # imports vs. declared dependencies
```

aiflow is not published yet. pixi installs it from the private
`pyiron/aiflow` repository at the commit pinned in
`[tool.pixi.pypi-dependencies]`, so git needs read access to that repository
(gh credential helper or SSH). To move to a newer aiflow, change `rev` and run
`pixi install`.

Without pixi, `pip install -e ".[all]"` works once aiflow is installed; some
dependencies (e.g. the LAMMPS binary) only come from conda-forge.

To hide mechanical commits (file moves, renames) from `git blame`:

```bash
git config blame.ignoreRevsFile .git-blame-ignore-revs
```
