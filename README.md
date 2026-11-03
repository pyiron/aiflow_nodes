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
`pyiron/aiflow` repository at the commit pinned in `[tool.pixi.dev]` (its
conda dependencies) and `[tool.pixi.pypi-dependencies]` (the package), so git
needs read access to that repository (gh credential helper or SSH). To move to
a newer aiflow, change both revs and run `pixi install`.

Without pixi, `pip install -e ".[all]"` works once aiflow is installed; some
dependencies (e.g. the LAMMPS binary) only come from conda-forge.

### Dependencies

`[project]` in `pyproject.toml` is the only hand-written list of requirements:
the core in `dependencies`, each node family's stack in an extra (`lammps`,
`vasp`, `structure`, `thermodynamics`, `ml`, `assyst`, `gpaw`, `pyiron`,
`database`, `viz`).  `recipe/recipe.yaml` is the conda packaging, one output
per extra (`aiflow-nodes`, `aiflow-nodes-lammps`, ...); it adds only what PyPI
cannot provide, the LAMMPS program and the potential and resource data.
`tools/check_conda_recipe.py` (pre-commit hook, CI) fails when the two
disagree, so change a requirement in both.  This workspace installs the conda
dependencies of every output through `[tool.pixi.dev]`, gpaw on linux-64 only;
exact versions live in `pixi.lock`, not in the manifest.

From PyPI, the `lammps` and `gpaw` extras install the Python side only; the
LAMMPS and GPAW programs come from conda-forge or a local build.

To hide mechanical commits (file moves, renames) from `git blame`:

```bash
git config blame.ignoreRevsFile .git-blame-ignore-revs
```
