# aiflow_nodes — instructions for coding agents

This file is for any coding agent (Claude Code, Codex, Copilot, Cursor, …)
and for human contributors.  `CLAUDE.md` only imports it; edit this file.

aiflow_nodes is the node library for [aiflow](https://github.com/pyiron/aiflow)
workflows: maths, DataFrames, plotting, atomistic structure building, DFT/MD
engines (LAMMPS, VASP, Quantum ESPRESSO, ASE), thermodynamics, ML potentials
and more.  Python ≥ 3.11, managed with pixi.

## How to work

These guidelines favour care over speed; use judgement on trivial tasks.

**Think before coding.**  State your assumptions.  If a request can be read in
more than one way, name the readings instead of picking one silently.  If
something is unclear, stop and ask.  If a simpler approach exists, say so.

**Simplicity first.**  Write the least code that solves the problem: no
features, options or abstractions nobody asked for, and no error handling for
cases that cannot happen.  If 200 lines could be 50, rewrite them.

**Surgical changes.**  Every changed line must trace back to the request.
Don't reformat, refactor or "improve" neighbouring code, comments or docs;
match the existing style.  Remove imports and helpers your own change made
unused, but only *mention* pre-existing dead code.

**Goal-driven execution.**  Turn a task into a check you can run: for a bug,
write a failing test that reproduces it, then make it pass; for a feature,
write tests for it first.  For multi-step work, state a short plan with a
verification step per item.  "Done" means the relevant tests,
`pixi run lint` and `pixi run pylint` pass.

## Layout

One distribution, `aiflow_nodes` (`src/aiflow_nodes/`):

| Path | What |
|---|---|
| `atomistic/` | Structures (`structure/`), engines (`engine/`: LAMMPS with Jinja templates, VASP, QE, ASE), calculators, properties (bulk, elastic, phonons, defects, calphy, …), ML potentials, assyst, thermodynamics |
| `electrochemistry/` | Electrode/electrolyte setup, VASP CCE plugins, analysis |
| `math_utils.py`, `utilities.py`, `dataframe.py`, `plotting.py`, `controls.py`, `executors.py` | General-purpose nodes |
| `databases/`, `experiment/`, `machine_learning/`, `scipy_nodes/`, `thermodynamics/`, `graphs/` | Smaller node families |
| `assistant_plugin.py` | Registers the library with the `aiflow.ai` assistant (`aiflow.node_libraries` entry point) |
| `guides/node_guide.md` | The library's guide for the assistant |

Outside the package: `workflows/` (example and reference workflows),
`tutorial/`, `tests/` (`unit/` mirrors the package, `integration/`, and
`static/` test data).

## Invariants (enforced by tests or easy to break)

- **Node modules import their heavy stack lazily.**  The core dependencies are
  in `[project] dependencies`; everything else belongs to an extra (`lammps`,
  `vasp`, `structure`, `thermodynamics`, `ml`, `assyst`, `gpaw`, `pyiron`,
  `database`, `viz`) and is imported inside the node that needs it.
- **Imports and calls must match the installed packages.**  pylint checks that
  imported names exist (including deferred imports inside node bodies),
  variables are bound and calls match the signature (`[tool.pylint]` in
  `pyproject.toml`).  Fix what it finds; most findings are real API drift.
- **Every node module imports and every referenced node path resolves.**
  `tests/unit/test_node_library_smoke.py` imports every node module, checks
  node bodies for undefined names, and resolves every `aiflow_nodes.…` path
  named in `workflows/*.py`, the node guide and aiflow's workflow guide.
  Moving or renaming a node means updating those references.
- **A node's hash includes its module path.**  Moving or renaming a node
  invalidates cached results.
- **`src/aiflow_nodes/guides/node_guide.md` is a runtime prompt** for the
  `aiflow.ai` assistant.  It must stay the only file in `guides/` (it is handed
  to an agent with `--add-dir`), keep the sections that
  `tests/unit/test_assistant_plugin.py` checks, and stay in step with the
  package layout.  ruff does not format Markdown here on purpose.
- **Describing the library must not import it.**  `assistant_plugin.py` parses
  the sources; don't make it import node modules.
- **Annotations are evaluated at runtime.**  Don't quote annotations in node
  signatures.

## Writing node or workflow code

aiflow's API differs from plain Python in ways that look wrong but are
intended; don't guess it from general Python knowledge.  Before writing any
`@as_function_node`, `@group_node` or `Workflow` code, including in tests:

1. Read the rules in the aiflow repository,
   `.claude/skills/aiflow-workflows/SKILL.md` (Claude Code loads it as the
   `aiflow-workflows` skill), and the guide it points to,
   `src/aiflow/ai/guides/llm_workflow_guide.md`.  In the `aiflow_env`
   workspace the aiflow clone is `../aiflow`; otherwise find it via
   `pixi run python -c "import aiflow; print(aiflow.__file__)"`.
2. Read `src/aiflow_nodes/guides/node_guide.md` and search the package for an
   existing node before writing a new one.

Two mistakes from this repository's history:

- To run a node's wrapped function directly, call `Node().func(...)`.
  `_func` belongs to the old core API and no longer exists.
- pylint reads a node call such as `BSpline()` as a call of the undecorated
  function and reports `no-value-for-parameter`.  That one is a false
  positive: disable it on that line with a comment saying why.

## Environment

The shell is usually not activated, and a bare `python` may be a system or
conda-base interpreter.  Run everything through pixi:

1. **A local, gitignored note** (`CLAUDE.local.md`, `AGENTS.local.md`) names an
   environment: use that.
2. **Otherwise** use this repository's workspace: `pixi run test`,
   `pixi run python -m pytest …`, `pixi run lint`.  When the checkout sits in
   the `aiflow_env` bundle (a `pixi.toml` in the parent directory that also
   holds `aiflow`), that environment works too; say which one you use.

Check once that `pixi run python -c "import aiflow_nodes; print(aiflow_nodes.__file__)"`
points into this checkout's `src/`.  Never create an environment or install,
upgrade or remove packages without asking, and never `pip install` into a
pixi env: changes go through `pyproject.toml` and `pixi.lock`.

aiflow is not on PyPI; pixi installs it from the private `pyiron/aiflow`
repository at the commit pinned in `[tool.pixi.pypi-dependencies]`.  Bump
`rev` only on purpose.

## Tasks

```bash
pixi run test                     # pytest with coverage (CI)
pixi run python -m pytest tests/unit/engine/test_lammps.py -v
pixi run pylint                   # imports, unbound names, call signatures (CI)
pixi run lint                     # ruff check + ruff format --check (CI)
pixi run hooks                    # all pre-commit hooks on all files (CI)
pixi run deptry                   # imports vs. declared dependencies
```

- pylint and the tests need the full dependency stack, so they run in the
  `test` (or `default`) environment; `lint` and `hooks` run in the light
  `lint` environment.
- The smoke test skips node modules whose third-party dependency is not
  installed; a dependency that is installed but no longer exposes a name is a
  failure.
- Integration tests under `tests/integration/` run real LAMMPS.

## Dependencies

`[project]` in `pyproject.toml` declares the requirements (core plus extras).
`[tool.pixi.package.run-dependencies]` is their conda side with the same
bounds, plus what only conda-forge provides (LAMMPS, potentials, pyiron
resources).  Change a requirement in both places, then `pixi install` to
update `pixi.lock`.  Exact versions live in the lock file only.  Run
`pixi run deptry` after touching imports or dependencies.

## Formatting and linting

- `ruff format .` and `ruff check .` (`--fix` for safe fixes and import
  sorting).  CI and the pre-commit hook check both; Markdown is not formatted.
- Configuration lives in `[tool.ruff]` of `pyproject.toml`; never add a
  `ruff.toml`.  ruff is pinned in both the `lint` feature and
  `.pre-commit-config.yaml`; bump them together.
- Lint rules are listed explicitly; add new ones one family at a time.
- pylint runs only the checks enabled in `[tool.pylint."messages control"]`,
  those that need the installed packages; ruff does the rest.  Don't enable
  `no-member` or `invalid-field-call`: aiflow builds node classes dynamically,
  so they only produce noise.
- Prefer fixing a finding over a `# noqa` or `# pylint: disable`; when one is
  right, say why in the comment.
- The pre-commit hooks strip notebook outputs (re-`git add` the notebook) and
  reject binary files and files over 500 kB.  Data belongs elsewhere (e.g.
  Zenodo).

## Branches and commits

- Never commit to `main`.  Work on a branch named in short kebab-case.  Push
  it and open a PR against `main` only when asked; the PR description says
  what changed, why, and how it was verified.
- Commit subjects are short and imperative, in sentence case ("Clean up the
  declared dependencies").  Keep mechanical changes such as a reformat or a
  file move in a commit of their own, and add it to `.git-blame-ignore-revs`.
