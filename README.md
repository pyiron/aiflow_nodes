# aiflow_nodes

Node library for pyiron/aiflow workflows.

## Layout

- `src/aiflow_nodes/` – the installable package
- `workflows/` – example and reference workflows (not part of the package)
- `tutorial/` – tutorial material (not part of the package)
- `tests/` – test suite

## Development

```bash
pip install -e .
python -m unittest discover tests
```

To hide mechanical commits (file moves, renames) from `git blame`:

```bash
git config blame.ignoreRevsFile .git-blame-ignore-revs
```
