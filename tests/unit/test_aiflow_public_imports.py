"""Every aiflow import in this repository uses aiflow's public API.

aiflow's public API is what its public packages list in ``__all__``; every
other module (``aiflow.core.remote_cache``, ``aiflow.core.node``,
``aiflow.gui``, …) is private and may change or disappear without notice.  A
private import that sits inside a node body only fails when the node runs, so
the check reads the sources instead of importing them: every ``.py`` file and
every notebook code cell, imports at any depth included.
"""

import ast
import importlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCANNED = ("src", "tests", "workflows", "tutorial")
PUBLIC_PACKAGES = (
    "aiflow.core",
    "aiflow.core.stdlib",
    "aiflow.database",
    "aiflow.database.nodes",
    "aiflow.ai",
)
#: Directories that hold no code of this repository, such as Jupyter's checkpoints.
SKIPPED = ("__pycache__", ".ipynb_checkpoints")


def _files(root, pattern):
    for path in sorted((REPO / root).rglob(pattern)):
        if not any(part in SKIPPED for part in path.parts):
            yield path


def _sources():
    """(path, source) of every Python file and notebook code cell."""
    for root in SCANNED:
        for path in _files(root, "*.py"):
            yield path, path.read_text(encoding="utf-8")
        for path in _files(root, "*.ipynb"):
            notebook = json.loads(path.read_text(encoding="utf-8"))
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code":
                    lines = cell["source"]
                    lines = (
                        lines.splitlines(keepends=True)
                        if isinstance(lines, str)
                        else lines
                    )
                    # IPython magics and shell escapes are not Python.
                    code = "".join(
                        "\n" if line.lstrip().startswith(("%", "!")) else line
                        for line in lines
                    )
                    yield path, code


def _aiflow_imports(source):
    """(line, module, name) per imported aiflow name; name is None for ``import``."""
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "aiflow":
                    yield node.lineno, alias.name, None
        elif (
            isinstance(node, ast.ImportFrom)
            and node.level == 0
            and node.module.split(".")[0] == "aiflow"
        ):
            for alias in node.names:
                yield node.lineno, node.module, alias.name


def _is_public(module, name):
    if name is None:
        return module in PUBLIC_PACKAGES
    if module == "aiflow":
        return f"aiflow.{name}" in PUBLIC_PACKAGES
    return module in PUBLIC_PACKAGES and name in importlib.import_module(module).__all__


def test_aiflow_is_imported_only_through_its_public_names():
    imports = [
        (path, *imp) for path, source in _sources() for imp in _aiflow_imports(source)
    ]
    # Every scanned root, and the notebooks, take part, so a scan that loses
    # one of them fails here instead of passing quietly.
    scanned = {path.relative_to(REPO).parts[0] for path, *_ in imports}
    assert scanned == set(SCANNED)
    assert any(path.suffix == ".ipynb" for path, *_ in imports)
    private = [
        f"{path.relative_to(REPO)}:{line}: "
        + (f"import {module}" if name is None else f"from {module} import {name}")
        for path, line, module, name in imports
        if not _is_public(module, name)
    ]
    assert not private, (
        "Import these from a public aiflow package ("
        + ", ".join(PUBLIC_PACKAGES)
        + ") instead:\n"
        + "\n".join(private)
    )
