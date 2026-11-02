"""``aiflow_nodes`` as a node library for the :mod:`aiflow.ai` assistant.

Registered under the ``aiflow.node_libraries`` entry point (see this
repository's ``pyproject.toml``), so installing ``aiflow_nodes`` is all it takes
for the assistant to compose workflows from it.

Describing the library does not import its node modules: the source directory
is located with :func:`importlib.util.find_spec` and parsed, never executed, so
the heavy dependencies behind the nodes load only once a workflow uses one.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from aiflow.ai.node_library import NodeLibrary

#: Import name of the library this plugin describes.
PACKAGE = "aiflow_nodes"

#: The library-specific companion to aiflow's ``docs/llm_workflow_guide.md``.
#: It sits in a directory of its own so that it can be handed to an agent with
#: ``--add-dir`` without also exposing the rest of the package.
GUIDES_DIR = Path(__file__).resolve().parent / "guides"
NODE_GUIDE = GUIDES_DIR / "node_guide.md"

#: Shown in ``NODE_INDEX.md`` as the example import.
EXAMPLE_IMPORT = f"from {PACKAGE}.atomistic.structure.build import Bulk"

#: Parts of the package that are not nodes for a workflow to use: the benchmark
#: harness's own nodes.
EXCLUDE = ("benchmark.py",)

DESCRIPTION = (
    "computational materials science: maths, DataFrames, plotting, control "
    "flow, atomistic structure building, interatomic potentials, DFT/MD "
    "engines and executors"
)


def package_root() -> Path | None:
    """The ``aiflow_nodes`` source directory, or ``None`` when not installed."""
    try:
        spec = importlib.util.find_spec(PACKAGE)
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    root = Path(list(spec.submodule_search_locations)[0])
    return root if root.is_dir() else None


def library(root: str | Path | None = None) -> NodeLibrary | None:
    """The :class:`~aiflow.ai.node_library.NodeLibrary` for ``aiflow_nodes``.

    *root* overrides where the sources are, e.g. a mirror the agent may read.
    ``None`` when the library is not installed.
    """
    root = Path(root) if root is not None else package_root()
    if root is None:
        return None
    return NodeLibrary(
        name=PACKAGE,
        root=root,
        description=DESCRIPTION,
        guide=NODE_GUIDE,
        example_import=EXAMPLE_IMPORT,
        exclude=EXCLUDE,
    )
