"""
Smoke test for the ``aiflow_nodes`` library itself.

The agentic benchmark asks the agent to reuse library nodes, and then charges it
for the outcome.  When a node is stale the agent pays twice: repair cycles for a
bug it did not write, and — because the node looked broken — a hand-rolled
reimplementation that is worse than what the library already had.  Two of the
three aiflow workflows that ran in the first full benchmark failed this way:
``M3GNet`` imported a class matgl had renamed, and a calphy node created its run
directory with a bare ``os.mkdir``.

Nothing here executes a node.  What is checked is that every module holding a
node imports, that node bodies read no undefined names, and that the node paths
the workflows and guides refer to exist.

Whether the deferred ``from X import Y`` statements inside node bodies still
resolve — where API drift hides, because a lazy import inside a function
survives collection and only explodes mid-run — is checked by pylint
(``pixi run -e test pylint``, ``no-name-in-module``), which reads the installed
packages instead of importing them.

Modules whose third-party dependency is simply *not installed* are skipped; a
dependency that *is* installed but no longer exposes the name is a failure.
"""

import ast
import builtins
import importlib
import re
from pathlib import Path

import pytest
from aiflow.ai.node_library import DEFAULT_EXCLUDE, NODE_DECORATORS, _decorator_name
from aiflow.ai.workflow_assistant import WORKFLOW_GUIDE

from aiflow_nodes.assistant_plugin import NODE_GUIDE, package_root

LIBRARY_ROOT = package_root()
REPO_ROOT = Path(__file__).resolve().parents[2]

#: What the scan skips: what the assistant's node index skips.
SCAN_EXCLUDE = DEFAULT_EXCLUDE


#: Packages that belong to this project.  A ``ModuleNotFoundError`` naming one of
#: these is never an uninstalled optional dependency — it is a stale import path
#: left behind by a refactoring, and it takes down every module that imports it.
#: ``pyiron_nodes.atomistic.property`` failed exactly this way: a node body still
#: said ``from pyiron_core.pyiron_workflow.data_fields import …``, whose root is
#: not ``pyiron_nodes``, so the whole package read as "optional dep missing".
FIRST_PARTY = frozenset(
    {
        "aiflow_nodes",
        "pyiron_nodes",
        "core",
        "pyiron_core",
        "pyiron_workflow",
        "pyiron_ai_nodes",
        "aiflow",
        # Now aiflow.database / aiflow.ai / aiflow.gui: an import of the old
        # name is stale, not an optional dependency.
        "pyiron_database",
        "pyiron_ai",
        "pyironflow",
    }
)


def _skip_unless_first_party(exc: ModuleNotFoundError, context: str) -> None:
    """Fail on a stale first-party import, skip on a missing third-party one."""
    root = (exc.name or "").split(".")[0]
    if root in FIRST_PARTY:
        pytest.fail(f"{context}: stale first-party import path — {exc}")
    pytest.skip(f"optional dependency missing: {exc.name}")


def _node_sources():
    """``(module_path, ast.FunctionDef | ast.ClassDef)`` for every node in the library."""
    package = LIBRARY_ROOT.name
    for path in sorted(LIBRARY_ROOT.rglob("*.py")):
        parts = path.relative_to(LIBRARY_ROOT).parts
        if any(p in SCAN_EXCLUDE for p in parts) or path.name == "__init__.py":
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:  # reported by test_every_node_module_parses
            continue
        module = ".".join((package, *parts[:-1], path.stem))
        for item in tree.body:
            if not isinstance(
                item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                continue
            if any(_decorator_name(d) in NODE_DECORATORS for d in item.decorator_list):
                yield module, item


def _package_modules():
    """Every ``__init__.py`` in the library, as a dotted module path.

    Package ``__init__``s hold no nodes, so the node scan never looked at them —
    yet two real bugs lived there: ``atomistic/property/__init__.py`` imported a
    ``calphy`` module that had been renamed to ``calphy.py_old``, and
    ``thermodynamics/landau/__init__.py`` said ``from landau import phases``,
    which silently resolved to the *installed third-party* ``landau`` package.
    """
    package = LIBRARY_ROOT.name
    for path in sorted(LIBRARY_ROOT.rglob("__init__.py")):
        parts = path.relative_to(LIBRARY_ROOT).parts
        if any(p in SCAN_EXCLUDE for p in parts):
            continue
        yield ".".join((package, *parts[:-1]))


NODE_MODULES = sorted({module for module, _ in _node_sources()})
PACKAGE_MODULES = sorted(set(_package_modules()))


@pytest.mark.parametrize("module", NODE_MODULES)
def test_every_node_module_imports(module):
    try:
        importlib.import_module(module)
    except ModuleNotFoundError as exc:
        _skip_unless_first_party(exc, module)


@pytest.mark.parametrize("module", PACKAGE_MODULES)
def test_every_package_init_imports(module):
    try:
        importlib.import_module(module)
    except ModuleNotFoundError as exc:
        _skip_unless_first_party(exc, module)
    except ImportError as exc:
        # ``from aiflow_nodes.x import y`` where ``y.py`` does not exist raises
        # ImportError, not ModuleNotFoundError — that is always our own bug.
        pytest.fail(f"{module}: {exc}")


def _global_names(code, seen=None):
    """Every ``LOAD_GLOBAL`` name reachable from *code*, nested functions included."""
    import dis

    seen = set() if seen is None else seen
    for instruction in dis.get_instructions(code):
        if instruction.opname in ("LOAD_GLOBAL", "LOAD_NAME"):
            # 3.11+ prefixes a ``NULL + `` marker onto method-call globals
            seen.add(str(instruction.argval).split()[-1])
    for const in code.co_consts:
        if hasattr(const, "co_code"):
            _global_names(const, seen)
    return seen


@pytest.mark.parametrize("module", NODE_MODULES)
def test_no_node_references_an_undefined_global(module):
    """A name a node body reads but nothing defines is a guaranteed ``NameError``.

    ``elastic.py`` carried this for a long time: ``import … as sym`` had been
    commented out while four ``sym.…`` calls stayed, so every node in the module
    raised at run time and nothing at import time noticed.  Bytecode is read, not
    executed, so this is cheap and safe.
    """
    try:
        imported = importlib.import_module(module)
    except ModuleNotFoundError as exc:
        _skip_unless_first_party(exc, module)
    except ImportError as exc:
        pytest.skip(f"{module} unusable in this environment: {exc}")

    defined = set(vars(imported)) | set(vars(builtins))
    missing = {}
    for name, obj in vars(imported).items():
        func = getattr(obj, "_original_func", None)
        if func is None or not hasattr(func, "__code__"):
            continue  # dataclass nodes have no function body
        if getattr(func, "__module__", None) != module:
            continue  # re-exported from elsewhere; checked where it lives
        undefined = sorted(n for n in _global_names(func.__code__) if n not in defined)
        if undefined:
            missing[name] = undefined

    assert not missing, (
        f"{module} — node bodies read names that are defined nowhere in the "
        f"module, in builtins, or as an import: {missing}. Each one is a "
        f"NameError the moment the node runs."
    )


#: Node names that still live in two places, each with a reason and an owner.
#: The point of the list is that it is *short and visible*: shrinking it is a
#: reviewable act, whereas a silent duplicate is a coin flip the agent has to win
#: every time it picks a node out of the generated index.
KNOWN_DUPLICATES = {
    # assyst vs assyst2 are two generations of the same pipeline; merging them
    # is a separate pass with its own physics review.
    "ElementInput",
    "PlotAtomsCells",
    "PlotAtomsHistogram",
    "PlotDistances",
    "PlotSPG",
    "SpaceGroupSampling",
    # assyst2's settings dataclass predates the ASE one and carries extra
    # ASSYST-specific fields; it goes when assyst2 is merged.
    "GenericOptimizerSettings",
    # structure/container's version operates on a StructureContainer, not on a
    # single Atoms object — same name, genuinely different input type.
    "CreateVacancy",
    # linearfit's builds an ACE design matrix; BSpline's builds a spline basis.
    # Same word, unrelated maths.
    "DesignMatrix",
    # controls.Slice slices a matrix by a string spec; cp_models.Slice is used
    # inside its own module with module-specific semantics.
    "Slice",
    # container_new.py is a richer replacement for container.py; merge is pending.
    "AddPristine",
    "GetStructureTable",
    "GetDefectTable",
    "GetPristineTable",
}


def test_no_duplicate_node_names():
    """One node name, one module — except for the explicitly allow-listed few."""
    homes: dict[str, set[str]] = {}
    for module, item in _node_sources():
        homes.setdefault(item.name, set()).add(module)

    duplicated = {
        name: sorted(mods)
        for name, mods in homes.items()
        if len(mods) > 1 and name not in KNOWN_DUPLICATES
    }
    assert not duplicated, (
        f"node names defined in more than one module: {duplicated}. The agent "
        f"picks nodes out of a flat generated index, so a duplicate is a choice "
        f"it can get wrong. Merge them, or add the name to KNOWN_DUPLICATES with "
        f"a reason."
    )

    stale = sorted(n for n in KNOWN_DUPLICATES if len(homes.get(n, ())) < 2)
    assert not stale, (
        f"KNOWN_DUPLICATES lists names that are no longer duplicated: {stale}. "
        f"Delete the entries so the list keeps meaning something."
    )


def _referenced_node_paths():
    """``(source, dotted_path)`` for every ``aiflow_nodes`` node referenced in-repo.

    Covers the two places a stale path hides where no test would ever run it:
    the example workflow scripts and the docs fed to the agent as context.
    """
    package = LIBRARY_ROOT.name
    import_re = re.compile(
        rf"^[ \t]*from[ \t]+({package}(?:\.\w+)+)[ \t]+import[ \t]+([\w, ]+)",
        re.MULTILINE,
    )
    attr_re = re.compile(rf"\b{package}((?:\.\w+){{2,}})\b")

    def _from_text(text, source):
        for module, names in import_re.findall(text):
            for name in names.replace("(", "").split(","):
                name = name.strip()
                if name and name != "*":
                    yield source, f"{module}.{name}"

    py_sources = sorted((REPO_ROOT / "workflows").glob("*.py"))
    doc_sources = [WORKFLOW_GUIDE, NODE_GUIDE]
    for path in py_sources + doc_sources:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        yield from _from_text(text, path.name)

    # attribute-style references such as ``pn.math_utils.Linspace`` inside docs
    for path in doc_sources:
        if not path.is_file():
            continue
        for tail in attr_re.findall(path.read_text(encoding="utf-8", errors="replace")):
            yield path.name, f"{package}{tail}"


REFERENCED_NODE_PATHS = sorted(set(_referenced_node_paths()))


@pytest.mark.parametrize("source,dotted", REFERENCED_NODE_PATHS, ids=lambda v: str(v))
def test_every_referenced_node_path_resolves(source, dotted):
    module, _, name = dotted.rpartition(".")
    try:
        imported = importlib.import_module(module)
    except ModuleNotFoundError as exc:
        if (exc.name or "").split(".")[0] in FIRST_PARTY:
            pytest.fail(f"{source} references {dotted}, but {exc}")
        pytest.skip(f"optional dependency missing: {exc.name}")
    except ImportError as exc:
        pytest.skip(f"{module} unusable in this environment: {exc}")

    if hasattr(imported, name):
        return
    try:
        importlib.import_module(dotted)
    except ImportError:
        pytest.fail(f"{source} references `{dotted}`, which does not exist.")


def test_the_library_index_is_not_empty():
    """Guards the scan itself: a broken walk would make every test above vacuous."""
    assert len(NODE_MODULES) > 50
    assert len(PACKAGE_MODULES) > 10
    assert len(REFERENCED_NODE_PATHS) > 20
