"""Node modules must not import heavy packages at module level.

aiflow's GUI imports every node module to list its nodes, so a heavy import at
module level slows down loading the whole library.  The check reads the sources
instead of importing them: it needs none of the packages installed, and it only
looks at what this library imports itself.
"""

import ast

from aiflow.ai.node_library import DEFAULT_EXCLUDE

from aiflow_nodes.assistant_plugin import package_root

LIBRARY_ROOT = package_root()

#: Packages measured to add noticeable import time on top of the core stack.
HEAVY = (
    "landau",
    "matplotlib.pyplot",
    "phonopy",
    "pymatgen",
    "pyiron_base",
    "seaborn",
    "sklearn",
    "structuretoolkit",
)

#: Imports that must stay at module level, keyed by path below the package.
EXEMPT = {
    "atomistic/property/phonons.py": {
        "phonopy": "Phonopy appears in node signatures, which are evaluated at runtime",
    },
    "atomistic/thermodynamics/landau/phases.py": {
        "landau": "landau.phases types appear in node signatures",
    },
    "atomistic/thermodynamics/landau/plot.py": {
        "landau": "CheckTemperatureInterpolation's annotation is resolved at runtime",
    },
}


def _module_level_imports(statements):
    """Import statements that run when the module is imported."""
    for stmt in statements:
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            yield stmt
        elif isinstance(stmt, ast.If):
            if "TYPE_CHECKING" not in ast.unparse(stmt.test):
                yield from _module_level_imports(stmt.body)
            yield from _module_level_imports(stmt.orelse)
        elif isinstance(stmt, ast.Try):
            for block in (stmt.body, stmt.orelse, stmt.finalbody):
                yield from _module_level_imports(block)
            for handler in stmt.handlers:
                yield from _module_level_imports(handler.body)
        elif isinstance(stmt, ast.ClassDef):
            yield from _module_level_imports(stmt.body)


def _imported_names(stmt):
    if isinstance(stmt, ast.Import):
        return [alias.name for alias in stmt.names]
    if stmt.level or not stmt.module:
        return []
    return [f"{stmt.module}.{alias.name}" for alias in stmt.names]


def _heavy_package(name):
    for package in HEAVY:
        if name == package or name.startswith(package + "."):
            return package
    return None


def _heavy_imports():
    """``(path, line, imported name, heavy package)`` for every module-level hit."""
    for path in sorted(LIBRARY_ROOT.rglob("*.py")):
        rel = path.relative_to(LIBRARY_ROOT)
        if any(part in DEFAULT_EXCLUDE for part in rel.parts):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for stmt in _module_level_imports(tree.body):
            for name in _imported_names(stmt):
                package = _heavy_package(name)
                if package is not None:
                    yield rel.as_posix(), stmt.lineno, name, package
                    break


def test_no_heavy_import_at_module_level():
    offenders = [
        f"{path}:{line} {name}"
        for path, line, name, package in _heavy_imports()
        if package not in EXEMPT.get(path, {})
    ]
    assert not offenders, "import these inside the node body:\n" + "\n".join(offenders)


def test_every_exemption_is_still_needed():
    found = {(path, package) for path, _, _, package in _heavy_imports()}
    stale = [
        f"{path}: {package}"
        for path, packages in EXEMPT.items()
        for package in packages
        if (path, package) not in found
    ]
    assert not stale, "remove these from EXEMPT:\n" + "\n".join(stale)
