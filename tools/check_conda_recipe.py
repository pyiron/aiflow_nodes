"""Check that recipe/recipe.yaml mirrors the requirements in pyproject.toml.

pyproject.toml [project] is the source of truth.  The recipe has one output per
extra: the output named like the project holds `dependencies`, and the output
`<name>-<extra>` holds that extra's requirements next to an exact
``pin_subpackage`` of the main output.  An extra that refers to another extra
of the same project (``aiflow[sqlalchemy]``) is a ``pin_subpackage`` of that
output.

Requirements are compared by PyPI name and version specifier.  Conda names are
translated to PyPI names with the mapping pixi uses (prefix-dev/parselmouth),
corrected by ``conda-to-pypi`` below.  Settings, in pyproject.toml::

    [tool.check-conda-recipe]
    ignore-extras = ["dev"]           # extras without a conda output
    conda-only = ["lammps"]           # recipe requirements without a PyPI twin
    [tool.check-conda-recipe.conda-to-pypi]
    python-ace = "python-ace"         # where the mapping is wrong

This file is kept identical in pyiron/aiflow and pyiron/aiflow_nodes.

Usage: python tools/check_conda_recipe.py [--root DIR] [--offline]
Exit status 1 when the two disagree.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import tomllib
import urllib.request
from pathlib import Path

import yaml

MAPPING_URL = (
    "https://raw.githubusercontent.com/prefix-dev/parselmouth/main/"
    "files/compressed_mapping.json"
)
MAPPING_CACHE = Path.home() / ".cache" / "check-conda-recipe" / "mapping.json"
MAPPING_MAX_AGE = 7 * 24 * 3600

PYPI_REQ = re.compile(
    r"^\s*(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)\s*"
    r"(?:\[(?P<extras>[^\]]*)\])?\s*(?P<spec>[^;]*?)\s*(?P<marker>;.*)?$"
)
PIN = re.compile(r"""^\$\{\{\s*pin_subpackage\(\s*["']([^"']+)["']""")


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def load_mapping(offline: bool) -> dict[str, str]:
    """Conda name -> PyPI name, cached for a week."""
    fresh = (
        MAPPING_CACHE.exists()
        and time.time() - MAPPING_CACHE.stat().st_mtime < MAPPING_MAX_AGE
    )
    if not fresh and not offline:
        try:
            with urllib.request.urlopen(MAPPING_URL, timeout=30) as response:
                data = response.read()
            MAPPING_CACHE.parent.mkdir(parents=True, exist_ok=True)
            MAPPING_CACHE.write_bytes(data)
        except OSError as error:
            print(f"warning: cannot fetch the conda->PyPI mapping: {error}")
    if MAPPING_CACHE.exists():
        return {k: v for k, v in json.loads(MAPPING_CACHE.read_text()).items() if v}
    print("warning: no conda->PyPI mapping; comparing conda names as PyPI names")
    return {}


def pypi_requirements(reqs: list[str], project: str) -> dict[str, str]:
    """PyPI requirements -> {normalized name or "pin:<output>": specifier}."""
    result = {}
    for req in reqs:
        match = PYPI_REQ.match(req)
        if not match:
            sys.exit(f"cannot parse requirement {req!r}")
        if match["marker"]:
            sys.exit(f"environment markers are not supported: {req!r}")
        name = normalize(match["name"])
        spec = match["spec"].replace(" ", "")
        if name == project:
            for extra in (match["extras"] or "").split(","):
                result[f"pin:{project}-{normalize(extra.strip())}"] = ""
        else:
            result[name] = spec
    return result


def conda_requirements(
    run: list, mapping: dict[str, str], overrides: dict[str, str], conda_only: set
) -> dict[str, str]:
    """Recipe run requirements -> {PyPI name or "pin:<output>": specifier}."""
    result = {}
    for req in run:
        if not isinstance(req, str):
            sys.exit(f"unsupported recipe requirement (selector?): {req!r}")
        pin = PIN.match(req)
        if pin:
            result[f"pin:{pin[1]}"] = ""
            continue
        name, *rest = req.split()
        if name == "python" or name in conda_only:
            continue
        spec = rest[0] if rest and rest[0] != "*" else ""
        pypi = overrides.get(name) or mapping.get(name) or name
        result[normalize(pypi)] = spec
    return result


def compare(output: str, expected: dict, actual: dict) -> list[str]:
    errors = []
    for name in sorted(expected.keys() - actual.keys()):
        errors.append(f"{output}: missing {name} {expected[name]}".rstrip())
    for name in sorted(actual.keys() - expected.keys()):
        errors.append(f"{output}: not in pyproject.toml: {name}")
    for name in sorted(expected.keys() & actual.keys()):
        if expected[name] != actual[name]:
            errors.append(
                f"{output}: {name}: pyproject.toml has {expected[name] or 'no bound'}"
                f", the recipe {actual[name] or 'no bound'}"
            )
    return errors


def check(root: Path, offline: bool) -> list[str]:
    pyproject = tomllib.loads((root / "pyproject.toml").read_text())
    recipe = yaml.safe_load((root / "recipe" / "recipe.yaml").read_text())
    project = pyproject["project"]
    config = pyproject.get("tool", {}).get("check-conda-recipe", {})
    ignore = {normalize(e) for e in config.get("ignore-extras", [])}
    conda_only = set(config.get("conda-only", []))
    overrides = config.get("conda-to-pypi", {})
    mapping = load_mapping(offline)
    core = normalize(project["name"])

    expected = {core: pypi_requirements(project.get("dependencies", []), core)}
    for extra, reqs in project.get("optional-dependencies", {}).items():
        if normalize(extra) not in ignore:
            # Every extra output pins the main output.
            expected[f"{core}-{normalize(extra)}"] = {
                f"pin:{core}": "",
                **pypi_requirements(reqs, core),
            }

    errors = []
    version = recipe.get("context", {}).get("version")
    if version != project["version"]:
        errors.append(
            f"version: pyproject.toml has {project['version']}, the recipe {version}"
        )
    outputs = {
        normalize(o["package"]["name"]): o.get("requirements", {}).get("run", [])
        for o in recipe["outputs"]
    }
    for name in sorted(expected.keys() - outputs.keys()):
        errors.append(f"no recipe output {name}")
    for name in sorted(outputs.keys() - expected.keys()):
        errors.append(f"recipe output {name} matches no extra in pyproject.toml")
    for name in sorted(expected.keys() & outputs.keys()):
        actual = conda_requirements(outputs[name], mapping, overrides, conda_only)
        errors += compare(name, expected[name], actual)
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    errors = check(args.root, args.offline)
    for error in errors:
        print(error)
    if errors:
        sys.exit(1)
    print("recipe/recipe.yaml matches pyproject.toml")


if __name__ == "__main__":
    main()
