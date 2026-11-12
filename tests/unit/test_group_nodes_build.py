"""Group nodes must build with their default arguments (issue #28)."""

import importlib

import pytest

GROUPS = [
    ("aiflow_nodes.atomistic.property.thermodynamics", "GetChemicalPotential"),
    ("aiflow_nodes.atomistic.structure.build", "CubicBulkCell"),
    ("aiflow_nodes.atomistic.structure.calc", "FitDiffPotential"),
    ("aiflow_nodes.atomistic.structure.calc", "FitDiffPotential2"),
    ("aiflow_nodes.atomistic.thermodynamics.landau.plot", "ComputPhaseDiagram"),
    ("aiflow_nodes.thermodynamics.cp_models", "compute_cp_from_G"),
]


@pytest.mark.parametrize(("module", "name"), GROUPS)
def test_group_builds_without_arguments(module, name):
    try:
        node_class = getattr(importlib.import_module(module), name)
    except ModuleNotFoundError as exc:
        pytest.skip(f"optional dependency missing: {exc.name}")
    node_class()
