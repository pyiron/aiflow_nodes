"""Node modules must not import their heavy dependencies at module level.

aiflow's GUI imports every node module to list its nodes, so a heavy
import at module level slows down loading the whole library.
"""

import subprocess
import sys

import pytest

CASES = [
    ("aiflow_nodes.dev_tools", "pyiron_base"),
    ("aiflow_nodes.atomistic.engine.vasp_new", "pymatgen"),
    ("aiflow_nodes.atomistic.structure.container_new", "pymatgen"),
    ("aiflow_nodes.atomistic.property.bulk", "matplotlib.pyplot"),
    ("aiflow_nodes.machine_learning.pipeline", "sklearn"),
    ("aiflow_nodes.validation_nodes", "sklearn"),
]


@pytest.mark.parametrize(("module", "heavy"), CASES)
def test_module_does_not_import_heavy_dependency(module, heavy):
    pytest.importorskip(heavy)
    script = f"import sys, {module}; print({heavy!r} in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False"
