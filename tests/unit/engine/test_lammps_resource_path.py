"""The LAMMPS potential catalogue is found in the environment that runs the code.

``iprpy-data`` installs the catalogue under ``<env>/share/iprpy``, and upstream
looks for it only through ``CONDA_PREFIX`` / ``CONDA_DIR``.  Those are set by
*activating* an environment, which a Jupyter kernel, the VS Code notebook kernel
or the workflow assistant's MCP server never do: they start the environment's
``python`` directly.  Every potential node then failed with "No resource_path
found" although the catalogue sat right next to the interpreter.
"""

import os
import sys

import pytest

pytest.importorskip("ase")
pytest.importorskip("pyiron_lammps")

from ase.build import bulk  # noqa: E402

from aiflow_nodes.atomistic.engine.lammps import (  # noqa: E402
    GetPotential,
    ListPotentials,
    get_usable_potential_by_name,
)

pytestmark = pytest.mark.skipif(
    not os.path.isdir(os.path.join(sys.prefix, "share", "iprpy")),
    reason="LAMMPS potential catalogue (iprpy-data) not installed in this env",
)

AL = bulk("Al", "fcc", a=4.05, cubic=True)


@pytest.fixture
def unactivated(monkeypatch):
    """The environment as an unactivated interpreter sees it."""
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    monkeypatch.delenv("CONDA_DIR", raising=False)


def test_listing_potentials_needs_no_activated_environment(unactivated):
    assert ListPotentials(structure=AL).run()


def test_choosing_a_potential_needs_no_activated_environment(unactivated):
    assert GetPotential(structure=AL).run() is not None


def test_looking_up_a_potential_needs_no_activated_environment(unactivated):
    name = ListPotentials(structure=AL).run()[0]
    assert get_usable_potential_by_name(name)["Name"] == name
