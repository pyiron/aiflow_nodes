"""The nodes that read stored outputs back from the instance database.

They passed ``node_hash=`` to aiflow's ``restore_node_outputs``, which has no
such parameter.  ``GetStoredOutput`` crashed; ``GetHashesByUpstreamQualname``
swallowed the ``TypeError`` and silently returned an empty table.

The database is a stand-in (an SQLite table with the columns the nodes read);
aiflow's restore functions are mocked with their real signatures, so a call
that does not match them fails here as it would in production.
"""

from types import SimpleNamespace
from unittest import mock

import pandas as pd
import pytest

sqlalchemy = pytest.importorskip("sqlalchemy")
node_module = pytest.importorskip("aiflow.database.instance_database.node")

import aiflow.database  # noqa: E402

from aiflow_nodes.databases.node_hash_db import (  # noqa: E402
    GetHashesByUpstreamQualname,
    GetStoredOutput,
)

ROWS = [
    {"hash": "bulk", "qualname": "Bulk", "inputs": {}, "connected_inputs": []},
    {
        "hash": "neb",
        "qualname": "RunNEB",
        "inputs": {"structure": "bulk@structure"},
        "connected_inputs": ["structure"],
    },
]
STORED = pd.DataFrame({"energy": [1.0, 2.0]})


@pytest.fixture
def db(tmp_path):
    engine = sqlalchemy.create_engine("sqlite://")
    metadata = sqlalchemy.MetaData()
    table = sqlalchemy.Table(
        "nodes",
        metadata,
        sqlalchemy.Column("hash", sqlalchemy.String, primary_key=True),
        sqlalchemy.Column("qualname", sqlalchemy.String),
        sqlalchemy.Column("inputs", sqlalchemy.JSON),
        sqlalchemy.Column("connected_inputs", sqlalchemy.JSON),
    )
    metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(table.insert(), ROWS)
    return SimpleNamespace(engine=engine, table=table, storage_path=str(tmp_path))


@pytest.fixture
def restore(monkeypatch):
    """Mock aiflow's restore functions, keeping their signatures."""
    node = SimpleNamespace(outputs={"df": SimpleNamespace(value=STORED)})
    from_db = mock.create_autospec(
        aiflow.database.restore_node_from_database, return_value=(node, None)
    )
    outputs = mock.create_autospec(node_module.restore_node_outputs, return_value=True)
    monkeypatch.setattr(aiflow.database, "restore_node_from_database", from_db)
    monkeypatch.setattr(node_module, "restore_node_outputs", outputs)
    return outputs


def test_get_stored_output_restores_the_node(db, restore):
    result = GetStoredOutput._original_func(db=db, index=1)
    pd.testing.assert_frame_equal(result, STORED)
    restore.assert_called_once()


def test_get_hashes_by_upstream_qualname_finds_the_match(db, restore):
    result = GetHashesByUpstreamQualname._original_func(
        db=db, qualname="RunNEB", contains="Bulk"
    )
    assert list(result.node_id) == [1, 1]
    assert list(result.energy) == [1.0, 2.0]
