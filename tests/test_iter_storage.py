"""Per-cell storage of ``iterate`` and ``IterToDataFrame`` sweeps.

A template with ``store=True`` stores every cell as its own row, linked to the
sweep, and a re-run restores the cells instead of executing them.  The sweeps
run in a ``Workflow`` against an in-memory database double, so no PostgreSQL
server is required.  Everything here goes through the public nodes, so it
characterizes the behaviour whichever sweep engine is underneath.
"""

import pathlib
import sys
import tempfile
import threading
from concurrent.futures import Future, ThreadPoolExecutor

import cloudpickle
import pytest
from aiflow.core import (
    FileStore,
    Workflow,
    as_function_node,
    get_hash,
    set_reserved_port,
)
from aiflow.database import PostgreSQLInstanceDatabase

from aiflow_nodes.controls import IterToDataFrame, iterate

# Jobs of the executorlib double below run in a fresh interpreter, which cannot
# import this test module; ship its functions by value instead.
cloudpickle.register_pickle_by_value(sys.modules[__name__])


class MemDB(PostgreSQLInstanceDatabase):
    """A ``PostgreSQLInstanceDatabase`` whose rows live in a dict.

    Its store methods (``lookup``, ``save``, ``mark_submitted``, …) are the real
    ones, so it is an ``aiflow.core.CheckpointStore``; only the row access is in
    memory, so no server is needed.
    """

    def __new__(cls, *args, **kwargs):
        # Not the per-connection singleton: every double is a database of its own.
        return object.__new__(cls)

    def __init__(self, storage_path):
        self.storage_path = pathlib.Path(storage_path)
        self.rows = {}

    def has_column(self, name):
        return True

    def init(self):
        pass

    def drop(self):
        self.rows.clear()

    def close(self):
        pass

    def create(self, node):
        """First write wins, as ``INSERT ... ON CONFLICT DO NOTHING`` does."""
        self.rows.setdefault(node.hash, node)
        return node.hash

    def read(self, hash):
        return self.rows.get(hash)

    def update(self, hash, **kwargs):
        for k, v in kwargs.items():
            setattr(self.rows[hash], k, v)

    def delete(self, hash):
        self.rows.pop(hash, None)


class LockedDB(MemDB):
    """Like a real database handle: holds a lock, so it cannot be pickled."""

    def __init__(self, storage_path):
        super().__init__(storage_path)
        self._lock = threading.Lock()


class PicklingExecutor:
    """Serialises every task the way executorlib does, then runs it here."""

    def __init__(self):
        self.submitted = 0

    def submit(self, fn, *args, **kwargs):
        self.submitted += 1
        payload = cloudpickle.dumps((fn, args, kwargs))  # raises if unpicklable
        fn, args, kwargs = cloudpickle.loads(payload)
        future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except Exception as exc:  # pragma: no cover - surfaced via the future
            future.set_exception(exc)
        return future


class CacheExecutor:
    """executorlib's file cache without a scheduler: recoverable, but local.

    Wraps ``SingleNodeExecutor`` and exposes ``info["executor_kwargs"]`` the way
    the cluster executors do, so aiflow treats it as one.
    """

    def __init__(self, directory, on_submit=None):
        from executorlib import SingleNodeExecutor

        self._ex = SingleNodeExecutor(cache_directory=directory)
        self.info = {"executor_kwargs": {"cache_directory": directory}}
        self.future_queue = None
        self.keys = []
        self._on_submit = on_submit

    def submit(self, fn, *args, resource_dict=None, **kwargs):
        key = (resource_dict or {}).get("cache_key")
        self.keys.append(key)
        if self._on_submit is not None:
            self._on_submit(key)
        return self._ex.submit(fn, *args, resource_dict=resource_dict or {}, **kwargs)

    def shutdown(self, wait=True, **kwargs):
        self._ex.shutdown(wait=wait)


EXEC = {"n": 0}


@as_function_node
def Square(x: int = 0):
    EXEC["n"] += 1
    y = x * x
    return y


@as_function_node
def Echo(x: int = 0):
    y = x
    return y


@as_function_node("executor")
def Provide(executor=None):
    return executor


def stored(template):
    set_reserved_port(template, "store", True)
    return template


SWEEPS = {
    "iterate": (iterate, lambda node: node.outputs.out_lst.value),
    "IterToDataFrame": (IterToDataFrame, lambda node: list(node.outputs.df.value["y"])),
}


def _sweep_workflow(template, sweep="IterToDataFrame", executor=None):
    """A workflow file's sweep, run as the GUI runs it: on a *copy*, which the
    tests run with ``run(store=db)``.

    The copy shares its nodes with the original, so the template's own graph
    has no database; the sweep has to store into the database of the run
    executing it.
    """
    node_class, _ = SWEEPS[sweep]
    wf = Workflow("cells")
    wf.T = template
    kwargs = {}
    if executor is not None:
        wf.Ex = Provide(executor=executor)
        kwargs["executor"] = wf.Ex
    wf.Iter = node_class(node=wf.T, input_label="x", values=[1, 2, 3], **kwargs)
    return wf.copy()


def _result(wf, sweep="IterToDataFrame"):
    return SWEEPS[sweep][1](wf.Iter)


def _cells(db):
    """Cell rows: the template's instances, each linked to the sweep."""
    return [r for r in db.rows.values() if r.qualname == "Square" and r.master_hash]


@pytest.mark.parametrize("sweep", sorted(SWEEPS))
def test_cells_are_not_stored_unless_the_template_asks(sweep):
    with tempfile.TemporaryDirectory() as tmp:
        EXEC["n"] = 0
        db = MemDB(tmp)
        wf = _sweep_workflow(Square(), sweep)
        wf.run(store=db)
        assert _result(wf, sweep) == [1, 4, 9]
        assert EXEC["n"] == 3
        assert _cells(db) == []


#: The stores a sweep's cells go into: a directory of files, and a database,
#: which also keeps a row per cell.
STORES = {"FileStore": FileStore, "CheckpointStore": MemDB}


@pytest.mark.parametrize("store", sorted(STORES))
@pytest.mark.parametrize("sweep", sorted(SWEEPS))
@pytest.mark.parametrize("mode", ["sequential", "threads", "remote"])
def test_every_cell_is_stored_linked_to_its_sweep_and_restored(sweep, mode, store):
    executor = {
        "sequential": None,
        "threads": ThreadPoolExecutor(max_workers=2),
        "remote": PicklingExecutor(),
    }[mode]
    with tempfile.TemporaryDirectory() as tmp:
        db = STORES[store](tmp)
        wf = _sweep_workflow(stored(Square()), sweep, executor)
        wf.run(store=db)
        assert _result(wf, sweep) == [1, 4, 9]
        if isinstance(db, MemDB):
            cells = _cells(db)
            assert len(cells) == 3
            assert {r.master_hash for r in cells} == {get_hash(wf.Iter)}
        assert len(list(pathlib.Path(tmp).glob("*.hdf5"))) >= 3

        # A second run over the same values restores every cell.
        EXEC["n"] = 0
        wf.run(store=db)
        assert _result(wf, sweep) == [1, 4, 9]
        if mode != "remote":  # remote cells run in copies; no counter here
            assert EXEC["n"] == 0
        if isinstance(db, MemDB):
            assert len(_cells(db)) == 3
    if mode == "threads":
        executor.shutdown()


def test_no_cells_are_stored_when_recording_is_off():
    with tempfile.TemporaryDirectory() as tmp:
        db = MemDB(tmp)
        wf = _sweep_workflow(stored(Square()))
        wf.storage_enabled = False
        wf.run(store=db)
        assert _result(wf) == [1, 4, 9]
        assert _cells(db) == []


# ---------------------------------------------------------------------------
# Out-of-process executors (Slurm, Flux, process pools)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sweep", sorted(SWEEPS))
def test_remote_sweep_never_ships_the_database(sweep):
    """An attached db must not travel with the task: it hung GUI sweeps."""
    with tempfile.TemporaryDirectory() as tmp:
        pool = PicklingExecutor()
        db = LockedDB(tmp)
        wf = _sweep_workflow(Square(), sweep, pool)
        wf.run(store=db)
        assert _result(wf, sweep) == [1, 4, 9]
        assert pool.submitted == 3
        assert _cells(db) == []


@pytest.mark.parametrize("store", ["FileStore", "LockedDB"])
@pytest.mark.parametrize("sweep", sorted(SWEEPS))
def test_remote_cells_restored_on_the_client_are_not_submitted_again(sweep, store):
    with tempfile.TemporaryDirectory() as tmp:
        db = {"FileStore": FileStore, "LockedDB": LockedDB}[store](tmp)
        first = _sweep_workflow(stored(Square()), sweep, PicklingExecutor())
        first.run(store=db)
        assert len(list(pathlib.Path(tmp).glob("*.hdf5"))) == 3
        if isinstance(db, MemDB):
            assert len(_cells(db)) == 3

        pool = PicklingExecutor()
        again = _sweep_workflow(stored(Square()), sweep, pool)
        again.run(store=db)
        assert _result(again, sweep) == [1, 4, 9]
        assert pool.submitted == 0


@pytest.mark.parametrize("sweep", [iterate, IterToDataFrame])
def test_unpicklable_template_fails_before_anything_is_submitted(sweep):
    pool = PicklingExecutor()
    template = Echo()
    template.inputs.x.value = threading.Lock()
    with pytest.raises(TypeError, match="cannot send 'Echo'"):
        sweep(node=template, input_label="x", values=[1], executor=pool).run()
    assert pool.submitted == 0


@pytest.mark.parametrize("sweep", sorted(SWEEPS))
@pytest.mark.parametrize("store", [False, True], ids=["plain", "stored"])
def test_remote_sweep_restores_the_swept_port(sweep, store):
    with tempfile.TemporaryDirectory() as tmp:
        template = Square(x=7)
        if store:
            template = stored(template)
        db = LockedDB(tmp)
        wf = _sweep_workflow(template, sweep, PicklingExecutor())
        wf.run(store=db)
        assert _result(wf, sweep) == [1, 4, 9]
        assert template.inputs.x.value == 7


# ---------------------------------------------------------------------------
# Cluster executors: cells check-pointed, recoverable, never resubmitted
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_queued_cells_are_checkpointed_keyed_on_their_hash_and_never_resubmitted():
    with tempfile.TemporaryDirectory() as tmp:
        cache = str(pathlib.Path(tmp) / "cache")
        db = MemDB(tmp)
        seen = {}

        def at_submit(key):  # the row must exist *before* the job does
            row = db.read(key)
            seen[key] = (getattr(row, "status", None), row.executor, row.master_hash)

        ex = CacheExecutor(cache, on_submit=at_submit)
        wf = _sweep_workflow(Square(), executor=ex)  # no store: still kept
        wf.run(store=db)
        ex.shutdown()
        assert _result(wf) == [1, 4, 9]

        sweep_hash = get_hash(wf.Iter)
        executor_hash = get_hash(wf.Ex)
        assert len(ex.keys) == 3 and all(ex.keys)
        for key in ex.keys:
            assert seen[key] == ("Submitted", executor_hash, sweep_hash)
            assert getattr(db.read(key), "status", None) != "Submitted"  # completed
            assert (pathlib.Path(cache) / f"{key}_o.h5").exists()

        # A new session over the same cache: everything is collected, nothing
        # goes to the queue again.
        ex2 = CacheExecutor(cache)
        again = _sweep_workflow(Square(), executor=ex2)
        again.run(store=db)
        ex2.shutdown()
        assert ex2.keys == []
        assert _result(again) == [1, 4, 9]
