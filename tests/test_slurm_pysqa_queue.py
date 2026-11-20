"""``SlurmExecutor`` with a pysqa config: ``partition`` must select the queue.

pysqa ignores ``partition`` and, without a ``queue``, falls back to the site's
``queue_primary`` — so the translation is what decides where the jobs run.
"""

import pytest

from aiflow_nodes.executors import _pysqa_queue

QUEUE_YAML = """\
queue_type: SLURM
queue_primary: cmti
queues:
  cmti: {cores_max: 1200, cores_min: 1, run_time_max: 345600, script: cmti.sh}
  s_cmmg: {cores_max: 256, cores_min: 1, run_time_max: 345600, script: cmmg.sh}
"""


@pytest.fixture
def config_dir(tmp_path):
    (tmp_path / "queue.yaml").write_text(QUEUE_YAML)
    return str(tmp_path)


@pytest.mark.parametrize("partition", ["s.cmmg", "s_cmmg"])
def test_partition_or_queue_name_selects_the_queue(config_dir, partition):
    assert _pysqa_queue(config_dir, partition) == "s_cmmg"


def test_unknown_partition_fails_instead_of_falling_back_to_the_primary(config_dir):
    with pytest.raises(ValueError, match="choose one of"):
        _pysqa_queue(config_dir, "normal")


def test_the_executor_fails_a_run_whose_job_dies_instead_of_hanging(tmp_path):
    """Registered with aiflow's queue watchdog as soon as it is built."""
    remote_cache = pytest.importorskip("aiflow.core.remote_cache")
    if not hasattr(remote_cache, "watch_queue"):
        pytest.skip("this aiflow has no queue watchdog")
    from aiflow_nodes.executors import SlurmExecutor

    with pytest.warns(UserWarning):  # executorlib cannot validate 'partition'
        executor = SlurmExecutor(partition="any", cache_directory=str(tmp_path)).run()
    try:
        assert executor in remote_cache._watchdogs
    finally:
        executor.shutdown(wait=False)
