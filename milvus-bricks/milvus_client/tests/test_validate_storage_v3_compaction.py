from types import SimpleNamespace

from milvus_client.requests import validate_storage_v3_compaction as validator


def segment(segment_id, rows, storage_version, state="Sealed", is_sorted=True):
    return SimpleNamespace(
        segment_id=segment_id,
        num_rows=rows,
        storage_version=storage_version,
        state_name=state,
        is_sorted=is_sorted,
    )


class FakeCompactionClient:
    def __init__(self):
        self.states = iter(["Executing", "Completed"])

    def get_compaction_state(self, job_id, timeout):
        return next(self.states)

    def get_compaction_plans(self, job_id, timeout):
        return SimpleNamespace(plans=[SimpleNamespace(sources=[11, 12], target=21)])


def test_wait_for_compaction_requires_terminal_state_and_returns_lineage():
    state, plans = validator._wait_for_compaction(
        FakeCompactionClient(), "coll", 7, timeout_sec=1, poll_interval_sec=0
    )

    assert state == "Completed"
    assert plans == [{"sources": [11, 12], "target": 21}]


class FakeStorageClient:
    def __init__(self):
        self.persistent = [
            [segment(21, 100, 2)],
            [segment(21, 100, 3)],
            [segment(21, 100, 3)],
        ]
        self.loaded = [
            [segment(21, 100, 2)],
            [segment(21, 100, 3)],
            [segment(21, 100, 3)],
        ]

    def list_persistent_segments(self, collection):
        return self.persistent.pop(0)

    def list_loaded_segments(self, collection):
        return self.loaded.pop(0)


def test_wait_for_storage_checkpoint_proves_loaded_storage_v3_is_stable(monkeypatch):
    client = FakeStorageClient()
    monkeypatch.setattr(validator, "sleep", lambda _: None)

    checkpoint = validator._wait_for_storage_checkpoint(
        client,
        "coll",
        expected_rows=100,
        expected_storage_version=3,
        timeout_sec=1,
        poll_interval_sec=0,
    )

    assert checkpoint["storage_versions"] == [3]
    assert checkpoint["active"] == checkpoint["serving"]
    assert checkpoint["serving"][21]["storage_version"] == 3


def test_wait_for_storage_checkpoint_can_ignore_stale_baseline_row_count(monkeypatch):
    client = FakeStorageClient()
    client.persistent = [
        [segment(21, 100, 2)],
        [segment(21, 120, 3)],
        [segment(21, 120, 3)],
    ]
    client.loaded = [
        [segment(21, 100, 2)],
        [segment(21, 120, 3)],
        [segment(21, 120, 3)],
    ]
    monkeypatch.setattr(validator, "sleep", lambda _: None)

    checkpoint = validator._wait_for_storage_checkpoint(
        client,
        "coll",
        expected_rows=None,
        expected_storage_version=3,
        timeout_sec=1,
        poll_interval_sec=0,
    )

    assert checkpoint["active_rows"] == 120
    assert checkpoint["storage_versions"] == [3]


def test_wait_for_storage_checkpoint_accepts_querycoord_omitted_loaded_version(
    monkeypatch,
):
    client = FakeStorageClient()
    client.persistent = [
        [segment(21, 100, 2)],
        [segment(21, 100, 3)],
        [segment(21, 100, 3)],
    ]
    client.loaded = [
        [segment(21, 100, 0)],
        [segment(21, 100, 0)],
        [segment(21, 100, 0)],
    ]
    monkeypatch.setattr(validator, "sleep", lambda _: None)

    checkpoint = validator._wait_for_storage_checkpoint(
        client,
        "coll",
        expected_rows=None,
        expected_storage_version=3,
        timeout_sec=1,
        poll_interval_sec=0,
    )

    assert checkpoint["storage_versions"] == [3]
    assert checkpoint["serving_storage_versions"] == [0]
