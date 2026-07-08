import json

from milvus_client.requests import _pressure
from milvus_client.requests import count_pressure
from milvus_client.requests import search_pressure
from milvus_client.common.workload import WorkloadSummary


def test_independent_pressure_brick_writes_structured_result(monkeypatch, tmp_path):
    output_json = tmp_path / "result.json"
    captured = {}

    def fake_run_pressure_workload(
        client,
        schema_matrix,
        collection_prefix,
        operations,
        seed,
        duration_sec,
        max_workers,
        batch_size,
        operation_interval_sec=0.0,
        baseline_start_id=0,
        baseline_rows_per_collection=0,
    ):
        del client, schema_matrix, seed, duration_sec, max_workers, batch_size, operation_interval_sec
        captured["collection_prefix"] = collection_prefix
        captured["operations"] = operations
        captured["baseline_start_id"] = baseline_start_id
        captured["baseline_rows_per_collection"] = baseline_rows_per_collection
        summary = WorkloadSummary()
        summary.record("search", 3)
        return summary

    monkeypatch.setattr(_pressure, "create_client", lambda *args, **kwargs: object())
    monkeypatch.setattr(_pressure, "run_pressure_workload", fake_run_pressure_workload)

    code = search_pressure.main(
        [
            "--uri",
            "http://localhost:19530",
            "--collection-prefix",
            "qa",
            "--schema-matrix",
            "schema.yaml",
            "--checkpoint-dir",
            str(tmp_path),
            "--output-json",
            str(output_json),
        ]
    )

    payload = json.loads(output_json.read_text())
    assert code == 0
    assert payload["brick"] == "search_pressure"
    assert payload["status"] == "passed"
    assert payload["metrics"]["search"] == 3
    assert captured["operations"] == ["search"]
    assert captured["collection_prefix"] == "qa"


def test_count_pressure_forwards_baseline_count_args(monkeypatch, tmp_path):
    output_json = tmp_path / "result.json"
    captured = {}

    def fake_run_pressure_workload(
        client,
        schema_matrix,
        collection_prefix,
        operations,
        seed,
        duration_sec,
        max_workers,
        batch_size,
        operation_interval_sec=0.0,
        baseline_start_id=0,
        baseline_rows_per_collection=0,
    ):
        del client, schema_matrix, collection_prefix, seed, duration_sec, max_workers, batch_size, operation_interval_sec
        captured["operations"] = operations
        captured["baseline_start_id"] = baseline_start_id
        captured["baseline_rows_per_collection"] = baseline_rows_per_collection
        summary = WorkloadSummary()
        summary.record("count", 1)
        return summary

    monkeypatch.setattr(_pressure, "create_client", lambda *args, **kwargs: object())
    monkeypatch.setattr(_pressure, "run_pressure_workload", fake_run_pressure_workload)

    code = count_pressure.main(
        [
            "--uri",
            "http://localhost:19530",
            "--collection-prefix",
            "qa",
            "--schema-matrix",
            "schema.yaml",
            "--baseline-start-id",
            "10",
            "--baseline-rows-per-collection",
            "5000",
            "--checkpoint-dir",
            str(tmp_path),
            "--output-json",
            str(output_json),
        ]
    )

    payload = json.loads(output_json.read_text())
    assert code == 0
    assert payload["brick"] == "count_pressure"
    assert payload["status"] == "passed"
    assert captured["operations"] == ["count"]
    assert captured["baseline_start_id"] == 10
    assert captured["baseline_rows_per_collection"] == 5000


def test_pressure_brick_retries_client_startup(monkeypatch, tmp_path):
    output_json = tmp_path / "result.json"
    attempts = {"count": 0}

    def flaky_create_client(*args, **kwargs):
        del args, kwargs
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("temporarily unavailable")
        return object()

    def fake_sleep(seconds):
        del seconds

    def fake_run_pressure_workload(*args, **kwargs):
        del args, kwargs
        summary = WorkloadSummary()
        summary.record("search", 1)
        return summary

    monkeypatch.setattr(_pressure, "create_client", flaky_create_client)
    monkeypatch.setattr(_pressure.time, "sleep", fake_sleep)
    monkeypatch.setattr(_pressure, "run_pressure_workload", fake_run_pressure_workload)

    code = search_pressure.main(
        [
            "--uri",
            "http://localhost:19530",
            "--collection-prefix",
            "qa",
            "--schema-matrix",
            "schema.yaml",
            "--startup-retry-sec",
            "5",
            "--checkpoint-dir",
            str(tmp_path),
            "--output-json",
            str(output_json),
        ]
    )

    payload = json.loads(output_json.read_text())
    assert code == 0
    assert payload["status"] == "passed"
    assert attempts["count"] == 2
