import json

from milvus_client.requests import _pressure
from milvus_client.requests import search_pressure
from milvus_client.common.workload import WorkloadSummary


def test_independent_pressure_brick_writes_structured_result(monkeypatch, tmp_path):
    output_json = tmp_path / "result.json"
    captured = {}

    def fake_run_pressure_workload(client, schema_matrix, collection_prefix, operations, seed, duration_sec, max_workers, batch_size, operation_interval_sec=0.0):
        del client, schema_matrix, seed, duration_sec, max_workers, batch_size, operation_interval_sec
        captured["collection_prefix"] = collection_prefix
        captured["operations"] = operations
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
