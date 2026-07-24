import json
from pathlib import Path

from milvus_client.common.data import vector_fields
from milvus_client.common.schema import FieldSpec, SchemaSpec, load_schema_matrix
from milvus_client.requests import mixed_rw_pressure
from milvus_client.requests.mixed_rw_pressure import _run_operation

ROOT = Path(__file__).resolve().parents[1]


class FakeSearchClient:
    def __init__(self):
        self.search_calls = []

    def search(self, **kwargs):
        self.search_calls.append(kwargs)
        return [[{"id": 1, "distance": 0.1}]]


class FakeQueryClient:
    def __init__(self):
        self.query_calls = []

    def query(self, **kwargs):
        self.query_calls.append(kwargs)
        return [{"custom_id": 1}]


def test_query_operation_uses_schema_primary_field():
    spec = load_schema_matrix(ROOT / "manifests" / "schema_matrix_2_6.yaml")[0]
    spec.fields[0] = spec.fields[0].__class__(
        name="custom_id", dtype="INT64", primary=True
    )
    client = FakeQueryClient()

    op, count = _run_operation(client, spec, "qa_dense", "query", 7, 10, 1)

    assert op == "query"
    assert count == 1
    assert client.query_calls[0]["filter"] == "custom_id >= 0"
    assert client.query_calls[0]["output_fields"] == ["custom_id"]


def test_query_operation_quotes_string_primary_field():
    spec = SchemaSpec(
        name="string_pk",
        version="test",
        fields=[
            FieldSpec(name="pk", dtype="VARCHAR", primary=True, max_length=64),
            FieldSpec(name="embedding", dtype="FLOAT_VECTOR", dim=2),
        ],
    )
    client = FakeQueryClient()

    op, count = _run_operation(client, spec, "qa_string", "query", 7, 10, 1)

    assert op == "query"
    assert count == 1
    assert client.query_calls[0]["filter"] == 'pk >= "pk_00000000000000000000"'


def test_failed_operation_is_returned_without_raising():
    spec = load_schema_matrix(ROOT / "manifests" / "schema_matrix_2_6.yaml")[0]

    class FailingClient:
        def query(self, **kwargs):
            del kwargs
            raise RuntimeError("temporary failure")

    assert _run_operation(FailingClient(), spec, "qa_dense", "query", 7, 10, 1) == (
        "failed_query",
        1,
    )


def test_pressure_workload_records_operation_failure_details():
    spec = load_schema_matrix(ROOT / "manifests" / "schema_matrix_2_6.yaml")[0]

    class FailingClient:
        def query(self, **kwargs):
            del kwargs
            raise RuntimeError("connection reset by peer")

    from milvus_client.common.workload import run_operation_outcome

    outcome = run_operation_outcome(
        FailingClient(),
        spec,
        "qa",
        "query",
        7,
        batch_size=10,
        op_index=1,
    )

    assert outcome.operation == "failed_query"
    assert outcome.failure_detail["collection"] == "qa"
    assert outcome.failure_detail["operation"] == "query"
    assert outcome.failure_detail["error_type"] == "RuntimeError"
    assert outcome.failure_detail["connectivity_transient"] is True


def test_search_operation_covers_all_vector_fields():
    spec = load_schema_matrix(ROOT / "manifests" / "schema_matrix_2_6.yaml")[1]
    client = FakeSearchClient()

    op, count = _run_operation(
        client, spec, "qa_multi_vector_numeric", "search", 7, 10, 1
    )

    assert op == "search"
    assert count == len(vector_fields(spec))
    assert [call["anns_field"] for call in client.search_calls] == [
        field.name for field in vector_fields(spec)
    ]
    metric_by_field = {
        call["anns_field"]: call["search_params"]["metric_type"]
        for call in client.search_calls
    }
    assert metric_by_field["dense_hnsw"] == "COSINE"
    assert metric_by_field["dense_diskann"] == "L2"
    assert metric_by_field["binary_ivf"] == "HAMMING"
    assert metric_by_field["sparse_bm25"] == "BM25"


def test_search_operation_records_empty_hits_as_failed_operation():
    spec = load_schema_matrix(ROOT / "manifests" / "schema_matrix_2_6.yaml")[0]

    class EmptySearchClient:
        def search(self, **kwargs):
            del kwargs
            return [[]]

    assert _run_operation(
        EmptySearchClient(), spec, "qa_dense", "search", 7, 10, 1
    ) == ("failed_search", 1)


def test_pressure_workload_marks_empty_search_as_non_connectivity_failure():
    spec = load_schema_matrix(ROOT / "manifests" / "schema_matrix_2_6.yaml")[0]

    class EmptySearchClient:
        def search(self, **kwargs):
            del kwargs
            return [[]]

    from milvus_client.common.workload import run_operation_outcome

    outcome = run_operation_outcome(
        EmptySearchClient(),
        spec,
        "qa",
        "search",
        7,
        batch_size=10,
        op_index=1,
    )

    assert outcome.operation == "failed_search"
    assert outcome.failure_detail["operation"] == "search"
    assert outcome.failure_detail["error_type"] == "AssertionError"
    assert outcome.failure_detail["connectivity_transient"] is False


def test_mixed_rw_pressure_writes_structured_connection_failure(monkeypatch, tmp_path):
    output_json = tmp_path / "result.json"

    def fail_connect(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("connect failed")

    monkeypatch.setattr(mixed_rw_pressure, "create_client_with_retry", fail_connect)

    code = mixed_rw_pressure.main(
        [
            "--uri",
            "http://localhost:19530",
            "--collection-prefix",
            "qa",
            "--schema-matrix",
            str(ROOT / "manifests" / "schema_matrix_2_6.yaml"),
            "--checkpoint-dir",
            str(tmp_path / "checkpoints"),
            "--output-json",
            str(output_json),
        ]
    )

    payload = json.loads(output_json.read_text())
    assert code == 1
    assert payload["status"] == "failed"
    assert payload["failures"][0]["type"] == "MIXED_RW_FAILED"
    assert payload["failures"][0]["error"] == "connect failed"


def test_mixed_rw_pressure_includes_count_operation(monkeypatch, tmp_path):
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
        from milvus_client.common.workload import WorkloadSummary

        del (
            client,
            schema_matrix,
            collection_prefix,
            seed,
            duration_sec,
            max_workers,
            batch_size,
            operation_interval_sec,
        )
        captured["operations"] = operations
        captured["baseline_rows_per_collection"] = baseline_rows_per_collection
        summary = WorkloadSummary()
        summary.record("count", 1)
        return summary

    def fake_create_client_with_retry(uri, token, db_name, retry_sec):
        captured["startup_retry_sec"] = retry_sec
        return {"uri": uri, "token": token, "db_name": db_name}

    monkeypatch.setattr(
        mixed_rw_pressure, "create_client_with_retry", fake_create_client_with_retry
    )
    monkeypatch.setattr(
        mixed_rw_pressure, "run_pressure_workload", fake_run_pressure_workload
    )

    code = mixed_rw_pressure.main(
        [
            "--uri",
            "http://localhost:19530",
            "--collection-prefix",
            "qa",
            "--schema-matrix",
            str(ROOT / "manifests" / "schema_matrix_2_6.yaml"),
            "--baseline-rows-per-collection",
            "5000",
            "--startup-retry-sec",
            "7.5",
            "--checkpoint-dir",
            str(tmp_path / "checkpoints"),
            "--output-json",
            str(output_json),
        ]
    )

    payload = json.loads(output_json.read_text())
    assert code == 0
    assert payload["status"] == "passed"
    assert "count" in captured["operations"]
    assert captured["baseline_rows_per_collection"] == 5000
    assert captured["startup_retry_sec"] == 7.5


def test_mixed_rw_pressure_writes_operation_failure_details(monkeypatch, tmp_path):
    output_json = tmp_path / "result.json"

    def fake_run_pressure_workload(*args, **kwargs):
        del args, kwargs
        from milvus_client.common.workload import WorkloadSummary

        summary = WorkloadSummary()
        summary.record(
            "failed_query",
            1,
            {
                "type": "PRESSURE_OPERATION_FAILED",
                "operation": "query",
                "started_at": "2026-07-23T20:42:00+00:00",
                "finished_at": "2026-07-23T20:42:00+00:00",
                "error": "connection refused",
                "connectivity_transient": True,
            },
        )
        return summary

    monkeypatch.setattr(
        mixed_rw_pressure, "create_client_with_retry", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(
        mixed_rw_pressure, "run_pressure_workload", fake_run_pressure_workload
    )

    code = mixed_rw_pressure.main(
        [
            "--uri",
            "http://localhost:19530",
            "--collection-prefix",
            "qa",
            "--schema-matrix",
            str(ROOT / "manifests" / "schema_matrix_2_6.yaml"),
            "--checkpoint-dir",
            str(tmp_path / "checkpoints"),
            "--output-json",
            str(output_json),
        ]
    )

    payload = json.loads(output_json.read_text())
    assert code == 1
    assert payload["status"] == "failed"
    assert payload["metrics"]["requests_failed"] == 1
    assert payload["failures"][0]["operation"] == "query"
    assert payload["failures"][0]["connectivity_transient"] is True
