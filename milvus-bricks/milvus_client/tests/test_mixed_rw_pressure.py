import json
from pathlib import Path

from milvus_client.common.schema import load_schema_matrix
from milvus_client.common.schema import FieldSpec, SchemaSpec
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
    spec.fields[0] = spec.fields[0].__class__(name="custom_id", dtype="INT64", primary=True)
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

    assert _run_operation(FailingClient(), spec, "qa_dense", "query", 7, 10, 1) == ("failed_query", 1)


def test_search_operation_covers_all_vector_fields():
    spec = load_schema_matrix(ROOT / "manifests" / "schema_matrix_2_6.yaml")[1]
    client = FakeSearchClient()

    op, count = _run_operation(client, spec, "qa_multi_vector_numeric", "search", 7, 10, 1)

    assert op == "search"
    assert count == 3
    assert [call["anns_field"] for call in client.search_calls] == [
        "float16_vec",
        "bfloat16_vec",
        "int8_vec",
    ]
    assert all(call["search_params"]["metric_type"] == "COSINE" for call in client.search_calls)


def test_search_operation_records_empty_hits_as_failed_operation():
    spec = load_schema_matrix(ROOT / "manifests" / "schema_matrix_2_6.yaml")[0]

    class EmptySearchClient:
        def search(self, **kwargs):
            del kwargs
            return [[]]

    assert _run_operation(EmptySearchClient(), spec, "qa_dense", "search", 7, 10, 1) == ("failed_search", 1)


def test_mixed_rw_pressure_writes_structured_connection_failure(monkeypatch, tmp_path):
    output_json = tmp_path / "result.json"

    def fail_connect(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("connect failed")

    monkeypatch.setattr(mixed_rw_pressure, "create_client", fail_connect)

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
