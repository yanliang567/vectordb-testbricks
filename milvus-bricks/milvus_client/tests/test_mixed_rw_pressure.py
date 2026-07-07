from pathlib import Path

from milvus_client.common.schema import load_schema_matrix
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


def test_search_operation_rejects_empty_hits():
    spec = load_schema_matrix(ROOT / "manifests" / "schema_matrix_2_6.yaml")[0]

    class EmptySearchClient:
        def search(self, **kwargs):
            del kwargs
            return [[]]

    try:
        _run_operation(EmptySearchClient(), spec, "qa_dense", "search", 7, 10, 1)
    except AssertionError as exc:
        assert "search returned no hits" in str(exc)
    else:
        raise AssertionError("expected empty search hits to fail")
