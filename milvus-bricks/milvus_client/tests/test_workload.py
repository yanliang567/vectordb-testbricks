from pathlib import Path

from milvus_client.common.schema import FieldSpec, FunctionSpec, IndexSpec, SchemaSpec, load_schema_matrix
from milvus_client.common.workload import run_operation


ROOT = Path(__file__).resolve().parents[1]


def test_delete_operation_uses_pressure_pk_range_not_seed_baseline():
    spec = load_schema_matrix(ROOT / "manifests" / "schema_matrix_2_6.yaml")[0]

    class DeleteClient:
        def __init__(self):
            self.delete_calls = []

        def delete(self, **kwargs):
            self.delete_calls.append(kwargs)
            return {"delete_count": 0}

    client = DeleteClient()

    op, count = run_operation(client, spec, "qa_dense", "delete", 7, 10, 2)

    assert op == "delete"
    assert count == 10
    assert client.delete_calls[0]["filter"] == "id >= 30000020 && id <= 30000029"


def test_query_iterator_operation_closes_iterator():
    spec = SchemaSpec(
        name="string_pk",
        version="test",
        fields=[
            FieldSpec(name="pk", dtype="VARCHAR", primary=True, max_length=64),
            FieldSpec(name="embedding", dtype="FLOAT_VECTOR", dim=2),
        ],
    )

    class FakeIterator:
        def __init__(self):
            self.closed = False
            self.calls = 0

        def next(self):
            self.calls += 1
            if self.calls == 1:
                return [{"pk": "pk_00000000000000000000"}]
            return []

        def close(self):
            self.closed = True

    class IteratorClient:
        def __init__(self):
            self.iterator = FakeIterator()
            self.query_iterator_calls = []

        def query_iterator(self, **kwargs):
            self.query_iterator_calls.append(kwargs)
            return self.iterator

    client = IteratorClient()

    op, count = run_operation(client, spec, "qa_string", "query_iterator", 7, 10, 1)

    assert op == "query_iterator"
    assert count == 1
    assert client.iterator.closed
    assert client.query_iterator_calls[0]["filter"] == 'pk >= "pk_00000000000000000000"'


def test_auto_id_collection_skips_destructive_pressure_operations():
    spec = SchemaSpec(
        name="auto",
        version="test",
        fields=[
            FieldSpec(name="id", dtype="INT64", primary=True, auto_id=True),
            FieldSpec(name="embedding", dtype="FLOAT_VECTOR", dim=2),
        ],
    )

    op, count = run_operation(object(), spec, "qa_auto", "delete", 7, 10, 1)

    assert op == "delete_skipped_auto_id"
    assert count == 0


def test_bm25_function_output_search_uses_text_query():
    spec = SchemaSpec(
        name="bm25",
        version="test",
        fields=[
            FieldSpec(name="id", dtype="INT64", primary=True),
            FieldSpec(name="document", dtype="VARCHAR", max_length=256),
            FieldSpec(name="sparse_bm25", dtype="SPARSE_FLOAT_VECTOR"),
        ],
        functions=[
            FunctionSpec(
                name="bm25_document",
                function_type="BM25",
                input_fields=["document"],
                output_fields=["sparse_bm25"],
            )
        ],
        indexes=[IndexSpec(field="sparse_bm25", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")],
    )

    class SearchClient:
        def __init__(self):
            self.search_calls = []

        def search(self, **kwargs):
            self.search_calls.append(kwargs)
            return [[{"id": 1}]]

    client = SearchClient()

    op, count = run_operation(client, spec, "qa_bm25", "search", 7, 10, 3)

    assert op == "search"
    assert count == 1
    assert client.search_calls[0]["data"] == ["milvus compatibility token_3"]
    assert client.search_calls[0]["search_params"]["metric_type"] == "BM25"
