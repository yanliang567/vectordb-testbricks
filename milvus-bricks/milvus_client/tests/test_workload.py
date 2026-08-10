from pathlib import Path

from milvus_client.common.schema import (
    FieldSpec,
    FunctionSpec,
    IndexSpec,
    SchemaSpec,
    StructArraySpec,
    load_schema_matrix,
)
from milvus_client.common.workload import run_operation, search_params_for_field


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
        indexes=[
            IndexSpec(
                field="sparse_bm25",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="BM25",
            )
        ],
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
    assert client.search_calls[0]["data"] == [
        "document 4 milvus compatibility upgrade rollback token_4"
    ]
    assert client.search_calls[0]["search_params"]["metric_type"] == "BM25"


def test_minhash_function_output_search_uses_text_query():
    spec = SchemaSpec(
        name="minhash",
        version="3.0",
        fields=[
            FieldSpec(name="id", dtype="INT64", primary=True),
            FieldSpec(
                name="document",
                dtype="VARCHAR",
                max_length=65535,
                value_profile="minhash_documents",
            ),
            FieldSpec(name="minhash", dtype="BINARY_VECTOR", dim=4096),
        ],
        functions=[
            FunctionSpec(
                name="text_to_minhash",
                function_type="MINHASH",
                input_fields=["document"],
                output_fields=["minhash"],
            )
        ],
        indexes=[
            IndexSpec(
                field="minhash",
                index_type="MINHASH_LSH",
                metric_type="MHJACCARD",
            )
        ],
    )

    class SearchClient:
        def __init__(self):
            self.search_calls = []

        def search(self, **kwargs):
            self.search_calls.append(kwargs)
            return [[{"id": 4}]]

    client = SearchClient()

    op, count = run_operation(client, spec, "qa_minhash", "search", 7, 10, 3)

    assert op == "search"
    assert count == 1
    assert client.search_calls[0]["data"] == [
        "the quick brown fox jumps over a lazy dog"
    ]
    assert "filter" not in client.search_calls[0]
    assert client.search_calls[0]["search_params"]["metric_type"] == "MHJACCARD"


def test_faiss_search_params_only_set_nprobe_for_ivf_factory_indexes():
    spec = SchemaSpec(
        name="faiss",
        version="3.0",
        fields=[
            FieldSpec(name="id", dtype="INT64", primary=True),
            FieldSpec(name="float_ivf", dtype="FLOAT_VECTOR", dim=64),
            FieldSpec(name="binary_flat", dtype="BINARY_VECTOR", dim=64),
        ],
        indexes=[
            IndexSpec(
                field="float_ivf",
                index_type="FAISS",
                metric_type="L2",
                params={"faiss_index_name": "IVF64,Flat"},
            ),
            IndexSpec(
                field="binary_flat",
                index_type="FAISS",
                metric_type="HAMMING",
                params={"faiss_index_name": "BFlat"},
            ),
        ],
    )

    assert search_params_for_field(spec, "float_ivf") == {"nprobe": 8}
    assert search_params_for_field(spec, "binary_flat") == {}


def test_lossy_faiss_pressure_search_does_not_require_exact_self_recall():
    spec = SchemaSpec(
        name="faiss_lossy",
        version="3.0",
        fields=[
            FieldSpec(name="id", dtype="INT64", primary=True),
            FieldSpec(name="embedding", dtype="FLOAT_VECTOR", dim=64),
        ],
        indexes=[
            IndexSpec(
                field="embedding",
                index_type="FAISS",
                metric_type="COSINE",
                params={"faiss_index_name": "OPQ16,IVF64,PQ16x4"},
                search_params={"nprobe": 8},
            )
        ],
    )

    class SearchClient:
        def __init__(self):
            self.search_calls = []

        def search(self, **kwargs):
            self.search_calls.append(kwargs)
            return [[{"id": 99, "distance": 0.8}]]

    client = SearchClient()

    op, count = run_operation(
        client,
        spec,
        "qa_faiss_lossy",
        "search",
        7,
        10,
        3,
        baseline_start_id=0,
        baseline_rows_per_collection=100,
    )

    assert op == "search"
    assert count == 1
    assert "filter" not in client.search_calls[0]


def test_search_operation_covers_struct_array_only_vector_index():
    spec = SchemaSpec(
        name="struct_only",
        version="3.0",
        fields=[FieldSpec(name="id", dtype="INT64", primary=True)],
        struct_arrays=[
            StructArraySpec(
                name="embeddings",
                max_capacity=4,
                fields=[FieldSpec(name="vector", dtype="FLOAT16_VECTOR", dim=4)],
            )
        ],
        indexes=[
            IndexSpec(
                field="embeddings[vector]",
                index_type="DISKANN",
                metric_type="MAX_SIM_COSINE",
            )
        ],
    )

    class SearchClient:
        def __init__(self):
            self.search_calls = []

        def search(self, **kwargs):
            self.search_calls.append(kwargs)
            return [[{"id": 3, "distance": 1.0}]]

    client = SearchClient()

    op, count = run_operation(
        client,
        spec,
        "qa_struct_only",
        "search",
        7,
        10,
        3,
        baseline_start_id=0,
        baseline_rows_per_collection=10,
    )

    assert op == "search"
    assert count == 1
    assert client.search_calls[0]["anns_field"] == "embeddings[vector]"
    assert client.search_calls[0]["filter"] == "id == 3"
    assert type(client.search_calls[0]["data"][0]).__name__ == "EmbeddingList"


def test_search_operation_uses_non_null_nullable_vector_probe():
    spec = SchemaSpec(
        name="nullable_vector",
        version="2.6",
        fields=[
            FieldSpec(name="id", dtype="INT64", primary=True),
            FieldSpec(name="embedding", dtype="FLOAT_VECTOR", dim=4, nullable=True),
        ],
        indexes=[IndexSpec(field="embedding", index_type="HNSW", metric_type="COSINE")],
    )

    class SearchClient:
        def __init__(self):
            self.search_calls = []

        def search(self, **kwargs):
            self.search_calls.append(kwargs)
            return [[{"id": 1, "distance": 1.0}]]

    client = SearchClient()

    op, count = run_operation(
        client,
        spec,
        "qa_nullable",
        "search",
        7,
        10,
        0,
        baseline_start_id=0,
        baseline_rows_per_collection=10,
    )

    assert op == "search"
    assert count == 1
    assert client.search_calls[0]["filter"] == "id == 1"


def test_count_operation_checks_seed_pk_range_exactly():
    spec = SchemaSpec(
        name="baseline",
        version="test",
        fields=[
            FieldSpec(name="pk", dtype="VARCHAR", primary=True, max_length=64),
            FieldSpec(name="embedding", dtype="FLOAT_VECTOR", dim=2),
        ],
    )

    class CountClient:
        def __init__(self):
            self.query_calls = []

        def query(self, **kwargs):
            self.query_calls.append(kwargs)
            return [{"count(*)": 5}]

    client = CountClient()

    op, count = run_operation(
        client,
        spec,
        "qa_baseline",
        "count",
        7,
        10,
        1,
        baseline_start_id=0,
        baseline_rows_per_collection=5,
    )

    assert op == "count"
    assert count == 1
    assert (
        client.query_calls[0]["filter"]
        == 'pk >= "pk_00000000000000000000" && pk <= "pk_00000000000000000004"'
    )


def test_count_operation_fails_on_baseline_count_drift():
    spec = SchemaSpec(
        name="baseline",
        version="test",
        fields=[
            FieldSpec(name="id", dtype="INT64", primary=True),
            FieldSpec(name="embedding", dtype="FLOAT_VECTOR", dim=2),
        ],
    )

    class DriftClient:
        def query(self, **kwargs):
            del kwargs
            return [{"count(*)": 4}]

    assert run_operation(
        DriftClient(),
        spec,
        "qa_baseline",
        "count",
        7,
        10,
        1,
        baseline_start_id=0,
        baseline_rows_per_collection=5,
    ) == ("failed_count", 1)


def test_count_operation_checks_auto_id_minimum_total_count():
    spec = SchemaSpec(
        name="auto",
        version="test",
        fields=[
            FieldSpec(name="id", dtype="INT64", primary=True, auto_id=True),
            FieldSpec(name="embedding", dtype="FLOAT_VECTOR", dim=2),
        ],
    )

    class CountClient:
        def __init__(self):
            self.query_calls = []

        def query(self, **kwargs):
            self.query_calls.append(kwargs)
            return [{"count(*)": 7}]

    client = CountClient()

    op, count = run_operation(
        client,
        spec,
        "qa_auto",
        "count",
        7,
        10,
        1,
        baseline_rows_per_collection=5,
    )

    assert op == "count"
    assert count == 1
    assert client.query_calls[0]["filter"] == ""


def test_search_params_prefer_explicit_matrix_values():
    spec = SchemaSpec(
        name="faiss",
        version="3.0",
        fields=[
            FieldSpec(name="id", dtype="INT64", primary=True),
            FieldSpec(name="embedding", dtype="FLOAT_VECTOR", dim=64),
        ],
        indexes=[
            IndexSpec(
                field="embedding",
                index_type="FAISS",
                metric_type="COSINE",
                search_params={"nprobe": 16, "refine_k": 2},
            )
        ],
    )

    assert search_params_for_field(spec, "embedding") == {
        "nprobe": 16,
        "refine_k": 2,
    }
