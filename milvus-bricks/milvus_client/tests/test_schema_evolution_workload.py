from milvus_client.common.schema import FieldSpec, FunctionSpec, IndexSpec, SchemaSpec
from milvus_client.requests.schema_evolution_workload import run_schema_evolution


class FakeClient:
    def __init__(self):
        self.calls = []

    def has_collection(self, collection_name):
        self.calls.append(("has_collection", collection_name))
        return True

    def add_collection_field(self, collection_name, field_name, data_type, **kwargs):
        self.calls.append(("add_collection_field", collection_name, field_name, data_type, kwargs))

    def add_collection_function(self, collection_name, function, **kwargs):
        self.calls.append(("add_collection_function", collection_name, function.name))

    def drop_collection_function(self, collection_name, function_name, **kwargs):
        self.calls.append(("drop_collection_function", collection_name, function_name))

    def upsert(self, collection_name, data):
        self.calls.append(("upsert", collection_name, data))
        return {"upsert_count": len(data)}

    def query(self, collection_name, filter, output_fields, limit=None):
        self.calls.append(("query", collection_name, filter, tuple(output_fields), limit))
        return [{"id": 1}]

    def search(self, collection_name, data, anns_field, limit, search_params):
        self.calls.append(("search", collection_name, anns_field, search_params, data))
        return [[{"id": 1, "distance": 0.1}]]


def _baseline_bm25_spec():
    return SchemaSpec(
        name="existing_bm25",
        version="2.6",
        fields=[
            FieldSpec(name="id", dtype="INT64", primary=True),
            FieldSpec(name="text", dtype="VARCHAR", max_length=256, enable_analyzer=True),
            FieldSpec(name="sparse_bm25", dtype="SPARSE_FLOAT_VECTOR"),
            FieldSpec(name="embedding", dtype="FLOAT_VECTOR", dim=8),
        ],
        functions=[
            FunctionSpec(
                name="text_bm25_emb",
                function_type="BM25",
                input_fields=["text"],
                output_fields=["sparse_bm25"],
            )
        ],
        indexes=[
            IndexSpec(field="sparse_bm25", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25"),
            IndexSpec(field="embedding", index_type="AUTOINDEX", metric_type="COSINE"),
        ],
    )


def test_schema_evolution_cycles_existing_collection_fields_functions_and_reads():
    client = FakeClient()
    metrics = run_schema_evolution(
        client,
        [_baseline_bm25_spec()],
        collection_prefix="qa",
        rows_per_collection=3,
        batch_size=2,
        start_id=5000,
        seed=7,
    )

    call_names = [call[0] for call in client.calls]
    assert "add_collection_field" in call_names
    assert ("drop_collection_function", "qa_existing_bm25", "text_bm25_emb") in client.calls
    assert ("add_collection_function", "qa_existing_bm25", "text_bm25_emb") in client.calls
    assert any(call[0] == "upsert" and call[1] == "qa_existing_bm25" for call in client.calls)
    assert any(call[0] == "query" and "evo_nullable_varchar" in call[3] for call in client.calls)
    assert any(call[0] == "search" and call[2] == "embedding" for call in client.calls)
    assert any(
        call[0] == "search" and call[2] == "sparse_bm25" and isinstance(call[4][0], str)
        for call in client.calls
    )
    assert metrics["collections_total"] == 1
    assert metrics["failed_total"] == 0
    assert metrics["function_cycles_total"] == 1
    assert metrics["drop_field_skipped_total"] == 1


def test_schema_evolution_updates_nullable_vector_collection():
    client = FakeClient()
    spec = SchemaSpec(
        name="nullable_vector",
        version="3.0",
        fields=[
            FieldSpec(name="id", dtype="INT64", primary=True),
            FieldSpec(name="category", dtype="INT64"),
            FieldSpec(name="embedding", dtype="FLOAT_VECTOR", dim=8, nullable=True),
        ],
        indexes=[IndexSpec(field="embedding", index_type="AUTOINDEX", metric_type="COSINE")],
        validators=["null_vector_semantics"],
    )

    metrics = run_schema_evolution(
        client,
        [spec],
        collection_prefix="qa3",
        rows_per_collection=4,
        batch_size=2,
        start_id=6000,
        seed=11,
    )

    upsert_rows = [row for call in client.calls if call[0] == "upsert" for row in call[2]]
    assert any(row["embedding"] is None for row in upsert_rows)
    assert any(row["embedding"] is not None for row in upsert_rows)
    assert metrics["nullable_updates_total"] == 4
    assert metrics["failed_total"] == 0


def test_schema_evolution_formats_string_primary_key_filters():
    client = FakeClient()
    spec = SchemaSpec(
        name="string_pk",
        version="2.6",
        fields=[
            FieldSpec(name="pk", dtype="VARCHAR", primary=True, max_length=64),
            FieldSpec(name="embedding", dtype="FLOAT_VECTOR", dim=8),
        ],
        indexes=[IndexSpec(field="embedding", index_type="AUTOINDEX", metric_type="COSINE")],
    )

    metrics = run_schema_evolution(
        client,
        [spec],
        collection_prefix="qa",
        rows_per_collection=2,
        batch_size=2,
        start_id=7000,
        seed=13,
    )

    query_filters = [call[2] for call in client.calls if call[0] == "query"]
    assert any('pk >= "pk_00000000000000007000"' in filter_expr for filter_expr in query_filters)
    assert metrics["failed_total"] == 0
