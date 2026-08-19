import json
import re

from milvus_client.common.schema import (
    FieldSpec,
    FunctionSpec,
    IndexSpec,
    SchemaSpec,
    StructArraySpec,
)
from milvus_client.requests.schema_evolution_workload import run_schema_evolution


class FakeClient:
    def __init__(self):
        self.calls = []
        self.rows = {}
        self.next_auto_id = 1000

    def has_collection(self, collection_name):
        self.calls.append(("has_collection", collection_name))
        return True

    def add_collection_field(self, collection_name, field_name, data_type, **kwargs):
        self.calls.append(
            ("add_collection_field", collection_name, field_name, data_type, kwargs)
        )

    def add_collection_function(self, collection_name, function, **kwargs):
        self.calls.append(("add_collection_function", collection_name, function.name))

    def drop_collection_function(self, collection_name, function_name, **kwargs):
        self.calls.append(("drop_collection_function", collection_name, function_name))

    def upsert(self, collection_name, data):
        self.calls.append(("upsert", collection_name, data))
        collection_rows = self.rows.setdefault(collection_name, {})
        for row in data:
            primary_name = "pk" if "pk" in row else "id"
            collection_rows[row[primary_name]] = row
        return {"upsert_count": len(data)}

    def insert(self, collection_name, data):
        self.calls.append(("insert", collection_name, data))
        collection_rows = self.rows.setdefault(collection_name, {})
        ids = []
        for row in data:
            inserted_id = self.next_auto_id
            self.next_auto_id += 1
            stored = {**row, "id": inserted_id}
            collection_rows[inserted_id] = stored
            ids.append(inserted_id)
        return {"ids": ids}

    def query(self, collection_name, filter, output_fields, limit=None):
        self.calls.append(
            ("query", collection_name, filter, tuple(output_fields), limit)
        )
        rows = list(self.rows.get(collection_name, {}).values())
        rows = [row for row in rows if self._matches_filter(row, filter)]
        if output_fields == ["count(*)"]:
            return [{"count(*)": len(rows)}]
        if limit is not None:
            rows = rows[:limit]
        return [{field: row.get(field) for field in output_fields} for row in rows]

    def search(
        self,
        collection_name,
        data,
        anns_field,
        limit,
        search_params,
        filter="",
        **kwargs,
    ):
        self.calls.append(
            ("search", collection_name, anns_field, search_params, data, filter)
        )
        rows = list(self.rows.get(collection_name, {}).values())
        rows = [row for row in rows if self._matches_filter(row, filter)]
        if not rows:
            return [[]]
        primary_name = "pk" if "pk" in rows[0] else "id"
        return [
            [
                {
                    "id": rows[0][primary_name],
                    primary_name: rows[0][primary_name],
                    "offset": 0,
                    "distance": 1.0,
                }
            ]
        ]

    @staticmethod
    def _parse_filter_value(value):
        value = value.strip()
        if value.startswith('"'):
            return json.loads(value)
        if "." in value:
            return float(value)
        return int(value)

    @classmethod
    def _matches_filter(cls, row, filter_expr):
        if not filter_expr:
            return True
        in_values = re.search(r"(\w+) in \[(.*)\]", filter_expr)
        if in_values:
            values = [
                cls._parse_filter_value(value)
                for value in in_values.group(2).split(",")
            ]
            return row.get(in_values.group(1)) in values
        exact = re.search(r"(\w+) == (\"(?:\\.|[^\"])*\"|-?\d+(?:\.\d+)?)", filter_expr)
        if exact:
            return row.get(exact.group(1)) == cls._parse_filter_value(exact.group(2))
        lower = re.search(r"(\w+) >= (\"(?:\\.|[^\"])*\"|-?\d+(?:\.\d+)?)", filter_expr)
        upper = re.search(r"(\w+) <= (\"(?:\\.|[^\"])*\"|-?\d+(?:\.\d+)?)", filter_expr)
        if lower and row.get(lower.group(1)) < cls._parse_filter_value(lower.group(2)):
            return False
        if upper and row.get(upper.group(1)) > cls._parse_filter_value(upper.group(2)):
            return False
        return True


class EmptyEvolutionReadClient(FakeClient):
    def query(self, collection_name, filter, output_fields, limit=None):
        self.calls.append(
            ("query", collection_name, filter, tuple(output_fields), limit)
        )
        if output_fields == ["count(*)"]:
            return [{"count(*)": 0}]
        return []


class IrrelevantSearchHitClient(FakeClient):
    def search(self, *args, **kwargs):
        super().search(*args, **kwargs)
        return [[{"id": -1, "distance": 1.0}]]


class LowSimilaritySearchClient(FakeClient):
    def search(self, *args, **kwargs):
        response = super().search(*args, **kwargs)
        if response and response[0]:
            response[0][0]["distance"] = 0.1
        return response


class MissingAutoIdResponseClient(FakeClient):
    def insert(self, collection_name, data):
        super().insert(collection_name, data)
        return {}


class DuplicateAutoIdResponseClient(FakeClient):
    def insert(self, collection_name, data):
        response = super().insert(collection_name, data)
        return {"ids": [response["ids"][0]] * len(response["ids"])}


class FakeBm25DropRequiresFieldClient(FakeClient):
    def drop_collection_function(self, collection_name, function_name, **kwargs):
        self.calls.append(("drop_collection_function", collection_name, function_name))
        raise RuntimeError(
            "BM25 function must be dropped with its output field in drop_function_field interface"
        )


class FakeFunctionFieldClient(FakeClient):
    def drop_function_field(self, collection_name, function_name, **kwargs):
        self.calls.append(("drop_function_field", collection_name, function_name))

    def add_function_field(
        self, collection_name, field_schema, func, index_params, **kwargs
    ):
        self.calls.append(
            (
                "add_function_field",
                collection_name,
                field_schema.name,
                func.name,
                index_params,
            )
        )


def _baseline_bm25_spec():
    return SchemaSpec(
        name="existing_bm25",
        version="2.6",
        fields=[
            FieldSpec(name="id", dtype="INT64", primary=True),
            FieldSpec(
                name="text", dtype="VARCHAR", max_length=256, enable_analyzer=True
            ),
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
            IndexSpec(
                field="sparse_bm25",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="BM25",
            ),
            IndexSpec(field="embedding", index_type="AUTOINDEX", metric_type="COSINE"),
        ],
    )


def _minhash_spec():
    return SchemaSpec(
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


def _struct_array_spec():
    return SchemaSpec(
        name="struct_array",
        version="3.0",
        fields=[
            FieldSpec(name="id", dtype="INT64", primary=True),
            FieldSpec(name="normal_vector", dtype="FLOAT_VECTOR", dim=8),
        ],
        struct_arrays=[
            StructArraySpec(
                name="items",
                max_capacity=4,
                fields=[
                    FieldSpec(name="embedding", dtype="FLOAT_VECTOR", dim=8),
                    FieldSpec(name="score", dtype="FLOAT"),
                    FieldSpec(name="category", dtype="VARCHAR", max_length=64),
                ],
            )
        ],
        indexes=[
            IndexSpec(
                field="normal_vector", index_type="AUTOINDEX", metric_type="COSINE"
            ),
            IndexSpec(
                field="items[embedding]",
                index_type="HNSW",
                metric_type="COSINE",
            ),
            IndexSpec(field="items[score]", index_type="STL_SORT"),
        ],
    )


def _auto_id_spec():
    return SchemaSpec(
        name="auto_id",
        version="3.0",
        fields=[
            FieldSpec(name="id", dtype="INT64", primary=True, auto_id=True),
            FieldSpec(name="category", dtype="INT64"),
            FieldSpec(name="embedding", dtype="FLOAT_VECTOR", dim=8),
        ],
        indexes=[
            IndexSpec(field="embedding", index_type="AUTOINDEX", metric_type="COSINE")
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
    assert (
        "drop_collection_function",
        "qa_existing_bm25",
        "text_bm25_emb",
    ) in client.calls
    assert (
        "add_collection_function",
        "qa_existing_bm25",
        "text_bm25_emb",
    ) in client.calls
    assert any(
        call[0] == "upsert" and call[1] == "qa_existing_bm25" for call in client.calls
    )
    assert any(
        call[0] == "query" and "evo_nullable_varchar" in call[3]
        for call in client.calls
    )
    assert any(call[0] == "search" and call[2] == "embedding" for call in client.calls)
    assert any(
        call[0] == "search" and call[2] == "sparse_bm25" and isinstance(call[4][0], str)
        for call in client.calls
    )
    assert metrics["collections_total"] == 1
    assert metrics["failed_total"] == 0
    assert metrics["function_cycles_total"] == 1
    assert metrics["drop_field_skipped_total"] == 1


class SchemaMismatchRetryableException(Exception):
    pass


class FlakySchemaMismatchClient(FakeClient):
    def __init__(self, fail_upserts: int = 1):
        super().__init__()
        self.fail_upserts = fail_upserts
        self.upsert_attempts = 0

    def upsert(self, collection_name, data):
        self.upsert_attempts += 1
        if self.upsert_attempts <= self.fail_upserts:
            raise SchemaMismatchRetryableException(
                f"collection schema mismatch[collection={collection_name}]"
            )
        return super().upsert(collection_name, data)


def test_schema_evolution_retries_upsert_on_schema_mismatch():
    client = FlakySchemaMismatchClient(fail_upserts=1)

    metrics = run_schema_evolution(
        client,
        [_baseline_bm25_spec()],
        collection_prefix="qa",
        rows_per_collection=2,
        batch_size=2,
        start_id=5000,
        seed=7,
    )

    assert client.upsert_attempts == 2
    assert metrics["failed_total"] == 0


def test_schema_evolution_gives_up_after_schema_mismatch_retries():
    client = FlakySchemaMismatchClient(fail_upserts=99)

    metrics = run_schema_evolution(
        client,
        [_baseline_bm25_spec()],
        collection_prefix="qa",
        rows_per_collection=2,
        batch_size=2,
        start_id=5000,
        seed=7,
    )

    assert client.upsert_attempts == 5
    assert metrics["failed_total"] == 1


def test_schema_evolution_uses_function_field_apis_when_available():
    client = FakeFunctionFieldClient()

    metrics = run_schema_evolution(
        client,
        [_baseline_bm25_spec()],
        collection_prefix="qa",
        rows_per_collection=2,
        batch_size=2,
        start_id=5000,
        seed=7,
    )

    assert metrics["failed_total"] == 0
    assert metrics["function_cycles_total"] == 1
    assert (
        "drop_function_field",
        "qa_existing_bm25",
        "text_bm25_emb",
    ) in client.calls
    add_call = next(call for call in client.calls if call[0] == "add_function_field")
    assert add_call[1:4] == (
        "qa_existing_bm25",
        "sparse_bm25",
        "text_bm25_emb",
    )
    assert not any(call[0] == "drop_collection_function" for call in client.calls)
    assert not any(call[0] == "add_collection_function" for call in client.calls)


def test_schema_evolution_skips_function_field_cycle_when_disabled():
    client = FakeFunctionFieldClient()

    metrics = run_schema_evolution(
        client,
        [_baseline_bm25_spec()],
        collection_prefix="qa",
        rows_per_collection=2,
        batch_size=2,
        start_id=5000,
        seed=7,
        function_field_cycle_enabled=False,
    )

    assert metrics["failed_total"] == 0
    assert metrics["function_cycles_total"] == 0
    assert metrics["function_cycle_skipped_total"] == 1
    assert metrics["collections"][0]["function_cycle_skip_reasons"] == [
        "skipped_disabled"
    ]
    assert not any("function_field" in call[0] for call in client.calls)


def test_schema_evolution_minhash_search_uses_function_input_text():
    client = FakeFunctionFieldClient()

    metrics = run_schema_evolution(
        client,
        [_minhash_spec()],
        collection_prefix="qa3",
        rows_per_collection=3,
        batch_size=2,
        start_id=5000,
        seed=7,
    )

    searches = [call for call in client.calls if call[0] == "search"]
    assert metrics["failed_total"] == 0
    assert metrics["function_cycles_total"] == 0
    assert metrics["function_cycle_skipped_total"] == 1
    assert metrics["collections"][0]["function_cycle_skip_reasons"] == [
        "skipped_only_vector_field"
    ]
    assert searches[0][2] == "minhash"
    assert searches[0][4] == [
        "distributed vector databases provide scalable similarity search"
    ]


def test_schema_evolution_skips_bm25_function_cycle_when_drop_function_field_api_is_missing():
    client = FakeBm25DropRequiresFieldClient()
    metrics = run_schema_evolution(
        client,
        [_baseline_bm25_spec()],
        collection_prefix="qa",
        rows_per_collection=2,
        batch_size=2,
        start_id=5000,
        seed=7,
    )

    collection = metrics["collections"][0]
    assert metrics["failed_total"] == 0
    assert metrics["function_cycles_total"] == 0
    assert metrics["function_cycle_skipped_total"] == 1
    assert collection["function_cycle_skip_reasons"] == [
        "skipped_drop_function_field_api_missing"
    ]
    assert not any(call[0] == "add_collection_function" for call in client.calls)


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
        indexes=[
            IndexSpec(field="embedding", index_type="AUTOINDEX", metric_type="COSINE")
        ],
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

    upsert_rows = [
        row for call in client.calls if call[0] == "upsert" for row in call[2]
    ]
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
        indexes=[
            IndexSpec(field="embedding", index_type="AUTOINDEX", metric_type="COSINE")
        ],
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
    assert any(
        'pk >= "pk_00000000000000007000"' in filter_expr
        for filter_expr in query_filters
    )
    assert metrics["failed_total"] == 0


def test_schema_evolution_fails_when_evolved_rows_are_not_queryable():
    metrics = run_schema_evolution(
        EmptyEvolutionReadClient(),
        [_baseline_bm25_spec()],
        collection_prefix="qa",
        rows_per_collection=2,
        batch_size=2,
        start_id=5000,
        seed=7,
    )

    assert metrics["failed_total"] == 1
    assert "expected 2 evolved rows" in metrics["collections"][0]["error"]


def test_schema_evolution_fails_when_search_returns_unrelated_primary_key():
    metrics = run_schema_evolution(
        IrrelevantSearchHitClient(),
        [_baseline_bm25_spec()],
        collection_prefix="qa",
        rows_per_collection=2,
        batch_size=2,
        start_id=5000,
        seed=7,
    )

    assert metrics["failed_total"] == 1
    assert "expected primary key" in metrics["collections"][0]["error"]


def test_schema_evolution_checkpoint_is_reused_read_only_after_rollback():
    client = FakeClient()
    checkpoint = {"version": 1, "collections": {}}
    upgrade = run_schema_evolution(
        client,
        [_struct_array_spec()],
        collection_prefix="qa3",
        rows_per_collection=3,
        batch_size=2,
        start_id=5000,
        seed=7,
        checkpoint_output=checkpoint,
    )

    assert upgrade["failed_total"] == 0
    collection_checkpoint = checkpoint["collections"]["qa3_struct_array"]
    assert collection_checkpoint["expected_count"] == 3
    assert collection_checkpoint["validation_fields"] == [
        "id",
        "normal_vector",
        "items",
        "evo_nullable_varchar",
    ]
    assert len(collection_checkpoint["sample_checksums"]) == 3
    assert {probe["field"] for probe in collection_checkpoint["search_probes"]} == {
        "normal_vector",
        "items[embedding]",
    }

    client.calls.clear()
    rollback = run_schema_evolution(
        client,
        [_struct_array_spec()],
        collection_prefix="qa3",
        rows_per_collection=999,
        batch_size=99,
        start_id=1,
        seed=999,
        phase="after-rollback",
        checkpoint=checkpoint,
    )

    assert rollback["failed_total"] == 0
    assert not any(
        call[0] in {"add_collection_field", "drop_collection_field", "upsert"}
        for call in client.calls
    )


def test_schema_evolution_rollback_detects_struct_array_payload_drift():
    client = FakeClient()
    checkpoint = {"version": 1, "collections": {}}
    upgrade = run_schema_evolution(
        client,
        [_struct_array_spec()],
        collection_prefix="qa3",
        rows_per_collection=3,
        batch_size=2,
        start_id=5000,
        seed=7,
        checkpoint_output=checkpoint,
    )
    assert upgrade["failed_total"] == 0
    client.rows["qa3_struct_array"][5001]["items"][0]["score"] = -123.0

    rollback = run_schema_evolution(
        client,
        [_struct_array_spec()],
        collection_prefix="qa3",
        rows_per_collection=3,
        batch_size=2,
        start_id=5000,
        seed=7,
        phase="after-rollback",
        checkpoint=checkpoint,
    )

    assert rollback["failed_total"] == 1
    assert "checkpoint checksum" in rollback["collections"][0]["error"]


def test_schema_evolution_rollback_rejects_checkpoint_without_struct_array_field():
    client = FakeClient()
    checkpoint = {"version": 1, "collections": {}}
    upgrade = run_schema_evolution(
        client,
        [_struct_array_spec()],
        collection_prefix="qa3",
        rows_per_collection=3,
        batch_size=2,
        start_id=5000,
        seed=7,
        checkpoint_output=checkpoint,
    )
    assert upgrade["failed_total"] == 0
    checkpoint["collections"]["qa3_struct_array"]["validation_fields"] = [
        "id",
        "evo_nullable_varchar",
    ]

    rollback = run_schema_evolution(
        client,
        [_struct_array_spec()],
        collection_prefix="qa3",
        rows_per_collection=3,
        batch_size=2,
        start_id=5000,
        seed=7,
        phase="after-rollback",
        checkpoint=checkpoint,
    )

    assert rollback["failed_total"] == 1
    assert "validation fields differ" in rollback["collections"][0]["error"]


def test_schema_evolution_rollback_detects_top_level_vector_content_drift():
    client = FakeClient()
    checkpoint = {"version": 1, "collections": {}}
    upgrade = run_schema_evolution(
        client,
        [_struct_array_spec()],
        collection_prefix="qa3",
        rows_per_collection=3,
        batch_size=2,
        start_id=5000,
        seed=7,
        checkpoint_output=checkpoint,
    )
    assert upgrade["failed_total"] == 0
    assert (
        "normal_vector"
        in checkpoint["collections"]["qa3_struct_array"]["validation_fields"]
    )
    for row in client.rows["qa3_struct_array"].values():
        row["normal_vector"] = [999.0] * 8

    rollback = run_schema_evolution(
        client,
        [_struct_array_spec()],
        collection_prefix="qa3",
        rows_per_collection=3,
        batch_size=2,
        start_id=5000,
        seed=7,
        phase="after-rollback",
        checkpoint=checkpoint,
    )

    assert rollback["failed_total"] == 1
    assert "checkpoint checksum" in rollback["collections"][0]["error"]


def test_schema_evolution_rollback_detects_nullable_vector_null_state_drift():
    client = FakeClient()
    spec = SchemaSpec(
        name="nullable_vector",
        version="3.0",
        fields=[
            FieldSpec(name="id", dtype="INT64", primary=True),
            FieldSpec(name="embedding", dtype="FLOAT_VECTOR", dim=8, nullable=True),
        ],
        indexes=[
            IndexSpec(field="embedding", index_type="AUTOINDEX", metric_type="COSINE")
        ],
    )
    checkpoint = {"version": 1, "collections": {}}
    upgrade = run_schema_evolution(
        client,
        [spec],
        collection_prefix="qa3",
        rows_per_collection=4,
        batch_size=2,
        start_id=6000,
        seed=11,
        checkpoint_output=checkpoint,
    )
    assert upgrade["failed_total"] == 0
    assert client.rows["qa3_nullable_vector"][6000]["embedding"] is None
    client.rows["qa3_nullable_vector"][6000]["embedding"] = [999.0] * 8

    rollback = run_schema_evolution(
        client,
        [spec],
        collection_prefix="qa3",
        rows_per_collection=4,
        batch_size=2,
        start_id=6000,
        seed=11,
        phase="after-rollback",
        checkpoint=checkpoint,
    )

    assert rollback["failed_total"] == 1
    assert "checkpoint checksum" in rollback["collections"][0]["error"]


def test_schema_evolution_search_requires_metric_specific_self_match_score():
    metrics = run_schema_evolution(
        LowSimilaritySearchClient(),
        [_struct_array_spec()],
        collection_prefix="qa3",
        rows_per_collection=3,
        batch_size=2,
        start_id=5000,
        seed=7,
    )

    assert metrics["failed_total"] == 1
    assert "self-match score" in metrics["collections"][0]["error"]


def test_schema_evolution_auto_id_writes_and_validates_checkpoint_after_rollback():
    client = FakeClient()
    checkpoint = {"version": 1, "collections": {}}
    upgrade = run_schema_evolution(
        client,
        [_auto_id_spec()],
        collection_prefix="qa3",
        rows_per_collection=3,
        batch_size=2,
        start_id=5000,
        seed=7,
        checkpoint_output=checkpoint,
    )

    assert upgrade["failed_total"] == 0
    collection_checkpoint = checkpoint["collections"]["qa3_auto_id"]
    assert collection_checkpoint["pk_values"] == [1000, 1001, 1002]
    assert collection_checkpoint["expected_count"] == 3

    client.calls.clear()
    rollback = run_schema_evolution(
        client,
        [_auto_id_spec()],
        collection_prefix="qa3",
        rows_per_collection=999,
        batch_size=99,
        start_id=1,
        seed=999,
        phase="after-rollback",
        checkpoint=checkpoint,
    )

    assert rollback["failed_total"] == 0
    assert any(call[0] == "query" for call in client.calls)
    assert any(call[0] == "search" for call in client.calls)
    assert not any(call[0] in {"insert", "upsert"} for call in client.calls)


def test_schema_evolution_auto_id_rejects_missing_insert_ids():
    metrics = run_schema_evolution(
        MissingAutoIdResponseClient(),
        [_auto_id_spec()],
        collection_prefix="qa3",
        rows_per_collection=3,
        batch_size=2,
        start_id=5000,
        seed=7,
    )

    assert metrics["failed_total"] == 1
    assert "returned 0 primary keys for 2 rows" in metrics["collections"][0]["error"]


def test_schema_evolution_auto_id_rejects_duplicate_insert_ids():
    metrics = run_schema_evolution(
        DuplicateAutoIdResponseClient(),
        [_auto_id_spec()],
        collection_prefix="qa3",
        rows_per_collection=3,
        batch_size=2,
        start_id=5000,
        seed=7,
    )

    assert metrics["failed_total"] == 1
    assert "duplicate primary keys" in metrics["collections"][0]["error"]
